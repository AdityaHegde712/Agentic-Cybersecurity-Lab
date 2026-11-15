"""Empirical Exploration: Imperceptible-Tier and Adversarial Cyber-Physical Attacks.

Investigates:
1. Imperceptible-tier attacks: Recovery limits when perturbation amplitude is within
   the surrogate noise floor (0.25*sigma_eps to 3.0*sigma_eps).
2. Differentiable Adversarial Attacks: Time-Series PGD on companion sensors to test
   surrogate vulnerability and whether the downstream ADMM solver detects or gets fooled.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.inverse.estimators.multi_order_trend_filter import MultiOrderTrendFilterEstimator  # noqa: E402
from src.inverse.surrogates import DilatedTCNSurrogate  # noqa: E402

OUTPUT_DIR = Path("results/experiments/adversarial_imperceptible")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_experiments() -> dict[str, object]:
    print("=" * 80)
    print("EMPIRICAL INVESTIGATION: IMPERCEPTIBLE & ADVERSARIAL ATTACK RECOVERY")
    print("=" * 80)

    # 1. Load BATADAL development split
    val_path = Path("data/processed/batadal/val.parquet")
    val_df_full = pd.read_parquet(val_path)
    # Strictly evaluate development rows 480-960
    dev_df = val_df_full.iloc[480:960].copy()
    del val_df_full

    target_channel = "L_T1"
    clean_series = dev_df[target_channel].to_numpy(dtype=np.float64)
    N = len(clean_series)

    # 2. Load frozen champion TCN surrogate
    ckpt_path = Path("results/surrogates/vista_tcn_48h_150ep/tcn.pt")
    surrogate = DilatedTCNSurrogate.load(ckpt_path)
    sigma_eps = 0.0418  # Champion surrogate residual noise std (meters)
    print(f"Loaded champion TCN. Baseline noise floor sigma_eps: {sigma_eps:.4f} m")

    # Predict clean baseline
    clean_pred_res = surrogate.predict(dev_df)
    clean_series = dev_df[target_channel].to_numpy(dtype=np.float64)

    # --------------------------------------------------------------------------
    # PART 1: Imperceptible-Tier Sweep (0.25*sigma_eps to 3.0*sigma_eps)
    # --------------------------------------------------------------------------
    print("\n--- Part 1: Imperceptible-Tier Attack Sweep ---")
    multipliers = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    imperceptible_results = []

    # Estimator: Multi-order trend filter
    estimator = MultiOrderTrendFilterEstimator(
        target_channel=target_channel,
        bounds=(0.264, 4.908),
        sigma_epsilon=sigma_eps,
        lambda_0=0.05,
        lambda_1=0.50,
        lambda_2=1.00,
    )

    onset = 100
    duration = 48  # 48-hour attack

    for mult in multipliers:
        amp = mult * sigma_eps  # Amplitude in meters

        for fam in ["step", "drift"]:
            delta_true = np.zeros(N, dtype=np.float64)
            if fam == "step":
                delta_true[onset : onset + duration] = amp
            else:
                # Drift accumulates from 0 to amp
                delta_true[onset : onset + duration] = amp * (np.arange(duration) / (duration - 1))

            y_obs = clean_series + delta_true
            result = estimator.reconstruct(y_obs, surrogate_x=clean_pred_res.y_hat)

            # Compute error on active support
            active_true = delta_true[onset : onset + duration]
            active_est = result.delta_hat[onset : onset + duration]
            nrmse = np.sqrt(np.mean((active_true - active_est) ** 2)) / (amp + 1e-8)
            est_amp = float(np.mean(active_est))

            # Non-zero detection fraction (did regularizer shrink it to zero?)
            non_zero_frac = float(np.mean(np.abs(active_est) > 0.005))

            imperceptible_results.append({
                "multiplier": mult,
                "amplitude_m": amp,
                "family": fam,
                "delta_nrmse": float(nrmse),
                "estimated_amplitude_m": est_amp,
                "non_zero_fraction": non_zero_frac,
            })
            print(f"  [{fam.upper():5s}] Amp: {amp:.4f}m ({mult:4.2f}*sigma) | NRMSE: {nrmse:.4f} | Non-Zero: {non_zero_frac*100:5.1f}% | Est Amp: {est_amp:.4f}m")

    # --------------------------------------------------------------------------
    # PART 2: Differentiable Adversarial Attack on Companion Telemetry (TS-PGD)
    # --------------------------------------------------------------------------
    print("\n--- Part 2: Adversarial TS-PGD Perturbation on Companion Sensors ---")
    adversarial_results = []

    # Model in PyTorch for autograd
    tcn_model = surrogate.model
    tcn_model.eval()

    # Take lookback window L=48
    lookback = surrogate.context_length
    t_eval = 200
    comp_cols = list(surrogate.companion_channels)
    x_comp_raw = dev_df[comp_cols].iloc[t_eval - lookback : t_eval].to_numpy(dtype=np.float32)

    # Normalize companion input
    comp_mean = surrogate.feature_mean_
    comp_std = surrogate.feature_std_
    x_norm = (x_comp_raw - comp_mean) / (comp_std + 1e-8)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)  # [1, L, C]

    y_target_clean = float(clean_series[t_eval])
    y_target_tensor = torch.tensor([y_target_clean], dtype=torch.float32)

    eps_levels = [0.01, 0.02, 0.05, 0.10, 0.20]  # Max norm perturbation on normalized scale (1% to 20%)

    for eps in eps_levels:
        delta_adv = torch.zeros_like(x_tensor)
        alpha = eps / 5.0  # 5-step PGD

        # 5-step TS-PGD to maximize surrogate prediction error
        for step in range(5):
            delta_adv.requires_grad_(True)
            pred = tcn_model(x_tensor + delta_adv)
            loss = nn.functional.mse_loss(pred.view(-1), y_target_tensor.view(-1))
            grad = torch.autograd.grad(loss, delta_adv)[0]

            with torch.no_grad():
                delta_adv = torch.clamp(delta_adv + alpha * grad.sign(), -eps, eps)

        # Clean vs Adversarial prediction
        with torch.no_grad():
            clean_pred = float(tcn_model(x_tensor).squeeze().item())
            adv_pred = float(tcn_model(x_tensor + delta_adv).squeeze().item())

        induced_error_m = abs(adv_pred - clean_pred)
        phantom_alarm = induced_error_m > (3.0 * sigma_eps)

        adversarial_results.append({
            "epsilon_norm": eps,
            "clean_prediction_m": clean_pred,
            "adv_prediction_m": adv_pred,
            "induced_error_m": induced_error_m,
            "phantom_alarm_triggered": bool(phantom_alarm),
        })
        print(f"  [TS-PGD] Epsilon: {eps*100:4.1f}% | Clean Pred: {clean_pred:.3f}m | Adv Pred: {adv_pred:.3f}m | Induced Error: {induced_error_m:.4f}m | Phantom Alarm: {phantom_alarm}")

    # 3. Save Summary JSON
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sigma_eps_m": sigma_eps,
        "imperceptible_tier_results": imperceptible_results,
        "adversarial_pgd_results": adversarial_results,
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 4. Generate Publication-Quality Visual Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot 1: Imperceptible Detection Curve
    df_imp = pd.DataFrame(imperceptible_results)
    for fam, color, marker in [("step", "#1f77b4", "o"), ("drift", "#ff7f0e", "s")]:
        sub = df_imp[df_imp["family"] == fam]
        ax1.plot(sub["multiplier"], sub["non_zero_fraction"] * 100, marker=marker, color=color, lw=2.5, label=f"{fam.capitalize()} Attack")

    ax1.axvline(1.0, color="#d62728", ls="--", lw=1.5, label=r"Noise Floor ($1\sigma_\epsilon \approx 0.042\,\mathrm{m}$)")
    ax1.axvline(3.0, color="#7f7f7f", ls=":", lw=1.5, label=r"Stealth Limit ($3\sigma_\epsilon \approx 0.125\,\mathrm{m}$)")
    ax1.set_xlabel(r"Attack Amplitude Multiplier ($\times \sigma_\epsilon$)", fontsize=13)
    ax1.set_ylabel("Solver Non-Zero Recovery Rate (%)", fontsize=13)
    ax1.set_title("Imperceptible-Tier Recovery Boundary", fontsize=14, fontweight="bold")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=11)

    # Plot 2: Adversarial Vulnerability Curve
    df_adv = pd.DataFrame(adversarial_results)
    ax2.plot(df_adv["epsilon_norm"] * 100, df_adv["induced_error_m"], marker="^", color="#2ca02c", lw=2.5, label="TS-PGD Induced Error")
    ax2.axhline(3.0 * sigma_eps, color="#d62728", ls="--", lw=1.5, label=r"Detection Threshold ($3\sigma_\epsilon$)")
    ax2.set_xlabel(r"Companion Adversarial Budget $\epsilon$ (% sensor span)", fontsize=13)
    ax2.set_ylabel("Surrogate Target Error (m)", fontsize=13)
    ax2.set_title("Adversarial Vulnerability to Companion Spoofing", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "imperceptible_adversarial_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to {plot_path}")

    return summary


if __name__ == "__main__":
    run_experiments()
