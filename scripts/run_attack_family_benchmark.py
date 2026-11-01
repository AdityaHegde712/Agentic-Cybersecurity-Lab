"""Attack Family Generalization Benchmark Runner (Sprint 4B).

Evaluates generalization across 5 distinct synthetic attack families:
1. Step attacks (instantaneous step injection)
2. Ramp attacks (linearly increasing slope over attack window)
3. Linear drifts (sustained unbounded slope injection)
4. Transient pulses (short boxcar pulses tau = 3h)
5. Sinusoids (frequency spoofing injection)

Estimators evaluated:
1. PersistenceEstimator (Statistical Baseline)
2. StepMAP_TCN (Phase 3 Causal Champion with pure first-order TV, lambda_2 = 0)
3. CausalMultiOrderTrendFilter_TCN (Sprint 4B Multi-Order Trend Filter, lambda_2 > 0)

Surrogate:
Stage 1 Champion Dilated TCN (`results/surrogates/vista_tcn_48h_150ep/tcn.pt`)
completely frozen and unmodified.

Data Seal:
Evaluates rows 480-960 of BATADAL val.parquet.
Strictly asserts zero access or evaluation on the sealed final test block (rows 960-1440).
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.inverse.benchmark_manifest import BenchmarkManifest
from src.inverse.channel_constraints import ChannelConstraintManifest
from src.inverse.estimators import (
    CausalMultiOrderTrendFilterEstimator,
    CausalStepMAPEstimator,
    PersistenceEstimator,
)
from src.inverse.harness import (
    BenchmarkScenario,
    compute_scenario_metrics,
)
from src.inverse.surrogates import DilatedTCNSurrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sprint 4B Attack Family Generalization Benchmark"
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="configs/manifests/batadal-benchmark-manifest-v1.json",
        help="Path to benchmark manifest JSON",
    )
    parser.add_argument(
        "--channel-manifest-path",
        type=str,
        default="configs/manifests/batadal-channel-manifest-v1.json",
        help="Path to channel constraint manifest JSON",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/surrogates/vista_tcn_48h_150ep/tcn.pt",
        help="Path to trained champion surrogate checkpoint (.pt)",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/processed/batadal/train.parquet",
        help="Path to BATADAL train.parquet",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default="data/processed/batadal/val.parquet",
        help="Path to BATADAL val.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/narrow_inverse_research/attack_family_generalization",
        help="Output directory for benchmark artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Compute device for neural inference",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_arg


def build_attack_trajectory(
    family: str,
    n_total: int,
    onset: int,
    offset: int,
    amplitude: float,
) -> np.ndarray:
    """Construct ground truth additive attack trajectory delta for given attack family."""
    delta = np.zeros(n_total, dtype=np.float64)
    duration = offset - onset

    if family == "step":
        delta[onset:offset] = amplitude
    elif family == "ramp":
        ramp_vals = amplitude * (np.arange(duration, dtype=np.float64) / max(1, duration - 1))
        delta[onset:offset] = ramp_vals
    elif family == "drift":
        slope = amplitude / max(1, duration)
        delta[onset:offset] = slope * np.arange(duration, dtype=np.float64)
    elif family == "pulse":
        pulse_len = min(3, duration)
        delta[onset : onset + pulse_len] = amplitude
    elif family == "sinusoid":
        period = 12.0
        t_arr = np.arange(duration, dtype=np.float64)
        delta[onset:offset] = amplitude * np.sin(2.0 * np.pi * t_arr / period)
    else:
        raise ValueError(f"Unknown attack family: {family}")

    return delta


def generate_attack_family_scenarios(
    train_std: float,
    dev_length: int = 480,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate deterministic stratified scenarios across 5 attack families."""
    rng = np.random.default_rng(seed)
    families = ["step", "ramp", "drift", "pulse", "sinusoid"]

    # Parameter grid per family
    amplitudes_std = [0.5, 1.0, 1.5, 2.0]
    signs = [1, -1]
    durations = [12, 24, 48]

    scenarios: list[dict[str, Any]] = []

    for fam in families:
        # Create 10 deterministic scenarios per family
        for s_idx in range(10):
            amp_std = amplitudes_std[s_idx % len(amplitudes_std)]
            sign = signs[(s_idx // len(amplitudes_std)) % len(signs)]
            dur = durations[s_idx % len(durations)]
            injected_amp = sign * amp_std * train_std

            # Buffer 24h at start and end of 480h block
            min_onset = 24
            max_onset = dev_length - dur - 24
            onset = int(min_onset + (s_idx * 37) % max(1, max_onset - min_onset))
            offset = onset + dur

            scenarios.append({
                "scenario_id": f"{fam}_amp{amp_std:.1f}_dur{dur:02d}_sign{sign:+d}_idx{s_idx:02d}",
                "family": fam,
                "amplitude_train_std": float(amp_std),
                "sign": int(sign),
                "duration_hours": int(dur),
                "onset_index": onset,
                "offset_index": offset,
                "injected_amplitude": float(injected_amp),
            })

    return scenarios


def run_benchmark() -> dict[str, Any]:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    print("=" * 80)
    print("SPRINT 4B: ATTACK FAMILY GENERALIZATION BENCHMARK")
    print(f"Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"Device: {device} | Output Directory: {output_dir}")
    print("=" * 80)

    # 1. Load manifests
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = BenchmarkManifest.from_json(manifest_path.read_text(encoding="utf-8").strip())
    target_channel = manifest.target_sensor

    channel_manifest_path = Path(args.channel_manifest_path)
    channel_manifest = ChannelConstraintManifest.from_json(
        channel_manifest_path.read_text(encoding="utf-8").strip()
    )
    target_ch_meta = next(c for c in channel_manifest.channels if c.name == target_channel)
    box_bounds = (target_ch_meta.empirical_bounds.minimum, target_ch_meta.empirical_bounds.maximum)
    print(f"Target Channel: {target_channel} | Physical Bounds: [{box_bounds[0]:.4f}, {box_bounds[1]:.4f}] m")

    # 2. Data loading and strict data seal validation
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    train_df = pd.read_parquet(train_path)
    val_df_full = pd.read_parquet(val_path)

    assert len(val_df_full) == 1440, f"Expected 1440 val rows, found {len(val_df_full)}"
    dev_block = next(b for b in manifest.blocks if b.name == "development_injection")
    calib_block = next(b for b in manifest.blocks if b.name == "clean_calibration")
    sealed_block = next(b for b in manifest.blocks if b.name == "sealed_final_injection")

    assert dev_block.row_start == 480 and dev_block.row_stop == 960
    assert sealed_block.row_start == 960 and sealed_block.row_stop == 1440

    calib_df = val_df_full.iloc[calib_block.row_start : calib_block.row_stop].copy()
    dev_df = val_df_full.iloc[dev_block.row_start : dev_block.row_stop].copy()
    calib_series = calib_df[target_channel].to_numpy(dtype=np.float64)

    # STRICT DATA SEAL: Purge sealed final block from memory
    del val_df_full
    print("Data Partition Verified: Sealed rows 960-1440 strictly untouched.")

    train_std = float(train_df[target_channel].std())
    print(f"Normal dynamics train_std: {train_std:.4f} m")

    # 3. Load champion plant surrogate (Dilated TCN) - frozen
    ckpt_path = Path(args.checkpoint)
    print(f"Loading frozen champion TCN surrogate from {ckpt_path}...")
    tcn_surrogate = DilatedTCNSurrogate.load(ckpt_path)
    tcn_surrogate.device_str = device
    if hasattr(tcn_surrogate, "model") and tcn_surrogate.model is not None:
        tcn_surrogate.model = tcn_surrogate.model.to(device)
        tcn_surrogate.model.eval()

    # Precompute causal surrogate prediction once across dev_df (companion SCADA is uncorrupted)
    print("Computing baseline causal surrogate predictions on development telemetry...")
    surr_res = tcn_surrogate.reconstruct(dev_df)
    clean_surrogate_x = surr_res.x_hat
    surrogate_unc = float(np.mean(surr_res.pointwise_uncertainty)) if surr_res.pointwise_uncertainty is not None else 0.0418
    print(f"Surrogate residual uncertainty: {surrogate_unc:.4f} m")

    clean_x_dev = dev_df[target_channel].to_numpy(dtype=np.float64)
    n_dev = len(clean_x_dev)

    # 4. Generate Attack Scenarios across 5 families
    scenarios = generate_attack_family_scenarios(train_std=train_std, dev_length=n_dev, seed=manifest.seeds["injection"])
    print(f"Generated {len(scenarios)} attack scenarios across 5 families (10 per family).")

    # 5. Initialize Estimators
    p_est = PersistenceEstimator(lag=1).fit_normal(calib_series)

    # StepMAP_TCN (Phase 3 champion, lambda_2 = 0)
    step_map_tcn = CausalStepMAPEstimator(
        window_size=48,
        lambda_tv=1.0,
        lambda_l1=0.05,
        target_channel=target_channel,
        bounds=box_bounds,
        residual_std=surrogate_unc,
        solver="admm",
    )

    # CausalMultiOrderTrendFilter_TCN (Sprint 4B multi-order filter, lambda_2 > 0)
    multi_order_tcn = CausalMultiOrderTrendFilterEstimator(
        window_size=48,
        lambda_0=0.05,
        lambda_1=0.50,
        lambda_2=1.00,
        target_channel=target_channel,
        bounds=box_bounds,
        residual_std=surrogate_unc,
        solver="admm",
    )

    estimators = [
        ("PersistenceEstimator", p_est),
        ("StepMAP_TCN", step_map_tcn),
        ("CausalMultiOrderTrendFilter_TCN", multi_order_tcn),
    ]

    # 6. Execute Benchmark
    print(f"\nExecuting {len(estimators)} estimators across {len(scenarios)} scenarios ({len(estimators) * len(scenarios)} evaluations)...")
    records: list[dict[str, Any]] = []
    sample_traces: dict[str, dict[str, np.ndarray]] = {}
    b_min, b_max = box_bounds

    t_start = time.perf_counter()

    for s_idx, sc in enumerate(scenarios):
        fam = sc["family"]
        onset = sc["onset_index"]
        offset = sc["offset_index"]
        amp = sc["injected_amplitude"]

        delta_true = build_attack_trajectory(
            family=fam,
            n_total=n_dev,
            onset=onset,
            offset=offset,
            amplitude=amp,
        )

        attacked_y = clean_x_dev + delta_true
        attacked_df = dev_df.copy()
        attacked_df[target_channel] = attacked_y

        trace_dict: dict[str, np.ndarray] = {
            "y": attacked_y,
            "x_true": clean_x_dev,
            "delta_true": delta_true,
        }

        for est_name, est in estimators:
            if isinstance(est, PersistenceEstimator):
                res = est.reconstruct(
                    attacked_df,
                    attack_interval=None,
                    target_channel=target_channel,
                    prior_clean_history=calib_series,
                )
            else:
                res = est.reconstruct(
                    attacked_y,
                    surrogate_x=clean_surrogate_x,
                    attack_interval=None,
                )

            trace_dict[f"delta_hat_{est_name}"] = res.delta_hat
            trace_dict[f"x_hat_{est_name}"] = res.x_hat

            # Hard constraint violations
            hard_violations = int(np.sum((res.x_hat < b_min - 1e-4) | (res.x_hat > b_max + 1e-4)))

            # Compute scenario metrics
            b_sc = BenchmarkScenario(
                scenario_id=sc["scenario_id"],
                target_channel=target_channel,
                amplitude_train_std=sc["amplitude_train_std"],
                sign=sc["sign"],
                duration_hours=sc["duration_hours"],
                onset_index=onset,
                offset_index=offset,
                regime="development",
                injected_amplitude=amp,
            )

            # For NRMSE denominator, use peak attack amplitude for non-constant geometries
            peak_amp = max(abs(amp), float(np.max(np.abs(delta_true[onset:offset]))), 1e-6)
            interval = slice(onset, offset)
            interval_rmse = float(np.sqrt(np.mean((res.delta_hat[interval] - delta_true[interval]) ** 2)))
            delta_nrmse = float(interval_rmse / peak_amp)

            support_threshold = float(0.1 * peak_amp)
            pred_support = np.abs(res.delta_hat) > support_threshold
            true_support = np.abs(delta_true) > support_threshold
            tp = int(np.logical_and(pred_support, true_support).sum())
            fp = int(np.logical_and(pred_support, ~true_support).sum())
            fn = int(np.logical_and(~pred_support, true_support).sum())
            precision = float(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recall = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            support_f1 = float((2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)

            med_delta = float(np.median(res.delta_hat[interval]))
            signed_amp_err = float((med_delta - amp) / train_std)
            sign_acc = 1.0 if np.sign(med_delta) == np.sign(amp) else 0.0

            m = {
                "scenario_id": sc["scenario_id"],
                "family": fam,
                "estimator_name": est_name,
                "amplitude_train_std": sc["amplitude_train_std"],
                "duration_hours": sc["duration_hours"],
                "sign": sc["sign"],
                "injected_amplitude": amp,
                "onset_index": onset,
                "offset_index": offset,
                "delta_nrmse": delta_nrmse,
                "support_f1": support_f1,
                "signed_amplitude_error_std": signed_amp_err,
                "sign_accuracy": sign_acc,
                "hard_constraint_violations": hard_violations,
            }
            records.append(m)

        # Save trace sample for plotting (first scenario of each family)
        if fam not in sample_traces:
            sample_traces[fam] = {
                "onset": onset,
                "offset": offset,
                **trace_dict,
            }

        if (s_idx + 1) % 10 == 0:
            print(f"Processed {s_idx + 1}/{len(scenarios)} scenarios...")

    total_time = time.perf_counter() - t_start
    print(f"Evaluations finished in {total_time:.2f} s ({total_time / len(records) * 1000:.2f} ms/eval).")

    results_df = pd.DataFrame(records)

    # 7. Aggregate Metrics & Stratify by Family
    summary: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "target_channel": target_channel,
            "train_std": train_std,
            "total_scenarios": len(scenarios),
            "evaluations_count": len(records),
            "total_time_seconds": total_time,
            "checkpoint": str(ckpt_path),
            "box_bounds": list(box_bounds),
        },
        "overall": {},
        "by_family": {},
        "marginal_reduction_tf_vs_tv": {},
    }

    for est_name, _ in estimators:
        sub = results_df[results_df["estimator_name"] == est_name]
        summary["overall"][est_name] = {
            "median_delta_nrmse": float(sub["delta_nrmse"].median()),
            "mean_delta_nrmse": float(sub["delta_nrmse"].mean()),
            "median_support_f1": float(sub["support_f1"].median()),
            "mean_support_f1": float(sub["support_f1"].mean()),
            "median_signed_amplitude_error_std": float(sub["signed_amplitude_error_std"].median()),
            "sign_accuracy": float(sub["sign_accuracy"].mean()),
            "total_hard_violations": int(sub["hard_constraint_violations"].sum()),
        }

    families = ["step", "ramp", "drift", "pulse", "sinusoid"]
    for fam in families:
        summary["by_family"][fam] = {}
        for est_name, _ in estimators:
            sub = results_df[(results_df["family"] == fam) & (results_df["estimator_name"] == est_name)]
            summary["by_family"][fam][est_name] = {
                "median_delta_nrmse": float(sub["delta_nrmse"].median()),
                "mean_delta_nrmse": float(sub["delta_nrmse"].mean()),
                "median_support_f1": float(sub["support_f1"].median()),
                "mean_support_f1": float(sub["support_f1"].mean()),
                "sign_accuracy": float(sub["sign_accuracy"].mean()),
                "total_hard_violations": int(sub["hard_constraint_violations"].sum()),
            }

        # Compute marginal NRMSE reduction of TrendFilter vs pure TV
        tv_nrmse = summary["by_family"][fam]["StepMAP_TCN"]["median_delta_nrmse"]
        tf_nrmse = summary["by_family"][fam]["CausalMultiOrderTrendFilter_TCN"]["median_delta_nrmse"]
        reduction_pct = float((tv_nrmse - tf_nrmse) / tv_nrmse * 100.0) if tv_nrmse > 0 else 0.0
        summary["marginal_reduction_tf_vs_tv"][fam] = {
            "tv_median_nrmse": tv_nrmse,
            "tf_median_nrmse": tf_nrmse,
            "reduction_percentage": reduction_pct,
        }

    # 8. Save Artifacts
    summary_csv_path = output_dir / "summary.csv"
    results_df.to_csv(summary_csv_path, index=False)
    print(f"Saved scenario records to {summary_csv_path}")

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary JSON to {summary_json_path}")

    # 9. Generate Comparison Plots
    _generate_plots(results_df, summary, sample_traces, plots_dir)

    print("=" * 80)
    print("BENCHMARK SUMMARY BY ATTACK FAMILY (MEDIAN DELTA NRMSE):")
    print(f"{'Family':<12} | {'Persistence':<12} | {'StepMAP (TV)':<14} | {'MultiOrder (TF)':<16} | {'TF Improvement':<14}")
    print("-" * 80)
    for fam in families:
        p_val = summary["by_family"][fam]["PersistenceEstimator"]["median_delta_nrmse"]
        tv_val = summary["by_family"][fam]["StepMAP_TCN"]["median_delta_nrmse"]
        tf_val = summary["by_family"][fam]["CausalMultiOrderTrendFilter_TCN"]["median_delta_nrmse"]
        imp = summary["marginal_reduction_tf_vs_tv"][fam]["reduction_percentage"]
        print(f"{fam:<12} | {p_val:<12.4f} | {tv_val:<14.4f} | {tf_val:<16.4f} | {imp:+6.2f}%")
    print("=" * 80)

    return summary


def _generate_plots(
    df: pd.DataFrame,
    summary: dict[str, Any],
    traces: dict[str, dict[str, Any]],
    plots_dir: Path,
) -> None:
    """Generate high-resolution comparison plots."""
    families = ["step", "ramp", "drift", "pulse", "sinusoid"]
    estimators = ["PersistenceEstimator", "StepMAP_TCN", "CausalMultiOrderTrendFilter_TCN"]
    labels = ["Persistence", "StepMAP (TV)", "MultiOrder (TF)"]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]

    # Plot 1: Grouped bar chart comparing median NRMSE
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    x = np.arange(len(families))
    width = 0.25

    for idx, (est_name, label, color) in enumerate(zip(estimators, labels, colors)):
        vals = [summary["by_family"][fam][est_name]["median_delta_nrmse"] for fam in families]
        ax.bar(x + (idx - 1) * width, vals, width, label=label, color=color, alpha=0.9, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in families], fontsize=11, fontweight="bold")
    ax.set_ylabel("Median Delta NRMSE", fontsize=11)
    ax.set_title("Attack Family Generalization: NRMSE by Morphology", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    p1_path = plots_dir / "family_nrmse_comparison.png"
    fig.savefig(p1_path)
    plt.close(fig)
    print(f"Saved plot: {p1_path}")

    # Plot 2: Marginal reduction percentage of TF over TV
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    reductions = [summary["marginal_reduction_tf_vs_tv"][fam]["reduction_percentage"] for fam in families]
    bar_colors = ["#2ca02c" if r >= 0 else "#d62728" for r in reductions]
    bars = ax.bar(families, reductions, color=bar_colors, edgecolor="black", width=0.5)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.axhline(15.0, color="#d62728", linestyle="--", linewidth=1.5, label="Target Gate (>= 15% on Ramps/Drifts)")

    for bar, r in zip(bars, reductions):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax.annotate(f"{r:+.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha="center", va=va, fontweight="bold", fontsize=10)

    ax.set_ylabel("NRMSE Reduction over Pure TV (%)", fontsize=11)
    ax.set_title("Marginal Benefit of Second-Order Trend Filtering (lambda_2 > 0)", fontsize=12, fontweight="bold")
    ax.set_xticklabels([f.capitalize() for f in families], fontsize=11, fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    p2_path = plots_dir / "nrmse_reduction_by_family.png"
    fig.savefig(p2_path)
    plt.close(fig)
    print(f"Saved plot: {p2_path}")

    # Plot 3: 5-row sample reconstruction trace showcase
    fig, axes = plt.subplots(len(families), 1, figsize=(12, 10), sharex=False, dpi=200)
    for idx, fam in enumerate(families):
        ax = axes[idx]
        tr = traces[fam]
        onset = tr["onset"]
        offset = tr["offset"]
        t_view = slice(max(0, onset - 12), min(len(tr["delta_true"]), offset + 24))
        t_axis = np.arange(len(tr["delta_true"]))[t_view]

        ax.plot(t_axis, tr["delta_true"][t_view], "k--", linewidth=2.0, label="True Delta", alpha=0.8)
        ax.plot(t_axis, tr["delta_hat_PersistenceEstimator"][t_view], color="#7f7f7f", linewidth=1.2, linestyle=":", label="Persistence")
        ax.plot(t_axis, tr["delta_hat_StepMAP_TCN"][t_view], color="#1f77b4", linewidth=1.5, label="StepMAP (TV)")
        ax.plot(t_axis, tr["delta_hat_CausalMultiOrderTrendFilter_TCN"][t_view], color="#2ca02c", linewidth=2.0, label="MultiOrder (TF)")

        ax.axvspan(onset, offset, color="#ffffcc", alpha=0.4, label="Active Attack Interval" if idx == 0 else "")
        ax.set_ylabel(r"$\delta$ (m)", fontsize=10)
        ax.set_title(f"Attack Family: {fam.capitalize()}", fontsize=11, fontweight="bold", loc="left")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", ncol=5, fontsize=8)

    axes[-1].set_xlabel("Time (hours)", fontsize=11)
    fig.tight_layout()
    p3_path = plots_dir / "attack_trajectories_sample.png"
    fig.savefig(p3_path)
    plt.close(fig)
    print(f"Saved plot: {p3_path}")


if __name__ == "__main__":
    run_benchmark()
