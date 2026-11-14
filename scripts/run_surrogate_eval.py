"""Unified multi-dataset benchmark evaluator for plant surrogates.

Supports BATADAL, SWaT, and WADI datasets across CausalTCN and CausalPatchTST.
Evaluates attack scenarios via AttackBatchGenerator and CausalMultiOrderTrendFilterEstimator.
Outputs summary.json, summary.csv, and scenarios.parquet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks.massive_generator import AttackBatchGenerator, AttackScenario  # noqa: E402
from src.data.surrogate_loaders import DEFAULT_TARGET_CHANNELS  # noqa: E402
from src.inverse.estimators import CausalMultiOrderTrendFilterEstimator  # noqa: E402
from src.inverse.harness import compute_scenario_metrics  # noqa: E402
from src.models.base import CausalTCN  # noqa: E402
from src.models.patch_tst import CausalPatchTST  # noqa: E402

DATASET_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "batadal": (0.264, 4.908),
    "swat": (450.0, 850.0),
    "wadi": (35.0, 75.0),
}


def parse_args() -> argparse.Namespace:
    """Parse evaluator command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Multi-Dataset Surrogate Evaluator"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["batadal", "swat", "wadi"],
        help="Target dataset to evaluate on",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tcn",
        choices=["tcn", "patch_tst"],
        help="Model architecture",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained surrogate checkpoint (.pt)",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="5family",
        choices=["5family", "10k"],
        help="Evaluation benchmark mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/surrogate_eval",
        help="Output directory for evaluation artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    """Resolve target execution device with fallback."""
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def load_evaluation_data(dataset: str) -> tuple[pd.DataFrame, float]:
    """Load evaluation telemetry split and train standard deviation.

    Enforces strict data seal for BATADAL.
    """
    root = Path("data/processed") / dataset
    train_df = pd.read_parquet(root / "train.parquet")
    val_df = pd.read_parquet(root / "val.parquet")

    target_sensor = DEFAULT_TARGET_CHANNELS[dataset]
    train_std = float(train_df[target_sensor].std())

    if dataset == "batadal" and len(val_df) == 1440:
        # Enforce strict data seal: development split rows 480-960
        val_df = val_df.iloc[480:960].copy().reset_index(drop=True)

    return val_df, train_std


def compute_surrogate_predictions(
    model: torch.nn.Module,
    val_df: pd.DataFrame,
    metadata: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Compute surrogate predictions across val_df and estimate residual uncertainty."""
    model.eval()
    companion_channels = metadata["companion_channels"]
    target_channel = metadata["target_channel"]
    lookback = metadata.get("lookback", 48)

    feature_mean = np.array(metadata["feature_mean"], dtype=np.float32)
    feature_std = np.array(metadata["feature_std"], dtype=np.float32)
    target_mean = float(metadata["target_mean"])
    target_std = float(metadata["target_std"])

    raw_feats = val_df[companion_channels].to_numpy(dtype=np.float32)
    norm_feats = (raw_feats - feature_mean) / feature_std
    n_steps = len(val_df)

    clean_x_hat = np.zeros(n_steps, dtype=np.float64)

    # First lookback steps: use target mean or first prediction
    windows: list[np.ndarray] = []
    for t in range(lookback, n_steps + 1):
        windows.append(norm_feats[t - lookback : t])

    if windows:
        windows_arr = np.stack(windows, axis=0)
        x_tensor = torch.from_numpy(windows_arr).to(device)
        with torch.no_grad():
            preds_norm = model(x_tensor).cpu().numpy().flatten()
        preds_denorm = preds_norm * target_std + target_mean
        clean_x_hat[lookback - 1 :] = preds_denorm
        clean_x_hat[: lookback - 1] = preds_denorm[0]
    else:
        clean_x_hat[:] = target_mean

    # Compute residual uncertainty
    actual = val_df[target_channel].to_numpy(dtype=np.float64)
    residuals = actual - clean_x_hat
    surrogate_unc = float(np.std(residuals))
    if surrogate_unc < 1e-4:
        surrogate_unc = 0.0418

    return clean_x_hat, surrogate_unc


def main() -> None:
    """Execute unified benchmark evaluation."""
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir) / f"{args.dataset}_{args.model}_{args.eval_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("UNIFIED MULTI-DATASET SURROGATE BENCHMARK EVALUATOR")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(
        f"Dataset: {args.dataset.upper()} | Model: {args.model.upper()} | Mode: {args.eval_mode}"
    )
    print(f"Device: {device} | Checkpoint: {args.checkpoint}")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)

    # 1. Load data
    val_df, train_std = load_evaluation_data(args.dataset)
    target_sensor = DEFAULT_TARGET_CHANNELS[args.dataset]
    box_bounds = DATASET_PHYSICAL_BOUNDS[args.dataset]
    print(
        f"Evaluation Telemetry: {len(val_df)} timesteps | Target: {target_sensor} "
        f"(train_std={train_std:.4f}) | Envelope: {box_bounds}"
    )

    # 2. Load model checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_channels = len(ckpt["companion_channels"])

    if args.model == "tcn":
        model: torch.nn.Module = CausalTCN(
            in_channels=in_channels,
            hidden_channels=ckpt.get("hidden_channels", 64),
            dilations=tuple(ckpt.get("dilations", (1, 2, 4, 8))),
            kernel_size=ckpt.get("kernel_size", 3),
            dropout=0.0,
        )
    elif args.model == "patch_tst":
        model = CausalPatchTST(
            in_channels=in_channels,
            d_model=ckpt.get("d_model", 64),
            patch_len=ckpt.get("patch_len", 16),
            stride=ckpt.get("stride", 8),
            nhead=ckpt.get("nhead", 4),
            num_layers=ckpt.get("num_layers", 3),
            dropout=0.0,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    # 3. Compute baseline surrogate predictions
    clean_surrogate_x, surrogate_unc = compute_surrogate_predictions(
        model, val_df, ckpt, device
    )
    print(f"Baseline Surrogate Residual Uncertainty: {surrogate_unc:.4f}")

    # 4. Generate Attack Scenarios
    sensor_cols = [
        c for c in val_df.columns if c not in ("timestamp", "label", "attack_id")
    ]
    val_telemetry = val_df[sensor_cols].to_numpy(dtype=np.float64)
    gen = AttackBatchGenerator(val_telemetry, sensor_cols, seed=args.seed)

    b_min, b_max = box_bounds
    bounds_min = {target_sensor: b_min}
    bounds_max = {target_sensor: b_max}

    scenarios: list[AttackScenario] = []
    if args.eval_mode == "5family":
        families = ["step", "ramp", "drift", "pulse", "sinusoid"]
        configs = [
            {"dur": 12, "sigma_scale": 0.5},
            {"dur": 12, "sigma_scale": -0.5},
            {"dur": 12, "sigma_scale": 1.0},
            {"dur": 12, "sigma_scale": -1.0},
            {"dur": 24, "sigma_scale": 1.5},
            {"dur": 24, "sigma_scale": -1.5},
            {"dur": 24, "sigma_scale": 2.0},
            {"dur": 24, "sigma_scale": -2.0},
            {"dur": 48, "sigma_scale": 0.5},
            {"dur": 48, "sigma_scale": -0.5},
        ]
        for fam in families:
            for idx, cfg in enumerate(configs):
                dur = min(cfg["dur"], len(val_df) // 4)
                scale = cfg["sigma_scale"]
                amp = float(scale * train_std)
                scen_id = f"{args.dataset}_{fam}_{idx:02d}_d{dur}_amp{scale:+.1f}s"
                for onset in range(
                    10, len(val_df) - dur - 10, max(1, (len(val_df) - dur) // 50)
                ):
                    scen = gen.generate_scenario(
                        scenario_id=scen_id,
                        family=fam,
                        target_channels=[target_sensor],
                        onset_step=onset,
                        duration_steps=dur,
                        amplitude=amp,
                    )
                    chk = gen.check_physical_envelope(scen, bounds_min, bounds_max)
                    if chk["is_valid"]:
                        scenarios.append(scen)
                        break
    elif args.eval_mode == "10k":
        # Generate 1,000 scenario test corpus for high throughput evaluation
        batch_scens = gen.generate_batch(
            families=["step", "ramp", "periodic", "trapezoid", "brownian"],
            channels=[target_sensor],
            n_scenarios=1000,
        )
        scenarios.extend(batch_scens)

    print(f"Generated {len(scenarios)} attack scenarios for benchmark evaluation.")

    # Save scenarios parquet
    scenarios_parquet = out_dir / "scenarios.parquet"
    gen.save_parquet(scenarios, scenarios_parquet)
    print(f"Exported scenario ledger to {scenarios_parquet}")

    # 5. Evaluate scenarios with CausalMultiOrderTrendFilterEstimator
    estimator = CausalMultiOrderTrendFilterEstimator(
        window_size=48,
        lambda_0=0.05,
        lambda_1=0.50,
        lambda_2=1.00,
        target_channel=target_sensor,
        bounds=box_bounds,
        residual_std=surrogate_unc,
        solver="admm",
    )

    clean_x = val_df[target_sensor].to_numpy(dtype=np.float64)
    target_idx = sensor_cols.index(target_sensor)

    print(
        f"Evaluating {len(scenarios)} scenarios with CausalMultiOrderTrendFilterEstimator..."
    )
    t0_eval = time.perf_counter()
    records: list[dict[str, Any]] = []

    for scen in scenarios:
        delta_true = scen.delta[:, target_idx]
        attacked_y = clean_x + delta_true

        recon = estimator.reconstruct(attacked_y, surrogate_x=clean_surrogate_x)
        box_violations = int(
            np.sum((recon.x_hat < b_min - 1e-4) | (recon.x_hat > b_max + 1e-4))
        )

        metrics = compute_scenario_metrics(
            delta_hat=recon.delta_hat,
            delta_true=delta_true,
            x_hat=recon.x_hat,
            x_true=clean_x,
            onset=scen.onset_step,
            offset=scen.onset_step + scen.duration_steps,
            injected_amplitude=scen.amplitude,
            train_std=train_std,
            pointwise_uncertainty=recon.pointwise_uncertainty,
            estimator_name="CausalMultiOrderTrendFilter",
        )
        metrics["scenario_id"] = scen.scenario_id
        metrics["family"] = scen.family
        metrics["box_violations"] = box_violations
        records.append(metrics)

    eval_time = time.perf_counter() - t0_eval
    print(
        f"Evaluations finished in {eval_time:.2f}s ({eval_time / max(1, len(scenarios)):.3f}s / scen)."
    )

    # Save summary CSV & JSON
    df_results = pd.DataFrame.from_records(records)
    summary_csv = out_dir / "summary.csv"
    df_results.to_csv(summary_csv, index=False)
    print(f"Saved results to {summary_csv}")

    family_stats: dict[str, Any] = {}
    for fam, group in df_results.groupby("family"):
        family_stats[str(fam)] = {
            "count": int(len(group)),
            "median_delta_nrmse": float(group["delta_nrmse"].median()),
            "mean_delta_nrmse": float(group["delta_nrmse"].mean()),
            "sign_accuracy": float(group["sign_accuracy"].mean()),
            "support_f1": float(group["support_f1"].mean()),
            "coverage_90": float(group["coverage_90"].mean()),
            "box_violations": int(group["box_violations"].sum()),
        }

    summary_json_data = {
        "dataset": args.dataset,
        "model": args.model,
        "eval_mode": args.eval_mode,
        "checkpoint": args.checkpoint,
        "total_scenarios": len(scenarios),
        "total_eval_time_seconds": round(eval_time, 3),
        "total_box_violations": int(df_results["box_violations"].sum()),
        "overall_median_delta_nrmse": float(df_results["delta_nrmse"].median()),
        "overall_mean_delta_nrmse": float(df_results["delta_nrmse"].mean()),
        "overall_sign_accuracy": float(df_results["sign_accuracy"].mean()),
        "overall_coverage_90": float(df_results["coverage_90"].mean()),
        "per_family_stats": family_stats,
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_json_data, f, indent=2)
    print(f"Saved summary metrics to {summary_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
