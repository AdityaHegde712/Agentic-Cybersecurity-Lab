"""Execution script for surrogate dynamics model bake-off on BATADAL.

Targeting local execution and TACC Vista GH200 compute.
Trains and benchmarks:
- RidgeCompanionSurrogate (linear baseline)
- PhysicsGuidedMLPSurrogate (PINN with mass conservation regularization)
- DilatedTCNSurrogate (1D causal Temporal Convolutional Network)

Saves checkpoints, evaluation metrics, and summary tables to output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from src.inverse.surrogates import (
    BasePlantSurrogate,
    DilatedTCNSurrogate,
    PhysicsGuidedMLPSurrogate,
    RidgeCompanionSurrogate,
    SurrogatePrediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BATADAL Dynamics Surrogate Bake-Off Training Pipeline (TACC Vista / Local)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["ridge", "pinn", "tcn", "all"],
        default="all",
        help="Surrogate model family to train (default: 'all')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs for neural models (default: 25)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW (default: 0.001)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=24,
        help="Context lookback window in hours (default: 24)",
    )
    parser.add_argument(
        "--gamma-mass",
        type=float,
        default=0.1,
        help="Weight for physical mass rate-of-change loss (default: 0.1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Compute device for neural training (default: 'auto')",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Execute fast smoke run with 1 epoch on 100 samples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/surrogates/vista_bakeoff",
        help="Output directory for model weights and metrics (default: 'results/surrogates/vista_bakeoff')",
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
        "--manifest-path",
        type=str,
        default="configs/manifests/batadal-channel-manifest-v1.json",
        help="Path to channel manifest JSON",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return device_arg


def run_training() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    print("=" * 75)
    print("BATADAL DYNAMICS SURROGATE MODEL BAKE-OFF")
    print(f"Device: {device} | Output Directory: {output_dir}")
    if args.smoke:
        print("[SMOKE MODE ENABLED] 1 epoch, reduced sample counts.")
    print("=" * 75)

    # 1. Load Parquet Datasets
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")

    print(f"Loading datasets from {train_path} and {val_path}...")
    full_train = pd.read_parquet(train_path)
    full_val = pd.read_parquet(val_path)

    # Blocks:
    # model_fit: train rows 0:5840
    # validation split: train rows 5840:7321
    # calibration split: val rows 0:480
    fit_df = full_train.iloc[0:5840].copy()
    val_df = full_train.iloc[5840:7321].copy()
    calib_df = full_val.iloc[0:480].copy()

    epochs = 1 if args.smoke else args.epochs

    if args.smoke:
        fit_df = fit_df.iloc[:100].copy()
        val_df = val_df.iloc[:50].copy()
        calib_df = calib_df.iloc[:50].copy()

    print(
        f"Data Partition: fit={len(fit_df)} rows | val={len(val_df)} rows | "
        f"calib={len(calib_df)} rows | epochs={epochs}"
    )

    models_to_run = (
        ["ridge", "pinn", "tcn"] if args.model == "all" else [args.model]
    )

    metrics_summary: dict[str, dict] = {}
    calib_predictions: dict[str, SurrogatePrediction] = {}

    for model_key in models_to_run:
        print("\n" + "-" * 75)
        surrogate: BasePlantSurrogate

        if model_key == "ridge":
            print(f"Training Ridge Companion Surrogate (alpha=1.0)...")
            surrogate = RidgeCompanionSurrogate(alpha=1.0)
        elif model_key == "pinn":
            print(
                f"Training Physics-Guided MLP Surrogate (context={args.context_length}h, "
                f"gamma_mass={args.gamma_mass}, lr={args.lr}, device={device})..."
            )
            surrogate = PhysicsGuidedMLPSurrogate(
                context_length=args.context_length,
                gamma_mass=args.gamma_mass,
                lr=args.lr,
                epochs=epochs,
                batch_size=args.batch_size,
                device=device,
                verbose=True,
            )
        elif model_key == "tcn":
            print(
                f"Training Dilated TCN Surrogate (context={args.context_length}h, "
                f"dilations=[1, 2, 4, 8], lr={args.lr}, device={device})..."
            )
            surrogate = DilatedTCNSurrogate(
                context_length=args.context_length,
                lr=args.lr,
                epochs=epochs,
                batch_size=args.batch_size,
                device=device,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown model key: {model_key}")

        t0 = time.perf_counter()
        surrogate.fit(fit_df, val_df)
        elapsed = time.perf_counter() - t0
        print(f"Fit completed in {elapsed:.2f} seconds.")

        # Evaluate on validation split
        val_pred = surrogate.predict(val_df)
        # Evaluate on clean calibration split
        calib_pred = surrogate.predict(calib_df)
        calib_predictions[model_key] = calib_pred

        print(
            f"Validation Results  -> RMSE: {val_pred.metrics.get('rmse', 0.0):.4f} m | "
            f"MAE: {val_pred.metrics.get('mae', 0.0):.4f} m | "
            f"R2: {val_pred.metrics.get('r2', 0.0):.4f} | "
            f"Res Std: {val_pred.residual_std:.4f} m"
        )
        print(
            f"Calibration Results -> RMSE: {calib_pred.metrics.get('rmse', 0.0):.4f} m | "
            f"MAE: {calib_pred.metrics.get('mae', 0.0):.4f} m | "
            f"R2: {calib_pred.metrics.get('r2', 0.0):.4f} | "
            f"Res Std: {calib_pred.residual_std:.4f} m"
        )

        # Save checkpoint
        ckpt_path = output_dir / f"{model_key}.pt"
        surrogate.save(ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

        metrics_summary[model_key] = {
            "model_type": surrogate.__class__.__name__,
            "training_time_seconds": round(elapsed, 3),
            "epochs": epochs,
            "device": device,
            "checkpoint_path": str(ckpt_path),
            "val_metrics": {k: round(v, 5) for k, v in val_pred.metrics.items()},
            "val_residual_std": round(float(val_pred.residual_std), 5),
            "calib_metrics": {k: round(v, 5) for k, v in calib_pred.metrics.items()},
            "calib_residual_std": round(float(calib_pred.residual_std), 5),
        }

    # 4. Save aggregate metrics.json
    metrics_path = output_dir / "metrics.json"
    full_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "smoke_mode": args.smoke,
            "fit_rows": len(fit_df),
            "val_rows": len(val_df),
            "calib_rows": len(calib_df),
            "target_channel": "L_T1",
        },
        "models": metrics_summary,
    }
    metrics_path.write_text(json.dumps(full_output, indent=2), encoding="utf-8")
    print(f"\nSaved aggregate metrics to {metrics_path}")

    # 5. Generate diagnostic plots
    plots_dir = output_dir / "plots"
    _generate_plots(calib_df, calib_predictions, metrics_summary, plots_dir)
    print(f"Generated diagnostic plots in {plots_dir}")

    # 6. Generate markdown summary report
    report_path = output_dir / "BAKEOFF_SUMMARY.md"
    _generate_markdown_summary(full_output, report_path)
    print(f"Saved bakeoff markdown report to {report_path}")
    print("\n" + "=" * 75)
    print("Surrogate Dynamics Bake-Off completed successfully.")
    print("=" * 75)


def _generate_plots(
    calib_df: pd.DataFrame,
    calib_predictions: dict[str, SurrogatePrediction],
    metrics_summary: dict[str, dict],
    plots_dir: Path,
) -> None:
    """Generate diagnostic visualization plots for calibration performance."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_styles = {
        "ridge": {"label": "RidgeCompanionSurrogate", "color": "#1f77b4", "linestyle": "--"},
        "pinn": {"label": "PhysicsGuidedMLPSurrogate", "color": "#d62728", "linestyle": "-."},
        "tcn": {"label": "DilatedTCNSurrogate", "color": "#2ca02c", "linestyle": ":"},
    }

    hours = np.arange(len(calib_df))
    y_true = calib_df["L_T1"].to_numpy()

    # -------------------------------------------------------------------------
    # Plot 1: plots/surrogate_predictions_calibration.png (12, 6, dpi=180)
    # -------------------------------------------------------------------------
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=180)

    # Panel 1: Complete calibration trajectory
    ax1.plot(hours, y_true, color="black", linewidth=1.8, label="Ground Truth ($y_{\\text{obs}}$)", zorder=10)
    for key, pred in calib_predictions.items():
        style = model_styles.get(key, {"label": key, "color": "#9467bd", "linestyle": "-"})
        ax1.plot(
            hours,
            pred.y_hat,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.4,
            alpha=0.85,
        )
    ax1.set_title(f"Complete Calibration Horizon ({len(hours)}h) — Tank Level $L_{{T1}}$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Water Level (m)", fontsize=10)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Panel 2: 72-hour focal zoom (or up to len(hours))
    zoom_len = min(72, len(hours))
    ax2.plot(hours[:zoom_len], y_true[:zoom_len], color="black", linewidth=1.8, label="Ground Truth ($y_{\\text{obs}}$)", zorder=10)
    for key, pred in calib_predictions.items():
        style = model_styles.get(key, {"label": key, "color": "#9467bd", "linestyle": "-"})
        ax2.plot(
            hours[:zoom_len],
            pred.y_hat[:zoom_len],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.4,
            alpha=0.85,
        )
    ax2.set_title(f"Diurnal Phase Tracking & Noise Characteristics (First {zoom_len}h Zoom)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (hours)", fontsize=10)
    ax2.set_ylabel("Water Level (m)", fontsize=10)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle="--")

    fig1.tight_layout()
    fig1.savefig(plots_dir / "surrogate_predictions_calibration.png", dpi=180)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Plot 2: plots/surrogate_residuals_distribution.png (9, 5, dpi=180)
    # -------------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="Zero Error ($r=0$)")

    for key, pred in calib_predictions.items():
        style = model_styles.get(key, {"label": key, "color": "#9467bd", "linestyle": "-"})
        res = pred.residuals
        sigma = pred.residual_std
        color = style["color"]
        label_name = f"{style['label']} ($\\sigma_\\epsilon={sigma:.3f}$ m)"

        n_bins = min(25, max(6, len(res) // 3))
        ax.hist(res, bins=n_bins, density=True, alpha=0.25, color=color)

        if len(res) > 3 and np.ptp(res) > 1e-5:
            try:
                kde = stats.gaussian_kde(res)
                span = max(float(np.ptp(res)), 0.5)
                x_grid = np.linspace(res.min() - 0.2 * span, res.max() + 0.2 * span, 200)
                ax.plot(x_grid, kde(x_grid), color=color, linewidth=2.0, label=label_name)
            except Exception:
                ax.plot([], [], color=color, linewidth=2.0, label=label_name)
        else:
            ax.plot([], [], color=color, linewidth=2.0, label=label_name)

    ax.set_title("Surrogate Calibration Residual Distribution ($r_t = y_{\\text{obs}, t} - \\hat{x}_t$)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Residual Error (m)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig2.tight_layout()
    fig2.savefig(plots_dir / "surrogate_residuals_distribution.png", dpi=180)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Plot 3: plots/surrogate_metrics_comparison.png (8, 4.5, dpi=180)
    # -------------------------------------------------------------------------
    fig3, ax_bar = plt.subplots(figsize=(8, 4.5), dpi=180)
    keys = list(metrics_summary.keys())
    x = np.arange(len(keys))
    width = 0.25

    val_rmse = [metrics_summary[k]["val_metrics"]["rmse"] for k in keys]
    calib_rmse = [metrics_summary[k]["calib_metrics"]["rmse"] for k in keys]
    calib_sigma = [metrics_summary[k]["calib_residual_std"] for k in keys]

    rects1 = ax_bar.bar(x - width, val_rmse, width, label="Val RMSE (m)", color="#4575b4", alpha=0.9)
    rects2 = ax_bar.bar(x, calib_rmse, width, label="Calib RMSE (m)", color="#74add1", alpha=0.9)
    rects3 = ax_bar.bar(x + width, calib_sigma, width, label=r"Calib $\sigma_\epsilon$ (m)", color="#f46d43", alpha=0.9)

    ax_bar.set_xticks(x)
    labels = [metrics_summary[k]["model_type"].replace("Surrogate", "") for k in keys]
    ax_bar.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax_bar.set_ylabel("Error / Spread (m)", fontsize=10)
    ax_bar.set_title("Surrogate Metrics Comparison (Validation vs. Calibration)", fontsize=11, fontweight="bold")
    ax_bar.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax_bar.grid(True, axis="y", alpha=0.3, linestyle="--")

    for rects in (rects1, rects2, rects3):
        for rect in rects:
            height = rect.get_height()
            ax_bar.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig3.tight_layout()
    fig3.savefig(plots_dir / "surrogate_metrics_comparison.png", dpi=180)
    plt.close(fig3)


def _generate_markdown_summary(data: dict, report_path: Path) -> None:
    meta = data["metadata"]
    models = data["models"]

    lines = [
        "# BATADAL Dynamics Surrogate Model Bake-Off Summary",
        "",
        "> **Target Sensor:** `L_T1`  ",
        f"> **Execution Mode:** {'Smoke Run' if meta['smoke_mode'] else 'Full Training'}  ",
        f"> **Compute Device:** `{meta['device']}` (CUDA Available: `{meta['cuda_available']}`)  ",
        f"> **Timestamp:** `{meta['timestamp']}`  ",
        "",
        "## Performance Comparison",
        "",
        "| Model | Val RMSE (m) | Val MAE (m) | Val $R^2$ | Calib RMSE (m) | Calib Res $\\sigma$ (m) | Train Time (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key, info in models.items():
        v = info["val_metrics"]
        c = info["calib_metrics"]
        lines.append(
            f"| **{info['model_type']}** (`{key}`) | "
            f"{v.get('rmse', 0.0):.4f} | {v.get('mae', 0.0):.4f} | {v.get('r2', 0.0):.4f} | "
            f"{c.get('rmse', 0.0):.4f} | {info['calib_residual_std']:.4f} | {info['training_time_seconds']:.2f} |"
        )

    lines.extend([
        "",
        "## Diagnostic Visualizations",
        "",
        "### 1. Calibration Trajectory & Diurnal Phase Tracking",
        "![Surrogate Calibration Predictions](plots/surrogate_predictions_calibration.png)",
        "",
        "### 2. Residual Distribution & Noise Scale ($\\sigma_\\epsilon$)",
        "![Surrogate Residual Distribution](plots/surrogate_residuals_distribution.png)",
        "",
        "### 3. Model Metrics Comparison",
        "![Surrogate Metrics Comparison](plots/surrogate_metrics_comparison.png)",
        "",
        "## Artifact Ledger",
        "",
    ])
    for key, info in models.items():
        lines.append(f"- **{info['model_type']} Checkpoint**: `{info['checkpoint_path']}`")
    lines.append(f"- **Metrics JSON**: `{report_path.parent / 'metrics.json'}`")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_training()
