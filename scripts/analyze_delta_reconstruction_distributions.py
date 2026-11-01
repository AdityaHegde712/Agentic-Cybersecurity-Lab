"""Statistical Distribution and Error Modeling Runner for Sprint 4A.

Analyzes empirical reconstruction errors from the sealed final benchmark:
1. Ingests sealed benchmark results (summary.csv and summary.json).
2. Fits parametric distribution models (Gaussian, Student-t, GED) across all estimators
   and stratified by amplitude (0.75 sigma, 1.5 sigma, 3.0 sigma).
3. Evaluates clean unattacked telemetry (rows 960-1440 of val.parquet) with StepMAPEstimator
   and CausalStepMAPEstimator to profile the empirical null distribution p(delta_hat | delta = 0).
4. Computes operational ROC detection curves and optimal threshold tables (tau* for FAR in {0.001, 0.01, 0.05}).
5. Generates 5 publication-ready diagnostic plots:
   - error_distribution_fits.png
   - qq_plots.png
   - null_distribution_clean.png
   - roc_curves_by_amplitude.png
   - error_vs_amplitude_regime.png
6. Saves summary.json and summary.csv to results/narrow_inverse_research/error_analysis/.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inverse.estimators import CausalStepMAPEstimator, StepMAPEstimator
from src.inverse.statistical_modeling import (
    DistributionFitResult,
    NullDistributionSummary,
    OperationalThresholdCurve,
    compute_operational_threshold_curves,
    evaluate_null_clean_distribution,
    fit_parametric_distributions,
    rank_models,
)
from src.inverse.surrogates import DilatedTCNSurrogate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sprint 4A: Statistical Distribution and Error Modeling Runner"
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/narrow_inverse_research/sealed_final_benchmark/summary.csv",
        help="Path to sealed benchmark summary.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="results/narrow_inverse_research/sealed_final_benchmark/summary.json",
        help="Path to sealed benchmark summary.json",
    )
    parser.add_argument(
        "--val-parquet",
        type=str,
        default="data/processed/batadal/val.parquet",
        help="Path to BATADAL val.parquet",
    )
    parser.add_argument(
        "--tcn-checkpoint",
        type=str,
        default="results/surrogates/vista_tcn_48h_150ep/tcn.pt",
        help="Path to champion TCN surrogate checkpoint (.pt)",
    )
    parser.add_argument(
        "--channel-manifest",
        type=str,
        default="configs/manifests/batadal-channel-manifest-v1.json",
        help="Path to channel manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/narrow_inverse_research/error_analysis",
        help="Directory to write analysis results and diagnostic plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for PyTorch inference ('cuda' or 'cpu')",
    )
    return parser.parse_args()


# =====================================================================
# Plot 1: Error Distribution Fits
# =====================================================================

def plot_error_distribution_fits(
    errors_by_estimator: dict[str, np.ndarray],
    fits_by_estimator: dict[str, dict[str, DistributionFitResult]],
    output_path: Path,
) -> None:
    """Plot empirical error histogram with overlaid Gaussian, Student-t, and GED curves."""
    key_estimators = [
        "CausalStepMAP_TCN_Undeclared",
        "StepMAP_TCN_Undeclared",
        "CompanionRegressionEstimator",
        "PersistenceEstimator",
    ]
    estimators_to_plot = [e for e in key_estimators if e in errors_by_estimator]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    for idx, est_name in enumerate(estimators_to_plot):
        ax = axes[idx]
        errs = errors_by_estimator[est_name]
        fits = fits_by_estimator[est_name]

        # Histogram
        n_bins = 25
        counts, bin_edges, _ = ax.hist(
            errs,
            bins=n_bins,
            density=True,
            alpha=0.45,
            color="#2b5c8f",
            edgecolor="black",
            linewidth=0.8,
            label="Empirical Error",
        )

        x_grid = np.linspace(min(errs.min(), -0.5), max(errs.max(), 0.5), 500)

        # 1. Gaussian
        g_fit = fits["gaussian"]
        g_pdf = stats.norm.pdf(x_grid, loc=g_fit.params["loc"], scale=g_fit.params["scale"])
        ax.plot(x_grid, g_pdf, label=f"Gaussian (AIC={g_fit.aic:.1f})", color="#d95f02", linestyle="--", linewidth=2.0)

        # 2. Student-t
        t_fit = fits["student_t"]
        t_pdf = stats.t.pdf(x_grid, df=t_fit.params["df"], loc=t_fit.params["loc"], scale=t_fit.params["scale"])
        ax.plot(x_grid, t_pdf, label=f"Student-t (AIC={t_fit.aic:.1f})", color="#7570b3", linestyle="-.", linewidth=2.0)

        # 3. GED
        ged_fit = fits["ged"]
        ged_pdf = stats.gennorm.pdf(x_grid, beta=ged_fit.params["beta"], loc=ged_fit.params["loc"], scale=ged_fit.params["scale"])
        ax.plot(x_grid, ged_pdf, label=f"GED (AIC={ged_fit.aic:.1f}, beta={ged_fit.params['beta']:.2f})", color="#1b9e77", linestyle="-", linewidth=2.5)

        ranked = rank_models(fits, criterion="aic")
        best = ranked[0]
        ax.set_title(
            f"{est_name}\nBest Fit: {best.distribution_name.upper()} (AIC={best.aic:.1f}, KS p={best.ks_pvalue:.3f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Reconstruction Amplitude Error e_A (meters)", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.suptitle("Parametric Distribution Fits to Scenario Reconstruction Error e_A", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


# =====================================================================
# Plot 2: Q-Q Plots
# =====================================================================

def plot_qq_plots(
    errors_by_estimator: dict[str, np.ndarray],
    fits_by_estimator: dict[str, dict[str, DistributionFitResult]],
    output_path: Path,
) -> None:
    """Plot Quantile-Quantile (Q-Q) plots comparing empirical quantiles vs theoretical quantiles."""
    estimators_to_plot = ["CausalStepMAP_TCN_Undeclared", "StepMAP_TCN_Undeclared"]
    quantiles = np.linspace(0.01, 0.99, 100)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    for row_idx, est_name in enumerate(estimators_to_plot):
        if est_name not in errors_by_estimator:
            continue
        errs = np.sort(errors_by_estimator[est_name])
        fits = fits_by_estimator[est_name]
        emp_q = np.quantile(errs, quantiles)

        # 1. Gaussian Q-Q
        ax_g = axes[row_idx, 0]
        g_fit = fits["gaussian"]
        g_theo = stats.norm.ppf(quantiles, loc=g_fit.params["loc"], scale=g_fit.params["scale"])
        ax_g.scatter(g_theo, emp_q, color="#d95f02", s=20, alpha=0.7)
        lims_g = [min(g_theo.min(), emp_q.min()), max(g_theo.max(), emp_q.max())]
        ax_g.plot(lims_g, lims_g, "k--", linewidth=1.5, alpha=0.75, label="y = x")
        ax_g.set_title(f"{est_name}\nGaussian Q-Q (KS={g_fit.ks_stat:.3f})", fontsize=10, fontweight="bold")
        ax_g.set_xlabel("Theoretical Quantiles (m)", fontsize=9)
        ax_g.set_ylabel("Empirical Quantiles (m)", fontsize=9)
        ax_g.grid(True, linestyle=":", alpha=0.6)
        ax_g.legend(loc="upper left", fontsize=8)

        # 2. Student-t Q-Q
        ax_t = axes[row_idx, 1]
        t_fit = fits["student_t"]
        t_theo = stats.t.ppf(quantiles, df=t_fit.params["df"], loc=t_fit.params["loc"], scale=t_fit.params["scale"])
        ax_t.scatter(t_theo, emp_q, color="#7570b3", s=20, alpha=0.7)
        lims_t = [min(t_theo.min(), emp_q.min()), max(t_theo.max(), emp_q.max())]
        ax_t.plot(lims_t, lims_t, "k--", linewidth=1.5, alpha=0.75, label="y = x")
        ax_t.set_title(f"{est_name}\nStudent-t Q-Q (df={t_fit.params['df']:.1f}, KS={t_fit.ks_stat:.3f})", fontsize=10, fontweight="bold")
        ax_t.set_xlabel("Theoretical Quantiles (m)", fontsize=9)
        ax_t.grid(True, linestyle=":", alpha=0.6)
        ax_t.legend(loc="upper left", fontsize=8)

        # 3. GED Q-Q
        ax_ged = axes[row_idx, 2]
        ged_fit = fits["ged"]
        ged_theo = stats.gennorm.ppf(quantiles, beta=ged_fit.params["beta"], loc=ged_fit.params["loc"], scale=ged_fit.params["scale"])
        ax_ged.scatter(ged_theo, emp_q, color="#1b9e77", s=20, alpha=0.7)
        lims_ged = [min(ged_theo.min(), emp_q.min()), max(ged_theo.max(), emp_q.max())]
        ax_ged.plot(lims_ged, lims_ged, "k--", linewidth=1.5, alpha=0.75, label="y = x")
        ax_ged.set_title(f"{est_name}\nGED Q-Q (beta={ged_fit.params['beta']:.2f}, KS={ged_fit.ks_stat:.3f})", fontsize=10, fontweight="bold")
        ax_ged.set_xlabel("Theoretical Quantiles (m)", fontsize=9)
        ax_ged.grid(True, linestyle=":", alpha=0.6)
        ax_ged.legend(loc="upper left", fontsize=8)

    plt.suptitle("Q-Q Goodness-of-Fit Comparison across Candidate Distributions", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


# =====================================================================
# Plot 3: Null Clean Distribution
# =====================================================================

def plot_null_clean_distribution(
    null_series_map: dict[str, np.ndarray],
    null_summaries: dict[str, NullDistributionSummary],
    output_path: Path,
) -> None:
    """Plot time-series reconstruction and empirical survival curves under authentic telemetry."""
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax_ts = fig.add_subplot(gs[0, :])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_surv = fig.add_subplot(gs[1, 1])

    colors = {
        "StepMAP_TCN_Undeclared": "#2b5c8f",
        "CausalStepMAP_TCN_Undeclared": "#d95f02",
    }

    # Top: Time Series
    hours = np.arange(len(next(iter(null_series_map.values()))))
    for est_name, delta_null in null_series_map.items():
        color = colors.get(est_name, "#333333")
        label = f"{est_name} (Sparsity: {null_summaries[est_name].sparsity_rate * 100:.1f}%)"
        ax_ts.plot(hours, delta_null, label=label, color=color, linewidth=1.4, alpha=0.85)

    ax_ts.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_ts.axhline(0.1254, color="crimson", linestyle=":", linewidth=1.2, label="Nominal 3*sigma_eps (0.125m)")
    ax_ts.axhline(-0.1254, color="crimson", linestyle=":", linewidth=1.2)
    ax_ts.set_title("Authentic Telemetry Reconstruction p(delta_hat | delta = 0) (Test Block Rows 960-1440)", fontsize=11, fontweight="bold")
    ax_ts.set_xlabel("Time Elapsed in Test Block (hours)", fontsize=10)
    ax_ts.set_ylabel("Reconstructed delta_hat (meters)", fontsize=10)
    ax_ts.set_ylim(-0.15, 0.15)
    ax_ts.grid(True, linestyle=":", alpha=0.6)
    ax_ts.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Bottom Left: Magnitude Histogram
    for est_name, delta_null in null_series_map.items():
        color = colors.get(est_name, "#333333")
        abs_err = np.abs(delta_null)
        ax_hist.hist(
            abs_err,
            bins=40,
            density=True,
            alpha=0.5,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label=f"{est_name} (|delta| > 0: {(abs_err > 1e-4).sum()}/480)",
        )
    ax_hist.set_title("Magnitude Distribution |delta_hat| on Clean Series", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Magnitude |delta_hat| (meters)", fontsize=10)
    ax_hist.set_ylabel("Empirical Density", fontsize=10)
    ax_hist.set_yscale("log")
    ax_hist.grid(True, linestyle=":", alpha=0.6)
    ax_hist.legend(loc="upper right", fontsize=8.5)

    # Bottom Right: Empirical False Alarm Rate (Survival Function)
    taus = np.linspace(0.0, 0.15, 300)
    for est_name, delta_null in null_series_map.items():
        color = colors.get(est_name, "#333333")
        abs_err = np.abs(delta_null)
        fars = [float(np.mean(abs_err >= t)) for t in taus]
        ax_surv.plot(taus, fars, label=f"{est_name}", color=color, linewidth=2.0)

    ax_surv.axvline(0.1254, color="crimson", linestyle=":", linewidth=1.2, label="Nominal 3*sigma_eps (0.125m)")
    ax_surv.axhline(0.01, color="grey", linestyle="--", linewidth=0.9, label="Target FAR = 1.0%")
    ax_surv.axhline(0.001, color="black", linestyle="--", linewidth=0.9, label="Target FAR = 0.1%")
    ax_surv.set_title("Empirical False Alarm Rate P(|delta_hat| >= tau | H0)", fontsize=11, fontweight="bold")
    ax_surv.set_xlabel("Alarm Threshold tau (meters)", fontsize=10)
    ax_surv.set_ylabel("False Alarm Rate (alpha)", fontsize=10)
    ax_surv.set_yscale("log")
    ax_surv.set_ylim(1e-4, 1.0)
    ax_surv.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_surv.legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Empirical Null Noise Envelope and False-Alarm Profiling", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


# =====================================================================
# Plot 4: ROC Curves by Amplitude
# =====================================================================

def plot_roc_curves_by_amplitude(
    null_deltas: np.ndarray,
    df_benchmark: pd.DataFrame,
    train_std: float,
    output_path: Path,
) -> None:
    """Plot operational ROC curves stratified by injection amplitude (0.75, 1.5, 3.0 sigma)."""
    est_name = "CausalStepMAP_TCN_Undeclared"
    est_df = df_benchmark[df_benchmark["estimator"] == est_name]

    fig, (ax_linear, ax_log) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    colors = {0.75: "#1b9e77", 1.5: "#d95f02", 3.0: "#7570b3", "overall": "#2b5c8f"}
    labels = {0.75: "0.75 sigma (~0.91m)", 1.5: "1.50 sigma (~1.81m)", 3.0: "3.00 sigma (~3.63m)"}

    # Amplitude-stratified ROCs
    for amp in [0.75, 1.5, 3.0]:
        sub = est_df[est_df["amplitude_train_std"] == amp]
        a_hat = np.abs(sub["injected_amplitude"] + sub["signed_amplitude_error_train_std"] * train_std)
        curve = compute_operational_threshold_curves(
            null_deltas=null_deltas,
            attack_deltas=a_hat,
            target_fars=(0.001, 0.01, 0.05),
            n_thresholds=300,
        )

        ax_linear.plot(curve.fpr, curve.tpr, color=colors[amp], linewidth=2.0, label=f"Amp {labels[amp]} (AUC={curve.auc:.4f})")
        ax_log.plot(curve.fpr, curve.tpr, color=colors[amp], linewidth=2.0, label=f"Amp {labels[amp]}")

    # Overall ROC
    all_ahat = np.abs(est_df["injected_amplitude"] + est_df["signed_amplitude_error_train_std"] * train_std)
    all_curve = compute_operational_threshold_curves(
        null_deltas=null_deltas,
        attack_deltas=all_ahat,
        target_fars=(0.001, 0.01, 0.05),
        n_thresholds=300,
    )
    ax_linear.plot(all_curve.fpr, all_curve.tpr, color=colors["overall"], linestyle="--", linewidth=2.2, label=f"Overall (AUC={all_curve.auc:.4f})")
    ax_log.plot(all_curve.fpr, all_curve.tpr, color=colors["overall"], linestyle="--", linewidth=2.2, label="Overall")

    # Annotate target FAR lines
    for far, ls in [(0.05, ":"), (0.01, "--"), (0.001, "-.")]:
        ax_linear.axvline(far, color="grey", linestyle=ls, alpha=0.7, label=f"FAR = {far*100:.1f}%" if far == 0.01 else "")
        ax_log.axvline(far, color="grey", linestyle=ls, alpha=0.7, label=f"FAR = {far*100:.1f}%")

    ax_linear.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Guess")
    ax_linear.set_title(f"Operational ROC Detection Curves ({est_name})", fontsize=11, fontweight="bold")
    ax_linear.set_xlabel("False Positive Rate (FPR / FAR)", fontsize=10)
    ax_linear.set_ylabel("True Positive Rate (Recall / TPR)", fontsize=10)
    ax_linear.set_xlim(-0.01, 1.01)
    ax_linear.set_ylim(-0.01, 1.02)
    ax_linear.grid(True, linestyle=":", alpha=0.6)
    ax_linear.legend(loc="lower right", fontsize=8.5)

    ax_log.set_title("Operational ROC on Log-FPR Scale (ICS Alerting Domain)", fontsize=11, fontweight="bold")
    ax_log.set_xlabel("False Positive Rate (log scale)", fontsize=10)
    ax_log.set_ylabel("True Positive Rate (Recall / TPR)", fontsize=10)
    ax_log.set_xscale("log")
    ax_log.set_xlim(1e-4, 1.0)
    ax_log.set_ylim(-0.01, 1.02)
    ax_log.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_log.legend(loc="lower right", fontsize=8.5)

    plt.suptitle("Operational Detection Tradeoff Bounds for Causal Step-MAP Champion", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


# =====================================================================
# Plot 5: Error vs Amplitude and Regime
# =====================================================================

def plot_error_vs_amplitude_regime(
    df_benchmark: pd.DataFrame,
    train_std: float,
    output_path: Path,
) -> None:
    """Plot reconstruction error across amplitudes, hydraulic regimes, and attack sign."""
    key_ests = [
        "CausalStepMAP_TCN_Undeclared",
        "StepMAP_TCN_Undeclared",
        "CompanionRegressionEstimator",
        "PersistenceEstimator",
    ]
    df_plot = df_benchmark[df_benchmark["estimator"].isin(key_ests)].copy()
    df_plot["error_meters"] = df_plot["signed_amplitude_error_train_std"] * train_std

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # 1. Error vs Amplitude
    amps = [0.75, 1.5, 3.0]
    for est in key_ests:
        means = [df_plot[(df_plot["estimator"] == est) & (df_plot["amplitude_train_std"] == a)]["error_meters"].mean() for a in amps]
        stds = [df_plot[(df_plot["estimator"] == est) & (df_plot["amplitude_train_std"] == a)]["error_meters"].std() for a in amps]
        ax1.errorbar(amps, means, yerr=stds, marker="o", capsize=4, linewidth=1.8, label=est)
    ax1.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_title("Reconstruction Error e_A vs Injected Amplitude", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Injected Attack Amplitude (sigma_train)", fontsize=10)
    ax1.set_ylabel("Signed Error e_A (meters)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Error vs Dynamics Regime
    regimes = ["low_dynamics", "medium_dynamics", "high_dynamics"]
    for est in key_ests:
        means = [df_plot[(df_plot["estimator"] == est) & (df_plot["regime"] == r)]["error_meters"].mean() for r in regimes]
        stds = [df_plot[(df_plot["estimator"] == est) & (df_plot["regime"] == r)]["error_meters"].std() for r in regimes]
        ax2.errorbar(range(len(regimes)), means, yerr=stds, marker="s", capsize=4, linewidth=1.8, label=est)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_xticks(range(len(regimes)))
    ax2.set_xticklabels(["Low", "Medium", "High"])
    ax2.set_title("Reconstruction Error e_A vs Hydraulic Regime", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Hydraulic Dynamics Regime", fontsize=10)
    ax2.set_ylabel("Signed Error e_A (meters)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    # 3. Error vs Attack Sign (Symmetry Audit)
    signs = [-1, 1]
    for est in key_ests:
        means = [df_plot[(df_plot["estimator"] == est) & (df_plot["sign"] == s)]["error_meters"].mean() for s in signs]
        stds = [df_plot[(df_plot["estimator"] == est) & (df_plot["sign"] == s)]["error_meters"].std() for s in signs]
        ax3.errorbar(["Negative (-)", "Positive (+)"], means, yerr=stds, marker="^", capsize=4, linewidth=1.8, label=est)
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.set_title("Reconstruction Error e_A vs Attack Sign (Symmetry)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Attack Injection Sign", fontsize=10)
    ax3.set_ylabel("Signed Error e_A (meters)", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8)

    plt.suptitle("Reconstruction Error Stability across Amplitude, Dynamics, and Sign Regimes", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


# =====================================================================
# Main Analysis Pipeline
# =====================================================================

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = Path(args.summary_csv)
    summary_json_path = Path(args.summary_json)
    val_parquet_path = Path(args.val_parquet)
    tcn_ckpt_path = Path(args.tcn_checkpoint)
    manifest_path = Path(args.channel_manifest)

    if not summary_csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV at {summary_csv_path}")
    if not summary_json_path.exists():
        raise FileNotFoundError(f"Missing summary JSON at {summary_json_path}")
    if not val_parquet_path.exists():
        raise FileNotFoundError(f"Missing validation parquet at {val_parquet_path}")
    if not tcn_ckpt_path.exists():
        raise FileNotFoundError(f"Missing TCN checkpoint at {tcn_ckpt_path}")

    print(f"[Sprint 4A] Loading benchmark results from {summary_csv_path}...")
    df_benchmark = pd.read_csv(summary_csv_path)
    with open(summary_json_path, "r", encoding="utf-8") as f:
        bench_json = json.load(f)

    train_std = float(bench_json.get("train_std_meters", 1.20852))
    target_channel = bench_json.get("target_channel", "L_T1")
    print(f"[Sprint 4A] Benchmark target sensor: {target_channel}, train_std: {train_std:.5f} m")

    # 1. Fit Parametric Distributions across All Estimators and Amplitudes
    estimators = df_benchmark["estimator"].unique().tolist()
    errors_by_estimator: dict[str, np.ndarray] = {}
    fits_by_estimator: dict[str, dict[str, DistributionFitResult]] = {}
    stratified_fits: dict[str, dict[str, dict[str, Any]]] = {}

    for est in estimators:
        sub = df_benchmark[df_benchmark["estimator"] == est].copy()
        errs = (sub["signed_amplitude_error_train_std"] * train_std).to_numpy(dtype=np.float64)
        errors_by_estimator[est] = errs

        overall_fits = fit_parametric_distributions(errs)
        fits_by_estimator[est] = overall_fits

        stratified_fits[est] = {
            "overall": {
                "fits": {k: v.to_dict() for k, v in overall_fits.items()},
                "best_model_aic": rank_models(overall_fits, "aic")[0].distribution_name,
                "best_model_bic": rank_models(overall_fits, "bic")[0].distribution_name,
            },
            "by_amplitude": {},
        }

        for amp in (0.75, 1.5, 3.0):
            amp_sub = sub[sub["amplitude_train_std"] == amp]
            amp_errs = (amp_sub["signed_amplitude_error_train_std"] * train_std).to_numpy(dtype=np.float64)
            amp_fits = fit_parametric_distributions(amp_errs)
            stratified_fits[est]["by_amplitude"][str(amp)] = {
                "fits": {k: v.to_dict() for k, v in amp_fits.items()},
                "best_model_aic": rank_models(amp_fits, "aic")[0].distribution_name,
                "best_model_bic": rank_models(amp_fits, "bic")[0].distribution_name,
            }

    print("[Sprint 4A] Parametric fitting completed across all estimators and amplitude tiers.")

    # 2. Evaluate Clean Null Distribution p(delta_hat | delta = 0)
    print(f"[Sprint 4A] Evaluating clean null distribution on rows 960-1440 of {val_parquet_path}...")
    val_df = pd.read_parquet(val_parquet_path)
    clean_test_block = val_df.iloc[960:1440].copy()

    tcn_surrogate = DilatedTCNSurrogate.load(tcn_ckpt_path)
    tcn_surrogate.device_str = args.device
    if hasattr(tcn_surrogate, "model") and tcn_surrogate.model is not None:
        tcn_surrogate.model.to(torch.device(args.device))

    step_map = StepMAPEstimator(
        target_channel=target_channel,
        channel_manifest_path=manifest_path,
        surrogate_estimator=tcn_surrogate,
    )
    causal_step_map = CausalStepMAPEstimator(
        target_channel=target_channel,
        channel_manifest_path=manifest_path,
        surrogate_estimator=tcn_surrogate,
    )

    res_step = step_map.reconstruct(clean_test_block, attack_interval=None)
    res_causal = causal_step_map.reconstruct(clean_test_block, attack_interval=None)

    null_series_map = {
        "StepMAP_TCN_Undeclared": res_step.delta_hat,
        "CausalStepMAP_TCN_Undeclared": res_causal.delta_hat,
    }

    null_summaries: dict[str, NullDistributionSummary] = {}
    null_summaries_dict: dict[str, Any] = {}
    threshold_eval_levels = [0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.1254, 0.15, 0.20, 0.25]

    for name, series in null_series_map.items():
        summary = evaluate_null_clean_distribution(
            series,
            estimator_name=name,
            tolerance=1e-4,
            threshold_levels=threshold_eval_levels,
        )
        null_summaries[name] = summary
        null_summaries_dict[name] = summary.to_dict()
        print(f"  {name}: Sparsity={summary.sparsity_rate*100:.2f}%, std={summary.std_null:.6f}m, max_abs={summary.max_abs:.6f}m")

    # 3. Operational Threshold Trade-offs & Curves
    print("[Sprint 4A] Computing operational detection curves and threshold tables...")
    operational_results: dict[str, Any] = {}
    target_fars = (0.001, 0.01, 0.05)

    for est_name in ["CausalStepMAP_TCN_Undeclared", "StepMAP_TCN_Undeclared"]:
        null_deltas = null_series_map[est_name]
        est_df = df_benchmark[df_benchmark["estimator"] == est_name]
        a_hat_all = np.abs(est_df["injected_amplitude"] + est_df["signed_amplitude_error_train_std"] * train_std)

        curve_overall = compute_operational_threshold_curves(
            null_deltas=null_deltas,
            attack_deltas=a_hat_all,
            target_fars=target_fars,
            n_thresholds=300,
        )

        op_entry: dict[str, Any] = {
            "overall": {
                "auc": curve_overall.auc,
                "tau_for_far": curve_overall.tau_for_far,
            },
            "by_amplitude": {},
        }

        for amp in (0.75, 1.5, 3.0):
            sub_amp = est_df[est_df["amplitude_train_std"] == amp]
            a_hat_amp = np.abs(sub_amp["injected_amplitude"] + sub_amp["signed_amplitude_error_train_std"] * train_std)
            curve_amp = compute_operational_threshold_curves(
                null_deltas=null_deltas,
                attack_deltas=a_hat_amp,
                target_fars=target_fars,
                n_thresholds=300,
            )
            op_entry["by_amplitude"][str(amp)] = {
                "auc": curve_amp.auc,
                "tau_for_far": curve_amp.tau_for_far,
                "tpr_at_nominal_3sigma": float(np.mean(a_hat_amp >= 0.1254)),
            }

        operational_results[est_name] = op_entry

    # 4. Generate Publication-Ready Plots
    print(f"[Sprint 4A] Rendering 5 diagnostic plots in {output_dir}...")
    plot_error_distribution_fits(
        errors_by_estimator=errors_by_estimator,
        fits_by_estimator=fits_by_estimator,
        output_path=output_dir / "error_distribution_fits.png",
    )
    plot_qq_plots(
        errors_by_estimator=errors_by_estimator,
        fits_by_estimator=fits_by_estimator,
        output_path=output_dir / "qq_plots.png",
    )
    plot_null_clean_distribution(
        null_series_map=null_series_map,
        null_summaries=null_summaries,
        output_path=output_dir / "null_distribution_clean.png",
    )
    plot_roc_curves_by_amplitude(
        null_deltas=null_series_map["CausalStepMAP_TCN_Undeclared"],
        df_benchmark=df_benchmark,
        train_std=train_std,
        output_path=output_dir / "roc_curves_by_amplitude.png",
    )
    plot_error_vs_amplitude_regime(
        df_benchmark=df_benchmark,
        train_std=train_std,
        output_path=output_dir / "error_vs_amplitude_regime.png",
    )

    # 5. Save Output Artifacts: summary.json and summary.csv
    print(f"[Sprint 4A] Saving summary.json and summary.csv...")
    summary_json_out = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "train_std_meters": train_std,
        "target_channel": target_channel,
        "parametric_fits": stratified_fits,
        "null_distribution": null_summaries_dict,
        "operational_detection": operational_results,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json_out, f, indent=2)

    # Enrich summary.csv with error_meters, reconstructed_amplitude, and log-likelihood densities
    df_out = df_benchmark.copy()
    df_out["error_meters"] = df_out["signed_amplitude_error_train_std"] * train_std
    df_out["reconstructed_amplitude"] = df_out["injected_amplitude"] + df_out["error_meters"]

    # Compute logpdfs under fitted models for each record
    gauss_lps = []
    t_lps = []
    ged_lps = []
    best_models = []

    for _, row in df_out.iterrows():
        est = row["estimator"]
        err = row["error_meters"]
        fits = fits_by_estimator[est]

        g = fits["gaussian"]
        t = fits["student_t"]
        ged = fits["ged"]

        lp_g = float(stats.norm.logpdf(err, loc=g.params["loc"], scale=g.params["scale"]))
        lp_t = float(stats.t.logpdf(err, df=t.params["df"], loc=t.params["loc"], scale=t.params["scale"]))
        lp_ged = float(stats.gennorm.logpdf(err, beta=ged.params["beta"], loc=ged.params["loc"], scale=ged.params["scale"]))

        gauss_lps.append(lp_g)
        t_lps.append(lp_t)
        ged_lps.append(lp_ged)
        best_models.append(rank_models(fits, "aic")[0].distribution_name)

    df_out["logpdf_gaussian"] = gauss_lps
    df_out["logpdf_student_t"] = t_lps
    df_out["logpdf_ged"] = ged_lps
    df_out["best_fit_model"] = best_models

    df_out.to_csv(output_dir / "summary.csv", index=False)
    print(f"[Sprint 4A] Execution complete. Artifacts saved in {output_dir}.")


if __name__ == "__main__":
    main()
