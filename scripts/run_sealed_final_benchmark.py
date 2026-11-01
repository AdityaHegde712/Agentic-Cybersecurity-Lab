"""Sealed Final Benchmark Evaluation Runner (November 18 Milestone Step 5).

Unseals and evaluates the final test block (rows 960-1440 of val.parquet)
across the 180 sealed test scenarios defined in batadal-benchmark-manifest-v1.json:
- Unseen amplitudes: {0.75, 1.5, 3.0} sigma
- Unseen durations: {9, 18, 36} hours
- 10 regime-stratified onsets per combination (180 scenarios total)

Evaluates 6 estimator configurations:
1. PersistenceEstimator (Causal Baseline)
2. CompanionRegressionEstimator (Linear Baseline)
3. StepMAP_TCN_Declared (Offline Champion)
4. StepMAP_TCN_Undeclared (Offline Free Inverse Recovery)
5. CausalStepMAP_TCN_Declared (Online Rolling Filter)
6. CausalStepMAP_TCN_Undeclared (Online Free Inverse Recovery)

Saves:
- results/narrow_inverse_research/sealed_final_benchmark/summary.json
- results/narrow_inverse_research/sealed_final_benchmark/summary.csv
- results/narrow_inverse_research/sealed_final_benchmark/plots/final_nrmse_by_duration.png
- results/narrow_inverse_research/sealed_final_benchmark/plots/final_nrmse_by_amplitude.png
- results/narrow_inverse_research/sealed_final_benchmark/plots/final_coverage_sharpness.png
- results/narrow_inverse_research/sealed_final_benchmark/plots/final_causal_vs_persistence.png
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
    CausalStepMAPEstimator,
    CompanionRegressionEstimator,
    PersistenceEstimator,
    SeasonalNaiveEstimator,
    StepMAPEstimator,
)
from src.inverse.harness import (
    BenchmarkScenario,
    compute_regime_cutoffs,
    compute_scenario_metrics,
    generate_final_grid_scenarios,
)
from src.inverse.surrogates import DilatedTCNSurrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="November 18 Milestone Sealed Final Benchmark Evaluation"
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
        default="results/narrow_inverse_research/sealed_final_benchmark",
        help="Output directory for final benchmark artifacts",
    )
    parser.add_argument(
        "--lambda-tv",
        type=float,
        default=1.0,
        help="Total variation penalty weight",
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=0.05,
        help="Sparsity penalty weight",
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


def run_sealed_final_benchmark() -> dict[str, Any]:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    print("=" * 80)
    print("SEALED FINAL BENCHMARK EVALUATION (NOVEMBER 18 MILESTONE STEP 5)")
    print(f"Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"Device: {device} | Output Directory: {output_dir}")
    print("=" * 80)

    # 1. Load and Verify Manifests
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = BenchmarkManifest.from_json(manifest_path.read_text(encoding="utf-8").strip())
    target_channel = manifest.target_sensor
    print(f"Dataset: {manifest.dataset} | Target Channel: {target_channel}")

    channel_manifest_path = Path(args.channel_manifest_path)
    if not channel_manifest_path.exists():
        raise FileNotFoundError(f"Channel manifest not found: {channel_manifest_path}")
    channel_manifest = ChannelConstraintManifest.from_json(
        channel_manifest_path.read_text(encoding="utf-8").strip()
    )
    target_ch_meta = next(c for c in channel_manifest.channels if c.name == target_channel)
    box_bounds = (target_ch_meta.empirical_bounds.minimum, target_ch_meta.empirical_bounds.maximum)
    print(f"Target Channel Physical Box Bounds: [{box_bounds[0]:.4f}, {box_bounds[1]:.4f}] m")

    # 2. Data Partitioning & Test Split Unsealing
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing train or val parquet datasets")

    train_df = pd.read_parquet(train_path)
    val_df_full = pd.read_parquet(val_path)
    assert len(val_df_full) == 1440, f"Expected 1440 val rows, got {len(val_df_full)}"

    fit_block = next(b for b in manifest.blocks if b.name == "model_fit")
    calib_block = next(b for b in manifest.blocks if b.name == "clean_calibration")
    dev_block = next(b for b in manifest.blocks if b.name == "development_injection")
    sealed_block = next(b for b in manifest.blocks if b.name == "sealed_final_injection")

    assert sealed_block.row_start == 960 and sealed_block.row_stop == 1440

    fit_df = train_df.iloc[fit_block.row_start : fit_block.row_stop].copy()
    calib_df = val_df_full.iloc[calib_block.row_start : calib_block.row_stop].copy()
    final_df = val_df_full.iloc[sealed_block.row_start : sealed_block.row_stop].copy()

    # Antecedent clean values prior to row 960
    prev_val = float(val_df_full[target_channel].iloc[sealed_block.row_start - 1])
    prior_clean_history = val_df_full[target_channel].iloc[: sealed_block.row_start].to_numpy(dtype=np.float64)

    train_std = float(fit_df[target_channel].std())
    train_mean = float(fit_df[target_channel].mean())
    print(f"Target Sensor Normal Dynamics: mean={train_mean:.4f} m, std={train_std:.4f} m")

    # 3. Stratified Final Scenario Generation (180 Scenarios)
    cutoffs = compute_regime_cutoffs(fit_df[target_channel].to_numpy())
    scenarios = generate_final_grid_scenarios(
        manifest=manifest,
        final_series=final_df[target_channel].to_numpy(),
        prev_val=prev_val,
        train_std=train_std,
        cutoffs=cutoffs,
    )
    print(f"Stratified Final Scenarios Generated: {len(scenarios)} scenarios (18 combinations x 10 onsets)")

    # 4. Load Models & Checkpoints
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Surrogate checkpoint not found: {ckpt_path}")
    print(f"Loading Champion Dilated TCN surrogate from {ckpt_path}...")
    tcn_surrogate = DilatedTCNSurrogate.load(ckpt_path)
    tcn_surrogate.device_str = device
    if hasattr(tcn_surrogate, "model") and tcn_surrogate.model is not None:
        tcn_surrogate.model.to(torch.device(device))

    calib_series = calib_df[target_channel].to_numpy(dtype=np.float64)

    # 5. Fit Baselines and Step-MAP Estimators
    print("Initializing estimators...")
    p_est = PersistenceEstimator(lag=1).fit_normal(calib_series)
    reg_est = CompanionRegressionEstimator(
        target_channel=target_channel,
        channel_manifest_path=channel_manifest_path,
        alpha=1.0,
    ).fit_normal(fit_df)

    step_map_tcn = StepMAPEstimator(
        lambda_tv=args.lambda_tv,
        lambda_l1=args.lambda_l1,
        target_channel=target_channel,
        channel_manifest_path=channel_manifest_path,
        surrogate_estimator=tcn_surrogate,
    )

    causal_step_map_tcn = CausalStepMAPEstimator(
        window_size=48,
        lambda_tv=args.lambda_tv,
        lambda_l1=args.lambda_l1,
        target_channel=target_channel,
        channel_manifest_path=channel_manifest_path,
        surrogate_estimator=tcn_surrogate,
    )

    configs = [
        ("PersistenceEstimator", p_est, True),
        ("CompanionRegressionEstimator", reg_est, True),
        ("StepMAP_TCN_Declared", step_map_tcn, True),
        ("StepMAP_TCN_Undeclared", step_map_tcn, False),
        ("CausalStepMAP_TCN_Declared", causal_step_map_tcn, True),
        ("CausalStepMAP_TCN_Undeclared", causal_step_map_tcn, False),
    ]

    # 6. Execute Sealed Final Benchmark Across All 180 Scenarios
    print(f"\nExecuting {len(configs)} configurations across 180 final scenarios ({len(configs) * len(scenarios)} evaluations)...")
    clean_x_final = final_df[target_channel].to_numpy(dtype=np.float64)
    n_final = len(clean_x_final)
    records: list[dict[str, Any]] = []

    t_start = time.perf_counter()
    b_min, b_max = box_bounds

    for s_idx, sc in enumerate(scenarios):
        delta_true = np.zeros(n_final, dtype=np.float64)
        delta_true[sc.onset_index : sc.offset_index] = sc.injected_amplitude

        attacked_df = final_df.copy()
        attacked_df[target_channel] = clean_x_final + delta_true

        for config_name, est, is_declared in configs:
            attack_interval = (sc.onset_index, sc.offset_index) if is_declared else None

            if isinstance(est, PersistenceEstimator):
                res = est.reconstruct(
                    attacked_df,
                    attack_interval=attack_interval,
                    target_channel=target_channel,
                    prior_clean_history=prior_clean_history,
                )
            else:
                res = est.reconstruct(
                    attacked_df,
                    attack_interval=attack_interval,
                )

            # Count hard box-constraint violations (tolerance 1e-4)
            hard_violations = int(np.sum((res.x_hat < b_min - 1e-4) | (res.x_hat > b_max + 1e-4)))

            m = compute_scenario_metrics(
                delta_hat=res.delta_hat,
                delta_true=delta_true,
                x_hat=res.x_hat,
                x_true=clean_x_final,
                onset=sc.onset_index,
                offset=sc.offset_index,
                injected_amplitude=sc.injected_amplitude,
                train_std=train_std,
                pointwise_uncertainty=res.pointwise_uncertainty,
                scenario=sc,
                estimator_name=config_name,
            )
            m["hard_constraint_violations"] = hard_violations
            m["block_name"] = sc.block_name
            records.append(m)

        if (s_idx + 1) % 30 == 0:
            print(f"Processed {s_idx + 1}/{len(scenarios)} scenarios...")

    total_eval_time = time.perf_counter() - t_start
    print(f"Final evaluations completed in {total_eval_time:.2f} s ({total_eval_time / len(records) * 1000:.2f} ms/eval).")

    results_df = pd.DataFrame(records)

    # 7. Aggregate Statistics and Acceptance Gates
    summary_dict = _build_final_summary(
        results_df=results_df,
        target_channel=target_channel,
        train_std=train_std,
        n_scenarios=len(scenarios),
        ckpt_path=str(ckpt_path),
        total_time=total_eval_time,
    )

    # 8. Save CSV and JSON
    csv_path = output_dir / "summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved scenario evaluation records to {csv_path}")

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary_dict, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved sealed final benchmark summary to {json_path}")

    # 9. Generate Diagnostic Visualizations
    print("Generating diagnostic benchmark plots...")
    _generate_final_plots(
        results_df=results_df,
        summary_dict=summary_dict,
        plots_dir=plots_dir,
    )
    print(f"Diagnostic plots saved to {plots_dir}")

    print("\n" + "=" * 80)
    print(f"ALL PRECOMMITTED GATES PASSED: {summary_dict['acceptance_ledger']['all_gates_passed']}")
    print("SEALED FINAL BENCHMARK EVALUATION COMPLETE.")
    print("=" * 80)
    return summary_dict


def _build_final_summary(
    results_df: pd.DataFrame,
    target_channel: str,
    train_std: float,
    n_scenarios: int,
    ckpt_path: str,
    total_time: float,
) -> dict[str, Any]:
    estimators_summary: dict[str, Any] = {}

    for est_name, group in results_df.groupby("estimator"):
        sign_subset = group[group["amplitude_train_std"] >= 1.0]
        sign_acc = float(sign_subset["sign_accuracy"].mean()) if len(sign_subset) > 0 else 1.0

        est_stats: dict[str, Any] = {
            "median_delta_nrmse": round(float(group["delta_nrmse"].median()), 5),
            "mean_delta_nrmse": round(float(group["delta_nrmse"].mean()), 5),
            "median_support_f1": round(float(group["support_f1"].median()), 5),
            "mean_support_f1": round(float(group["support_f1"].mean()), 5),
            "median_signed_amplitude_error_train_std": round(
                float(group["signed_amplitude_error_train_std"].abs().median()), 5
            ),
            "mean_clean_x_nrmse": round(float(group["clean_x_nrmse"].mean()), 5),
            "sign_accuracy_ge_1std": round(sign_acc, 5),
            "mean_coverage_90": round(float(group["coverage_90"].mean()), 5) if "coverage_90" in group else 0.0,
            "mean_interval_sharpness": round(float(group["interval_sharpness"].mean()), 5) if "interval_sharpness" in group else 0.0,
            "hard_constraint_violations": int(group["hard_constraint_violations"].sum()),
            "by_amplitude": {},
            "by_duration": {},
            "by_regime": {},
        }

        for amp, amp_grp in group.groupby("amplitude_train_std"):
            est_stats["by_amplitude"][str(amp)] = {
                "median_delta_nrmse": round(float(amp_grp["delta_nrmse"].median()), 5),
                "median_support_f1": round(float(amp_grp["support_f1"].median()), 5),
                "median_signed_amplitude_error": round(
                    float(amp_grp["signed_amplitude_error_train_std"].abs().median()), 5
                ),
            }

        for dur, dur_grp in group.groupby("duration_hours"):
            est_stats["by_duration"][str(dur)] = {
                "median_delta_nrmse": round(float(dur_grp["delta_nrmse"].median()), 5),
                "median_support_f1": round(float(dur_grp["support_f1"].median()), 5),
            }

        for reg, reg_grp in group.groupby("regime"):
            est_stats["by_regime"][str(reg)] = {
                "median_delta_nrmse": round(float(reg_grp["delta_nrmse"].median()), 5),
                "median_support_f1": round(float(reg_grp["support_f1"].median()), 5),
            }

        estimators_summary[str(est_name)] = est_stats

    # Precommitted Acceptance Gate Auditing
    persistence_nrmse = estimators_summary["PersistenceEstimator"]["median_delta_nrmse"]
    causal_decl_nrmse = estimators_summary["CausalStepMAP_TCN_Declared"]["median_delta_nrmse"]
    causal_undecl_nrmse = estimators_summary["CausalStepMAP_TCN_Undeclared"]["median_delta_nrmse"]

    causal_decl_impr = (persistence_nrmse - causal_decl_nrmse) / persistence_nrmse
    causal_undecl_impr = (persistence_nrmse - causal_undecl_nrmse) / persistence_nrmse

    gate1_passed = bool(causal_decl_impr >= 0.20 and causal_undecl_impr >= 0.20)

    step_map_configs = [
        "StepMAP_TCN_Declared",
        "StepMAP_TCN_Undeclared",
        "CausalStepMAP_TCN_Declared",
        "CausalStepMAP_TCN_Undeclared",
    ]

    amp_errors = {cfg: estimators_summary[cfg]["median_signed_amplitude_error_train_std"] for cfg in step_map_configs}
    gate2_passed = bool(all(err <= 0.25 for err in amp_errors.values()))

    sign_accs = {cfg: estimators_summary[cfg]["sign_accuracy_ge_1std"] for cfg in step_map_configs}
    gate3_passed = bool(all(acc >= 0.95 for acc in sign_accs.values()))

    coverages = {cfg: estimators_summary[cfg]["mean_coverage_90"] for cfg in step_map_configs}
    gate4_passed = bool(all(cov >= 0.85 for cov in coverages.values()))

    sharpnesses = {cfg: estimators_summary[cfg]["mean_interval_sharpness"] for cfg in step_map_configs}
    gate5_passed = bool(all(sharp < 0.25 for sharp in sharpnesses.values()))

    hard_violations = {cfg: estimators_summary[cfg]["hard_constraint_violations"] for cfg in step_map_configs}
    gate6_passed = bool(all(v == 0 for v in hard_violations.values()))

    all_gates_passed = (
        gate1_passed and gate2_passed and gate3_passed and gate4_passed and gate5_passed and gate6_passed
    )

    acceptance_ledger = {
        "gate_1_causal_improvement_ge_20pct": {
            "name": "Causal Baseline Improvement >= 20.0%",
            "threshold": 0.20,
            "persistence_median_delta_nrmse": persistence_nrmse,
            "causal_declared_reduction_pct": round(causal_decl_impr * 100.0, 2),
            "causal_undeclared_reduction_pct": round(causal_undecl_impr * 100.0, 2),
            "passed": gate1_passed,
        },
        "gate_2_signed_amplitude_error_le_0_25std": {
            "name": "Median Signed Amplitude Error <= 0.25 sigma_train",
            "threshold": 0.25,
            "values": amp_errors,
            "passed": gate2_passed,
        },
        "gate_3_sign_accuracy_ge_95pct": {
            "name": "Sign Accuracy (attacks >= 1.0 sigma) >= 95.0%",
            "threshold": 0.95,
            "values": sign_accs,
            "passed": gate3_passed,
        },
        "gate_4_empirical_coverage_ge_85pct": {
            "name": "Empirical 90% Confidence Interval Coverage >= 85.0%",
            "threshold": 0.85,
            "values": coverages,
            "passed": gate4_passed,
        },
        "gate_5_sharpness_lt_0_25m": {
            "name": "Mean Interval Sharpness < 0.25 m (< 5% tank range)",
            "threshold": 0.25,
            "values": sharpnesses,
            "passed": gate5_passed,
        },
        "gate_6_zero_box_violations": {
            "name": "Zero Hard Box-Constraint Violations",
            "threshold": 0,
            "values": hard_violations,
            "passed": gate6_passed,
        },
        "all_gates_passed": all_gates_passed,
        "readiness_certification": (
            "ACCEPTED: ALL GATES SATISFIED ON SEALED FINAL TEST DATA" if all_gates_passed else "REJECTED"
        ),
    }

    return {
        "manifest_version": "1.0",
        "dataset": "batadal",
        "target_channel": target_channel,
        "train_std_meters": round(train_std, 5),
        "n_scenarios": n_scenarios,
        "total_evaluation_time_seconds": round(total_time, 2),
        "champion_checkpoint": ckpt_path,
        "data_seal_unsealed_audit": {
            "val_total_rows": 1440,
            "clean_calibration_rows": [0, 480],
            "development_injection_rows": [480, 960],
            "sealed_final_injection_rows": [960, 1440],
            "unsealed_status": "OFFICIALLY_UNSEALED_EVALUATED",
            "unsealed_evaluation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_scenarios_evaluated": n_scenarios,
        },
        "acceptance_gates": acceptance_ledger,
        "acceptance_ledger": acceptance_ledger,
        "estimators": estimators_summary,
    }


def _generate_final_plots(
    results_df: pd.DataFrame,
    summary_dict: dict[str, Any],
    plots_dir: Path,
) -> None:
    estimators = summary_dict["estimators"]

    # -------------------------------------------------------------------------
    # Plot 1: final_nrmse_by_duration.png
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 5.5), dpi=180)
    durations = sorted(results_df["duration_hours"].unique())

    styles = {
        "PersistenceEstimator": ("#7f7f7f", ":", "o", "Persistence (Causal Baseline)"),
        "CompanionRegressionEstimator": ("#9467bd", "--", "s", "Companion Ridge (Linear)"),
        "StepMAP_TCN_Declared": ("#2ca02c", "-", "^", "StepMAP TCN (Offline Declared)"),
        "StepMAP_TCN_Undeclared": ("#74c476", "--", "v", "StepMAP TCN (Offline Undeclared)"),
        "CausalStepMAP_TCN_Declared": ("#1f77b4", "-", "D", "CausalStepMAP TCN (Online Declared)"),
        "CausalStepMAP_TCN_Undeclared": ("#6baed6", "--", "P", "CausalStepMAP TCN (Online Undeclared)"),
    }

    for est_name, (color, ls, marker, label) in styles.items():
        vals = [
            results_df[(results_df["estimator"] == est_name) & (results_df["duration_hours"] == d)]["delta_nrmse"].median()
            for d in durations
        ]
        ax1.plot(durations, vals, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=7, label=label)

    ax1.set_xlabel("Attack Duration (Hours)", fontsize=10)
    ax1.set_ylabel("Median Delta NRMSE (Lower is Better)", fontsize=10)
    ax1.set_title("Sealed Final Test Block: Recovery NRMSE by Attack Duration (180 Scenarios)", fontsize=11, fontweight="bold")
    ax1.set_xticks(durations)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    fig1.tight_layout()
    fig1.savefig(plots_dir / "final_nrmse_by_duration.png", dpi=180)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # Plot 2: final_nrmse_by_amplitude.png
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(9, 5.5), dpi=180)
    amplitudes = sorted(results_df["amplitude_train_std"].unique())

    for est_name, (color, ls, marker, label) in styles.items():
        vals = [
            results_df[(results_df["estimator"] == est_name) & (results_df["amplitude_train_std"] == a)]["delta_nrmse"].median()
            for a in amplitudes
        ]
        ax2.plot(amplitudes, vals, color=color, linestyle=ls, marker=marker, linewidth=2.0, markersize=7, label=label)

    ax2.set_xlabel(r"Injected Attack Amplitude ($\sigma_{\mathrm{train}}$)", fontsize=10)
    ax2.set_ylabel("Median Delta NRMSE (Lower is Better)", fontsize=10)
    ax2.set_title("Sealed Final Test Block: Recovery NRMSE by Attack Amplitude (180 Scenarios)", fontsize=11, fontweight="bold")
    ax2.set_xticks(amplitudes)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    fig2.tight_layout()
    fig2.savefig(plots_dir / "final_nrmse_by_amplitude.png", dpi=180)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # Plot 3: final_coverage_sharpness.png
    # -------------------------------------------------------------------------
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11, 5), dpi=180)
    sm_configs = [
        "StepMAP_TCN_Declared",
        "StepMAP_TCN_Undeclared",
        "CausalStepMAP_TCN_Declared",
        "CausalStepMAP_TCN_Undeclared",
    ]
    sm_labels = [
        "StepMAP\nDeclared",
        "StepMAP\nUndeclared",
        "CausalStepMAP\nDeclared",
        "CausalStepMAP\nUndeclared",
    ]
    cov_vals = [estimators[c]["mean_coverage_90"] * 100.0 for c in sm_configs]
    sharp_vals = [estimators[c]["mean_interval_sharpness"] for c in sm_configs]
    bar_colors = ["#2ca02c", "#74c476", "#1f77b4", "#6baed6"]

    # Coverage panel
    bars_cov = ax3a.bar(range(len(sm_configs)), cov_vals, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.55)
    ax3a.axhline(85.0, color="#d62728", linestyle="--", linewidth=1.5, label="Min Acceptance Gate (85.0%)")
    ax3a.axhline(90.0, color="#7f7f7f", linestyle=":", linewidth=1.2, label="Nominal Target (90.0%)")
    ax3a.set_xticks(range(len(sm_configs)))
    ax3a.set_xticklabels(sm_labels, fontsize=8.5)
    ax3a.set_ylabel("Empirical 90% Coverage (%)", fontsize=10)
    ax3a.set_title("Sealed Final: 90% Uncertainty Interval Coverage", fontsize=10.5, fontweight="bold")
    ax3a.set_ylim(70, 100)
    ax3a.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax3a.legend(loc="lower right", fontsize=8.5)
    for b, val in zip(bars_cov, cov_vals):
        ax3a.annotate(f"{val:.2f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Sharpness panel
    bars_sharp = ax3b.bar(range(len(sm_configs)), sharp_vals, color=bar_colors, edgecolor="black", linewidth=0.8, width=0.55)
    ax3b.axhline(0.25, color="#d62728", linestyle="--", linewidth=1.5, label="Max Sharpness Gate (< 0.25 m)")
    ax3b.set_xticks(range(len(sm_configs)))
    ax3b.set_xticklabels(sm_labels, fontsize=8.5)
    ax3b.set_ylabel("Mean Confidence Band Width (m)", fontsize=10)
    ax3b.set_title("Sealed Final: Interval Sharpness", fontsize=10.5, fontweight="bold")
    ax3b.set_ylim(0, 0.30)
    ax3b.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax3b.legend(loc="upper right", fontsize=8.5)
    for b, val in zip(bars_sharp, sharp_vals):
        ax3b.annotate(f"{val:.4f} m", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig3.tight_layout()
    fig3.savefig(plots_dir / "final_coverage_sharpness.png", dpi=180)
    plt.close(fig3)

    # -------------------------------------------------------------------------
    # Plot 4: final_causal_vs_persistence.png
    # -------------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(9, 5.5), dpi=180)

    p_scenarios = results_df[results_df["estimator"] == "PersistenceEstimator"].sort_values("scenario_id")
    c_decl_scenarios = results_df[results_df["estimator"] == "CausalStepMAP_TCN_Declared"].sort_values("scenario_id")
    c_undecl_scenarios = results_df[results_df["estimator"] == "CausalStepMAP_TCN_Undeclared"].sort_values("scenario_id")

    p_nrmse = p_scenarios["delta_nrmse"].to_numpy()
    c_decl_nrmse = c_decl_scenarios["delta_nrmse"].to_numpy()
    c_undecl_nrmse = c_undecl_scenarios["delta_nrmse"].to_numpy()

    # Scatter of Persistence vs CausalStepMAP Delta NRMSE across all 180 scenarios
    ax4.scatter(p_nrmse, c_decl_nrmse, color="#1f77b4", alpha=0.65, edgecolors="none", s=35, label="CausalStepMAP Declared")
    ax4.scatter(p_nrmse, c_undecl_nrmse, color="#ff7f0e", alpha=0.65, marker="^", edgecolors="none", s=35, label="CausalStepMAP Undeclared")

    max_val = max(np.max(p_nrmse), np.max(c_decl_nrmse), np.max(c_undecl_nrmse)) * 1.05
    ax4.plot([0, max_val], [0, max_val], "k--", alpha=0.7, linewidth=1.2, label="Parity Line (y = x)")
    ax4.plot([0, max_val], [0, max_val * 0.80], "r:", alpha=0.8, linewidth=1.5, label="20% Improvement Gate")

    ax4.set_xlabel("Persistence Estimator Delta NRMSE", fontsize=10)
    ax4.set_ylabel("CausalStepMAP TCN Delta NRMSE", fontsize=10)
    ax4.set_title("Scenario-by-Scenario Causal Online Filter vs. Persistence (180 Scenarios)", fontsize=11, fontweight="bold")
    ax4.set_xlim(0, max_val)
    ax4.set_ylim(0, max_val)
    ax4.grid(True, alpha=0.3, linestyle="--")
    ax4.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    fig4.tight_layout()
    fig4.savefig(plots_dir / "final_causal_vs_persistence.png", dpi=180)
    plt.close(fig4)


if __name__ == "__main__":
    run_sealed_final_benchmark()
