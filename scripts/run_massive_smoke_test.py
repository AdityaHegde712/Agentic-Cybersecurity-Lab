"""Local End-to-End Smoke Test for the Massive Attack Scenario Suite.

Evaluates 50 attack scenarios across 5 morphology families on the development split
of BATADAL telemetry using the champion Dilated TCN surrogate and Causal Multi-Order
Trend Filter Estimator. Enforces strict data sealing of rows 960-1440.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks.massive_generator import AttackBatchGenerator, AttackScenario  # noqa: E402
from src.inverse.estimators import CausalMultiOrderTrendFilterEstimator  # noqa: E402
from src.inverse.harness import compute_scenario_metrics  # noqa: E402
from src.inverse.surrogates import DilatedTCNSurrogate  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for smoke test runner."""
    parser = argparse.ArgumentParser(description="Massive Attack Suite Smoke Test Runner")
    parser.add_argument(
        "--val-path",
        type=str,
        default="data/processed/batadal/val.parquet",
        help="Path to BATADAL validation dataset",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default="data/processed/batadal/train.parquet",
        help="Path to BATADAL train dataset (for std calibration)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/surrogates/vista_tcn_48h_150ep/tcn.pt",
        help="Path to frozen champion TCN checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/massive_attacks/smoke",
        help="Directory to save smoke test artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility",
    )
    return parser.parse_args()


def enforce_strict_data_seal(val_path: Path) -> pd.DataFrame:
    """Load validation telemetry and strictly enforce data seal on rows 960-1440.

    Args:
        val_path: Path to validation dataset parquet file.

    Returns:
        DataFrame containing development split (rows 480-960).
    """
    if not val_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")

    val_df_full = pd.read_parquet(val_path)
    total_rows = len(val_df_full)
    if total_rows != 1440:
        raise ValueError(f"Expected exactly 1440 validation rows, got {total_rows}")

    # Development split is rows 480 to 960
    dev_df = val_df_full.iloc[480:960].copy().reset_index(drop=True)
    assert len(dev_df) == 480, f"Expected 480 development rows, got {len(dev_df)}"

    # STRICT DATA SEAL: Purge full validation dataframe to guarantee zero access to rows 960-1440
    del val_df_full
    if "val_df_full" in locals():
        raise RuntimeError("Data seal violation: val_df_full could not be purged")

    return dev_df


def generate_smoke_scenarios(
    dev_df: pd.DataFrame,
    sensor_cols: list[str],
    train_std: float,
    seed: int,
    target_sensor: str = "L_T1",
    box_bounds: tuple[float, float] = (0.264, 4.908),
) -> list[AttackScenario]:
    """Generate 50 attack scenarios across 5 families and validate against physical bounds.

    Args:
        dev_df: Development telemetry dataframe.
        sensor_cols: List of numeric sensor column names.
        train_std: Normal training standard deviation of target sensor.
        seed: Random seed.
        target_sensor: Sensor identifier.
        box_bounds: Physical operational envelope (min, max).

    Returns:
        List of 50 physically valid AttackScenario instances.
    """
    dev_telemetry = dev_df[sensor_cols].to_numpy(dtype=np.float64)
    gen = AttackBatchGenerator(dev_telemetry, sensor_cols, seed=seed)

    b_min, b_max = box_bounds
    bounds_min = {target_sensor: b_min}
    bounds_max = {target_sensor: b_max}

    families = ["step", "ramp", "drift", "pulse", "sinusoid"]
    # 10 scenario configurations covering durations [12, 24, 48] and amplitudes [0.5, 1.0, 1.5, 2.0] sigma
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

    scenarios: list[AttackScenario] = []

    for fam in families:
        for idx, cfg in enumerate(configs):
            dur = cfg["dur"]
            scale = cfg["sigma_scale"]
            amp = float(scale * train_std)
            scen_id = f"smoke_{fam}_{idx:02d}_d{dur}h_amp{scale:+.1f}s"

            # Search for onset that satisfies the physical envelope
            found = False
            for onset in range(10, 480 - dur - 10):
                scen = gen.generate_scenario(
                    scenario_id=scen_id,
                    family=fam,
                    target_channels=[target_sensor],
                    onset_step=onset,
                    duration_steps=dur,
                    amplitude=amp,
                )
                check = gen.check_physical_envelope(scen, bounds_min, bounds_max)
                if check["is_valid"]:
                    scenarios.append(scen)
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    f"Could not find valid onset satisfying physical bounds for "
                    f"family={fam}, dur={dur}, scale={scale}"
                )

    if len(scenarios) != 50:
        raise ValueError(f"Expected exactly 50 scenarios, generated {len(scenarios)}")

    return scenarios


def main() -> None:
    """Execute the end-to-end smoke test runner."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MASSIVE ATTACK SUITE: 50-SCENARIO LOCAL E2E SMOKE TEST")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # 1. Enforce strict data seal
    val_path = Path(args.val_path)
    dev_df = enforce_strict_data_seal(val_path)
    print("Strict Data Seal Verified: rows 480-960 loaded, rows 960-1440 purged from memory.")

    # 2. Training standard deviation calibration
    train_path = Path(args.train_path)
    train_df = pd.read_parquet(train_path)
    target_sensor = "L_T1"
    train_std = float(train_df[target_sensor].std())
    print(f"Normal dynamics train_std for {target_sensor}: {train_std:.4f} m")

    sensor_cols = [c for c in dev_df.columns if c not in ["timestamp", "label", "attack_id"]]
    box_bounds = (0.264, 4.908)
    b_min, b_max = box_bounds

    # 3. Generate 50 scenarios across 5 families
    t_gen_start = time.perf_counter()
    scenarios = generate_smoke_scenarios(
        dev_df=dev_df,
        sensor_cols=sensor_cols,
        train_std=train_std,
        seed=args.seed,
        target_sensor=target_sensor,
        box_bounds=box_bounds,
    )
    t_gen_elapsed = time.perf_counter() - t_gen_start
    print(f"Generated {len(scenarios)} valid scenarios in {t_gen_elapsed * 1000:.2f} ms.")

    # 4. Export scenarios to Parquet
    scenarios_parquet = output_dir / "smoke_scenarios.parquet"
    gen_dummy = AttackBatchGenerator(
        dev_df[sensor_cols].to_numpy(dtype=np.float64), sensor_cols, seed=args.seed
    )
    gen_dummy.save_parquet(scenarios, scenarios_parquet)
    print(f"Exported scenario ledger to {scenarios_parquet} ({scenarios_parquet.stat().st_size} bytes).")

    # 5. Load champion plant surrogate (TCN)
    ckpt_path = Path(args.checkpoint)
    print(f"Loading champion surrogate from {ckpt_path}...")
    tcn_surrogate = DilatedTCNSurrogate.load(ckpt_path)
    surr_res = tcn_surrogate.reconstruct(dev_df)
    clean_surrogate_x = surr_res.x_hat
    surrogate_unc = (
        float(np.mean(surr_res.pointwise_uncertainty))
        if surr_res.pointwise_uncertainty is not None
        else 0.0418
    )
    print(f"Surrogate baseline uncertainty: {surrogate_unc:.4f} m")

    # 6. Initialize CausalMultiOrderTrendFilterEstimator
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

    clean_x = dev_df[target_sensor].to_numpy(dtype=np.float64)
    target_ch_idx = sensor_cols.index(target_sensor)

    # 7. Evaluate all 50 scenarios
    print(f"\nEvaluating all {len(scenarios)} scenarios with CausalMultiOrderTrendFilterEstimator...")
    t_eval_start = time.perf_counter()
    records: list[dict[str, Any]] = []

    for s_idx, scen in enumerate(scenarios):
        delta_true = scen.delta[:, target_ch_idx]
        attacked_y = clean_x + delta_true

        recon = estimator.reconstruct(attacked_y, surrogate_x=clean_surrogate_x)

        # Check physical boundary violations on recovered clean signal
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
        metrics["onset_step"] = scen.onset_step
        metrics["duration_hours"] = scen.duration_steps
        metrics["amplitude"] = scen.amplitude
        metrics["amplitude_train_std"] = float(scen.amplitude / train_std)
        metrics["box_violations"] = box_violations

        records.append(metrics)

    total_eval_time = time.perf_counter() - t_eval_start
    print(
        f"Completed 50 scenario evaluations in {total_eval_time:.2f}s "
        f"({total_eval_time / len(scenarios):.3f}s / scenario)."
    )

    # 8. Save CSV and JSON summary
    df_results = pd.DataFrame.from_records(records)
    summary_csv = output_dir / "summary.csv"
    df_results.to_csv(summary_csv, index=False)
    print(f"Saved evaluation metrics to {summary_csv}")

    # Aggregate statistics per family
    family_stats: dict[str, Any] = {}
    for fam, group in df_results.groupby("family"):
        family_stats[str(fam)] = {
            "count": int(len(group)),
            "median_delta_nrmse": float(group["delta_nrmse"].median()),
            "mean_delta_nrmse": float(group["delta_nrmse"].mean()),
            "std_delta_nrmse": float(group["delta_nrmse"].std()),
            "mean_signed_amplitude_error_std": float(
                group["signed_amplitude_error_train_std"].mean()
            ),
            "sign_accuracy": float(group["sign_accuracy"].mean()),
            "support_f1": float(group["support_f1"].mean()),
            "coverage_90": float(group["coverage_90"].mean()),
            "interval_sharpness": float(group["interval_sharpness"].mean()),
            "box_violations": int(group["box_violations"].sum()),
        }

    total_box_violations = int(df_results["box_violations"].sum())

    summary_json_data = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_scenarios": len(scenarios),
        "total_execution_time_seconds": round(total_eval_time, 3),
        "total_box_violations": total_box_violations,
        "overall_median_delta_nrmse": float(df_results["delta_nrmse"].median()),
        "overall_mean_delta_nrmse": float(df_results["delta_nrmse"].mean()),
        "overall_sign_accuracy": float(df_results["sign_accuracy"].mean()),
        "overall_coverage_90": float(df_results["coverage_90"].mean()),
        "overall_interval_sharpness": float(df_results["interval_sharpness"].mean()),
        "per_family_stats": family_stats,
    }

    summary_json = output_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_json_data, f, indent=2)
    print(f"Saved summary metrics to {summary_json}")

    # Print summary table
    print("\n" + "=" * 80)
    print("PER-FAMILY SUMMARY RESULTS")
    print(f"{'Family':<12} {'Count':<6} {'Median NRMSE':<14} {'Sign Acc':<10} {'F1':<8} {'Cov 90%':<10} {'Box Viol'}")
    print("-" * 80)
    for fam, stats in family_stats.items():
        print(
            f"{fam:<12} {stats['count']:<6} {stats['median_delta_nrmse']:<14.4f} "
            f"{stats['sign_accuracy']:<10.2f} {stats['support_f1']:<8.4f} "
            f"{stats['coverage_90']:<10.4f} {stats['box_violations']}"
        )
    print("-" * 80)
    print(f"Total Box Violations: {total_box_violations}")
    print(f"Overall Median Delta NRMSE: {summary_json_data['overall_median_delta_nrmse']:.4f}")
    print(f"Total Execution Time: {total_eval_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
