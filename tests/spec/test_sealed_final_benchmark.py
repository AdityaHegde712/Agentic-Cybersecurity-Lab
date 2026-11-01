"""LOCKED specification tests for Sealed Final Benchmark Evaluation (November 18 Milestone Step 5).

Verifies the final sealed test benchmark evaluation across all 180 final scenarios:
1. Final Scenario Generation:
   Verify `generate_final_grid_scenarios` generates exactly 180 deterministic scenarios
   covering all 18 combinations (3 amplitudes x 3 durations x 2 signs) with 10 regime-stratified
   onsets each.
2. Partition Integrity:
   Verify `sealed_final_injection` block is strictly rows 960-1440 of `val.parquet`.
3. Deterministic Ranking & Seed Invariance:
   Verify repeatable scenario IDs and onsets using `manifest.seeds['injection']`.
4. Result Schema & Acceptance Invariants:
   Verify `results/narrow_inverse_research/sealed_final_benchmark/summary.json` contains
   all precommitted gate evaluations across all 6 estimators and certifies acceptance.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from src.inverse.benchmark_manifest import BenchmarkManifest
from src.inverse.harness import (
    BenchmarkScenario,
    compute_regime_cutoffs,
    generate_final_grid_scenarios,
)


MANIFEST_PATH = Path("configs/manifests/batadal-benchmark-manifest-v1.json")
VAL_PATH = Path("data/processed/batadal/val.parquet")
TRAIN_PATH = Path("data/processed/batadal/train.parquet")
RESULTS_DIR = Path("results/narrow_inverse_research/sealed_final_benchmark")
SUMMARY_JSON_PATH = RESULTS_DIR / "summary.json"
SUMMARY_CSV_PATH = RESULTS_DIR / "summary.csv"


# =====================================================================
# Test 1: Final Scenario Generation
# =====================================================================

def test_final_scenario_generation() -> None:
    """Verify generate_final_grid_scenarios generates exactly 180 deterministic scenarios."""
    assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
    manifest = BenchmarkManifest.from_json(MANIFEST_PATH.read_text(encoding="utf-8").strip())
    grid = manifest.final_injections

    assert grid.amplitudes_train_std == (0.75, 1.5, 3.0)
    assert grid.durations_hours == (9, 18, 36)
    assert grid.regime_stratified_onset_count == 10
    assert grid.signs == (-1, 1)

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    fit_df = train_df.iloc[0:5840]
    train_std = float(fit_df[manifest.target_sensor].std())
    cutoffs = compute_regime_cutoffs(fit_df[manifest.target_sensor].to_numpy())

    final_df = val_df.iloc[960:1440]
    prev_val = float(val_df[manifest.target_sensor].iloc[959])

    scenarios = generate_final_grid_scenarios(
        manifest=manifest,
        final_series=final_df[manifest.target_sensor].to_numpy(),
        prev_val=prev_val,
        train_std=train_std,
        cutoffs=cutoffs,
    )

    # 1. Exactly 180 scenarios
    assert len(scenarios) == 180, f"Expected 180 final scenarios, got {len(scenarios)}"

    # 2. Check coverage of all 18 combinations (3 amps x 3 durs x 2 signs)
    combos_seen = set()
    for sc in scenarios:
        assert isinstance(sc, BenchmarkScenario)
        assert sc.block_name == "sealed_final_injection"
        assert sc.target_channel == manifest.target_sensor
        assert sc.amplitude_train_std in (0.75, 1.5, 3.0)
        assert sc.duration_hours in (9, 18, 36)
        assert sc.sign in (-1, 1)
        assert sc.offset_index == sc.onset_index + sc.duration_hours
        assert 0 <= sc.onset_index < 480
        assert sc.offset_index <= 480
        assert sc.regime in grid.regime_labels

        combos_seen.add((sc.amplitude_train_std, sc.duration_hours, sc.sign))

    assert len(combos_seen) == 18, f"Expected 18 parameter combinations, found {len(combos_seen)}"

    # 3. Exactly 10 onsets per combo
    for amp in grid.amplitudes_train_std:
        for dur in grid.durations_hours:
            for sign in grid.signs:
                sub = [
                    s for s in scenarios
                    if s.amplitude_train_std == amp and s.duration_hours == dur and s.sign == sign
                ]
                assert len(sub) == 10, f"Expected 10 scenarios for ({amp}, {dur}, {sign}), got {len(sub)}"
                # Distinct onsets within the combination
                onsets = [s.onset_index for s in sub]
                assert len(set(onsets)) == 10, f"Onsets must be without replacement within combo, got {onsets}"


# =====================================================================
# Test 2: Partition Integrity
# =====================================================================

def test_partition_integrity() -> None:
    """Verify sealed_final_injection block is strictly rows 960-1440 of val.parquet."""
    assert MANIFEST_PATH.exists()
    manifest = BenchmarkManifest.from_json(MANIFEST_PATH.read_text(encoding="utf-8").strip())

    sealed_block = next((b for b in manifest.blocks if b.name == "sealed_final_injection"), None)
    assert sealed_block is not None, "sealed_final_injection block missing from manifest"
    assert sealed_block.source_split == "val"
    assert sealed_block.row_start == 960
    assert sealed_block.row_stop == 1440

    val_df = pd.read_parquet(VAL_PATH)
    assert len(val_df) == 1440, f"Expected exactly 1440 rows in val.parquet, got {len(val_df)}"

    final_slice = val_df.iloc[sealed_block.row_start : sealed_block.row_stop]
    assert len(final_slice) == 480, f"Expected 480 rows in final test block, got {len(final_slice)}"


# =====================================================================
# Test 3: Deterministic Ranking & Seed Invariance
# =====================================================================

def test_deterministic_ranking_and_seed_invariance() -> None:
    """Verify repeatable scenario IDs and onsets using manifest.seeds['injection']."""
    manifest = BenchmarkManifest.from_json(MANIFEST_PATH.read_text(encoding="utf-8").strip())
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    fit_df = train_df.iloc[0:5840]
    train_std = float(fit_df[manifest.target_sensor].std())
    cutoffs = compute_regime_cutoffs(fit_df[manifest.target_sensor].to_numpy())

    final_df = val_df.iloc[960:1440]
    prev_val = float(val_df[manifest.target_sensor].iloc[959])

    scenarios_run1 = generate_final_grid_scenarios(
        manifest=manifest,
        final_series=final_df[manifest.target_sensor].to_numpy(),
        prev_val=prev_val,
        train_std=train_std,
        cutoffs=cutoffs,
    )
    scenarios_run2 = generate_final_grid_scenarios(
        manifest=manifest,
        final_series=final_df[manifest.target_sensor].to_numpy(),
        prev_val=prev_val,
        train_std=train_std,
        cutoffs=cutoffs,
    )

    ids1 = [s.scenario_id for s in scenarios_run1]
    ids2 = [s.scenario_id for s in scenarios_run2]
    assert ids1 == ids2, "Scenario IDs must be byte-for-byte identical across runs"

    onsets1 = [s.onset_index for s in scenarios_run1]
    onsets2 = [s.onset_index for s in scenarios_run2]
    assert onsets1 == onsets2, "Scenario onset indices must be identical across deterministic runs"


# =====================================================================
# Test 4: Result Schema & Acceptance Invariants
# =====================================================================

def test_sealed_final_benchmark_result_schema() -> None:
    """Verify run_sealed_final_benchmark outputs valid summary.json with all gates passed."""
    assert SUMMARY_JSON_PATH.exists(), (
        f"Sealed final benchmark summary missing at {SUMMARY_JSON_PATH}. "
        "Run scripts/run_sealed_final_benchmark.py first."
    )
    assert SUMMARY_CSV_PATH.exists(), f"Sealed final benchmark CSV missing at {SUMMARY_CSV_PATH}"

    data = json.loads(SUMMARY_JSON_PATH.read_text(encoding="utf-8"))

    # Estimators check
    assert "estimators" in data
    estimators = data["estimators"]
    required_configs = [
        "PersistenceEstimator",
        "CompanionRegressionEstimator",
        "StepMAP_TCN_Declared",
        "StepMAP_TCN_Undeclared",
        "CausalStepMAP_TCN_Declared",
        "CausalStepMAP_TCN_Undeclared",
    ]
    for cfg in required_configs:
        assert cfg in estimators, f"Missing estimator {cfg} in final benchmark summary"
        est_data = estimators[cfg]
        assert "median_delta_nrmse" in est_data
        assert "median_support_f1" in est_data
        assert "median_signed_amplitude_error_train_std" in est_data
        assert "sign_accuracy_ge_1std" in est_data
        assert "mean_coverage_90" in est_data
        assert "mean_interval_sharpness" in est_data
        assert "hard_constraint_violations" in est_data

    # Acceptance ledger check
    assert "acceptance_ledger" in data
    ledger = data["acceptance_ledger"]
    gates = [
        "gate_1_causal_improvement_ge_20pct",
        "gate_2_signed_amplitude_error_le_0_25std",
        "gate_3_sign_accuracy_ge_95pct",
        "gate_4_empirical_coverage_ge_85pct",
        "gate_5_sharpness_lt_0_25m",
        "gate_6_zero_box_violations",
        "all_gates_passed",
    ]
    for g in gates:
        assert g in ledger, f"Missing gate {g} in acceptance ledger"

    assert ledger["all_gates_passed"] is True, "Precommitted acceptance gates failed on sealed final block!"

    # Seal and metadata check
    assert "data_seal_unsealed_audit" in data
    seal_audit = data["data_seal_unsealed_audit"]
    assert seal_audit["sealed_final_injection_rows"] == [960, 1440]
    assert seal_audit["total_scenarios_evaluated"] == 180

    # CSV validation
    df = pd.read_csv(SUMMARY_CSV_PATH)
    assert len(df) == 180 * len(required_configs), (
        f"Expected {180 * len(required_configs)} rows in summary.csv, got {len(df)}"
    )
    if "block_name" in df.columns:
        assert (df["block_name"] == "sealed_final_injection").all()
