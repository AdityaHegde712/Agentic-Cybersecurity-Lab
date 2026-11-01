"""LOCKED specification and contract tests for the local inverse research harness.

Covers:
- Deterministic SHA-256 hash ranking without replacement.
- Model-fit regime cutoff extraction and onset regime classification.
- Full development injection grid scenario generation (120 scenarios).
- Exact metric computation (NRMSE on declared interval, support F1, signed amplitude error, clean x NRMSE).
- End-to-end execution of the 4 estimators and result artifact schemas.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inverse.benchmark_manifest import build_batadal_benchmark_manifest
from src.inverse.harness import (
    BenchmarkScenario,
    classify_series_regimes,
    compute_regime_cutoffs,
    compute_scenario_metrics,
    generate_development_grid_scenarios,
    run_benchmark_suite,
    sha256_seeded_rank,
)
from src.inverse.estimators import (
    CompanionRegressionEstimator,
    PersistenceEstimator,
    SeasonalNaiveEstimator,
    StepMAPEstimator,
)


CHANNEL_MANIFEST_PATH = Path("configs/manifests/batadal-channel-manifest-v1.json")
BENCHMARK_MANIFEST_PATH = Path("configs/manifests/batadal-benchmark-manifest-v1.json")

BATADAL_CHANNELS = (
    "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7", "F_PU1",
    "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4",
    "F_PU5", "S_PU5", "F_PU6", "S_PU6", "F_PU7", "S_PU7", "F_PU8",
    "S_PU8", "F_PU9", "S_PU9", "F_PU10", "S_PU10", "F_PU11", "S_PU11",
    "F_V2", "S_V2", "P_J280", "P_J269", "P_J300", "P_J256", "P_J289",
    "P_J415", "P_J302", "P_J306", "P_J307", "P_J317", "P_J14", "P_J422",
)


def _synthetic_fit_data(n_rows: int = 5840, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    cycle = np.sin(2 * np.pi * t / 24)
    data: dict[str, np.ndarray] = {
        "L_T1": 2.5 + 1.2 * cycle + 0.1 * rng.standard_normal(n_rows)
    }
    for ch in BATADAL_CHANNELS:
        if ch == "L_T1":
            continue
        if ch.startswith("S_"):
            data[ch] = (cycle > 0).astype(float)
        elif ch.startswith("F_"):
            data[ch] = np.maximum(0.0, 100.0 * cycle + 10.0 * rng.standard_normal(n_rows))
        else:
            data[ch] = 3.0 + 0.8 * cycle + 0.15 * rng.standard_normal(n_rows)
    return pd.DataFrame(data, columns=BATADAL_CHANNELS)


# =====================================================================
# 1. Deterministic SHA-256 Hash Ranking Tests
# =====================================================================

def test_sha256_seeded_rank_is_deterministic_and_complete() -> None:
    candidates = list(range(50))
    seed = 20261013
    context = "low_dynamics:dur24"

    ranked_1 = sha256_seeded_rank(seed, candidates, context=context)
    ranked_2 = sha256_seeded_rank(seed, candidates, context=context)

    # Identical results for identical inputs
    assert ranked_1 == ranked_2
    # All candidates preserved without loss or duplicate
    assert sorted(ranked_1) == candidates
    # Non-trivial permutation (not identical to original ascending order)
    assert ranked_1 != candidates


def test_sha256_seeded_rank_different_seeds_or_contexts_give_different_orders() -> None:
    candidates = list(range(50))
    ranked_seed_a = sha256_seeded_rank(20261013, candidates, context="test")
    ranked_seed_b = sha256_seeded_rank(20261014, candidates, context="test")
    ranked_ctx_c = sha256_seeded_rank(20261013, candidates, context="different_context")

    assert ranked_seed_a != ranked_seed_b
    assert ranked_seed_a != ranked_ctx_c


# =====================================================================
# 2. Regime Cutoff Extraction and Classification Tests
# =====================================================================

def test_compute_regime_cutoffs_returns_ascending_quantiles() -> None:
    series = np.array([1.0, 1.1, 1.3, 1.8, 2.5, 3.5, 4.0])
    q1, q2 = compute_regime_cutoffs(series, quantiles=(1 / 3, 2 / 3))

    assert 0.0 < q1 < q2


def test_classify_series_regimes_partitions_correctly() -> None:
    cutoffs = (0.2, 0.5)
    # Differences: [abs(1.1-1.0)=0.1, abs(1.4-1.1)=0.3, abs(2.0-1.4)=0.6]
    series = np.array([1.1, 1.4, 2.0])
    regimes = classify_series_regimes(series, prev_val=1.0, cutoffs=cutoffs)

    assert regimes == ["low_dynamics", "medium_dynamics", "high_dynamics"]


# =====================================================================
# 3. Development Grid Scenario Generation Tests
# =====================================================================

def test_generate_development_grid_scenarios_matches_manifest_specification() -> None:
    manifest = build_batadal_benchmark_manifest()
    fit_df = _synthetic_fit_data(5840, seed=1)
    dev_df = _synthetic_fit_data(480, seed=2)
    
    train_std = float(fit_df["L_T1"].std())
    cutoffs = compute_regime_cutoffs(fit_df["L_T1"].to_numpy())
    prev_val = float(fit_df["L_T1"].iloc[-1])

    scenarios = generate_development_grid_scenarios(
        manifest=manifest,
        dev_series=dev_df["L_T1"].to_numpy(),
        prev_val=prev_val,
        train_std=train_std,
        cutoffs=cutoffs,
    )

    # 3 amplitudes * 4 durations * 2 signs * 5 onsets = 120 scenarios
    assert len(scenarios) == 120

    scenario_ids = [s.scenario_id for s in scenarios]
    assert len(set(scenario_ids)) == 120, "All scenario IDs must be unique"

    # Validate each scenario conforms to constraints
    durations_seen = set()
    amplitudes_seen = set()
    signs_seen = set()
    regimes_count = {"low_dynamics": 0, "medium_dynamics": 0, "high_dynamics": 0}

    for sc in scenarios:
        assert sc.target_channel == "L_T1"
        assert 0 <= sc.onset_index < 480
        assert sc.offset_index == sc.onset_index + sc.duration_hours
        assert sc.offset_index <= 480, f"Offset {sc.offset_index} exceeds block boundary 480"
        assert sc.sign in (-1, 1)
        assert sc.amplitude_train_std in (0.5, 1.0, 2.0)
        assert sc.duration_hours in (6, 12, 24, 48)
        assert np.isclose(sc.injected_amplitude, sc.sign * sc.amplitude_train_std * train_std)
        assert sc.regime in regimes_count

        durations_seen.add(sc.duration_hours)
        amplitudes_seen.add(sc.amplitude_train_std)
        signs_seen.add(sc.sign)
        regimes_count[sc.regime] += 1

    assert durations_seen == {6, 12, 24, 48}
    assert amplitudes_seen == {0.5, 1.0, 2.0}
    assert signs_seen == {-1, 1}

    # Stratified balance: each regime receives exactly 40 scenarios (120 / 3)
    assert regimes_count["low_dynamics"] == 40
    assert regimes_count["medium_dynamics"] == 40
    assert regimes_count["high_dynamics"] == 40


def test_scenario_generation_is_strictly_deterministic() -> None:
    manifest = build_batadal_benchmark_manifest()
    fit_df = _synthetic_fit_data(5840, seed=10)
    dev_df = _synthetic_fit_data(480, seed=20)
    train_std = float(fit_df["L_T1"].std())
    cutoffs = compute_regime_cutoffs(fit_df["L_T1"].to_numpy())
    prev_val = float(fit_df["L_T1"].iloc[-1])

    scenarios_a = generate_development_grid_scenarios(
        manifest, dev_df["L_T1"].to_numpy(), prev_val, train_std, cutoffs
    )
    scenarios_b = generate_development_grid_scenarios(
        manifest, dev_df["L_T1"].to_numpy(), prev_val, train_std, cutoffs
    )

    assert scenarios_a == scenarios_b


# =====================================================================
# 4. Metric Computation Unit Tests
# =====================================================================

def test_compute_scenario_metrics_perfect_recovery() -> None:
    N = 100
    onset, offset = 20, 50
    amp = 2.0
    train_std = 1.0

    delta_true = np.zeros(N)
    delta_true[onset:offset] = amp
    delta_hat = delta_true.copy()

    x_true = np.linspace(1.0, 3.0, N)
    x_hat = x_true.copy()

    metrics = compute_scenario_metrics(
        delta_hat=delta_hat,
        delta_true=delta_true,
        x_hat=x_hat,
        x_true=x_true,
        onset=onset,
        offset=offset,
        injected_amplitude=amp,
        train_std=train_std,
    )

    assert metrics["delta_nrmse"] == 0.0
    assert metrics["support_f1"] == 1.0
    assert metrics["support_precision"] == 1.0
    assert metrics["support_recall"] == 1.0
    assert metrics["signed_amplitude_error_train_std"] == 0.0
    assert metrics["clean_x_nrmse"] == 0.0
    assert metrics["sign_accuracy"] == 1.0


def test_compute_scenario_metrics_zero_recovery_failure() -> None:
    N = 100
    onset, offset = 20, 50
    amp = 2.0
    train_std = 1.0

    delta_true = np.zeros(N)
    delta_true[onset:offset] = amp
    # Estimator guesses zero everywhere
    delta_hat = np.zeros(N)

    x_true = np.linspace(1.0, 3.0, N)
    # Since observed = x_true + delta_true, if delta_hat=0 then x_hat = observed
    x_hat = x_true + delta_true

    metrics = compute_scenario_metrics(
        delta_hat=delta_hat,
        delta_true=delta_true,
        x_hat=x_hat,
        x_true=x_true,
        onset=onset,
        offset=offset,
        injected_amplitude=amp,
        train_std=train_std,
    )

    assert metrics["delta_nrmse"] == 1.0
    assert metrics["support_f1"] == 0.0
    assert metrics["support_recall"] == 0.0
    # Median guess on support is 0, so error is (0 - amp) / train_std = -2.0
    assert metrics["signed_amplitude_error_train_std"] == -2.0
    assert np.isclose(metrics["clean_x_nrmse"], 2.0)
    assert metrics["sign_accuracy"] == 0.0


def test_compute_scenario_metrics_thresholding_and_negative_sign() -> None:
    N = 100
    onset, offset = 10, 30
    amp = -3.0
    train_std = 1.5

    delta_true = np.zeros(N)
    delta_true[onset:offset] = amp

    # Estimator recovers -2.7 on attack, and 0.2 noise outside
    delta_hat = np.zeros(N)
    delta_hat[onset:offset] = -2.7
    delta_hat[0:5] = 0.2  # below 0.1 * |amp| = 0.3 threshold, so should be ignored by support mask

    x_true = np.full(N, 2.0)
    x_hat = (x_true + delta_true) - delta_hat

    metrics = compute_scenario_metrics(
        delta_hat=delta_hat,
        delta_true=delta_true,
        x_hat=x_hat,
        x_true=x_true,
        onset=onset,
        offset=offset,
        injected_amplitude=amp,
        train_std=train_std,
    )

    # 0.2 noise outside is < 0.1 * 3.0 = 0.3, so support FP = 0 -> F1 = 1.0
    assert metrics["support_f1"] == 1.0
    # RMSE = 0.3, NRMSE = 0.3 / 3.0 = 0.1
    assert np.isclose(metrics["delta_nrmse"], 0.1)
    # Signed error = (-2.7 - (-3.0)) / 1.5 = +0.3 / 1.5 = +0.2
    assert np.isclose(metrics["signed_amplitude_error_train_std"], 0.2)
    assert metrics["sign_accuracy"] == 1.0


# =====================================================================
# 5. End-to-End Benchmark Suite Execution & Artifact Schema Tests
# =====================================================================

def test_run_benchmark_suite_synthetic_smoke() -> None:
    fit_df = _synthetic_fit_data(n_rows=5840, seed=100)
    calib_df = _synthetic_fit_data(n_rows=480, seed=200)
    dev_df = _synthetic_fit_data(n_rows=480, seed=300)
    train_std = float(fit_df["L_T1"].std())

    # Build 2 synthetic test scenarios
    scenarios = [
        BenchmarkScenario(
            scenario_id="smoke_scenario_1",
            target_channel="L_T1",
            amplitude_train_std=1.0,
            sign=1,
            duration_hours=12,
            onset_index=20,
            offset_index=32,
            regime="low_dynamics",
            injected_amplitude=train_std,
        ),
        BenchmarkScenario(
            scenario_id="smoke_scenario_2",
            target_channel="L_T1",
            amplitude_train_std=2.0,
            sign=-1,
            duration_hours=24,
            onset_index=50,
            offset_index=74,
            regime="high_dynamics",
            injected_amplitude=-2.0 * train_std,
        ),
    ]

    results_df, summary_dict = run_benchmark_suite(
        scenarios=scenarios,
        fit_df=fit_df,
        calib_df=calib_df,
        dev_df=dev_df,
        target_channel="L_T1",
        channel_manifest_path=CHANNEL_MANIFEST_PATH,
        train_std=train_std,
    )

    # 2 scenarios * 4 estimators = 8 result rows
    assert len(results_df) == 8
    expected_cols = {
        "scenario_id", "estimator", "amplitude_train_std", "duration_hours",
        "sign", "regime", "onset_index", "offset_index", "injected_amplitude",
        "delta_nrmse", "support_f1", "support_precision", "support_recall",
        "signed_amplitude_error_train_std", "clean_x_nrmse", "sign_accuracy",
    }
    assert expected_cols.issubset(results_df.columns)

    # Verify summary schema
    assert "metadata" in summary_dict
    assert "estimators" in summary_dict
    assert "acceptance_checks" in summary_dict
    assert set(summary_dict["estimators"]) == {
        "PersistenceEstimator",
        "SeasonalNaiveEstimator",
        "CompanionRegressionEstimator",
        "StepMAPEstimator",
    }
