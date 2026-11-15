"""Specification tests for High-Throughput Attack Scenario Generator.

Validates deterministic reproducibility, waveform mathematical correctness,
support mask alignment, physical envelope validation, scaling throughput (< 2.0s
for 1,000 scenarios), metadata conversion, and Parquet serialization.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.attacks.massive_generator import (
    AttackBatchGenerator,
    AttackScenario,
)


@pytest.fixture
def synthetic_telemetry() -> tuple[np.ndarray, list[str]]:
    """Create a synthetic 1,000-timestep, 5-channel baseline telemetry dataset."""
    rng = np.random.default_rng(12345)
    n_steps = 1000
    n_channels = 5
    channel_names = [f"sensor_{i}" for i in range(n_channels)]
    # Base telemetry around nominal level 50.0 +/- 5.0
    telemetry = 50.0 + rng.normal(0.0, 1.0, size=(n_steps, n_channels))
    return telemetry, channel_names


def test_deterministic_reproducibility(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify identical seeds yield byte-identical scenarios across stochastic families."""
    telemetry, channels = synthetic_telemetry

    gen1 = AttackBatchGenerator(telemetry, channels, seed=42)
    gen2 = AttackBatchGenerator(telemetry, channels, seed=42)
    gen3 = AttackBatchGenerator(telemetry, channels, seed=99)

    scen1_ou = gen1.generate_scenario(
        scenario_id="ou_test_1",
        family="ou",
        target_channels=["sensor_0"],
        onset_step=100,
        duration_steps=50,
        amplitude=10.0,
    )

    scen2_ou = gen2.generate_scenario(
        scenario_id="ou_test_1",
        family="ou",
        target_channels=["sensor_0"],
        onset_step=100,
        duration_steps=50,
        amplitude=10.0,
    )

    scen3_ou = gen3.generate_scenario(
        scenario_id="ou_test_1",
        family="ou",
        target_channels=["sensor_0"],
        onset_step=100,
        duration_steps=50,
        amplitude=10.0,
    )

    np.testing.assert_array_equal(scen1_ou.delta, scen2_ou.delta)
    np.testing.assert_array_equal(scen1_ou.support_mask, scen2_ou.support_mask)

    # Different seed must yield distinct stochastic trajectories
    assert not np.allclose(scen1_ou.delta, scen3_ou.delta)


def test_waveform_canonical_families(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify mathematical correctness of canonical profiles: step, ramp, drift, pulse, periodic."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)
    onset = 100
    duration = 40
    amp = 5.0
    ch = "sensor_1"
    ch_idx = channels.index(ch)

    # 1. Step
    step_scen = gen.generate_scenario("s1", "step", [ch], onset, duration, amp)
    active_step = step_scen.delta[onset : onset + duration, ch_idx]
    np.testing.assert_allclose(active_step, amp)

    # 2. Ramp (starts at 0.0, ends at amp)
    ramp_scen = gen.generate_scenario("s2", "ramp", [ch], onset, duration, amp)
    active_ramp = ramp_scen.delta[onset : onset + duration, ch_idx]
    assert np.isclose(active_ramp[0], 0.0)
    assert np.isclose(active_ramp[-1], amp)
    # Monotonically increasing
    assert np.all(np.diff(active_ramp) >= 0.0)

    # 3. Drift (linear accumulating)
    drift_scen = gen.generate_scenario("s3", "drift", [ch], onset, duration, amp)
    active_drift = drift_scen.delta[onset : onset + duration, ch_idx]
    assert np.isclose(active_drift[0], 0.0)
    assert np.all(np.diff(active_drift) >= 0.0)

    # 4. Pulse (shock of constant amplitude)
    pulse_scen = gen.generate_scenario("s4", "pulse", [ch], onset, duration, amp)
    active_pulse = pulse_scen.delta[onset : onset + duration, ch_idx]
    np.testing.assert_allclose(active_pulse, amp)

    # 5. Periodic (sinusoid)
    period = 10
    periodic_scen = gen.generate_scenario(
        "s5",
        "periodic",
        [ch],
        onset,
        duration,
        amp,
        parameters={"period": period, "phase": 0.0},
    )
    active_periodic = periodic_scen.delta[onset : onset + duration, ch_idx]
    expected_sin = amp * np.sin(2.0 * np.pi * np.arange(duration) / period)
    np.testing.assert_allclose(active_periodic, expected_sin, atol=1e-12)


def test_waveform_complex_families(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify mathematical correctness of complex profiles: trapezoid, sigmoidal, chirp."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)
    onset = 200
    duration = 50
    amp = 8.0
    ch = "sensor_2"
    ch_idx = channels.index(ch)

    # 1. Trapezoid (ramp-hold-decay)
    trap_scen = gen.generate_scenario(
        "c1",
        "trapezoid",
        [ch],
        onset,
        duration,
        amp,
        parameters={"u1": 0.2, "u2": 0.8},
    )
    active_trap = trap_scen.delta[onset : onset + duration, ch_idx]
    assert np.isclose(active_trap[0], 0.0)
    assert np.isclose(active_trap[-1], 0.0, atol=1e-1)
    # Peak hold region at amplitude (where u >= u1 and u < u2)
    hold_start = int(np.ceil(0.2 * (duration - 1)))
    hold_end = int(0.8 * (duration - 1))
    np.testing.assert_allclose(active_trap[hold_start:hold_end], amp)

    # 2. Sigmoidal (S-curve)
    sig_scen = gen.generate_scenario("c2", "sigmoidal", [ch], onset, duration, amp)
    active_sig = sig_scen.delta[onset : onset + duration, ch_idx]
    assert np.isclose(active_sig[0], 0.0, atol=1e-1)
    assert np.isclose(active_sig[-1], amp, atol=1e-1)
    # S-curve should be monotonic
    assert np.all(np.diff(active_sig) >= -1e-12)

    # 3. Chirp (frequency-modulated sinusoid)
    chirp_scen = gen.generate_scenario(
        "c3", "chirp", [ch], onset, duration, amp, parameters={"f0": 0.02, "f1": 0.2}
    )
    active_chirp = chirp_scen.delta[onset : onset + duration, ch_idx]
    # Chirp amplitude bounded by amp
    assert np.max(np.abs(active_chirp)) <= amp + 1e-12
    # Chirp should oscillate (both positive and negative values)
    assert np.any(active_chirp > 0.0)
    assert np.any(active_chirp < 0.0)


def test_waveform_stochastic_families(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify stochastic properties of brownian random walk and Ornstein-Uhlenbeck processes."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)
    onset = 50
    duration = 100
    amp = 4.0
    ch = "sensor_0"
    ch_idx = channels.index(ch)

    # 1. Brownian walk
    brown_scen = gen.generate_scenario("st1", "brownian", [ch], onset, duration, amp)
    active_brown = brown_scen.delta[onset : onset + duration, ch_idx]
    assert len(active_brown) == duration
    # Differences should be stochastic increments
    increments = np.diff(active_brown)
    assert np.std(increments) > 0.0

    # 2. Ornstein-Uhlenbeck process (mean-reverting towards target)
    ou_scen = gen.generate_scenario(
        "st2",
        "ou",
        [ch],
        onset,
        duration,
        amp,
        parameters={"theta": 0.3, "sigma": 0.2, "target": amp},
    )
    active_ou = ou_scen.delta[onset : onset + duration, ch_idx]
    assert len(active_ou) == duration
    # Later half should fluctuate around target amplitude
    assert np.isclose(np.mean(active_ou[duration // 2 :]), amp, atol=1.5)


def test_exact_support_mask_alignment(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify support_mask is True only on injection window and targeted channels with zero off-support perturbation."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)
    onset = 300
    duration = 80
    amp = 7.5
    targets = ["sensor_1", "sensor_3"]
    target_indices = [channels.index(c) for c in targets]
    untargeted_indices = [i for i in range(len(channels)) if i not in target_indices]

    scen = gen.generate_scenario("s_mask", "step", targets, onset, duration, amp)

    # 1. Support mask shape matches telemetry
    assert isinstance(scen, AttackScenario)
    assert scen.support_mask.shape == telemetry.shape
    assert scen.delta.shape == telemetry.shape

    # 2. Targeted window is active
    for idx in target_indices:
        assert np.all(scen.support_mask[onset : onset + duration, idx])
        assert np.all(scen.delta[onset : onset + duration, idx] == amp)

    # 3. Outside injection window is False
    assert not np.any(scen.support_mask[:onset, :])
    assert not np.any(scen.support_mask[onset + duration :, :])

    # 4. Untargeted channels are False everywhere
    for idx in untargeted_indices:
        assert not np.any(scen.support_mask[:, idx])

    # 5. Zero off-support perturbation invariant
    assert np.all(scen.delta[~scen.support_mask] == 0.0)


def test_multi_channel_injection(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify injection correctly corrupts multiple selected channels simultaneously."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)
    onset = 150
    duration = 60
    amp = 3.0
    all_targets = ["sensor_0", "sensor_2", "sensor_4"]

    scen = gen.generate_scenario("multi_1", "ramp", all_targets, onset, duration, amp)

    assert scen.target_channels == all_targets
    for target in all_targets:
        idx = channels.index(target)
        active_slice = scen.delta[onset : onset + duration, idx]
        assert np.isclose(active_slice[0], 0.0)
        assert np.isclose(active_slice[-1], amp)


def test_physical_boundary_checking(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify validation against physical operational bounds (envelope constraints)."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)

    min_bounds = {ch: 40.0 for ch in channels}
    max_bounds = {ch: 60.0 for ch in channels}

    # Safe scenario: base is ~50, amplitude +2 -> total ~52, within [40, 60]
    safe_scen = gen.generate_scenario(
        "safe", "step", ["sensor_0"], 100, 50, amplitude=2.0
    )
    check_safe = gen.check_physical_envelope(
        safe_scen, min_bounds=min_bounds, max_bounds=max_bounds
    )
    assert check_safe["is_valid"] is True
    assert check_safe["violation_count"] == 0

    # Violating scenario: amplitude +30 -> total ~80, violates max bound 60.0
    violating_scen = gen.generate_scenario(
        "violating", "step", ["sensor_0"], 100, 50, amplitude=30.0
    )
    check_viol = gen.check_physical_envelope(
        violating_scen, min_bounds=min_bounds, max_bounds=max_bounds
    )
    assert check_viol["is_valid"] is False
    assert check_viol["violation_count"] > 0
    assert "sensor_0" in check_viol["channels_violated"]


def test_high_throughput_scaling(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify high-throughput scaling: generating 1,000 scenarios in < 2.0 seconds."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)

    start_time = time.perf_counter()
    scenarios = gen.generate_batch(
        families=["step", "ramp", "periodic", "trapezoid", "brownian"],
        channels=channels,
        onsets=[50, 150, 300, 500],
        durations=[20, 50, 100],
        amplitudes=[1.0, 5.0, 10.0],
        n_scenarios=1000,
    )
    elapsed = time.perf_counter() - start_time

    assert len(scenarios) == 1000
    assert elapsed < 2.0, (
        f"Generating 1,000 scenarios took {elapsed:.3f}s (budget: < 2.0s)"
    )


def test_metadata_and_dataframe_conversion(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify scenarios convert to pandas DataFrame with complete scenario ledger metadata."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)

    scenarios = gen.generate_batch(
        families=["step", "ramp", "chirp"],
        channels=["sensor_0", "sensor_1"],
        n_scenarios=10,
    )

    df = gen.to_dataframe(scenarios)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    required_cols = {
        "scenario_id",
        "family",
        "target_channels",
        "onset_step",
        "duration_steps",
        "amplitude",
        "l2_norm",
        "max_abs_delta",
    }
    assert required_cols.issubset(df.columns)


def test_parquet_serialization_roundtrip(
    synthetic_telemetry: tuple[np.ndarray, list[str]], tmp_path: Path
) -> None:
    """Verify Parquet export and roundtrip deserialization."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)

    scenarios = gen.generate_batch(
        families=["step", "periodic", "sigmoidal"],
        channels=["sensor_2"],
        n_scenarios=25,
    )

    parquet_file = tmp_path / "scenario_ledger.parquet"
    gen.save_parquet(scenarios, parquet_file)

    assert parquet_file.exists()
    assert parquet_file.stat().st_size > 0

    df_loaded = pd.read_parquet(parquet_file)
    assert len(df_loaded) == 25
    assert "scenario_id" in df_loaded.columns
    assert "family" in df_loaded.columns


def test_invalid_parameters_raise_clean_errors(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify input validation fails explicitly on invalid configuration parameters."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)

    # 1. Invalid onset step (< 0)
    with pytest.raises(ValueError, match="onset_step"):
        gen.generate_scenario("e1", "step", ["sensor_0"], -5, 50, 1.0)

    # 2. Invalid duration (<= 0)
    with pytest.raises(ValueError, match="duration_steps"):
        gen.generate_scenario("e2", "step", ["sensor_0"], 50, 0, 1.0)

    # 3. Attack window exceeds telemetry length
    with pytest.raises(ValueError, match="exceeds"):
        gen.generate_scenario("e3", "step", ["sensor_0"], 950, 100, 1.0)

    # 4. Unknown channel name
    with pytest.raises(ValueError, match="Unknown channel"):
        gen.generate_scenario("e4", "step", ["nonexistent_sensor"], 50, 50, 1.0)

    # 5. Unknown family
    with pytest.raises(ValueError, match="Unsupported family"):
        gen.generate_scenario("e5", "invalid_family", ["sensor_0"], 50, 50, 1.0)


def test_family_aliases(synthetic_telemetry: tuple[np.ndarray, list[str]]) -> None:
    """Verify alias mapping canonicalizes correctly."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)

    scen_sin = gen.generate_scenario("al1", "sinusoid", ["sensor_0"], 10, 20, 1.0)
    assert scen_sin.family == "periodic"

    scen_sig = gen.generate_scenario("al2", "sigmoid", ["sensor_0"], 10, 20, 1.0)
    assert scen_sig.family == "sigmoidal"

    scen_rw = gen.generate_scenario("al3", "random_walk", ["sensor_0"], 10, 20, 1.0)
    assert scen_rw.family == "brownian"

    scen_ou = gen.generate_scenario(
        "al4", "ornstein_uhlenbeck", ["sensor_0"], 10, 20, 1.0
    )
    assert scen_ou.family == "ou"


def test_generate_family_and_massive_corpus(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify generate_family and generate_massive_corpus across all 10 families."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)

    family_scens = gen.generate_family("chirp", n_scenarios=5)
    assert len(family_scens) == 5
    assert all(s.family == "chirp" for s in family_scens)

    corpus_df = gen.generate_massive_corpus(n_total=50)
    assert isinstance(corpus_df, pd.DataFrame)
    assert len(corpus_df) == 50
    # Must contain multiple families
    assert len(corpus_df["family"].unique()) >= 5


def test_bounds_array_and_defaults(
    synthetic_telemetry: tuple[np.ndarray, list[str]],
) -> None:
    """Verify envelope check with ndarray bounds and None defaults."""
    telemetry, channels = synthetic_telemetry
    gen = AttackBatchGenerator(telemetry, channels, seed=42)

    scen = gen.generate_scenario(
        "safe_arr", "step", ["sensor_0"], 50, 20, amplitude=0.0
    )

    # Defaults (None bounds uses telemetry min/max)
    res_default = gen.check_physical_envelope(scen)
    assert res_default["is_valid"] is True

    # Array bounds
    min_arr = np.full(len(channels), -100.0)
    max_arr = np.full(len(channels), 100.0)
    res_arr = gen.check_physical_envelope(scen, min_bounds=min_arr, max_bounds=max_arr)
    assert res_arr["is_valid"] is True
