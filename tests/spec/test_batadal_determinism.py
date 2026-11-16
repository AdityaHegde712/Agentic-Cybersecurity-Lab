"""Specification tests for BATADAL CausalTCN and ADMM trend filter determinism audit.

Verifies:
1. Script and function contract compliance.
2. Attack trajectory generation across 5 families (step, ramp, trapezoid, pulse, periodic).
3. Representative multi-family slice scenario generation.
4. Exact numerical determinism of DilatedTCNSurrogate inference across runs.
5. Exact numerical determinism of CausalMultiOrderTrendFilterEstimator ADMM solver across runs.
6. Numerical parity calculation and threshold assertion.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from scripts.test_batadal_determinism import (
    build_attack_trajectory,
    compute_scenario_metrics_isolated,
    generate_determinism_scenarios,
    run_determinism_audit,
)
from src.inverse.estimators.multi_order_trend_filter import CausalMultiOrderTrendFilterEstimator


def test_build_attack_trajectory_five_families() -> None:
    """Verify build_attack_trajectory outputs non-zero attack onsets for all 5 families."""
    n = 100
    onset = 20
    offset = 50
    amp = 2.5
    families = ["step", "ramp", "trapezoid", "pulse", "periodic"]

    for fam in families:
        traj = build_attack_trajectory(fam, n_total=n, onset=onset, offset=offset, amplitude=amp)
        assert isinstance(traj, np.ndarray)
        assert traj.shape == (n,)
        assert traj.dtype == np.float64
        # Before onset and after offset must be strictly 0.0
        assert np.all(traj[:onset] == 0.0)
        assert np.all(traj[offset:] == 0.0)
        # Inside attack interval must contain active perturbation
        assert np.any(traj[onset:offset] != 0.0)


def test_generate_determinism_scenarios_ten_slice() -> None:
    """Verify generate_determinism_scenarios returns 10 scenarios spanning 5 families."""
    scenarios = generate_determinism_scenarios(train_std=0.45, total_length=480)
    assert len(scenarios) == 10

    family_counts: dict[str, int] = {}
    for sc in scenarios:
        assert "scenario_id" in sc
        assert "family" in sc
        assert "onset_index" in sc
        assert "offset_index" in sc
        assert "injected_amplitude" in sc
        assert 0 <= sc["onset_index"] < sc["offset_index"] <= 480
        fam = sc["family"]
        family_counts[fam] = family_counts.get(fam, 0) + 1

    expected_families = {"step", "ramp", "trapezoid", "pulse", "periodic"}
    assert set(family_counts.keys()) == expected_families
    for fam in expected_families:
        assert family_counts[fam] == 2, f"Expected 2 scenarios for {fam}, got {family_counts[fam]}"


def test_compute_scenario_metrics_isolated_structure() -> None:
    """Verify isolated scenario metric computation returns required metric keys."""
    n = 100
    onset = 20
    offset = 50
    amp = 1.0

    y = np.ones(n, dtype=np.float64)
    x_true = np.ones(n, dtype=np.float64)
    delta_true = np.zeros(n, dtype=np.float64)
    delta_true[onset:offset] = amp
    delta_hat = delta_true.copy()
    x_hat = y - delta_hat

    metrics = compute_scenario_metrics_isolated(
        y=y,
        x_true=x_true,
        delta_true=delta_true,
        delta_hat=delta_hat,
        x_hat=x_hat,
        onset=onset,
        offset=offset,
        peak_amp=amp,
    )

    required_keys = [
        "nrmse",
        "mae",
        "max_error",
        "delta_l2_norm",
        "delta_linf_norm",
        "tpr",
        "far",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key {key} in compute_scenario_metrics_isolated output"

    # Perfect recovery should yield 0 error
    assert metrics["nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["max_error"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["tpr"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["far"] == pytest.approx(0.0, abs=1e-12)


def test_causal_multi_order_estimator_deterministic_admm() -> None:
    """Verify CausalMultiOrderTrendFilterEstimator produces identical output over repeated runs."""
    rng = np.random.default_rng(12345)
    n = 60
    y = 2.5 + rng.standard_normal(n)
    surrogate_x = 2.5 * np.ones(n)

    est1 = CausalMultiOrderTrendFilterEstimator(
        window_size=24,
        lambda_0=0.05,
        lambda_1=0.5,
        lambda_2=1.0,
        bounds=(0.0, 5.0),
        solver="admm",
    )
    est2 = CausalMultiOrderTrendFilterEstimator(
        window_size=24,
        lambda_0=0.05,
        lambda_1=0.5,
        lambda_2=1.0,
        bounds=(0.0, 5.0),
        solver="admm",
    )

    res1 = est1.reconstruct(y, surrogate_x=surrogate_x)
    res2 = est2.reconstruct(y, surrogate_x=surrogate_x)

    delta_diff = np.max(np.abs(res1.delta_hat - res2.delta_hat))
    x_diff = np.max(np.abs(res1.x_hat - res2.x_hat))

    assert delta_diff == pytest.approx(0.0, abs=1e-12)
    assert x_diff == pytest.approx(0.0, abs=1e-12)


def test_run_determinism_audit_end_to_end(tmp_path: Path) -> None:
    """Verify run_determinism_audit produces valid summary.json and DETERMINISM_REPORT.md."""
    ckpt_path = Path("results/surrogates/vista_tcn_48h_150ep/tcn.pt")
    data_path = Path("data/processed/batadal/test.parquet")
    ch_manifest = Path("configs/manifests/batadal-channel-manifest-v1.json")

    # Fast 2-scenario, 2-pass test on 60-row block
    sub_scenarios = [
        {
            "scenario_id": "test_step",
            "family": "step",
            "amplitude_train_std": 1.0,
            "sign": 1,
            "duration_hours": 12,
            "onset_index": 20,
            "offset_index": 32,
            "injected_amplitude": 0.5,
        },
        {
            "scenario_id": "test_ramp",
            "family": "ramp",
            "amplitude_train_std": 1.0,
            "sign": -1,
            "duration_hours": 12,
            "onset_index": 35,
            "offset_index": 47,
            "injected_amplitude": -0.5,
        },
    ]

    summary = run_determinism_audit(
        data_path=data_path,
        checkpoint_path=ckpt_path,
        channel_manifest_path=ch_manifest,
        num_passes=2,
        num_rows=60,
        output_dir=tmp_path,
        device="cpu",
        scenarios=sub_scenarios,
    )

    assert summary["determinism_verified"] is True
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "DETERMINISM_REPORT.md").exists()
    assert summary["global_max_metric_diff"] <= 1e-12

