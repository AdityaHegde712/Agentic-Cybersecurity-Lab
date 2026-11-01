"""LOCKED specification tests for Undeclared-Support Inverse Recovery.

Verifies:
1. `src.inverse.harness.evaluate_inverse_benchmark` accepts `declared_support: bool = True` (default)
   and `declared_support: bool = False`.
2. When `declared_support=False`, `reconstruct()` is invoked with `attack_interval=None`.
3. Negative control on clean, unattacked normal series: `StepMAPEstimator` reconstructs
   delta_hat approx 0 with 0 false positive support points.
4. Undeclared step recovery on synthetic step attack (1.0 sigma, 24h) embedded in validation data:
   `StepMAPEstimator` achieves Support F1 >= 0.75, sign accuracy = 1.0, and bounded NRMSE without oracle interval.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.inverse.estimators import (
    BaseDeltaEstimator,
    ReconstructionResult,
    StepMAPEstimator,
)
from src.inverse.harness import (
    BenchmarkScenario,
    compute_scenario_metrics,
    evaluate_inverse_benchmark,
)
from src.inverse.surrogates import DilatedTCNSurrogate


CHANNEL_MANIFEST_PATH = Path("configs/manifests/batadal-channel-manifest-v1.json")
TCN_CHECKPOINT_PATH = Path("results/surrogates/vista_tcn_48h_150ep/tcn.pt")

BATADAL_CHANNELS = (
    "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7", "F_PU1",
    "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4",
    "F_PU5", "S_PU5", "F_PU6", "S_PU6", "F_PU7", "S_PU7", "F_PU8",
    "S_PU8", "F_PU9", "S_PU9", "F_PU10", "S_PU10", "F_PU11", "S_PU11",
    "F_V2", "S_V2", "P_J280", "P_J269", "P_J300", "P_J256", "P_J289",
    "P_J415", "P_J302", "P_J306", "P_J307", "P_J317", "P_J14", "P_J422",
)


def _synthetic_df(n_rows: int = 480, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    cycle = np.sin(2 * np.pi * t / 24)
    data: dict[str, np.ndarray] = {
        "L_T1": 2.5 + 1.2 * cycle + 0.05 * rng.standard_normal(n_rows)
    }
    for ch in BATADAL_CHANNELS:
        if ch == "L_T1":
            continue
        if ch.startswith("S_"):
            data[ch] = (cycle > 0).astype(float)
        elif ch.startswith("F_"):
            data[ch] = np.maximum(0.0, 100.0 * cycle + 10.0 * rng.standard_normal(n_rows))
        else:
            data[ch] = 3.0 + 0.8 * cycle + 0.1 * rng.standard_normal(n_rows)
    return pd.DataFrame(data, columns=BATADAL_CHANNELS)


def test_evaluate_inverse_benchmark_signature_and_default() -> None:
    """evaluate_inverse_benchmark accepts declared_support: bool = True as default."""
    sig = inspect.signature(evaluate_inverse_benchmark)
    assert "declared_support" in sig.parameters, "declared_support parameter missing"
    param = sig.parameters["declared_support"]
    assert param.default is True, f"declared_support default must be True, got {param.default}"


def test_evaluate_inverse_benchmark_invokes_reconstruct_with_interval_mapping() -> None:
    """When declared_support=False, reconstruct() must be invoked with attack_interval=None."""
    fit_df = _synthetic_df(n_rows=480, seed=1)
    calib_df = _synthetic_df(n_rows=480, seed=2)
    dev_df = _synthetic_df(n_rows=480, seed=3)
    train_std = float(fit_df["L_T1"].std())

    scenario = BenchmarkScenario(
        scenario_id="mock_scenario",
        target_channel="L_T1",
        amplitude_train_std=1.0,
        sign=1,
        duration_hours=24,
        onset_index=50,
        offset_index=74,
        regime="low_dynamics",
        injected_amplitude=train_std,
    )

    # Mock estimator recording reconstruct calls
    mock_est = MagicMock(spec=BaseDeltaEstimator)
    n = len(dev_df)
    mock_est.reconstruct.return_value = ReconstructionResult(
        x_hat=dev_df["L_T1"].to_numpy(),
        delta_hat=np.zeros(n),
        pointwise_uncertainty=np.zeros(n),
        pointwise_variance=np.zeros(n),
        objective_loss=0.0,
        ambiguity_flag=False,
    )

    # 1. Declared support: True -> attack_interval must be passed
    evaluate_inverse_benchmark(
        scenarios=[scenario],
        fit_df=fit_df,
        calib_df=calib_df,
        dev_df=dev_df,
        target_channel="L_T1",
        channel_manifest_path=CHANNEL_MANIFEST_PATH,
        train_std=train_std,
        declared_support=True,
        estimators={"MockEstimator": mock_est},
    )
    assert mock_est.reconstruct.called
    call_kwargs_declared = mock_est.reconstruct.call_args[1]
    assert call_kwargs_declared.get("attack_interval") == (50, 74)

    mock_est.reconstruct.reset_mock()

    # 2. Declared support: False -> attack_interval must be None
    evaluate_inverse_benchmark(
        scenarios=[scenario],
        fit_df=fit_df,
        calib_df=calib_df,
        dev_df=dev_df,
        target_channel="L_T1",
        channel_manifest_path=CHANNEL_MANIFEST_PATH,
        train_std=train_std,
        declared_support=False,
        estimators={"MockEstimator": mock_est},
    )
    assert mock_est.reconstruct.called
    call_kwargs_undeclared = mock_est.reconstruct.call_args[1]
    assert call_kwargs_undeclared.get("attack_interval") is None


def test_step_map_negative_control_clean_series() -> None:
    """Negative control on unattacked normal series: delta_hat approx 0 and 0 false positive support points."""
    T = 96
    t = np.arange(T)
    surrogate_x = 2.5 + 0.8 * np.sin(2 * np.pi * t / 24)
    # Observed matches surrogate with negligible sub-threshold noise
    rng = np.random.default_rng(1234)
    noise = rng.normal(0, 0.01, T)
    observed = surrogate_x + noise

    est = StepMAPEstimator(
        lambda_tv=1.0,
        lambda_l1=0.05,
        bounds=(0.0, 5.0),
        solver="admm",
    )

    res = est.reconstruct(observed, surrogate_x=surrogate_x, attack_interval=None)

    # Pointwise delta_hat must be tightly bounded around 0
    np.testing.assert_allclose(res.delta_hat, 0.0, atol=1e-4)

    # Support threshold at 10% of surrogate dynamic range (~0.16)
    threshold = 0.1 * float(np.std(surrogate_x))
    fp_points = int(np.sum(np.abs(res.delta_hat) > threshold))
    assert fp_points == 0, f"Expected 0 false positive support points, found {fp_points}"


def test_step_map_undeclared_step_recovery_validation_data() -> None:
    """On synthetic step attack (1.0 sigma, 24h) embedded in validation data, StepMAP achieves F1 >= 0.75."""
    val_path = Path("data/processed/batadal/val.parquet")
    train_path = Path("data/processed/batadal/train.parquet")
    if not val_path.exists() or not train_path.exists() or not TCN_CHECKPOINT_PATH.exists():
        pytest.skip("Benchmark data or TCN checkpoint not available on local path")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    train_std = float(train_df["L_T1"].std())

    # Use development injection slice (val rows 480:960)
    dev_df = val_df.iloc[480:960].copy()
    clean_x = dev_df["L_T1"].to_numpy(dtype=np.float64)
    n = len(clean_x)

    onset, offset = 100, 124
    injected_amplitude = 1.0 * train_std
    delta_true = np.zeros(n, dtype=np.float64)
    delta_true[onset:offset] = injected_amplitude

    attacked_df = dev_df.copy()
    attacked_df["L_T1"] = clean_x + delta_true

    tcn_surrogate = DilatedTCNSurrogate.load(TCN_CHECKPOINT_PATH)
    tcn_surrogate.device_str = "cpu"

    step_map = StepMAPEstimator(
        lambda_tv=1.0,
        lambda_l1=0.05,
        target_channel="L_T1",
        channel_manifest_path=CHANNEL_MANIFEST_PATH,
        surrogate_estimator=tcn_surrogate,
    )

    res = step_map.reconstruct(attacked_df, attack_interval=None)

    metrics = compute_scenario_metrics(
        delta_hat=res.delta_hat,
        delta_true=delta_true,
        x_hat=res.x_hat,
        x_true=clean_x,
        onset=onset,
        offset=offset,
        injected_amplitude=injected_amplitude,
        train_std=train_std,
    )

    # Acceptance assertions from milestone requirements:
    assert metrics["support_f1"] >= 0.75, f"Expected support F1 >= 0.75, got {metrics['support_f1']:.4f}"
    assert metrics["sign_accuracy"] == 1.0, f"Expected sign accuracy 1.0, got {metrics['sign_accuracy']}"
    assert metrics["delta_nrmse"] < 0.25, f"Expected bounded delta NRMSE < 0.25, got {metrics['delta_nrmse']:.4f}"
