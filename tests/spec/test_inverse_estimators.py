"""LOCKED specification and contract tests for the inverse estimator suite.

Covers:
- ReconstructionResult and BaseDeltaEstimator interface.
- Persistence / Seasonal Naive estimators.
- Companion Regression estimator (using declared-unaffected sensors from channel manifest).
- 1D Total-Variation / Fused Lasso (MAP) Step Estimator (with manifest bounds).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.inverse.estimators import (
    BaseDeltaEstimator,
    CompanionRegressionEstimator,
    PersistenceEstimator,
    ReconstructionResult,
    SeasonalNaiveEstimator,
    StepMAPEstimator,
)
from src.inverse.channel_constraints import ChannelConstraintManifest


MANIFEST_PATH = Path("configs/manifests/batadal-channel-manifest-v1.json")
BATADAL_CHANNELS = (
    "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7", "F_PU1",
    "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4",
    "F_PU5", "S_PU5", "F_PU6", "S_PU6", "F_PU7", "S_PU7", "F_PU8",
    "S_PU8", "F_PU9", "S_PU9", "F_PU10", "S_PU10", "F_PU11", "S_PU11",
    "F_V2", "S_V2", "P_J280", "P_J269", "P_J300", "P_J256", "P_J289",
    "P_J415", "P_J302", "P_J306", "P_J307", "P_J317", "P_J14", "P_J422",
)


def _synthetic_fit_data(n_rows: int = 5_840, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic training data matching the 43 BATADAL channels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    # Seasonal diurnal cycle (24h period)
    cycle = np.sin(2 * np.pi * t / 24)
    data: dict[str, np.ndarray] = {}
    
    # L_T1 base level around 2.5 with diurnal variation in [0.5, 4.5]
    data["L_T1"] = 2.5 + 1.2 * cycle + 0.1 * rng.standard_normal(n_rows)
    
    for ch in BATADAL_CHANNELS:
        if ch == "L_T1":
            continue
        if ch.startswith("S_"):
            # Binary pump status
            data[ch] = (cycle > 0).astype(float)
        elif ch.startswith("F_"):
            # Pump flow correlated with cycle
            data[ch] = np.maximum(0.0, 100.0 * cycle + 10.0 * rng.standard_normal(n_rows))
        else:
            # Tank levels and junction pressures
            data[ch] = 3.0 + 0.8 * cycle + 0.15 * rng.standard_normal(n_rows)
            
    return pd.DataFrame(data, columns=BATADAL_CHANNELS)


# =====================================================================
# 1. ReconstructionResult and BaseDeltaEstimator Interface Tests
# =====================================================================

def test_reconstruction_result_valid_fields_and_properties() -> None:
    x_hat = np.array([1.0, 2.0, 3.0])
    delta_hat = np.array([0.0, 0.5, 0.0])
    uncertainty = np.array([0.1, 0.1, 0.1])
    variance = np.array([0.01, 0.01, 0.01])
    prov = {"estimator": "test", "iterations": 10}

    result = ReconstructionResult(
        x_hat=x_hat,
        delta_hat=delta_hat,
        pointwise_uncertainty=uncertainty,
        pointwise_variance=variance,
        objective_loss=0.05,
        ambiguity_flag=False,
        provenance=prov,
    )

    assert len(result) == 3
    np.testing.assert_allclose(result.reconstructed_y, np.array([1.0, 2.5, 3.0]))
    np.testing.assert_allclose(result.residual(np.array([1.0, 2.5, 3.0])), np.zeros(3), atol=1e-12)
    assert result.ambiguity_flag is False
    assert result.objective_loss == 0.05
    assert result.provenance["estimator"] == "test"


def test_reconstruction_result_auto_derives_uncertainty_or_variance() -> None:
    x_hat = np.array([1.0, 2.0])
    delta_hat = np.array([0.0, 0.0])
    variance = np.array([4.0, 9.0])

    # Supplying variance without uncertainty automatically derives uncertainty
    result = ReconstructionResult(
        x_hat=x_hat,
        delta_hat=delta_hat,
        pointwise_variance=variance,
    )
    np.testing.assert_allclose(result.pointwise_uncertainty, np.array([2.0, 3.0]))

    # Supplying uncertainty without variance automatically derives variance
    result2 = ReconstructionResult(
        x_hat=x_hat,
        delta_hat=delta_hat,
        pointwise_uncertainty=np.array([2.0, 3.0]),
    )
    np.testing.assert_allclose(result2.pointwise_variance, np.array([4.0, 9.0]))


def test_reconstruction_result_validates_dimensions_and_finite_values() -> None:
    x_hat = np.array([1.0, 2.0])
    
    # Delta length mismatch
    with pytest.raises(ValueError, match="delta_hat length"):
        ReconstructionResult(x_hat=x_hat, delta_hat=np.array([1.0]))

    # Uncertainty length mismatch
    with pytest.raises(ValueError, match="uncertainty length"):
        ReconstructionResult(
            x_hat=x_hat,
            delta_hat=np.array([0.0, 0.0]),
            pointwise_uncertainty=np.array([0.1]),
        )

    # Variance length mismatch
    with pytest.raises(ValueError, match="variance length"):
        ReconstructionResult(
            x_hat=x_hat,
            delta_hat=np.array([0.0, 0.0]),
            pointwise_variance=np.array([0.1, 0.1, 0.1]),
        )

    # Negative uncertainty
    with pytest.raises(ValueError, match="non-negative"):
        ReconstructionResult(
            x_hat=x_hat,
            delta_hat=np.array([0.0, 0.0]),
            pointwise_uncertainty=np.array([-0.1, 0.1]),
        )

    # Non-finite values
    with pytest.raises(ValueError, match="finite"):
        ReconstructionResult(
            x_hat=np.array([1.0, np.nan]),
            delta_hat=np.array([0.0, 0.0]),
        )


def test_base_estimator_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseDeltaEstimator()  # type: ignore[abstract]


# =====================================================================
# 2. Persistence / Seasonal Naive Estimators Tests
# =====================================================================

def test_persistence_estimator_zero_attack_identity() -> None:
    clean_history = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    est = PersistenceEstimator(lag=1)
    est.fit_normal(clean_history)

    observed = np.array([3.0, 3.0, 3.0])
    res = est.reconstruct(observed)

    assert isinstance(res, ReconstructionResult)
    np.testing.assert_allclose(res.x_hat, np.array([3.0, 3.0, 3.0]))
    np.testing.assert_allclose(res.delta_hat, np.zeros(3))
    assert res.ambiguity_flag is False


def test_persistence_estimator_recovers_step_attack_with_declared_interval() -> None:
    clean = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    delta_true = np.array([0.0, 0.0, 1.5, 1.5, 0.0, 0.0])
    observed = clean + delta_true

    est = PersistenceEstimator(lag=1)
    est.fit_normal(clean)

    # With declared attack interval [2, 4)
    res = est.reconstruct(observed, attack_interval=(2, 4))
    np.testing.assert_allclose(res.delta_hat, delta_true, atol=1e-12)
    np.testing.assert_allclose(res.x_hat, clean, atol=1e-12)
    # Outside interval must be exactly zero
    assert res.delta_hat[0] == 0.0
    assert res.delta_hat[1] == 0.0
    assert res.delta_hat[4] == 0.0
    assert res.delta_hat[5] == 0.0


def test_seasonal_naive_estimator_periodic_recovery() -> None:
    t = np.arange(96)
    clean = 3.0 + 1.0 * np.sin(2 * np.pi * t / 24)
    
    est = SeasonalNaiveEstimator(period=24)
    est.fit_normal(clean[:48])

    # Attack on hours 48 to 72 with step magnitude 2.0
    delta_true = np.zeros(48)
    delta_true[12:36] = 2.0
    observed = clean[48:96] + delta_true

    res = est.reconstruct(observed, attack_interval=(12, 36), prior_clean_history=clean[24:48])
    np.testing.assert_allclose(res.delta_hat[12:36], 2.0, atol=1e-10)
    np.testing.assert_allclose(res.x_hat, clean[48:96], atol=1e-10)
    assert res.pointwise_uncertainty is not None
    assert len(res.pointwise_uncertainty) == 48


# =====================================================================
# 3. Companion Regression Estimator Tests
# =====================================================================

def test_companion_regression_extracts_declared_unaffected_channels_from_manifest() -> None:
    assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
    est = CompanionRegressionEstimator(
        target_channel="L_T1",
        channel_manifest_path=MANIFEST_PATH,
    )
    
    # Must declare exactly 42 unaffected companion sensors
    assert len(est.companion_channels) == 42
    assert "L_T1" not in est.companion_channels
    assert "L_T2" in est.companion_channels
    assert "F_PU1" in est.companion_channels
    assert "P_J280" in est.companion_channels


def test_companion_regression_fits_model_fit_split_and_reconstructs_step_attack() -> None:
    fit_df = _synthetic_fit_data(n_rows=5_840, seed=100)
    est = CompanionRegressionEstimator(
        target_channel="L_T1",
        channel_manifest_path=MANIFEST_PATH,
        alpha=1.0,
    )
    est.fit_normal(fit_df)

    assert est.is_fitted is True
    assert est.fit_r2 > 0.8
    assert est.residual_variance > 0.0

    # Evaluate on a 48-hour evaluation window
    eval_df = _synthetic_fit_data(n_rows=48, seed=200)
    delta_true = np.zeros(48)
    delta_true[12:36] = 1.5
    
    # Inject additive step attack on L_T1
    attacked_eval_df = eval_df.copy()
    attacked_eval_df["L_T1"] = eval_df["L_T1"] + delta_true

    res = est.reconstruct(attacked_eval_df, attack_interval=(12, 36))

    assert isinstance(res, ReconstructionResult)
    # Outside interval: delta_hat must be exactly zero
    np.testing.assert_allclose(res.delta_hat[:12], 0.0, atol=1e-12)
    np.testing.assert_allclose(res.delta_hat[36:], 0.0, atol=1e-12)

    # Inside interval: recovered delta should be close to true delta (within regression noise)
    delta_recovered_slice = res.delta_hat[12:36]
    assert np.median(np.abs(delta_recovered_slice - 1.5)) < 0.25
    assert res.pointwise_variance is not None
    assert np.all(res.pointwise_variance > 0)


def test_companion_regression_rejects_target_channel_in_companions() -> None:
    with pytest.raises(ValueError, match="Target channel cannot be a companion"):
        CompanionRegressionEstimator(
            target_channel="L_T1",
            companion_channels=("L_T1", "L_T2"),
        )


# =====================================================================
# 4. 1D Total-Variation / Fused Lasso (MAP) Step Estimator Tests
# =====================================================================

def test_step_map_recovers_step_attack_piecewise_constant() -> None:
    t = np.arange(48)
    clean_surrogate = 2.5 + 0.8 * np.sin(2 * np.pi * t / 24)
    # Actual clean signal has small noise around surrogate
    clean_true = clean_surrogate + 0.02 * np.sin(10 * t)
    
    delta_true = np.zeros(48)
    delta_true[15:35] = 1.8
    observed = clean_true + delta_true

    # L_T1 empirical bounds from manifest: [0.3201, 4.8713]
    est = StepMAPEstimator(
        lambda_tv=1.0,
        lambda_l1=0.1,
        target_channel="L_T1",
        bounds=(0.32011184096336365, 4.871329307556152),
    )

    res = est.reconstruct(
        observed=observed,
        surrogate_x=clean_surrogate,
        attack_interval=(15, 35),
    )

    assert isinstance(res, ReconstructionResult)
    # Outside attack interval: zero
    np.testing.assert_allclose(res.delta_hat[:15], 0.0, atol=1e-8)
    np.testing.assert_allclose(res.delta_hat[35:], 0.0, atol=1e-8)

    # Inside attack interval: recovered step close to 1.8 and piecewise constant
    step_hat = res.delta_hat[15:35]
    assert np.all(step_hat > 1.4)
    # First differences should be very small inside the plateau (total variation penalty)
    inner_diffs = np.diff(step_hat[1:-1])
    assert np.max(np.abs(inner_diffs)) < 0.05


def test_step_map_strictly_enforces_manifest_bounds() -> None:
    # y is large, surrogate is normal
    T = 20
    surrogate_x = np.full(T, 3.0)
    # Observed value is pushed to 6.0
    observed = np.full(T, 6.0)
    
    # Manifest bounds: max L_T1 is 4.8713.
    # Therefore, x = y - delta <= 4.8713 implies delta >= 6.0 - 4.8713 = 1.1287
    x_min, x_max = 0.3201, 4.8713
    est = StepMAPEstimator(
        lambda_tv=0.5,
        lambda_l1=0.05,
        bounds=(x_min, x_max),
    )

    res = est.reconstruct(
        observed=observed,
        surrogate_x=surrogate_x,
    )

    # Verify x_hat = observed - delta_hat satisfies bounds strictly
    assert np.all(res.x_hat >= x_min - 1e-10)
    assert np.all(res.x_hat <= x_max + 1e-10)


def test_step_map_deterministic_reproducibility() -> None:
    T = 30
    rng = np.random.default_rng(1234)
    surrogate_x = rng.uniform(1.0, 3.0, size=T)
    observed = surrogate_x + rng.uniform(-0.5, 0.5, size=T)
    observed[10:20] += 1.0

    est1 = StepMAPEstimator(lambda_tv=0.8, lambda_l1=0.2, bounds=(0.0, 5.0))
    est2 = StepMAPEstimator(lambda_tv=0.8, lambda_l1=0.2, bounds=(0.0, 5.0))

    res1 = est1.reconstruct(observed, surrogate_x=surrogate_x)
    res2 = est2.reconstruct(observed, surrogate_x=surrogate_x)

    np.testing.assert_array_equal(res1.x_hat, res2.x_hat)
    np.testing.assert_array_equal(res1.delta_hat, res2.delta_hat)
    assert res1.objective_loss == res2.objective_loss


def test_step_map_extreme_regularization_limits() -> None:
    T = 20
    surrogate_x = np.linspace(1.0, 3.0, T)
    observed = surrogate_x.copy()
    observed[5:15] += 2.0

    # 1. When lambda_l1 is very large, delta_hat must shrink to 0
    est_sparse = StepMAPEstimator(lambda_tv=0.0, lambda_l1=100.0, bounds=(-10.0, 10.0))
    res_sparse = est_sparse.reconstruct(observed, surrogate_x=surrogate_x)
    np.testing.assert_allclose(res_sparse.delta_hat, 0.0, atol=1e-5)
    np.testing.assert_allclose(res_sparse.x_hat, observed, atol=1e-5)

    # 2. When lambda_tv and lambda_l1 are 0, delta_hat is exact unregularized residual clipped to bounds
    est_unreg = StepMAPEstimator(lambda_tv=0.0, lambda_l1=0.0, bounds=(-10.0, 10.0))
    res_unreg = est_unreg.reconstruct(observed, surrogate_x=surrogate_x)
    np.testing.assert_allclose(res_unreg.delta_hat, observed - surrogate_x, atol=1e-5)
    np.testing.assert_allclose(res_unreg.x_hat, surrogate_x, atol=1e-5)


def test_step_map_scipy_and_admm_solvers_agree() -> None:
    T = 24
    t = np.arange(T)
    surrogate_x = 2.0 + 0.5 * np.cos(2 * np.pi * t / 24)
    observed = surrogate_x.copy()
    observed[8:16] += 1.2

    est_admm = StepMAPEstimator(lambda_tv=0.5, lambda_l1=0.1, bounds=(0.0, 5.0), solver="admm")
    est_scipy = StepMAPEstimator(lambda_tv=0.5, lambda_l1=0.1, bounds=(0.0, 5.0), solver="scipy")

    res_admm = est_admm.reconstruct(observed, surrogate_x=surrogate_x)
    res_scipy = est_scipy.reconstruct(observed, surrogate_x=surrogate_x)

    # Both should find very close objectives and delta reconstructions
    assert abs(res_admm.objective_loss - res_scipy.objective_loss) < 0.05
    np.testing.assert_allclose(res_admm.delta_hat, res_scipy.delta_hat, atol=0.05)


def test_end_to_end_pipeline_composition() -> None:
    """Compose CompanionRegressionEstimator as surrogate for StepMAPEstimator."""
    fit_df = _synthetic_fit_data(n_rows=5_840, seed=555)
    reg_est = CompanionRegressionEstimator(
        target_channel="L_T1",
        channel_manifest_path=MANIFEST_PATH,
    )
    reg_est.fit_normal(fit_df)

    eval_df = _synthetic_fit_data(n_rows=48, seed=777)
    delta_true = np.zeros(48)
    delta_true[10:30] = 2.0
    attacked_eval_df = eval_df.copy()
    attacked_eval_df["L_T1"] = eval_df["L_T1"] + delta_true

    map_est = StepMAPEstimator(
        lambda_tv=1.0,
        lambda_l1=0.05,
        target_channel="L_T1",
        channel_manifest_path=MANIFEST_PATH,
        surrogate_estimator=reg_est,
    )

    res = map_est.reconstruct(
        observed=attacked_eval_df,
        attack_interval=(10, 30),
    )

    assert isinstance(res, ReconstructionResult)
    # y == x_hat + delta_hat identity holds
    np.testing.assert_allclose(
        attacked_eval_df["L_T1"].to_numpy(),
        res.x_hat + res.delta_hat,
        atol=1e-10,
    )
    # Step recovered with median error < 0.2
    assert np.median(np.abs(res.delta_hat[10:30] - 2.0)) < 0.2
