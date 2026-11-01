"""LOCKED specification and contract tests for MultiOrderTrendFilter estimators.

Sprint 4B: Attack Family Generalization via Multi-Order Trend Filtering.
Formulation:
    min_delta  0.5 / sigma_eps^2 * ||y - x_hat - delta||_2^2
               + lambda_0 * ||delta||_1
               + lambda_1 ||D^1 delta||_1
               + lambda_2 ||D^2 delta||_1
subject to physical bounds:
    x_min <= y - delta <= x_max

Covers:
1. Interface Compliance: BaseDeltaEstimator inheritance and typed ReconstructionResult.
2. ADMM Banded Cholesky Solver Correctness: Agreement with SciPy numerical baseline.
3. Ramp and Linear Drift Superiority: lambda_2 > 0 achieves >= 15% NRMSE reduction over pure TV (lambda_2 = 0).
4. Step Recovery: High support F1 (>= 0.80), 100% sign accuracy, and bounded NRMSE.
5. Strict Causality Invariance: CausalMultiOrderTrendFilterEstimator has zero future lookahead.
6. Clean Negative Control: Temporal sparsity P(delta_hat = 0 | delta = 0) > 95% on unattacked series.
7. Physical Box Projection: Zero violations of hydraulic tank limits [x_min, x_max].
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.inverse.estimators.base import BaseDeltaEstimator, ReconstructionResult
from src.inverse.estimators.multi_order_trend_filter import (
    CausalMultiOrderTrendFilterEstimator,
    MultiOrderTrendFilterEstimator,
)


BATADAL_CHANNELS = (
    "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7", "F_PU1",
    "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4",
    "F_PU5", "S_PU5", "F_PU6", "S_PU6", "F_PU7", "S_PU7", "F_PU8",
    "S_PU8", "F_PU9", "S_PU9", "F_PU10", "S_PU10", "F_PU11", "S_PU11",
    "F_V2", "S_V2", "P_J280", "P_J269", "P_J300", "P_J256", "P_J289",
    "P_J415", "P_J302", "P_J306", "P_J307", "P_J317", "P_J14", "P_J422",
)


def _generate_synthetic_clean_data(n_rows: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate diurnal cycle companion and target sensor data for tank L_T1."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    cycle = np.sin(2 * np.pi * t / 24)
    data: dict[str, np.ndarray] = {}
    data["L_T1"] = 2.5 + 1.2 * cycle + 0.04 * rng.standard_normal(n_rows)

    for ch in BATADAL_CHANNELS:
        if ch == "L_T1":
            continue
        if ch.startswith("S_"):
            data[ch] = (cycle > 0).astype(float)
        elif ch.startswith("F_"):
            data[ch] = np.maximum(0.0, 100.0 * cycle + 5.0 * rng.standard_normal(n_rows))
        else:
            data[ch] = 3.0 + 0.8 * cycle + 0.04 * rng.standard_normal(n_rows)

    return pd.DataFrame(data, columns=BATADAL_CHANNELS)


# =====================================================================
# Test 1: Interface Compliance
# =====================================================================

def test_interface_compliance() -> None:
    """Verify inheritance from BaseDeltaEstimator and ReconstructionResult compliance."""
    assert issubclass(MultiOrderTrendFilterEstimator, BaseDeltaEstimator)
    assert issubclass(CausalMultiOrderTrendFilterEstimator, BaseDeltaEstimator)

    est_batch = MultiOrderTrendFilterEstimator(
        lambda_0=0.05,
        lambda_1=0.50,
        lambda_2=1.00,
        target_channel="L_T1",
        bounds=(0.3, 4.9),
    )
    est_causal = CausalMultiOrderTrendFilterEstimator(
        window_size=24,
        lambda_0=0.05,
        lambda_1=0.50,
        lambda_2=1.00,
        target_channel="L_T1",
        bounds=(0.3, 4.9),
    )

    df = _generate_synthetic_clean_data(n_rows=48, seed=11)
    for est in [est_batch, est_causal]:
        fitted = est.fit_normal(df)
        assert fitted is est
        res = est.reconstruct(df)
        assert isinstance(res, ReconstructionResult)
        assert len(res) == len(df)
        assert res.x_hat.shape == (len(df),)
        assert res.delta_hat.shape == (len(df),)
        assert res.pointwise_uncertainty is not None
        assert res.pointwise_variance is not None
        assert "estimator" in res.provenance
        np.testing.assert_allclose(res.x_hat + res.delta_hat, df["L_T1"].to_numpy(), atol=1e-10)


# =====================================================================
# Test 2: ADMM Banded Cholesky Solver Correctness
# =====================================================================

def test_admm_banded_cholesky_solver_correctness() -> None:
    """Verify ADMM banded Cholesky solver converges to global MAP objective."""
    n_rows = 50
    rng = np.random.default_rng(123)
    t = np.linspace(0, 10, n_rows)
    x_clean = 2.5 + 0.8 * np.sin(t)
    delta_true = np.zeros(n_rows)
    delta_true[15:35] = np.linspace(0.0, 1.5, 20)
    y = x_clean + delta_true + 0.03 * rng.standard_normal(n_rows)

    est_admm = MultiOrderTrendFilterEstimator(
        lambda_0=0.05,
        lambda_1=0.50,
        lambda_2=1.00,
        bounds=(0.0, 5.0),
        solver="admm",
        max_iter=1500,
        tol=1e-6,
    )
    est_scipy = MultiOrderTrendFilterEstimator(
        lambda_0=0.05,
        lambda_1=0.50,
        lambda_2=1.00,
        bounds=(0.0, 5.0),
        solver="scipy",
        max_iter=500,
        tol=1e-6,
    )

    res_admm = est_admm.reconstruct(y, surrogate_x=x_clean)
    res_scipy = est_scipy.reconstruct(y, surrogate_x=x_clean)

    # Both solvers should yield closely matching objectives and bounded delta difference
    assert abs(res_admm.objective_loss - res_scipy.objective_loss) / max(1.0, res_scipy.objective_loss) < 0.05
    np.testing.assert_allclose(res_admm.delta_hat, res_scipy.delta_hat, atol=0.08)


# =====================================================================
# Test 3: Ramp and Linear Drift Superiority (>= 15% NRMSE Reduction)
# =====================================================================

def test_ramp_and_drift_reconstruction_outperforms_pure_tv() -> None:
    """Mathematical Contract: Second-order filtering (lambda_2 > 0) achieves >= 15% NRMSE

    reduction over pure first-order TV (lambda_2 = 0) on sustained linear drifts and ramps.
    """
    n_rows = 80
    clean_df = _generate_synthetic_clean_data(n_rows=n_rows, seed=22)
    x_clean = clean_df["L_T1"].to_numpy().copy()

    # Sustained linear drift attack starting at onset=20
    onset = 20
    delta_drift = np.zeros(n_rows, dtype=np.float64)
    delta_drift[onset:] = 0.03 * (np.arange(onset, n_rows) - onset)

    rng = np.random.default_rng(42)
    y_drift = x_clean + delta_drift + 0.05 * rng.standard_normal(n_rows)

    # Pure first-order TV (lambda_2 = 0)
    est_tv = MultiOrderTrendFilterEstimator(
        lambda_0=0.01,
        lambda_1=0.80,
        lambda_2=0.00,
        bounds=(0.0, 6.0),
    )
    res_tv = est_tv.reconstruct(y_drift, surrogate_x=x_clean)

    # Multi-order trend filter (lambda_2 > 0)
    est_tf = MultiOrderTrendFilterEstimator(
        lambda_0=0.01,
        lambda_1=0.10,
        lambda_2=2.00,
        bounds=(0.0, 6.0),
    )
    res_tf = est_tf.reconstruct(y_drift, surrogate_x=x_clean)

    drift_range = float(np.max(delta_drift[onset:]) - np.min(delta_drift[onset:]))
    assert drift_range > 0.0

    rmse_tv = float(np.sqrt(np.mean((res_tv.delta_hat[onset:] - delta_drift[onset:]) ** 2)))
    rmse_tf = float(np.sqrt(np.mean((res_tf.delta_hat[onset:] - delta_drift[onset:]) ** 2)))
    nrmse_tv = rmse_tv / drift_range
    nrmse_tf = rmse_tf / drift_range

    reduction = (nrmse_tv - nrmse_tf) / nrmse_tv
    assert reduction >= 0.15, (
        f"Second-order trend filtering failed >= 15% NRMSE improvement gate: "
        f"achieved {reduction * 100:.2f}% (TV NRMSE: {nrmse_tv:.4f}, TF NRMSE: {nrmse_tf:.4f})"
    )


# =====================================================================
# Test 4: Step Attack Reconstruction
# =====================================================================

def test_step_attack_reconstruction() -> None:
    """Verify that multi-order trend filtering preserves step reconstruction with lambda_1 > 0."""
    n_rows = 96
    clean_df = _generate_synthetic_clean_data(n_rows=n_rows, seed=33)
    x_clean = clean_df["L_T1"].to_numpy().copy()

    onset, offset = 30, 66
    injected_amp = 1.4
    delta_true = np.zeros(n_rows, dtype=np.float64)
    delta_true[onset:offset] = injected_amp

    rng = np.random.default_rng(77)
    y_attacked = x_clean + delta_true + 0.04 * rng.standard_normal(n_rows)

    est = MultiOrderTrendFilterEstimator(
        lambda_0=0.05,
        lambda_1=0.80,
        lambda_2=0.50,
        bounds=(0.0, 5.0),
    )
    res = est.reconstruct(y_attacked, surrogate_x=x_clean)

    # 1. Sign accuracy on the active attack interval must be 100%
    plateau = res.delta_hat[onset + 2 : offset - 2]
    sign_acc = float(np.mean(np.sign(plateau) == np.sign(injected_amp)))
    assert sign_acc == 1.0, f"Expected 100% sign accuracy, got {sign_acc * 100:.1f}%"

    # 2. Support F1 score (detection threshold 0.20 m)
    pred_support = np.abs(res.delta_hat) > 0.20
    true_support = np.abs(delta_true) > 0.20
    tp = np.sum(pred_support & true_support)
    fp = np.sum(pred_support & ~true_support)
    fn = np.sum(~pred_support & true_support)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    assert f1 >= 0.80, f"Expected Support F1 >= 0.80, got {f1:.4f}"

    # 3. Active interval NRMSE bounded
    interval_rmse = float(np.sqrt(np.mean((res.delta_hat[onset:offset] - injected_amp) ** 2)))
    interval_nrmse = interval_rmse / abs(injected_amp)
    assert interval_nrmse < 0.35, f"Expected NRMSE < 0.35, got {interval_nrmse:.4f}"


# =====================================================================
# Test 5: Strictly Causal Rolling Filter Zero Lookahead Invariant
# =====================================================================

def test_causal_multi_order_strict_causality_invariant() -> None:
    """Mathematical Contract: delta_hat_{0:t} must be perfectly invariant to future observations y_{t+1:N}."""
    n_rows = 72
    clean_df = _generate_synthetic_clean_data(n_rows=n_rows, seed=44)
    x_clean = clean_df["L_T1"].to_numpy().copy()

    # Base attacked trajectory
    y_base = x_clean.copy()
    y_base[20:50] += 1.2

    est = CausalMultiOrderTrendFilterEstimator(
        window_size=24,
        lambda_0=0.05,
        lambda_1=0.50,
        lambda_2=1.00,
        bounds=(0.0, 5.0),
    )

    res_base = est.reconstruct(y_base, surrogate_x=x_clean)

    # Test perturbation invariance across multiple intermediate evaluation points
    for t_eval in [15, 25, 35, 45]:
        y_perturbed = y_base.copy()
        rng = np.random.default_rng(200 + t_eval)
        y_perturbed[t_eval + 1 :] += rng.uniform(4.0, 10.0, size=n_rows - (t_eval + 1))

        res_perturbed = est.reconstruct(y_perturbed, surrogate_x=x_clean)

        # Invariance for all timesteps <= t_eval
        np.testing.assert_allclose(
            res_perturbed.delta_hat[: t_eval + 1],
            res_base.delta_hat[: t_eval + 1],
            atol=1e-10,
            err_msg=f"Causality violation at evaluation timestep t={t_eval}",
        )
        np.testing.assert_allclose(
            res_perturbed.x_hat[: t_eval + 1],
            res_base.x_hat[: t_eval + 1],
            atol=1e-10,
            err_msg=f"State causality violation at evaluation timestep t={t_eval}",
        )

        # Future timesteps must reflect perturbation
        assert not np.allclose(
            res_perturbed.delta_hat[t_eval + 1 :],
            res_base.delta_hat[t_eval + 1 :],
        )


# =====================================================================
# Test 6: Clean Negative Control
# =====================================================================

def test_negative_control_clean_sparsity() -> None:
    """On unattacked telemetry, delta_hat is sparse with P(delta_hat = 0 | delta = 0) > 95%."""
    n_rows = 96
    clean_df = _generate_synthetic_clean_data(n_rows=n_rows, seed=55)
    x_clean = clean_df["L_T1"].to_numpy().copy()
    y_clean = x_clean.copy()

    est = CausalMultiOrderTrendFilterEstimator(
        window_size=36,
        lambda_0=0.10,
        lambda_1=0.50,
        lambda_2=1.00,
        bounds=(0.0, 5.0),
    )

    res = est.reconstruct(y_clean, surrogate_x=x_clean)

    # Negligible error
    np.testing.assert_allclose(res.delta_hat, 0.0, atol=0.08)

    # Sparsity check: fraction of points with |delta_hat| < 0.05 must be > 95%
    sparsity = float(np.mean(np.abs(res.delta_hat) < 0.05))
    assert sparsity > 0.95, f"Expected clean sparsity > 0.95, got {sparsity:.4f}"


# =====================================================================
# Test 7: Hard Physical Box Constraint Projection
# =====================================================================

def test_hard_physical_constraint_projection() -> None:
    """Verify zero physical box-constraint violations under severe spoofing."""
    n_rows = 48
    clean_df = _generate_synthetic_clean_data(n_rows=n_rows, seed=66)
    x_clean = clean_df["L_T1"].to_numpy().copy()

    # Extreme attack spoofing levels far outside [0.5, 4.5]
    y_corrupted = x_clean.copy()
    y_corrupted[10:25] = -3.0  # Physically impossible negative SCADA reading
    y_corrupted[30:45] = 12.0  # Above physical tank overflow limit

    box_min, box_max = 0.5, 4.5

    est_batch = MultiOrderTrendFilterEstimator(
        lambda_0=0.05,
        lambda_1=0.50,
        lambda_2=1.00,
        bounds=(box_min, box_max),
    )
    est_causal = CausalMultiOrderTrendFilterEstimator(
        window_size=24,
        lambda_0=0.05,
        lambda_1=0.50,
        lambda_2=1.00,
        bounds=(box_min, box_max),
    )

    for est in [est_batch, est_causal]:
        res = est.reconstruct(y_corrupted, surrogate_x=x_clean)
        # Check x_hat satisfies bounds
        assert np.all(res.x_hat >= box_min - 1e-8), f"Lower bound violation in {est.__class__.__name__}"
        assert np.all(res.x_hat <= box_max + 1e-8), f"Upper bound violation in {est.__class__.__name__}"
