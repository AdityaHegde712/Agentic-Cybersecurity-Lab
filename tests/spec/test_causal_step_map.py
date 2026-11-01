"""LOCKED specification and contract tests for CausalStepMAPEstimator.

Phase 3 Milestone: Past-Only Causal Rolling Filter.
Mathematical Contract:
Estimates the attack trajectory delta_t and clean state x_t strictly forward in time,
without conditioning on or observing future samples y_{t+1:N}.

Covers:
1. Interface Contract (BaseDeltaEstimator inheritance, ReconstructionResult validation).
2. Strict Causality Invariant (y_{t+1:N} perturbations produce zero change in delta_hat_{0:t}).
3. Clean Negative Control (zero false positive support points on unattacked series).
4. Step Recovery (bounded NRMSE and 100% sign accuracy on synthetic step attack).
5. Phase 3 Acceptance Gate (at least 20% lower NRMSE than PersistenceEstimator).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.inverse.estimators.base import BaseDeltaEstimator, ReconstructionResult
from src.inverse.estimators.causal_step_map import CausalStepMAPEstimator
from src.inverse.estimators.statistical import PersistenceEstimator


MANIFEST_PATH = Path("configs/manifests/batadal-channel-manifest-v1.json")
BATADAL_CHANNELS = (
    "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7", "F_PU1",
    "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4",
    "F_PU5", "S_PU5", "F_PU6", "S_PU6", "F_PU7", "S_PU7", "F_PU8",
    "S_PU8", "F_PU9", "S_PU9", "F_PU10", "S_PU10", "F_PU11", "S_PU11",
    "F_V2", "S_V2", "P_J280", "P_J269", "P_J300", "P_J256", "P_J289",
    "P_J415", "P_J302", "P_J306", "P_J307", "P_J317", "P_J14", "P_J422",
)


def _generate_synthetic_clean_data(n_rows: int = 240, seed: int = 42) -> pd.DataFrame:
    """Generate diurnal cycle companion and target sensor data."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    cycle = np.sin(2 * np.pi * t / 24)
    data: dict[str, np.ndarray] = {}
    data["L_T1"] = 2.5 + 1.2 * cycle + 0.05 * rng.standard_normal(n_rows)

    for ch in BATADAL_CHANNELS:
        if ch == "L_T1":
            continue
        if ch.startswith("S_"):
            data[ch] = (cycle > 0).astype(float)
        elif ch.startswith("F_"):
            data[ch] = np.maximum(0.0, 100.0 * cycle + 5.0 * rng.standard_normal(n_rows))
        else:
            data[ch] = 3.0 + 0.8 * cycle + 0.05 * rng.standard_normal(n_rows)

    return pd.DataFrame(data, columns=BATADAL_CHANNELS)


# =====================================================================
# Test 1: Interface Contract
# =====================================================================

def test_causal_step_map_interface_compliance() -> None:
    """Verify inheritance from BaseDeltaEstimator and ReconstructionResult compliance."""
    assert issubclass(CausalStepMAPEstimator, BaseDeltaEstimator)

    est = CausalStepMAPEstimator(
        window_size=24,
        lambda_tv=1.0,
        lambda_l1=0.1,
        target_channel="L_T1",
        bounds=(0.3, 4.9),
    )

    df = _generate_synthetic_clean_data(n_rows=48, seed=10)
    fitted_est = est.fit_normal(df)
    assert fitted_est is est

    res = est.reconstruct(df)
    assert isinstance(res, ReconstructionResult)
    assert len(res) == len(df)
    assert res.x_hat.shape == (len(df),)
    assert res.delta_hat.shape == (len(df),)
    assert res.pointwise_uncertainty is not None
    assert res.pointwise_variance is not None
    assert res.provenance["estimator"] == "CausalStepMAPEstimator"
    assert res.provenance["causal"] is True
    assert res.provenance["window_size"] == 24
    np.testing.assert_allclose(res.x_hat + res.delta_hat, df["L_T1"].to_numpy(), atol=1e-10)


# =====================================================================
# Test 2: Strict Causality Invariant (Zero Future Lookahead)
# =====================================================================

def test_causal_step_map_strict_causality_invariant() -> None:
    """Mathematical Contract: delta_hat_{0:t} must be perfectly invariant to y_{t+1:N}.
    
    Perturbing future observation samples while keeping past and current telemetry
    identical must yield the EXACT same estimate at all timesteps <= t.
    """
    n_rows = 72
    df_clean = _generate_synthetic_clean_data(n_rows=n_rows, seed=20)
    surrogate_clean = df_clean["L_T1"].to_numpy().copy()

    # Create baseline attacked sequence
    df_base = df_clean.copy()
    y_base = df_base["L_T1"].to_numpy().copy()
    y_base[20:50] += 1.5
    df_base["L_T1"] = y_base

    est = CausalStepMAPEstimator(
        window_size=24,
        lambda_tv=1.0,
        lambda_l1=0.1,
        bounds=(0.0, 5.0),
    )

    res_base = est.reconstruct(df_base, surrogate_x=surrogate_clean)

    # Test causality invariant at multiple intermediate evaluation timesteps t_eval
    test_eval_timesteps = [15, 25, 35, 45]
    for t_eval in test_eval_timesteps:
        df_perturbed = df_base.copy()
        y_perturbed = y_base.copy()

        # Severely corrupt future observations strictly after t_eval
        rng = np.random.default_rng(seed=100 + t_eval)
        future_distortion = rng.uniform(5.0, 15.0, size=n_rows - (t_eval + 1))
        y_perturbed[t_eval + 1 :] += future_distortion
        df_perturbed["L_T1"] = y_perturbed

        res_perturbed = est.reconstruct(df_perturbed, surrogate_x=surrogate_clean)

        # 1. Past and present estimates must be IDENTICAL within machine precision
        np.testing.assert_allclose(
            res_perturbed.delta_hat[: t_eval + 1],
            res_base.delta_hat[: t_eval + 1],
            atol=1e-12,
            err_msg=f"Causality violation detected at evaluation timestep t={t_eval}",
        )
        np.testing.assert_allclose(
            res_perturbed.x_hat[: t_eval + 1],
            res_base.x_hat[: t_eval + 1],
            atol=1e-12,
            err_msg=f"State reconstruction causality violation at evaluation timestep t={t_eval}",
        )

        # 2. Future estimates after t_eval must differ due to the future perturbation
        assert not np.allclose(
            res_perturbed.delta_hat[t_eval + 1 :],
            res_base.delta_hat[t_eval + 1 :],
        ), f"Expected future to reflect perturbation after t={t_eval}"


# =====================================================================
# Test 3: Clean Negative Control
# =====================================================================

def test_causal_step_map_clean_negative_control() -> None:
    """On unattacked normal series, delta_hat is ~0 with 0 false positive support points."""
    df_clean = _generate_synthetic_clean_data(n_rows=96, seed=30)
    surrogate_clean = df_clean["L_T1"].to_numpy().copy()

    est = CausalStepMAPEstimator(
        window_size=36,
        lambda_tv=1.0,
        lambda_l1=0.2,
        bounds=(0.0, 5.0),
    )

    res = est.reconstruct(df_clean, surrogate_x=surrogate_clean, attack_interval=None)

    # Max reconstruction error on delta should be negligible (< 0.05)
    np.testing.assert_allclose(res.delta_hat, 0.0, atol=0.05)

    # 0 false positive support points at detection threshold 0.1
    fp_points = np.sum(np.abs(res.delta_hat) > 0.1)
    assert fp_points == 0, f"Expected 0 false positive support points, found {fp_points}"


# =====================================================================
# Test 4: Step Recovery
# =====================================================================

def test_causal_step_map_step_recovery() -> None:
    """On synthetic step attack, accurately recover step with bounded NRMSE and 100% sign accuracy."""
    n_rows = 96
    df_clean = _generate_synthetic_clean_data(n_rows=n_rows, seed=40)
    surrogate_clean = df_clean["L_T1"].to_numpy().copy()

    # Inject 1.5m step attack between t=30 and t=66 (36 hours duration)
    onset, offset = 30, 66
    injected_amp = 1.5
    delta_true = np.zeros(n_rows, dtype=np.float64)
    delta_true[onset:offset] = injected_amp

    df_attacked = df_clean.copy()
    df_attacked["L_T1"] = df_clean["L_T1"] + delta_true

    est = CausalStepMAPEstimator(
        window_size=48,
        lambda_tv=1.0,
        lambda_l1=0.05,
        bounds=(0.0, 5.0),
    )

    res = est.reconstruct(df_attacked, surrogate_x=surrogate_clean, attack_interval=None)

    # Recovered delta on the plateau
    step_hat = res.delta_hat[onset + 2 : offset - 2]
    
    # 1. Sign accuracy on attack plateau must be 100%
    sign_correct = np.mean(np.sign(step_hat) == np.sign(injected_amp))
    assert sign_correct == 1.0, f"Expected 100% sign accuracy, got {sign_correct * 100:.1f}%"

    # 2. Bounded NRMSE on the active attack interval
    interval_rmse = np.sqrt(np.mean((res.delta_hat[onset:offset] - injected_amp) ** 2))
    interval_nrmse = interval_rmse / abs(injected_amp)
    assert interval_nrmse < 0.35, f"Interval NRMSE {interval_nrmse:.4f} exceeded 0.35"

    # 3. Outside attack interval, delta should be close to 0
    np.testing.assert_allclose(res.delta_hat[:onset], 0.0, atol=0.1)


# =====================================================================
# Test 5: Benchmark Gate: Beats PersistenceEstimator by >= 20%
# =====================================================================

def test_causal_step_map_beats_persistence_gate() -> None:
    """Phase 3 Acceptance Gate: CausalStepMAPEstimator independently beats Persistence by >= 20%."""
    n_rows = 96
    df_clean = _generate_synthetic_clean_data(n_rows=n_rows, seed=50)
    surrogate_clean = df_clean["L_T1"].to_numpy().copy()

    onset, offset = 24, 72
    injected_amp = 1.2
    delta_true = np.zeros(n_rows, dtype=np.float64)
    delta_true[onset:offset] = injected_amp

    df_attacked = df_clean.copy()
    df_attacked["L_T1"] = df_clean["L_T1"] + delta_true

    # 1. Persistence baseline
    p_est = PersistenceEstimator(lag=1).fit_normal(df_clean["L_T1"].to_numpy())
    p_res = p_est.reconstruct(
        df_attacked,
        attack_interval=(onset, offset),
        target_channel="L_T1",
        prior_clean_history=df_clean["L_T1"].to_numpy(),
    )
    p_rmse = np.sqrt(np.mean((p_res.delta_hat[onset:offset] - delta_true[onset:offset]) ** 2))
    p_nrmse = p_rmse / abs(injected_amp)

    # 2. CausalStepMAPEstimator
    c_est = CausalStepMAPEstimator(
        window_size=48,
        lambda_tv=1.0,
        lambda_l1=0.05,
        bounds=(0.0, 5.0),
    )
    c_res = c_est.reconstruct(
        df_attacked,
        surrogate_x=surrogate_clean,
        attack_interval=(onset, offset),
    )
    c_rmse = np.sqrt(np.mean((c_res.delta_hat[onset:offset] - delta_true[onset:offset]) ** 2))
    c_nrmse = c_rmse / abs(injected_amp)

    improvement = (p_nrmse - c_nrmse) / p_nrmse
    assert improvement >= 0.20, (
        f"Phase 3 Gate Failed: CausalStepMAP improvement was {improvement * 100:.2f}%, "
        f"required >= 20.00% (Causal NRMSE: {c_nrmse:.4f}, Persistence NRMSE: {p_nrmse:.4f})"
    )
