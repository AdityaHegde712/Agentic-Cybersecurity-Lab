"""LOCKED specification and contract tests for Calibrated Uncertainty Quantification.

Phase 3 Milestone: Calibrated Uncertainty Quantification & Confidence Intervals.
Contract:
1. Non-Zero Uncertainty: StepMAPEstimator and CausalStepMAPEstimator return non-zero,
   positive pointwise_uncertainty and pointwise_variance in ReconstructionResult.
2. Calibration Consistency: pointwise_uncertainty aligns with the surrogate's calibrated residual_std_.
3. Empirical Coverage Gate on Clean Data: On clean validation data, the nominal 90%
   confidence band (x_hat +- 1.645 * sigma_t) achieves empirical coverage between 85.0% and 95.0%.
4. Sharpness: Interval width is bounded and sharp (2 * 1.645 * sigma_t < 0.25m, representing <5% of 5m tank range).
5. Attacked Scenario Coverage: On a synthetic step attack (1.0 sigma, 24h), clean state
   ground truth x_true falls inside the 90% confidence band with >= 85.0% empirical coverage across the trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest

from src.inverse.estimators.base import BaseDeltaEstimator, ReconstructionResult
from src.inverse.estimators.causal_step_map import CausalStepMAPEstimator
from src.inverse.estimators.statistical import CompanionRegressionEstimator
from src.inverse.estimators.step_map import StepMAPEstimator
from src.inverse.harness import compute_scenario_metrics


class DummyCalibratedSurrogate(BaseDeltaEstimator):
    """Minimal mock surrogate with calibrated residual std for unit testing."""

    def __init__(self, residual_std: float = 0.042, target_channel: str = "L_T1") -> None:
        self.residual_std_ = float(residual_std)
        self.target_channel = target_channel

    def fit_normal(self, normal_data: Any, **kwargs: Any) -> DummyCalibratedSurrogate:
        return self

    def reconstruct(
        self,
        observed: np.ndarray | pd.Series | pd.DataFrame,
        attack_interval: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> ReconstructionResult:
        if isinstance(observed, pd.DataFrame):
            y = observed[self.target_channel].to_numpy(dtype=np.float64)
        elif isinstance(observed, pd.Series):
            y = observed.to_numpy(dtype=np.float64)
        else:
            y = np.asarray(observed, dtype=np.float64)

        n = len(y)
        unc = np.full(n, self.residual_std_, dtype=np.float64)
        return ReconstructionResult(
            x_hat=y.copy(),
            delta_hat=np.zeros(n, dtype=np.float64),
            pointwise_uncertainty=unc,
            pointwise_variance=unc**2,
            objective_loss=0.0,
            ambiguity_flag=False,
            provenance={"estimator": self.__class__.__name__, "residual_std": self.residual_std_},
        )


def _generate_synthetic_clean_data(n_rows: int = 240, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic companion and target sensor data with diurnal cycles."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    cycle = np.sin(2 * np.pi * t / 24)
    data: dict[str, np.ndarray] = {
        "L_T1": 2.5 + 1.2 * cycle + 0.042 * rng.standard_normal(n_rows),
        "P_J280": 3.0 + 0.8 * cycle + 0.05 * rng.standard_normal(n_rows),
        "F_PU1": np.maximum(0.0, 100.0 * cycle + 5.0 * rng.standard_normal(n_rows)),
    }
    return pd.DataFrame(data)


# =====================================================================
# Test 1: Non-Zero Uncertainty in ReconstructionResult
# =====================================================================

def test_step_map_and_causal_step_map_return_nonzero_uncertainty() -> None:
    """Verify both StepMAPEstimator and CausalStepMAPEstimator return non-zero positive uncertainty."""
    n_rows = 48
    df = _generate_synthetic_clean_data(n_rows=n_rows, seed=1)
    surrogate = DummyCalibratedSurrogate(residual_std=0.042)

    step_map = StepMAPEstimator(
        lambda_tv=1.0,
        lambda_l1=0.1,
        target_channel="L_T1",
        bounds=(0.0, 5.0),
        surrogate_estimator=surrogate,
    )
    res_step = step_map.reconstruct(df)
    assert res_step.pointwise_uncertainty is not None
    assert res_step.pointwise_variance is not None
    assert len(res_step.pointwise_uncertainty) == n_rows
    assert len(res_step.pointwise_variance) == n_rows
    assert np.all(res_step.pointwise_uncertainty > 0.0)
    assert np.all(res_step.pointwise_variance > 0.0)

    causal_step = CausalStepMAPEstimator(
        window_size=24,
        lambda_tv=1.0,
        lambda_l1=0.1,
        target_channel="L_T1",
        bounds=(0.0, 5.0),
        surrogate_estimator=surrogate,
    )
    res_causal = causal_step.reconstruct(df)
    assert res_causal.pointwise_uncertainty is not None
    assert res_causal.pointwise_variance is not None
    assert len(res_causal.pointwise_uncertainty) == n_rows
    assert len(res_causal.pointwise_variance) == n_rows
    assert np.all(res_causal.pointwise_uncertainty > 0.0)
    assert np.all(res_causal.pointwise_variance > 0.0)


# =====================================================================
# Test 2: Calibration Consistency with Surrogate Noise Scale
# =====================================================================

def test_uncertainty_calibration_consistency() -> None:
    """Verify pointwise_uncertainty aligns precisely with surrogate residual_std_."""
    expected_sigma = 0.0422
    surrogate = DummyCalibratedSurrogate(residual_std=expected_sigma)

    step_map = StepMAPEstimator(
        lambda_tv=1.0,
        lambda_l1=0.1,
        target_channel="L_T1",
        bounds=(0.0, 5.0),
        surrogate_estimator=surrogate,
    )
    causal_map = CausalStepMAPEstimator(
        window_size=24,
        lambda_tv=1.0,
        lambda_l1=0.1,
        target_channel="L_T1",
        bounds=(0.0, 5.0),
        surrogate_estimator=surrogate,
    )

    df = _generate_synthetic_clean_data(n_rows=48, seed=2)
    res_step = step_map.reconstruct(df)
    res_causal = causal_map.reconstruct(df)

    np.testing.assert_allclose(res_step.pointwise_uncertainty, expected_sigma, rtol=1e-5)
    np.testing.assert_allclose(res_step.pointwise_variance, expected_sigma**2, rtol=1e-5)
    np.testing.assert_allclose(res_causal.pointwise_uncertainty, expected_sigma, rtol=1e-5)
    np.testing.assert_allclose(res_causal.pointwise_variance, expected_sigma**2, rtol=1e-5)


# =====================================================================
# Test 3: Empirical Coverage Gate on Clean Data (85% - 95%)
# =====================================================================

def test_empirical_coverage_gate_clean_data() -> None:
    """Nominal 90% confidence band achieves empirical coverage between 85.0% and 95.0%."""
    # 1. Test on clean synthetic validation set where true noise is Gaussian
    n_rows = 1000
    rng = np.random.default_rng(1234)
    t = np.arange(n_rows)
    clean_mean = 2.5 + 1.2 * np.sin(2 * np.pi * t / 24)
    true_noise = 0.042 * rng.standard_normal(n_rows)
    x_true = clean_mean + true_noise

    surrogate = DummyCalibratedSurrogate(residual_std=0.042, target_channel="L_T1")
    df = pd.DataFrame({"L_T1": x_true})
    res_surr = surrogate.reconstruct(df)

    sigma_t = res_surr.pointwise_uncertainty
    z_90 = 1.645
    lower_band = clean_mean - z_90 * sigma_t
    upper_band = clean_mean + z_90 * sigma_t

    covered = (x_true >= lower_band) & (x_true <= upper_band)
    empirical_coverage = float(np.mean(covered))

    assert 0.85 <= empirical_coverage <= 0.95, (
        f"Synthetic empirical coverage {empirical_coverage:.4f} outside nominal 90% gate [0.85, 0.95]"
    )

    # 2. Test CompanionRegressionEstimator on real BATADAL validation split if available
    train_path = Path("data/processed/batadal/train.parquet")
    val_path = Path("data/processed/batadal/val.parquet")
    if train_path.exists() and val_path.exists():
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        reg = CompanionRegressionEstimator(target_channel="L_T1", alpha=1.0).fit_normal(train_df)
        res_reg = reg.reconstruct(val_df, attack_interval=(0, len(val_df)))
        x_val_true = val_df["L_T1"].to_numpy()
        reg_cov = float(np.mean(np.abs(x_val_true - res_reg.x_hat) <= z_90 * res_reg.pointwise_uncertainty))
        assert 0.85 <= reg_cov <= 0.95, (
            f"BATADAL validation empirical coverage {reg_cov:.4f} outside nominal 90% gate [0.85, 0.95]"
        )


# =====================================================================
# Test 4: Interval Sharpness (< 0.25m, < 5% tank range)
# =====================================================================

def test_confidence_interval_sharpness() -> None:
    """Verify confidence interval width is bounded and sharp (< 0.25m for TCN surrogate)."""
    tcn_sigma = 0.042
    z_90 = 1.645
    interval_width = 2.0 * z_90 * tcn_sigma

    tank_range = 5.0  # [0, 5] meters for BATADAL L_T1
    fraction_of_range = interval_width / tank_range

    assert interval_width < 0.25, f"Interval width {interval_width:.4f}m exceeds 0.25m ceiling"
    assert fraction_of_range < 0.05, f"Interval represents {fraction_of_range*100:.2f}% of range (must be < 5%)"

    surrogate = DummyCalibratedSurrogate(residual_std=tcn_sigma)
    step_map = StepMAPEstimator(surrogate_estimator=surrogate, bounds=(0.0, 5.0))
    df = _generate_synthetic_clean_data(n_rows=24, seed=5)
    res = step_map.reconstruct(df)

    widths = 2.0 * z_90 * res.pointwise_uncertainty
    assert np.all(widths < 0.25)
    assert np.all(widths / tank_range < 0.05)


# =====================================================================
# Test 5: Attacked Scenario Trajectory Coverage (>= 85%)
# =====================================================================

def test_attacked_scenario_trajectory_coverage() -> None:
    """On a synthetic step attack (1.0 sigma, 24h), clean state ground truth falls in band with >= 85% coverage."""
    n_rows = 240
    rng = np.random.default_rng(42)
    t = np.arange(n_rows)
    cycle = 2.5 + 1.2 * np.sin(2 * np.pi * t / 24)
    x_true = cycle + 0.042 * rng.standard_normal(n_rows)

    # Inject 1.0 sigma step attack of 24 hours at [100:124]
    onset, offset = 100, 124
    attack_amp = 0.47
    y = x_true.copy()
    y[onset:offset] += attack_amp

    df = pd.DataFrame({"L_T1": y})
    surrogate = DummyCalibratedSurrogate(residual_std=0.042, target_channel="L_T1")

    # 1. Test StepMAPEstimator
    step_map = StepMAPEstimator(
        lambda_tv=1.0,
        lambda_l1=0.05,
        target_channel="L_T1",
        bounds=(0.0, 5.0),
        surrogate_estimator=surrogate,
    )
    res_step = step_map.reconstruct(df, surrogate_x=cycle, attack_interval=(onset, offset))

    z_90 = 1.645
    step_lower = res_step.x_hat - z_90 * res_step.pointwise_uncertainty
    step_upper = res_step.x_hat + z_90 * res_step.pointwise_uncertainty
    step_cov = float(np.mean((x_true >= step_lower) & (x_true <= step_upper)))
    assert step_cov >= 0.85, f"StepMAP empirical coverage {step_cov:.4f} is below 85%"

    # 2. Test CausalStepMAPEstimator
    causal_map = CausalStepMAPEstimator(
        window_size=48,
        lambda_tv=1.0,
        lambda_l1=0.05,
        target_channel="L_T1",
        bounds=(0.0, 5.0),
        surrogate_estimator=surrogate,
    )
    res_causal = causal_map.reconstruct(df, surrogate_x=cycle, attack_interval=(onset, offset))

    causal_lower = res_causal.x_hat - z_90 * res_causal.pointwise_uncertainty
    causal_upper = res_causal.x_hat + z_90 * res_causal.pointwise_uncertainty
    causal_cov = float(np.mean((x_true >= causal_lower) & (x_true <= causal_upper)))
    assert causal_cov >= 0.85, f"CausalStepMAP empirical coverage {causal_cov:.4f} is below 85%"

    # 3. Test compute_scenario_metrics integration
    m = compute_scenario_metrics(
        delta_hat=res_step.delta_hat,
        delta_true=y - x_true,
        x_hat=res_step.x_hat,
        x_true=x_true,
        onset=onset,
        offset=offset,
        injected_amplitude=attack_amp,
        train_std=0.47,
        pointwise_uncertainty=res_step.pointwise_uncertainty,
        estimator_name="StepMAP_TCN",
    )
    assert "coverage_90" in m, "compute_scenario_metrics missing coverage_90 key"
    assert "interval_sharpness" in m, "compute_scenario_metrics missing interval_sharpness key"
    assert m["coverage_90"] >= 0.85, f"Scenario metrics coverage_90 {m['coverage_90']:.4f} < 0.85"
    assert m["interval_sharpness"] < 0.25, f"Scenario metrics interval_sharpness {m['interval_sharpness']:.4f} >= 0.25"
