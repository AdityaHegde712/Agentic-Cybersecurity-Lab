"""LOCKED specification tests for Delta Distribution Analysis and Error Modeling (Sprint 4A).

Verifies the mathematical and statistical contracts for:
1. Parametric fitting of Gaussian, Student-t, and Generalized Error Distribution (GED / gennorm) via MLE.
2. Model selection ranking via AIC and BIC information criteria.
3. Null clean distribution profiling (sparsity rate, tail envelope, empirical FAR).
4. Operational threshold curves (ROC, TPR/FPR trade-offs, optimal tau* lookup for target FARs).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.inverse.statistical_modeling import (
    DistributionFitResult,
    NullDistributionSummary,
    OperationalThresholdCurve,
    compute_operational_threshold_curves,
    evaluate_null_clean_distribution,
    fit_parametric_distributions,
    rank_models,
)


# =====================================================================
# Test 1: Parametric Fitting on Normal Data
# =====================================================================

def test_parametric_fitting_gaussian() -> None:
    """Verify Gaussian MLE parameter recovery and goodness-of-fit metrics."""
    rng = np.random.default_rng(42)
    mu_true = 0.45
    sigma_true = 1.25
    data = rng.normal(loc=mu_true, scale=sigma_true, size=1500)

    fits = fit_parametric_distributions(data)

    assert "gaussian" in fits
    assert "student_t" in fits
    assert "ged" in fits

    gauss_fit = fits["gaussian"]
    assert isinstance(gauss_fit, DistributionFitResult)
    assert gauss_fit.distribution_name == "gaussian"
    assert gauss_fit.n_params == 2
    assert gauss_fit.n_samples == 1500
    assert pytest.approx(gauss_fit.params["loc"], abs=0.08) == mu_true
    assert pytest.approx(gauss_fit.params["scale"], abs=0.08) == sigma_true

    # Log-likelihood, AIC, BIC sanity
    assert gauss_fit.log_likelihood < 0.0
    expected_aic = 2 * 2 - 2 * gauss_fit.log_likelihood
    assert pytest.approx(gauss_fit.aic, rel=1e-5) == expected_aic
    expected_bic = 2 * np.log(1500) - 2 * gauss_fit.log_likelihood
    assert pytest.approx(gauss_fit.bic, rel=1e-5) == expected_bic

    # KS test should not reject the true Gaussian hypothesis
    assert 0.0 <= gauss_fit.ks_stat <= 1.0
    assert gauss_fit.ks_pvalue > 0.05


# =====================================================================
# Test 2: Heavy-Tailed Fitting and Model Selection
# =====================================================================

def test_parametric_fitting_student_t_and_model_selection() -> None:
    """Verify Student-t heavy tail identification and AIC/BIC ranking superiority."""
    rng = np.random.default_rng(123)
    # Generate heavy-tailed t-distribution data (df=3.0)
    df_true = 3.0
    loc_true = 0.1
    scale_true = 0.8
    data = stats.t.rvs(df=df_true, loc=loc_true, scale=scale_true, size=2000, random_state=rng)

    fits = fit_parametric_distributions(data)
    t_fit = fits["student_t"]
    gauss_fit = fits["gaussian"]

    assert t_fit.distribution_name == "student_t"
    assert t_fit.n_params == 3
    assert 1.5 <= t_fit.params["df"] <= 5.0
    assert pytest.approx(t_fit.params["loc"], abs=0.1) == loc_true
    assert pytest.approx(t_fit.params["scale"], abs=0.15) == scale_true

    # Heavy tails should make Student-t have substantially superior (lower) AIC & BIC than Gaussian
    assert t_fit.aic < gauss_fit.aic - 20.0
    assert t_fit.bic < gauss_fit.bic - 20.0
    assert t_fit.ks_pvalue > 0.01

    ranked = rank_models(fits, criterion="aic")
    assert ranked[0].distribution_name in ("student_t", "ged")
    assert ranked[-1].distribution_name == "gaussian"


# =====================================================================
# Test 3: Generalized Error Distribution (GED / Laplace case)
# =====================================================================

def test_parametric_fitting_ged_laplace() -> None:
    """Verify GED captures Laplacian peakiness (beta ~ 1.0)."""
    rng = np.random.default_rng(999)
    # Laplace corresponds to GED with beta=1.0
    loc_true = 0.0
    scale_true = 0.5
    data = stats.laplace.rvs(loc=loc_true, scale=scale_true, size=2500, random_state=rng)

    fits = fit_parametric_distributions(data)
    ged_fit = fits["ged"]

    assert ged_fit.distribution_name == "ged"
    assert ged_fit.n_params == 3
    # Shape beta should be close to 1.0
    assert 0.7 <= ged_fit.params["beta"] <= 1.3
    assert pytest.approx(ged_fit.params["loc"], abs=0.08) == loc_true
    assert ged_fit.ks_pvalue > 0.01

    # Laplace data should penalize Gaussian (beta=2.0)
    assert ged_fit.aic < fits["gaussian"].aic - 50.0


# =====================================================================
# Test 4: Null Clean Distribution Profiling
# =====================================================================

def test_null_clean_distribution_profiling() -> None:
    """Verify null distribution sparsity rate and false-alarm profiling."""
    rng = np.random.default_rng(2026)
    n_samples = 2000

    # 97% exact or near-zeros (tolerance 1e-4), 3% small noise outliers
    delta_null = np.zeros(n_samples, dtype=np.float64)
    outlier_idx = rng.choice(n_samples, size=int(0.03 * n_samples), replace=False)
    delta_null[outlier_idx] = rng.uniform(0.05, 0.20, size=len(outlier_idx))

    summary = evaluate_null_clean_distribution(
        delta_null,
        estimator_name="TestEstimator",
        tolerance=1e-4,
        threshold_levels=[0.01, 0.05, 0.10, 0.15, 0.25],
    )

    assert isinstance(summary, NullDistributionSummary)
    assert summary.estimator_name == "TestEstimator"
    assert summary.n_points == n_samples
    assert summary.n_zero == int(0.97 * n_samples)
    assert pytest.approx(summary.sparsity_rate, abs=1e-4) == 0.97

    # Non-zero stats
    assert summary.mean_abs_nonzero > 0.0
    assert summary.max_abs >= 0.15

    # Empirical FAR should monotonically decrease with threshold
    fars = [summary.empirical_far_at_thresholds[t] for t in [0.01, 0.05, 0.10, 0.15, 0.25]]
    assert fars == sorted(fars, reverse=True)
    assert summary.empirical_far_at_thresholds[0.25] == 0.0

    # Percentiles dict check
    assert "p50" in summary.percentiles
    assert "p95" in summary.percentiles
    assert "p99" in summary.percentiles
    assert summary.percentiles["p50"] == 0.0


# =====================================================================
# Test 5: Operational Detection Threshold Curve
# =====================================================================

def test_operational_threshold_curves() -> None:
    """Verify ROC computation and optimal threshold tau* lookup for target FARs."""
    rng = np.random.default_rng(777)
    n = 3000

    # Null distribution: mostly zero, small noise std=0.03
    null_deltas = np.zeros(n, dtype=np.float64)
    active_null = rng.choice(n, size=150, replace=False)
    null_deltas[active_null] = rng.normal(0, 0.04, size=150)

    # Attack distribution: signal with mean 0.5, std 0.1
    attack_deltas = rng.normal(0.5, 0.1, size=n)

    target_fars = (0.001, 0.01, 0.05)
    curve = compute_operational_threshold_curves(
        null_deltas=null_deltas,
        attack_deltas=attack_deltas,
        target_fars=target_fars,
        n_thresholds=150,
    )

    assert isinstance(curve, OperationalThresholdCurve)
    assert len(curve.thresholds) == 150
    assert len(curve.fpr) == 150
    assert len(curve.tpr) == 150
    assert 0.90 <= curve.auc <= 1.0  # Well separated

    # Check tau* lookup for target FARs
    for far in target_fars:
        assert far in curve.tau_for_far
        tau_star = curve.tau_for_far[far]
        assert tau_star > 0.0
        # Verify empirical FPR at tau_star is <= far
        empirical_fpr = float(np.mean(np.abs(null_deltas) >= tau_star))
        assert empirical_fpr <= far + 1e-6

    # Smaller FAR requires equal or strictly higher threshold
    assert curve.tau_for_far[0.001] >= curve.tau_for_far[0.01] >= curve.tau_for_far[0.05]


# =====================================================================
# Test 6: Input Validation Contracts
# =====================================================================

def test_input_validation_contracts() -> None:
    """Verify robust error handling on degenerate inputs without silent fallbacks."""
    with pytest.raises(ValueError, match="data must contain at least 10 samples"):
        fit_parametric_distributions(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_parametric_distributions(np.array([1.0, np.nan, 3.0] * 5))

    with pytest.raises(ValueError, match="non-empty"):
        evaluate_null_clean_distribution(np.array([]))

    with pytest.raises(ValueError, match="target_fars must be strictly between 0 and 1"):
        compute_operational_threshold_curves(
            null_deltas=np.zeros(20),
            attack_deltas=np.ones(20),
            target_fars=(0.0, 1.5),
        )
