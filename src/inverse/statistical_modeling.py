"""Statistical Distribution and Error Modeling for Inverse Telemetry Recovery.

Provides mathematical foundations, MLE parametric fitting, null clean false-alarm
profiling, and operational detection threshold trade-off curves for reconstructed
sensor attack trajectories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy import integrate, stats


@dataclass(frozen=True)
class DistributionFitResult:
    """Container for parametric distribution MLE fit results and goodness-of-fit metrics.

    Attributes:
        distribution_name: Canonical identifier of distribution ('gaussian', 'student_t', 'ged').
        params: Dictionary of fitted distribution parameters (e.g., loc, scale, df, beta).
        log_likelihood: Maximized log-likelihood of empirical data under the fitted distribution.
        aic: Akaike Information Criterion (2k - 2*logL).
        bic: Bayesian Information Criterion (k*ln(n) - 2*logL).
        ks_stat: Two-sided Kolmogorov-Smirnov test statistic.
        ks_pvalue: Asymptotic p-value for Kolmogorov-Smirnov test.
        n_params: Number of free estimated parameters (k).
        n_samples: Number of data points used in the fit (n).
    """

    distribution_name: str
    params: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float
    n_params: int
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class NullDistributionSummary:
    """Summary of null clean telemetry reconstruction distribution p(delta_hat | delta = 0).

    Attributes:
        estimator_name: Identifier of the estimator evaluated.
        n_points: Total number of temporal evaluations under authentic telemetry.
        n_zero: Count of points where |delta_hat| <= tolerance (sparse zero reconstruction).
        sparsity_rate: Empirical probability P(delta_hat = 0 | delta = 0).
        mean_abs_nonzero: Mean magnitude of spurious residual when |delta_hat| > tolerance.
        max_abs: Peak spurious amplitude across entire clean evaluation window.
        std_null: Standard deviation of reconstructed delta under null hypothesis.
        empirical_far_at_thresholds: Mapping of detection threshold tau to empirical FAR.
        percentiles: Distribution quantiles of |delta_hat| (e.g. p50, p90, p95, p99).
    """

    estimator_name: str
    n_points: int
    n_zero: int
    sparsity_rate: float
    mean_abs_nonzero: float
    max_abs: float
    std_null: float
    empirical_far_at_thresholds: dict[float, float]
    percentiles: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary representation."""
        return asdict(self)


@dataclass(frozen=True)
class OperationalThresholdCurve:
    """Operational ROC and detection threshold curve across alarm cutoffs tau.

    Attributes:
        thresholds: Array of alarm threshold values tau.
        fpr: False Positive Rate (False Alarm Rate) across thresholds: P(|delta_hat| >= tau | H0).
        tpr: True Positive Rate (Detection Recall) across thresholds: P(|delta_hat| >= tau | H1).
        auc: Area Under the empirical ROC Curve.
        tau_for_far: Minimal operational thresholds tau* ensuring FAR <= target_far.
    """

    thresholds: list[float]
    fpr: list[float]
    tpr: list[float]
    auc: float
    tau_for_far: dict[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert curve to dictionary representation."""
        return asdict(self)


def _validate_1d_array(data: np.ndarray | Sequence[float], name: str = "data", min_samples: int = 10) -> np.ndarray:
    """Validate 1D numeric array without NaNs or Infs."""
    arr = np.asarray(data, dtype=np.float64).ravel()
    if len(arr) < min_samples:
        raise ValueError(f"{name} must contain at least {min_samples} samples, got {len(arr)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values")
    return arr


def fit_parametric_distributions(data: np.ndarray | Sequence[float]) -> dict[str, DistributionFitResult]:
    """Fit Gaussian, Student-t, and Generalized Error Distribution (GED) via MLE.

    Args:
        data: Continuous empirical error samples.

    Returns:
        Dictionary mapping distribution name to DistributionFitResult.
    """
    arr = _validate_1d_array(data, name="data", min_samples=10)
    n = len(arr)
    results: dict[str, DistributionFitResult] = {}

    # 1. Gaussian (Normal) Distribution (k=2: loc, scale)
    mu, sigma = stats.norm.fit(arr)
    sigma = max(float(sigma), 1e-12)
    ll_gauss = float(np.sum(stats.norm.logpdf(arr, loc=mu, scale=sigma)))
    k_gauss = 2
    aic_gauss = 2 * k_gauss - 2 * ll_gauss
    bic_gauss = k_gauss * np.log(n) - 2 * ll_gauss
    ks_gauss, p_gauss = stats.kstest(arr, stats.norm.cdf, args=(mu, sigma))
    results["gaussian"] = DistributionFitResult(
        distribution_name="gaussian",
        params={"loc": float(mu), "scale": float(sigma)},
        log_likelihood=ll_gauss,
        aic=float(aic_gauss),
        bic=float(bic_gauss),
        ks_stat=float(ks_gauss),
        ks_pvalue=float(p_gauss),
        n_params=k_gauss,
        n_samples=n,
    )

    # 2. Student-t Distribution (k=3: df, loc, scale)
    df_t, loc_t, scale_t = stats.t.fit(arr)
    scale_t = max(float(scale_t), 1e-12)
    ll_t = float(np.sum(stats.t.logpdf(arr, df=df_t, loc=loc_t, scale=scale_t)))
    k_t = 3
    aic_t = 2 * k_t - 2 * ll_t
    bic_t = k_t * np.log(n) - 2 * ll_t
    ks_t, p_t = stats.kstest(arr, stats.t.cdf, args=(df_t, loc_t, scale_t))
    results["student_t"] = DistributionFitResult(
        distribution_name="student_t",
        params={"df": float(df_t), "loc": float(loc_t), "scale": float(scale_t)},
        log_likelihood=ll_t,
        aic=float(aic_t),
        bic=float(bic_t),
        ks_stat=float(ks_t),
        ks_pvalue=float(p_t),
        n_params=k_t,
        n_samples=n,
    )

    # 3. Generalized Error Distribution (GED / Subbotin via gennorm, k=3: beta, loc, scale)
    beta_ged, loc_ged, scale_ged = stats.gennorm.fit(arr)
    scale_ged = max(float(scale_ged), 1e-12)
    ll_ged = float(np.sum(stats.gennorm.logpdf(arr, beta=beta_ged, loc=loc_ged, scale=scale_ged)))
    k_ged = 3
    aic_ged = 2 * k_ged - 2 * ll_ged
    bic_ged = k_ged * np.log(n) - 2 * ll_ged
    ks_ged, p_ged = stats.kstest(arr, stats.gennorm.cdf, args=(beta_ged, loc_ged, scale_ged))
    results["ged"] = DistributionFitResult(
        distribution_name="ged",
        params={"beta": float(beta_ged), "loc": float(loc_ged), "scale": float(scale_ged)},
        log_likelihood=ll_ged,
        aic=float(aic_ged),
        bic=float(bic_ged),
        ks_stat=float(ks_ged),
        ks_pvalue=float(p_ged),
        n_params=k_ged,
        n_samples=n,
    )

    return results


def rank_models(
    fits: dict[str, DistributionFitResult],
    criterion: str = "aic",
) -> list[DistributionFitResult]:
    """Rank fitted distribution models by information criterion (ascending order).

    Args:
        fits: Dictionary of distribution fit results.
        criterion: Selection metric ('aic' or 'bic').

    Returns:
        List of DistributionFitResult sorted from best (lowest) to worst.
    """
    if criterion.lower() not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion}")
    key_func = (lambda res: res.aic) if criterion.lower() == "aic" else (lambda res: res.bic)
    return sorted(fits.values(), key=key_func)


def evaluate_null_clean_distribution(
    delta_hat: np.ndarray | Sequence[float],
    estimator_name: str = "Estimator",
    tolerance: float = 1e-4,
    threshold_levels: Sequence[float] | None = None,
) -> NullDistributionSummary:
    """Evaluate empirical null distribution properties and false-alarm rates under clean telemetry.

    Args:
        delta_hat: Time series of reconstructed attack delta under authentic (unattacked) conditions.
        estimator_name: Human-readable name of the evaluated inverse solver.
        tolerance: Absolute value below which delta_hat is considered exactly zero.
        threshold_levels: Threshold cutoffs at which empirical false alarm rate is measured.

    Returns:
        NullDistributionSummary dataclass.
    """
    arr = np.asarray(delta_hat, dtype=np.float64).ravel()
    if len(arr) == 0:
        raise ValueError("delta_hat must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("delta_hat contains NaN or Inf values")

    if threshold_levels is None:
        threshold_levels = [0.01, 0.02, 0.05, 0.10, 0.125, 0.15, 0.20, 0.25]

    abs_arr = np.abs(arr)
    n_points = len(arr)
    zero_mask = abs_arr <= tolerance
    n_zero = int(np.sum(zero_mask))
    sparsity_rate = float(n_zero / n_points)

    nonzero_vals = abs_arr[~zero_mask]
    mean_abs_nonzero = float(np.mean(nonzero_vals)) if len(nonzero_vals) > 0 else 0.0
    max_abs = float(np.max(abs_arr))
    std_null = float(np.std(arr))

    empirical_far: dict[float, float] = {}
    for thresh in threshold_levels:
        empirical_far[float(thresh)] = float(np.mean(abs_arr >= thresh))

    percentiles = {
        "p50": float(np.percentile(abs_arr, 50)),
        "p75": float(np.percentile(abs_arr, 75)),
        "p90": float(np.percentile(abs_arr, 90)),
        "p95": float(np.percentile(abs_arr, 95)),
        "p99": float(np.percentile(abs_arr, 99)),
        "p99_9": float(np.percentile(abs_arr, 99.9)),
    }

    return NullDistributionSummary(
        estimator_name=estimator_name,
        n_points=n_points,
        n_zero=n_zero,
        sparsity_rate=sparsity_rate,
        mean_abs_nonzero=mean_abs_nonzero,
        max_abs=max_abs,
        std_null=std_null,
        empirical_far_at_thresholds=empirical_far,
        percentiles=percentiles,
    )


def compute_operational_threshold_curves(
    null_deltas: np.ndarray | Sequence[float],
    attack_deltas: np.ndarray | Sequence[float],
    target_fars: Sequence[float] = (0.001, 0.01, 0.05),
    n_thresholds: int = 200,
) -> OperationalThresholdCurve:
    """Compute empirical ROC curve and optimal alarm thresholds for target false alarm rates.

    Args:
        null_deltas: Array of reconstructed deltas under null hypothesis (H0: unattacked).
        attack_deltas: Array of reconstructed deltas under alternative hypothesis (H1: attacked).
        target_fars: Collection of target False Alarm Rates (alpha in (0, 1)).
        n_thresholds: Number of threshold evaluation points between 0 and peak magnitude.

    Returns:
        OperationalThresholdCurve dataclass.
    """
    arr_null = np.asarray(null_deltas, dtype=np.float64).ravel()
    arr_attack = np.asarray(attack_deltas, dtype=np.float64).ravel()

    if len(arr_null) == 0 or len(arr_attack) == 0:
        raise ValueError("null_deltas and attack_deltas must be non-empty")
    if not (np.all(np.isfinite(arr_null)) and np.all(np.isfinite(arr_attack))):
        raise ValueError("Inputs contain NaN or Inf values")

    for far in target_fars:
        if not (0.0 < far < 1.0):
            raise ValueError(f"target_fars must be strictly between 0 and 1, got {far}")

    abs_null = np.abs(arr_null)
    abs_attack = np.abs(arr_attack)

    max_val = max(float(np.max(abs_null)), float(np.max(abs_attack)), 0.01) * 1.05
    thresholds = np.linspace(0.0, max_val, n_thresholds)

    fpr_list: list[float] = []
    tpr_list: list[float] = []

    for tau in thresholds:
        fpr_list.append(float(np.mean(abs_null >= tau)))
        tpr_list.append(float(np.mean(abs_attack >= tau)))

    # Compute ROC AUC: as tau increases, FPR goes 1 -> 0, TPR goes 1 -> 0.
    # We sort by FPR ascending to compute trapezoidal integration.
    sorted_order = np.argsort(fpr_list)
    fpr_sorted = np.array(fpr_list)[sorted_order]
    tpr_sorted = np.array(tpr_list)[sorted_order]

    # Prepend (0, 0) and append (1, 1) if not strictly reached
    all_fpr = np.concatenate(([0.0], fpr_sorted, [1.0]))
    all_tpr = np.concatenate(([0.0], tpr_sorted, [1.0]))
    auc = float(integrate.trapezoid(all_tpr, all_fpr))
    auc = min(max(auc, 0.0), 1.0)

    # Optimal tau* for target FAR:
    # Smallest tau such that P(|delta_null| >= tau) <= far
    tau_for_far: dict[float, float] = {}
    for far in target_fars:
        # Exact empirical quantile guaranteeing empirical FAR <= far
        tau_q = float(np.quantile(abs_null, 1.0 - far))
        # Refine by checking if tau_q strictly satisfies empirical condition
        if np.mean(abs_null >= tau_q) > far:
            # Shift slightly to the next distinct value above
            greater_vals = abs_null[abs_null > tau_q]
            if len(greater_vals) > 0:
                tau_q = float(np.min(greater_vals))
        tau_for_far[float(far)] = tau_q

    return OperationalThresholdCurve(
        thresholds=[float(t) for t in thresholds],
        fpr=fpr_list,
        tpr=tpr_list,
        auc=auc,
        tau_for_far=tau_for_far,
    )
