"""Local-first inverse research benchmark harness for BATADAL.

Implements:
- Regime cutoff extraction and onset dynamic classification.
- Deterministic SHA-256 hash ranking for stratified scenario sampling.
- Injection scenario generation matching the sealed benchmark grid.
- Precise metric computation (NRMSE on interval, support F1, signed amplitude error, clean x NRMSE).
- Execution harness across Persistence, Seasonal Naive, Companion Regression, and StepMAP estimators.
- Aggregation, serialization, and report rendering for the October 18 benchmark artifact.
"""

from __future__ import annotations

import hashlib
import json
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.inverse.benchmark_manifest import BenchmarkManifest
from src.inverse.estimators import (
    BaseDeltaEstimator,
    CompanionRegressionEstimator,
    PersistenceEstimator,
    ReconstructionResult,
    SeasonalNaiveEstimator,
    StepMAPEstimator,
)


@dataclass(frozen=True)
class BenchmarkScenario:
    """Declared parameterization of a single synthetic additive step scenario."""

    scenario_id: str
    target_channel: str
    amplitude_train_std: float
    sign: int
    duration_hours: int
    onset_index: int
    offset_index: int
    regime: str
    injected_amplitude: float
    block_name: str = "development_injection"


def sha256_seeded_rank(
    seed: int,
    candidates: list[int],
    context: str = "",
) -> list[int]:
    """Deterministic permutation ranking of candidate integers using SHA-256."""
    scored: list[tuple[str, int]] = []
    for c in candidates:
        token = f"{seed}:{context}:{c}".encode("utf-8")
        digest = hashlib.sha256(token).hexdigest()
        scored.append((digest, c))
    scored.sort(key=lambda pair: pair[0])
    return [c for _, c in scored]


def compute_regime_cutoffs(
    series: np.ndarray,
    quantiles: tuple[float, float] = (1 / 3, 2 / 3),
) -> tuple[float, float]:
    """Compute empirical quantile cutoffs on absolute first differences."""
    arr = np.asarray(series, dtype=np.float64)
    if len(arr) < 2:
        raise ValueError("Series must contain at least 2 points to compute differences")
    diffs = np.abs(np.diff(arr))
    q1 = float(np.quantile(diffs, quantiles[0]))
    q2 = float(np.quantile(diffs, quantiles[1]))
    if q1 > q2:
        raise ValueError("Lower quantile exceeds upper quantile")
    return q1, q2


def classify_series_regimes(
    series: np.ndarray,
    prev_val: float | None,
    cutoffs: tuple[float, float],
    regime_labels: tuple[str, ...] = ("low_dynamics", "medium_dynamics", "high_dynamics"),
) -> list[str]:
    """Classify each timestep into dynamic regimes based on absolute first difference."""
    arr = np.asarray(series, dtype=np.float64)
    n = len(arr)
    regimes: list[str] = []
    q1, q2 = cutoffs
    for t in range(n):
        if t > 0:
            diff = abs(arr[t] - arr[t - 1])
        elif prev_val is not None:
            diff = abs(arr[t] - float(prev_val))
        else:
            diff = 0.0

        if diff < q1:
            regimes.append(regime_labels[0])
        elif diff < q2:
            regimes.append(regime_labels[1])
        else:
            regimes.append(regime_labels[2])
    return regimes


def generate_development_grid_scenarios(
    manifest: BenchmarkManifest,
    dev_series: np.ndarray,
    prev_val: float,
    train_std: float,
    cutoffs: tuple[float, float],
) -> list[BenchmarkScenario]:
    """Generate 120 deterministic regime-stratified step scenarios from manifest grid."""
    dev_arr = np.asarray(dev_series, dtype=np.float64)
    n = len(dev_arr)
    grid = manifest.development_injections
    seed = int(manifest.seeds["injection"])
    regime_labels = grid.regime_labels

    classified_regimes = classify_series_regimes(dev_arr, prev_val, cutoffs, regime_labels)

    # Pre-index candidates by (duration, regime)
    eligible_by_dur_regime: dict[tuple[int, str], list[int]] = {}
    for dur in grid.durations_hours:
        for reg in regime_labels:
            eligible = [
                t for t in range(n)
                if classified_regimes[t] == reg and (t + dur <= n)
            ]
            eligible_by_dur_regime[(dur, reg)] = eligible

    scenarios: list[BenchmarkScenario] = []
    combos = list(itertools.product(grid.amplitudes_train_std, grid.durations_hours, grid.signs))

    for combo_idx, (amp_std, dur, sign) in enumerate(combos):
        injected_amp = sign * amp_std * train_std
        picked_onsets_in_combo: set[int] = set()

        for rep in range(grid.regime_stratified_onset_count):
            # Rotate starting regime per combination to achieve exact balanced distribution across 120 scenarios
            reg_idx = (combo_idx + rep) % len(regime_labels)
            target_regime = regime_labels[reg_idx]

            candidates = eligible_by_dur_regime[(dur, target_regime)]
            context = f"{amp_std:.2f}_{dur}_{sign}_{target_regime}"
            ranked = sha256_seeded_rank(seed, candidates, context=context)

            chosen_onset: int | None = None
            for cand in ranked:
                if cand not in picked_onsets_in_combo:
                    chosen_onset = cand
                    picked_onsets_in_combo.add(cand)
                    break

            if chosen_onset is None:
                raise RuntimeError(
                    f"Exhausted candidates for combo amp={amp_std}, dur={dur}, sign={sign}, regime={target_regime}"
                )

            scenario_id = (
                f"dev_amp{amp_std:.2f}_dur{dur:02d}_sign{sign:+d}_{target_regime}_rep{rep}"
            )
            scenario = BenchmarkScenario(
                scenario_id=scenario_id,
                target_channel=manifest.target_sensor,
                amplitude_train_std=float(amp_std),
                sign=int(sign),
                duration_hours=int(dur),
                onset_index=int(chosen_onset),
                offset_index=int(chosen_onset + dur),
                regime=target_regime,
                injected_amplitude=float(injected_amp),
            )
            scenarios.append(scenario)

    return scenarios


def compute_scenario_metrics(
    *,
    delta_hat: np.ndarray,
    delta_true: np.ndarray,
    x_hat: np.ndarray,
    x_true: np.ndarray,
    onset: int,
    offset: int,
    injected_amplitude: float,
    train_std: float,
    scenario: BenchmarkScenario | None = None,
    estimator_name: str = "",
) -> dict[str, Any]:
    """Compute the 4 benchmark evaluation metrics for a single scenario."""
    d_hat = np.asarray(delta_hat, dtype=np.float64)
    d_true = np.asarray(delta_true, dtype=np.float64)
    xh = np.asarray(x_hat, dtype=np.float64)
    xt = np.asarray(x_true, dtype=np.float64)

    # 1. Pointwise NRMSE on delta trajectory over declared interval
    interval = slice(onset, offset)
    delta_interval_rmse = float(np.sqrt(np.mean((d_hat[interval] - d_true[interval]) ** 2)))
    delta_nrmse = float(delta_interval_rmse / abs(injected_amplitude))

    # 2. Support F1 score (threshold |delta_hat| > 0.1 * |amplitude|)
    support_threshold = float(0.1 * abs(injected_amplitude))
    pred_support = np.abs(d_hat) > support_threshold
    true_support = np.zeros(len(d_true), dtype=bool)
    true_support[interval] = True

    tp = int(np.logical_and(pred_support, true_support).sum())
    fp = int(np.logical_and(pred_support, ~true_support).sum())
    fn = int(np.logical_and(~pred_support, true_support).sum())

    precision = float(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    recall = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    support_f1 = float((2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)

    # 3. Signed Amplitude Error in units of train_std
    med_delta = float(np.median(d_hat[interval]))
    signed_amplitude_error_std = float((med_delta - injected_amplitude) / train_std)

    # 4. Pointwise clean x NRMSE
    clean_x_rmse = float(np.sqrt(np.mean((xh[interval] - xt[interval]) ** 2)))
    clean_x_nrmse = float(clean_x_rmse / train_std)

    # 5. Sign accuracy
    sign_accuracy = 1.0 if np.sign(med_delta) == np.sign(injected_amplitude) else 0.0

    result: dict[str, Any] = {
        "estimator": estimator_name,
        "delta_nrmse": delta_nrmse,
        "support_f1": support_f1,
        "support_precision": precision,
        "support_recall": recall,
        "signed_amplitude_error_train_std": signed_amplitude_error_std,
        "clean_x_nrmse": clean_x_nrmse,
        "sign_accuracy": sign_accuracy,
    }

    if scenario is not None:
        result.update({
            "scenario_id": scenario.scenario_id,
            "target_channel": scenario.target_channel,
            "amplitude_train_std": scenario.amplitude_train_std,
            "duration_hours": scenario.duration_hours,
            "sign": scenario.sign,
            "regime": scenario.regime,
            "onset_index": scenario.onset_index,
            "offset_index": scenario.offset_index,
            "injected_amplitude": scenario.injected_amplitude,
        })

    return result


def run_benchmark_suite(
    scenarios: list[BenchmarkScenario],
    fit_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    target_channel: str = "L_T1",
    channel_manifest_path: str | Path | None = None,
    train_std: float | None = None,
    lambda_tv: float = 1.0,
    lambda_l1: float = 0.05,
    alpha_ridge: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute all 4 inverse estimators on the scenarios and aggregate results."""
    if train_std is None:
        train_std = float(fit_df[target_channel].std())

    calib_series = calib_df[target_channel].to_numpy(dtype=np.float64)
    clean_x_dev = dev_df[target_channel].to_numpy(dtype=np.float64)
    n_dev = len(clean_x_dev)

    # 1. Instantiate and fit estimators
    p_est = PersistenceEstimator(lag=1).fit_normal(calib_series)
    s_est = SeasonalNaiveEstimator(period=24).fit_normal(calib_series)
    reg_est = CompanionRegressionEstimator(
        target_channel=target_channel,
        channel_manifest_path=channel_manifest_path,
        alpha=alpha_ridge,
    ).fit_normal(fit_df)
    map_est = StepMAPEstimator(
        lambda_tv=lambda_tv,
        lambda_l1=lambda_l1,
        target_channel=target_channel,
        channel_manifest_path=channel_manifest_path,
        surrogate_estimator=reg_est,
    )

    estimators: dict[str, BaseDeltaEstimator] = {
        "PersistenceEstimator": p_est,
        "SeasonalNaiveEstimator": s_est,
        "CompanionRegressionEstimator": reg_est,
        "StepMAPEstimator": map_est,
    }

    records: list[dict[str, Any]] = []

    # 2. Iterate through scenarios
    for sc in scenarios:
        # Construct exact attacked time series
        delta_true = np.zeros(n_dev, dtype=np.float64)
        delta_true[sc.onset_index : sc.offset_index] = sc.injected_amplitude

        attacked_df = dev_df.copy()
        attacked_df[target_channel] = clean_x_dev + delta_true
        attack_interval = (sc.onset_index, sc.offset_index)

        for est_name, est in estimators.items():
            if isinstance(est, (PersistenceEstimator, SeasonalNaiveEstimator)):
                res = est.reconstruct(
                    attacked_df,
                    attack_interval=attack_interval,
                    target_channel=target_channel,
                    prior_clean_history=calib_series,
                )
            else:
                res = est.reconstruct(
                    attacked_df,
                    attack_interval=attack_interval,
                )

            m = compute_scenario_metrics(
                delta_hat=res.delta_hat,
                delta_true=delta_true,
                x_hat=res.x_hat,
                x_true=clean_x_dev,
                onset=sc.onset_index,
                offset=sc.offset_index,
                injected_amplitude=sc.injected_amplitude,
                train_std=train_std,
                scenario=sc,
                estimator_name=est_name,
            )
            records.append(m)

    results_df = pd.DataFrame(records)

    # 3. Compute aggregate summary
    summary_dict = _build_summary_dict(
        results_df=results_df,
        target_channel=target_channel,
        train_std=train_std,
        n_scenarios=len(scenarios),
    )

    return results_df, summary_dict


def _build_summary_dict(
    results_df: pd.DataFrame,
    target_channel: str,
    train_std: float,
    n_scenarios: int,
) -> dict[str, Any]:
    """Build canonical summary structure with aggregations and acceptance checks."""
    estimators_summary: dict[str, Any] = {}

    for est_name, group in results_df.groupby("estimator"):
        sign_subset = group[group["amplitude_train_std"] >= 1.0]
        sign_acc = float(sign_subset["sign_accuracy"].mean()) if len(sign_subset) > 0 else 1.0

        est_stats: dict[str, Any] = {
            "median_delta_nrmse": float(group["delta_nrmse"].median()),
            "mean_delta_nrmse": float(group["delta_nrmse"].mean()),
            "median_support_f1": float(group["support_f1"].median()),
            "mean_support_f1": float(group["support_f1"].mean()),
            "median_signed_amplitude_error_train_std": float(
                group["signed_amplitude_error_train_std"].abs().median()
            ),
            "mean_clean_x_nrmse": float(group["clean_x_nrmse"].mean()),
            "sign_accuracy_ge_1std": sign_acc,
            "by_amplitude": {},
            "by_duration": {},
            "by_regime": {},
        }

        for amp, amp_grp in group.groupby("amplitude_train_std"):
            est_stats["by_amplitude"][str(amp)] = {
                "median_delta_nrmse": float(amp_grp["delta_nrmse"].median()),
                "median_support_f1": float(amp_grp["support_f1"].median()),
                "median_signed_amplitude_error": float(amp_grp["signed_amplitude_error_train_std"].abs().median()),
            }

        for dur, dur_grp in group.groupby("duration_hours"):
            est_stats["by_duration"][str(dur)] = {
                "median_delta_nrmse": float(dur_grp["delta_nrmse"].median()),
                "median_support_f1": float(dur_grp["support_f1"].median()),
            }

        for reg, reg_grp in group.groupby("regime"):
            est_stats["by_regime"][str(reg)] = {
                "median_delta_nrmse": float(reg_grp["delta_nrmse"].median()),
                "median_support_f1": float(reg_grp["support_f1"].median()),
            }

        estimators_summary[str(est_name)] = est_stats

    # Compare MAP / Companion against best causal baseline (Persistence vs Seasonal Naive)
    causal_candidates = [
        name for name in ("PersistenceEstimator", "SeasonalNaiveEstimator")
        if name in estimators_summary
    ]
    if causal_candidates:
        best_causal = min(
            causal_candidates,
            key=lambda name: estimators_summary[name]["median_delta_nrmse"],
        )
        base_nrmse = estimators_summary[best_causal]["median_delta_nrmse"]
    else:
        best_causal = "none"
        base_nrmse = 1.0

    improvements: dict[str, float] = {}
    for name, stats in estimators_summary.items():
        if base_nrmse > 0:
            improvements[name] = float((base_nrmse - stats["median_delta_nrmse"]) / base_nrmse)
        else:
            improvements[name] = 0.0

    # Acceptance threshold checks (from BATADAL benchmark manifest)
    step_map_stats = estimators_summary.get("StepMAPEstimator", {})
    step_map_nrmse_imp = improvements.get("StepMAPEstimator", 0.0)
    step_map_amp_err = step_map_stats.get("median_signed_amplitude_error_train_std", 1.0)
    step_map_sign_acc = step_map_stats.get("sign_accuracy_ge_1std", 0.0)

    acceptance_checks = {
        "best_causal_baseline": best_causal,
        "best_causal_median_delta_nrmse": base_nrmse,
        "step_map_median_delta_nrmse_improvement": step_map_nrmse_imp,
        "nrmse_improvement_ge_20pct": bool(step_map_nrmse_imp >= 0.20),
        "signed_amplitude_error_le_0.25_std": bool(step_map_amp_err <= 0.25),
        "sign_accuracy_ge_95pct": bool(step_map_sign_acc >= 0.95),
        "overall_development_acceptance_met": bool(
            step_map_nrmse_imp >= 0.20
            and step_map_amp_err <= 0.25
            and step_map_sign_acc >= 0.95
        ),
    }

    return {
        "metadata": {
            "target_channel": target_channel,
            "train_std": train_std,
            "n_scenarios": n_scenarios,
            "n_evaluations": len(results_df),
        },
        "estimators": estimators_summary,
        "improvements_over_best_causal": improvements,
        "acceptance_checks": acceptance_checks,
    }
