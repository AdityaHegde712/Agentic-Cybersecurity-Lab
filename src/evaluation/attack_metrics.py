"""Recovery metrics for additive attacks with known injected ground truth."""

import numpy as np


def additive_recovery_metrics(
    estimated_delta: np.ndarray,
    true_delta: np.ndarray,
) -> dict[str, float]:
    """Score support, magnitude, and sign without rewarding sparse zero guesses.

    Metrics require an injected ``true_delta``.  Do not apply them to real ICS
    labels, which identify attack intervals but do not supply a clean
    counterfactual measurement for every feature.
    """
    _validate_shapes(estimated_delta, true_delta)
    true_support = true_delta != 0.0
    estimated_support = estimated_delta != 0.0

    true_positive = int(np.logical_and(true_support, estimated_support).sum())
    false_positive = int(np.logical_and(~true_support, estimated_support).sum())
    false_negative = int(np.logical_and(true_support, ~estimated_support).sum())

    support_precision = _safe_ratio(true_positive, true_positive + false_positive)
    support_recall = _safe_ratio(true_positive, true_positive + false_negative)
    support_f1 = _safe_ratio(2.0 * support_precision * support_recall, support_precision + support_recall)
    support_iou = _safe_ratio(true_positive, true_positive + false_positive + false_negative)

    if not true_support.any():
        magnitude_mae = 0.0
        sign_accuracy = 1.0
    else:
        magnitude_mae = float(np.abs(estimated_delta[true_support] - true_delta[true_support]).mean())
        sign_accuracy = float(
            (np.sign(estimated_delta[true_support]) == np.sign(true_delta[true_support])).mean()
        )

    return {
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_f1": support_f1,
        "support_iou": support_iou,
        "magnitude_mae_on_support": magnitude_mae,
        "sign_accuracy_on_support": sign_accuracy,
    }


def _validate_shapes(estimated_delta: np.ndarray, true_delta: np.ndarray) -> None:
    if estimated_delta.shape != true_delta.shape:
        raise ValueError("estimated_delta and true_delta must have the same shape")
    if estimated_delta.ndim != 2:
        raise ValueError("delta arrays must be two-dimensional [time, sensor]")
    if not np.isfinite(estimated_delta).all() or not np.isfinite(true_delta).all():
        raise ValueError("delta arrays must contain only finite values")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
