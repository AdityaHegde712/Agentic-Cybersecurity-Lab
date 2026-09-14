"""LOCKED contract for leakage-safe statistical-baseline evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.baseline_runner import (
    calibrate_blocked_threshold,
    prepare_feature_matrices,
    summarize_score_series,
)


def test_summary_calibrates_threshold_from_validation_and_scores_test_events() -> None:
    validation_scores = np.array([0.0, 1.0, 2.0, 3.0])
    test_scores = np.array([0.0, 3.0, 0.0, 4.0])
    test_labels = np.array([0, 1, 0, 1])

    summary = summarize_score_series(
        validation_scores,
        test_scores,
        test_labels,
        threshold_quantile=0.75,
    )

    assert summary["threshold"] == pytest.approx(2.25)
    assert summary["point_fpr"] == pytest.approx(0.0)
    assert summary["point_tpr"] == pytest.approx(1.0)
    assert summary["event_recall"] == pytest.approx(1.0)
    assert summary["median_detection_delay"] == pytest.approx(0.0)


def test_blocked_calibration_uses_the_most_conservative_contiguous_normal_block() -> None:
    validation_scores = np.array([0.0, 100.0, 40.0, 40.0])

    threshold = calibrate_blocked_threshold(
        validation_scores,
        threshold_quantile=0.75,
        validation_blocks=2,
    )

    assert threshold == pytest.approx(75.0)
    assert threshold > np.quantile(validation_scores, 0.75)


def test_summary_uses_blocked_validation_calibration() -> None:
    summary = summarize_score_series(
        np.array([0.0, 100.0, 40.0, 40.0]),
        np.array([60.0, 80.0]),
        np.array([0, 0]),
        threshold_quantile=0.75,
        validation_blocks=2,
    )

    assert summary["threshold"] == pytest.approx(75.0)
    assert summary["point_fpr"] == pytest.approx(0.5)


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.1])
def test_summary_rejects_invalid_threshold_quantiles(quantile: float) -> None:
    with pytest.raises(ValueError):
        summarize_score_series(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0, 1]),
            threshold_quantile=quantile,
        )


def test_feature_preparation_drops_all_null_columns_and_uses_train_only_medians() -> None:
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "all_null": [np.nan, np.nan, np.nan]})
    validation = pd.DataFrame({"a": [np.nan, 5.0], "all_null": [np.nan, np.nan]})
    test = pd.DataFrame({"a": [np.nan, 7.0], "all_null": [np.nan, np.nan]})

    train_values, validation_values, test_values, sensors = prepare_feature_matrices(
        train,
        validation,
        test,
        ["a", "all_null"],
    )

    assert sensors == ["a"]
    assert np.array_equal(train_values[:, 0], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(validation_values[:, 0], np.array([2.0, 5.0]))
    assert np.array_equal(test_values[:, 0], np.array([2.0, 7.0]))
