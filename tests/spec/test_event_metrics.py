"""LOCKED contract for event-level detection evaluation."""

import numpy as np
import pytest

from src.evaluation.event_metrics import event_detection_metrics


def test_event_metrics_score_each_attack_segment_once_and_measure_delay() -> None:
    labels = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    predictions = np.array([0, 0, 1, 0, 0, 0, 1, 0])

    metrics = event_detection_metrics(labels, predictions)

    assert metrics["n_events"] == 2
    assert metrics["detected_events"] == 2
    assert metrics["event_recall"] == pytest.approx(1.0)
    assert metrics["median_detection_delay"] == pytest.approx(1.5)


def test_event_metrics_distinguish_a_missed_event() -> None:
    labels = np.array([0, 1, 1, 0, 1, 1, 0])
    predictions = np.array([0, 1, 0, 0, 0, 0, 0])

    metrics = event_detection_metrics(labels, predictions)

    assert metrics["n_events"] == 2
    assert metrics["detected_events"] == 1
    assert metrics["event_recall"] == pytest.approx(0.5)
    assert metrics["median_detection_delay"] == pytest.approx(0.0)


def test_event_metrics_reject_non_binary_or_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        event_detection_metrics(np.array([0, 1]), np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        event_detection_metrics(np.array([0, 2]), np.array([0, 1]))
    with pytest.raises(ValueError):
        event_detection_metrics(np.array([[0, 1]]), np.array([[0, 1]]))
