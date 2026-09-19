"""LOCKED contract for known-delta temporal attack recovery evaluation."""

import numpy as np
import pytest

from src.evaluation.injected_recovery import (
    InjectionScenario,
    inject_standardized_attack,
    temporal_recovery_metrics,
    threshold_estimated_delta,
)
from src.models.lstm_forecaster import FeatureNormalizer


def test_standardized_injection_converts_known_delta_back_to_sensor_units() -> None:
    clean = np.array([[10.0, 100.0]] * 6)
    normalizer = FeatureNormalizer(
        mean=np.array([10.0, 100.0], dtype=np.float32),
        scale=np.array([2.0, 10.0], dtype=np.float32),
    )
    scenario = InjectionScenario(
        attack_id="step-sensor-one",
        family="step",
        start=2,
        duration=3,
        magnitude_standard_deviations=2.0,
        sensors=(1,),
    )

    injected = inject_standardized_attack(clean, normalizer, scenario)

    assert np.array_equal(injected.observed, clean + injected.delta)
    assert np.all(injected.delta[:2] == 0.0)
    assert np.all(injected.delta[2:5, 0] == 0.0)
    assert np.all(injected.delta[2:5, 1] == pytest.approx(20.0))


def test_thresholded_estimate_and_temporal_metrics_recover_known_event() -> None:
    true_delta = np.zeros((5, 2))
    true_delta[1:4, 0] = 4.0
    estimated_delta = np.zeros((5, 2))
    estimated_delta[0, 0] = 1.0
    estimated_delta[1:4, 0] = 4.5

    thresholded = threshold_estimated_delta(
        estimated_delta,
        sensor_scale=np.array([2.0, 5.0]),
        threshold_standard_deviations=np.array([1.5, 1.5]),
    )
    metrics = temporal_recovery_metrics(
        thresholded,
        true_delta,
        target_indices=np.array([10, 11, 12, 13, 14]),
        event_start=11,
        event_end=14,
    )

    assert thresholded[0, 0] == 0.0
    assert metrics["support_f1"] == pytest.approx(1.0)
    assert metrics["sign_accuracy_on_support"] == pytest.approx(1.0)
    assert metrics["onset_error_samples"] == pytest.approx(0.0)
    assert metrics["duration_error_samples"] == pytest.approx(0.0)


def test_temporal_metrics_expose_missing_attack_as_infinite_timing_error() -> None:
    metrics = temporal_recovery_metrics(
        estimated_delta=np.zeros((4, 1)),
        true_delta=np.array([[0.0], [2.0], [2.0], [0.0]]),
        target_indices=np.array([20, 21, 22, 23]),
        event_start=21,
        event_end=23,
    )

    assert metrics["support_recall"] == pytest.approx(0.0)
    assert np.isinf(metrics["onset_error_samples"])
    assert np.isinf(metrics["duration_error_samples"])
