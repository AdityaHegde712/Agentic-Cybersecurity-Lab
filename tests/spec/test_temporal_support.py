"""LOCKED contract for label-free temporal support selection."""

import numpy as np

from src.evaluation.temporal_support import temporal_support_mask


def test_temporal_support_keeps_a_sustained_hysteresis_segment() -> None:
    residual = np.array([[0.0], [2.0], [1.2], [1.1], [0.0]])

    support = temporal_support_mask(
        residual,
        sensor_scale=np.array([1.0]),
        start_thresholds=np.array([1.5]),
        continuation_thresholds=np.array([1.0]),
        minimum_duration=3,
    )

    assert support[:, 0].tolist() == [False, True, True, True, False]


def test_temporal_support_rejects_an_isolated_high_residual_spike() -> None:
    residual = np.array([[0.0], [2.0], [0.0], [0.0]])

    support = temporal_support_mask(
        residual,
        sensor_scale=np.array([1.0]),
        start_thresholds=np.array([1.5]),
        continuation_thresholds=np.array([1.0]),
        minimum_duration=2,
    )

    assert not support.any()


def test_temporal_support_requires_a_high_threshold_seed() -> None:
    residual = np.array([[1.1], [1.2], [1.3]])

    support = temporal_support_mask(
        residual,
        sensor_scale=np.array([1.0]),
        start_thresholds=np.array([1.5]),
        continuation_thresholds=np.array([1.0]),
        minimum_duration=3,
    )

    assert not support.any()
