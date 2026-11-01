"""Label-free temporal support selection for forecast residuals."""

import numpy as np


def temporal_support_mask(
    residual: np.ndarray,
    *,
    sensor_scale: np.ndarray,
    start_thresholds: np.ndarray,
    continuation_thresholds: np.ndarray,
    minimum_duration: int,
) -> np.ndarray:
    """Select sustained per-sensor residual segments using hysteresis thresholds."""
    _validate_residual(residual)
    sensor_count = residual.shape[1]
    _validate_positive_vector(sensor_scale, "sensor_scale", sensor_count)
    _validate_nonnegative_vector(start_thresholds, "start_thresholds", sensor_count)
    _validate_nonnegative_vector(
        continuation_thresholds,
        "continuation_thresholds",
        sensor_count,
    )
    if (continuation_thresholds > start_thresholds).any():
        raise ValueError("continuation_thresholds must not exceed start_thresholds")
    if minimum_duration < 1:
        raise ValueError("minimum_duration must be positive")

    standardized_magnitude = np.abs(residual / sensor_scale)
    continuation_active = standardized_magnitude >= continuation_thresholds
    starts = standardized_magnitude >= start_thresholds
    support = np.zeros_like(continuation_active, dtype=bool)
    for sensor_index in range(sensor_count):
        _select_sensor_segments(
            continuation_active[:, sensor_index],
            starts[:, sensor_index],
            minimum_duration,
            support[:, sensor_index],
        )
    return support


def apply_temporal_support(
    residual: np.ndarray,
    *,
    sensor_scale: np.ndarray,
    start_thresholds: np.ndarray,
    continuation_thresholds: np.ndarray,
    minimum_duration: int,
) -> np.ndarray:
    """Return residual estimates restricted to label-free sustained support."""
    support = temporal_support_mask(
        residual,
        sensor_scale=sensor_scale,
        start_thresholds=start_thresholds,
        continuation_thresholds=continuation_thresholds,
        minimum_duration=minimum_duration,
    )
    return np.where(support, residual, 0.0)


def _select_sensor_segments(
    continuation_active: np.ndarray,
    starts: np.ndarray,
    minimum_duration: int,
    support: np.ndarray,
) -> None:
    segment_start: int | None = None
    for index, is_active in enumerate(continuation_active):
        if is_active and segment_start is None:
            segment_start = index
            continue
        if is_active:
            continue
        if segment_start is not None:
            _keep_segment_if_qualified(starts, support, segment_start, index, minimum_duration)
            segment_start = None
    if segment_start is not None:
        _keep_segment_if_qualified(
            starts,
            support,
            segment_start,
            len(continuation_active),
            minimum_duration,
        )


def _keep_segment_if_qualified(
    starts: np.ndarray,
    support: np.ndarray,
    segment_start: int,
    segment_end: int,
    minimum_duration: int,
) -> None:
    segment = slice(segment_start, segment_end)
    has_high_threshold_seed = bool(starts[segment].any())
    is_long_enough = segment_end - segment_start >= minimum_duration
    if has_high_threshold_seed and is_long_enough:
        support[segment] = True


def _validate_residual(residual: np.ndarray) -> None:
    if residual.ndim != 2 or not residual.shape[0] or not residual.shape[1]:
        raise ValueError("residual must be a non-empty [time, sensor] array")
    if not np.isfinite(residual).all():
        raise ValueError("residual must contain only finite values")


def _validate_positive_vector(values: np.ndarray, name: str, length: int) -> None:
    _validate_vector(values, name, length)
    if (values <= 0.0).any():
        raise ValueError(f"{name} must contain only positive values")


def _validate_nonnegative_vector(values: np.ndarray, name: str, length: int) -> None:
    _validate_vector(values, name, length)
    if (values < 0.0).any():
        raise ValueError(f"{name} must contain only non-negative values")


def _validate_vector(values: np.ndarray, name: str, length: int) -> None:
    if values.ndim != 1 or len(values) != length:
        raise ValueError(f"{name} must have one value per sensor")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
