"""Calibration and temporal-support diagnostics for residual attack recovery."""

import numpy as np
import pandas as pd


def clean_window_activation_summary(
    residual: np.ndarray,
    *,
    sensor_scale: np.ndarray,
    thresholds: np.ndarray,
    target_indices: np.ndarray,
    window_length: int,
    calibration_end: int,
) -> pd.DataFrame:
    """Summarize threshold activity across clean chronological windows."""
    active = _active_mask(residual, sensor_scale, thresholds)
    _validate_target_indices(target_indices, len(active))
    if window_length < 1:
        raise ValueError("window_length must be positive")

    records: list[dict[str, object]] = []
    start = 0
    while start < len(active):
        end = min(start + window_length, len(active))
        calibration_boundary = int(np.searchsorted(target_indices, calibration_end))
        crosses_calibration_boundary = start < calibration_boundary < end
        if crosses_calibration_boundary:
            end = calibration_boundary
        window_indices = target_indices[start:end]
        role = "calibration" if int(window_indices[-1]) < calibration_end else "assessment"
        records.append(
            {
                "window_start": int(window_indices[0]),
                "window_end": int(window_indices[-1]) + 1,
                "window_role": role,
                **_activation_fractions(active[start:end]),
            }
        )
        start = end
    return pd.DataFrame(records)


def scenario_phase_activation_summary(
    residual: np.ndarray,
    *,
    sensor_scale: np.ndarray,
    thresholds: np.ndarray,
    target_indices: np.ndarray,
    attack_sensors: tuple[int, ...],
    event_start: int,
    event_end: int,
) -> pd.DataFrame:
    """Measure attacked-sensor threshold activity before, during, and after an event."""
    active = _active_mask(residual, sensor_scale, thresholds)
    _validate_target_indices(target_indices, len(active))
    sensor_indices = _validate_attack_sensors(attack_sensors, active.shape[1])
    if event_start < 0 or event_end <= event_start:
        raise ValueError("event bounds must define a non-empty interval")

    phase_masks = {
        "pre_attack": target_indices < event_start,
        "attack": (target_indices >= event_start) & (target_indices < event_end),
        "post_attack": target_indices >= event_end,
    }
    records: list[dict[str, object]] = []
    for phase, mask in phase_masks.items():
        phase_active = active[mask][:, sensor_indices]
        if not len(phase_active):
            fractions = {
                "active_sensor_fraction": float("nan"),
                "active_timestep_fraction": float("nan"),
            }
        else:
            fractions = _activation_fractions(phase_active)
        records.append(
            {
                "phase": phase,
                "sample_count": int(mask.sum()),
                **fractions,
            }
        )
    return pd.DataFrame(records)


def attacked_sensor_trace(
    residual: np.ndarray,
    true_delta: np.ndarray,
    *,
    sensor_scale: np.ndarray,
    target_indices: np.ndarray,
    attack_sensors: tuple[int, ...],
) -> pd.DataFrame:
    """Aggregate residual and known delta traces over injected attack sensors only."""
    _validate_residual(residual, "residual")
    _validate_residual(true_delta, "true_delta")
    if residual.shape != true_delta.shape:
        raise ValueError("residual and true_delta must have the same shape")
    _validate_sensor_scale(sensor_scale, residual.shape[1])
    _validate_target_indices(target_indices, len(residual))
    sensor_indices = _validate_attack_sensors(attack_sensors, residual.shape[1])

    standardized_residual = residual / sensor_scale
    standardized_delta = true_delta / sensor_scale
    return pd.DataFrame(
        {
            "target_index": target_indices.astype(int),
            "mean_signed_residual_standard_deviations": standardized_residual[
                :, sensor_indices
            ].mean(axis=1),
            "mean_signed_true_delta_standard_deviations": standardized_delta[
                :, sensor_indices
            ].mean(axis=1),
        }
    )


def _active_mask(
    residual: np.ndarray,
    sensor_scale: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    _validate_residual(residual, "residual")
    _validate_sensor_scale(sensor_scale, residual.shape[1])
    _validate_thresholds(thresholds, residual.shape[1])
    return np.abs(residual / sensor_scale) >= thresholds


def _activation_fractions(active: np.ndarray) -> dict[str, float]:
    return {
        "active_sensor_fraction": float(active.mean()),
        "active_timestep_fraction": float(np.any(active, axis=1).mean()),
    }


def _validate_residual(values: np.ndarray, name: str) -> None:
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError(f"{name} must be a non-empty [time, sensor] array")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_sensor_scale(sensor_scale: np.ndarray, sensor_count: int) -> None:
    if sensor_scale.ndim != 1 or len(sensor_scale) != sensor_count:
        raise ValueError("sensor_scale must have one value per sensor")
    if not np.isfinite(sensor_scale).all() or (sensor_scale <= 0.0).any():
        raise ValueError("sensor_scale must contain only positive finite values")


def _validate_thresholds(thresholds: np.ndarray, sensor_count: int) -> None:
    if thresholds.ndim != 1 or len(thresholds) != sensor_count:
        raise ValueError("thresholds must have one value per sensor")
    if not np.isfinite(thresholds).all() or (thresholds < 0.0).any():
        raise ValueError("thresholds must contain only non-negative finite values")


def _validate_target_indices(target_indices: np.ndarray, expected_length: int) -> None:
    if target_indices.ndim != 1 or len(target_indices) != expected_length:
        raise ValueError("target_indices must match the residual time dimension")
    if not np.issubdtype(target_indices.dtype, np.integer):
        raise ValueError("target_indices must be integers")


def _validate_attack_sensors(
    attack_sensors: tuple[int, ...],
    sensor_count: int,
) -> np.ndarray:
    if not attack_sensors:
        raise ValueError("attack_sensors must not be empty")
    sensor_indices = np.asarray(attack_sensors)
    if not np.issubdtype(sensor_indices.dtype, np.integer):
        raise ValueError("attack_sensors must contain integer indices")
    if (sensor_indices < 0).any() or (sensor_indices >= sensor_count).any():
        raise ValueError("attack_sensors must be valid sensor indices")
    return sensor_indices
