"""Known-delta evaluation utilities for observation-only additive attacks."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.attacks.additive import AdditiveAttackResult, AdditiveAttackSpec, inject_additive_attack
from src.evaluation.attack_metrics import additive_recovery_metrics
from src.models.lstm_forecaster import FeatureNormalizer


AttackFamily = Literal["step", "ramp", "periodic"]


@dataclass(frozen=True)
class InjectionScenario:
    """Specify a temporal attack in train-derived sensor standard deviations."""

    attack_id: str
    family: AttackFamily
    start: int
    duration: int
    magnitude_standard_deviations: float
    sensors: tuple[int, ...]
    period: int | None = None

    def to_attack_spec(self) -> AdditiveAttackSpec:
        """Build the existing additive generator specification in standardized space."""
        return AdditiveAttackSpec(
            attack_id=self.attack_id,
            family=self.family,
            start=self.start,
            duration=self.duration,
            magnitude=self.magnitude_standard_deviations,
            sensors=self.sensors,
            period=self.period,
        )


def inject_standardized_attack(
    clean: np.ndarray,
    normalizer: FeatureNormalizer,
    scenario: InjectionScenario,
) -> AdditiveAttackResult:
    """Inject a scale-comparable attack and return data in original sensor units."""
    standardized_clean = normalizer.transform(clean)
    standardized_result = inject_additive_attack(
        standardized_clean,
        scenario.to_attack_spec(),
    )
    observed = normalizer.inverse(standardized_result.observed)
    delta = observed.astype(np.float64) - clean.astype(np.float64)
    return AdditiveAttackResult(
        observed=observed,
        delta=delta,
        event=standardized_result.event,
    )


def threshold_estimated_delta(
    estimated_delta: np.ndarray,
    sensor_scale: np.ndarray,
    threshold_standard_deviations: np.ndarray,
) -> np.ndarray:
    """Suppress residuals below per-sensor clean calibration thresholds."""
    _validate_delta_array(estimated_delta, "estimated_delta")
    _validate_sensor_vector(sensor_scale, "sensor_scale", estimated_delta.shape[1])
    _validate_sensor_vector(
        threshold_standard_deviations,
        "threshold_standard_deviations",
        estimated_delta.shape[1],
    )
    if (sensor_scale <= 0.0).any() or (threshold_standard_deviations < 0.0).any():
        raise ValueError("sensor scale must be positive and thresholds must be non-negative")

    standardized_residual = np.abs(estimated_delta / sensor_scale)
    is_active = standardized_residual >= threshold_standard_deviations
    return np.where(is_active, estimated_delta, 0.0)


def temporal_recovery_metrics(
    estimated_delta: np.ndarray,
    true_delta: np.ndarray,
    *,
    target_indices: np.ndarray,
    event_start: int,
    event_end: int,
) -> dict[str, float]:
    """Combine support recovery with attack-onset and duration error metrics."""
    _validate_delta_array(estimated_delta, "estimated_delta")
    _validate_delta_array(true_delta, "true_delta")
    if estimated_delta.shape != true_delta.shape:
        raise ValueError("estimated_delta and true_delta must have the same shape")
    if target_indices.ndim != 1 or len(target_indices) != len(estimated_delta):
        raise ValueError("target_indices must match the delta time dimension")
    if event_start < 0 or event_end <= event_start:
        raise ValueError("event bounds must be a non-empty interval")

    metrics = additive_recovery_metrics(estimated_delta, true_delta)
    active_timesteps = np.flatnonzero(np.any(estimated_delta != 0.0, axis=1))
    if not len(active_timesteps):
        return {
            **metrics,
            "onset_error_samples": float("inf"),
            "duration_error_samples": float("inf"),
        }

    estimated_start = int(target_indices[active_timesteps[0]])
    estimated_end = int(target_indices[active_timesteps[-1]]) + 1
    return {
        **metrics,
        "onset_error_samples": float(abs(estimated_start - event_start)),
        "duration_error_samples": float(
            abs((estimated_end - estimated_start) - (event_end - event_start))
        ),
    }


def _validate_delta_array(values: np.ndarray, name: str) -> None:
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError(f"{name} must be a non-empty [time, sensor] array")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_sensor_vector(values: np.ndarray, name: str, sensor_count: int) -> None:
    if values.ndim != 1 or len(values) != sensor_count or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite with one value per sensor")
