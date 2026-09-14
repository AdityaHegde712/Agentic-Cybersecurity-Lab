"""Ground-truth generators for observation-only temporal additive attacks."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


AttackFamily = Literal["step", "ramp", "periodic"]


@dataclass(frozen=True)
class AdditiveAttackSpec:
    """Describe a finite-duration additive perturbation on selected sensors."""

    attack_id: str
    family: AttackFamily
    start: int
    duration: int
    magnitude: float
    sensors: tuple[int, ...]
    period: int | None = None


@dataclass(frozen=True)
class AdditiveAttackResult:
    """Keep observed data and its injected ground truth together."""

    observed: np.ndarray
    delta: np.ndarray
    event: dict[str, object]


def inject_additive_attack(
    clean: np.ndarray,
    spec: AdditiveAttackSpec,
) -> AdditiveAttackResult:
    """Return ``clean + delta`` without mutating the clean time series.

    This function models sensor-observation spoofing only. It must not be used
    to represent an actuator attack that changes the underlying process state.
    """
    _validate_clean(clean)
    _validate_spec(spec, clean.shape)

    delta = np.zeros_like(clean, dtype=np.float64)
    end = spec.start + spec.duration
    waveform = _waveform(spec)
    delta[spec.start:end, list(spec.sensors)] = waveform[:, np.newaxis]
    observed = clean.astype(np.float64, copy=True) + delta

    event = {
        "attack_id": spec.attack_id,
        "family": spec.family,
        "start": spec.start,
        "end": end,
        "sensors": list(spec.sensors),
    }
    return AdditiveAttackResult(observed=observed, delta=delta, event=event)


def _validate_clean(clean: np.ndarray) -> None:
    if clean.ndim != 2:
        raise ValueError("clean must be a two-dimensional [time, sensor] array")
    if clean.shape[0] == 0 or clean.shape[1] == 0:
        raise ValueError("clean must contain at least one timestep and one sensor")
    if not np.isfinite(clean).all():
        raise ValueError("clean must contain only finite values")


def _validate_spec(spec: AdditiveAttackSpec, shape: tuple[int, int]) -> None:
    n_timesteps, n_sensors = shape
    if spec.family not in {"step", "ramp", "periodic"}:
        raise ValueError(f"unsupported additive attack family: {spec.family!r}")
    if not spec.attack_id:
        raise ValueError("attack_id must not be empty")
    if spec.start < 0:
        raise ValueError("start must be non-negative")
    if spec.duration <= 0:
        raise ValueError("duration must be positive")
    if spec.start + spec.duration > n_timesteps:
        raise ValueError("attack interval must fit within clean")
    if not spec.sensors:
        raise ValueError("sensors must not be empty")
    if len(set(spec.sensors)) != len(spec.sensors):
        raise ValueError("sensors must not contain duplicates")
    if min(spec.sensors) < 0 or max(spec.sensors) >= n_sensors:
        raise ValueError("sensor indices must be within clean")
    if not np.isfinite(spec.magnitude):
        raise ValueError("magnitude must be finite")
    if spec.family == "periodic" and (spec.period is None or spec.period <= 0):
        raise ValueError("periodic attacks require a positive period")


def _waveform(spec: AdditiveAttackSpec) -> np.ndarray:
    indices = np.arange(spec.duration, dtype=np.float64)
    if spec.family == "step":
        return np.full(spec.duration, spec.magnitude, dtype=np.float64)
    if spec.family == "ramp":
        if spec.duration == 1:
            return np.array([spec.magnitude], dtype=np.float64)
        return spec.magnitude * indices / (spec.duration - 1)
    if spec.family == "periodic":
        assert spec.period is not None
        return spec.magnitude * np.sin(2.0 * np.pi * indices / spec.period)
    raise ValueError(f"unsupported additive attack family: {spec.family!r}")
