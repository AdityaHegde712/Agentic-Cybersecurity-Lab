"""Typed configuration for the BATADAL known-delta recovery pilot."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from src.evaluation.injected_recovery import InjectionScenario


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INJECTED_RECOVERY_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "batadal_injected_recovery.json"
)


@dataclass(frozen=True)
class InjectedRecoveryConfig:
    """Immutable inputs for a known-delta recovery experiment."""

    dataset: str
    source_split: str
    checkpoint_path: Path
    output_dir: Path
    limit: int
    calibration_length: int
    threshold_quantile: float
    scenarios: tuple[InjectionScenario, ...]


def load_injected_recovery_config(config_path: Path) -> InjectedRecoveryConfig:
    """Load a portable BATADAL injection plan and reject unsafe paths."""
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"injected recovery config not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"injected recovery config is not valid JSON: {config_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("injected recovery config must be a JSON object")

    dataset = _required_string(payload, "dataset")
    source_split = _required_string(payload, "source_split")
    limit = _positive_integer(payload, "limit")
    calibration_length = _positive_integer(payload, "calibration_length")
    if calibration_length >= limit:
        raise ValueError("calibration_length must be smaller than limit")

    scenarios = tuple(_scenario(item) for item in _required_list(payload, "scenarios"))
    if len({scenario.attack_id for scenario in scenarios}) != len(scenarios):
        raise ValueError("injected recovery scenario names must be unique")
    if any(scenario.start < calibration_length for scenario in scenarios):
        raise ValueError("injected attacks must start after calibration")
    if any(scenario.start + scenario.duration > limit for scenario in scenarios):
        raise ValueError("injected attacks must fit within limit")

    threshold_quantile = _required_float(payload, "threshold_quantile")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be strictly between zero and one")

    return InjectedRecoveryConfig(
        dataset=dataset,
        source_split=source_split,
        checkpoint_path=_relative_path(_required_string(payload, "checkpoint_path"), "checkpoint_path"),
        output_dir=_relative_path(_required_string(payload, "output_dir"), "output_dir"),
        limit=limit,
        calibration_length=calibration_length,
        threshold_quantile=threshold_quantile,
        scenarios=scenarios,
    )


def _scenario(payload: object) -> InjectionScenario:
    if not isinstance(payload, dict):
        raise ValueError("each injected recovery scenario must be a JSON object")
    sensors = payload.get("sensors")
    if not isinstance(sensors, list) or not sensors or any(
        isinstance(sensor, bool) or not isinstance(sensor, int) or sensor < 0
        for sensor in sensors
    ):
        raise ValueError("scenario sensors must be non-negative integer indices")
    period = payload.get("period")
    if period is not None and (isinstance(period, bool) or not isinstance(period, int)):
        raise ValueError("scenario period must be an integer when provided")
    return InjectionScenario(
        attack_id=_required_string(payload, "attack_id"),
        family=_required_string(payload, "family"),  # type: ignore[arg-type]
        start=_positive_integer(payload, "start", allow_zero=True),
        duration=_positive_integer(payload, "duration"),
        magnitude_standard_deviations=_required_float(
            payload,
            "magnitude_standard_deviations",
        ),
        sensors=tuple(sensors),
        period=period,
    )


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"injected recovery field '{field}' must be a non-empty string")
    return value


def _required_list(payload: dict[str, Any], field: str) -> list[object]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"injected recovery field '{field}' must be a non-empty list")
    return value


def _positive_integer(
    payload: dict[str, Any],
    field: str,
    *,
    allow_zero: bool = False,
) -> int:
    value = payload.get(field)
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"injected recovery field '{field}' must be an integer >= {minimum}")
    return value


def _required_float(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"injected recovery field '{field}' must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"injected recovery field '{field}' must be finite")
    return parsed


def _relative_path(value: str, field: str) -> Path:
    path = Path(value)
    is_absolute = value.startswith(("/", "\\")) or path.is_absolute()
    if is_absolute or ".." in path.parts:
        raise ValueError(f"injected recovery {field} must be repository-relative")
    if not path.parts or path.parts[0] != "results":
        raise ValueError(f"injected recovery {field} must be inside results/")
    return path
