"""Typed loader for the bounded BATADAL LSTM ablation contract."""

from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABLATION_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "experiments" / "batadal_lstm_ablation.json"
)


@dataclass(frozen=True)
class AblationRunConfig:
    """One named LSTM configuration within a controlled experiment grid."""

    name: str
    context_length: int
    hidden_size: int
    max_epochs: int


@dataclass(frozen=True)
class LstmAblationConfig:
    """Immutable parameters shared by all runs in a BATADAL ablation."""

    experiment_name: str
    dataset: str
    seed: int
    limit: int
    output_dir: Path
    runs: tuple[AblationRunConfig, ...]


def load_ablation_config(config_path: Path) -> LstmAblationConfig:
    """Load and validate a repository-relative BATADAL ablation configuration."""
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"ablation config not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"ablation config is not valid JSON: {config_path}") from error

    if not isinstance(payload, dict):
        raise ValueError("ablation config must be a JSON object")

    output_dir = _relative_output_dir(_required_string(payload, "output_dir"))
    runs_payload = _required_list(payload, "runs")
    runs = tuple(_run_config(item) for item in runs_payload)
    _validate_run_names(runs)

    dataset = _required_string(payload, "dataset")
    if dataset != "batadal":
        raise ValueError("BATADAL ablation config dataset must be 'batadal'")

    return LstmAblationConfig(
        experiment_name=_required_string(payload, "experiment_name"),
        dataset=dataset,
        seed=_positive_integer(payload, "seed", allow_zero=True),
        limit=_positive_integer(payload, "limit"),
        output_dir=output_dir,
        runs=runs,
    )


def _run_config(payload: object) -> AblationRunConfig:
    if not isinstance(payload, dict):
        raise ValueError("each ablation run must be a JSON object")

    return AblationRunConfig(
        name=_required_string(payload, "name"),
        context_length=_positive_integer(payload, "context_length"),
        hidden_size=_positive_integer(payload, "hidden_size"),
        max_epochs=_positive_integer(payload, "max_epochs"),
    )


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ablation config field '{field}' must be a non-empty string")
    return value


def _required_list(payload: dict[str, Any], field: str) -> list[object]:
    value = payload.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"ablation config field '{field}' must be a non-empty list")
    return value


def _positive_integer(
    payload: dict[str, Any], field: str, *, allow_zero: bool = False
) -> int:
    value = payload.get(field)
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"ablation config field '{field}' must be an integer >= {minimum}")
    return value


def _relative_output_dir(value: str) -> Path:
    output_dir = Path(value)
    is_absolute_path = (
        value.startswith(("/", "\\"))
        or output_dir.is_absolute()
        or PureWindowsPath(value).is_absolute()
    )
    if is_absolute_path or ".." in output_dir.parts:
        raise ValueError("ablation output_dir must be repository-relative")
    if not output_dir.parts or output_dir.parts[0] != "results":
        raise ValueError("ablation output_dir must be inside results/")
    return output_dir


def _validate_run_names(runs: tuple[AblationRunConfig, ...]) -> None:
    names = [run.name for run in runs]
    if len(names) != len(set(names)):
        raise ValueError("ablation run names must be unique")
