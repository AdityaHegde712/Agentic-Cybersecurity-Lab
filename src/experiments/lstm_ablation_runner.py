"""Plan and describe reproducible LSTM ablation runs."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from src.experiments.lstm_ablation_config import LstmAblationConfig
from src.models.lstm_forecaster import ForecasterConfig


@dataclass(frozen=True)
class PlannedAblationRun:
    """One executable forecaster configuration with an isolated result directory."""

    name: str
    output_dir: Path
    forecaster_config: ForecasterConfig


def plan_ablation_runs(config: LstmAblationConfig) -> tuple[PlannedAblationRun, ...]:
    """Translate a versioned experiment contract into named forecaster runs."""
    return tuple(
        PlannedAblationRun(
            name=run.name,
            output_dir=config.output_dir / run.name,
            forecaster_config=ForecasterConfig(
                context_length=run.context_length,
                hidden_size=run.hidden_size,
                max_epochs=run.max_epochs,
                seed=config.seed,
            ),
        )
        for run in config.runs
    )


def write_ablation_manifest(config: LstmAblationConfig, output_root: Path) -> Path:
    """Write portable experiment provenance before any model training begins."""
    runs = plan_ablation_runs(config)
    manifest = {
        "experiment_name": config.experiment_name,
        "dataset": config.dataset,
        "seed": config.seed,
        "limit": config.limit,
        "output_dir": str(config.output_dir),
        "runs": [
            {
                "name": run.name,
                "output_dir": str(run.output_dir),
                **asdict(run.forecaster_config),
            }
            for run in runs
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
