"""Evaluate a selected forecaster against known injected additive attacks."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import get_meta, load_dataset
from src.evaluation.baseline_runner import prepare_feature_matrices
from src.evaluation.injected_recovery import (
    inject_standardized_attack,
    temporal_recovery_metrics,
    threshold_estimated_delta,
)
from src.experiments.injected_recovery_config import (
    DEFAULT_INJECTED_RECOVERY_CONFIG_PATH,
    InjectedRecoveryConfig,
    load_injected_recovery_config,
)
from src.models.lstm_forecaster import load_forecaster_checkpoint, predict_residuals


def args() -> argparse.Namespace:
    """Parse the portable injection-plan path and execution device."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_INJECTED_RECOVERY_CONFIG_PATH,
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_clean_validation_values(config: InjectedRecoveryConfig) -> np.ndarray:
    """Load a held-out normal BATADAL sequence without using attack labels as truth."""
    if config.dataset != "batadal" or config.source_split != "val":
        raise ValueError("the current pilot supports BATADAL validation data only")

    sensors = list(get_meta(config.dataset)["sensor_columns"])
    train = load_dataset(config.dataset, "train")
    validation = load_dataset(config.dataset, config.source_split)
    train_normal = train.loc[train.label == 0, sensors]
    validation_normal = validation.loc[validation.label == 0, sensors]
    _, clean_values, _, _ = prepare_feature_matrices(
        train_normal,
        validation_normal,
        validation_normal,
        sensors,
    )
    if len(clean_values) < config.limit:
        raise ValueError(
            f"only {len(clean_values)} normal validation rows are available; "
            f"pilot requires {config.limit}"
        )
    return clean_values[: config.limit]


def calibrate_sensor_thresholds(
    clean_values: np.ndarray,
    calibration_length: int,
    threshold_quantile: float,
    checkpoint_path: Path,
    device: str,
) -> tuple[object, np.ndarray]:
    """Calibrate per-sensor thresholds on a clean prefix disjoint from injection windows."""
    forecaster = load_forecaster_checkpoint(checkpoint_path, device=device)
    calibration_prediction = predict_residuals(
        forecaster,
        clean_values[:calibration_length],
        device=device,
    )
    standardized_residual = np.abs(
        calibration_prediction.residual / forecaster.normalizer.scale
    )
    return forecaster, np.quantile(standardized_residual, threshold_quantile, axis=0)


def run(config: InjectedRecoveryConfig, device: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Run every configured injection and return compact aggregate metrics."""
    clean_values = load_clean_validation_values(config)
    checkpoint_path = PROJECT_ROOT / config.checkpoint_path
    forecaster, thresholds = calibrate_sensor_thresholds(
        clean_values,
        config.calibration_length,
        config.threshold_quantile,
        checkpoint_path,
        device,
    )
    records: list[dict[str, object]] = []
    for scenario in config.scenarios:
        injected = inject_standardized_attack(clean_values, forecaster.normalizer, scenario)
        prediction = predict_residuals(forecaster, injected.observed, device=device)
        true_delta = injected.delta[prediction.target_indices]
        thresholded_delta = threshold_estimated_delta(
            prediction.residual,
            forecaster.normalizer.scale,
            thresholds,
        )
        metrics = temporal_recovery_metrics(
            thresholded_delta,
            true_delta,
            target_indices=prediction.target_indices,
            event_start=scenario.start,
            event_end=scenario.start + scenario.duration,
        )
        records.append(
            {
                "attack_id": scenario.attack_id,
                "family": scenario.family,
                "sensor_count": len(scenario.sensors),
                "magnitude_standard_deviations": scenario.magnitude_standard_deviations,
                "duration": scenario.duration,
                **metrics,
            }
        )
    return pd.DataFrame(records), thresholds


def write_artifacts(
    config: InjectedRecoveryConfig,
    summary: pd.DataFrame,
    thresholds: np.ndarray,
) -> None:
    """Persist compact provenance and aggregate evaluation results."""
    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    manifest = {
        **asdict(config),
        "checkpoint_path": str(config.checkpoint_path),
        "output_dir": str(config.output_dir),
        "sensor_threshold_standard_deviations": thresholds.tolist(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Execute the selected checkpoint against the configured known-delta pilot."""
    settings = args()
    config_path = settings.config if settings.config.is_absolute() else PROJECT_ROOT / settings.config
    config = load_injected_recovery_config(config_path)
    device = "cuda" if settings.device == "auto" and torch.cuda.is_available() else (
        "cpu" if settings.device == "auto" else settings.device
    )
    summary, thresholds = run(config, device)
    write_artifacts(config, summary, thresholds)
    print(f"Saved injected recovery artifacts to {config.output_dir} on {device}", flush=True)


if __name__ == "__main__":
    main()
