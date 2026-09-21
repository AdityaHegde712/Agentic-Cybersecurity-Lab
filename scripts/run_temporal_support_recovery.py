"""Compare pointwise and temporal-support residual recovery on known injections."""

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

from scripts.run_injected_recovery import calibrate_sensor_thresholds, load_clean_validation_values
from src.evaluation.injected_recovery import (
    inject_standardized_attack,
    temporal_recovery_metrics,
    threshold_estimated_delta,
)
from src.evaluation.temporal_support import apply_temporal_support
from src.experiments.injected_recovery_config import (
    DEFAULT_INJECTED_RECOVERY_CONFIG_PATH,
    load_injected_recovery_config,
)
from src.models.lstm_forecaster import predict_residuals


DEFAULT_OUTPUT_DIR = Path("results/lstm_forecaster/batadal_temporal_support_recovery")


def args() -> argparse.Namespace:
    """Parse portable temporal-support experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_INJECTED_RECOVERY_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--continuation-quantile", type=float, default=0.95)
    parser.add_argument("--minimum-duration", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run(
    config_path: Path,
    output_dir: Path,
    continuation_quantile: float,
    minimum_duration: int,
    device: str,
) -> Path:
    """Evaluate a label-free temporal-support layer over all sensor residuals."""
    if not 0.0 < continuation_quantile < 1.0:
        raise ValueError("continuation_quantile must be strictly between zero and one")
    if minimum_duration < 1:
        raise ValueError("minimum_duration must be positive")
    if output_dir.is_absolute() or ".." in output_dir.parts or output_dir.parts[0] != "results":
        raise ValueError("output_dir must be repository-relative inside results/")

    config = load_injected_recovery_config(config_path)
    clean_values = load_clean_validation_values(config)
    forecaster, start_thresholds = calibrate_sensor_thresholds(
        clean_values,
        config.calibration_length,
        config.threshold_quantile,
        PROJECT_ROOT / config.checkpoint_path,
        device,
    )
    calibration_prediction = predict_residuals(
        forecaster,
        clean_values[: config.calibration_length],
        device=device,
    )
    standardized_calibration = np.abs(
        calibration_prediction.residual / forecaster.normalizer.scale
    )
    continuation_thresholds = np.quantile(
        standardized_calibration,
        continuation_quantile,
        axis=0,
    )

    records: list[dict[str, object]] = []
    for scenario in config.scenarios:
        injected = inject_standardized_attack(clean_values, forecaster.normalizer, scenario)
        prediction = predict_residuals(forecaster, injected.observed, device=device)
        true_delta = injected.delta[prediction.target_indices]
        pointwise = threshold_estimated_delta(
            prediction.residual,
            forecaster.normalizer.scale,
            start_thresholds,
        )
        temporal = apply_temporal_support(
            prediction.residual,
            sensor_scale=forecaster.normalizer.scale,
            start_thresholds=start_thresholds,
            continuation_thresholds=continuation_thresholds,
            minimum_duration=minimum_duration,
        )
        common = {
            "attack_id": scenario.attack_id,
            "family": scenario.family,
            "sensor_count": len(scenario.sensors),
            "magnitude_standard_deviations": scenario.magnitude_standard_deviations,
            "duration": scenario.duration,
        }
        for estimator, estimate in (("pointwise", pointwise), ("temporal_support", temporal)):
            records.append(
                {
                    **common,
                    "estimator": estimator,
                    **temporal_recovery_metrics(
                        estimate,
                        true_delta,
                        target_indices=prediction.target_indices,
                        event_start=scenario.start,
                        event_end=scenario.start + scenario.duration,
                    ),
                }
            )

    artifact_dir = PROJECT_ROOT / output_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(artifact_dir / "summary.csv", index=False)
    (artifact_dir / "manifest.json").write_text(
        json.dumps(
            {
                "injection_config": {**asdict(config), "checkpoint_path": str(config.checkpoint_path)},
                "continuation_quantile": continuation_quantile,
                "minimum_duration": minimum_duration,
                "start_threshold_standard_deviations": start_thresholds.tolist(),
                "continuation_threshold_standard_deviations": continuation_thresholds.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_dir


def main() -> None:
    """Run the support comparison on CPU or CUDA."""
    settings = args()
    config_path = settings.config if settings.config.is_absolute() else PROJECT_ROOT / settings.config
    device = "cuda" if settings.device == "auto" and torch.cuda.is_available() else (
        "cpu" if settings.device == "auto" else settings.device
    )
    artifact_dir = run(
        config_path,
        settings.output_dir,
        settings.continuation_quantile,
        settings.minimum_duration,
        device,
    )
    print(f"Saved temporal-support recovery artifacts to {artifact_dir} on {device}", flush=True)


if __name__ == "__main__":
    main()
