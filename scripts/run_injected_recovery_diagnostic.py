"""Generate compact calibration and temporal-support diagnostic artifacts."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_injected_recovery import (
    calibrate_sensor_thresholds,
    load_clean_validation_values,
)
from src.evaluation.injected_recovery import inject_standardized_attack
from src.evaluation.injected_recovery_diagnostic import (
    attacked_sensor_trace,
    clean_window_activation_summary,
    scenario_phase_activation_summary,
)
from src.experiments.injected_recovery_config import (
    DEFAULT_INJECTED_RECOVERY_CONFIG_PATH,
    load_injected_recovery_config,
)
from src.models.lstm_forecaster import predict_residuals


def args() -> argparse.Namespace:
    """Parse portable diagnostic settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_INJECTED_RECOVERY_CONFIG_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-length", type=int, default=128)
    return parser.parse_args()


def run(config_path: Path, device: str, window_length: int) -> Path:
    """Generate clean-drift and injected temporal-support diagnostic tables."""
    if window_length < 1:
        raise ValueError("window_length must be positive")
    config = load_injected_recovery_config(config_path)
    clean_values = load_clean_validation_values(config)
    forecaster, thresholds = calibrate_sensor_thresholds(
        clean_values,
        config.calibration_length,
        config.threshold_quantile,
        PROJECT_ROOT / config.checkpoint_path,
        device,
    )
    clean_prediction = predict_residuals(forecaster, clean_values, device=device)
    clean_summary = clean_window_activation_summary(
        clean_prediction.residual,
        sensor_scale=forecaster.normalizer.scale,
        thresholds=thresholds,
        target_indices=clean_prediction.target_indices,
        window_length=window_length,
        calibration_end=config.calibration_length,
    )

    phase_summaries: list[pd.DataFrame] = []
    traces: dict[str, pd.DataFrame] = {}
    for scenario in config.scenarios:
        injected = inject_standardized_attack(clean_values, forecaster.normalizer, scenario)
        prediction = predict_residuals(forecaster, injected.observed, device=device)
        true_delta = injected.delta[prediction.target_indices]
        phase_summary = scenario_phase_activation_summary(
            prediction.residual,
            sensor_scale=forecaster.normalizer.scale,
            thresholds=thresholds,
            target_indices=prediction.target_indices,
            attack_sensors=scenario.sensors,
            event_start=scenario.start,
            event_end=scenario.start + scenario.duration,
        )
        phase_summaries.append(
            phase_summary.assign(attack_id=scenario.attack_id, family=scenario.family)
        )
        traces[scenario.attack_id] = attacked_sensor_trace(
            prediction.residual,
            true_delta,
            sensor_scale=forecaster.normalizer.scale,
            target_indices=prediction.target_indices,
            attack_sensors=scenario.sensors,
        )

    diagnostic_dir = PROJECT_ROOT / config.output_dir / "diagnostic"
    trace_dir = diagnostic_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    clean_summary.to_csv(diagnostic_dir / "clean_window_activation_summary.csv", index=False)
    phase_summary = pd.concat(phase_summaries, ignore_index=True)
    phase_summary.to_csv(diagnostic_dir / "scenario_phase_activation_summary.csv", index=False)
    for attack_id, trace in traces.items():
        trace.to_csv(trace_dir / f"{attack_id}.csv", index=False)
    (diagnostic_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    **asdict(config),
                    "checkpoint_path": str(config.checkpoint_path),
                    "output_dir": str(config.output_dir),
                },
                "window_length": window_length,
                "sensor_threshold_standard_deviations": thresholds.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return diagnostic_dir


def main() -> None:
    """Run the diagnostic with the selected CPU or CUDA device."""
    settings = args()
    config_path = settings.config if settings.config.is_absolute() else PROJECT_ROOT / settings.config
    device = "cuda" if settings.device == "auto" and torch.cuda.is_available() else (
        "cpu" if settings.device == "auto" else settings.device
    )
    diagnostic_dir = run(config_path, device, settings.window_length)
    print(f"Saved injected recovery diagnostic artifacts to {diagnostic_dir} on {device}", flush=True)


if __name__ == "__main__":
    main()
