"""LOCKED contract for the selected BATADAL injected-recovery pilot."""

from pathlib import Path

from src.experiments.injected_recovery_config import (
    DEFAULT_INJECTED_RECOVERY_CONFIG_PATH,
    load_injected_recovery_config,
)


def test_versioned_injected_recovery_pilot_uses_selected_checkpoint_and_holdout_window() -> None:
    config = load_injected_recovery_config(DEFAULT_INJECTED_RECOVERY_CONFIG_PATH)

    assert config.dataset == "batadal"
    assert config.source_split == "val"
    assert config.limit == 10_000
    assert config.calibration_length == 2_000
    assert config.threshold_quantile == 0.99
    assert config.checkpoint_path == Path(
        "results/lstm_forecaster/batadal_ablation/"
        "context-32_hidden-128_epochs-10/batadal/checkpoints/best.pt"
    )
    assert not config.checkpoint_path.is_absolute()
    assert not config.output_dir.is_absolute()
    assert [scenario.family for scenario in config.scenarios] == [
        "step",
        "ramp",
        "periodic",
        "step",
    ]
    assert any(len(scenario.sensors) > 1 for scenario in config.scenarios)
    assert all(scenario.start >= config.calibration_length for scenario in config.scenarios)
    assert all(scenario.start + scenario.duration <= config.limit for scenario in config.scenarios)
