"""LOCKED contract for a reproducible named LSTM ablation runner."""

import json
from pathlib import Path

from src.experiments.lstm_ablation_config import (
    DEFAULT_ABLATION_CONFIG_PATH,
    load_ablation_config,
)
from src.experiments.lstm_ablation_runner import (
    plan_ablation_runs,
    write_ablation_manifest,
)


def test_ablation_run_plan_preserves_named_configurations_and_isolated_outputs() -> None:
    config = load_ablation_config(DEFAULT_ABLATION_CONFIG_PATH)

    runs = plan_ablation_runs(config)

    assert [run.name for run in runs] == [
        "context-16_hidden-32_epochs-5",
        "context-32_hidden-64_epochs-5",
        "context-64_hidden-64_epochs-10",
        "context-32_hidden-128_epochs-10",
    ]
    assert [run.output_dir for run in runs] == [
        Path("results/lstm_forecaster/batadal_ablation") / run.name
        for run in config.runs
    ]
    assert all(not run.output_dir.is_absolute() for run in runs)
    assert [run.forecaster_config.seed for run in runs] == [7, 7, 7, 7]
    assert [run.forecaster_config.context_length for run in runs] == [16, 32, 64, 32]
    assert [run.forecaster_config.hidden_size for run in runs] == [32, 64, 64, 128]
    assert [run.forecaster_config.max_epochs for run in runs] == [5, 5, 10, 10]


def test_ablation_manifest_records_shared_parameters_and_relative_run_paths(
    tmp_path: Path,
) -> None:
    config = load_ablation_config(DEFAULT_ABLATION_CONFIG_PATH)

    manifest_path = write_ablation_manifest(config, tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path == tmp_path / "manifest.json"
    assert manifest["experiment_name"] == "batadal_lstm_ablation"
    assert manifest["dataset"] == "batadal"
    assert manifest["seed"] == 7
    assert manifest["limit"] == 10_000
    assert [run["name"] for run in manifest["runs"]] == [run.name for run in config.runs]
    assert [run["output_dir"] for run in manifest["runs"]] == [
        str(Path("results/lstm_forecaster/batadal_ablation") / run.name)
        for run in config.runs
    ]
