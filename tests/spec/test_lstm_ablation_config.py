"""LOCKED contract for the repository-tracked BATADAL LSTM ablation grid."""

import json
from pathlib import Path

import pytest

from src.experiments.lstm_ablation_config import (
    DEFAULT_ABLATION_CONFIG_PATH,
    load_ablation_config,
)


def test_versioned_batadal_ablation_contract_is_deterministic_and_relative() -> None:
    config = load_ablation_config(DEFAULT_ABLATION_CONFIG_PATH)

    assert config.experiment_name == "batadal_lstm_ablation"
    assert config.dataset == "batadal"
    assert config.seed == 7
    assert config.limit == 10_000
    assert config.output_dir == Path("results/lstm_forecaster/batadal_ablation")
    assert not config.output_dir.is_absolute()
    assert [(run.context_length, run.hidden_size, run.max_epochs) for run in config.runs] == [
        (16, 32, 5),
        (32, 64, 5),
        (64, 64, 10),
        (32, 128, 10),
    ]


def test_versioned_batadal_ablation_run_names_are_unique_and_complete() -> None:
    config = load_ablation_config(DEFAULT_ABLATION_CONFIG_PATH)

    names = [run.name for run in config.runs]

    assert len(names) == len(set(names))
    assert all(run.context_length > 0 for run in config.runs)
    assert all(run.hidden_size > 0 for run in config.runs)
    assert all(run.max_epochs > 0 for run in config.runs)


def test_ablation_config_rejects_an_absolute_output_directory(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid_ablation.json"
    invalid_path.write_text(
        json.dumps(
            {
                "experiment_name": "batadal_lstm_ablation",
                "dataset": "batadal",
                "seed": 7,
                "limit": 10_000,
                "output_dir": "/machine/specific/results",
                "runs": [
                    {
                        "name": "baseline",
                        "context_length": 32,
                        "hidden_size": 64,
                        "max_epochs": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative"):
        load_ablation_config(invalid_path)
