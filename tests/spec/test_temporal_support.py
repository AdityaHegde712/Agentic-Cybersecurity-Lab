"""LOCKED contract for label-free temporal support selection."""

import numpy as np
import json
from pathlib import Path

from scripts.run_temporal_support_recovery import manifest_config
from src.evaluation.injected_recovery import InjectionScenario
from src.experiments.injected_recovery_config import InjectedRecoveryConfig
from src.evaluation.temporal_support import temporal_support_mask


def test_temporal_support_keeps_a_sustained_hysteresis_segment() -> None:
    residual = np.array([[0.0], [2.0], [1.2], [1.1], [0.0]])

    support = temporal_support_mask(
        residual,
        sensor_scale=np.array([1.0]),
        start_thresholds=np.array([1.5]),
        continuation_thresholds=np.array([1.0]),
        minimum_duration=3,
    )

    assert support[:, 0].tolist() == [False, True, True, True, False]


def test_temporal_support_rejects_an_isolated_high_residual_spike() -> None:
    residual = np.array([[0.0], [2.0], [0.0], [0.0]])

    support = temporal_support_mask(
        residual,
        sensor_scale=np.array([1.0]),
        start_thresholds=np.array([1.5]),
        continuation_thresholds=np.array([1.0]),
        minimum_duration=2,
    )

    assert not support.any()


def test_temporal_support_requires_a_high_threshold_seed() -> None:
    residual = np.array([[1.1], [1.2], [1.3]])

    support = temporal_support_mask(
        residual,
        sensor_scale=np.array([1.0]),
        start_thresholds=np.array([1.5]),
        continuation_thresholds=np.array([1.0]),
        minimum_duration=3,
    )

    assert not support.any()


def test_manifest_config_converts_repository_paths_to_json_strings() -> None:
    config = InjectedRecoveryConfig(
        dataset="batadal",
        source_split="val",
        checkpoint_path=Path("results/checkpoint.pt"),
        output_dir=Path("results/output"),
        limit=100,
        calibration_length=20,
        threshold_quantile=0.99,
        scenarios=(
            InjectionScenario(
                attack_id="step",
                family="step",
                start=30,
                duration=10,
                magnitude_standard_deviations=3.0,
                sensors=(0,),
            ),
        ),
    )

    payload = manifest_config(config)

    assert payload["checkpoint_path"] == "results/checkpoint.pt"
    assert payload["output_dir"] == "results/output"
    json.dumps(payload)
