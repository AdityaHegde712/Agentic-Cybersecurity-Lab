"""LOCKED contract for calibration and temporal-support recovery diagnostics."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.injected_recovery_diagnostic import (
    attacked_sensor_trace,
    clean_window_activation_summary,
    scenario_phase_activation_summary,
)


def test_clean_window_summary_exposes_calibration_drift() -> None:
    residual = np.array(
        [
            [0.2, 0.2],
            [0.2, 0.2],
            [2.0, 0.2],
            [2.0, 2.0],
        ]
    )

    summary = clean_window_activation_summary(
        residual,
        sensor_scale=np.array([1.0, 1.0]),
        thresholds=np.array([1.0, 1.0]),
        target_indices=np.array([10, 11, 12, 13]),
        window_length=2,
        calibration_end=12,
    )

    assert list(summary["window_role"]) == ["calibration", "assessment"]
    assert list(summary["active_sensor_fraction"]) == pytest.approx([0.0, 0.75])
    assert list(summary["active_timestep_fraction"]) == pytest.approx([0.0, 1.0])


def test_scenario_phase_summary_is_limited_to_configured_attack_sensors() -> None:
    residual = np.array(
        [
            [0.0, 5.0],
            [0.0, 5.0],
            [3.0, 5.0],
            [3.0, 5.0],
            [0.0, 5.0],
            [0.0, 5.0],
        ]
    )

    summary = scenario_phase_activation_summary(
        residual,
        sensor_scale=np.array([1.0, 1.0]),
        thresholds=np.array([1.0, 1.0]),
        target_indices=np.arange(10, 16),
        attack_sensors=(0,),
        event_start=12,
        event_end=14,
    )

    assert list(summary["phase"]) == ["pre_attack", "attack", "post_attack"]
    assert list(summary["active_sensor_fraction"]) == pytest.approx([0.0, 1.0, 0.0])
    assert list(summary["active_timestep_fraction"]) == pytest.approx([0.0, 1.0, 0.0])


def test_attacked_sensor_trace_aggregates_only_attack_support() -> None:
    trace = attacked_sensor_trace(
        residual=np.array([[1.0, 9.0], [3.0, 9.0]]),
        true_delta=np.array([[2.0, 0.0], [4.0, 0.0]]),
        sensor_scale=np.array([1.0, 1.0]),
        target_indices=np.array([20, 21]),
        attack_sensors=(0,),
    )

    expected = pd.DataFrame(
        {
            "target_index": [20, 21],
            "mean_signed_residual_standard_deviations": [1.0, 3.0],
            "mean_signed_true_delta_standard_deviations": [2.0, 4.0],
        }
    )
    pd.testing.assert_frame_equal(trace, expected)
