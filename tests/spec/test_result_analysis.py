"""LOCKED contract for bounded baseline-result visualization inputs."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.result_analysis import (
    downsample_trace,
    score_distribution_frame,
    validate_summary,
)


def test_downsample_trace_preserves_every_attack_row_and_trace_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "label": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            "cusum": np.arange(10, dtype=float),
        }
    )

    sampled = downsample_trace(frame, max_points=5)

    assert 0 in sampled.index
    assert 9 in sampled.index
    assert {2, 6}.issubset(sampled.index)
    assert len(sampled) <= 5


def test_downsample_trace_rejects_invalid_label_contract() -> None:
    with pytest.raises(ValueError, match="label"):
        downsample_trace(pd.DataFrame({"cusum": [1.0, 2.0]}), max_points=10)
    with pytest.raises(ValueError, match="positive"):
        downsample_trace(pd.DataFrame({"label": [0, 1]}), max_points=0)


def test_validate_summary_requires_baseline_evaluation_fields() -> None:
    valid = pd.DataFrame(
        {
            "dataset": ["hai"],
            "baseline": ["cusum"],
            "point_fpr": [0.1],
            "point_tpr": [0.9],
            "event_recall": [1.0],
        }
    )

    validate_summary(valid)

    with pytest.raises(ValueError, match="event_recall"):
        validate_summary(valid.drop(columns="event_recall"))


def test_score_distribution_frame_requires_and_returns_aggregate_shift_fields() -> None:
    summary = pd.DataFrame(
        {
            "dataset": ["hai"],
            "baseline": ["residual_ewma"],
            "validation_median": [1.0],
            "test_normal_median": [1.5],
            "test_normal_q99": [2.0],
        }
    )

    diagnostics = score_distribution_frame(summary)

    assert diagnostics.equals(summary)
    with pytest.raises(ValueError, match="test_normal_q99"):
        score_distribution_frame(summary.drop(columns="test_normal_q99"))
