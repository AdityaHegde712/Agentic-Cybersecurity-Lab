"""LOCKED contract for bounded baseline-result visualization inputs."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.result_analysis import downsample_trace, validate_summary


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
