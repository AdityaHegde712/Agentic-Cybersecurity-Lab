"""Evaluation metrics and benchmark helpers."""

from .statistical_baselines import statistical_baseline_scores
from .event_metrics import event_detection_metrics
from .baseline_runner import (
    prepare_feature_matrices,
    run_dataset_statistical_baselines,
    summarize_score_series,
)

__all__ = [
    "event_detection_metrics",
    "prepare_feature_matrices",
    "run_dataset_statistical_baselines",
    "statistical_baseline_scores",
    "summarize_score_series",
]
