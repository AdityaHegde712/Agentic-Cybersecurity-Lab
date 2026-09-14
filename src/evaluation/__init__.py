"""Evaluation metrics and benchmark helpers."""

from .statistical_baselines import statistical_baseline_scores
from .event_metrics import event_detection_metrics
from .baseline_runner import (
    calibrate_blocked_threshold,
    prepare_feature_matrices,
    run_dataset_statistical_baselines,
    summarize_score_series,
)
from .result_analysis import downsample_trace, validate_summary

__all__ = [
    "event_detection_metrics",
    "calibrate_blocked_threshold",
    "prepare_feature_matrices",
    "downsample_trace",
    "run_dataset_statistical_baselines",
    "statistical_baseline_scores",
    "summarize_score_series",
    "validate_summary",
]
