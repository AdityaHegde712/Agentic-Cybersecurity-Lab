"""Bounded visualization helpers for baseline-result artifacts."""

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = {"dataset", "baseline", "point_fpr", "point_tpr", "event_recall"}


def validate_summary(summary: pd.DataFrame) -> None:
    """Fail loudly when a baseline summary lacks required evaluation fields."""
    missing = SUMMARY_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(f"summary is missing required columns: {', '.join(sorted(missing))}")


def downsample_trace(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Bound a trace while retaining attack rows and first/last observations."""
    if "label" not in frame.columns:
        raise ValueError("trace frame must contain a label column")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if len(frame) <= max_points:
        return frame.copy()

    attack_positions = np.flatnonzero(frame["label"].to_numpy() == 1)
    retained_positions = {0, len(frame) - 1, *attack_positions.tolist()}
    remaining_budget = max(0, max_points - len(retained_positions))
    normal_positions = np.flatnonzero(frame["label"].to_numpy() == 0)
    if remaining_budget and len(normal_positions):
        sampled_normal = np.linspace(0, len(normal_positions) - 1, remaining_budget, dtype=int)
        retained_positions.update(normal_positions[sampled_normal].tolist())

    return frame.iloc[sorted(retained_positions)].copy()
