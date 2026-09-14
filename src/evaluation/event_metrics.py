"""Event-level metrics for binary time-series attack labels."""

import numpy as np


def event_detection_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    """Return one hit per contiguous attack event and its first-alarm delay."""
    _validate_binary_series(labels, predictions)
    event_starts = np.flatnonzero(np.diff(np.concatenate(([0], labels))) == 1)
    delays: list[int] = []

    for start in event_starts:
        end = start
        while end < len(labels) and labels[end] == 1:
            end += 1
        hits = np.flatnonzero(predictions[start:end])
        if len(hits):
            delays.append(int(hits[0]))

    detected_events = len(delays)
    n_events = len(event_starts)
    event_recall = detected_events / n_events if n_events else 0.0
    median_delay = float(np.median(delays)) if delays else None
    return {
        "n_events": n_events,
        "detected_events": detected_events,
        "event_recall": event_recall,
        "median_detection_delay": median_delay,
    }


def _validate_binary_series(labels: np.ndarray, predictions: np.ndarray) -> None:
    if labels.ndim != 1 or predictions.ndim != 1:
        raise ValueError("labels and predictions must be one-dimensional")
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("labels and predictions must be binary")
