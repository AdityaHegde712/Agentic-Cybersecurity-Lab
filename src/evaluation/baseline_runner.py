"""Leakage-safe adapters from canonical datasets to statistical baselines."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.registry import get_meta, load_dataset
from src.evaluation.event_metrics import event_detection_metrics
from src.evaluation.metrics import false_positive_rate, true_positive_rate
from src.evaluation.statistical_baselines import statistical_baseline_scores


@dataclass(frozen=True)
class BaselineRun:
    """Per-timestep scores and per-method evaluation for one canonical dataset."""

    scores: pd.DataFrame
    summary: pd.DataFrame


def summarize_score_series(
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    threshold_quantile: float,
) -> dict[str, float | int | None]:
    """Calibrate on validation-normal scores, then evaluate held-out test data."""
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be in (0, 1)")
    if validation_scores.ndim != 1 or test_scores.ndim != 1 or test_labels.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if test_scores.shape != test_labels.shape:
        raise ValueError("test_scores and test_labels must have the same shape")
    if len(validation_scores) == 0:
        raise ValueError("validation_scores must not be empty")
    if not np.isfinite(validation_scores).all() or not np.isfinite(test_scores).all():
        raise ValueError("scores must contain only finite values")

    threshold = float(np.quantile(validation_scores, threshold_quantile))
    predictions = (test_scores > threshold).astype(np.int8)
    event_metrics = event_detection_metrics(test_labels.astype(np.int8), predictions)
    return {
        "threshold": threshold,
        "point_fpr": false_positive_rate(test_labels, predictions),
        "point_tpr": true_positive_rate(test_labels, predictions),
        **event_metrics,
    }


def run_dataset_statistical_baselines(
    dataset: str,
    *,
    limit: int | None = None,
    threshold_quantile: float = 0.99,
    ewma_alpha: float = 0.2,
    pca_components: int = 5,
) -> BaselineRun:
    """Fit baseline scores on normal train data and evaluate the held-out test split."""
    train, validation, test, sensors = _load_canonical_splits(dataset, limit)
    train_normal = train.loc[train["label"] == 0, sensors]
    validation_normal = validation.loc[validation["label"] == 0, sensors]
    test_values = test.loc[:, sensors]
    test_labels = test["label"].to_numpy(dtype=np.int8)

    if len(train_normal) == 0 or len(validation_normal) == 0:
        raise ValueError(f"{dataset} requires normal rows in train and validation splits")

    train_values, validation_values, test_values, sensors = prepare_feature_matrices(
        train_normal,
        validation_normal,
        test_values,
        sensors,
    )

    usable_components = min(pca_components, len(sensors))
    validation_scores = statistical_baseline_scores(
        train_values,
        validation_values,
        ewma_alpha=ewma_alpha,
        pca_components=usable_components,
    )
    test_scores = statistical_baseline_scores(
        train_values,
        test_values,
        ewma_alpha=ewma_alpha,
        pca_components=usable_components,
    )

    score_frame = test.loc[:, ["timestamp", "label", "attack_id"]].copy()
    summaries: list[dict[str, object]] = []
    for baseline, test_score in test_scores.items():
        score_frame[baseline] = test_score
        metrics = summarize_score_series(
            validation_scores[baseline],
            test_score,
            test_labels,
            threshold_quantile=threshold_quantile,
        )
        summaries.append({"dataset": dataset, "baseline": baseline, **metrics})

    return BaselineRun(scores=score_frame, summary=pd.DataFrame(summaries))


def _load_canonical_splits(
    dataset: str,
    limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    meta = get_meta(dataset)
    sensors = list(meta["sensor_columns"])
    train = _limit_rows(load_dataset(dataset, "train"), limit)
    validation = _limit_rows(load_dataset(dataset, "val"), limit)
    test = _limit_rows(load_dataset(dataset, "test"), limit)
    return train, validation, test, sensors


def prepare_feature_matrices(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    sensors: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Drop train-all-null columns and median-impute from normal training data.

    Canonical Parquet remains raw. This function is the explicit modeling
    boundary for instrumentation gaps such as WADI's unlogged channels.
    """
    train_features = train.loc[:, sensors].copy()
    validation_features = validation.loc[:, sensors].copy()
    test_features = test.loc[:, sensors].copy()
    usable_sensors = [sensor for sensor in sensors if not train_features[sensor].isna().all()]
    if not usable_sensors:
        raise ValueError("no usable sensors remain after dropping all-null training columns")

    train_features = train_features.loc[:, usable_sensors]
    validation_features = validation_features.loc[:, usable_sensors]
    test_features = test_features.loc[:, usable_sensors]
    train_medians = train_features.median(axis=0)

    train_filled = train_features.fillna(train_medians)
    validation_filled = validation_features.fillna(train_medians)
    test_filled = test_features.fillna(train_medians)
    return (
        train_filled.to_numpy(dtype=np.float64),
        validation_filled.to_numpy(dtype=np.float64),
        test_filled.to_numpy(dtype=np.float64),
        usable_sensors,
    )


def _limit_rows(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    return frame if limit is None else frame.iloc[:limit].copy()
