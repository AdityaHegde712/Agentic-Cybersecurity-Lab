"""Continuous statistical baseline scores for canonical multivariate datasets."""

import numpy as np


def statistical_baseline_scores(
    train: np.ndarray,
    test: np.ndarray,
    *,
    ewma_alpha: float = 0.2,
    pca_components: int = 1,
    cusum_slack: float = 0.5,
    cusum_reset_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    """Return continuous per-timestep anomaly scores fit on normal train data.

    Scores are intentionally unthresholded. Threshold selection belongs to a
    later blocked-time validation phase, preventing test-label leakage during
    baseline generation.
    """
    _validate_inputs(
        train,
        test,
        ewma_alpha,
        pca_components,
        cusum_slack,
        cusum_reset_threshold,
    )
    train_standardized, test_standardized = _standardize_from_train(train, test)

    return {
        "cusum": _cusum_score(test_standardized, cusum_slack, cusum_reset_threshold),
        "ewma": _ewma_score(test_standardized, ewma_alpha),
        "pca_spe": _pca_spe_score(train_standardized, test_standardized, pca_components),
    }


def _validate_inputs(
    train: np.ndarray,
    test: np.ndarray,
    ewma_alpha: float,
    pca_components: int,
    cusum_slack: float,
    cusum_reset_threshold: float | None,
) -> None:
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("train and test must be two-dimensional [time, sensor] arrays")
    if train.shape[0] == 0 or test.shape[0] == 0:
        raise ValueError("train and test must each contain at least one timestep")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train and test must have the same sensor count")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("train and test must contain only finite values")
    if not 0.0 < ewma_alpha <= 1.0:
        raise ValueError("ewma_alpha must be in (0, 1]")
    if not 1 <= pca_components <= train.shape[1]:
        raise ValueError("pca_components must be between 1 and the sensor count")
    if cusum_slack < 0.0:
        raise ValueError("cusum_slack must be non-negative")
    if cusum_reset_threshold is not None and (
        not np.isfinite(cusum_reset_threshold) or cusum_reset_threshold <= 0.0
    ):
        raise ValueError("cusum_reset_threshold must be finite and positive when provided")


def _standardize_from_train(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    stable_scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    return (train - mean) / stable_scale, (test - mean) / stable_scale


def _cusum_score(
    values: np.ndarray,
    slack: float,
    reset_threshold: float | None,
) -> np.ndarray:
    positive = np.zeros(values.shape[1], dtype=np.float64)
    negative = np.zeros(values.shape[1], dtype=np.float64)
    scores = np.empty(values.shape[0], dtype=np.float64)

    for index, row in enumerate(values):
        positive = np.maximum(0.0, positive + row - slack)
        negative = np.maximum(0.0, negative - row - slack)
        score = float(np.maximum(positive, negative).max())
        scores[index] = score
        if reset_threshold is not None and score > reset_threshold:
            positive.fill(0.0)
            negative.fill(0.0)
    return scores


def _ewma_score(values: np.ndarray, alpha: float) -> np.ndarray:
    state = np.zeros(values.shape[1], dtype=np.float64)
    scores = np.empty(values.shape[0], dtype=np.float64)

    for index, row in enumerate(values):
        state = alpha * row + (1.0 - alpha) * state
        scores[index] = float(np.abs(state).max())
    return scores


def _pca_spe_score(
    train: np.ndarray,
    test: np.ndarray,
    components: int,
) -> np.ndarray:
    _, _, right_singular_vectors = np.linalg.svd(train, full_matrices=False)
    basis = right_singular_vectors[:components]
    reconstruction = (test @ basis.T) @ basis
    residual = test - reconstruction
    return np.einsum("ij,ij->i", residual, residual)
