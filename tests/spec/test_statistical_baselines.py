"""LOCKED contract for dataset-agnostic statistical baseline scores."""

import numpy as np
import pytest

from src.evaluation.statistical_baselines import statistical_baseline_scores


def _normal_series() -> np.ndarray:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(240, 1))
    noise = rng.normal(scale=0.08, size=(240, 3))
    return latent * np.array([[1.0, 0.7, -0.4]]) + noise


def test_baselines_return_finite_continuous_scores_for_each_timestep() -> None:
    train = _normal_series()
    test = _normal_series()[:80]

    scores = statistical_baseline_scores(train, test, ewma_alpha=0.2, pca_components=1)

    assert set(scores) == {"cusum", "ewma", "pca_spe"}
    for score in scores.values():
        assert score.shape == (len(test),)
        assert np.isfinite(score).all()
        assert np.all(score >= 0.0)


def test_each_baseline_increases_on_a_large_additive_sensor_attack() -> None:
    train = _normal_series()
    test = _normal_series()[:80]
    attacked = test.copy()
    attacked[40:, 0] += 8.0

    clean_scores = statistical_baseline_scores(train, test, ewma_alpha=0.2, pca_components=1)
    attacked_scores = statistical_baseline_scores(train, attacked, ewma_alpha=0.2, pca_components=1)

    for name in attacked_scores:
        clean_mean = clean_scores[name][40:].mean()
        attacked_mean = attacked_scores[name][40:].mean()
        assert attacked_mean > clean_mean * 5.0, name


def test_cusum_resets_after_crossing_its_calibrated_reset_threshold() -> None:
    train = np.array([[-1.0], [1.0]])
    test = np.array([[2.0], [2.0], [0.0]])

    scores = statistical_baseline_scores(
        train,
        test,
        pca_components=1,
        cusum_slack=0.0,
        cusum_reset_threshold=3.0,
    )

    assert np.array_equal(scores["cusum"], np.array([2.0, 4.0, 0.0]))


@pytest.mark.parametrize(
    ("train", "test", "alpha", "components"),
    [
        (np.zeros((2, 2)), np.zeros((2, 3)), 0.2, 1),
        (np.zeros((2, 2)), np.zeros((2, 2)), 0.0, 1),
        (np.zeros((2, 2)), np.zeros((2, 2)), 1.1, 1),
        (np.zeros((2, 2)), np.zeros((2, 2)), 0.2, 0),
        (np.zeros((2, 2)), np.zeros((2, 2)), 0.2, 3),
    ],
)
def test_baselines_reject_invalid_inputs(
    train: np.ndarray,
    test: np.ndarray,
    alpha: float,
    components: int,
) -> None:
    with pytest.raises(ValueError):
        statistical_baseline_scores(train, test, ewma_alpha=alpha, pca_components=components)
