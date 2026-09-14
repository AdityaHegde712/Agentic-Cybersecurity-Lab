"""LOCKED contract for one-step temporal-residual statistical baselines."""

import numpy as np

from src.evaluation.statistical_baselines import (
    persistence_residuals,
    temporal_residual_baseline_scores,
)


def _normal_series() -> np.ndarray:
    rng = np.random.default_rng(19)
    latent = rng.normal(size=(240, 1))
    noise = rng.normal(scale=0.05, size=(240, 3))
    return latent * np.array([[1.0, 0.6, -0.3]]) + noise


def test_persistence_residuals_preserve_time_alignment_with_preceding_state() -> None:
    values = np.array([[2.0, 4.0], [5.0, 9.0], [9.0, 16.0]])

    residuals = persistence_residuals(values, preceding_values=np.array([1.0, 1.0]))

    assert np.array_equal(residuals, np.array([[1.0, 3.0], [3.0, 5.0], [4.0, 7.0]]))


def test_temporal_residual_scores_are_invariant_to_constant_operating_level_offset() -> None:
    train = _normal_series()
    test = _normal_series()[:80]
    offset = np.array([500.0, -200.0, 75.0])

    reference = temporal_residual_baseline_scores(train, test, pca_components=1)
    offset_scores = temporal_residual_baseline_scores(
        train + offset,
        test + offset,
        pca_components=1,
    )

    assert set(reference) == {"residual_cusum", "residual_ewma", "residual_pca_spe"}
    for baseline in reference:
        assert np.allclose(reference[baseline], offset_scores[baseline])


def test_temporal_residual_scores_increase_at_a_large_additive_step_onset() -> None:
    train = _normal_series()
    test = _normal_series()[:80]
    attacked = test.copy()
    attacked[40:, 0] += 20.0

    clean_scores = temporal_residual_baseline_scores(train, test, pca_components=1)
    attacked_scores = temporal_residual_baseline_scores(train, attacked, pca_components=1)

    for baseline in clean_scores:
        clean_peak = clean_scores[baseline][40:42].max()
        attack_peak = attacked_scores[baseline][40:42].max()
        assert attack_peak > clean_peak * 5.0, baseline
