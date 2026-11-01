"""LOCKED contract for declared support in standardized attack injection."""

import numpy as np

from src.evaluation.attack_metrics import additive_recovery_metrics
from src.evaluation.injected_recovery import (
    InjectionScenario,
    declared_support_mask,
    inject_standardized_attack,
)
from src.models.lstm_forecaster import FeatureNormalizer


def test_standardized_injection_preserves_declared_support_and_exact_delta() -> None:
    clean = np.array(
        [[10.1, 100.3], [10.2, 100.4], [10.3, 100.5], [10.4, 100.6]],
        dtype=np.float64,
    )
    normalizer = FeatureNormalizer(
        mean=np.array([10.0, 100.0], dtype=np.float32),
        scale=np.array([0.3, 1.7], dtype=np.float32),
    )
    scenario = InjectionScenario(
        attack_id="single-step",
        family="step",
        start=1,
        duration=2,
        magnitude_standard_deviations=-2.0,
        sensors=(1,),
    )

    injected = inject_standardized_attack(clean, normalizer, scenario)
    support = declared_support_mask(clean.shape, scenario)

    assert np.array_equal(injected.support, support)
    assert np.array_equal(injected.observed, clean + injected.delta)
    assert np.all(injected.delta[~support] == 0.0)
    assert np.all(injected.delta[support] == -2.0 * normalizer.scale[1])


def test_explicit_support_mask_ignores_numerical_residue_outside_attack() -> None:
    true_delta = np.array([[1.0e-8], [4.0], [4.0], [-1.0e-8]])
    estimated_delta = np.array([[0.0], [4.0], [4.0], [0.0]])
    support = np.array([[False], [True], [True], [False]])

    metrics = additive_recovery_metrics(
        estimated_delta,
        true_delta,
        true_support=support,
    )

    assert metrics["support_f1"] == 1.0
    assert metrics["magnitude_mae_on_support"] == 0.0
