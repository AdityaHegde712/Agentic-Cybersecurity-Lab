"""LOCKED contract for temporal additive-attack ground truth.

These tests define the benchmark boundary for observation-only spoofing:
``observed = clean + delta``.  Real ICS labels may exercise detection and
localization, but only an injected benchmark supplies ``delta`` for recovery
metrics.
"""

import numpy as np
import pytest

from src.attacks.additive import AdditiveAttackSpec, inject_additive_attack
from src.evaluation.attack_metrics import additive_recovery_metrics


def test_step_injection_preserves_clean_input_and_exposes_ground_truth() -> None:
    clean = np.zeros((10, 3), dtype=np.float64)
    spec = AdditiveAttackSpec(
        attack_id="step-1",
        family="step",
        start=3,
        duration=4,
        magnitude=2.5,
        sensors=(0, 2),
    )

    result = inject_additive_attack(clean, spec)

    assert np.array_equal(clean, np.zeros((10, 3)))
    assert np.array_equal(result.observed, clean + result.delta)
    assert result.delta.shape == clean.shape
    assert np.all(result.delta[:3] == 0.0)
    assert np.all(result.delta[3:7, (0, 2)] == 2.5)
    assert np.all(result.delta[3:7, 1] == 0.0)
    assert np.all(result.delta[7:] == 0.0)
    assert result.event == {
        "attack_id": "step-1",
        "family": "step",
        "start": 3,
        "end": 7,
        "sensors": [0, 2],
    }


def test_ramp_and_periodic_injections_are_temporal_and_finite() -> None:
    clean = np.zeros((12, 2), dtype=np.float64)
    ramp = AdditiveAttackSpec(
        attack_id="ramp-1",
        family="ramp",
        start=2,
        duration=4,
        magnitude=3.0,
        sensors=(1,),
    )
    periodic = AdditiveAttackSpec(
        attack_id="periodic-1",
        family="periodic",
        start=4,
        duration=6,
        magnitude=2.0,
        sensors=(0,),
        period=4,
    )

    ramp_result = inject_additive_attack(clean, ramp)
    periodic_result = inject_additive_attack(clean, periodic)

    assert np.array_equal(ramp_result.delta[2:6, 1], np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.all(ramp_result.delta[:, 0] == 0.0)
    assert np.all(periodic_result.delta[:4] == 0.0)
    assert np.all(periodic_result.delta[10:] == 0.0)
    assert np.isclose(np.max(periodic_result.delta[:, 0]), 2.0)


@pytest.mark.parametrize(
    "spec",
    [
        AdditiveAttackSpec("bad", "step", -1, 2, 1.0, (0,)),
        AdditiveAttackSpec("bad", "step", 2, 0, 1.0, (0,)),
        AdditiveAttackSpec("bad", "step", 9, 2, 1.0, (0,)),
        AdditiveAttackSpec("bad", "step", 2, 2, 1.0, (3,)),
        AdditiveAttackSpec("bad", "unknown", 2, 2, 1.0, (0,)),
        AdditiveAttackSpec("bad", "periodic", 2, 2, 1.0, (0,)),
    ],
)
def test_invalid_attack_specs_fail_loudly(spec: AdditiveAttackSpec) -> None:
    with pytest.raises(ValueError):
        inject_additive_attack(np.zeros((10, 3)), spec)


def test_recovery_metrics_score_support_magnitude_and_sign_separately() -> None:
    true_delta = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, -3.0], [0.0, 0.0]]
    )
    estimated_delta = np.array(
        [[0.0, 0.0], [2.0, 0.5], [1.0, -3.0], [0.0, 0.0]]
    )

    metrics = additive_recovery_metrics(estimated_delta, true_delta)

    assert metrics["support_precision"] == pytest.approx(0.75)
    assert metrics["support_recall"] == pytest.approx(1.0)
    assert metrics["support_f1"] == pytest.approx(6.0 / 7.0)
    assert metrics["support_iou"] == pytest.approx(0.75)
    assert metrics["magnitude_mae_on_support"] == pytest.approx(1.0 / 3.0)
    assert metrics["sign_accuracy_on_support"] == pytest.approx(1.0)


def test_recovery_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        additive_recovery_metrics(np.zeros((3, 2)), np.zeros((3, 3)))
