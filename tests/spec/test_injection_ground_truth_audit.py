"""LOCKED contract for aggregate injection ground-truth verification."""

import numpy as np

from src.experiments.injection_ground_truth_audit import audit_injection
from src.evaluation.injected_recovery import InjectionScenario
from src.models.lstm_forecaster import FeatureNormalizer


def test_ground_truth_audit_confirms_declared_support_and_clean_identity() -> None:
    clean = np.array([[0.11], [0.22], [0.33], [0.44]], dtype=np.float64)
    normalizer = FeatureNormalizer(
        mean=np.array([0.1], dtype=np.float32),
        scale=np.array([0.3], dtype=np.float32),
    )
    scenario = InjectionScenario(
        attack_id="audit-step",
        family="step",
        start=1,
        duration=2,
        magnitude_standard_deviations=1.5,
        sensors=(0,),
    )

    audit = audit_injection(clean, normalizer, scenario)

    assert audit["attack_id"] == "audit-step"
    assert audit["declared_support_count"] == 2
    assert audit["delta_nonzero_outside_declared_support"] == 0
    assert audit["observed_equals_clean_plus_delta"] is True
    assert audit["declared_delta_matches_expected"] is True
