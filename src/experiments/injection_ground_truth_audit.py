"""Aggregate verification for standardized additive-attack ground truth."""

from typing import Any

import numpy as np

from src.attacks.additive import inject_additive_attack
from src.evaluation.injected_recovery import (
    InjectionScenario,
    declared_support_mask,
    inject_standardized_attack,
)
from src.models.lstm_forecaster import FeatureNormalizer


def audit_injection(
    clean: np.ndarray,
    normalizer: FeatureNormalizer,
    scenario: InjectionScenario,
) -> dict[str, Any]:
    """Verify one injected scenario without exposing time-series rows."""
    standardized_clean = normalizer.transform(clean)
    standardized = inject_additive_attack(
        standardized_clean,
        scenario.to_attack_spec(),
    )
    injected = inject_standardized_attack(clean, normalizer, scenario)
    support = declared_support_mask(clean.shape, scenario)
    expected_delta = standardized.delta * normalizer.scale.astype(np.float64)

    legacy_observed = normalizer.inverse(standardized.observed)
    legacy_delta = legacy_observed.astype(np.float64) - clean.astype(np.float64)

    return {
        "attack_id": scenario.attack_id,
        "declared_support_count": int(support.sum()),
        "delta_nonzero_outside_declared_support": int(
            np.count_nonzero(injected.delta[~support])
        ),
        "observed_equals_clean_plus_delta": bool(
            np.array_equal(injected.observed, clean + injected.delta)
        ),
        "declared_delta_matches_expected": bool(
            np.array_equal(injected.delta, expected_delta)
        ),
        "legacy_round_trip_nonzero_outside_declared_support": int(
            np.count_nonzero(legacy_delta[~support])
        ),
        "legacy_round_trip_max_abs_outside_declared_support": float(
            np.abs(legacy_delta[~support]).max(initial=0.0)
        ),
    }
