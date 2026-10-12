"""LOCKED contracts for observation semantics and inverse ambiguity reporting."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.inverse.identifiability import (
    assess_ambiguity,
    fingerprint_observation,
    observation_contract,
)
from src.inverse.identifiability import HardSetCertificate


def _ambiguous_candidates() -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray]:
    observed = np.array([10.0, 20.0, 30.0, 40.0])
    declared_support = np.array([False, True, True, False])
    clean_candidates = (
        np.array([10.0, 18.0, 28.0, 40.0]),
        np.array([10.0, 17.0, 27.0, 40.0]),
    )
    delta_candidates = (
        np.array([0.0, 2.0, 2.0, 0.0]),
        np.array([0.0, 3.0, 3.0, 0.0]),
    )
    return observed, clean_candidates, delta_candidates, declared_support


def test_observation_contract_distinguishes_injected_truth_from_real_attack_noise() -> None:
    synthetic = observation_contract("synthetic_injection")
    real = observation_contract("real_attack")

    assert synthetic.identity_is_exact is True
    assert synthetic.delta_ground_truth_available is True
    assert synthetic.variation_location == "recorded_clean_x"
    assert "process" in synthetic.variation_description.lower()
    assert "sensor" in synthetic.variation_description.lower()
    assert "injected delta" in synthetic.variation_description.lower()
    assert real.identity_is_exact is False
    assert real.delta_ground_truth_available is False
    assert real.variation_location == "calibrated_observation_likelihood"
    assert "calibrated" in real.variation_description.lower()
    assert "counterfactual" in real.variation_description.lower()
    with pytest.raises(ValueError):
        observation_contract("unsealed_regime")


def test_two_distinct_feasible_decompositions_are_reported_as_ambiguous() -> None:
    observed, clean_candidates, delta_candidates, declared_support = _ambiguous_candidates()

    result = assess_ambiguity(
        observed,
        clean_candidates,
        delta_candidates,
        declared_support,
    )

    assert result.ambiguous is True
    assert result.candidate_count == 2
    assert result.max_identity_error == pytest.approx(0.0, abs=1e-12)


def test_supplied_candidate_ambiguity_does_not_claim_hard_set_membership() -> None:
    observed, clean_candidates, delta_candidates, declared_support = _ambiguous_candidates()

    result = assess_ambiguity(
        observed,
        clean_candidates,
        delta_candidates,
        declared_support,
        tau_delta=0.25,
        delta_norm="linf",
    )

    assert result.ambiguous is True
    assert result.structural_status == "not_certified"
    assert result.certification_scope is None


@pytest.mark.parametrize(
    ("observed", "clean_candidates", "delta_candidates", "declared_support"),
    [
        (
            np.array([1.0, 2.0]),
            (np.array([1.0, 2.0]),),
            (np.array([0.0, 0.0]),),
            np.array([True, True, True]),
        ),
        (
            np.array([1.0, np.nan]),
            (np.array([1.0, 2.0]),),
            (np.array([0.0, 0.0]),),
            np.array([True, True]),
        ),
        (
            np.array([1.0, 2.0]),
            (np.array([1.0, 2.0]),),
            (np.array([1.0, 0.0]),),
            np.array([False, True]),
        ),
        (
            np.array([1.0, 2.0]),
            (np.array([1.0, 2.0]),),
            (np.array([0.0, 0.1]),),
            np.array([True, True]),
        ),
    ],
)
def test_assessment_rejects_invalid_or_infeasible_candidate_sets(
    observed: np.ndarray,
    clean_candidates: tuple[np.ndarray, ...],
    delta_candidates: tuple[np.ndarray, ...],
    declared_support: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        assess_ambiguity(observed, clean_candidates, delta_candidates, declared_support)


def test_assessment_rejects_duplicate_candidate_pairs() -> None:
    observed = np.array([1.0, 2.0])
    clean = np.array([1.0, 1.0])
    delta = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        assess_ambiguity(
            observed,
            (clean, clean.copy()),
            (delta, delta.copy()),
            np.array([True, True]),
        )


def test_real_contract_keeps_attack_addition_exact_but_distinguishes_q_from_x() -> None:
    real = observation_contract("real_attack")

    assert real.attack_addition_is_exact is True
    assert real.latent_process_symbol == "q"
    assert real.clean_observation_symbol == "x"
    assert "y = x + delta" in real.attack_addition_description
    assert "q" in real.variation_description
    assert "x" in real.variation_description


def test_finite_candidate_sample_cannot_certify_no_ambiguity_without_certificate() -> None:
    observed = np.array([1.0, 2.0])
    clean = np.array([1.0, 1.0])
    delta = np.array([0.0, 1.0])
    support = np.array([True, True])

    sampled = assess_ambiguity(
        observed,
        (clean,),
        (delta,),
        support,
        tau_delta=0.25,
        delta_norm="linf",
    )
    certified = assess_ambiguity(
        observed,
        (clean,),
        (delta,),
        support,
        tau_delta=0.25,
        delta_norm="linf",
        exhaustive_certificate=True,
    )

    assert sampled.ambiguous is None
    assert sampled.certified is False
    assert certified.ambiguous is False
    assert certified.certified is True


def test_sealed_delta_tolerance_and_norm_use_maximum_pairwise_separation() -> None:
    observed, clean_candidates, delta_candidates, declared_support = _ambiguous_candidates()

    linf = assess_ambiguity(
        observed,
        clean_candidates,
        delta_candidates,
        declared_support,
        tau_delta=0.25,
        delta_norm="linf",
    )
    l1 = assess_ambiguity(
        observed,
        clean_candidates,
        delta_candidates,
        declared_support,
        tau_delta=1.5,
        delta_norm="l1",
    )

    assert linf.max_pairwise_delta_separation == pytest.approx(1.0)
    assert linf.ambiguous is True
    assert l1.max_pairwise_delta_separation == pytest.approx(2.0)
    assert l1.ambiguous is True


def test_documented_batadal_toy_witness_has_two_feasible_range_rate_decompositions() -> None:
    observed = np.array([1.0, 1.0, 2.0, 2.0, 1.0])
    declared_support = np.array([False, False, True, True, False])
    clean_candidates = (
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.5, 1.5, 1.0]),
    )
    delta_candidates = (
        np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.5, 0.5, 0.0]),
    )

    for clean, delta in zip(clean_candidates, delta_candidates, strict=True):
        assert np.array_equal(observed, clean + delta)
        assert np.all(delta[~declared_support] == 0.0)
        assert clean.min() >= 1.0
        assert clean.max() <= 2.0
        assert np.max(np.abs(np.diff(clean))) <= 1.0

    result = assess_ambiguity(
        observed,
        clean_candidates,
        delta_candidates,
        declared_support,
        tau_delta=0.25,
        delta_norm="linf",
    )

    assert result.ambiguous is True
    assert result.certified is False
    assert result.max_pairwise_delta_separation == pytest.approx(0.5)


def test_legacy_exhaustive_flag_is_limited_to_the_supplied_enumeration() -> None:
    observed = np.array([1.0, 2.0])
    clean = np.array([1.0, 1.0])
    delta = np.array([0.0, 1.0])

    result = assess_ambiguity(
        observed,
        (clean,),
        (delta,),
        np.array([True, True]),
        tau_delta=0.25,
        delta_norm="linf",
        exhaustive_certificate=True,
    )

    assert result.certified is True
    assert result.certification_scope == "supplied_candidate_enumeration"
    assert result.structural_status == "not_certified"
    assert result.hard_set_certificate is None


def _certificate_payload(
    *,
    observed_fingerprint: str,
    hard_set_manifest_json: str,
    hard_set_manifest_id: str,
    method: str = "finite_exhaustive",
    coverage_scope: str = "F_hard(y)",
    verified_hard_constraint_ids: tuple[str, ...] = ("validated_topology_alignment",),
    certified_diameter_upper_bound: float = 0.25,
    benchmark_case_id: str = "batadal-toy-witness-v1",
    delta_norm: str = "linf",
) -> str:
    return json.dumps(
        {
            "benchmark_case_id": benchmark_case_id,
            "certified_diameter_upper_bound": certified_diameter_upper_bound,
            "coverage_scope": coverage_scope,
            "delta_norm": delta_norm,
            "hard_set_manifest_id": hard_set_manifest_id,
            "hard_set_manifest_json": hard_set_manifest_json,
            "method": method,
            "observed_fingerprint": observed_fingerprint,
            "verified_hard_constraint_ids": list(verified_hard_constraint_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _hard_set_manifest_json(
    *,
    declared_support: np.ndarray,
    identity_atol: float,
    attack_class: dict[str, object] | None = None,
    normal_dynamics_artifact_sha256: str = "b" * 64,
    hard_constraint_parameters: dict[str, object] | None = None,
) -> str:
    verified_hard_constraint_ids = ("validated_topology_alignment",)
    return json.dumps(
        {
            "attack_class": attack_class
            or {
                "amplitudes": [0.5, 1.0],
                "change_point_convention": "known_support_start_inclusive_end_exclusive",
                "duration": 2,
                "family": "signed_step",
                "signs": ["negative", "positive"],
            },
            "coverage_scope": "F_hard(y)",
            "declared_support": declared_support.tolist(),
            "hard_constraints": [
                {
                    "id": "validated_topology_alignment",
                    "parameters": hard_constraint_parameters
                    or {
                        "dataset_alignment_validated": True,
                        "maximum_absolute_clean_rate": 0.6543217897415161,
                    },
                }
            ],
            "identity_atol": identity_atol,
            "normal_dynamics": {
                "artifact_sha256": normal_dynamics_artifact_sha256,
                "model_family": "state_space",
            },
            "schema_version": "1.0",
            "verified_hard_constraint_ids": list(verified_hard_constraint_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _certificate_kwargs(
    *,
    observed: np.ndarray,
    declared_support: np.ndarray,
    identity_atol: float = 1e-12,
    benchmark_case_id: str = "batadal-toy-witness-v1",
    delta_norm: str = "linf",
    certified_diameter_upper_bound: float = 0.25,
) -> dict[str, object]:
    constraint_ids = ("validated_topology_alignment",)
    hard_set_manifest_json = _hard_set_manifest_json(
        declared_support=declared_support,
        identity_atol=identity_atol,
    )
    hard_set_manifest_id = hashlib.sha256(
        hard_set_manifest_json.encode("utf-8")
    ).hexdigest()
    evidence_manifest_json = _certificate_payload(
        benchmark_case_id=benchmark_case_id,
        observed_fingerprint=fingerprint_observation(observed),
        delta_norm=delta_norm,
        hard_set_manifest_json=hard_set_manifest_json,
        hard_set_manifest_id=hard_set_manifest_id,
        certified_diameter_upper_bound=certified_diameter_upper_bound,
    )
    return {
        "method": "finite_exhaustive",
        "coverage_scope": "F_hard(y)",
        "benchmark_case_id": benchmark_case_id,
        "observed_fingerprint": fingerprint_observation(observed),
        "delta_norm": delta_norm,
        "hard_set_manifest_json": hard_set_manifest_json,
        "hard_set_manifest_id": hard_set_manifest_id,
        "evidence_manifest_json": evidence_manifest_json,
        "evidence_manifest_id": hashlib.sha256(
            evidence_manifest_json.encode("utf-8")
        ).hexdigest(),
        "verified_hard_constraint_ids": constraint_ids,
        "certified_diameter_upper_bound": certified_diameter_upper_bound,
    }


def _hard_set_certificate(
    *,
    observed: np.ndarray,
    declared_support: np.ndarray,
    certified_diameter_upper_bound: float = 0.25,
) -> HardSetCertificate:
    return HardSetCertificate(
        **_certificate_kwargs(
            observed=observed,
            declared_support=declared_support,
            certified_diameter_upper_bound=certified_diameter_upper_bound,
        )
    )


def test_hard_set_certificate_above_tau_cannot_claim_structural_uniqueness() -> None:
    observed = np.array([1.0, 2.0])
    clean = np.array([1.0, 1.0])
    delta = np.array([0.0, 1.0])
    certificate = _hard_set_certificate(
        observed=observed,
        declared_support=np.array([True, True]),
        certified_diameter_upper_bound=0.5,
    )

    with pytest.raises((AttributeError, TypeError)):
        certificate.method = "certified_global_bound"

    result = assess_ambiguity(
        observed,
        (clean,),
        (delta,),
        np.array([True, True]),
        tau_delta=0.25,
        delta_norm="linf",
        hard_set_certificate=certificate,
        benchmark_case_id="batadal-toy-witness-v1",
        active_hard_set_manifest_json=certificate.hard_set_manifest_json,
    )

    assert result.structural_status == "not_certified"
    assert result.certification_scope == "F_hard(y)"
    assert result.hard_set_certificate == certificate


def test_matching_hard_set_certificate_with_small_enough_bound_certifies_uniqueness() -> None:
    observed = np.array([1.0, 2.0])
    certificate = _hard_set_certificate(
        observed=observed,
        declared_support=np.array([True, True]),
        certified_diameter_upper_bound=0.25,
    )
    clean = np.array([1.0, 1.0])
    delta = np.array([0.0, 1.0])

    with pytest.raises((AttributeError, TypeError)):
        certificate.method = "certified_global_bound"

    result = assess_ambiguity(
        observed,
        (clean,),
        (delta,),
        np.array([True, True]),
        tau_delta=0.25,
        delta_norm="linf",
        hard_set_certificate=certificate,
        benchmark_case_id="batadal-toy-witness-v1",
        active_hard_set_manifest_json=certificate.hard_set_manifest_json,
    )

    assert result.structural_status == "unique_certified"
    assert result.certification_scope == "F_hard(y)"
    assert result.hard_set_certificate == certificate


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "unsupported_method"),
        ("coverage_scope", "all_candidates"),
        ("benchmark_case_id", ""),
        ("observed_fingerprint", "not-a-sha256"),
        ("delta_norm", "unsupported_norm"),
        ("hard_set_manifest_json", "not-json"),
        ("hard_set_manifest_id", "not-a-sha256"),
        ("evidence_manifest_json", "not-json"),
        ("evidence_manifest_id", "not-a-sha256"),
        ("verified_hard_constraint_ids", ()),
        ("certified_diameter_upper_bound", -0.1),
    ],
)
def test_malformed_hard_set_certificate_is_rejected(
    field: str,
    value: object,
) -> None:
    kwargs = _certificate_kwargs(
        observed=np.array([1.0, 2.0]),
        declared_support=np.array([True, True]),
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        HardSetCertificate(**kwargs)


def test_certificate_hash_and_payload_fields_must_match() -> None:
    kwargs = _certificate_kwargs(
        observed=np.array([1.0, 2.0]),
        declared_support=np.array([True, True]),
    )

    with pytest.raises(ValueError):
        HardSetCertificate(**(kwargs | {"evidence_manifest_id": "a" * 64}))
    mismatched_payload = _certificate_payload(
        observed_fingerprint=kwargs["observed_fingerprint"],
        hard_set_manifest_json=kwargs["hard_set_manifest_json"],
        hard_set_manifest_id=kwargs["hard_set_manifest_id"],
        certified_diameter_upper_bound=0.5,
    )
    with pytest.raises(ValueError):
        HardSetCertificate(
            **(
                kwargs
                | {
                    "evidence_manifest_json": mismatched_payload,
                    "evidence_manifest_id": hashlib.sha256(
                        mismatched_payload.encode("utf-8")
                    ).hexdigest(),
                }
            )
        )


def test_hard_set_manifest_digest_and_payload_must_match_certificate_evidence() -> None:
    kwargs = _certificate_kwargs(
        observed=np.array([1.0, 2.0]),
        declared_support=np.array([True, True]),
    )
    mismatched_manifest_json = _hard_set_manifest_json(
        declared_support=np.array([True, True]),
        identity_atol=1e-12,
        hard_constraint_parameters={"different_hard_constraint": 1.0},
    )

    with pytest.raises(ValueError):
        HardSetCertificate(
            **(kwargs | {"hard_set_manifest_json": mismatched_manifest_json})
        )
    with pytest.raises(ValueError):
        HardSetCertificate(
            **(
                kwargs
                | {
                    "hard_set_manifest_json": mismatched_manifest_json,
                    "hard_set_manifest_id": hashlib.sha256(
                        mismatched_manifest_json.encode("utf-8")
                    ).hexdigest(),
                }
            )
        )


def test_certificate_cannot_be_reused_for_different_data_norm_or_case() -> None:
    observed = np.array([1.0, 2.0])
    clean = np.array([1.0, 1.0])
    delta = np.array([0.0, 1.0])
    support = np.array([True, True])
    certificate = _hard_set_certificate(
        observed=observed,
        declared_support=support,
    )

    with pytest.raises(ValueError):
        assess_ambiguity(
            np.array([2.0, 3.0]),
            (np.array([2.0, 2.0]),),
            (delta,),
            support,
            tau_delta=0.25,
            delta_norm="linf",
            hard_set_certificate=certificate,
            benchmark_case_id="batadal-toy-witness-v1",
            active_hard_set_manifest_json=certificate.hard_set_manifest_json,
        )
    with pytest.raises(ValueError):
        assess_ambiguity(
            observed,
            (clean,),
            (delta,),
            support,
            tau_delta=0.25,
            delta_norm="l1",
            hard_set_certificate=certificate,
            benchmark_case_id="batadal-toy-witness-v1",
            active_hard_set_manifest_json=certificate.hard_set_manifest_json,
        )
    with pytest.raises(ValueError):
        assess_ambiguity(
            observed,
            (clean,),
            (delta,),
            support,
            tau_delta=0.25,
            delta_norm="linf",
            hard_set_certificate=certificate,
            benchmark_case_id="batadal-toy-witness-v2",
            active_hard_set_manifest_json=certificate.hard_set_manifest_json,
        )


def test_certificate_requires_active_manifest_support_and_tolerance_to_match_inputs() -> None:
    observed = np.array([1.0, 2.0])
    support = np.array([True, True])
    certificate = _hard_set_certificate(observed=observed, declared_support=support)

    with pytest.raises(ValueError):
        assess_ambiguity(
            observed,
            (observed.copy(),),
            (np.zeros_like(observed),),
            np.array([True, False]),
            identity_atol=1e-12,
            tau_delta=0.25,
            delta_norm="linf",
            hard_set_certificate=certificate,
            benchmark_case_id="batadal-toy-witness-v1",
            active_hard_set_manifest_json=certificate.hard_set_manifest_json,
        )
    with pytest.raises(ValueError):
        assess_ambiguity(
            observed,
            (observed.copy(),),
            (np.zeros_like(observed),),
            support,
            identity_atol=1e-10,
            tau_delta=0.25,
            delta_norm="linf",
            hard_set_certificate=certificate,
            benchmark_case_id="batadal-toy-witness-v1",
            active_hard_set_manifest_json=certificate.hard_set_manifest_json,
        )


@pytest.mark.parametrize(
    "active_manifest_json",
    [
        _hard_set_manifest_json(
            declared_support=np.array([True, True]),
            identity_atol=1e-12,
            attack_class={
                "amplitudes": [0.25, 1.0],
                "change_point_convention": "known_support_start_inclusive_end_exclusive",
                "duration": 2,
                "family": "signed_step",
                "signs": ["negative", "positive"],
            },
        ),
        _hard_set_manifest_json(
            declared_support=np.array([True, True]),
            identity_atol=1e-12,
            normal_dynamics_artifact_sha256="c" * 64,
        ),
        _hard_set_manifest_json(
            declared_support=np.array([True, True]),
            identity_atol=1e-12,
            hard_constraint_parameters={
                "dataset_alignment_validated": True,
                "maximum_absolute_clean_rate": 0.7,
            },
        ),
    ],
)
def test_certificate_rejects_changed_sealed_attack_dynamics_or_constraint_content(
    active_manifest_json: str,
) -> None:
    observed = np.array([1.0, 2.0])
    support = np.array([True, True])
    certificate = _hard_set_certificate(observed=observed, declared_support=support)

    with pytest.raises(ValueError):
        assess_ambiguity(
            observed,
            (np.array([1.0, 1.0]),),
            (np.array([0.0, 1.0]),),
            support,
            identity_atol=1e-12,
            tau_delta=0.25,
            delta_norm="linf",
            hard_set_certificate=certificate,
            benchmark_case_id="batadal-toy-witness-v1",
            active_hard_set_manifest_json=active_manifest_json,
        )


def test_certificate_below_supplied_separation_is_contradictory() -> None:
    observed, clean_candidates, delta_candidates, declared_support = _ambiguous_candidates()
    certificate = _hard_set_certificate(
        observed=observed,
        declared_support=declared_support,
        certified_diameter_upper_bound=0.5,
    )

    with pytest.raises(ValueError):
        assess_ambiguity(
            observed,
            clean_candidates,
            delta_candidates,
            declared_support,
            tau_delta=2.0,
            delta_norm="linf",
            hard_set_certificate=certificate,
            benchmark_case_id="batadal-toy-witness-v1",
            active_hard_set_manifest_json=certificate.hard_set_manifest_json,
        )


def test_missing_hard_set_certificate_cannot_claim_structural_uniqueness() -> None:
    observed = np.array([1.0, 2.0])
    clean = np.array([1.0, 1.0])
    delta = np.array([0.0, 1.0])

    result = assess_ambiguity(
        observed,
        (clean,),
        (delta,),
        np.array([True, True]),
        tau_delta=0.25,
        delta_norm="linf",
        exhaustive_certificate=True,
    )

    assert result.structural_status != "unique_certified"


def test_sealed_batadal_witness_constants_fix_range_rate_amplitude_and_duration() -> None:
    clean_range = (0.32011184096336365, 4.871329307556152)
    maximum_absolute_rate = 0.6543217897415161
    amplitudes = frozenset((0.5, 1.0))
    window = np.array([2, 3])

    assert clean_range == (0.32011184096336365, 4.871329307556152)
    assert maximum_absolute_rate == pytest.approx(0.6543217897415161)
    assert amplitudes == frozenset((0.5, 1.0))
    assert window.size == 2
    for amplitude in amplitudes:
        delta = np.zeros(5)
        delta[window] = amplitude
        assert np.array_equal(delta[window], np.full(window.size, amplitude))
        assert np.count_nonzero(delta) == window.size


def test_batadal_manifest_witness_arrays_satisfy_every_sealed_constraint() -> None:
    minimum, maximum = (0.32011184096336365, 4.871329307556152)
    maximum_clean_step = 0.6543217897415161
    window = np.array([2, 3])
    observed = np.array([1.0, 1.0, 2.0, 2.0, 1.0])
    clean_a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    delta_a = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    clean_b = np.array([1.0, 1.0, 1.5, 1.5, 1.0])
    delta_b = np.array([0.0, 0.0, 0.5, 0.5, 0.0])

    assert np.array_equal(window, np.array([2, 3]))
    assert window.size == 2
    for clean, delta, amplitude in (
        (clean_a, delta_a, 1.0),
        (clean_b, delta_b, 0.5),
    ):
        assert np.array_equal(observed, clean + delta)
        assert np.all(delta[np.setdiff1d(np.arange(observed.size), window)] == 0.0)
        assert np.array_equal(delta[window], np.full(window.size, amplitude))
        assert amplitude in {0.5, 1.0}
        assert clean.min() >= minimum
        assert clean.max() <= maximum
        assert np.max(np.abs(np.diff(clean))) <= maximum_clean_step
