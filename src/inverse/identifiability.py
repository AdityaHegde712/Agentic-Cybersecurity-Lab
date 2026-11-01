"""Contracts for stating when additive inverse recovery is non-identifiable."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Final, Literal

import numpy as np


ObservationRegime = Literal["synthetic_injection", "real_attack"]
DeltaNorm = Literal["linf", "l1"]
StructuralStatus = Literal["ambiguous", "unique_certified", "not_certified"]
HardSetMethod = Literal["finite_exhaustive", "certified_global_bound"]

# Strict float64 tolerance for the asserted y = x + delta observation identity.
IDENTITY_ATOL: Final[float] = 1e-12


def fingerprint_observation(observed: np.ndarray) -> str:
    """Hash a finite numeric 1D observation using shape and little-endian float64 bytes."""
    observed_vector = _validated_numeric_vector(observed, "observed")
    canonical_values = np.ascontiguousarray(observed_vector.astype("<f8", copy=False))
    shape_bytes = observed_vector.size.to_bytes(8, byteorder="big", signed=False)
    payload = b"inverse-observation-v1\x00" + shape_bytes + canonical_values.tobytes()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ObservationContract:
    """Truth and variation semantics for a sealed observation regime."""

    regime: ObservationRegime
    identity_is_exact: bool
    delta_ground_truth_available: bool
    variation_location: str
    variation_description: str
    attack_addition_is_exact: bool
    attack_addition_description: str
    latent_process_symbol: str
    clean_observation_symbol: str


@dataclass(frozen=True)
class AmbiguityResult:
    """Status for supplied candidates, distinct from proof over F_hard(y)."""

    ambiguous: bool | None
    certified: bool
    candidate_count: int
    max_identity_error: float
    max_pairwise_delta_separation: float
    tau_delta: float
    delta_norm: DeltaNorm
    structural_status: StructuralStatus
    certification_scope: str | None
    hard_set_certificate: HardSetCertificate | None


@dataclass(frozen=True)
class HardSetCertificate:
    """Evidence that all of F_hard(y), not just supplied candidates, was checked."""

    method: HardSetMethod
    coverage_scope: Literal["F_hard(y)"]
    benchmark_case_id: str
    observed_fingerprint: str
    delta_norm: DeltaNorm
    hard_set_manifest_json: str
    hard_set_manifest_id: str
    evidence_manifest_json: str
    evidence_manifest_id: str
    verified_hard_constraint_ids: tuple[str, ...]
    certified_diameter_upper_bound: float

    def __post_init__(self) -> None:
        if self.method not in ("finite_exhaustive", "certified_global_bound"):
            raise ValueError("method must be a supported hard-set certification method")
        if self.coverage_scope != "F_hard(y)":
            raise ValueError("coverage_scope must be exactly 'F_hard(y)'")
        _validate_nonblank_text(self.benchmark_case_id, "benchmark_case_id")
        _validate_sha256_digest(self.observed_fingerprint, "observed_fingerprint")
        _validate_delta_norm(self.delta_norm)
        _validate_certificate_bound(
            self.certified_diameter_upper_bound,
            "certified_diameter_upper_bound",
        )
        _validate_sha256_digest(self.hard_set_manifest_id, "hard_set_manifest_id")
        _validate_sha256_digest(self.evidence_manifest_id, "evidence_manifest_id")
        _validate_verified_constraint_ids(self.verified_hard_constraint_ids)
        hard_set_manifest = _parse_canonical_json(
            self.hard_set_manifest_json,
            "hard_set_manifest_json",
        )
        _validate_hard_set_manifest(hard_set_manifest, self)
        _validate_manifest_digest(
            self.hard_set_manifest_json,
            self.hard_set_manifest_id,
            "hard_set_manifest_id",
        )
        payload = _parse_certificate_payload(self.evidence_manifest_json)
        _validate_certificate_payload(payload, self)
        _validate_manifest_digest(
            self.evidence_manifest_json,
            self.evidence_manifest_id,
            "evidence_manifest_id",
        )


def observation_contract(regime: ObservationRegime) -> ObservationContract:
    """Return the observation semantics for a supported benchmark regime."""
    if regime == "synthetic_injection":
        return ObservationContract(
            regime=regime,
            identity_is_exact=True,
            delta_ground_truth_available=True,
            variation_location="recorded_clean_x",
            variation_description=(
                "Recorded clean x retains process and sensor variation; injected delta "
                "is added exactly to create y."
            ),
            attack_addition_is_exact=True,
            attack_addition_description="y = x + delta is exact for a synthetic injection.",
            latent_process_symbol="q",
            clean_observation_symbol="x",
        )
    if regime == "real_attack":
        return ObservationContract(
            regime=regime,
            identity_is_exact=False,
            delta_ground_truth_available=False,
            variation_location="calibrated_observation_likelihood",
            variation_description=(
                "A calibrated observation likelihood maps latent process q to clean "
                "observation x; the counterfactual clean x and delta are not observed "
                "ground truth."
            ),
            attack_addition_is_exact=True,
            attack_addition_description=(
                "y = x + delta is exact after defining clean observation x."
            ),
            latent_process_symbol="q",
            clean_observation_symbol="x",
        )
    raise ValueError("regime must be 'synthetic_injection' or 'real_attack'")


def assess_ambiguity(
    observed: np.ndarray,
    clean_candidates: tuple[np.ndarray, ...],
    delta_candidates: tuple[np.ndarray, ...],
    declared_support: np.ndarray,
    *,
    identity_atol: float = IDENTITY_ATOL,
    tau_delta: float = 0.0,
    delta_norm: DeltaNorm = "linf",
    exhaustive_certificate: bool = False,
    hard_set_certificate: HardSetCertificate | None = None,
    benchmark_case_id: str | None = None,
    active_hard_set_manifest_json: str | None = None,
) -> AmbiguityResult:
    """Assess supplied candidates; they are not internally proven F_hard(y)-feasible."""
    observed_vector = _validated_numeric_vector(observed, "observed")
    support_vector = _validated_support_vector(declared_support, observed_vector.shape)
    _validate_tolerance(identity_atol)
    _validate_tau_delta(tau_delta)
    _validate_delta_norm(delta_norm)
    _validate_exhaustive_certificate(exhaustive_certificate)
    _validate_hard_set_certificate(hard_set_certificate)
    _validate_certificate_binding(
        hard_set_certificate,
        observed_vector,
        delta_norm,
        benchmark_case_id,
    )
    _validate_active_hard_set_manifest(
        hard_set_certificate,
        active_hard_set_manifest_json,
        support_vector,
        identity_atol,
    )
    _validate_candidate_collections(clean_candidates, delta_candidates)

    candidate_errors: list[float] = []
    seen_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (clean, delta) in enumerate(zip(clean_candidates, delta_candidates)):
        clean_vector = _validated_numeric_vector(clean, f"clean_candidates[{index}]")
        delta_vector = _validated_numeric_vector(delta, f"delta_candidates[{index}]")
        _validate_candidate_shape(clean_vector, observed_vector.shape, "clean", index)
        _validate_candidate_shape(delta_vector, observed_vector.shape, "delta", index)
        _reject_unsupported_delta(delta_vector, support_vector, index)
        identity_error = float(np.max(np.abs(observed_vector - clean_vector - delta_vector)))
        if identity_error > identity_atol:
            raise ValueError(f"candidate pair {index} violates observed = clean + delta")
        _reject_duplicate_pair(clean_vector, delta_vector, seen_pairs)
        seen_pairs.append((clean_vector, delta_vector))
        candidate_errors.append(identity_error)

    max_separation = _max_pairwise_delta_separation(delta_candidates, delta_norm)
    _reject_contradictory_certificate(hard_set_certificate, max_separation)
    ambiguity, certified = _ambiguity_status(
        max_separation,
        tau_delta,
        exhaustive_certificate,
        hard_set_certificate,
    )
    structural_status = _structural_status(max_separation, tau_delta, hard_set_certificate)
    certification_scope = _certification_scope(
        exhaustive_certificate,
        hard_set_certificate,
    )
    return AmbiguityResult(
        ambiguous=ambiguity,
        certified=certified,
        candidate_count=len(candidate_errors),
        max_identity_error=max(candidate_errors),
        max_pairwise_delta_separation=max_separation,
        tau_delta=float(tau_delta),
        delta_norm=delta_norm,
        structural_status=structural_status,
        certification_scope=certification_scope,
        hard_set_certificate=hard_set_certificate,
    )


def _validated_numeric_vector(values: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional numpy array")
    is_numeric = np.issubdtype(values.dtype, np.number)
    is_boolean = np.issubdtype(values.dtype, np.bool_)
    if not is_numeric or is_boolean:
        raise ValueError(f"{name} must contain numeric values")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validated_support_vector(
    support: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(support, np.ndarray) or support.ndim != 1:
        raise ValueError("declared_support must be a one-dimensional numpy boolean array")
    if support.shape != expected_shape:
        raise ValueError("declared_support must have the same shape as observed")
    if not np.issubdtype(support.dtype, np.bool_):
        raise ValueError("declared_support must contain boolean values")
    return support


def _validate_tolerance(identity_atol: float) -> None:
    if isinstance(identity_atol, bool) or not isinstance(identity_atol, (int, float)):
        raise ValueError("identity_atol must be a non-negative finite number")
    if not np.isfinite(identity_atol) or identity_atol < 0.0:
        raise ValueError("identity_atol must be a non-negative finite number")


def _validate_tau_delta(tau_delta: float) -> None:
    if isinstance(tau_delta, bool) or not isinstance(tau_delta, (int, float)):
        raise ValueError("tau_delta must be a non-negative finite number")
    if not np.isfinite(tau_delta) or tau_delta < 0.0:
        raise ValueError("tau_delta must be a non-negative finite number")


def _validate_delta_norm(delta_norm: DeltaNorm) -> None:
    if delta_norm not in ("linf", "l1"):
        raise ValueError("delta_norm must be 'linf' or 'l1'")


def _validate_exhaustive_certificate(exhaustive_certificate: bool) -> None:
    if not isinstance(exhaustive_certificate, bool):
        raise ValueError("exhaustive_certificate must be a boolean")


def _validate_hard_set_certificate(certificate: HardSetCertificate | None) -> None:
    if certificate is not None and not isinstance(certificate, HardSetCertificate):
        raise ValueError("hard_set_certificate must be a HardSetCertificate or None")


def _validate_certificate_binding(
    certificate: HardSetCertificate | None,
    observed: np.ndarray,
    delta_norm: DeltaNorm,
    benchmark_case_id: str | None,
) -> None:
    if certificate is None:
        return
    _validate_nonblank_text(benchmark_case_id, "benchmark_case_id")
    if benchmark_case_id != certificate.benchmark_case_id:
        raise ValueError("benchmark_case_id does not match hard_set_certificate")
    if fingerprint_observation(observed) != certificate.observed_fingerprint:
        raise ValueError("observed does not match hard_set_certificate fingerprint")
    if delta_norm != certificate.delta_norm:
        raise ValueError("delta_norm does not match hard_set_certificate")


def _validate_nonblank_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


def _validate_sha256_digest(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


def _validate_certificate_bound(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative finite number")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be a non-negative finite number")


def _parse_canonical_json(serialized: object, field_name: str) -> dict[str, object]:
    if not isinstance(serialized, str):
        raise ValueError(f"{field_name} must be a JSON string")
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field_name} must contain valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must contain a JSON object")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if serialized != canonical:
        raise ValueError(f"{field_name} must use canonical JSON serialization")
    return payload


def _parse_certificate_payload(serialized: object) -> dict[str, object]:
    return _parse_canonical_json(serialized, "evidence_manifest_json")


def _validate_manifest_digest(
    serialized: str,
    digest: str,
    field_name: str,
) -> None:
    expected = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if digest != expected:
        raise ValueError(f"{field_name} must match the UTF-8 manifest SHA-256")


def _validate_hard_set_manifest(
    payload: dict[str, object],
    certificate: HardSetCertificate,
) -> None:
    required_keys = {
        "schema_version",
        "coverage_scope",
        "declared_support",
        "attack_class",
        "normal_dynamics",
        "hard_constraints",
        "identity_atol",
        "verified_hard_constraint_ids",
    }
    if set(payload) != required_keys:
        raise ValueError("hard_set_manifest_json has an invalid key set")
    if payload["schema_version"] != "1.0":
        raise ValueError("hard-set manifest schema_version must be '1.0'")
    if payload["coverage_scope"] != certificate.coverage_scope:
        raise ValueError("hard-set manifest coverage_scope does not match certificate")
    _validate_manifest_support(payload["declared_support"])
    _validate_manifest_identity_atol(payload["identity_atol"])
    _validate_attack_class(payload["attack_class"])
    _validate_normal_dynamics(payload["normal_dynamics"])
    hard_constraint_ids = _validate_hard_constraints(payload["hard_constraints"])
    if payload["verified_hard_constraint_ids"] != hard_constraint_ids:
        raise ValueError("hard-set manifest verified IDs must match hard constraints")
    if hard_constraint_ids != list(certificate.verified_hard_constraint_ids):
        raise ValueError("hard-set manifest constraint IDs do not match certificate")


def _validate_manifest_support(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("hard-set manifest declared_support must be a non-empty list")
    if any(type(entry) is not bool for entry in value):
        raise ValueError("hard-set manifest declared_support must contain booleans")


def _validate_manifest_identity_atol(value: object) -> None:
    _validate_certificate_bound(value, "hard-set manifest identity_atol")


def _validate_attack_class(value: object) -> None:
    required_keys = {
        "amplitudes",
        "signs",
        "duration",
        "change_point_convention",
        "family",
    }
    if not isinstance(value, dict) or set(value) != required_keys:
        raise ValueError("hard-set manifest attack_class has an invalid schema")
    amplitudes = value["amplitudes"]
    signs = value["signs"]
    if not isinstance(amplitudes, list) or not amplitudes:
        raise ValueError("hard-set manifest attack_class amplitudes must be non-empty")
    for amplitude in amplitudes:
        _validate_certificate_bound(amplitude, "hard-set manifest attack amplitude")
    if not isinstance(signs, list) or not signs:
        raise ValueError("hard-set manifest attack_class signs must be non-empty")
    for sign in signs:
        _validate_nonblank_text(sign, "hard-set manifest attack sign")
    duration = value["duration"]
    if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
        raise ValueError("hard-set manifest attack duration must be a positive integer")
    _validate_nonblank_text(
        value["change_point_convention"],
        "hard-set manifest change_point_convention",
    )
    _validate_nonblank_text(value["family"], "hard-set manifest attack family")


def _validate_normal_dynamics(value: object) -> None:
    required_keys = {"model_family", "artifact_sha256"}
    if not isinstance(value, dict) or set(value) != required_keys:
        raise ValueError("hard-set manifest normal_dynamics has an invalid schema")
    _validate_nonblank_text(value["model_family"], "hard-set manifest model_family")
    _validate_sha256_digest(
        value["artifact_sha256"],
        "hard-set manifest artifact_sha256",
    )


def _validate_hard_constraints(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("hard-set manifest hard_constraints must be non-empty")
    constraint_ids: list[str] = []
    for constraint in value:
        if not isinstance(constraint, dict) or set(constraint) != {"id", "parameters"}:
            raise ValueError("hard-set manifest hard constraint has an invalid schema")
        constraint_id = constraint["id"]
        _validate_nonblank_text(constraint_id, "hard-set manifest hard constraint ID")
        _validate_typed_parameters(constraint["parameters"])
        constraint_ids.append(constraint_id)
    if len(constraint_ids) != len(set(constraint_ids)):
        raise ValueError("hard-set manifest hard constraint IDs must be unique")
    return constraint_ids


def _validate_typed_parameters(value: object) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError("hard-set manifest constraint parameters must be non-empty")
    for parameter_name, parameter_value in value.items():
        _validate_nonblank_text(parameter_name, "hard-set manifest parameter name")
        if isinstance(parameter_value, bool):
            continue
        if isinstance(parameter_value, str) and parameter_value.strip():
            continue
        if isinstance(parameter_value, (int, float)) and not isinstance(parameter_value, bool):
            if np.isfinite(parameter_value):
                continue
        raise ValueError("hard-set manifest constraint parameters must be typed scalars")


def _validate_active_hard_set_manifest(
    certificate: HardSetCertificate | None,
    active_manifest_json: str | None,
    declared_support: np.ndarray,
    identity_atol: float,
) -> None:
    if certificate is None:
        return
    if not isinstance(active_manifest_json, str):
        raise ValueError("active_hard_set_manifest_json is required with a certificate")
    if active_manifest_json != certificate.hard_set_manifest_json:
        raise ValueError("active_hard_set_manifest_json does not match certificate")
    _validate_manifest_digest(
        active_manifest_json,
        certificate.hard_set_manifest_id,
        "hard_set_manifest_id",
    )
    active_manifest = _parse_canonical_json(
        active_manifest_json,
        "active_hard_set_manifest_json",
    )
    if active_manifest["declared_support"] != declared_support.tolist():
        raise ValueError("active hard-set declared_support does not match input")
    if active_manifest["identity_atol"] != identity_atol:
        raise ValueError("active hard-set identity_atol does not match input")


def _validate_certificate_payload(
    payload: dict[str, object],
    certificate: HardSetCertificate,
) -> None:
    required_keys = {
        "method",
        "coverage_scope",
        "benchmark_case_id",
        "observed_fingerprint",
        "delta_norm",
        "hard_set_manifest_json",
        "hard_set_manifest_id",
        "verified_hard_constraint_ids",
        "certified_diameter_upper_bound",
    }
    if set(payload) != required_keys:
        raise ValueError("evidence_manifest_json has an invalid key set")
    payload_bound = payload["certified_diameter_upper_bound"]
    _validate_certificate_bound(payload_bound, "payload certified_diameter_upper_bound")
    if payload["method"] != certificate.method:
        raise ValueError("evidence manifest method does not match certificate")
    if payload["coverage_scope"] != certificate.coverage_scope:
        raise ValueError("evidence manifest coverage_scope does not match certificate")
    if payload["benchmark_case_id"] != certificate.benchmark_case_id:
        raise ValueError("evidence manifest benchmark_case_id does not match certificate")
    if payload["observed_fingerprint"] != certificate.observed_fingerprint:
        raise ValueError("evidence manifest fingerprint does not match certificate")
    if payload["delta_norm"] != certificate.delta_norm:
        raise ValueError("evidence manifest delta_norm does not match certificate")
    if payload["hard_set_manifest_json"] != certificate.hard_set_manifest_json:
        raise ValueError("evidence manifest hard-set JSON does not match certificate")
    if payload["hard_set_manifest_id"] != certificate.hard_set_manifest_id:
        raise ValueError("evidence manifest hard-set digest does not match certificate")
    if payload["verified_hard_constraint_ids"] != list(
        certificate.verified_hard_constraint_ids
    ):
        raise ValueError("evidence manifest constraint IDs do not match certificate")
    if payload_bound != certificate.certified_diameter_upper_bound:
        raise ValueError("evidence manifest diameter bound does not match certificate")


def _reject_contradictory_certificate(
    certificate: HardSetCertificate | None,
    max_separation: float,
) -> None:
    if certificate is None:
        return
    if certificate.certified_diameter_upper_bound < max_separation:
        raise ValueError("certificate diameter bound contradicts supplied candidate separation")


def _validate_verified_constraint_ids(constraint_ids: tuple[str, ...]) -> None:
    if not isinstance(constraint_ids, tuple) or not constraint_ids:
        raise ValueError("verified_hard_constraint_ids must be a non-empty tuple")
    has_invalid_identifier = any(
        not isinstance(identifier, str) or not identifier.strip()
        for identifier in constraint_ids
    )
    if has_invalid_identifier:
        raise ValueError("verified_hard_constraint_ids must contain non-blank strings")
    if len(constraint_ids) != len(set(constraint_ids)):
        raise ValueError("verified_hard_constraint_ids must be unique")


def _validate_candidate_collections(
    clean_candidates: tuple[np.ndarray, ...],
    delta_candidates: tuple[np.ndarray, ...],
) -> None:
    if not isinstance(clean_candidates, tuple) or not isinstance(delta_candidates, tuple):
        raise ValueError("candidate collections must be tuples")
    if not clean_candidates or len(clean_candidates) != len(delta_candidates):
        raise ValueError("candidate collections must have equal non-zero lengths")


def _validate_candidate_shape(
    candidate: np.ndarray,
    expected_shape: tuple[int, ...],
    kind: str,
    index: int,
) -> None:
    if candidate.shape != expected_shape:
        raise ValueError(f"{kind}_candidates[{index}] must have the same shape as observed")


def _reject_unsupported_delta(
    delta: np.ndarray,
    support: np.ndarray,
    index: int,
) -> None:
    if np.any(delta[~support] != 0.0):
        raise ValueError(f"delta_candidates[{index}] must be zero outside declared_support")


def _reject_duplicate_pair(
    clean: np.ndarray,
    delta: np.ndarray,
    seen_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    is_duplicate = any(
        np.array_equal(clean, prior_clean) and np.array_equal(delta, prior_delta)
        for prior_clean, prior_delta in seen_pairs
    )
    if is_duplicate:
        raise ValueError("clean/delta candidate pairs must be distinct")


def _max_pairwise_delta_separation(
    delta_candidates: tuple[np.ndarray, ...],
    delta_norm: DeltaNorm,
) -> float:
    maximum = 0.0
    for first_index, first_delta in enumerate(delta_candidates):
        for second_delta in delta_candidates[first_index + 1 :]:
            difference = np.abs(first_delta - second_delta)
            separation = (
                float(difference.max())
                if delta_norm == "linf"
                else float(difference.sum())
            )
            maximum = max(maximum, separation)
    return maximum


def _ambiguity_status(
    max_separation: float,
    tau_delta: float,
    exhaustive_certificate: bool,
    hard_set_certificate: HardSetCertificate | None,
) -> tuple[bool | None, bool]:
    if max_separation > tau_delta:
        return True, False
    certificate_proves_uniqueness = (
        hard_set_certificate is not None
        and hard_set_certificate.certified_diameter_upper_bound <= tau_delta
    )
    if certificate_proves_uniqueness:
        return False, True
    if exhaustive_certificate:
        return False, True
    return None, False


def _structural_status(
    max_separation: float,
    tau_delta: float,
    hard_set_certificate: HardSetCertificate | None,
) -> StructuralStatus:
    if max_separation > tau_delta:
        # Supplied candidates are diagnostics, not a verified F_hard(y) witness.
        return "not_certified"
    certificate_proves_uniqueness = (
        hard_set_certificate is not None
        and hard_set_certificate.certified_diameter_upper_bound <= tau_delta
    )
    if certificate_proves_uniqueness:
        return "unique_certified"
    return "not_certified"


def _certification_scope(
    exhaustive_certificate: bool,
    hard_set_certificate: HardSetCertificate | None,
) -> str | None:
    if hard_set_certificate is not None:
        return hard_set_certificate.coverage_scope
    if exhaustive_certificate:
        return "supplied_candidate_enumeration"
    return None
