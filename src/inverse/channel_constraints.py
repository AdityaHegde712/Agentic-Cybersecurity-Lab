"""Immutable channel constraints for inverse-reconstruction benchmarks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping


ChannelRole = Literal["target_measurement", "recorded_companion"]
ValueType = Literal["continuous", "discrete", "binary"]
Attackability = Literal["attackable", "declared_unaffected", "unknown"]
TrustAssumption = Literal["independently_trusted", "declared_unaffected", "unknown"]
ConstraintMode = Literal["soft", "hard"]
SemanticStatus = Literal["candidate_unverified", "validated"]
EvidenceScope = Literal["model_fit_only"]

_CHANNEL_ROLES = frozenset(("target_measurement", "recorded_companion"))
_VALUE_TYPES = frozenset(("continuous", "discrete", "binary"))
_ATTACKABILITIES = frozenset(("attackable", "declared_unaffected", "unknown"))
_TRUST_ASSUMPTIONS = frozenset(
    ("independently_trusted", "declared_unaffected", "unknown")
)
_CONSTRAINT_MODES = frozenset(("soft", "hard"))
_SEMANTIC_STATUSES = frozenset(("candidate_unverified", "validated"))
_EVIDENCE_SCOPES = frozenset(("model_fit_only",))


def _require_nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be a finite number")
    return numeric_value


@dataclass(frozen=True)
class TopologyMapping:
    """Dataset-to-public-system mapping and its validation status."""

    public_reference: str | None
    dataset_alignment_validated: bool

    def __post_init__(self) -> None:
        if self.public_reference is not None:
            _require_nonempty_text(self.public_reference, "public_reference")
        if not isinstance(self.dataset_alignment_validated, bool):
            raise ValueError("dataset_alignment_validated must be a boolean")


@dataclass(frozen=True)
class EmpiricalBounds:
    """Observed value limits estimated exclusively from the model-fit block."""

    minimum: float
    maximum: float
    source_block: Literal["model_fit"]

    def __post_init__(self) -> None:
        minimum = _require_finite_number(self.minimum, "minimum")
        maximum = _require_finite_number(self.maximum, "maximum")
        if minimum > maximum:
            raise ValueError("minimum must not exceed maximum")


@dataclass(frozen=True)
class EmpiricalRate:
    """Observed per-step rate limit estimated exclusively from model-fit data."""

    maximum_absolute_change_per_step: float
    source_block: Literal["model_fit"]

    def __post_init__(self) -> None:
        maximum_change = _require_finite_number(
            self.maximum_absolute_change_per_step,
            "maximum_absolute_change_per_step",
        )
        if maximum_change < 0.0:
            raise ValueError("maximum_absolute_change_per_step must be non-negative")


@dataclass(frozen=True)
class ChannelConstraint:
    """Semantics, admissible limits, and provenance for one dataset channel."""

    name: str
    role: ChannelRole
    unit: str
    value_type: ValueType
    attackability: Attackability
    trust_assumption: TrustAssumption
    topology_mapping: TopologyMapping
    empirical_bounds: EmpiricalBounds
    empirical_rate: EmpiricalRate
    constraint_mode: ConstraintMode
    source_provenance: tuple[str, ...]
    semantic_status: SemanticStatus = "candidate_unverified"
    stored_unit_status: SemanticStatus = "candidate_unverified"
    constraint_enforcement: ConstraintMode = "soft"
    evidence_scope: EvidenceScope = "model_fit_only"
    candidate_public_unit: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_text(self.name, "name")
        _require_nonempty_text(self.unit, "unit")
        _require_enum_value(self.role, _CHANNEL_ROLES, "role")
        _require_enum_value(self.value_type, _VALUE_TYPES, "value_type")
        _require_enum_value(self.attackability, _ATTACKABILITIES, "attackability")
        _require_enum_value(self.trust_assumption, _TRUST_ASSUMPTIONS, "trust_assumption")
        _require_enum_value(self.constraint_mode, _CONSTRAINT_MODES, "constraint_mode")
        _require_enum_value(self.semantic_status, _SEMANTIC_STATUSES, "semantic_status")
        _require_enum_value(
            self.stored_unit_status,
            _SEMANTIC_STATUSES,
            "stored_unit_status",
        )
        _require_enum_value(
            self.constraint_enforcement,
            _CONSTRAINT_MODES,
            "constraint_enforcement",
        )
        _require_enum_value(self.evidence_scope, _EVIDENCE_SCOPES, "evidence_scope")
        if self.candidate_public_unit is not None:
            _require_nonempty_text(
                self.candidate_public_unit,
                "candidate_public_unit",
            )
        _require_provenance(self.source_provenance)
        if not isinstance(self.topology_mapping, TopologyMapping):
            raise ValueError("topology_mapping must be a TopologyMapping")
        if not isinstance(self.empirical_bounds, EmpiricalBounds):
            raise ValueError("empirical_bounds must be an EmpiricalBounds")
        if not isinstance(self.empirical_rate, EmpiricalRate):
            raise ValueError("empirical_rate must be an EmpiricalRate")
        if self.empirical_bounds.source_block != "model_fit":
            raise ValueError("empirical bounds must use source_block='model_fit'")
        if self.empirical_rate.source_block != "model_fit":
            raise ValueError("empirical rate must use source_block='model_fit'")
        if self.constraint_mode != self.constraint_enforcement:
            raise ValueError("constraint_mode and constraint_enforcement must match")
        is_hard_constraint = self.constraint_mode == "hard"
        if is_hard_constraint and not self.topology_mapping.dataset_alignment_validated:
            raise ValueError("hard constraints require validated dataset topology alignment")


@dataclass(frozen=True)
class ChannelConstraintManifest:
    """Deterministic, complete channel-constraint declaration for one dataset."""

    manifest_version: str
    dataset: str
    dataset_channels: tuple[str, ...]
    channels: tuple[ChannelConstraint, ...]

    def __post_init__(self) -> None:
        _require_nonempty_text(self.manifest_version, "manifest_version")
        _require_nonempty_text(self.dataset, "dataset")
        _require_channel_names(self.dataset_channels, "dataset_channels")
        _require_channel_constraints(self.channels)
        declared_names = tuple(channel.name for channel in self.channels)
        if set(declared_names) != set(self.dataset_channels):
            raise ValueError("channels must declare every dataset channel exactly once")
        if len(declared_names) != len(set(declared_names)):
            raise ValueError("channels must not contain duplicate names")

    def to_json(self) -> str:
        """Return a stable JSON representation suitable for sealed manifests."""
        return json.dumps(self._as_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, serialized: str) -> ChannelConstraintManifest:
        """Restore a manifest from the deterministic JSON representation."""
        if not isinstance(serialized, str):
            raise ValueError("serialized manifest must be a JSON string")
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError("serialized manifest must contain valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("serialized manifest must be a JSON object")
        expected_keys = {"manifest_version", "dataset", "dataset_channels", "channels"}
        if set(payload) != expected_keys:
            raise ValueError("serialized manifest has an invalid schema")
        return cls(
            manifest_version=_mapping_text(payload, "manifest_version"),
            dataset=_mapping_text(payload, "dataset"),
            dataset_channels=_mapping_text_tuple(payload, "dataset_channels"),
            channels=tuple(_channel_from_payload(item) for item in _mapping_list(payload, "channels")),
        )

    def _as_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "dataset": self.dataset,
            "dataset_channels": list(self.dataset_channels),
            "channels": [_channel_as_dict(channel) for channel in self.channels],
        }


def _require_enum_value(value: object, allowed: frozenset[str], field_name: str) -> None:
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {allowed_values}")


def _require_provenance(provenance: object) -> None:
    if not isinstance(provenance, tuple) or not provenance:
        raise ValueError("source_provenance must be a non-empty tuple")
    for source in provenance:
        _require_nonempty_text(source, "source_provenance entry")


def _require_channel_names(names: object, field_name: str) -> None:
    if not isinstance(names, tuple) or not names:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    for name in names:
        _require_nonempty_text(name, field_name)
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _require_channel_constraints(channels: object) -> None:
    if not isinstance(channels, tuple) or not channels:
        raise ValueError("channels must be a non-empty tuple")
    if not all(isinstance(channel, ChannelConstraint) for channel in channels):
        raise ValueError("channels must contain ChannelConstraint instances")


def _channel_as_dict(channel: ChannelConstraint) -> dict[str, object]:
    return {
        "name": channel.name,
        "role": channel.role,
        "unit": channel.unit,
        "value_type": channel.value_type,
        "attackability": channel.attackability,
        "trust_assumption": channel.trust_assumption,
        "topology_mapping": {
            "public_reference": channel.topology_mapping.public_reference,
            "dataset_alignment_validated": channel.topology_mapping.dataset_alignment_validated,
        },
        "empirical_bounds": {
            "minimum": channel.empirical_bounds.minimum,
            "maximum": channel.empirical_bounds.maximum,
            "source_block": channel.empirical_bounds.source_block,
        },
        "empirical_rate": {
            "maximum_absolute_change_per_step": channel.empirical_rate.maximum_absolute_change_per_step,
            "source_block": channel.empirical_rate.source_block,
        },
        "constraint_mode": channel.constraint_mode,
        "source_provenance": list(channel.source_provenance),
        "semantic_status": channel.semantic_status,
        "stored_unit_status": channel.stored_unit_status,
        "constraint_enforcement": channel.constraint_enforcement,
        "evidence_scope": channel.evidence_scope,
        "candidate_public_unit": channel.candidate_public_unit,
    }


def _channel_from_payload(payload: object) -> ChannelConstraint:
    if not isinstance(payload, dict):
        raise ValueError("channel entries must be JSON objects")
    legacy_keys = {
        "name",
        "role",
        "unit",
        "value_type",
        "attackability",
        "trust_assumption",
        "topology_mapping",
        "empirical_bounds",
        "empirical_rate",
        "constraint_mode",
        "source_provenance",
    }
    optional_keys = {
        "semantic_status",
        "stored_unit_status",
        "constraint_enforcement",
        "evidence_scope",
        "candidate_public_unit",
    }
    if not legacy_keys.issubset(payload) or not set(payload).issubset(
        legacy_keys | optional_keys
    ):
        raise ValueError("channel entry has an invalid schema")
    topology_mapping = _mapping_object(payload, "topology_mapping")
    empirical_bounds = _mapping_object(payload, "empirical_bounds")
    empirical_rate = _mapping_object(payload, "empirical_rate")
    return ChannelConstraint(
        name=_mapping_text(payload, "name"),
        role=_mapping_text(payload, "role"),
        unit=_mapping_text(payload, "unit"),
        value_type=_mapping_text(payload, "value_type"),
        attackability=_mapping_text(payload, "attackability"),
        trust_assumption=_mapping_text(payload, "trust_assumption"),
        topology_mapping=TopologyMapping(
            public_reference=topology_mapping.get("public_reference"),
            dataset_alignment_validated=topology_mapping.get("dataset_alignment_validated"),
        ),
        empirical_bounds=EmpiricalBounds(
            minimum=empirical_bounds.get("minimum"),
            maximum=empirical_bounds.get("maximum"),
            source_block=empirical_bounds.get("source_block"),
        ),
        empirical_rate=EmpiricalRate(
            maximum_absolute_change_per_step=empirical_rate.get(
                "maximum_absolute_change_per_step"
            ),
            source_block=empirical_rate.get("source_block"),
        ),
        constraint_mode=_mapping_text(payload, "constraint_mode"),
        source_provenance=_mapping_text_tuple(payload, "source_provenance"),
        semantic_status=payload.get("semantic_status", "candidate_unverified"),
        stored_unit_status=payload.get("stored_unit_status", "candidate_unverified"),
        constraint_enforcement=payload.get("constraint_enforcement", "soft"),
        evidence_scope=payload.get("evidence_scope", "model_fit_only"),
        candidate_public_unit=payload.get("candidate_public_unit"),
    )


def _mapping_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    return _require_nonempty_text(value, key)


def _mapping_text_tuple(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = mapping.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a JSON array")
    return tuple(_require_nonempty_text(value, key) for value in values)


def _mapping_list(mapping: Mapping[str, Any], key: str) -> list[object]:
    values = mapping.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a JSON array")
    return values


def _mapping_object(mapping: Mapping[str, Any], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value
