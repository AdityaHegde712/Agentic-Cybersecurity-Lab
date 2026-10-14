"""Sealed, deterministic benchmark definition for narrowed BATADAL experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Mapping
from types import MappingProxyType


_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"
_ONE_HOUR = timedelta(hours=1)
_REQUIRED_PROVENANCE = (
    "selected_checkpoint_sha256",
    "estimator_config_sha256",
    "baseline_config_sha256",
)
_ARTIFACT_HASHES = {
    "batadal_train_parquet": "6a7c7232b5b24eb7b475dcca3d20e6fc6ada93f4f50769af454e1e5a9560d343",
    "batadal_val_parquet": "98a12650fc75cec6554e61f7ede2e09266d3858821b3e7faa973647ee2dc9967",
    "batadal_metadata_json": "d8d0bec8f1a93bf711939a8c017820fe26c9bac9a943b439c371530eacda0974",
    "channel_constraint_manifest": "e20b7728cbf4d22ea6fd6dbd4f5eca0b26f65a9165108fa8f8d8d890ad0bd347",
}


@dataclass(frozen=True)
class ChronologicalBlock:
    """A contiguous hourly interval from one source split."""

    name: str
    source_split: str
    row_start: int
    row_stop: int
    start_timestamp: str
    end_timestamp: str

    def __post_init__(self) -> None:
        _require_text(self.name, "block name")
        _require_text(self.source_split, "source_split")
        _require_nonnegative_int(self.row_start, "row_start")
        _require_nonnegative_int(self.row_stop, "row_stop")
        if self.row_stop <= self.row_start:
            raise ValueError("row_stop must be greater than row_start")
        start = _parse_timestamp(self.start_timestamp, "start_timestamp")
        end = _parse_timestamp(self.end_timestamp, "end_timestamp")
        if end < start:
            raise ValueError("end_timestamp must not precede start_timestamp")
        expected_end = start + _ONE_HOUR * (self.row_count - 1)
        if end != expected_end:
            raise ValueError("block timestamps must progress hourly for every row")

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start


@dataclass(frozen=True)
class InjectionGrid:
    """Grid of attack amplitudes, durations, and regime-stratified onsets."""

    amplitudes_train_std: tuple[float, ...]
    durations_hours: tuple[int, ...]
    regime_stratified_onset_count: int
    onsets_are_unseen: bool = False
    signs: tuple[int, ...] = (-1, 1)
    onset_selection_strategy: str = "regime_stratified_without_replacement"
    regime_labels: tuple[str, ...] = ("low_dynamics", "medium_dynamics", "high_dynamics")

    def __post_init__(self) -> None:
        _require_positive_numeric_tuple(self.amplitudes_train_std, "amplitudes_train_std")
        _require_positive_int_tuple(self.durations_hours, "durations_hours")
        _require_positive_int(self.regime_stratified_onset_count, "regime_stratified_onset_count")
        if not isinstance(self.onsets_are_unseen, bool):
            raise ValueError("onsets_are_unseen must be a boolean")
        if self.signs != (-1, 1):
            raise ValueError("signs must be the sealed (-1, 1) tuple")
        if self.onset_selection_strategy != "regime_stratified_without_replacement":
            raise ValueError("onset_selection_strategy must use the sealed strategy")
        _require_text_tuple(self.regime_labels, "regime_labels")


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Precommitted success thresholds for the benchmark."""

    minimum_median_delta_nrmse_improvement: float
    maximum_median_signed_amplitude_error_train_std: float
    minimum_sign_accuracy: float
    minimum_sign_magnitude_train_std: float
    nominal_interval_coverage: float
    minimum_empirical_interval_coverage: float
    maximum_empirical_interval_coverage: float
    maximum_hard_constraint_violations: int
    require_causal_baseline_improvement: bool

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_median_delta_nrmse_improvement",
            "maximum_median_signed_amplitude_error_train_std",
            "minimum_sign_accuracy",
            "minimum_sign_magnitude_train_std",
            "nominal_interval_coverage",
            "minimum_empirical_interval_coverage",
            "maximum_empirical_interval_coverage",
        ):
            _require_nonnegative_number(getattr(self, field_name), field_name)
        probability_fields = (
            self.minimum_sign_accuracy,
            self.nominal_interval_coverage,
            self.minimum_empirical_interval_coverage,
            self.maximum_empirical_interval_coverage,
        )
        if any(value > 1.0 for value in probability_fields):
            raise ValueError("coverage and accuracy thresholds must not exceed one")
        if self.minimum_empirical_interval_coverage > self.maximum_empirical_interval_coverage:
            raise ValueError("empirical interval coverage bounds are reversed")
        _require_nonnegative_int(
            self.maximum_hard_constraint_violations,
            "maximum_hard_constraint_violations",
        )
        if not isinstance(self.require_causal_baseline_improvement, bool):
            raise ValueError("require_causal_baseline_improvement must be a boolean")


@dataclass(frozen=True)
class SealedReadiness:
    """Fail-closed readiness state based on required provenance hashes."""

    ready: bool
    status: Literal["ready", "incomplete_missing_required_artifacts"]
    missing_required_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ready, bool):
            raise ValueError("ready must be a boolean")
        _require_text(self.status, "status")
        if not isinstance(self.missing_required_artifacts, tuple):
            raise ValueError("missing_required_artifacts must be a tuple")
        for artifact in self.missing_required_artifacts:
            _require_text(artifact, "missing required artifact")
        if len(set(self.missing_required_artifacts)) != len(self.missing_required_artifacts):
            raise ValueError("missing required artifacts must be unique")
        expected_ready = not self.missing_required_artifacts
        expected_status = "ready" if expected_ready else "incomplete_missing_required_artifacts"
        if self.ready != expected_ready or self.status != expected_status:
            raise ValueError("readiness fields are inconsistent")


@dataclass(frozen=True)
class BenchmarkManifest:
    """Complete, serializable narrowed BATADAL benchmark declaration."""

    manifest_version: str
    dataset: str
    target_sensor: str
    sampling_interval_seconds: int
    blocks: tuple[ChronologicalBlock, ...]
    development_injections: InjectionGrid
    final_injections: InjectionGrid
    seeds: Mapping[str, int]
    acceptance: AcceptanceThresholds
    artifact_hashes: Mapping[str, str]
    readiness: SealedReadiness
    generator_spec_json: str
    generator_spec_sha256: str
    metric_spec_json: str
    metric_spec_sha256: str
    final_evaluation_count: int = 0

    def __post_init__(self) -> None:
        _require_text(self.manifest_version, "manifest_version")
        _require_text(self.dataset, "dataset")
        _require_text(self.target_sensor, "target_sensor")
        _require_positive_int(self.sampling_interval_seconds, "sampling_interval_seconds")
        if self.sampling_interval_seconds != 3_600:
            raise ValueError("benchmark blocks require hourly sampling")
        _require_blocks(self.blocks)
        _require_batadal_block_layout(self.blocks)
        self.validate_chronology()
        _require_seed_mapping(self.seeds)
        _require_hash_mapping(self.artifact_hashes, "artifact_hashes")
        _require_artifact_roles(self.artifact_hashes)
        _validate_readiness(self.readiness, self.artifact_hashes)
        object.__setattr__(self, "seeds", MappingProxyType(dict(self.seeds)))
        object.__setattr__(self, "artifact_hashes", MappingProxyType(dict(self.artifact_hashes)))
        _validate_spec_seal(
            self.generator_spec_json,
            self.generator_spec_sha256,
            "generator_spec",
        )
        _validate_spec_seal(self.metric_spec_json, self.metric_spec_sha256, "metric_spec")
        _validate_generator_spec(self.generator_spec_json)
        _validate_metric_spec(self.metric_spec_json, self.acceptance)
        _require_nonnegative_int(self.final_evaluation_count, "final_evaluation_count")
        if self.final_evaluation_count != 0:
            raise ValueError("final_evaluation_count must remain zero before sealed evaluation")

    def validate_chronology(self) -> None:
        """Reject row/timestamp overlaps, gaps, reversals, and non-hourly boundaries."""
        previous_end: datetime | None = None
        split_row_stops: dict[str, int] = {}
        for block in self.blocks:
            start = _parse_timestamp(block.start_timestamp, "start_timestamp")
            end = _parse_timestamp(block.end_timestamp, "end_timestamp")
            if previous_end is not None and start != previous_end + _ONE_HOUR:
                raise ValueError("chronological blocks must have hourly boundary progression")
            expected_row_start = split_row_stops.get(block.source_split)
            if expected_row_start is not None and block.row_start != expected_row_start:
                raise ValueError("blocks in a source split must be contiguous without overlap or gaps")
            if end < start:
                raise ValueError("block timestamps are reversed")
            split_row_stops[block.source_split] = block.row_stop
            previous_end = end

    def require_sealed_readiness(self) -> None:
        """Raise unless all required provenance for a sealed evaluation is present."""
        if not self.readiness.ready:
            missing = ", ".join(self.readiness.missing_required_artifacts)
            raise ValueError(f"sealed readiness is incomplete: {missing}")

    def to_json(self) -> str:
        """Serialize the manifest as compact, sorted canonical JSON."""
        return json.dumps(self._as_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        """Return the SHA-256 fingerprint of the canonical manifest JSON."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, serialized: str) -> BenchmarkManifest:
        """Restore a strictly validated manifest from canonical JSON."""
        if not isinstance(serialized, str):
            raise ValueError("serialized manifest must be a string")
        try:
            payload = json.loads(serialized)
        except json.JSONDecodeError as error:
            raise ValueError("serialized manifest must contain valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("serialized manifest must be a JSON object")
        if serialized != json.dumps(payload, sort_keys=True, separators=(",", ":")):
            raise ValueError("serialized manifest must use canonical compact sorted JSON")
        return _manifest_from_payload(payload)

    def _as_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "dataset": self.dataset,
            "target_sensor": self.target_sensor,
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "blocks": [_block_as_dict(block) for block in self.blocks],
            "development_injections": _grid_as_dict(self.development_injections),
            "final_injections": _grid_as_dict(self.final_injections),
            "seeds": dict(self.seeds),
            "acceptance": _acceptance_as_dict(self.acceptance),
            "artifact_hashes": dict(self.artifact_hashes),
            "readiness": _readiness_as_dict(self.readiness),
            "generator_spec_json": self.generator_spec_json,
            "generator_spec_sha256": self.generator_spec_sha256,
            "metric_spec_json": self.metric_spec_json,
            "metric_spec_sha256": self.metric_spec_sha256,
            "final_evaluation_count": self.final_evaluation_count,
        }


def build_batadal_benchmark_manifest(
    *,
    seeds: Mapping[str, int] | None = None,
    artifact_hashes: Mapping[str, str] | None = None,
    required_provenance_hashes: Mapping[str, str] | None = None,
    claim_sealed_readiness: bool = False,
) -> BenchmarkManifest:
    """Build the precommitted BATADAL benchmark, fail-closed by default."""
    if not isinstance(claim_sealed_readiness, bool):
        raise ValueError("claim_sealed_readiness must be a boolean")
    seed_values = dict(seeds) if seeds is not None else {
        "dynamics": 20261012,
        "injection": 20261013,
        "uncertainty": 20261014,
    }
    hash_values = dict(artifact_hashes) if artifact_hashes is not None else dict(_ARTIFACT_HASHES)
    provenance = dict(required_provenance_hashes or {})
    if provenance:
        _require_hash_mapping(provenance, "required_provenance_hashes")
    _require_artifact_roles(hash_values)
    _require_provenance_roles(provenance)
    hash_values.update(provenance)
    missing = tuple(key for key in _REQUIRED_PROVENANCE if key not in hash_values)
    readiness = SealedReadiness(
        ready=not missing,
        status="ready" if not missing else "incomplete_missing_required_artifacts",
        missing_required_artifacts=missing,
    )
    manifest = BenchmarkManifest(
        manifest_version="1.0",
        dataset="batadal",
        target_sensor="L_T1",
        sampling_interval_seconds=3_600,
        blocks=_batadal_blocks(),
        development_injections=InjectionGrid((0.5, 1.0, 2.0), (6, 12, 24, 48), 5),
        final_injections=InjectionGrid((0.75, 1.5, 3.0), (9, 18, 36), 10, True),
        seeds=seed_values,
        acceptance=AcceptanceThresholds(0.20, 0.25, 0.95, 1.0, 0.90, 0.85, 0.95, 0, True),
        artifact_hashes=hash_values,
        readiness=readiness,
        generator_spec_json=_canonical_json(_generator_spec()),
        generator_spec_sha256=_json_sha256(_canonical_json(_generator_spec())),
        metric_spec_json=_canonical_json(_metric_spec()),
        metric_spec_sha256=_json_sha256(_canonical_json(_metric_spec())),
    )
    if claim_sealed_readiness:
        manifest.require_sealed_readiness()
    return manifest


def _batadal_blocks() -> tuple[ChronologicalBlock, ...]:
    return (
        ChronologicalBlock("model_fit", "train", 0, 5_840, "2014-01-06 00:00", "2014-09-06 07:00"),
        ChronologicalBlock("model_selection", "train", 5_840, 7_321, "2014-09-06 08:00", "2014-11-07 00:00"),
        ChronologicalBlock("clean_calibration", "val", 0, 480, "2014-11-07 01:00", "2014-11-27 00:00"),
        ChronologicalBlock("development_injection", "val", 480, 960, "2014-11-27 01:00", "2014-12-17 00:00"),
        ChronologicalBlock("sealed_final_injection", "val", 960, 1_440, "2014-12-17 01:00", "2015-01-06 00:00"),
    )


def _require_batadal_block_layout(blocks: tuple[ChronologicalBlock, ...]) -> None:
    expected = tuple(
        (
            block.name,
            block.source_split,
            block.row_start,
            block.row_stop,
        )
        for block in _batadal_blocks()
    )
    actual = tuple(
        (block.name, block.source_split, block.row_start, block.row_stop)
        for block in blocks
    )
    if actual != expected:
        raise ValueError("blocks must use the sealed BATADAL names, splits, and row ranges")


def _generator_spec() -> dict[str, object]:
    return {
        "attack_family": "finite_signed_step",
        "duration_unit": "hours",
        "duration_fit_boundary_rule": "onset_index + duration_hours <= block_row_stop",
        "eligible_blocks": {
            "development": "development_injection",
            "final": "sealed_final_injection",
        },
        "no_replacement_hash_ranking": {
            "algorithm": "sha256_seeded_rank",
            "version": "1",
        },
        "onset_selection_strategy": "regime_stratified_without_replacement",
        "regime_feature": "absolute_first_difference",
        "regime_quantiles": [1 / 3, 2 / 3],
        "regime_source_block": "model_fit",
        "seed_role": "injection",
        "signs": [-1, 1],
        "target_sensor": "L_T1",
    }


def _metric_spec() -> dict[str, object]:
    return {
        "acceptance_thresholds": _acceptance_as_dict(
            AcceptanceThresholds(0.20, 0.25, 0.95, 1.0, 0.90, 0.85, 0.95, 0, True)
        ),
        "constraint_reporting": {
            "hard": "count_zero_required",
            "soft": "report_by_source_and_severity",
        },
        "delta_trajectory_nrmse": {
            "aggregation": "median",
            "denominator": "absolute_injected_amplitude",
            "unit": "dimensionless",
        },
        "improvement": {
            "causal_comparison": "best_eligible_causal_baseline",
            "formula": "(baseline_nrmse - estimator_nrmse) / baseline_nrmse",
        },
        "paired_bootstrap": {
            "confidence_interval": "percentile_95",
            "replicates": 10_000,
            "seed_role": "uncertainty",
            "unit": "scenario_pair",
        },
        "posterior_interval_coverage": {
            "coverage": "pointwise_90_percent_support_coverage",
            "nominal": 0.90,
            "sharpness": "median_interval_width",
        },
        "sign_accuracy": {
            "eligibility": "absolute_injected_amplitude_train_std >= 1.0",
            "minimum_magnitude_train_std": 1.0,
            "unit": "proportion",
        },
        "signed_amplitude_error": {
            "error": "absolute_signed_error",
            "estimator": "median_over_declared_support",
            "unit": "train_std",
        },
    }


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _json_sha256(serialized: str) -> str:
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_spec_seal(serialized: object, digest: object, field_name: str) -> None:
    if not isinstance(serialized, str):
        raise ValueError(f"{field_name}_json must be a string")
    try:
        payload = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field_name}_json must contain valid JSON") from error
    if not isinstance(payload, dict) or serialized != _canonical_json(payload):
        raise ValueError(f"{field_name}_json must be canonical compact sorted JSON")
    if not isinstance(digest, str) or digest != _json_sha256(serialized):
        raise ValueError(f"{field_name}_sha256 must match canonical JSON")


def _validate_generator_spec(serialized: str) -> None:
    if json.loads(serialized) != _generator_spec():
        raise ValueError("generator_spec_json does not match the sealed generator contract")


def _validate_metric_spec(serialized: str, acceptance: AcceptanceThresholds) -> None:
    expected = _metric_spec()
    expected["acceptance_thresholds"] = _acceptance_as_dict(acceptance)
    if json.loads(serialized) != expected:
        raise ValueError("metric_spec_json does not match the sealed metric contract")


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


def _require_nonnegative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_positive_int(value: object, field_name: str) -> None:
    _require_nonnegative_int(value, field_name)
    if value == 0:
        raise ValueError(f"{field_name} must be positive")


def _require_nonnegative_number(value: object, field_name: str) -> None:
    is_invalid = (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    )
    if is_invalid:
        raise ValueError(f"{field_name} must be a non-negative number")


def _require_positive_numeric_tuple(values: object, field_name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    for value in values:
        _require_nonnegative_number(value, field_name)
        if value == 0:
            raise ValueError(f"{field_name} values must be positive")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique")


def _require_positive_int_tuple(values: object, field_name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    for value in values:
        _require_positive_int(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique")


def _require_text_tuple(values: object, field_name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    for value in values:
        _require_text(value, field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} values must be unique")


def _parse_timestamp(value: object, field_name: str) -> datetime:
    _require_text(value, field_name)
    try:
        return datetime.strptime(value, _TIMESTAMP_FORMAT)
    except ValueError as error:
        raise ValueError(f"{field_name} must use YYYY-MM-DD HH:MM") from error


def _require_blocks(blocks: object) -> None:
    if not isinstance(blocks, tuple) or len(blocks) != 5:
        raise ValueError("blocks must be a tuple of five chronological blocks")
    if not all(isinstance(block, ChronologicalBlock) for block in blocks):
        raise ValueError("blocks must contain ChronologicalBlock instances")
    names = tuple(block.name for block in blocks)
    if len(set(names)) != len(names):
        raise ValueError("block names must be unique")


def _require_seed_mapping(seeds: object) -> None:
    if not isinstance(seeds, Mapping) or not seeds:
        raise ValueError("seeds must be a non-empty mapping")
    for name, seed in seeds.items():
        _require_text(name, "seed name")
        _require_nonnegative_int(seed, "seed")
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("seed values must be unique")
    if set(seeds) != {"dynamics", "injection", "uncertainty"}:
        raise ValueError("seeds must use the exact dynamics, injection, and uncertainty roles")


def _require_hash_mapping(hashes: object, field_name: str) -> None:
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    for name, digest in hashes.items():
        _require_text(name, field_name)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{field_name} values must be lowercase SHA-256 digests")


def _require_artifact_roles(hashes: Mapping[str, str]) -> None:
    if not set(_ARTIFACT_HASHES).issubset(hashes):
        raise ValueError("artifact_hashes must include every base BATADAL artifact role")
    allowed_roles = set(_ARTIFACT_HASHES) | set(_REQUIRED_PROVENANCE)
    if not set(hashes).issubset(allowed_roles):
        raise ValueError("artifact_hashes contains an unsupported role")
    for role, expected_hash in _ARTIFACT_HASHES.items():
        if hashes[role] != expected_hash:
            raise ValueError("base BATADAL artifact hashes must match the sealed values")


def _require_provenance_roles(provenance: Mapping[str, str]) -> None:
    allowed_roles = set(_REQUIRED_PROVENANCE)
    if not set(provenance).issubset(allowed_roles):
        raise ValueError("required_provenance_hashes has an unsupported role")


def _validate_readiness(readiness: SealedReadiness, hashes: Mapping[str, str]) -> None:
    expected_missing = tuple(key for key in _REQUIRED_PROVENANCE if key not in hashes)
    expected_ready = not expected_missing
    expected_status = "ready" if expected_ready else "incomplete_missing_required_artifacts"
    if readiness.missing_required_artifacts != expected_missing:
        raise ValueError("readiness missing artifacts must be recomputed from artifact_hashes")
    if readiness.ready != expected_ready or readiness.status != expected_status:
        raise ValueError("readiness must match the artifact_hashes provenance state")


def _block_as_dict(block: ChronologicalBlock) -> dict[str, object]:
    return {
        "name": block.name,
        "source_split": block.source_split,
        "row_start": block.row_start,
        "row_stop": block.row_stop,
        "start_timestamp": block.start_timestamp,
        "end_timestamp": block.end_timestamp,
    }


def _grid_as_dict(grid: InjectionGrid) -> dict[str, object]:
    return {
        "amplitudes_train_std": list(grid.amplitudes_train_std),
        "durations_hours": list(grid.durations_hours),
        "regime_stratified_onset_count": grid.regime_stratified_onset_count,
        "onsets_are_unseen": grid.onsets_are_unseen,
        "signs": list(grid.signs),
        "onset_selection_strategy": grid.onset_selection_strategy,
        "regime_labels": list(grid.regime_labels),
    }


def _acceptance_as_dict(acceptance: AcceptanceThresholds) -> dict[str, object]:
    return dict(acceptance.__dict__)


def _readiness_as_dict(readiness: SealedReadiness) -> dict[str, object]:
    return {
        "ready": readiness.ready,
        "status": readiness.status,
        "missing_required_artifacts": list(readiness.missing_required_artifacts),
    }


def _manifest_from_payload(payload: dict[str, object]) -> BenchmarkManifest:
    expected_keys = {
        "manifest_version", "dataset", "target_sensor", "sampling_interval_seconds", "blocks",
        "development_injections", "final_injections", "seeds", "acceptance", "artifact_hashes",
        "readiness", "generator_spec_json", "generator_spec_sha256", "metric_spec_json",
        "metric_spec_sha256", "final_evaluation_count",
    }
    if set(payload) != expected_keys:
        raise ValueError("serialized manifest has an invalid schema")
    blocks = payload["blocks"]
    if not isinstance(blocks, list):
        raise ValueError("blocks must be a JSON array")
    return BenchmarkManifest(
        manifest_version=payload["manifest_version"],
        dataset=payload["dataset"],
        target_sensor=payload["target_sensor"],
        sampling_interval_seconds=payload["sampling_interval_seconds"],
        blocks=tuple(_block_from_payload(item) for item in blocks),
        development_injections=_grid_from_payload(payload["development_injections"]),
        final_injections=_grid_from_payload(payload["final_injections"]),
        seeds=_mapping_value(payload["seeds"], "seeds"),
        acceptance=_acceptance_from_payload(payload["acceptance"]),
        artifact_hashes=_mapping_value(payload["artifact_hashes"], "artifact_hashes"),
        readiness=_readiness_from_payload(payload["readiness"]),
        generator_spec_json=payload["generator_spec_json"],
        generator_spec_sha256=payload["generator_spec_sha256"],
        metric_spec_json=payload["metric_spec_json"],
        metric_spec_sha256=payload["metric_spec_sha256"],
        final_evaluation_count=payload["final_evaluation_count"],
    )


def _block_from_payload(payload: object) -> ChronologicalBlock:
    mapping = _mapping_value(payload, "block")
    return ChronologicalBlock(**mapping)


def _grid_from_payload(payload: object) -> InjectionGrid:
    mapping = _mapping_value(payload, "injection grid")
    mapping["amplitudes_train_std"] = tuple(mapping["amplitudes_train_std"])
    mapping["durations_hours"] = tuple(mapping["durations_hours"])
    mapping["signs"] = tuple(mapping["signs"])
    mapping["regime_labels"] = tuple(mapping["regime_labels"])
    return InjectionGrid(**mapping)


def _acceptance_from_payload(payload: object) -> AcceptanceThresholds:
    return AcceptanceThresholds(**_mapping_value(payload, "acceptance"))


def _readiness_from_payload(payload: object) -> SealedReadiness:
    mapping = _mapping_value(payload, "readiness")
    mapping["missing_required_artifacts"] = tuple(mapping["missing_required_artifacts"])
    return SealedReadiness(**mapping)


def _mapping_value(payload: object, field_name: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return dict(payload)
