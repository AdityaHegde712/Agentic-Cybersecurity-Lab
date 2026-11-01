"""LOCKED deterministic benchmark-definition contract for narrowed BATADAL work."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from src.inverse.benchmark_manifest import (
    BenchmarkManifest,
    build_batadal_benchmark_manifest,
)


EXPECTED_ARTIFACT_HASHES = {
    "batadal_train_parquet": "6a7c7232b5b24eb7b475dcca3d20e6fc6ada93f4f50769af454e1e5a9560d343",
    "batadal_val_parquet": "98a12650fc75cec6554e61f7ede2e09266d3858821b3e7faa973647ee2dc9967",
    "batadal_metadata_json": "d8d0bec8f1a93bf711939a8c017820fe26c9bac9a943b439c371530eacda0974",
    "channel_constraint_manifest": "e20b7728cbf4d22ea6fd6dbd4f5eca0b26f65a9165108fa8f8d8d890ad0bd347",
}


def test_batadal_manifest_locks_exact_chronological_blocks_and_injection_grids() -> None:
    manifest = build_batadal_benchmark_manifest()

    assert manifest.dataset == "batadal"
    assert manifest.target_sensor == "L_T1"
    assert manifest.sampling_interval_seconds == 3_600
    assert [
        (
            block.name,
            block.source_split,
            block.row_start,
            block.row_stop,
            block.start_timestamp,
            block.end_timestamp,
        )
        for block in manifest.blocks
    ] == [
        ("model_fit", "train", 0, 5_840, "2014-01-06 00:00", "2014-09-06 07:00"),
        ("model_selection", "train", 5_840, 7_321, "2014-09-06 08:00", "2014-11-07 00:00"),
        ("clean_calibration", "val", 0, 480, "2014-11-07 01:00", "2014-11-27 00:00"),
        ("development_injection", "val", 480, 960, "2014-11-27 01:00", "2014-12-17 00:00"),
        ("sealed_final_injection", "val", 960, 1_440, "2014-12-17 01:00", "2015-01-06 00:00"),
    ]
    assert [block.row_count for block in manifest.blocks] == [5_840, 1_481, 480, 480, 480]
    assert manifest.validate_chronology() is None
    assert manifest.development_injections.amplitudes_train_std == (0.5, 1.0, 2.0)
    assert manifest.development_injections.durations_hours == (6, 12, 24, 48)
    assert manifest.development_injections.regime_stratified_onset_count == 5
    assert manifest.final_injections.amplitudes_train_std == (0.75, 1.5, 3.0)
    assert manifest.final_injections.durations_hours == (9, 18, 36)
    assert manifest.final_injections.regime_stratified_onset_count == 10
    assert manifest.final_injections.onsets_are_unseen is True


def test_manifest_locks_nonnegative_unique_named_seeds_and_acceptance_thresholds() -> None:
    manifest = build_batadal_benchmark_manifest()

    assert manifest.seeds
    assert all(isinstance(name, str) and name for name in manifest.seeds)
    assert all(isinstance(seed, int) and seed >= 0 for seed in manifest.seeds.values())
    assert len(set(manifest.seeds.values())) == len(manifest.seeds)
    assert manifest.acceptance.minimum_median_delta_nrmse_improvement == 0.20
    assert manifest.acceptance.maximum_median_signed_amplitude_error_train_std == 0.25
    assert manifest.acceptance.minimum_sign_accuracy == 0.95
    assert manifest.acceptance.minimum_sign_magnitude_train_std == 1.0
    assert manifest.acceptance.nominal_interval_coverage == 0.90
    assert manifest.acceptance.minimum_empirical_interval_coverage == 0.85
    assert manifest.acceptance.maximum_empirical_interval_coverage == 0.95
    assert manifest.acceptance.maximum_hard_constraint_violations == 0
    assert manifest.acceptance.require_causal_baseline_improvement is True

    with pytest.raises(ValueError):
        build_batadal_benchmark_manifest(seeds={"dynamics": -1})
    with pytest.raises(ValueError):
        build_batadal_benchmark_manifest(seeds={"dynamics": 7, "injection": 7})


def test_manifest_records_current_artifacts_but_remains_incomplete_without_required_provenance() -> None:
    manifest = build_batadal_benchmark_manifest()

    assert manifest.artifact_hashes == EXPECTED_ARTIFACT_HASHES
    assert manifest.readiness.ready is False
    assert manifest.readiness.status == "incomplete_missing_required_artifacts"
    assert set(manifest.readiness.missing_required_artifacts) == {
        "selected_checkpoint_sha256",
        "estimator_config_sha256",
        "baseline_config_sha256",
    }
    assert manifest.final_evaluation_count == 0
    with pytest.raises(ValueError):
        manifest.require_sealed_readiness()
    with pytest.raises(ValueError):
        build_batadal_benchmark_manifest(claim_sealed_readiness=True)


def test_manifest_json_and_hash_are_canonical_and_deterministic() -> None:
    manifest = build_batadal_benchmark_manifest()

    serialized = manifest.to_json()
    restored = BenchmarkManifest.from_json(serialized)

    assert restored == manifest
    assert restored.to_json() == serialized
    assert manifest.sha256() == hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    assert manifest.sha256() == manifest.sha256().lower()


def test_seeds_and_artifacts_are_immutable_and_have_exact_required_roles() -> None:
    manifest = build_batadal_benchmark_manifest()

    assert set(manifest.seeds) == {"dynamics", "injection", "uncertainty"}
    with pytest.raises(TypeError):
        manifest.seeds["dynamics"] = 0
    with pytest.raises(TypeError):
        manifest.artifact_hashes["batadal_train_parquet"] = "a" * 64
    with pytest.raises(ValueError):
        build_batadal_benchmark_manifest(seeds={"dynamics": 1, "injection": 2})
    with pytest.raises(ValueError):
        build_batadal_benchmark_manifest(
            seeds={"dynamics": 1, "injection": 2, "other": 3}
        )


def test_ready_manifest_merges_all_required_provenance_and_base_artifact_roles() -> None:
    required_provenance = {
        "selected_checkpoint_sha256": "a" * 64,
        "estimator_config_sha256": "b" * 64,
        "baseline_config_sha256": "c" * 64,
    }
    manifest = build_batadal_benchmark_manifest(
        required_provenance_hashes=required_provenance,
        claim_sealed_readiness=True,
    )

    assert manifest.readiness.ready is True
    assert manifest.readiness.status == "ready"
    assert manifest.artifact_hashes == EXPECTED_ARTIFACT_HASHES | required_provenance
    assert BenchmarkManifest.from_json(manifest.to_json()).artifact_hashes == (
        EXPECTED_ARTIFACT_HASHES | required_provenance
    )
    with pytest.raises(ValueError):
        build_batadal_benchmark_manifest(
            artifact_hashes={"batadal_train_parquet": EXPECTED_ARTIFACT_HASHES["batadal_train_parquet"]},
            required_provenance_hashes=required_provenance,
            claim_sealed_readiness=True,
        )
    with pytest.raises(ValueError):
        build_batadal_benchmark_manifest(
            required_provenance_hashes={"selected_checkpoint_sha256": "not-a-hash"},
            claim_sealed_readiness=True,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("minimum_sign_accuracy", 1.01),
        ("nominal_interval_coverage", 1.01),
        ("minimum_empirical_interval_coverage", 1.01),
        ("maximum_empirical_interval_coverage", 1.01),
    ],
)
def test_acceptance_probability_fields_must_be_in_unit_interval(
    field: str,
    invalid_value: float,
) -> None:
    acceptance = build_batadal_benchmark_manifest().acceptance

    with pytest.raises(ValueError):
        replace(acceptance, **{field: invalid_value})


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("amplitudes_train_std", (0.0, 1.0)),
        ("amplitudes_train_std", (0.5, 0.5)),
        ("durations_hours", (0, 6)),
        ("durations_hours", (6, 6)),
    ],
)
def test_injection_grid_amplitudes_and_durations_must_be_positive_and_unique(
    field: str,
    invalid_value: tuple[float, ...] | tuple[int, ...],
) -> None:
    grid = build_batadal_benchmark_manifest().development_injections

    with pytest.raises(ValueError):
        replace(grid, **{field: invalid_value})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["blocks"][0].update({"name": "wrong_first_block"}),
        lambda payload: payload["blocks"][0].update({"source_split": "val"}),
        lambda payload: [
            payload["blocks"].__setitem__(0, payload["blocks"][1]),
            payload["blocks"].__setitem__(1, payload["blocks"][0]),
        ],
    ],
)
def test_serialized_manifest_rejects_semantic_block_changes_despite_valid_chronology(
    mutate: object,
) -> None:
    payload = json.loads(build_batadal_benchmark_manifest().to_json())
    mutate(payload)  # type: ignore[operator]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    with pytest.raises(ValueError):
        BenchmarkManifest.from_json(serialized)


def test_injection_grids_seal_signed_step_onset_rules_and_regimes() -> None:
    manifest = build_batadal_benchmark_manifest()

    for grid in (manifest.development_injections, manifest.final_injections):
        assert grid.signs == (-1, 1)
        assert grid.onset_selection_strategy == "regime_stratified_without_replacement"
        assert grid.regime_labels
        assert all(isinstance(label, str) and label for label in grid.regime_labels)


def test_generator_and_metric_specs_are_canonical_hashed_replay_seals() -> None:
    manifest = build_batadal_benchmark_manifest()
    generator_spec = json.loads(manifest.generator_spec_json)
    metric_spec = json.loads(manifest.metric_spec_json)

    assert manifest.generator_spec_json == json.dumps(
        generator_spec, sort_keys=True, separators=(",", ":")
    )
    assert manifest.metric_spec_json == json.dumps(
        metric_spec, sort_keys=True, separators=(",", ":")
    )
    assert manifest.generator_spec_sha256 == hashlib.sha256(
        manifest.generator_spec_json.encode("utf-8")
    ).hexdigest()
    assert manifest.metric_spec_sha256 == hashlib.sha256(
        manifest.metric_spec_json.encode("utf-8")
    ).hexdigest()
    assert generator_spec["target_sensor"] == "L_T1"
    assert generator_spec["attack_family"] == "finite_signed_step"
    assert generator_spec["signs"] == [-1, 1]
    assert generator_spec["onset_selection_strategy"] == (
        "regime_stratified_without_replacement"
    )
    assert generator_spec["duration_unit"] == "hours"
    assert metric_spec["delta_trajectory_nrmse"]["aggregation"] == "median"
    assert metric_spec["signed_amplitude_error"]["unit"] == "train_std"
    assert metric_spec["sign_accuracy"]["minimum_magnitude_train_std"] == 1.0
    assert metric_spec["posterior_interval_coverage"]["nominal"] == 0.90
    assert metric_spec["acceptance_thresholds"] == json.loads(
        json.dumps(manifest.acceptance.__dict__, sort_keys=True)
    )
    assert BenchmarkManifest.from_json(manifest.to_json()) == manifest


@pytest.mark.parametrize("spec_key", ["generator", "metric"])
def test_serialized_manifest_rejects_mismatched_spec_digest_or_noncanonical_spec(
    spec_key: str,
) -> None:
    manifest = build_batadal_benchmark_manifest()
    payload = json.loads(manifest.to_json())
    json_key = f"{spec_key}_spec_json"
    digest_key = f"{spec_key}_spec_sha256"

    mismatched_digest = dict(payload)
    mismatched_digest[digest_key] = "a" * 64
    with pytest.raises(ValueError):
        BenchmarkManifest.from_json(
            json.dumps(mismatched_digest, sort_keys=True, separators=(",", ":"))
        )

    noncanonical_spec = dict(payload)
    parsed_spec = json.loads(noncanonical_spec[json_key])
    noncanonical_spec[json_key] = json.dumps(parsed_spec, sort_keys=True)
    noncanonical_spec[digest_key] = hashlib.sha256(
        noncanonical_spec[json_key].encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError):
        BenchmarkManifest.from_json(
            json.dumps(noncanonical_spec, sort_keys=True, separators=(",", ":"))
        )


def test_deserialization_recomputes_readiness_and_rejects_base_artifact_drift() -> None:
    manifest = build_batadal_benchmark_manifest()
    false_ready_payload = json.loads(manifest.to_json())
    false_ready_payload["readiness"] = {
        "ready": True,
        "status": "ready",
        "missing_required_artifacts": [],
    }
    with pytest.raises(ValueError):
        BenchmarkManifest.from_json(
            json.dumps(false_ready_payload, sort_keys=True, separators=(",", ":"))
        )

    with pytest.raises(ValueError):
        build_batadal_benchmark_manifest(
            artifact_hashes=EXPECTED_ARTIFACT_HASHES | {"unsupported_role": "a" * 64}
        )
    drifted_hashes = dict(EXPECTED_ARTIFACT_HASHES)
    drifted_hashes["batadal_train_parquet"] = "a" * 64
    with pytest.raises(ValueError):
        build_batadal_benchmark_manifest(artifact_hashes=drifted_hashes)


def test_generator_spec_seals_all_deterministic_regime_and_onset_replay_inputs() -> None:
    generator_spec = json.loads(
        build_batadal_benchmark_manifest().generator_spec_json
    )

    assert generator_spec["regime_feature"] == "absolute_first_difference"
    assert generator_spec["regime_source_block"] == "model_fit"
    assert generator_spec["regime_quantiles"] == [1 / 3, 2 / 3]
    assert generator_spec["eligible_blocks"] == {
        "development": "development_injection",
        "final": "sealed_final_injection",
    }
    assert generator_spec["duration_fit_boundary_rule"] == (
        "onset_index + duration_hours <= block_row_stop"
    )
    assert generator_spec["no_replacement_hash_ranking"] == {
        "algorithm": "sha256_seeded_rank",
        "version": "1",
    }
    assert generator_spec["seed_role"] == "injection"


def test_metric_spec_seals_formulas_units_uncertainty_and_constraint_reporting() -> None:
    metric_spec = json.loads(build_batadal_benchmark_manifest().metric_spec_json)

    assert metric_spec["delta_trajectory_nrmse"] == {
        "aggregation": "median",
        "denominator": "absolute_injected_amplitude",
        "unit": "dimensionless",
    }
    assert metric_spec["signed_amplitude_error"] == {
        "estimator": "median_over_declared_support",
        "error": "absolute_signed_error",
        "unit": "train_std",
    }
    assert metric_spec["sign_accuracy"] == {
        "eligibility": "absolute_injected_amplitude_train_std >= 1.0",
        "minimum_magnitude_train_std": 1.0,
        "unit": "proportion",
    }
    assert metric_spec["posterior_interval_coverage"] == {
        "coverage": "pointwise_90_percent_support_coverage",
        "nominal": 0.90,
        "sharpness": "median_interval_width",
    }
    assert metric_spec["paired_bootstrap"] == {
        "confidence_interval": "percentile_95",
        "replicates": 10_000,
        "seed_role": "uncertainty",
        "unit": "scenario_pair",
    }
    assert metric_spec["improvement"] == {
        "causal_comparison": "best_eligible_causal_baseline",
        "formula": "(baseline_nrmse - estimator_nrmse) / baseline_nrmse",
    }
    assert metric_spec["constraint_reporting"] == {
        "hard": "count_zero_required",
        "soft": "report_by_source_and_severity",
    }
