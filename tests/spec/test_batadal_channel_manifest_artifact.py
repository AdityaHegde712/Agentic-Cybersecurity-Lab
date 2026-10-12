"""LOCKED artifact-writing contract for the sealed BATADAL channel manifest."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from scripts.build_batadal_channel_manifest import (
    DEFAULT_MANIFEST_OUTPUT_PATH,
    FIT_ROW_COUNT,
    write_manifest,
)
from src.inverse.channel_constraints import (
    ChannelConstraint,
    ChannelConstraintManifest,
    EmpiricalBounds,
    EmpiricalRate,
    TopologyMapping,
)


def _manifest() -> ChannelConstraintManifest:
    channel = ChannelConstraint(
        name="L_T1",
        role="target_measurement",
        unit="unknown",
        value_type="continuous",
        attackability="attackable",
        trust_assumption="unknown",
        topology_mapping=TopologyMapping(
            public_reference="C-Town:T1",
            dataset_alignment_validated=False,
        ),
        empirical_bounds=EmpiricalBounds(
            minimum=0.0,
            maximum=1.0,
            source_block="model_fit",
        ),
        empirical_rate=EmpiricalRate(
            maximum_absolute_change_per_step=0.25,
            source_block="model_fit",
        ),
        constraint_mode="soft",
        constraint_enforcement="soft",
        source_provenance=("local model-fit data", "C-Town public topology"),
    )
    return ChannelConstraintManifest(
        manifest_version="1.0",
        dataset="batadal",
        dataset_channels=("L_T1",),
        channels=(channel,),
    )


def test_writer_creates_utf8_lf_artifact_with_sha256_of_exact_bytes(tmp_path: Path) -> None:
    manifest = _manifest()
    output_path = tmp_path / "sealed" / "batadal-channel-manifest-v1.json"

    digest = write_manifest(manifest, output_path)

    exact_bytes = output_path.read_bytes()
    text = exact_bytes.decode("utf-8")
    assert output_path.parent.is_dir()
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    assert ChannelConstraintManifest.from_json(text.rstrip("\n")) == manifest
    assert digest == hashlib.sha256(exact_bytes).hexdigest()
    assert len(digest) == 64
    assert digest == digest.lower()


def test_writer_is_reproducible_and_leaves_only_the_stable_final_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "sealed" / "batadal-channel-manifest-v1.json"
    manifest = _manifest()

    first_digest = write_manifest(manifest, output_path)
    first_bytes = output_path.read_bytes()
    second_digest = write_manifest(manifest, output_path)
    assert output_path.read_bytes() == first_bytes
    assert second_digest == first_digest

    replacement = replace(manifest, manifest_version="1.1")
    replacement_digest = write_manifest(replacement, output_path)
    assert replacement_digest != first_digest
    assert ChannelConstraintManifest.from_json(
        output_path.read_text(encoding="utf-8").rstrip("\n")
    ) == replacement
    assert [path.name for path in output_path.parent.iterdir()] == [output_path.name]


def test_manifest_artifact_defaults_are_sealed_to_the_authorized_fit_block() -> None:
    assert FIT_ROW_COUNT == 5_840
    assert not DEFAULT_MANIFEST_OUTPUT_PATH.is_absolute()
    assert DEFAULT_MANIFEST_OUTPUT_PATH.parent == Path("configs/manifests")
    assert DEFAULT_MANIFEST_OUTPUT_PATH.name.startswith("batadal-channel-manifest-v")
    assert DEFAULT_MANIFEST_OUTPUT_PATH.suffix == ".json"
