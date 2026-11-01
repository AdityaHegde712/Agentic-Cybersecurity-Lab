"""LOCKED contract for the typed inverse-benchmark channel manifest."""

from __future__ import annotations

import json

import pytest

from src.inverse.channel_constraints import (
    ChannelConstraint,
    ChannelConstraintManifest,
    EmpiricalBounds,
    EmpiricalRate,
    TopologyMapping,
)


def _channel(
    name: str,
    *,
    trust_assumption: str = "independently_trusted",
    constraint_mode: str = "soft",
    topology_validated: bool = True,
    provenance: tuple[str, ...] = ("BATADAL normal model-fit block",),
) -> ChannelConstraint:
    return ChannelConstraint(
        name=name,
        role="target_measurement" if name == "L_T1" else "recorded_companion",
        unit="m",
        value_type="continuous",
        attackability="attackable" if name == "L_T1" else "declared_unaffected",
        trust_assumption=trust_assumption,
        topology_mapping=TopologyMapping(
            public_reference="C-Town:T1" if name == "L_T1" else None,
            dataset_alignment_validated=topology_validated,
        ),
        empirical_bounds=EmpiricalBounds(
            minimum=0.25,
            maximum=5.00,
            source_block="model_fit",
        ),
        empirical_rate=EmpiricalRate(
            maximum_absolute_change_per_step=0.20,
            source_block="model_fit",
        ),
        constraint_mode=constraint_mode,
        source_provenance=provenance,
    )


def _manifest(*channels: ChannelConstraint) -> ChannelConstraintManifest:
    return ChannelConstraintManifest(
        manifest_version="1.0",
        dataset="batadal",
        dataset_channels=("L_T1", "F_PU1"),
        channels=channels,
    )


def test_manifest_requires_each_declared_dataset_channel_exactly_once() -> None:
    l_t1 = _channel("L_T1")
    f_pu1 = _channel("F_PU1")

    manifest = _manifest(l_t1, f_pu1)

    assert [channel.name for channel in manifest.channels] == ["L_T1", "F_PU1"]
    with pytest.raises(ValueError):
        _manifest(l_t1)
    with pytest.raises(ValueError):
        _manifest(l_t1, l_t1)


def test_channel_schema_captures_semantics_constraints_and_provenance() -> None:
    channel = _channel("L_T1")

    assert channel.name == "L_T1"
    assert channel.role == "target_measurement"
    assert channel.unit == "m"
    assert channel.value_type == "continuous"
    assert channel.attackability == "attackable"
    assert channel.trust_assumption == "independently_trusted"
    assert channel.constraint_mode == "soft"
    assert channel.source_provenance == ("BATADAL normal model-fit block",)
    assert channel.empirical_bounds.source_block == "model_fit"
    assert channel.empirical_rate.source_block == "model_fit"


def test_manifest_rejects_non_model_fit_empirical_limits() -> None:
    invalid = _channel("L_T1")
    invalid_bounds = EmpiricalBounds(
        minimum=0.25,
        maximum=5.00,
        source_block="model_selection",
    )

    with pytest.raises(ValueError):
        ChannelConstraint(
            name=invalid.name,
            role=invalid.role,
            unit=invalid.unit,
            value_type=invalid.value_type,
            attackability=invalid.attackability,
            trust_assumption=invalid.trust_assumption,
            topology_mapping=invalid.topology_mapping,
            empirical_bounds=invalid_bounds,
            empirical_rate=invalid.empirical_rate,
            constraint_mode=invalid.constraint_mode,
            source_provenance=invalid.source_provenance,
        )


def test_manifest_rejects_correlation_trust_and_unvalidated_hard_physics() -> None:
    with pytest.raises(ValueError):
        _channel(
            "L_T1",
            trust_assumption="trusted_from_correlation",
            provenance=("correlation analysis",),
        )
    with pytest.raises(ValueError):
        _channel("L_T1", constraint_mode="hard", topology_validated=False)


def test_manifest_json_round_trip_is_deterministic() -> None:
    manifest = _manifest(_channel("L_T1"), _channel("F_PU1"))

    serialized = manifest.to_json()
    restored = ChannelConstraintManifest.from_json(serialized)

    assert restored == manifest
    assert restored.to_json() == serialized
    assert list(json.loads(serialized)) == [
        "channels",
        "dataset",
        "dataset_channels",
        "manifest_version",
    ]
