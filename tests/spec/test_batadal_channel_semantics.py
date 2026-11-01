"""LOCKED semantic and provenance contract for BATADAL channel manifests."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from src.inverse.batadal_manifest import build_batadal_channel_manifest


BATADAL_CHANNELS = (
    "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7",
    "F_PU1", "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4",
    "S_PU4", "F_PU5", "S_PU5", "F_PU6", "S_PU6", "F_PU7", "S_PU7",
    "F_PU8", "S_PU8", "F_PU9", "S_PU9", "F_PU10", "S_PU10", "F_PU11",
    "S_PU11", "F_V2", "S_V2", "P_J280", "P_J269", "P_J300", "P_J256",
    "P_J289", "P_J415", "P_J302", "P_J306", "P_J307", "P_J317", "P_J14",
    "P_J422",
)


def _fit_values() -> pd.DataFrame:
    values: dict[str, list[float]] = {}
    for index, name in enumerate(BATADAL_CHANNELS):
        values[name] = [0.0, 1.0, 1.0] if name.startswith("S_") else [float(index), float(index + 1), float(index + 2)]
    return pd.DataFrame(values, columns=BATADAL_CHANNELS)


def _channels_by_name() -> dict[str, object]:
    manifest = build_batadal_channel_manifest(_fit_values(), BATADAL_CHANNELS)
    return {channel.name: channel for channel in manifest.channels}


def test_candidate_units_are_explicit_but_stored_units_remain_unknown() -> None:
    channels = _channels_by_name()

    assert all(channel.unit == "unknown" for channel in channels.values())
    assert channels["L_T1"].candidate_public_unit == "m"
    assert channels["P_J280"].candidate_public_unit == "m"
    assert channels["F_PU1"].candidate_public_unit == "L/s"
    assert channels["S_PU1"].candidate_public_unit == "dimensionless"


def test_topology_references_use_canonical_public_entity_ids_without_validation() -> None:
    channels = _channels_by_name()

    assert channels["L_T1"].topology_mapping.public_reference == "C-Town:T1"
    assert channels["F_PU1"].topology_mapping.public_reference == "C-Town:PU1"
    assert channels["S_PU1"].topology_mapping.public_reference == "C-Town:PU1"
    assert channels["F_V2"].topology_mapping.public_reference == "C-Town:V2"
    assert channels["S_V2"].topology_mapping.public_reference == "C-Town:V2"
    assert channels["P_J280"].topology_mapping.public_reference == "C-Town:J280"
    assert all(
        not channel.topology_mapping.dataset_alignment_validated
        for channel in channels.values()
    )


def test_each_channel_records_scoped_local_and_public_source_evidence() -> None:
    channels = _channels_by_name()

    for channel in channels.values():
        assert any("model-fit" in source.lower() for source in channel.source_provenance)
        assert any(
            "http" in source.lower() or "c-town" in source.lower()
            for source in channel.source_provenance
        )
        assert channel.evidence_scope == "model_fit_only"


def test_constraint_mode_and_enforcement_cannot_diverge_or_harden_without_validation() -> None:
    channel = _channels_by_name()["L_T1"]

    with pytest.raises(ValueError):
        replace(channel, constraint_enforcement="hard")
    with pytest.raises(ValueError):
        replace(channel, constraint_mode="hard", constraint_enforcement="hard")
