"""LOCKED contract for the BATADAL channel-constraint manifest builder."""

from __future__ import annotations

import pandas as pd
import pytest

from src.inverse.batadal_manifest import build_batadal_channel_manifest


BATADAL_CHANNELS = (
    "L_T1",
    "L_T2",
    "L_T3",
    "L_T4",
    "L_T5",
    "L_T6",
    "L_T7",
    "F_PU1",
    "S_PU1",
    "F_PU2",
    "S_PU2",
    "F_PU3",
    "S_PU3",
    "F_PU4",
    "S_PU4",
    "F_PU5",
    "S_PU5",
    "F_PU6",
    "S_PU6",
    "F_PU7",
    "S_PU7",
    "F_PU8",
    "S_PU8",
    "F_PU9",
    "S_PU9",
    "F_PU10",
    "S_PU10",
    "F_PU11",
    "S_PU11",
    "F_V2",
    "S_V2",
    "P_J280",
    "P_J269",
    "P_J300",
    "P_J256",
    "P_J289",
    "P_J415",
    "P_J302",
    "P_J306",
    "P_J307",
    "P_J317",
    "P_J14",
    "P_J422",
)


def _fit_values() -> pd.DataFrame:
    values: dict[str, list[float]] = {}
    for index, name in enumerate(BATADAL_CHANNELS):
        if name.startswith("S_"):
            values[name] = [float(index % 2), 1.0, 1.0]
        else:
            values[name] = [float(index), float(index + 2), float(index + 1)]
    values["F_PU3"] = [0.0, 0.0, 0.0]
    return pd.DataFrame(values, columns=BATADAL_CHANNELS)


def _by_name(manifest: object) -> dict[str, object]:
    return {channel.name: channel for channel in manifest.channels}  # type: ignore[attr-defined]


def test_builder_preserves_the_complete_batadal_channel_order_and_fit_limits() -> None:
    fit_values = _fit_values()

    manifest = build_batadal_channel_manifest(fit_values, BATADAL_CHANNELS)
    channels = _by_name(manifest)

    assert tuple(channel.name for channel in manifest.channels) == BATADAL_CHANNELS
    assert manifest.dataset_channels == BATADAL_CHANNELS
    assert len(manifest.channels) == 43
    assert channels["L_T1"].empirical_bounds.minimum == 0.0
    assert channels["L_T1"].empirical_bounds.maximum == 2.0
    assert channels["L_T1"].empirical_rate.maximum_absolute_change_per_step == 2.0
    assert channels["F_PU3"].empirical_bounds.minimum == 0.0
    assert channels["F_PU3"].empirical_bounds.maximum == 0.0
    assert channels["F_PU3"].empirical_rate.maximum_absolute_change_per_step == 0.0
    assert all(channel.empirical_bounds.source_block == "model_fit" for channel in manifest.channels)
    assert all(channel.empirical_rate.source_block == "model_fit" for channel in manifest.channels)


def test_builder_declares_the_narrow_threat_model_and_unvalidated_semantics() -> None:
    manifest = build_batadal_channel_manifest(_fit_values(), BATADAL_CHANNELS)
    channels = _by_name(manifest)

    assert channels["L_T1"].role == "target_measurement"
    assert channels["L_T1"].attackability == "attackable"
    assert all(
        channel.attackability == "declared_unaffected"
        for name, channel in channels.items()
        if name != "L_T1"
    )
    assert all(channel.constraint_mode == "soft" for channel in manifest.channels)
    assert all(channel.constraint_enforcement == "soft" for channel in manifest.channels)
    assert all(channel.semantic_status == "candidate_unverified" for channel in manifest.channels)
    assert all(channel.stored_unit_status == "candidate_unverified" for channel in manifest.channels)
    assert all(channel.evidence_scope == "model_fit_only" for channel in manifest.channels)
    assert all(
        not channel.topology_mapping.dataset_alignment_validated
        for channel in manifest.channels
    )
    assert all(channel.source_provenance for channel in manifest.channels)


def test_builder_classifies_status_as_binary_only_for_observed_zero_one_values() -> None:
    fit_values = _fit_values()

    manifest = build_batadal_channel_manifest(fit_values, BATADAL_CHANNELS)

    assert _by_name(manifest)["S_PU2"].value_type == "binary"
    invalid_status_values = fit_values.copy()
    invalid_status_values.loc[1, "S_PU2"] = 0.5
    with pytest.raises(ValueError, match="S_PU2"):
        build_batadal_channel_manifest(invalid_status_values, BATADAL_CHANNELS)


def test_builder_does_not_impose_a_generic_nonnegative_flow_constraint() -> None:
    fit_values = _fit_values()
    fit_values["F_PU1"] = [-3.0, 1.0, -1.0]

    manifest = build_batadal_channel_manifest(fit_values, BATADAL_CHANNELS)

    flow = _by_name(manifest)["F_PU1"]
    assert flow.empirical_bounds.minimum == -3.0
    assert flow.constraint_mode == "soft"
