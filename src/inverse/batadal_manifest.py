"""BATADAL channel-manifest builder based solely on supplied model-fit data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.inverse.channel_constraints import (
    ChannelConstraint,
    ChannelConstraintManifest,
    EmpiricalBounds,
    EmpiricalRate,
    TopologyMapping,
)


BATADAL_CHANNELS = (
    "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7", "F_PU1",
    "S_PU1", "F_PU2", "S_PU2", "F_PU3", "S_PU3", "F_PU4", "S_PU4",
    "F_PU5", "S_PU5", "F_PU6", "S_PU6", "F_PU7", "S_PU7", "F_PU8",
    "S_PU8", "F_PU9", "S_PU9", "F_PU10", "S_PU10", "F_PU11", "S_PU11",
    "F_V2", "S_V2", "P_J280", "P_J269", "P_J300", "P_J256", "P_J289",
    "P_J415", "P_J302", "P_J306", "P_J307", "P_J317", "P_J14", "P_J422",
)

_TARGET_CHANNEL = "L_T1"
_THREAT_MODEL_PROVENANCE = (
    "BATADAL model-fit block supplied to build_batadal_channel_manifest",
    "C-Town candidate topology: https://www.epa.gov/water-research/ctown",
    "BATADAL benchmark threat model: L_T1 attackable; other channels declared unaffected",
)


def build_batadal_channel_manifest(
    fit_values: pd.DataFrame,
    dataset_channels: tuple[str, ...],
) -> ChannelConstraintManifest:
    """Build a soft, candidate-only manifest from the model-fit block."""
    _validate_inputs(fit_values, dataset_channels)
    channels = tuple(
        _build_channel(name, fit_values[name]) for name in dataset_channels
    )
    return ChannelConstraintManifest(
        manifest_version="1.0",
        dataset="batadal",
        dataset_channels=dataset_channels,
        channels=channels,
    )


def _validate_inputs(fit_values: pd.DataFrame, dataset_channels: tuple[str, ...]) -> None:
    if not isinstance(fit_values, pd.DataFrame):
        raise ValueError("fit_values must be a pandas DataFrame")
    if fit_values.empty:
        raise ValueError("fit_values must contain at least one row")
    if dataset_channels != BATADAL_CHANNELS:
        raise ValueError("dataset_channels must match the canonical BATADAL channel order")
    if tuple(fit_values.columns) != dataset_channels:
        raise ValueError("fit_values columns must exactly match dataset_channels in order")
    for name in dataset_channels:
        values = fit_values[name]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise ValueError(f"{name} must contain numeric model-fit values")
        numeric_values = values.to_numpy(dtype=float)
        if not np.isfinite(numeric_values).all():
            raise ValueError(f"{name} must contain only finite model-fit values")
        if name.startswith("S_") and not np.isin(numeric_values, (0.0, 1.0)).all():
            raise ValueError(f"{name} status values must be observed 0 or 1")


def _build_channel(name: str, values: pd.Series) -> ChannelConstraint:
    numeric_values = values.to_numpy(dtype=float)
    maximum_rate = float(np.abs(np.diff(numeric_values)).max()) if len(values) > 1 else 0.0
    is_status = name.startswith("S_")
    return ChannelConstraint(
        name=name,
        role="target_measurement" if name == _TARGET_CHANNEL else "recorded_companion",
        unit="unknown",
        value_type="binary" if is_status else "continuous",
        attackability="attackable" if name == _TARGET_CHANNEL else "declared_unaffected",
        trust_assumption="unknown" if name == _TARGET_CHANNEL else "declared_unaffected",
        topology_mapping=TopologyMapping(
            public_reference=f"C-Town:{_public_entity_id(name)}",
            dataset_alignment_validated=False,
        ),
        empirical_bounds=EmpiricalBounds(
            minimum=float(numeric_values.min()),
            maximum=float(numeric_values.max()),
            source_block="model_fit",
        ),
        empirical_rate=EmpiricalRate(
            maximum_absolute_change_per_step=maximum_rate,
            source_block="model_fit",
        ),
        constraint_mode="soft",
        source_provenance=_THREAT_MODEL_PROVENANCE,
        semantic_status="candidate_unverified",
        stored_unit_status="candidate_unverified",
        constraint_enforcement="soft",
        evidence_scope="model_fit_only",
        candidate_public_unit=_candidate_public_unit(name),
    )


def _public_entity_id(channel_name: str) -> str:
    return channel_name.split("_", maxsplit=1)[1]


def _candidate_public_unit(channel_name: str) -> str:
    if channel_name.startswith(("L_", "P_")):
        return "m"
    if channel_name.startswith("F_"):
        return "L/s"
    return "dimensionless"
