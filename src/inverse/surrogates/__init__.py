"""Dynamics surrogate models for plant telemetry inversion and bake-off benchmarks."""

from __future__ import annotations

from src.inverse.surrogates.base import BasePlantSurrogate, SurrogatePrediction
from src.inverse.surrogates.dataset import (
    CompanionDataset,
    create_companion_dataloaders,
    get_batadal_companion_channels,
)
from src.inverse.surrogates.models import (
    DilatedTCN,
    DilatedTCNSurrogate,
    PhysicsGuidedMLP,
    PhysicsGuidedMLPSurrogate,
)
from src.inverse.surrogates.ridge import RidgeCompanionSurrogate

__all__ = [
    "BasePlantSurrogate",
    "SurrogatePrediction",
    "CompanionDataset",
    "create_companion_dataloaders",
    "get_batadal_companion_channels",
    "RidgeCompanionSurrogate",
    "PhysicsGuidedMLP",
    "PhysicsGuidedMLPSurrogate",
    "DilatedTCN",
    "DilatedTCNSurrogate",
]
