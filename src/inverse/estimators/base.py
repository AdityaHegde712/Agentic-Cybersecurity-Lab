"""Base interfaces and data structures for inverse attack estimators."""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReconstructionResult:
    """Standardized output of an inverse attack estimator."""

    x_hat: np.ndarray
    delta_hat: np.ndarray
    pointwise_uncertainty: np.ndarray | None = None
    pointwise_variance: np.ndarray | None = None
    objective_loss: float | None = None
    ambiguity_flag: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x_arr = np.asarray(self.x_hat, dtype=np.float64)
        delta_arr = np.asarray(self.delta_hat, dtype=np.float64)

        if x_arr.ndim != 1 or delta_arr.ndim != 1:
            raise ValueError("x_hat and delta_hat must be 1D arrays")
        if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(delta_arr)):
            raise ValueError("x_hat and delta_hat must contain only finite numbers")
        if len(x_arr) != len(delta_arr):
            raise ValueError(
                f"delta_hat length ({len(delta_arr)}) does not match x_hat length ({len(x_arr)})"
            )

        object.__setattr__(self, "x_hat", x_arr)
        object.__setattr__(self, "delta_hat", delta_arr)

        if self.pointwise_uncertainty is None and self.pointwise_variance is not None:
            var_arr = np.asarray(self.pointwise_variance, dtype=np.float64)
            if var_arr.ndim != 1 or len(var_arr) != len(x_arr):
                raise ValueError("pointwise_variance length must match x_hat length")
            if not np.all(np.isfinite(var_arr)) or np.any(var_arr < 0.0):
                raise ValueError("pointwise_variance must be finite and non-negative")
            unc_arr = np.sqrt(var_arr)
            object.__setattr__(self, "pointwise_variance", var_arr)
            object.__setattr__(self, "pointwise_uncertainty", unc_arr)
        elif self.pointwise_uncertainty is not None and self.pointwise_variance is None:
            unc_arr = np.asarray(self.pointwise_uncertainty, dtype=np.float64)
            if unc_arr.ndim != 1 or len(unc_arr) != len(x_arr):
                raise ValueError("pointwise_uncertainty length must match x_hat length")
            if not np.all(np.isfinite(unc_arr)) or np.any(unc_arr < 0.0):
                raise ValueError("pointwise_uncertainty must be finite and non-negative")
            var_arr = unc_arr ** 2
            object.__setattr__(self, "pointwise_uncertainty", unc_arr)
            object.__setattr__(self, "pointwise_variance", var_arr)
        elif self.pointwise_uncertainty is not None and self.pointwise_variance is not None:
            unc_arr = np.asarray(self.pointwise_uncertainty, dtype=np.float64)
            var_arr = np.asarray(self.pointwise_variance, dtype=np.float64)
            if unc_arr.ndim != 1 or len(unc_arr) != len(x_arr):
                raise ValueError("pointwise_uncertainty length must match x_hat length")
            if var_arr.ndim != 1 or len(var_arr) != len(x_arr):
                raise ValueError("pointwise_variance length must match x_hat length")
            if not np.all(np.isfinite(unc_arr)) or np.any(unc_arr < 0.0):
                raise ValueError("pointwise_uncertainty must be finite and non-negative")
            if not np.all(np.isfinite(var_arr)) or np.any(var_arr < 0.0):
                raise ValueError("pointwise_variance must be finite and non-negative")
            object.__setattr__(self, "pointwise_uncertainty", unc_arr)
            object.__setattr__(self, "pointwise_variance", var_arr)

        if not isinstance(self.ambiguity_flag, bool):
            raise ValueError("ambiguity_flag must be a boolean")

        if self.objective_loss is not None:
            if not isinstance(self.objective_loss, (int, float)) or not np.isfinite(self.objective_loss):
                raise ValueError("objective_loss must be a finite float or None")
            object.__setattr__(self, "objective_loss", float(self.objective_loss))

        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a dictionary or mapping")

    @property
    def reconstructed_y(self) -> np.ndarray:
        """The total reconstructed observed signal: x_hat + delta_hat."""
        return self.x_hat + self.delta_hat

    def residual(self, y: np.ndarray) -> np.ndarray:
        """Pointwise reconstruction error against true observation y: y - (x_hat + delta_hat)."""
        return np.asarray(y, dtype=np.float64) - self.reconstructed_y

    def __len__(self) -> int:
        return len(self.x_hat)


class BaseDeltaEstimator(abc.ABC):
    """Abstract interface for additive attack trajectory estimators."""

    @abc.abstractmethod
    def fit_normal(
        self,
        normal_data: pd.DataFrame | np.ndarray,
        **kwargs: Any,
    ) -> BaseDeltaEstimator:
        """Fit baseline normal dynamics on unattacked calibration/training data."""
        ...

    @abc.abstractmethod
    def reconstruct(
        self,
        observed: np.ndarray | pd.Series | pd.DataFrame,
        attack_interval: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> ReconstructionResult:
        """Reconstruct clean measurement x and attack delta from observed data."""
        ...

    def fit(
        self,
        normal_data: pd.DataFrame | np.ndarray,
        **kwargs: Any,
    ) -> BaseDeltaEstimator:
        """Alias for fit_normal."""
        return self.fit_normal(normal_data, **kwargs)
