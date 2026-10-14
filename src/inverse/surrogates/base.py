"""Base interfaces and prediction schemas for dynamics surrogate models.

Defines the contract for surrogate models predicting target plant sensor telemetry
from uncorrupted companion sensor telemetry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SurrogatePrediction:
    """Prediction result emitted by a plant dynamics surrogate."""

    y_hat: np.ndarray
    residuals: np.ndarray
    residual_std: float
    metrics: dict[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.y_hat, np.ndarray):
            raise TypeError("y_hat must be a numpy ndarray")
        if not isinstance(self.residuals, np.ndarray):
            raise TypeError("residuals must be a numpy ndarray")
        if len(self.y_hat) != len(self.residuals):
            raise ValueError(
                f"y_hat length ({len(self.y_hat)}) must match residuals length ({len(self.residuals)})"
            )
        if not isinstance(self.residual_std, (int, float)):
            raise TypeError("residual_std must be a numeric float")
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dictionary")


class BasePlantSurrogate(ABC):
    """Abstract base class for all plant telemetry surrogate models."""

    target_channel: str = "L_T1"
    is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> BasePlantSurrogate:
        """Fit the surrogate model on clean training data.

        Args:
            train_df: Clean training split DataFrame.
            val_df: Optional clean validation split DataFrame.

        Returns:
            Fitted instance of self.
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Predict the target sensor level from companion telemetry.

        Args:
            df: Input DataFrame containing companion channels (and optionally target channel).

        Returns:
            SurrogatePrediction containing pointwise predictions, residuals, and metrics.
        """

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Serialize model weights, scaler parameters, and metadata to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path | str) -> BasePlantSurrogate:
        """Load serialized surrogate model from disk."""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Compute standard regression metrics between ground truth and predictions.

        Args:
            y_true: Ground truth target array.
            y_pred: Predicted target array.

        Returns:
            Dictionary with 'rmse', 'mae', and 'r2'.
        """
        if len(y_true) == 0:
            return {"rmse": 0.0, "mae": 0.0, "r2": 0.0}
        residuals = y_true.astype(np.float64) - y_pred.astype(np.float64)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        total_ss = float(np.sum((y_true - np.mean(y_true)) ** 2))
        res_ss = float(np.sum(residuals**2))
        r2 = float(1.0 - (res_ss / total_ss)) if total_ss > 1e-12 else 0.0
        return {"rmse": rmse, "mae": mae, "r2": r2}
