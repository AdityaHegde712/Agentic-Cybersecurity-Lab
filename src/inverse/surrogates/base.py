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

from src.inverse.estimators.base import ReconstructionResult


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

    def reconstruct(
        self,
        observed: np.ndarray | pd.Series | pd.DataFrame,
        attack_interval: tuple[int, int] | None = None,
        target_channel: str | None = None,
        **kwargs: Any,
    ) -> ReconstructionResult:
        """Reconstruct clean measurement signal and additive attack trajectory.

        Polymorphic interface enabling plant dynamics surrogates to act directly as
        inverse attack estimators and surrogate estimators for MAP solvers.

        Args:
            observed: Observed telemetry (DataFrame with companion channels, or Series/Array).
            attack_interval: Optional tuple (onset, offset) declaring known attack bounds.
            target_channel: Target sensor name (defaults to self.target_channel).
            **kwargs: Extra parameters (e.g. 'companion_data' or 'df' if observed is an array).

        Returns:
            ReconstructionResult containing clean x_hat, delta_hat, and uncertainty bounds.
        """
        ch = target_channel or self.target_channel

        if isinstance(observed, pd.DataFrame):
            pred = self.predict(observed)
            y_hat = pred.y_hat
            if ch in observed.columns:
                y = observed[ch].to_numpy(dtype=np.float64)
            else:
                y = y_hat.copy()
        elif isinstance(observed, pd.Series):
            y = observed.to_numpy(dtype=np.float64)
            df = kwargs.get("df") or kwargs.get("companion_data")
            if isinstance(df, pd.DataFrame):
                pred = self.predict(df)
                y_hat = pred.y_hat
            else:
                raise ValueError(
                    f"{self.__class__.__name__}.reconstruct requires a DataFrame with companion channels, "
                    "or 'df' passed in kwargs."
                )
        else:
            y = np.asarray(observed, dtype=np.float64)
            df = kwargs.get("df") or kwargs.get("companion_data")
            if isinstance(df, pd.DataFrame):
                pred = self.predict(df)
                y_hat = pred.y_hat
            else:
                raise ValueError(
                    f"{self.__class__.__name__}.reconstruct requires a DataFrame with companion channels, "
                    "or 'df' passed in kwargs."
                )

        n = len(y)
        if len(y_hat) != n:
            raise ValueError(
                f"Predicted y_hat length ({len(y_hat)}) does not match observed length ({n})"
            )

        # Compute delta_hat = y - x_hat (where surrogate predicted clean x_hat = y_hat)
        delta_hat = y - y_hat

        # If attack_interval is provided, clamp delta_hat = 0 outside [onset, offset]
        if attack_interval is not None:
            onset, offset = attack_interval
            if not (0 <= onset <= offset <= n):
                raise ValueError(f"Invalid attack_interval {attack_interval} for length {n}")
            delta_hat = delta_hat.copy()
            delta_hat[:onset] = 0.0
            delta_hat[offset:] = 0.0

        # Set x_hat = y - delta_hat
        x_hat = y - delta_hat

        unc = np.full(n, float(pred.residual_std), dtype=np.float64)
        var = unc ** 2

        return ReconstructionResult(
            x_hat=x_hat,
            delta_hat=delta_hat,
            pointwise_uncertainty=unc,
            pointwise_variance=var,
            objective_loss=float(pred.residual_std),
            ambiguity_flag=False,
            provenance={
                "estimator": self.__class__.__name__,
                "residual_std": float(pred.residual_std),
                "metrics": pred.metrics,
                "attack_interval": attack_interval,
            },
        )
