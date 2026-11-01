"""Ridge linear companion surrogate model baseline.

Wraps closed-form ridge regression on the 42 declared-unaffected companion channels
to predict target tank level (L_T1) without autoregressive feedback corruption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from src.inverse.surrogates.base import BasePlantSurrogate, SurrogatePrediction
from src.inverse.surrogates.dataset import get_batadal_companion_channels


class RidgeCompanionSurrogate(BasePlantSurrogate):
    """Linear ridge regression surrogate on unaffected companion sensor telemetry."""

    def __init__(
        self,
        alpha: float = 1.0,
        target_channel: str = "L_T1",
        companion_channels: Sequence[str] | None = None,
        standardize: bool = True,
    ) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

        self.alpha = float(alpha)
        self.target_channel = target_channel
        if companion_channels is None:
            self.companion_channels = tuple(get_batadal_companion_channels())
        else:
            self.companion_channels = tuple(companion_channels)

        if self.target_channel in self.companion_channels:
            raise ValueError("Target channel cannot be included in companion channels")

        self.standardize = standardize
        self.weights_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.residual_std_: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> RidgeCompanionSurrogate:
        """Fit regularized linear regression on clean training data.

        Args:
            train_df: Clean training split DataFrame.
            val_df: Optional validation DataFrame (unused by linear closed-form).

        Returns:
            Fitted instance of self.
        """
        if self.target_channel not in train_df.columns:
            raise KeyError(f"Target channel {self.target_channel} not found in train_df")
        for ch in self.companion_channels:
            if ch not in train_df.columns:
                raise KeyError(f"Companion channel {ch} not found in train_df")

        X = train_df[list(self.companion_channels)].to_numpy(dtype=np.float64)
        y = train_df[self.target_channel].to_numpy(dtype=np.float64)
        n, p = X.shape

        if self.standardize:
            x_mean = np.mean(X, axis=0)
            x_std = np.std(X, axis=0)
            x_std[x_std == 0.0] = 1.0
            X_norm = (X - x_mean) / x_std

            y_mean = float(np.mean(y))
            y_norm = y - y_mean

            reg_matrix = X_norm.T @ X_norm + self.alpha * np.eye(p)
            w_norm = np.linalg.solve(reg_matrix, X_norm.T @ y_norm)

            self.weights_ = w_norm / x_std
            self.intercept_ = float(y_mean - np.dot(x_mean, self.weights_))
            self.feature_mean_ = x_mean
            self.feature_std_ = x_std
        else:
            X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float64)])
            reg_matrix = X_aug.T @ X_aug + self.alpha * np.eye(p + 1)
            reg_matrix[-1, -1] = X_aug.T @ X_aug[-1, -1]
            w_all = np.linalg.solve(reg_matrix, X_aug.T @ y)
            self.weights_ = w_all[:-1]
            self.intercept_ = float(w_all[-1])
            self.feature_mean_ = np.zeros(p, dtype=np.float64)
            self.feature_std_ = np.ones(p, dtype=np.float64)

        y_pred = X @ self.weights_ + self.intercept_
        residuals = y - y_pred
        self.residual_std_ = float(np.std(residuals))
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Predict target plant sensor levels from companion telemetry.

        Args:
            df: Input DataFrame containing companion channels.

        Returns:
            SurrogatePrediction with point predictions, residuals, and metrics.
        """
        if not self.is_fitted or self.weights_ is None:
            raise RuntimeError("RidgeCompanionSurrogate must be fitted before predict()")

        for ch in self.companion_channels:
            if ch not in df.columns:
                raise KeyError(f"Companion channel {ch} not found in input DataFrame")

        X = df[list(self.companion_channels)].to_numpy(dtype=np.float64)
        y_hat = X @ self.weights_ + self.intercept_

        if self.target_channel in df.columns:
            y_obs = df[self.target_channel].to_numpy(dtype=np.float64)
            residuals = y_obs - y_hat
            residual_std = float(np.std(residuals))
            metrics = self.compute_metrics(y_obs, y_hat)
        else:
            residuals = np.zeros(len(df), dtype=np.float64)
            residual_std = self.residual_std_
            metrics = {}

        return SurrogatePrediction(
            y_hat=y_hat,
            residuals=residuals,
            residual_std=residual_std,
            metrics=metrics,
        )

    def save(self, path: Path | str) -> None:
        """Save model parameters and metadata to disk."""
        if not self.is_fitted or self.weights_ is None:
            raise RuntimeError("Cannot save unfitted surrogate")

        save_dict = {
            "model_type": "RidgeCompanionSurrogate",
            "alpha": self.alpha,
            "target_channel": self.target_channel,
            "companion_channels": list(self.companion_channels),
            "standardize": self.standardize,
            "weights": self.weights_,
            "intercept": self.intercept_,
            "feature_mean": self.feature_mean_,
            "feature_std": self.feature_std_,
            "residual_std": self.residual_std_,
        }
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, dest)

    @classmethod
    def load(cls, path: Path | str) -> RidgeCompanionSurrogate:
        """Load saved surrogate model from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Surrogate checkpoint not found at {load_path}")

        checkpoint = torch.save if False else torch.load(load_path, map_location="cpu", weights_only=False)
        surrogate = cls(
            alpha=checkpoint["alpha"],
            target_channel=checkpoint["target_channel"],
            companion_channels=checkpoint["companion_channels"],
            standardize=checkpoint["standardize"],
        )
        surrogate.weights_ = np.asarray(checkpoint["weights"], dtype=np.float64)
        surrogate.intercept_ = float(checkpoint["intercept"])
        surrogate.feature_mean_ = np.asarray(checkpoint["feature_mean"], dtype=np.float64)
        surrogate.feature_std_ = np.asarray(checkpoint["feature_std"], dtype=np.float64)
        surrogate.residual_std_ = float(checkpoint["residual_std"])
        surrogate.is_fitted = True
        return surrogate
