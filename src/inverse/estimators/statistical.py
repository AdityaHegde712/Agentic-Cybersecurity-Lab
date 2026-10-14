"""Statistical and regression inverse estimators for sensor additive attack recovery.

Implements:
- PersistenceEstimator (lag-1 naive clean history propagation).
- SeasonalNaiveEstimator (lag-S periodic clean history propagation).
- CompanionRegressionEstimator (regularized linear prediction from declared-unaffected companion sensors).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.inverse.channel_constraints import ChannelConstraintManifest
from src.inverse.estimators.base import BaseDeltaEstimator, ReconstructionResult


class PersistenceEstimator(BaseDeltaEstimator):
    """Predicts clean signal from past unattacked history via lag persistence."""

    def __init__(self, lag: int = 1) -> None:
        if lag < 1:
            raise ValueError("lag must be a positive integer")
        self.lag = lag
        self.history_: np.ndarray | None = None
        self.residual_variance_: float = 0.0

    def fit_normal(
        self,
        normal_data: pd.DataFrame | pd.Series | np.ndarray,
        target_channel: str | None = None,
        **kwargs: Any,
    ) -> PersistenceEstimator:
        """Fit baseline on normal data and compute residual variance."""
        values = self._extract_1d(normal_data, target_channel)
        self.history_ = values.copy()
        if len(values) > self.lag:
            diffs = values[self.lag :] - values[: -self.lag]
            self.residual_variance_ = float(np.var(diffs))
        else:
            self.residual_variance_ = 0.0
        return self

    def reconstruct(
        self,
        observed: np.ndarray | pd.Series | pd.DataFrame,
        attack_interval: tuple[int, int] | None = None,
        target_channel: str | None = None,
        prior_clean_history: np.ndarray | None = None,
        **kwargs: Any,
    ) -> ReconstructionResult:
        """Reconstruct clean measurement and attack delta via persistence."""
        y = self._extract_1d(observed, target_channel)
        n = len(y)

        # Context history to look back into
        if prior_clean_history is not None:
            clean_history = np.asarray(prior_clean_history, dtype=np.float64)
        elif self.history_ is not None:
            clean_history = self.history_
        else:
            clean_history = np.array([], dtype=np.float64)

        x_hat = np.empty(n, dtype=np.float64)
        delta_hat = np.zeros(n, dtype=np.float64)

        if attack_interval is not None:
            start, stop = attack_interval
            if not (0 <= start <= stop <= n):
                raise ValueError(f"Invalid attack_interval {attack_interval} for length {n}")

            # Pre-attack: unattacked
            x_hat[:start] = y[:start]
            delta_hat[:start] = 0.0

            # During attack: propagate recursively from clean values
            for t in range(start, stop):
                lookback = t - self.lag
                if lookback >= start:
                    pred = x_hat[lookback]
                elif lookback >= 0:
                    pred = y[lookback]
                else:
                    hist_idx = len(clean_history) + lookback
                    if hist_idx >= 0 and len(clean_history) > 0:
                        pred = clean_history[hist_idx]
                    elif start > 0:
                        pred = y[start - 1]
                    elif len(clean_history) > 0:
                        pred = clean_history[-1]
                    else:
                        pred = y[0]
                x_hat[t] = pred
                delta_hat[t] = y[t] - pred

            # Post-attack: unattacked
            x_hat[stop:] = y[stop:]
            delta_hat[stop:] = 0.0
        else:
            for t in range(n):
                lookback = t - self.lag
                if lookback >= 0:
                    pred = y[lookback]
                else:
                    hist_idx = len(clean_history) + lookback
                    if hist_idx >= 0 and len(clean_history) > 0:
                        pred = clean_history[hist_idx]
                    elif len(clean_history) > 0:
                        pred = clean_history[-1]
                    else:
                        pred = y[0]
                x_hat[t] = pred
                delta_hat[t] = y[t] - pred

        var = np.full(n, self.residual_variance_, dtype=np.float64)
        unc = np.sqrt(var)

        prov = {
            "estimator": self.__class__.__name__,
            "lag": self.lag,
            "residual_variance": self.residual_variance_,
        }

        return ReconstructionResult(
            x_hat=x_hat,
            delta_hat=delta_hat,
            pointwise_uncertainty=unc,
            pointwise_variance=var,
            objective_loss=0.0,
            ambiguity_flag=False,
            provenance=prov,
        )

    def _extract_1d(
        self,
        data: np.ndarray | pd.Series | pd.DataFrame,
        channel: str | None = None,
    ) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            if channel is not None and channel in data.columns:
                return data[channel].to_numpy(dtype=np.float64)
            if data.shape[1] == 1:
                return data.iloc[:, 0].to_numpy(dtype=np.float64)
            raise ValueError("DataFrame provided without target channel specification")
        if isinstance(data, pd.Series):
            return data.to_numpy(dtype=np.float64)
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D data, got shape {arr.shape}")
        return arr


class SeasonalNaiveEstimator(PersistenceEstimator):
    """Predicts clean signal from past unattacked history via seasonal periodicity (default 24h)."""

    def __init__(self, period: int = 24) -> None:
        super().__init__(lag=period)
        self.period = period


class CompanionRegressionEstimator(BaseDeltaEstimator):
    """Fits regularized linear model predicting target sensor from unaffected companions."""

    def __init__(
        self,
        target_channel: str = "L_T1",
        companion_channels: tuple[str, ...] | None = None,
        channel_manifest_path: str | Path | None = None,
        alpha: float = 1.0,
        standardize: bool = True,
    ) -> None:
        self.target_channel = target_channel
        self.alpha = float(alpha)
        self.standardize = standardize
        self.weights_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.fit_r2: float = 0.0
        self.residual_variance: float = 0.0
        self.is_fitted: bool = False

        if companion_channels is not None:
            if target_channel in companion_channels:
                raise ValueError("Target channel cannot be a companion sensor")
            self.companion_channels = tuple(companion_channels)
        elif channel_manifest_path is not None:
            manifest_file = Path(channel_manifest_path)
            if not manifest_file.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_file}")
            manifest = ChannelConstraintManifest.from_json(manifest_file.read_text(encoding="utf-8"))
            unaffected = [
                ch.name
                for ch in manifest.channels
                if ch.role == "recorded_companion" and ch.attackability == "declared_unaffected"
            ]
            if target_channel in unaffected:
                unaffected.remove(target_channel)
            self.companion_channels = tuple(unaffected)
        else:
            default_path = Path("configs/manifests/batadal-channel-manifest-v1.json")
            if default_path.exists():
                manifest = ChannelConstraintManifest.from_json(default_path.read_text(encoding="utf-8"))
                unaffected = [
                    ch.name
                    for ch in manifest.channels
                    if ch.role == "recorded_companion" and ch.attackability == "declared_unaffected"
                ]
                if target_channel in unaffected:
                    unaffected.remove(target_channel)
                self.companion_channels = tuple(unaffected)
            else:
                self.companion_channels = ()

    def fit_normal(
        self,
        normal_data: pd.DataFrame,
        **kwargs: Any,
    ) -> CompanionRegressionEstimator:
        """Fit regularized linear regression on the clean model-fit dataset."""
        if not isinstance(normal_data, pd.DataFrame):
            raise ValueError("normal_data must be a pandas DataFrame")
        if self.target_channel not in normal_data.columns:
            raise ValueError(f"Target channel {self.target_channel} not in data")
        for ch in self.companion_channels:
            if ch not in normal_data.columns:
                raise ValueError(f"Companion channel {ch} not in data")

        y = normal_data[self.target_channel].to_numpy(dtype=np.float64)
        X = normal_data[list(self.companion_channels)].to_numpy(dtype=np.float64)
        n, p = X.shape

        if self.standardize:
            x_mean = np.mean(X, axis=0)
            x_std = np.std(X, axis=0)
            x_std[x_std == 0.0] = 1.0
            X_norm = (X - x_mean) / x_std

            y_mean = float(np.mean(y))
            y_norm = y - y_mean

            # Ridge regression normal equations: (X^T X + alpha * I)^(-1) X^T y
            reg_matrix = X_norm.T @ X_norm + self.alpha * np.eye(p)
            w_norm = np.linalg.solve(reg_matrix, X_norm.T @ y_norm)

            self.weights_ = w_norm / x_std
            self.intercept_ = float(y_mean - np.dot(x_mean, self.weights_))
        else:
            X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float64)])
            reg_matrix = X_aug.T @ X_aug + self.alpha * np.eye(p + 1)
            reg_matrix[-1, -1] = X_aug.T @ X_aug[-1, -1]  # do not penalize intercept
            w_all = np.linalg.solve(reg_matrix, X_aug.T @ y)
            self.weights_ = w_all[:-1]
            self.intercept_ = float(w_all[-1])

        y_pred = X @ self.weights_ + self.intercept_
        residuals = y - y_pred
        self.residual_variance = float(np.var(residuals))
        total_var = float(np.var(y))
        self.fit_r2 = 1.0 - (self.residual_variance / total_var) if total_var > 0 else 0.0
        self.is_fitted = True
        return self

    def reconstruct(
        self,
        observed: pd.DataFrame | np.ndarray,
        attack_interval: tuple[int, int] | None = None,
        companion_data: pd.DataFrame | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ReconstructionResult:
        """Reconstruct clean measurement and delta using companion regression."""
        if not self.is_fitted or self.weights_ is None:
            raise RuntimeError("Estimator must be fitted before reconstruction")

        if isinstance(observed, pd.DataFrame):
            y = observed[self.target_channel].to_numpy(dtype=np.float64)
            X = observed[list(self.companion_channels)].to_numpy(dtype=np.float64)
        else:
            y = np.asarray(observed, dtype=np.float64)
            if companion_data is None:
                raise ValueError("companion_data must be provided when observed is a numpy array")
            if isinstance(companion_data, pd.DataFrame):
                X = companion_data[list(self.companion_channels)].to_numpy(dtype=np.float64)
            else:
                X = np.asarray(companion_data, dtype=np.float64)

        n = len(y)
        surrogate_x = X @ self.weights_ + self.intercept_
        x_hat = np.empty(n, dtype=np.float64)
        delta_hat = np.zeros(n, dtype=np.float64)

        if attack_interval is not None:
            start, stop = attack_interval
            if not (0 <= start <= stop <= n):
                raise ValueError(f"Invalid attack_interval {attack_interval} for length {n}")

            x_hat[:start] = y[:start]
            delta_hat[:start] = 0.0

            x_hat[start:stop] = surrogate_x[start:stop]
            delta_hat[start:stop] = y[start:stop] - surrogate_x[start:stop]

            x_hat[stop:] = y[stop:]
            delta_hat[stop:] = 0.0
        else:
            x_hat[:] = surrogate_x
            delta_hat[:] = y - surrogate_x

        var = np.full(n, self.residual_variance, dtype=np.float64)
        unc = np.sqrt(var)

        prov = {
            "estimator": self.__class__.__name__,
            "target_channel": self.target_channel,
            "alpha": self.alpha,
            "fit_r2": self.fit_r2,
            "residual_variance": self.residual_variance,
            "companion_count": len(self.companion_channels),
        }

        return ReconstructionResult(
            x_hat=x_hat,
            delta_hat=delta_hat,
            pointwise_uncertainty=unc,
            pointwise_variance=var,
            objective_loss=float(np.mean((y - x_hat - delta_hat) ** 2)),
            ambiguity_flag=False,
            provenance=prov,
        )
