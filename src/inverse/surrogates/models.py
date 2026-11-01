"""PyTorch dynamics surrogate models for plant telemetry reconstruction.

Implements:
- PhysicsGuidedMLPSurrogate: Multi-layer perceptron with hydraulic mass balance regularization.
- DilatedTCNSurrogate: 1D Temporal Convolutional Network with causal dilated convolutions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.inverse.surrogates.base import BasePlantSurrogate, SurrogatePrediction
from src.inverse.surrogates.dataset import CompanionDataset, get_batadal_companion_channels


# =====================================================================
# 1. Physics-Guided Multi-Layer Perceptron (PINN / Mass-Regularized MLP)
# =====================================================================

class PhysicsGuidedMLP(nn.Module):
    """3-layer MLP predicting target plant level from companion telemetry."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        curr_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input has sliding window context (batch, context_len, n_features), flatten to (batch, -1)
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.network(x)


class PhysicsGuidedMLPSurrogate(BasePlantSurrogate):
    """Surrogate model regularized by physical hydraulic mass conservation."""

    def __init__(
        self,
        context_length: int = 24,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.1,
        gamma_mass: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 64,
        device: str = "cpu",
        target_channel: str = "L_T1",
        companion_channels: Sequence[str] | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.context_length = context_length
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.gamma_mass = float(gamma_mass)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device_str = device
        self.target_channel = target_channel
        self.seed = seed
        self.verbose = verbose

        if companion_channels is None:
            self.companion_channels = tuple(get_batadal_companion_channels())
        else:
            self.companion_channels = tuple(companion_channels)

        if self.target_channel in self.companion_channels:
            raise ValueError("Target channel cannot be included in companion channels")

        self.model: PhysicsGuidedMLP | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.flow_indices_: list[int] = []
        self.flow_weights_: np.ndarray | None = None
        self.flow_intercept_: float = 0.0
        self.residual_std_: float = 0.0
        self.is_fitted: bool = False

    def _resolve_device(self) -> torch.device:
        if self.device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> PhysicsGuidedMLPSurrogate:
        """Fit MLP surrogate using combined MSE and hydraulic mass balance loss."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = self._resolve_device()

        # 1. Fit flow proxy: estimate delta_L_T1 from companion flow channels
        self.flow_indices_ = [
            i for i, ch in enumerate(self.companion_channels) if ch.startswith("F_")
        ]
        flow_cols = [self.companion_channels[i] for i in self.flow_indices_]

        if flow_cols and len(train_df) > 1 and self.target_channel in train_df.columns:
            dL = train_df[self.target_channel].diff().iloc[1:].to_numpy(dtype=np.float64)
            F_flows = train_df[flow_cols].iloc[1:].to_numpy(dtype=np.float64)
            # Least squares fit: F_flows @ w + b approx dL
            F_aug = np.hstack([F_flows, np.ones((len(dL), 1), dtype=np.float64)])
            reg_sol, _, _, _ = np.linalg.lstsq(F_aug, dL, rcond=None)
            self.flow_weights_ = reg_sol[:-1]
            self.flow_intercept_ = float(reg_sol[-1])
        else:
            self.flow_weights_ = np.zeros(len(self.flow_indices_), dtype=np.float64)
            self.flow_intercept_ = 0.0

        flow_w_t = torch.tensor(self.flow_weights_, dtype=torch.float32, device=dev).unsqueeze(1)
        flow_b_t = torch.tensor(self.flow_intercept_, dtype=torch.float32, device=dev)

        # 2. Build datasets
        train_ds = CompanionDataset(
            df=train_df,
            target_channel=self.target_channel,
            companion_channels=self.companion_channels,
            context_length=self.context_length,
            normalize=True,
            pad_initial=False,
        )
        self.feature_mean_ = train_ds.feature_mean
        self.feature_std_ = train_ds.feature_std

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=False,  # preserve sequential order for rate-of-change physics loss
            drop_last=False,
        )

        val_loader: DataLoader | None = None
        if val_df is not None:
            val_ds = CompanionDataset(
                df=val_df,
                target_channel=self.target_channel,
                companion_channels=self.companion_channels,
                context_length=self.context_length,
                feature_mean=self.feature_mean_,
                feature_std=self.feature_std_,
                normalize=True,
                pad_initial=False,
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # 3. Instantiate model
        in_features = len(self.companion_channels) * self.context_length
        self.model = PhysicsGuidedMLP(
            in_features=in_features,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(dev)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # 4. Training loop
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            n_batches = 0

            for bx, by in train_loader:
                bx = bx.to(dev)
                by = by.to(dev)

                optimizer.zero_grad()
                y_hat = self.model(bx)

                # Loss component 1: Pointwise MSE
                loss_mse = F.mse_loss(y_hat, by)

                # Loss component 2: Physics mass rate-of-change loss
                # L_mass = 1/(T-1) sum |(y_hat_{t+1} - y_hat_t) - delta_x_flow|^2
                if bx.size(0) > 1 and self.gamma_mass > 0.0 and len(self.flow_indices_) > 0:
                    diff_y_hat = y_hat[1:] - y_hat[:-1]  # shape (T-1, 1)

                    # Extract unnormalized flows at the current step (last row of sliding window)
                    # Note: bx is standardized. Reconstruct raw flow values:
                    # x_raw = bx[:, -1, flow_idx] * std + mean
                    flow_std = torch.tensor(
                        self.feature_std_[0, self.flow_indices_],
                        dtype=torch.float32,
                        device=dev,
                    )
                    flow_mean = torch.tensor(
                        self.feature_mean_[0, self.flow_indices_],
                        dtype=torch.float32,
                        device=dev,
                    )
                    raw_flows = bx[1:, -1, self.flow_indices_] * flow_std + flow_mean
                    delta_flow = raw_flows @ flow_w_t + flow_b_t

                    loss_mass = torch.mean((diff_y_hat - delta_flow) ** 2)
                    loss = loss_mse + self.gamma_mass * loss_mass
                else:
                    loss = loss_mse

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            if self.verbose and epoch % max(1, self.epochs // 5) == 0:
                print(f"[PINN Epoch {epoch:02d}/{self.epochs}] Loss: {total_loss / max(1, n_batches):.5f}")

        # Compute training residual std
        self.is_fitted = True
        train_pred = self.predict(train_df)
        self.residual_std_ = train_pred.residual_std
        return self

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Predict target plant sensor levels from companion telemetry."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("PhysicsGuidedMLPSurrogate must be fitted before predict()")

        dev = self._resolve_device()
        self.model.eval()

        # Evaluate over entire df with pad_initial=True so output matches len(df)
        ds = CompanionDataset(
            df=df,
            target_channel=self.target_channel,
            companion_channels=self.companion_channels,
            context_length=self.context_length,
            feature_mean=self.feature_mean_,
            feature_std=self.feature_std_,
            normalize=True,
            pad_initial=True,
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        preds: list[np.ndarray] = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(dev)
                y_hat = self.model(bx)
                preds.append(y_hat.cpu().numpy().squeeze(-1))

        if preds:
            y_hat_all = np.concatenate(preds, axis=0).astype(np.float64)
        else:
            y_hat_all = np.zeros(0, dtype=np.float64)

        if self.target_channel in df.columns:
            y_obs = df[self.target_channel].to_numpy(dtype=np.float64)
            residuals = y_obs - y_hat_all
            residual_std = float(np.std(residuals))
            metrics = self.compute_metrics(y_obs, y_hat_all)
        else:
            residuals = np.zeros(len(df), dtype=np.float64)
            residual_std = self.residual_std_
            metrics = {}

        return SurrogatePrediction(
            y_hat=y_hat_all,
            residuals=residuals,
            residual_std=residual_std,
            metrics=metrics,
        )

    def save(self, path: Path | str) -> None:
        """Save model checkpoint, parameters, and metadata to disk."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted surrogate")

        save_dict = {
            "model_type": "PhysicsGuidedMLPSurrogate",
            "state_dict": self.model.state_dict(),
            "context_length": self.context_length,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "gamma_mass": self.gamma_mass,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "target_channel": self.target_channel,
            "companion_channels": list(self.companion_channels),
            "feature_mean": self.feature_mean_,
            "feature_std": self.feature_std_,
            "flow_indices": self.flow_indices_,
            "flow_weights": self.flow_weights_,
            "flow_intercept": self.flow_intercept_,
            "residual_std": self.residual_std_,
            "seed": self.seed,
        }
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, dest)

    @classmethod
    def load(cls, path: Path | str) -> PhysicsGuidedMLPSurrogate:
        """Load saved surrogate model from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Surrogate checkpoint not found at {load_path}")

        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        surrogate = cls(
            context_length=checkpoint["context_length"],
            hidden_dims=tuple(checkpoint["hidden_dims"]),
            dropout=checkpoint["dropout"],
            gamma_mass=checkpoint["gamma_mass"],
            lr=checkpoint["lr"],
            weight_decay=checkpoint["weight_decay"],
            epochs=checkpoint["epochs"],
            batch_size=checkpoint["batch_size"],
            target_channel=checkpoint["target_channel"],
            companion_channels=checkpoint["companion_channels"],
            seed=checkpoint.get("seed", 42),
        )
        surrogate.feature_mean_ = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        surrogate.feature_std_ = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        surrogate.flow_indices_ = checkpoint["flow_indices"]
        surrogate.flow_weights_ = np.asarray(checkpoint["flow_weights"], dtype=np.float64)
        surrogate.flow_intercept_ = float(checkpoint["flow_intercept"])
        surrogate.residual_std_ = float(checkpoint["residual_std"])

        in_features = len(surrogate.companion_channels) * surrogate.context_length
        surrogate.model = PhysicsGuidedMLP(
            in_features=in_features,
            hidden_dims=surrogate.hidden_dims,
            dropout=surrogate.dropout,
        )
        surrogate.model.load_state_dict(checkpoint["state_dict"])
        surrogate.is_fitted = True
        return surrogate


# =====================================================================
# 2. Dilated Temporal Convolutional Network (TCN Surrogate)
# =====================================================================

class CausalConv1dBlock(nn.Module):
    """Residual 1D causal dilated convolution block with GroupNorm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.drop1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.drop2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        # Conv 1
        x1 = F.pad(x, (self.pad, 0))
        out = self.drop1(F.silu(self.norm1(self.conv1(x1))))
        # Conv 2
        x2 = F.pad(out, (self.pad, 0))
        out = self.drop2(self.norm2(self.conv2(x2)))
        return F.silu(out + res)


class DilatedTCN(nn.Module):
    """Feed-forward Temporal Convolutional Network with causal dilations [1, 2, 4, 8]."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        curr_in = in_channels

        for d in dilations:
            layers.append(
                CausalConv1dBlock(
                    in_channels=curr_in,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
            curr_in = hidden_channels

        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, context_length, n_features) -> transpose to (batch, n_features, context_length)
        if x.dim() == 3:
            x = x.transpose(1, 2)
        h = self.network(x)
        # Extract representation at current timestep (final position in lookback window)
        h_last = h[:, :, -1]
        return self.head(h_last)


class DilatedTCNSurrogate(BasePlantSurrogate):
    """Causal Dilated Temporal Convolutional Network surrogate model."""

    def __init__(
        self,
        context_length: int = 24,
        hidden_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 64,
        device: str = "cpu",
        target_channel: str = "L_T1",
        companion_channels: Sequence[str] | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.context_length = context_length
        self.hidden_channels = hidden_channels
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device_str = device
        self.target_channel = target_channel
        self.seed = seed
        self.verbose = verbose

        if companion_channels is None:
            self.companion_channels = tuple(get_batadal_companion_channels())
        else:
            self.companion_channels = tuple(companion_channels)

        if self.target_channel in self.companion_channels:
            raise ValueError("Target channel cannot be included in companion channels")

        self.model: DilatedTCN | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.residual_std_: float = 0.0
        self.is_fitted: bool = False

    def _resolve_device(self) -> torch.device:
        if self.device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> DilatedTCNSurrogate:
        """Fit Dilated TCN surrogate using MSE loss over temporal companion context."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = self._resolve_device()

        # Build datasets
        train_ds = CompanionDataset(
            df=train_df,
            target_channel=self.target_channel,
            companion_channels=self.companion_channels,
            context_length=self.context_length,
            normalize=True,
            pad_initial=False,
        )
        self.feature_mean_ = train_ds.feature_mean
        self.feature_std_ = train_ds.feature_std

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        in_channels = len(self.companion_channels)
        self.model = DilatedTCN(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            dilations=self.dilations,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        ).to(dev)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Training loop
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            n_batches = 0

            for bx, by in train_loader:
                bx = bx.to(dev)
                by = by.to(dev)

                optimizer.zero_grad()
                y_hat = self.model(bx)
                loss = F.mse_loss(y_hat, by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            if self.verbose and epoch % max(1, self.epochs // 5) == 0:
                print(f"[TCN Epoch {epoch:02d}/{self.epochs}] Loss: {total_loss / max(1, n_batches):.5f}")

        self.is_fitted = True
        train_pred = self.predict(train_df)
        self.residual_std_ = train_pred.residual_std
        return self

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Predict target plant sensor levels using causal companion sequence processing."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("DilatedTCNSurrogate must be fitted before predict()")

        dev = self._resolve_device()
        self.model.eval()

        ds = CompanionDataset(
            df=df,
            target_channel=self.target_channel,
            companion_channels=self.companion_channels,
            context_length=self.context_length,
            feature_mean=self.feature_mean_,
            feature_std=self.feature_std_,
            normalize=True,
            pad_initial=True,
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        preds: list[np.ndarray] = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(dev)
                y_hat = self.model(bx)
                preds.append(y_hat.cpu().numpy().squeeze(-1))

        if preds:
            y_hat_all = np.concatenate(preds, axis=0).astype(np.float64)
        else:
            y_hat_all = np.zeros(0, dtype=np.float64)

        if self.target_channel in df.columns:
            y_obs = df[self.target_channel].to_numpy(dtype=np.float64)
            residuals = y_obs - y_hat_all
            residual_std = float(np.std(residuals))
            metrics = self.compute_metrics(y_obs, y_hat_all)
        else:
            residuals = np.zeros(len(df), dtype=np.float64)
            residual_std = self.residual_std_
            metrics = {}

        return SurrogatePrediction(
            y_hat=y_hat_all,
            residuals=residuals,
            residual_std=residual_std,
            metrics=metrics,
        )

    def save(self, path: Path | str) -> None:
        """Save model checkpoint, parameters, and metadata to disk."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted surrogate")

        save_dict = {
            "model_type": "DilatedTCNSurrogate",
            "state_dict": self.model.state_dict(),
            "context_length": self.context_length,
            "hidden_channels": self.hidden_channels,
            "dilations": self.dilations,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "target_channel": self.target_channel,
            "companion_channels": list(self.companion_channels),
            "feature_mean": self.feature_mean_,
            "feature_std": self.feature_std_,
            "residual_std": self.residual_std_,
            "seed": self.seed,
        }
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, dest)

    @classmethod
    def load(cls, path: Path | str) -> DilatedTCNSurrogate:
        """Load saved surrogate model from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Surrogate checkpoint not found at {load_path}")

        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        surrogate = cls(
            context_length=checkpoint["context_length"],
            hidden_channels=checkpoint["hidden_channels"],
            dilations=tuple(checkpoint["dilations"]),
            kernel_size=checkpoint["kernel_size"],
            dropout=checkpoint["dropout"],
            lr=checkpoint["lr"],
            weight_decay=checkpoint["weight_decay"],
            epochs=checkpoint["epochs"],
            batch_size=checkpoint["batch_size"],
            target_channel=checkpoint["target_channel"],
            companion_channels=checkpoint["companion_channels"],
            seed=checkpoint.get("seed", 42),
        )
        surrogate.feature_mean_ = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        surrogate.feature_std_ = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        surrogate.residual_std_ = float(checkpoint["residual_std"])

        in_channels = len(surrogate.companion_channels)
        surrogate.model = DilatedTCN(
            in_channels=in_channels,
            hidden_channels=surrogate.hidden_channels,
            dilations=surrogate.dilations,
            kernel_size=surrogate.kernel_size,
            dropout=surrogate.dropout,
        )
        surrogate.model.load_state_dict(checkpoint["state_dict"])
        surrogate.is_fitted = True
        return surrogate
