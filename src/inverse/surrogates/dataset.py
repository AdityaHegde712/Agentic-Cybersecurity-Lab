"""PyTorch Dataset and DataLoader constructors for companion plant telemetry.

Implements sliding window feature representation over declared-unaffected companion sensors
for predicting the target plant sensor level (L_T1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_MANIFEST_PATH = Path("configs/manifests/batadal-channel-manifest-v1.json")
DEFAULT_TARGET_CHANNEL = "L_T1"


def get_batadal_companion_channels(manifest_path: Path | str | None = None) -> list[str]:
    """Extract ordered list of declared unaffected companion channels from manifest.

    Args:
        manifest_path: Optional path to channel manifest JSON.

    Returns:
        List of 42 companion channel names.
    """
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(f"Channel manifest not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    channels = [
        ch["name"]
        for ch in data.get("channels", [])
        if ch.get("role") == "recorded_companion"
        and ch.get("attackability") == "declared_unaffected"
    ]
    if not channels:
        raise ValueError(f"No declared unaffected companion channels found in {path}")
    return channels


class CompanionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset returning sliding window companion telemetry and target level."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_channel: str = DEFAULT_TARGET_CHANNEL,
        companion_channels: Sequence[str] | None = None,
        context_length: int = 24,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        normalize: bool = True,
        pad_initial: bool = False,
    ) -> None:
        """Initialize CompanionDataset.

        Args:
            df: DataFrame containing companion and target telemetry.
            target_channel: Name of target sensor channel (e.g. 'L_T1').
            companion_channels: Optional list of companion channels (defaults to 42 channels).
            context_length: Sliding window length in timesteps.
            feature_mean: Precomputed mean for standardization.
            feature_std: Precomputed std for standardization.
            normalize: Whether to z-score standardize features.
            pad_initial: If True, left-pad with initial value so dataset length matches len(df).
        """
        if context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {context_length}")

        self.target_channel = target_channel
        if companion_channels is None:
            self.companion_channels = tuple(get_batadal_companion_channels())
        else:
            self.companion_channels = tuple(companion_channels)

        if self.target_channel in self.companion_channels:
            raise ValueError(f"Target channel {self.target_channel} cannot be in companion channels")

        for ch in self.companion_channels:
            if ch not in df.columns:
                raise KeyError(f"Companion channel '{ch}' missing from DataFrame")

        self.context_length = context_length
        self.normalize = normalize
        self.pad_initial = pad_initial

        # Extract features
        X_raw = df[list(self.companion_channels)].to_numpy(dtype=np.float32)

        # Fit or apply standardization
        if self.normalize:
            if feature_mean is not None and feature_std is not None:
                self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
                self.feature_std = np.asarray(feature_std, dtype=np.float32)
            else:
                self.feature_mean = np.mean(X_raw, axis=0, keepdims=True).astype(np.float32)
                self.feature_std = np.std(X_raw, axis=0, keepdims=True).astype(np.float32)
                self.feature_std[self.feature_std == 0.0] = 1.0

            X_norm = (X_raw - self.feature_mean) / self.feature_std
        else:
            self.feature_mean = np.zeros((1, X_raw.shape[1]), dtype=np.float32)
            self.feature_std = np.ones((1, X_raw.shape[1]), dtype=np.float32)
            X_norm = X_raw

        # Extract target if present
        if self.target_channel in df.columns:
            y_raw = df[self.target_channel].to_numpy(dtype=np.float32)
        else:
            y_raw = np.zeros(len(df), dtype=np.float32)

        if self.pad_initial and len(df) > 0 and self.context_length > 1:
            # Replicate initial row (context_length - 1) times
            pad_count = self.context_length - 1
            pad_features = np.tile(X_norm[:1], (pad_count, 1))
            self.X = np.vstack([pad_features, X_norm])
            self.y = y_raw
            self.length = len(df)
        else:
            self.X = X_norm
            self.y = y_raw
            self.length = max(0, len(df) - self.context_length + 1)

        # Cache flow indices for physics-guided loss
        self.flow_indices = [
            i for i, ch in enumerate(self.companion_channels) if ch.startswith("F_")
        ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} out of range [0, {self.length})")

        if self.pad_initial:
            # Window spans [index, index + context_length) in padded array
            x_win = self.X[index : index + self.context_length]
            y_val = self.y[index]
        else:
            x_win = self.X[index : index + self.context_length]
            y_val = self.y[index + self.context_length - 1]

        return torch.from_numpy(x_win), torch.tensor([y_val], dtype=torch.float32)


def create_companion_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    context_length: int = 24,
    batch_size: int = 64,
    shuffle_train: bool = False,
    num_workers: int = 0,
    target_channel: str = DEFAULT_TARGET_CHANNEL,
    manifest_path: Path | str | None = None,
) -> tuple[DataLoader, ...]:
    """Construct PyTorch DataLoaders for train, val, and optional test splits.

    Standardization parameters are fit strictly on the train_df split and transferred
    to validation and test splits without leakage.

    Args:
        train_df: Training DataFrame.
        val_df: Optional validation DataFrame.
        test_df: Optional test DataFrame.
        context_length: Context lookback window length.
        batch_size: Batch size.
        shuffle_train: Whether to shuffle train DataLoader.
        num_workers: Number of DataLoader worker processes.
        target_channel: Target sensor name.
        manifest_path: Path to channel manifest.

    Returns:
        Tuple of (train_loader, val_loader) or (train_loader, val_loader, test_loader).
    """
    companion_channels = get_batadal_companion_channels(manifest_path)

    train_ds = CompanionDataset(
        df=train_df,
        target_channel=target_channel,
        companion_channels=companion_channels,
        context_length=context_length,
        normalize=True,
        pad_initial=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=False,
    )

    loaders: list[DataLoader] = [train_loader]

    if val_df is not None:
        val_ds = CompanionDataset(
            df=val_df,
            target_channel=target_channel,
            companion_channels=companion_channels,
            context_length=context_length,
            feature_mean=train_ds.feature_mean,
            feature_std=train_ds.feature_std,
            normalize=True,
            pad_initial=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        loaders.append(val_loader)

    if test_df is not None:
        test_ds = CompanionDataset(
            df=test_df,
            target_channel=target_channel,
            companion_channels=companion_channels,
            context_length=context_length,
            feature_mean=train_ds.feature_mean,
            feature_std=train_ds.feature_std,
            normalize=True,
            pad_initial=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        loaders.append(test_loader)

    return tuple(loaders)
