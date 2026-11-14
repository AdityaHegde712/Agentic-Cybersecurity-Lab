"""Unified PyTorch datasets and DataLoaders for plant surrogate training.

Supports BATADAL, SWaT, and WADI industrial telemetry datasets with strict
train-split-only normalization and temporal context windowing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_TARGET_CHANNELS: dict[str, str] = {
    "batadal": "L_T1",
    "swat": "LIT101",
    "wadi": "1_LT_001_PV",
}

EXCLUDE_COLUMNS: set[str] = {
    "timestamp",
    "label",
    "attack_id",
}


class PlantSurrogateDataset(Dataset):
    """Temporal sliding-window dataset for plant surrogate prediction."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        lookback: int = 48,
    ) -> None:
        """Initialize sliding-window dataset.

        Args:
            features: Normalized 2D companion feature array of shape [N, d].
            targets: Normalized 1D target array of shape [N].
            lookback: Lookback context window length (timesteps).
        """
        if len(features) != len(targets):
            raise ValueError(
                f"features length ({len(features)}) must match targets length ({len(targets)})"
            )
        if len(features) < lookback:
            raise ValueError(
                f"Data length ({len(features)}) is shorter than lookback window ({lookback})"
            )

        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.lookback = int(lookback)
        self.n_samples = len(self.features) - self.lookback + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_window = self.features[idx : idx + self.lookback]
        y_scalar = self.targets[idx + self.lookback - 1]
        return (
            torch.from_numpy(x_window),
            torch.tensor([y_scalar], dtype=torch.float32),
        )


def get_surrogate_loaders(
    dataset_name: str,
    lookback: int = 48,
    batch_size: int = 64,
    target_channel: str | None = None,
    num_workers: int = 0,
    data_root: Path | str = "data/processed",
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Construct train and validation DataLoaders for a plant dataset.

    Normalizes features and targets strictly using training split statistics.

    Args:
        dataset_name: Dataset name ('batadal', 'swat', or 'wadi').
        lookback: Temporal lookback context size (default 48).
        batch_size: Batch size for training and validation loaders.
        target_channel: Optional override for primary target channel.
        num_workers: Number of background worker processes.
        data_root: Root directory containing processed dataset folders.

    Returns:
        Tuple of (train_loader, val_loader, metadata_dict).
    """
    ds_key = dataset_name.strip().lower()
    if ds_key not in DEFAULT_TARGET_CHANNELS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: {list(DEFAULT_TARGET_CHANNELS.keys())}"
        )

    root = Path(data_root)
    train_path = root / ds_key / "train.parquet"
    val_path = root / ds_key / "val.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # For BATADAL: enforce data seal (development split rows 480-960)
    if ds_key == "batadal" and len(val_df) == 1440:
        val_df = val_df.iloc[480:960].copy().reset_index(drop=True)

    target_ch = (
        target_channel
        if target_channel is not None
        else DEFAULT_TARGET_CHANNELS[ds_key]
    )
    if target_ch not in train_df.columns:
        raise KeyError(
            f"Target channel '{target_ch}' not found in {ds_key} train columns: {list(train_df.columns)}"
        )

    companion_channels = [
        c for c in train_df.columns if c not in EXCLUDE_COLUMNS and c != target_ch
    ]

    # Compute mean and standard deviation strictly on the training set
    raw_train_feats = train_df[companion_channels].to_numpy(dtype=np.float64)
    raw_val_feats = val_df[companion_channels].to_numpy(dtype=np.float64)

    feature_mean = np.mean(raw_train_feats, axis=0)
    feature_std = np.std(raw_train_feats, axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    raw_train_target = train_df[target_ch].to_numpy(dtype=np.float64)
    raw_val_target = val_df[target_ch].to_numpy(dtype=np.float64)

    target_mean = float(np.mean(raw_train_target))
    target_std = float(np.std(raw_train_target))
    if target_std < 1e-6:
        target_std = 1.0

    # Normalize training and validation sets
    norm_train_feats = (raw_train_feats - feature_mean) / feature_std
    norm_val_feats = (raw_val_feats - feature_mean) / feature_std

    norm_train_target = (raw_train_target - target_mean) / target_std
    norm_val_target = (raw_val_target - target_mean) / target_std

    train_ds = PlantSurrogateDataset(
        norm_train_feats, norm_train_target, lookback=lookback
    )
    val_ds = PlantSurrogateDataset(norm_val_feats, norm_val_target, lookback=lookback)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    metadata: dict[str, Any] = {
        "dataset_name": ds_key,
        "target_channel": target_ch,
        "companion_channels": companion_channels,
        "in_channels": len(companion_channels),
        "lookback": lookback,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean,
        "target_std": target_std,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }

    return train_loader, val_loader, metadata
