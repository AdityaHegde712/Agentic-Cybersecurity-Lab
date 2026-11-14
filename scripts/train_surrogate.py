"""Unified surrogate training runner for industrial SCADA telemetry.

Supports BATADAL, SWaT, and WADI datasets across CausalTCN and CausalPatchTST
architectures. Features terminal-friendly progress logging, best checkpoint
persistence, history recording, and loss curve plotting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm import tqdm
except ImportError:

    class _DummyTqdm:
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            pass

    def tqdm(iterable, *args, **kwargs):
        return _DummyTqdm(iterable, *args, **kwargs)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.surrogate_loaders import get_surrogate_loaders  # noqa: E402
from src.models.base import CausalTCN  # noqa: E402
from src.models.patch_tst import CausalPatchTST  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Plant Surrogate Training Runner"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["batadal", "swat", "wadi"],
        help="Target industrial telemetry dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tcn",
        choices=["tcn", "patch_tst"],
        help="Surrogate model architecture",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=48,
        help="Lookback context window length (timesteps)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/surrogates",
        help="Directory to save checkpoints and artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension / d_model for the surrogate",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast 1-epoch reduced-batch smoke test for pipeline pre-flight validation",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    """Resolve target execution device with fallback."""
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def plot_loss_curve(history: list[dict[str, Any]], output_path: Path) -> None:
    """Plot and save training and validation loss curves."""
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(epochs, train_losses, label="Train Loss (MSE)", color="#1f77b4", lw=2)
    ax.plot(
        epochs,
        val_losses,
        label="Val Loss (MSE)",
        color="#ff7f0e",
        lw=2,
        linestyle="--",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Normalized Scale)")
    ax.set_title("Surrogate Model Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    """Execute surrogate model training loop."""
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.smoke:
        args.epochs = 1
        args.batch_size = 16
        print("[SMOKE MODE] Configured 1 epoch with batch size 16.")

    device = resolve_device(args.device)
    out_dir = Path(args.checkpoint_dir) / f"{args.dataset}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PLANT SURROGATE TRAINING RUNNER")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    print(
        f"Dataset: {args.dataset.upper()} | Model: {args.model.upper()} | Device: {device}"
    )
    print(f"Output Directory: {out_dir}")
    print("=" * 80)

    # 1. Load data
    train_loader, val_loader, metadata = get_surrogate_loaders(
        dataset_name=args.dataset,
        lookback=args.lookback,
        batch_size=args.batch_size,
    )
    in_channels = metadata["in_channels"]
    print(
        f"Data Loaded: {in_channels} companion features, target={metadata['target_channel']} | "
        f"Train: {metadata['train_samples']} samples, Val: {metadata['val_samples']} samples"
    )

    # 2. Build model
    if args.model == "tcn":
        model: nn.Module = CausalTCN(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            dilations=(1, 2, 4, 8),
            kernel_size=3,
            dropout=0.1,
        )
    elif args.model == "patch_tst":
        model = CausalPatchTST(
            in_channels=in_channels,
            d_model=args.hidden_dim,
            patch_len=16,
            stride=8,
            nhead=4,
            num_layers=3,
            dropout=0.1,
            max_patches=64,
        )
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model Architecture: {model.__class__.__name__} ({n_params:,} trainable parameters)"
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-6
    )

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_checkpoint_path = out_dir / "checkpoint_best.pt"

    t_total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        # Progress bar: mininterval=1.0 prevents Slurm log flooding
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{args.epochs:02d}",
            mininterval=1.0,
            leave=False,
        )
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = float(loss.item())
            train_loss_sum += loss_val
            train_batches += 1
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        scheduler.step()
        train_loss = train_loss_sum / max(1, train_batches)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_residuals: list[float] = []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                preds_val = model(x_val)
                v_loss = criterion(preds_val, y_val)
                val_loss_sum += float(v_loss.item())
                val_batches += 1
                res = (preds_val - y_val).cpu().numpy().flatten()
                val_residuals.extend(res.tolist())

        val_loss = val_loss_sum / max(1, val_batches)
        val_res_std = float(np.std(val_residuals)) * float(metadata["target_std"])
        epoch_elapsed = time.perf_counter() - epoch_start

        # Record history
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_residual_std_m": round(val_res_std, 6),
            "lr": round(float(scheduler.get_last_lr()[0]), 8),
            "time_seconds": round(epoch_elapsed, 2),
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"Val Res Std: {val_res_std:.4f} m | Time: {epoch_elapsed:.1f}s"
        )

        # Checkpoint persistence for best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_type": model.__class__.__name__,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dataset_name": args.dataset,
                    "model_name": args.model,
                    "target_channel": metadata["target_channel"],
                    "companion_channels": metadata["companion_channels"],
                    "lookback": args.lookback,
                    "feature_mean": metadata["feature_mean"],
                    "feature_std": metadata["feature_std"],
                    "target_mean": metadata["target_mean"],
                    "target_std": metadata["target_std"],
                    "best_epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "val_residual_std": val_res_std,
                },
                best_checkpoint_path,
            )

    total_time = time.perf_counter() - t_total_start
    print("=" * 80)
    print(
        f"Training Complete in {total_time:.1f}s | Best Val Loss: {best_val_loss:.5f}"
    )

    # Save training history JSON
    history_path = out_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Plot loss curve
    loss_curve_path = out_dir / "loss_curve.png"
    plot_loss_curve(history, loss_curve_path)
    print(f"Saved loss curve plot to {loss_curve_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
