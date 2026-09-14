"""Normal-only LSTM next-step forecasts and observation residuals."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ForecasterConfig:
    context_length: int = 32
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 256
    max_epochs: int = 20
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 7

    def __post_init__(self) -> None:
        if min(self.context_length, self.hidden_size, self.num_layers, self.batch_size, self.max_epochs, self.patience) < 1:
            raise ValueError("model dimensions and training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid optimization or dropout parameter")


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureNormalizer":
        _validate(values, "values")
        scale = values.std(axis=0)
        return cls(values.mean(axis=0).astype(np.float32), np.where(scale > np.finfo(np.float32).eps, scale, 1.0).astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        _validate(values, "values")
        if values.shape[1] != len(self.mean):
            raise ValueError("sensor count differs from checkpoint")
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)


class NextStepWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, values: np.ndarray, context_length: int) -> None:
        _validate(values, "values")
        if context_length < 1 or len(values) <= context_length:
            raise ValueError("values must be longer than context_length")
        self.values, self.context_length = torch.from_numpy(values.astype(np.float32, copy=False)), context_length

    def __len__(self) -> int:
        return len(self.values) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError("window index out of range")
        end = index + self.context_length
        return self.values[index:end], self.values[end]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.recurrent(context)
        return self.output(hidden[-1])


@dataclass(frozen=True)
class TrainingResult:
    model: LSTMForecaster
    normalizer: FeatureNormalizer
    config: ForecasterConfig
    history: tuple[dict[str, float | int], ...]
    best_validation_loss: float


@dataclass(frozen=True)
class LoadedForecaster:
    model: LSTMForecaster
    normalizer: FeatureNormalizer
    config: ForecasterConfig


@dataclass(frozen=True)
class ResidualPrediction:
    target_indices: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray


def train_forecaster(train_values: np.ndarray, validation_values: np.ndarray, *, config: ForecasterConfig, checkpoint_path: Path, device: str) -> TrainingResult:
    _validate(train_values, "train_values"); _validate(validation_values, "validation_values")
    if train_values.shape[1] != validation_values.shape[1]: raise ValueError("sensor counts must match")
    _seed(config.seed); normalizer = FeatureNormalizer.fit(train_values); target = torch.device(device)
    model = LSTMForecaster(train_values.shape[1], config.hidden_size, config.num_layers, config.dropout).to(target)
    train_loader = _loader(normalizer.transform(train_values), config, True); validation_loader = _loader(normalizer.transform(validation_values), config, False)
    optimizer, criterion = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay), nn.MSELoss()
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _epoch(model, train_loader, criterion, target, optimizer); validation_loss = _epoch(model, validation_loader, criterion, target)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best:
            best, stale = validation_loss, 0; _save(checkpoint_path, model, normalizer, config, epoch, validation_loss)
        else:
            stale += 1
            if stale >= config.patience: break
    loaded = load_forecaster_checkpoint(checkpoint_path, device=device)
    return TrainingResult(loaded.model, loaded.normalizer, loaded.config, tuple(history), best)


def load_forecaster_checkpoint(checkpoint_path: Path, *, device: str) -> LoadedForecaster:
    if not checkpoint_path.exists(): raise FileNotFoundError(f"forecaster checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True); config = ForecasterConfig(**payload["config"])
    model = LSTMForecaster(**payload["model_kwargs"]); model.load_state_dict(payload["model_state_dict"]); model.to(torch.device(device)).eval()
    return LoadedForecaster(model, FeatureNormalizer(payload["normalizer_mean"].cpu().numpy(), payload["normalizer_scale"].cpu().numpy()), config)


def predict_residuals(forecaster: LoadedForecaster, values: np.ndarray, *, device: str) -> ResidualPrediction:
    loader = _loader(forecaster.normalizer.transform(values), forecaster.config, False); target = torch.device(device); predicted, observed = [], []
    forecaster.model.eval()
    with torch.no_grad():
        for context, labels in loader:
            predicted.append(forecaster.model(context.to(target)).cpu().numpy()); observed.append(labels.numpy())
    predicted_values = forecaster.normalizer.inverse(np.concatenate(predicted)); observed_values = forecaster.normalizer.inverse(np.concatenate(observed))
    return ResidualPrediction(np.arange(forecaster.config.context_length, len(values)), observed_values, predicted_values, estimate_additive_delta(observed_values, predicted_values))


def estimate_additive_delta(observed: np.ndarray, predicted_clean: np.ndarray) -> np.ndarray:
    _validate(observed, "observed"); _validate(predicted_clean, "predicted_clean")
    if observed.shape != predicted_clean.shape: raise ValueError("observed and predicted_clean must have the same shape")
    return observed - predicted_clean


def _loader(values: np.ndarray, config: ForecasterConfig, shuffle: bool) -> DataLoader:
    return DataLoader(NextStepWindowDataset(values, config.context_length), batch_size=config.batch_size, shuffle=shuffle)


def _epoch(model: LSTMForecaster, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> float:
    model.train(optimizer is not None); total = count = 0
    for context, target in loader:
        context, target = context.to(device), target.to(device)
        if optimizer: optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(context), target)
        if optimizer: loss.backward(); optimizer.step()
        total += float(loss.detach().cpu()) * len(target); count += len(target)
    return total / count


def _save(path: Path, model: LSTMForecaster, normalizer: FeatureNormalizer, config: ForecasterConfig, epoch: int, validation_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_kwargs": {"input_size": model.output.out_features, "hidden_size": model.recurrent.hidden_size, "num_layers": model.recurrent.num_layers, "dropout": config.dropout}, "normalizer_mean": torch.from_numpy(normalizer.mean), "normalizer_scale": torch.from_numpy(normalizer.scale), "config": asdict(config), "epoch": epoch, "validation_loss": validation_loss}, path)


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _validate(values: np.ndarray, name: str) -> None:
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or not np.isfinite(values).all(): raise ValueError(f"{name} must be finite non-empty [time, sensor]")
