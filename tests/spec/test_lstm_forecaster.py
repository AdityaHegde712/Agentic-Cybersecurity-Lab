"""LOCKED contract for normal-only LSTM forecasting and residual recovery."""

from pathlib import Path

import numpy as np
import torch

from src.models.lstm_forecaster import (
    ForecasterConfig,
    LSTMForecaster,
    NextStepWindowDataset,
    estimate_additive_delta,
    load_forecaster_checkpoint,
    predict_residuals,
    train_forecaster,
)


def _smooth_series(length: int = 64) -> np.ndarray:
    indices = np.arange(length, dtype=np.float32)
    return np.column_stack((np.sin(indices / 5.0), np.cos(indices / 7.0))).astype(np.float32)


def test_window_dataset_produces_history_and_next_observation_pairs() -> None:
    values = _smooth_series(10)
    dataset = NextStepWindowDataset(values, context_length=4)

    context, target = dataset[0]

    assert len(dataset) == 6
    assert context.shape == (4, 2)
    assert target.shape == (2,)
    assert torch.equal(context[-1], torch.from_numpy(values[3]))
    assert torch.equal(target, torch.from_numpy(values[4]))


def test_lstm_forecaster_preserves_batch_time_and_sensor_dimensions() -> None:
    model = LSTMForecaster(input_size=2, hidden_size=8, num_layers=1)

    prediction = model(torch.zeros((3, 5, 2), dtype=torch.float32))

    assert prediction.shape == (3, 2)


def test_training_checkpoint_load_and_residual_prediction_round_trip(tmp_path: Path) -> None:
    values = _smooth_series(80)
    config = ForecasterConfig(context_length=4, hidden_size=8, batch_size=8, max_epochs=2, patience=2)
    checkpoint_path = tmp_path / "best.pt"

    trained = train_forecaster(
        values[:56],
        values[56:],
        config=config,
        checkpoint_path=checkpoint_path,
        device="cpu",
    )
    loaded = load_forecaster_checkpoint(checkpoint_path, device="cpu")
    prediction = predict_residuals(loaded, values[56:], device="cpu")

    assert checkpoint_path.exists()
    assert trained.best_validation_loss >= 0.0
    assert prediction.predicted.shape == prediction.observed.shape
    assert prediction.residual.shape == prediction.observed.shape
    assert prediction.target_indices[0] == config.context_length


def test_estimated_delta_is_the_observation_minus_normal_prediction() -> None:
    predicted = np.array([[1.0, 2.0], [3.0, 4.0]])
    observed = np.array([[1.5, 1.0], [2.0, 6.0]])

    estimated_delta = estimate_additive_delta(observed, predicted)

    assert np.array_equal(estimated_delta, np.array([[0.5, -1.0], [-1.0, 2.0]]))


def test_training_reports_epoch_metrics_to_an_optional_progress_callback(tmp_path: Path) -> None:
    values = _smooth_series(40)
    records: list[dict[str, float | int]] = []

    train_forecaster(
        values[:28],
        values[28:],
        config=ForecasterConfig(context_length=4, hidden_size=4, batch_size=8, max_epochs=1),
        checkpoint_path=tmp_path / "progress.pt",
        device="cpu",
        progress_callback=records.append,
    )

    assert len(records) == 1
    assert records[0]["epoch"] == 1
    assert "train_loss" in records[0]
