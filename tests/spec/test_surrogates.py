"""Specification tests for plant dynamics surrogate models and dataset pipeline.

Validates:
- SurrogatePrediction dataclass contract and immutability.
- CompanionDataset sliding window feature extraction and standardization.
- create_companion_dataloaders construction and batch integrity.
- RidgeCompanionSurrogate: fit, predict, metrics, serialization round-trip.
- PhysicsGuidedMLPSurrogate: mass-regularized fit, predict, serialization round-trip.
- DilatedTCNSurrogate: causal feed-forward sequence processing, serialization round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.inverse.surrogates import (
    BasePlantSurrogate,
    CompanionDataset,
    DilatedTCNSurrogate,
    PhysicsGuidedMLPSurrogate,
    RidgeCompanionSurrogate,
    SurrogatePrediction,
    create_companion_dataloaders,
    get_batadal_companion_channels,
)


@pytest.fixture
def mock_manifest_and_df(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """Create a synthetic dataset and minimal manifest for fast contract testing."""
    n_samples = 120
    # 6 mock companion channels (including 2 flow channels)
    companion_channels = ["L_T2", "F_PU1", "F_PU2", "P_J280", "S_PU1", "S_PU2"]
    target_channel = "L_T1"

    rng = np.random.RandomState(42)
    data: dict[str, np.ndarray] = {
        "timestamp": np.arange(n_samples),
        target_channel: 2.5 + np.sin(np.linspace(0, 10, n_samples)) + 0.05 * rng.randn(n_samples),
    }
    for ch in companion_channels:
        if ch.startswith("F_"):
            data[ch] = 50.0 + 10.0 * rng.randn(n_samples)
        elif ch.startswith("S_"):
            data[ch] = (rng.rand(n_samples) > 0.5).astype(float)
        else:
            data[ch] = 2.0 + rng.randn(n_samples)

    df = pd.DataFrame(data)

    manifest_dict = {
        "channels": [
            {
                "name": target_channel,
                "role": "target_measurement",
                "attackability": "attackable",
            }
        ]
        + [
            {
                "name": ch,
                "role": "recorded_companion",
                "attackability": "declared_unaffected",
            }
            for ch in companion_channels
        ]
    }
    manifest_path = tmp_path / "mock-manifest.json"
    manifest_path.write_text(json.dumps(manifest_dict), encoding="utf-8")
    return manifest_path, df


# =====================================================================
# 1. Prediction Schema & Contract Tests
# =====================================================================

def test_surrogate_prediction_schema_and_immutability() -> None:
    """SurrogatePrediction must enforce type safety and frozen immutability."""
    y_hat = np.array([2.1, 2.2, 2.3])
    residuals = np.array([0.1, -0.1, 0.0])
    metrics = {"rmse": 0.1, "mae": 0.067, "r2": 0.95}

    pred = SurrogatePrediction(
        y_hat=y_hat,
        residuals=residuals,
        residual_std=0.0816,
        metrics=metrics,
    )

    assert len(pred.y_hat) == 3
    assert len(pred.residuals) == 3
    assert pred.residual_std == pytest.approx(0.0816)
    assert pred.metrics["rmse"] == 0.1

    # Immutability
    with pytest.raises(AttributeError):
        pred.residual_std = 0.5  # type: ignore

    # Length mismatch
    with pytest.raises(ValueError, match="must match residuals length"):
        SurrogatePrediction(
            y_hat=np.array([1.0, 2.0]),
            residuals=np.array([0.1]),
            residual_std=0.1,
            metrics={},
        )


def test_base_plant_surrogate_compute_metrics() -> None:
    """BasePlantSurrogate.compute_metrics computes exact RMSE, MAE, and R2."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.0, 4.2, 4.8])

    m = BasePlantSurrogate.compute_metrics(y_true, y_pred)
    assert "rmse" in m and "mae" in m and "r2" in m
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    assert m["mae"] == pytest.approx(np.mean(np.abs(y_true - y_pred)))
    assert m["r2"] > 0.95


# =====================================================================
# 2. CompanionDataset & DataLoader Tests
# =====================================================================

def test_companion_dataset_sliding_window_unpadded(mock_manifest_and_df: tuple[Path, pd.DataFrame]) -> None:
    """CompanionDataset unpadded length is N - context_length + 1 with correct window shape."""
    manifest_path, df = mock_manifest_and_df
    context_length = 12
    channels = ["L_T2", "F_PU1", "F_PU2", "P_J280", "S_PU1", "S_PU2"]

    ds = CompanionDataset(
        df=df,
        target_channel="L_T1",
        companion_channels=channels,
        context_length=context_length,
        normalize=True,
        pad_initial=False,
    )

    expected_len = len(df) - context_length + 1
    assert len(ds) == expected_len

    x_win, y_target = ds[0]
    assert x_win.shape == (context_length, len(channels))
    assert y_target.shape == (1,)
    assert x_win.dtype == torch.float32
    assert y_target.dtype == torch.float32

    # y_target at index 0 corresponds to df['L_T1'].iloc[context_length - 1]
    assert y_target.item() == pytest.approx(float(df["L_T1"].iloc[context_length - 1]), abs=1e-5)


def test_companion_dataset_sliding_window_padded(mock_manifest_and_df: tuple[Path, pd.DataFrame]) -> None:
    """CompanionDataset with pad_initial=True produces exactly len(df) items."""
    manifest_path, df = mock_manifest_and_df
    context_length = 12
    channels = ["L_T2", "F_PU1", "F_PU2", "P_J280", "S_PU1", "S_PU2"]

    ds = CompanionDataset(
        df=df,
        target_channel="L_T1",
        companion_channels=channels,
        context_length=context_length,
        normalize=True,
        pad_initial=True,
    )

    assert len(ds) == len(df)
    x_win0, y_target0 = ds[0]
    assert x_win0.shape == (context_length, len(channels))
    assert y_target0.item() == pytest.approx(float(df["L_T1"].iloc[0]), abs=1e-5)


def test_create_companion_dataloaders(mock_manifest_and_df: tuple[Path, pd.DataFrame]) -> None:
    """create_companion_dataloaders constructs PyTorch DataLoaders with preserved scaler."""
    manifest_path, df = mock_manifest_and_df
    train_df = df.iloc[:80]
    val_df = df.iloc[80:]

    loaders = create_companion_dataloaders(
        train_df=train_df,
        val_df=val_df,
        context_length=8,
        batch_size=16,
        manifest_path=manifest_path,
    )

    assert len(loaders) == 2
    train_loader, val_loader = loaders

    bx, by = next(iter(train_loader))
    assert bx.shape[0] == 16
    assert bx.shape[1] == 8
    assert by.shape == (16, 1)


# =====================================================================
# 3. Ridge Companion Surrogate Tests
# =====================================================================

def test_ridge_companion_surrogate_fit_predict_save_load(
    mock_manifest_and_df: tuple[Path, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """RidgeCompanionSurrogate fits closed-form ridge, predicts, and survives round-trip serialization."""
    manifest_path, df = mock_manifest_and_df
    channels = ["L_T2", "F_PU1", "F_PU2", "P_J280", "S_PU1", "S_PU2"]
    train_df = df.iloc[:80]
    test_df = df.iloc[80:]

    model = RidgeCompanionSurrogate(
        alpha=1.0,
        target_channel="L_T1",
        companion_channels=channels,
    )
    assert not model.is_fitted

    model.fit(train_df)
    assert model.is_fitted
    assert model.weights_ is not None
    assert len(model.weights_) == len(channels)

    pred = model.predict(test_df)
    assert isinstance(pred, SurrogatePrediction)
    assert len(pred.y_hat) == len(test_df)
    assert len(pred.residuals) == len(test_df)
    assert pred.residual_std > 0.0
    assert "rmse" in pred.metrics

    # Save and load round-trip
    save_path = tmp_path / "ridge_test.pt"
    model.save(save_path)
    assert save_path.exists()

    loaded_model = RidgeCompanionSurrogate.load(save_path)
    assert loaded_model.is_fitted
    assert loaded_model.alpha == model.alpha

    pred_loaded = loaded_model.predict(test_df)
    np.testing.assert_allclose(pred.y_hat, pred_loaded.y_hat, rtol=1e-6)
    np.testing.assert_allclose(pred.residuals, pred_loaded.residuals, rtol=1e-6)


# =====================================================================
# 4. PhysicsGuidedMLPSurrogate Tests
# =====================================================================

def test_physics_guided_mlp_surrogate_fit_predict_save_load(
    mock_manifest_and_df: tuple[Path, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """PhysicsGuidedMLPSurrogate trains with mass loss, predicts len(df), and round-trips."""
    manifest_path, df = mock_manifest_and_df
    channels = ["L_T2", "F_PU1", "F_PU2", "P_J280", "S_PU1", "S_PU2"]
    train_df = df.iloc[:80]
    test_df = df.iloc[80:]

    model = PhysicsGuidedMLPSurrogate(
        context_length=6,
        hidden_dims=(32, 16),
        dropout=0.0,
        gamma_mass=0.1,
        lr=0.01,
        epochs=3,
        batch_size=16,
        device="cpu",
        target_channel="L_T1",
        companion_channels=channels,
    )

    model.fit(train_df)
    assert model.is_fitted
    assert len(model.flow_indices_) == 2  # F_PU1 and F_PU2
    assert model.flow_weights_ is not None

    pred = model.predict(test_df)
    assert isinstance(pred, SurrogatePrediction)
    assert len(pred.y_hat) == len(test_df)
    assert len(pred.residuals) == len(test_df)
    assert pred.residual_std >= 0.0

    # Serialization round-trip
    save_path = tmp_path / "pinn_test.pt"
    model.save(save_path)
    assert save_path.exists()

    loaded_model = PhysicsGuidedMLPSurrogate.load(save_path)
    assert loaded_model.is_fitted
    assert loaded_model.context_length == 6
    assert loaded_model.hidden_dims == (32, 16)

    pred_loaded = loaded_model.predict(test_df)
    np.testing.assert_allclose(pred.y_hat, pred_loaded.y_hat, atol=1e-4)


# =====================================================================
# 5. DilatedTCNSurrogate Tests
# =====================================================================

def test_dilated_tcn_surrogate_fit_predict_save_load(
    mock_manifest_and_df: tuple[Path, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """DilatedTCNSurrogate trains causal dilated convs, predicts len(df), and round-trips."""
    manifest_path, df = mock_manifest_and_df
    channels = ["L_T2", "F_PU1", "F_PU2", "P_J280", "S_PU1", "S_PU2"]
    train_df = df.iloc[:80]
    test_df = df.iloc[80:]

    model = DilatedTCNSurrogate(
        context_length=6,
        hidden_channels=16,
        dilations=(1, 2),
        kernel_size=3,
        dropout=0.0,
        lr=0.01,
        epochs=3,
        batch_size=16,
        device="cpu",
        target_channel="L_T1",
        companion_channels=channels,
    )

    model.fit(train_df)
    assert model.is_fitted

    pred = model.predict(test_df)
    assert isinstance(pred, SurrogatePrediction)
    assert len(pred.y_hat) == len(test_df)
    assert len(pred.residuals) == len(test_df)
    assert pred.residual_std >= 0.0

    # Serialization round-trip
    save_path = tmp_path / "tcn_test.pt"
    model.save(save_path)
    assert save_path.exists()

    loaded_model = DilatedTCNSurrogate.load(save_path)
    assert loaded_model.is_fitted
    assert loaded_model.hidden_channels == 16
    assert loaded_model.dilations == (1, 2)

    pred_loaded = loaded_model.predict(test_df)
    np.testing.assert_allclose(pred.y_hat, pred_loaded.y_hat, atol=1e-4)


# =====================================================================
# 6. Contract Consistency Across All 3 Surrogates
# =====================================================================

def test_surrogate_polymorphic_contract(mock_manifest_and_df: tuple[Path, pd.DataFrame]) -> None:
    """All 3 surrogates inherit from BasePlantSurrogate and emit identical prediction contracts."""
    manifest_path, df = mock_manifest_and_df
    channels = ["L_T2", "F_PU1", "F_PU2", "P_J280", "S_PU1", "S_PU2"]
    train_df = df.iloc[:60]
    eval_df = df.iloc[60:]

    models: list[BasePlantSurrogate] = [
        RidgeCompanionSurrogate(companion_channels=channels),
        PhysicsGuidedMLPSurrogate(context_length=4, hidden_dims=(16,), epochs=1, companion_channels=channels),
        DilatedTCNSurrogate(context_length=4, hidden_channels=8, dilations=(1,), epochs=1, companion_channels=channels),
    ]

    for m in models:
        assert isinstance(m, BasePlantSurrogate)
        m.fit(train_df)
        pred = m.predict(eval_df)
        assert isinstance(pred, SurrogatePrediction)
        assert len(pred.y_hat) == len(eval_df)
        assert len(pred.residuals) == len(eval_df)
        assert isinstance(pred.residual_std, float)
        assert "rmse" in pred.metrics
        assert "mae" in pred.metrics
        assert "r2" in pred.metrics
