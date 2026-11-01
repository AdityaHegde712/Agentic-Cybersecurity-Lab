"""Train or load normal-only LSTM forecasters and score held-out test data."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.data.registry import get_meta, load_dataset
from src.evaluation.baseline_runner import calibrate_blocked_threshold, prepare_feature_matrices, summarize_score_series
from src.experiments.lstm_ablation_config import load_ablation_config
from src.experiments.lstm_ablation_runner import plan_ablation_runs, write_ablation_manifest
from src.models.lstm_forecaster import ForecasterConfig, load_forecaster_checkpoint, predict_residuals, train_forecaster

DATASETS = ("hai", "swat", "wadi", "batadal")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["all", *DATASETS], default=["all"])
    parser.add_argument("--limit", type=int, default=None); parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=32); parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256); parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--validation-blocks", type=int, default=5); parser.add_argument("--threshold-quantile", type=float, default=.99)
    parser.add_argument("--device", default="auto"); parser.add_argument("--mode", choices=["train-predict", "predict"], default="train-predict")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ablation-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "lstm_forecaster")
    return parser.parse_args()


def run(dataset: str, settings: argparse.Namespace, device: str) -> pd.DataFrame:
    sensors = list(get_meta(dataset)["sensor_columns"]); train = load_dataset(dataset, "train"); validation = load_dataset(dataset, "val"); test = load_dataset(dataset, "test")
    if settings.limit: train, validation, test = train.iloc[:settings.limit], validation.iloc[:settings.limit], test.iloc[:settings.limit]
    train_values, validation_values, test_values, sensors = prepare_feature_matrices(train.loc[train.label == 0, sensors], validation.loc[validation.label == 0, sensors], test.loc[:, sensors], sensors)
    run_dir = settings.output_dir / dataset; checkpoint = run_dir / "checkpoints" / "best.pt"
    config = ForecasterConfig(context_length=settings.context_length, hidden_size=settings.hidden_size, batch_size=settings.batch_size, max_epochs=settings.epochs, patience=settings.patience, seed=settings.seed)
    best_validation_loss: float | None = None
    if settings.mode == "train-predict":
        def report(record: dict[str, float | int]) -> None:
            print(f"[{dataset}] epoch {record['epoch']}/{config.max_epochs}: train_loss={record['train_loss']:.6f} validation_loss={record['validation_loss']:.6f}", flush=True)
        result = train_forecaster(train_values, validation_values, config=config, checkpoint_path=checkpoint, device=device, progress_callback=report)
        pd.DataFrame(result.history).to_csv(run_dir / "training_metrics.csv", index=False)
        best_validation_loss = result.best_validation_loss
    forecaster = load_forecaster_checkpoint(checkpoint, device=device)
    validation_prediction = predict_residuals(forecaster, validation_values, device=device); test_prediction = predict_residuals(forecaster, test_values, device=device)
    validation_scores = np.max(np.abs(validation_prediction.residual / forecaster.normalizer.scale), axis=1)
    test_scores = np.max(np.abs(test_prediction.residual / forecaster.normalizer.scale), axis=1)
    labels = test.label.to_numpy(dtype=np.int8)[test_prediction.target_indices]
    threshold = calibrate_blocked_threshold(validation_scores, threshold_quantile=settings.threshold_quantile, validation_blocks=settings.validation_blocks)
    summary = summarize_score_series(validation_scores, test_scores, labels, threshold_quantile=settings.threshold_quantile, validation_blocks=settings.validation_blocks)
    frame = test.iloc[test_prediction.target_indices][["timestamp", "label", "attack_id"]].copy(); frame["lstm_score"] = test_scores
    frame.to_csv(run_dir / "scores.csv", index=False)
    result = {"dataset": dataset, "baseline": "lstm_forecaster", **summary}
    if best_validation_loss is not None:
        result["best_validation_loss"] = best_validation_loss
    if hasattr(settings, "run_name"):
        result["run_name"] = settings.run_name
    return pd.DataFrame([result])


def run_ablation(config_path: Path, settings: argparse.Namespace, device: str) -> None:
    """Execute each named configuration and collect a single grid-level summary."""
    experiment = load_ablation_config(config_path)
    planned_runs = plan_ablation_runs(experiment)
    output_root = PROJECT_ROOT / experiment.output_dir
    write_ablation_manifest(experiment, output_root)
    summaries: list[pd.DataFrame] = []

    for planned_run in planned_runs:
        run_settings = argparse.Namespace(**vars(settings))
        run_settings.datasets = [experiment.dataset]
        run_settings.limit = experiment.limit
        run_settings.context_length = planned_run.forecaster_config.context_length
        run_settings.hidden_size = planned_run.forecaster_config.hidden_size
        run_settings.epochs = planned_run.forecaster_config.max_epochs
        run_settings.seed = planned_run.forecaster_config.seed
        run_settings.output_dir = PROJECT_ROOT / planned_run.output_dir
        run_settings.run_name = planned_run.name
        summaries.append(run(experiment.dataset, run_settings, device))

    pd.concat(summaries, ignore_index=True).to_csv(
        output_root / "summary.csv",
        index=False,
    )
    print(f"Saved ablation artifacts to {experiment.output_dir} on {device}", flush=True)


def main() -> None:
    settings = args(); device = "cuda" if settings.device == "auto" and torch.cuda.is_available() else ("cpu" if settings.device == "auto" else settings.device)
    if settings.ablation_config is not None:
        run_ablation(settings.ablation_config, settings, device)
        return
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = DATASETS if "all" in settings.datasets else settings.datasets; summaries = [run(dataset, settings, device) for dataset in datasets]
    pd.concat(summaries, ignore_index=True).to_csv(settings.output_dir / "summary.csv", index=False)
    print(f"Saved LSTM artifacts to {settings.output_dir} on {device}", flush=True)


if __name__ == "__main__": main()
