"""Create compact LSTM score and training plots without conversational raw-data loading."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input-dir", type=Path, default=Path("results") / "lstm_forecaster"); parser.add_argument("--output-dir", type=Path, default=Path("results") / "lstm_forecaster_analysis"); settings = parser.parse_args(); settings.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(settings.input_dir / "summary.csv"); summary.to_markdown(settings.output_dir / "INSPECTION.md", index=False)
    for dataset in summary.dataset:
        scores = pd.read_csv(settings.input_dir / dataset / "scores.csv"); threshold = summary.loc[summary.dataset == dataset, "threshold"].iloc[0]
        figure, axis = plt.subplots(figsize=(14, 4)); axis.plot(np.log1p(scores.lstm_score), linewidth=.6); axis.axhline(np.log1p(threshold), color="tab:orange", linestyle="--"); axis.fill_between(np.arange(len(scores)), 0, 1, where=scores.label.to_numpy() == 1, transform=axis.get_xaxis_transform(), alpha=.2, color="tab:red"); axis.set_title(f"{dataset}: LSTM residual score"); figure.tight_layout(); figure.savefig(settings.output_dir / f"{dataset}_scores.png", dpi=180); plt.close(figure)
        history_path = settings.input_dir / dataset / "training_metrics.csv"
        if history_path.exists():
            history = pd.read_csv(history_path); figure, axis = plt.subplots(); axis.plot(history.epoch, history.train_loss, label="train"); axis.plot(history.epoch, history.validation_loss, label="validation"); axis.legend(); figure.tight_layout(); figure.savefig(settings.output_dir / f"{dataset}_loss.png", dpi=180); plt.close(figure)
    print(f"Saved inspection artifacts to {settings.output_dir}", flush=True)

if __name__ == "__main__": main()
