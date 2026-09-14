"""Create compact plots and a report from statistical-baseline CSV artifacts."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.result_analysis import downsample_trace, validate_summary


METRICS = ("point_fpr", "point_tpr", "event_recall")


def parse_args() -> argparse.Namespace:
    """Parse repository-relative input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results") / "statistical_baselines" / "smoke",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "statistical_baselines" / "smoke_analysis",
    )
    parser.add_argument("--max-trace-points", type=int, default=2_000)
    return parser.parse_args()


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Save grouped score metrics without loading per-timestep CSV data."""
    datasets = list(summary["dataset"].unique())
    baselines = list(summary["baseline"].unique())
    figure, axes = plt.subplots(1, len(METRICS), figsize=(15, 4), sharey=False)

    for axis, metric in zip(axes, METRICS):
        positions = np.arange(len(datasets), dtype=float)
        width = 0.8 / len(baselines)
        for index, baseline in enumerate(baselines):
            values = [
                summary.loc[
                    (summary["dataset"] == dataset) & (summary["baseline"] == baseline),
                    metric,
                ].iloc[0]
                for dataset in datasets
            ]
            offset = (index - (len(baselines) - 1) / 2) * width
            axis.bar(positions + offset, values, width, label=baseline)
        axis.set_title(metric.replace("_", " "))
        axis.set_xticks(positions, datasets)
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("rate")
    axes[-1].legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_dataset_trace(
    dataset: str,
    summary: pd.DataFrame,
    input_dir: Path,
    output_path: Path,
    max_points: int,
) -> None:
    """Save score traces with threshold references and attack shading."""
    dataset_summary = summary.loc[summary["dataset"] == dataset]
    baselines = list(dataset_summary["baseline"])
    use_columns = ["timestamp", "label", *baselines]
    scores = pd.read_csv(input_dir / f"{dataset}_scores.csv", usecols=use_columns)
    sampled = downsample_trace(scores, max_points)
    figure, axes = plt.subplots(len(baselines), 1, figsize=(14, 2.8 * len(baselines)), sharex=True)
    axes = np.atleast_1d(axes)

    attack_mask = sampled["label"].to_numpy() == 1
    for axis, baseline in zip(axes, baselines):
        values = np.log1p(sampled[baseline].to_numpy())
        threshold = dataset_summary.loc[dataset_summary["baseline"] == baseline, "threshold"].iloc[0]
        axis.plot(sampled.index, values, linewidth=0.8, color="#4c78a8", label=f"log1p({baseline})")
        axis.axhline(np.log1p(threshold), color="#f58518", linestyle="--", label="validation threshold")
        axis.fill_between(sampled.index, 0.0, 1.0, where=attack_mask, transform=axis.get_xaxis_transform(), color="#e45756", alpha=0.18, label="attack label")
        axis.set_ylabel(baseline)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("sample index (downsampled; attack rows retained)")
    figure.suptitle(f"{dataset}: statistical-baseline score traces", y=1.01)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a compact, aggregate-only inspection report."""
    rows = [
        "# Statistical Baseline Result Inspection",
        "",
        "This report is derived from aggregate baseline summaries; raw per-timestep score rows are not embedded.",
        "",
        "| Dataset | Baseline | Point FPR | Point TPR | Event Recall | Median Delay |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        delay = row.get("median_detection_delay")
        delay_text = "n/a" if pd.isna(delay) else f"{delay:.2f}"
        rows.append(
            f"| {row['dataset']} | {row['baseline']} | {row['point_fpr']:.4f} | "
            f"{row['point_tpr']:.4f} | {row['event_recall']:.4f} | {delay_text} |"
        )
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    """Generate compact inspection artifacts from a baseline result directory."""
    args = parse_args()
    if args.max_trace_points <= 0:
        raise ValueError("max-trace-points must be positive")
    summary_path = args.input_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"baseline summary not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    validate_summary(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_summary(summary, args.output_dir / "summary_metrics.png")
    write_report(summary, args.output_dir / "INSPECTION.md")

    for dataset in summary["dataset"].unique():
        plot_dataset_trace(
            dataset,
            summary,
            args.input_dir,
            args.output_dir / f"{dataset}_traces.png",
            args.max_trace_points,
        )
    print(f"Saved inspection artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
