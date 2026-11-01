"""Compact summary, ranking, and plotting utilities for LSTM ablations."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_METRICS = (
    "run_name",
    "best_validation_loss",
    "point_fpr",
    "event_recall",
    "median_detection_delay",
)


def load_ranked_ablation_summary(input_dir: Path) -> pd.DataFrame:
    """Load the grid summary and return the deterministic selection ranking."""
    summary_path = input_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"ablation summary not found: {summary_path}")
    return rank_ablation_summary(pd.read_csv(summary_path))


def rank_ablation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank runs by validation loss, false positives, recall, then delay."""
    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in summary]
    if missing_metrics:
        raise ValueError(f"ablation summary missing columns: {missing_metrics}")

    ranked = summary.loc[:, REQUIRED_METRICS].copy()
    numeric_metrics = list(REQUIRED_METRICS[1:])
    for metric in numeric_metrics:
        ranked[metric] = pd.to_numeric(ranked[metric], errors="raise")
    if ranked[numeric_metrics].isna().any().any():
        raise ValueError("ablation summary metrics must be complete")
    if ranked["run_name"].duplicated().any():
        raise ValueError("ablation summary run names must be unique")

    ranked = ranked.sort_values(
        ["best_validation_loss", "point_fpr", "event_recall", "median_detection_delay"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "selection_rank", range(1, len(ranked) + 1))
    return ranked


def write_selection_report(summary: pd.DataFrame, output_dir: Path) -> Path:
    """Write an inspection report from aggregate metrics only."""
    ranked = rank_ablation_summary(summary)
    required_ranked_columns = ("selection_rank", *REQUIRED_METRICS)
    missing_columns = [column for column in required_ranked_columns if column not in ranked]
    if missing_columns:
        raise ValueError(f"ranked summary missing columns: {missing_columns}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "INSPECTION.md"
    lines = [
        "# BATADAL LSTM Ablation Inspection",
        "",
        "## Selection Ranking",
        "",
        "| Rank | Run | best_validation_loss | Point FPR | Event recall | Median delay |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in ranked.itertuples(index=False):
        lines.append(
            "| "
            f"{row.selection_rank} | {row.run_name} | {row.best_validation_loss:.6f} | "
            f"{row.point_fpr:.6f} | {row.event_recall:.6f} | "
            f"{row.median_detection_delay:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This ranking uses aggregate validation and detection metrics. "
            "It selects a forecaster candidate; it does not establish additive "
            "attack recovery. That claim requires the later known-delta injection benchmark.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def save_ranked_metrics_plot(ranked: pd.DataFrame, output_dir: Path) -> Path:
    """Save a compact four-metric comparison plot for the ranked runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("best_validation_loss", "Best validation loss"),
        ("point_fpr", "Point FPR"),
        ("event_recall", "Event recall"),
        ("median_detection_delay", "Median delay"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    labels = [f"{row.selection_rank}: {row.run_name}" for row in ranked.itertuples()]
    for axis, (metric, title) in zip(axes.flat, metrics):
        axis.bar(labels, ranked[metric])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
    plot_path = output_dir / "ranked_metrics.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path
