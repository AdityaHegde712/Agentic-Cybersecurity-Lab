"""LOCKED contract for compact BATADAL ablation selection analysis."""

from pathlib import Path

import pandas as pd

from src.experiments.lstm_ablation_analysis import (
    load_ranked_ablation_summary,
    save_ranked_metrics_plot,
    write_selection_report,
)


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_name": ["slow", "balanced", "noisy"],
            "best_validation_loss": [0.15, 0.10, 0.12],
            "point_fpr": [0.01, 0.02, 0.08],
            "event_recall": [0.80, 1.00, 1.00],
            "median_detection_delay": [5.0, 2.0, 1.0],
        }
    )


def test_ranked_ablation_summary_prioritizes_validation_loss_then_operating_metrics(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "ablation"
    input_dir.mkdir()
    _summary().to_csv(input_dir / "summary.csv", index=False)

    ranked = load_ranked_ablation_summary(input_dir)

    assert ranked["run_name"].tolist() == ["balanced", "noisy", "slow"]
    assert ranked["selection_rank"].tolist() == [1, 2, 3]


def test_selection_report_records_ranking_and_excludes_raw_score_inputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "analysis"

    report_path = write_selection_report(_summary(), output_dir)
    report = report_path.read_text(encoding="utf-8")

    assert report_path == output_dir / "INSPECTION.md"
    assert "balanced" in report
    assert "best_validation_loss" in report
    assert "scores.csv" not in report
    assert "raw score" not in report.lower()


def test_ranked_metrics_plot_is_saved_from_aggregate_summary_only(tmp_path: Path) -> None:
    plot_path = save_ranked_metrics_plot(
        load_ranked_ablation_summary(_write_summary(tmp_path)),
        tmp_path / "analysis",
    )

    assert plot_path == tmp_path / "analysis" / "ranked_metrics.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def _write_summary(tmp_path: Path) -> Path:
    input_dir = tmp_path / "ablation"
    input_dir.mkdir()
    _summary().to_csv(input_dir / "summary.csv", index=False)
    return input_dir
