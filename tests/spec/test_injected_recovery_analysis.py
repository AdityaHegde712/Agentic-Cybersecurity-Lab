"""LOCKED contract for compact known-delta recovery inspection artifacts."""

from pathlib import Path

import pandas as pd

from src.experiments.injected_recovery_analysis import (
    save_recovery_metrics_plot,
    write_recovery_inspection_report,
)


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "attack_id": ["step-single", "ramp-single"],
            "family": ["step", "ramp"],
            "sensor_count": [1, 1],
            "magnitude_standard_deviations": [3.0, 3.0],
            "duration": [400, 500],
            "support_f1": [0.8, 0.6],
            "magnitude_mae_on_support": [0.2, 0.5],
            "sign_accuracy_on_support": [1.0, 0.9],
            "onset_error_samples": [1.0, 3.0],
            "duration_error_samples": [10.0, 20.0],
        }
    )


def test_recovery_inspection_report_preserves_known_delta_metric_names(
    tmp_path: Path,
) -> None:
    report_path = write_recovery_inspection_report(_summary(), tmp_path)
    report = report_path.read_text(encoding="utf-8")

    assert report_path == tmp_path / "INSPECTION.md"
    assert "support_f1" in report
    assert "magnitude_mae_on_support" in report
    assert "step-single" in report
    assert "scores.csv" not in report


def test_recovery_metrics_plot_is_saved_from_aggregate_summary(tmp_path: Path) -> None:
    plot_path = save_recovery_metrics_plot(_summary(), tmp_path)

    assert plot_path == tmp_path / "recovery_metrics.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
