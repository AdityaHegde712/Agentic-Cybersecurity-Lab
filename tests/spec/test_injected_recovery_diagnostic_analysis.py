"""LOCKED contract for compact recovery-diagnostic inspection artifacts."""

from pathlib import Path

import pandas as pd

from src.experiments.injected_recovery_diagnostic_analysis import (
    save_recovery_diagnostic_plots,
    write_recovery_diagnostic_report,
)


def test_diagnostic_report_preserves_clean_drift_and_attack_phase_evidence(
    tmp_path: Path,
) -> None:
    report_path = write_recovery_diagnostic_report(
        clean_summary=pd.DataFrame(
            {
                "window_start": [32, 64],
                "window_end": [64, 96],
                "window_role": ["calibration", "assessment"],
                "active_sensor_fraction": [0.01, 0.30],
                "active_timestep_fraction": [0.02, 0.55],
            }
        ),
        phase_summary=pd.DataFrame(
            {
                "attack_id": ["step", "step", "step"],
                "phase": ["pre_attack", "attack", "post_attack"],
                "sample_count": [5, 3, 2],
                "active_sensor_fraction": [0.0, 1.0, 0.0],
                "active_timestep_fraction": [0.0, 1.0, 0.0],
            }
        ),
        output_dir=tmp_path,
    )

    report = report_path.read_text(encoding="utf-8")
    assert report_path == tmp_path / "INSPECTION.md"
    assert "Clean calibration drift" in report
    assert "pre_attack" in report
    assert "scores.csv" not in report


def test_diagnostic_plot_is_saved_from_compact_aggregate_inputs(tmp_path: Path) -> None:
    plot_paths = save_recovery_diagnostic_plots(
        clean_summary=pd.DataFrame(
            {
                "window_start": [32, 64],
                "window_end": [64, 96],
                "window_role": ["calibration", "assessment"],
                "active_sensor_fraction": [0.01, 0.30],
                "active_timestep_fraction": [0.02, 0.55],
            }
        ),
        phase_summary=pd.DataFrame(
            {
                "attack_id": ["step", "step", "step"],
                "phase": ["pre_attack", "attack", "post_attack"],
                "sample_count": [5, 3, 2],
                "active_sensor_fraction": [0.0, 1.0, 0.0],
                "active_timestep_fraction": [0.0, 1.0, 0.0],
            }
        ),
        traces={
            "step": pd.DataFrame(
                {
                    "target_index": [32, 33, 34],
                    "mean_signed_residual_standard_deviations": [0.0, 2.0, 2.0],
                    "mean_signed_true_delta_standard_deviations": [0.0, 3.0, 3.0],
                }
            )
        },
        output_dir=tmp_path,
    )

    assert set(plot_paths) == {"activation", "traces"}
    assert all(path.exists() and path.stat().st_size > 0 for path in plot_paths.values())
