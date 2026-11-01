"""Render compact plots and an inspection report from recovery diagnostics."""

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.injected_recovery_diagnostic_analysis import (
    save_recovery_diagnostic_plots,
    write_recovery_diagnostic_report,
)


def args() -> argparse.Namespace:
    """Parse repository-relative diagnostic artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/lstm_forecaster/batadal_injected_recovery/diagnostic"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Create compact inspection artifacts without opening raw experiment output."""
    settings = args()
    input_dir = PROJECT_ROOT / settings.input_dir
    output_dir = PROJECT_ROOT / settings.output_dir if settings.output_dir else input_dir / "analysis"
    clean_summary = pd.read_csv(input_dir / "clean_window_activation_summary.csv")
    phase_summary = pd.read_csv(input_dir / "scenario_phase_activation_summary.csv")
    traces = {
        path.stem: pd.read_csv(path)
        for path in sorted((input_dir / "traces").glob("*.csv"))
    }
    report_path = write_recovery_diagnostic_report(clean_summary, phase_summary, output_dir)
    plot_paths = save_recovery_diagnostic_plots(clean_summary, phase_summary, traces, output_dir)
    print(f"Saved recovery diagnostic inspection report to {report_path}", flush=True)
    for name, path in plot_paths.items():
        print(f"Saved recovery diagnostic {name} plot to {path}", flush=True)


if __name__ == "__main__":
    main()
