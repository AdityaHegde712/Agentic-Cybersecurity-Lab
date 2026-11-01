"""Create compact inspection artifacts for injected additive-attack recovery."""

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.injected_recovery_analysis import (
    save_recovery_metrics_plot,
    write_recovery_inspection_report,
)


def args() -> argparse.Namespace:
    """Parse repository-relative recovery output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/lstm_forecaster/batadal_injected_recovery"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Render compact recovery artifacts from the aggregate scenario summary."""
    settings = args()
    input_dir = PROJECT_ROOT / settings.input_dir
    output_dir = PROJECT_ROOT / settings.output_dir if settings.output_dir else input_dir / "analysis"
    summary_path = input_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"injected recovery summary not found: {summary_path}")
    summary = pd.read_csv(summary_path)
    report_path = write_recovery_inspection_report(summary, output_dir)
    plot_path = save_recovery_metrics_plot(summary, output_dir)
    print(f"Saved injected recovery inspection report to {report_path}", flush=True)
    print(f"Saved injected recovery inspection plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()
