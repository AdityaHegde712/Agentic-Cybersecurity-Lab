"""Create compact report and plots for a completed LSTM ablation grid."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.lstm_ablation_analysis import (
    load_ranked_ablation_summary,
    save_ranked_metrics_plot,
    write_selection_report,
)


def args() -> argparse.Namespace:
    """Parse repository-relative ablation inspection paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/lstm_forecaster/batadal_ablation"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Rank aggregate results and save compact human-inspectable artifacts."""
    settings = args()
    input_dir = PROJECT_ROOT / settings.input_dir
    output_dir = PROJECT_ROOT / settings.output_dir if settings.output_dir else input_dir / "analysis"
    ranked = load_ranked_ablation_summary(input_dir)
    report_path = write_selection_report(ranked, output_dir)
    plot_path = save_ranked_metrics_plot(ranked, output_dir)
    print(f"Saved ablation inspection report to {report_path}", flush=True)
    print(f"Saved ablation inspection plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()
