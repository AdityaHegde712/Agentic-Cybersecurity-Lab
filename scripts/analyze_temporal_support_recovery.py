"""Create compact comparison artifacts for temporal-support recovery."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def args() -> argparse.Namespace:
    """Parse repository-relative comparison paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/lstm_forecaster/batadal_temporal_support_recovery"),
    )
    return parser.parse_args()


def main() -> None:
    """Write a table and plot from the aggregate estimator comparison."""
    settings = args()
    input_dir = PROJECT_ROOT / settings.input_dir
    summary = pd.read_csv(input_dir / "summary.csv")
    required = {"attack_id", "estimator", "support_f1", "sign_accuracy_on_support", "onset_error_samples"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"temporal-support summary missing columns: {missing}")
    analysis_dir = input_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    pivot = summary.pivot(index="attack_id", columns="estimator", values="support_f1")
    report = [
        "# BATADAL Temporal-Support Recovery Comparison",
        "",
        "## Support F1 by estimator",
        "",
        "| Attack | Pointwise | Temporal support |",
        "|---|---:|---:|",
    ]
    for attack_id, row in pivot.iterrows():
        report.append(
            "| "
            f"{attack_id} | {row.get('pointwise', float('nan')):.4f} | "
            f"{row.get('temporal_support', float('nan')):.4f} |"
        )
    report.extend(
        [
        "",
        "## Interpretation boundary",
        "",
        "The temporal-support estimator is label-free at inference: it applies per-sensor thresholds and duration rules calibrated on clean data. Injected sensor identities are used only for this known-delta evaluation.",
        ]
    )
    (analysis_dir / "INSPECTION.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Known-delta support F1")
    axes[0].set_ylabel("F1")
    axes[0].tick_params(axis="x", rotation=25)
    timing = summary.pivot(index="attack_id", columns="estimator", values="onset_error_samples")
    timing.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Onset error")
    axes[1].set_ylabel("Samples")
    axes[1].tick_params(axis="x", rotation=25)
    figure.savefig(analysis_dir / "estimator_comparison.png", dpi=160)
    plt.close(figure)
    print(f"Saved temporal-support inspection artifacts to {analysis_dir}", flush=True)


if __name__ == "__main__":
    main()
