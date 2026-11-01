"""Compact reports and plots for known-delta recovery experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = (
    "attack_id",
    "family",
    "sensor_count",
    "magnitude_standard_deviations",
    "duration",
    "support_f1",
    "magnitude_mae_on_support",
    "sign_accuracy_on_support",
    "onset_error_samples",
    "duration_error_samples",
)


def write_recovery_inspection_report(summary: pd.DataFrame, output_dir: Path) -> Path:
    """Write a compact table of known-delta recovery metrics."""
    _validate_summary(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "INSPECTION.md"
    lines = [
        "# BATADAL Injected Additive-Attack Recovery Inspection",
        "",
        "## Known-Delta Recovery Summary",
        "",
        "| Attack | Family | Sensors | Magnitude (std) | support_f1 | magnitude_mae_on_support | sign_accuracy_on_support | Onset error | Duration error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.attack_id} | {row.family} | {row.sensor_count} | "
            f"{row.magnitude_standard_deviations:.2f} | {row.support_f1:.4f} | "
            f"{row.magnitude_mae_on_support:.6f} | "
            f"{row.sign_accuracy_on_support:.4f} | "
            f"{row.onset_error_samples:.2f} | {row.duration_error_samples:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These metrics compare forecast-residual estimates against known injected "
            "additive error. They validate the declared synthetic threat model only; "
            "they do not turn real ICS event labels into clean-signal ground truth.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def save_recovery_metrics_plot(summary: pd.DataFrame, output_dir: Path) -> Path:
    """Save a compact per-scenario recovery metric comparison."""
    _validate_summary(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    labels = summary["attack_id"].tolist()
    _bar(axes[0, 0], labels, summary["support_f1"], "Support F1")
    _bar(
        axes[0, 1],
        labels,
        summary["magnitude_mae_on_support"],
        "Magnitude MAE on support",
    )
    _bar(
        axes[1, 0],
        labels,
        summary["sign_accuracy_on_support"],
        "Sign accuracy on support",
    )
    axes[1, 1].bar(labels, summary["onset_error_samples"], label="Onset")
    axes[1, 1].bar(labels, summary["duration_error_samples"], bottom=summary["onset_error_samples"], label="Duration")
    axes[1, 1].set_title("Timing error samples")
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis="x", rotation=25)
    plot_path = output_dir / "recovery_metrics.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def _bar(axis: plt.Axes, labels: list[str], values: pd.Series, title: str) -> None:
    axis.bar(labels, values)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=25)


def _validate_summary(summary: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in summary]
    if missing_columns:
        raise ValueError(f"injected recovery summary missing columns: {missing_columns}")
