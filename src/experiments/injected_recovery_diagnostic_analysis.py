"""Compact reports and plots for calibration and temporal-support diagnostics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_recovery_diagnostic_report(
    clean_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write a compact interpretation-ready calibration diagnostic report."""
    _validate_clean_summary(clean_summary)
    _validate_phase_summary(phase_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "INSPECTION.md"
    lines = [
        "# BATADAL Recovery Calibration and Temporal-Support Diagnostic",
        "",
        "## Clean calibration drift",
        "",
        "| Window | Role | Active sensor fraction | Active timestep fraction |",
        "|---:|---|---:|---:|",
    ]
    for row in clean_summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.window_start}-{row.window_end} | {row.window_role} | "
            f"{row.active_sensor_fraction:.4f} | {row.active_timestep_fraction:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Attacked-sensor temporal support",
            "",
            "| Attack | Phase | Samples | Active sensor fraction | Active timestep fraction |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in phase_summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.attack_id} | {row.phase} | {row.sample_count} | "
            f"{row.active_sensor_fraction:.4f} | {row.active_timestep_fraction:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Clean-window activity tests whether thresholds remain calibrated outside "
            "their prefix. Attack-phase activity is restricted to injected sensors so "
            "unrelated sensor residuals cannot obscure recovery behavior. Trace plots "
            "compare forecast residual with the known synthetic delta; they are a "
            "diagnostic, not a delta-recovery claim.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def save_recovery_diagnostic_plots(
    clean_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    traces: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Path]:
    """Render aggregate activity and known-delta trace plots."""
    _validate_clean_summary(clean_summary)
    _validate_phase_summary(phase_summary)
    _validate_traces(traces)
    output_dir.mkdir(parents=True, exist_ok=True)

    activation_path = output_dir / "activation_drift_and_support.png"
    _save_activation_plot(clean_summary, phase_summary, activation_path)
    trace_path = output_dir / "attacked_sensor_traces.png"
    _save_trace_plot(traces, trace_path)
    return {"activation": activation_path, "traces": trace_path}


def _save_activation_plot(
    clean_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    labels = [f"{start}-{end}" for start, end in zip(clean_summary["window_start"], clean_summary["window_end"])]
    axes[0].plot(labels, clean_summary["active_sensor_fraction"], marker="o", label="Sensor fraction")
    axes[0].plot(labels, clean_summary["active_timestep_fraction"], marker="o", label="Timestep fraction")
    axes[0].set_title("Clean threshold activity by time window")
    axes[0].set_ylabel("Fraction active")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend()

    pivot = phase_summary.pivot(index="attack_id", columns="phase", values="active_sensor_fraction")
    pivot = pivot.reindex(columns=["pre_attack", "attack", "post_attack"])
    pivot.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Attacked-sensor activity by phase")
    axes[1].set_ylabel("Active sensor fraction")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(title="Phase")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_trace_plot(traces: dict[str, pd.DataFrame], output_path: Path) -> None:
    figure, axes = plt.subplots(len(traces), 1, figsize=(12, max(3, 2.7 * len(traces))), squeeze=False, constrained_layout=True)
    for axis, (attack_id, trace) in zip(axes[:, 0], traces.items()):
        axis.plot(
            trace["target_index"],
            trace["mean_signed_residual_standard_deviations"],
            label="Forecast residual",
        )
        axis.plot(
            trace["target_index"],
            trace["mean_signed_true_delta_standard_deviations"],
            label="Known injected delta",
        )
        axis.set_title(f"{attack_id}: attacked-sensor mean trace")
        axis.set_xlabel("Target index")
        axis.set_ylabel("Standard deviations")
        axis.legend(loc="upper right")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _validate_clean_summary(summary: pd.DataFrame) -> None:
    _require_columns(
        summary,
        {
            "window_start",
            "window_end",
            "window_role",
            "active_sensor_fraction",
            "active_timestep_fraction",
        },
        "clean summary",
    )


def _validate_phase_summary(summary: pd.DataFrame) -> None:
    _require_columns(
        summary,
        {
            "attack_id",
            "phase",
            "sample_count",
            "active_sensor_fraction",
            "active_timestep_fraction",
        },
        "phase summary",
    )


def _validate_traces(traces: dict[str, pd.DataFrame]) -> None:
    if not traces:
        raise ValueError("at least one attack trace is required")
    for attack_id, trace in traces.items():
        if not attack_id:
            raise ValueError("attack trace ids must be non-empty")
        _require_columns(
            trace,
            {
                "target_index",
                "mean_signed_residual_standard_deviations",
                "mean_signed_true_delta_standard_deviations",
            },
            "attack trace",
        )


def _require_columns(summary: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
