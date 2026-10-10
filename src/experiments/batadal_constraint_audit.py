"""Aggregate constraint inspection for the narrow BATADAL inverse benchmark."""

from typing import Any

import pandas as pd


def summarize_constraints(
    values: pd.DataFrame,
    *,
    target_sensor: str,
    companion_count: int,
) -> dict[str, Any]:
    """Summarize clean sensor ranges and correlation-ranked candidates."""
    if values.empty:
        raise ValueError("values must contain at least one row")
    if target_sensor not in values.columns:
        raise ValueError("target_sensor must be one of the value columns")
    if companion_count < 1:
        raise ValueError("companion_count must be positive")
    if not all(pd.api.types.is_numeric_dtype(values[column]) for column in values.columns):
        raise ValueError("values must contain only numeric sensor columns")

    target = values[target_sensor]
    sensor_standard_deviations = values.std(axis=0, ddof=0)
    if sensor_standard_deviations[target_sensor] == 0.0:
        raise ValueError("target_sensor must vary across clean rows")
    variable_columns = sensor_standard_deviations[sensor_standard_deviations > 0.0].index
    correlations = values.loc[:, variable_columns].corrwith(target)
    candidates = [
        {
            "sensor": sensor,
            "correlation": float(correlation),
            "absolute_correlation": float(abs(correlation)),
        }
        for sensor, correlation in correlations.items()
        if sensor != target_sensor and pd.notna(correlation)
    ]
    candidates.sort(key=lambda candidate: (-candidate["absolute_correlation"], candidate["sensor"]))

    return {
        "target_sensor": target_sensor,
        "row_count": int(len(values)),
        "sensor_count": int(len(values.columns)),
        "constant_sensor_count": int((sensor_standard_deviations == 0.0).sum()),
        "target_range": {
            "minimum": float(target.min()),
            "quantile_01": float(target.quantile(0.01)),
            "median": float(target.median()),
            "quantile_99": float(target.quantile(0.99)),
            "maximum": float(target.max()),
            "standard_deviation": float(target.std(ddof=0)),
            "missing_count": int(target.isna().sum()),
        },
        "companion_candidates": candidates[:companion_count],
    }
