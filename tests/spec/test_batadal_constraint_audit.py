"""LOCKED contract for aggregate inverse-benchmark constraint inspection."""

import pandas as pd
import pytest

from src.experiments.batadal_constraint_audit import summarize_constraints


def test_constraint_summary_reports_ranges_and_ranked_companion_candidates() -> None:
    values = pd.DataFrame(
        {
            "target": [0.0, 1.0, 2.0, 3.0, 4.0],
            "companion": [0.0, 2.0, 4.0, 6.0, 8.0],
            "opposed": [4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )

    summary = summarize_constraints(values, target_sensor="target", companion_count=2)

    assert summary["target_sensor"] == "target"
    assert summary["sensor_count"] == 3
    assert summary["constant_sensor_count"] == 0
    assert summary["row_count"] == 5
    assert summary["target_range"]["minimum"] == 0.0
    assert summary["target_range"]["maximum"] == 4.0
    assert summary["companion_candidates"][0]["sensor"] == "companion"
    assert summary["companion_candidates"][0]["correlation"] == pytest.approx(1.0)
    assert summary["companion_candidates"][1]["sensor"] == "opposed"
    assert summary["companion_candidates"][1]["correlation"] == pytest.approx(-1.0)
