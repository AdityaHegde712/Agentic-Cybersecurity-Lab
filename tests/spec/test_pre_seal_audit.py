"""LOCKED specification tests for Consolidated Pre-Seal Audit (November 18 Milestone Step 4).

Verifies the consolidated pre-seal benchmark audit across 120 development scenarios:
1. Data Seal Integrity: Strictly reads only rows 0-960 of val.parquet (clean calibration: 0-480,
   development: 480-960) and asserts zero access or evaluation on the sealed final test block
   (rows 960-1440).
2. Unified Audit Schema: summary.json contains all 4 key configurations:
   - StepMAP_TCN_Declared
   - StepMAP_TCN_Undeclared
   - CausalStepMAP_TCN_Declared
   - CausalStepMAP_TCN_Undeclared
   and baseline estimators (PersistenceEstimator, CompanionRegressionEstimator).
3. Precommitted Gate Invariants across 120 development scenarios:
   - CausalStepMAP_TCN (both declared and undeclared) beats PersistenceEstimator by >= 20.0% median Delta NRMSE.
   - Median signed amplitude error <= 0.25 sigma_train.
   - Sign accuracy on attacks >= 1.0 sigma is >= 95.0%.
   - Empirical 90% coverage is >= 85.0%.
   - Sharpness is < 0.25 m.
   - Zero hard box-constraint violations.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from src.inverse.benchmark_manifest import BenchmarkManifest


AUDIT_DIR = Path("results/narrow_inverse_research/pre_seal_audit")
SUMMARY_JSON_PATH = AUDIT_DIR / "summary.json"
SUMMARY_CSV_PATH = AUDIT_DIR / "summary.csv"
MANIFEST_PATH = Path("configs/manifests/batadal-benchmark-manifest-v1.json")


# =====================================================================
# Test 1: Data Seal Integrity
# =====================================================================

def test_data_seal_integrity() -> None:
    """Assert pre-seal audit strictly reads only rows 0-960 and leaves rows 960-1440 untouched."""
    assert SUMMARY_JSON_PATH.exists(), f"Pre-seal audit summary missing at {SUMMARY_JSON_PATH}"
    data = json.loads(SUMMARY_JSON_PATH.read_text(encoding="utf-8"))

    # Verify audit seal metadata
    assert "data_seal_integrity" in data, "summary.json missing 'data_seal_integrity' block"
    seal = data["data_seal_integrity"]

    assert seal["clean_calibration_rows"] == [0, 480], (
        f"clean_calibration must be rows [0, 480], got {seal['clean_calibration_rows']}"
    )
    assert seal["development_injection_rows"] == [480, 960], (
        f"development_injection must be rows [480, 960], got {seal['development_injection_rows']}"
    )
    assert seal["sealed_final_injection_rows"] == [960, 1440], (
        f"sealed_final_injection must be rows [960, 1440], got {seal['sealed_final_injection_rows']}"
    )
    assert seal["max_val_row_accessed"] <= 960, (
        f"Seal breach! Max val row accessed {seal['max_val_row_accessed']} > 960"
    )
    assert seal["sealed_block_accessed"] is False, "Sealed final test block was accessed!"
    assert seal["sealed_status"] in ("SEALED_LOCKED", "LOCKED", "UNTOUCHED"), (
        f"Unexpected sealed_status: {seal['sealed_status']}"
    )

    # Cross-verify with BenchmarkManifest
    assert MANIFEST_PATH.exists()
    manifest = BenchmarkManifest.from_json(MANIFEST_PATH.read_text(encoding="utf-8").strip())
    final_block = next(b for b in manifest.blocks if b.name == "sealed_final_injection")
    assert final_block.row_start == 960
    assert final_block.row_stop == 1440

    # Cross-verify with summary.csv
    assert SUMMARY_CSV_PATH.exists(), f"summary.csv missing at {SUMMARY_CSV_PATH}"
    df = pd.read_csv(SUMMARY_CSV_PATH)
    if "block_name" in df.columns:
        assert (df["block_name"] == "development_injection").all(), (
            "Non-development block scenarios found in pre-seal audit CSV"
        )
    # Each scenario onset and offset must be within 0..480 relative to the development block
    assert (df["offset_index"] <= 480).all(), (
        f"Scenario offset exceeds development block window length: max={df['offset_index'].max()}"
    )


# =====================================================================
# Test 2: Unified Audit Schema
# =====================================================================

def test_unified_audit_schema() -> None:
    """Assert consolidated audit generates valid summary.json with all 4 configurations plus baselines."""
    assert SUMMARY_JSON_PATH.exists(), f"Pre-seal audit summary missing at {SUMMARY_JSON_PATH}"
    data = json.loads(SUMMARY_JSON_PATH.read_text(encoding="utf-8"))

    assert "estimators" in data, "summary.json missing 'estimators' dict"
    estimators = data["estimators"]

    required_configs = [
        "StepMAP_TCN_Declared",
        "StepMAP_TCN_Undeclared",
        "CausalStepMAP_TCN_Declared",
        "CausalStepMAP_TCN_Undeclared",
        "PersistenceEstimator",
        "CompanionRegressionEstimator",
    ]

    for cfg in required_configs:
        assert cfg in estimators, f"Missing required configuration '{cfg}' in estimators summary"
        est_data = estimators[cfg]
        for field in [
            "median_delta_nrmse",
            "mean_delta_nrmse",
            "median_support_f1",
            "mean_support_f1",
            "median_signed_amplitude_error_train_std",
            "sign_accuracy_ge_1std",
            "mean_coverage_90",
            "mean_interval_sharpness",
            "hard_constraint_violations",
        ]:
            assert field in est_data, f"Estimator '{cfg}' missing required field '{field}'"
            assert isinstance(est_data[field], (int, float)), (
                f"Field '{field}' in '{cfg}' must be numeric, got {type(est_data[field])}"
            )

    # Verify CSV scenario coverage
    assert SUMMARY_CSV_PATH.exists()
    df = pd.read_csv(SUMMARY_CSV_PATH)
    for cfg in required_configs:
        cfg_rows = df[df["estimator"] == cfg]
        assert len(cfg_rows) == 120, (
            f"Expected 120 scenario records for '{cfg}', found {len(cfg_rows)}"
        )


# =====================================================================
# Test 3: Precommitted Gate Invariants
# =====================================================================

def test_precommitted_gate_invariants() -> None:
    """Verify precommitted Phase 3 acceptance gates across 120 development scenarios."""
    assert SUMMARY_JSON_PATH.exists(), f"Pre-seal audit summary missing at {SUMMARY_JSON_PATH}"
    data = json.loads(SUMMARY_JSON_PATH.read_text(encoding="utf-8"))
    estimators = data["estimators"]

    p_nrmse = estimators["PersistenceEstimator"]["median_delta_nrmse"]
    assert p_nrmse > 0.0

    # Gate 1: Causal baseline improvement >= 20.0% median Delta NRMSE
    for causal_cfg in ["CausalStepMAP_TCN_Declared", "CausalStepMAP_TCN_Undeclared"]:
        causal_nrmse = estimators[causal_cfg]["median_delta_nrmse"]
        improvement = (p_nrmse - causal_nrmse) / p_nrmse
        assert improvement >= 0.20, (
            f"{causal_cfg} improvement {improvement * 100:.2f}% < 20.0% acceptance gate "
            f"(causal={causal_nrmse}, persistence={p_nrmse})"
        )

    # Evaluate key Step-MAP configurations against precommitted Phase 3 gates
    eval_configs = [
        "StepMAP_TCN_Declared",
        "StepMAP_TCN_Undeclared",
        "CausalStepMAP_TCN_Declared",
        "CausalStepMAP_TCN_Undeclared",
    ]

    for cfg in eval_configs:
        est = estimators[cfg]

        # Gate 2: Median signed amplitude error <= 0.25 sigma_train
        amp_err = est["median_signed_amplitude_error_train_std"]
        assert amp_err <= 0.25, (
            f"{cfg} signed amplitude error {amp_err:.4f} > 0.25 sigma_train acceptance gate"
        )

        # Gate 3: Sign accuracy on attacks >= 1.0 sigma is >= 95.0%
        sign_acc = est["sign_accuracy_ge_1std"]
        assert sign_acc >= 0.95, (
            f"{cfg} sign accuracy {sign_acc * 100:.2f}% < 95.0% acceptance gate"
        )

        # Gate 4: Empirical 90% coverage >= 85.0%
        cov = est["mean_coverage_90"]
        assert cov >= 0.85, (
            f"{cfg} empirical coverage {cov * 100:.2f}% < 85.0% acceptance gate"
        )

        # Gate 5: Sharpness < 0.25 m
        sharp = est["mean_interval_sharpness"]
        assert sharp < 0.25, (
            f"{cfg} sharpness {sharp:.4f} m >= 0.25 m acceptance gate"
        )

        # Gate 6: Zero hard box-constraint violations
        hard_viol = est["hard_constraint_violations"]
        assert hard_viol == 0, (
            f"{cfg} produced {hard_viol} hard box-constraint violations (must be 0)"
        )
