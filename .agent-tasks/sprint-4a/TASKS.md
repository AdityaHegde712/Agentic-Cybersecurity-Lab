# Sprint 4A Tasks: Statistical Distribution & Error Modeling

- [x] 1. Author locked specification test contract `tests/spec/test_delta_distribution_analysis.py`.
  - **Acceptance:** Verify parametric fit schemas, MLE convergence contracts for Gaussian, Student-$t$, and GED (`scipy.stats.gennorm`), KS test calculation, null distribution evaluation schema, and operational threshold derivations.
  - **Status:** Complete. Red phase verified; all 6 contract tests pass.

- [x] 2. Implement statistical modeling module `src/inverse/statistical_modeling.py`.
  - **Acceptance:** Clean, type-annotated functions for MLE parametric fitting, AIC/BIC computation, KS goodness-of-fit testing, null clean false-alarm rate profiling, and operational ROC/threshold computation.
  - **Status:** Complete. Green phase verified.

- [x] 3. Implement CLI runner `scripts/analyze_delta_reconstruction_distributions.py`.
  - **Acceptance:** Ingests `results/narrow_inverse_research/sealed_final_benchmark/summary.csv`, evaluates null clean telemetry on unattacked test block, fits parametric models, derives threshold tables, and generates 5 publication-ready diagnostic plots in `results/narrow_inverse_research/error_analysis/`.
  - **Status:** Complete. Verified deterministic CLI execution.

- [x] 4. Execute distribution analysis and generate output artifacts.
  - **Acceptance:** `results/narrow_inverse_research/error_analysis/summary.json`, `summary.csv`, and all 5 plots generated; full test suite (277 spec tests) passes 100%.
  - **Status:** Complete. All artifacts verified on disk; full suite passes.

- [x] 5. Author Sprint 4A Error Modeling Technical Report.
  - **Acceptance:** Comprehensive report authored at `results/narrow_inverse_research/error_analysis/ERROR_MODELING_REPORT.md` analyzing fitted distributions, tail risk, and operational detector guidelines.
  - **Status:** Complete. Detailed report published.

- [x] 6. Synchronize Sprint 4A completion into `.agent-tasks/narrow-inverse-research/TASKS.md` and root `HANDOFF.md`.
  - **Acceptance:** Task 11 marked complete with full evidence trail.
  - **Status:** Complete. State files and root HANDOFF.md synchronized.

