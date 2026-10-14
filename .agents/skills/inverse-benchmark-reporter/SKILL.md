---
name: inverse-benchmark-reporter
description: Standardized framework and template for reporting empirical inverse modeling benchmark results in time-series additive attack recovery (y = x + delta). Use whenever synthesizing, presenting, or auditing empirical results from inverse estimators, mathematical baselines, and surrogate dynamics bake-offs.
---

# Inverse Benchmark Reporter Skill

Use this skill to produce consistent, scientifically rigorous, accomplishment-oriented reports for inverse modeling of sensor attacks ($y = x + \delta$).

## Reporting Principles

1. **Focus on Mechanism & Solution**: Highlight *how* the method solves or bounds the inverse problem, rather than exploring abstract non-identifiability without empirical results.
2. **Deterministic & Programmatic**: Ground claims in exact mathematical formulations (loss objectives, regularizers, physical constraints) and deterministic evaluation scripts.
3. **Structured Template**: Follow the structure in [references/TEMPLATE.md](references/TEMPLATE.md) for every empirical benchmark deliverable.
4. **Aggregate & Visual Only**: Do not dump high-volume raw outputs into reports. Reference summary metrics (`summary.csv`, `summary.json`) and rendered static plots under `results/`.
