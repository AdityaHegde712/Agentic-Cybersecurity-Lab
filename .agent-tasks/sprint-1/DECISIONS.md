# Sprint 1 Decisions: Accepted

## S1-D01: Observation-Only Attack Benchmark

**Decision:** Model additive observation corruption, not actuator/process attacks. Use synthetic injection for delta recovery and real labels for detector evidence.

**Consequence:** Do not report a real-dataset residual as recovered ground-truth attack magnitude.

## S1-D02: Residuals Before Raw Levels

**Decision:** Retain raw-level scores only as a negative-control baseline. Treat temporal residuals as the main statistical measurement layer.

**Evidence:** Raw EWMA/PCA/SPE produced approximately 99-100% FPR on HAI/SWaT; residual scores produced bounded FPR with useful HAI and BATADAL event evidence.

## S1-D03: Normal-Only Forecaster

**Decision:** Train the LSTM on known-normal rows; persist architecture, weights, and train-derived normalization in `best.pt`; do not refit normalization at prediction time.

**Consequence:** Forecast residuals are candidate delta estimates only for injected benchmarks.

## S1-D04: Artifact-First Inspection

**Decision:** Large score outputs are read only by repository scripts. Agents inspect aggregate Markdown reports and rendered PNGs.

**Consequence:** Never paste or manually open high-volume result CSVs in conversational context.
