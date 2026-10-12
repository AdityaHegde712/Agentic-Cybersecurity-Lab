# Narrow Additive Inverse Sprint Execution Report

## Sprint State

- **Started:** 2026-10-11 America/Los_Angeles
- **Branch:** `feature/narrow-additive-inverse-research`
- **Active phase:** Phase 0 - local-first research harness
- **Overall status:** active; checkpoint-independent preparation is proceeding
- **Compute used this sprint:** local CPU only, 0 TACC SUs

## Evidence Boundary

This report distinguishes proposed work from locally reproduced evidence and
remote HPC evidence. No `ssh`, `idev`, `sbatch`, `scancel`, remote repository
synchronization, or allocation-consuming command has been run in this sprint.

## Progress Log

### 2026-10-11 - Sprint Started

- [x] Confirmed the active branch is
  `feature/narrow-additive-inverse-research` at `d71ece9`.
- [x] Preserved unrelated existing worktree changes and untracked paths.
- [x] Confirmed the selected BATADAL checkpoint remains absent locally.
- [x] Activated Phase 0 without treating the checkpoint as a blocker for Tasks
  3-6.
- [x] Added the locked channel/constraint manifest specification before
  implementation.
- [x] Reproduced the expected red-stage failure: `ModuleNotFoundError` for the
  intentionally absent `src.inverse` package.
- [x] Implemented the immutable typed schema with deterministic JSON round-trip
  and strict trust, provenance, source-block, and topology guards.
- [x] Passed 5 focused manifest tests and all 101 immutable specification tests.
- [x] Retrieved and quote-validated current public C-Town/EPANET unit evidence;
  adversarial review narrowed it to candidate semantics only.
- [x] Profiled the 5,840-row model-fit block locally: 43 finite channels, binary
  observed `S_*` values, and seven constant channels.
- [x] Added the locked complete 43-channel BATADAL builder contract.
- [x] Reproduced its expected red-stage failure at the absent builder module.
- [x] Implemented the complete builder and passed its first 9 focused tests.
- [x] Added and reproduced a final semantic red gate with four isolated failures:
  stored units, canonical public IDs, scoped provenance, and hardening guards.
- [x] Corrected all four production semantics and passed 13 focused tests plus
  all 109 immutable specification tests.
- [x] Added the locked deterministic versioned-artifact writer contract.
- [x] Reproduced its expected red-stage failure at the absent script module.
- [x] Implemented and passed the 3-test artifact-writer contract.
- [x] Generated `configs/manifests/batadal-channel-manifest-v1.json` from exactly
  the first 5,840 model-fit rows.
- [x] Verified artifact SHA-256
  `e20b7728cbf4d22ea6fd6dbd4f5eca0b26f65a9165108fa8f8d8d890ad0bd347`,
  deterministic round-trip, 43-channel coverage, and UTF-8/LF formatting.
- [x] Passed all 112 immutable specification tests after artifact generation.
- [/] Began Task 4: identifiability and measurement-noise contract.

## Experiment and Verification Ledger

| Date | Scope | Environment | Result | Evidence |
| --- | --- | --- | --- | --- |
| 2026-10-11 | Sprint-start Git and task-state inspection | Local Windows | Active feature branch confirmed; checkpoint blocker isolated | Workspace-confirmed |
| 2026-10-11 | Manifest contract red stage | Local CPU, focused pytest | Expected collection failure: `src.inverse` absent | Locally reproduced |
| 2026-10-11 | Manifest schema green gate | Local CPU, focused and full pytest | 5 focused and 101 full specification tests passed | Locally reproduced |
| 2026-10-11 | Model-fit constraint profile | Local CPU, `sengupta_research` Conda environment | 5,840 rows; 43 finite channels; status and constant-channel profile recorded | Locally reproduced |
| 2026-10-11 | C-Town/EPANET semantics review | Public primary sources plus adversarial review | Units/topology restricted to candidate, soft-only evidence | Retrieved and reviewed |
| 2026-10-11 | Complete BATADAL manifest red stage | Local CPU, focused pytest | Expected collection failure: builder module absent | Locally reproduced |
| 2026-10-11 | BATADAL builder first green gate | Local CPU, focused pytest | 9 focused manifest tests passed | Locally reproduced |
| 2026-10-11 | Manifest semantic red gate | Local CPU, focused pytest | 4 expected semantic/provenance assertions failed | Locally reproduced |
| 2026-10-11 | Complete manifest schema/builder green gate | Local CPU, focused and full pytest | 13 focused and 109 full specification tests passed | Locally reproduced |
| 2026-10-11 | Manifest artifact writer red stage | Local CPU, focused pytest | Expected collection failure: writer script absent | Locally reproduced |
| 2026-10-11 | Versioned manifest artifact | Local CPU, existing Conda environment | 43-channel artifact generated from 5,840 rows; SHA-256 verified | Locally reproduced |
| 2026-10-11 | Task 3 completion gate | Local CPU, full pytest | 112 immutable specification tests passed | Locally reproduced |

## Git Ledger

| Date | Commit | Push | Scope |
| --- | --- | --- | --- |
| 2026-10-11 | `a01d81d` | Confirmed on `origin/feature/narrow-additive-inverse-research` | Approved trajectory and completed Task 3 manifest milestone |

## Compute Ledger

| Date | System | Queue / Node | Walltime | Estimated SUs | Actual status |
| --- | --- | --- | --- | --- | --- |
| 2026-10-11 | Local workstation | CPU | Less than 1 minute | 0 | State inspection only |
| 2026-10-11 | Local workstation | CPU | Less than 1 minute | 0 | Model-fit profile and manifest generation complete |

## Active Blockers

1. The exact selected BATADAL checkpoint is missing, so Task 2 and all
   checkpoint-dependent experiments remain paused pending restoration or exact
   regeneration with provenance and SHA-256.
2. Public C-Town topology/physics cannot become a hard constraint until its
   alignment to the processed BATADAL channels is validated.

## Immediate Next Action

Write the identifiability and measurement-noise contract with an executable
same-observation ambiguity example.
