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
- [x] Drafted the mathematical identifiability/noise contract with a
  BATADAL-scaled same-observation ambiguity witness.
- [x] Added the locked Task 4 observation/ambiguity specification.
- [x] Reproduced its expected red-stage failure at the absent production module.
- [x] Implemented the first typed observation and ambiguity API; its seven
  original focused tests pass.
- [x] Completed adversarial mathematical review; it rejected six underspecified
  claim, certificate, witness, leakage, and trust rules.
- [x] Corrected the written contract without promoting any soft BATADAL rule to
  hard evidence.
- [x] Added four immutable adversarial tests and reproduced their expected red
  failures for missing exact-addition and certified-diameter semantics.
- [x] Implemented the sealed ambiguity tolerance, norm, and certificate status;
  all 11 focused tests pass.
- [x] Passed all 123 immutable specification tests after hardening.
- [x] Reproduced the exact toy witness locally: `linf` separation `0.5`, maximum
  clean step `0.5`, `ambiguous=true`, and `certified=false`.
- [x] Second adversarial review rejected caller-controlled global certification
  and loose witness constants.
- [x] Tightened the written certificate contract and added immutable hard-set
  certificate plus exact witness-constant assertions.
- [x] Reproduced the final expected Red-stage import failure for the absent
  `HardSetCertificate`.
- [x] Implemented the first evidence-reference certificate and structural status;
  20 focused and 132 full specification tests pass.
- [x] Third adversarial review found the remaining false-uniqueness path: a
  metadata-only SHA-shaped identifier is not resolved evidence and carries no
  certified global diameter.
- [x] Aligned the documented legacy scope literal with implementation.
- [x] Added a true exact-constant witness regression; 21 focused tests pass.
- [x] Owner authorized correction of the flawed metadata-only certificate test.
- [x] Corrected only the owner-authorized certificate assertions; the Red stage
  produced 11 expected failures for missing evidence/bound fields.
- [x] Implemented canonical evidence JSON, SHA-256 verification, exact payload
  matching, certified global-diameter semantics, and contradiction rejection.
- [x] Passed 26 focused and 138 full immutable specification tests.
- [x] Final review found cross-observation and cross-norm certificate reuse was
  still possible because the certificate omitted exact benchmark bindings.
- [x] Added owner-authorized tests for observation fingerprint, benchmark case,
  delta norm, and canonical hard-set manifest bindings; reproduced the expected
  missing-fingerprint Red failure.
- [x] Implemented exact observation, benchmark-case, norm, hard-set-manifest, and
  evidence-manifest certificate bindings.
- [x] Passed 32 focused and 144 full immutable specification tests.
- [x] Final review required the hard-set manifest to seal the actual support,
  identity tolerance, attack class, dynamics artifact, and typed constraint
  parameters rather than IDs alone.
- [x] Reproduced 8 expected old-schema Red failures, implemented the complete
  schema, and passed 36 focused plus 148 full immutable specification tests.
- [x] Downgraded unverified supplied-candidate separation to a diagnostic while
  keeping structural status `not_certified`; 37 focused and 149 full tests pass.
- [x] Final adversarial review accepted the Task 4 mathematical claim boundary.
- [/] Task 5 chronological benchmark manifest and seal are active.

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
| 2026-10-11 | Task 4 observation/ambiguity red stage | Local CPU, focused pytest | Expected collection failure: identifiability module absent | Locally reproduced |
| 2026-10-11 | Task 4 adversarial semantic red stage | Local CPU, focused pytest | 7 passed; 4 expected failures for missing certificate API | Locally reproduced |
| 2026-10-11 | Task 4 hardened green gate | Local CPU, focused and full pytest | 11 focused and 123 full specification tests passed | Locally reproduced |
| 2026-10-11 | Task 4 exact ambiguity witness | Local CPU, direct execution | `linf` separation 0.5; ambiguous and not uniqueness-certified | Locally reproduced |
| 2026-10-11 | Task 4 hard-set certificate red stage | Local CPU, focused pytest | Expected import failure: `HardSetCertificate` absent | Locally reproduced |
| 2026-10-11 | Task 4 first certificate green gate | Local CPU, focused and full pytest | 20 focused and 132 full tests passed; adversary still rejected global-certificate semantics | Locally reproduced, not accepted |
| 2026-10-11 | Task 4 resolved-certificate red stage | Local CPU, focused pytest | 15 passed; 11 expected failures for missing resolved evidence/bound API | Locally reproduced |
| 2026-10-11 | Task 4 resolved-certificate green gate | Local CPU, focused and full pytest | 26 focused and 138 full specification tests passed | Locally reproduced |
| 2026-10-11 | Task 4 exact-binding red stage | Local CPU, focused pytest | Expected import failure: `fingerprint_observation` absent | Locally reproduced |
| 2026-10-11 | Task 4 exact-binding green gate | Local CPU, focused and full pytest | 32 focused and 144 full specification tests passed | Locally reproduced |
| 2026-10-11 | Task 4 complete-hard-set red stage | Local CPU, focused pytest | 28 passed; 8 expected old-schema failures | Locally reproduced |
| 2026-10-11 | Task 4 complete-hard-set green gate | Local CPU, focused and full pytest | 36 focused and 148 full specification tests passed | Locally reproduced |
| 2026-10-11 | Task 4 accepted completion gate | Local CPU, focused and full pytest plus adversarial review | 37 focused and 149 full tests passed; claim boundary accepted | Locally reproduced and reviewed |

## Git Ledger

| Date | Commit | Push | Scope |
| --- | --- | --- | --- |
| 2026-10-11 | `a01d81d` | Confirmed on `origin/feature/narrow-additive-inverse-research` | Approved trajectory and completed Task 3 manifest milestone |
| 2026-10-11 | `aee210d` | Pending | Accepted Task 4 identifiability and noise contract |

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

Write locked chronological-split and deterministic benchmark-seal tests before
implementing the Task 5 manifest.
