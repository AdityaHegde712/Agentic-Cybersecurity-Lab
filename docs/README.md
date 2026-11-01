# Documentation Index

This index separates current repository guidance, active research state, operations, and historical artifacts. Rendered PDFs are local-only; Markdown is the canonical editable source.

## Authority map

| Need | Authoritative location |
| :--- | :--- |
| Public project orientation and setup | [`../README.md`](../README.md) |
| Code architecture and navigation | [`../CODEBASE.md`](../CODEBASE.md) |
| Current branch and continuation state | [`../HANDOFF.md`](../HANDOFF.md) |
| Active inverse-recovery research plan | [`../.agent-tasks/narrow-inverse-research/`](../.agent-tasks/narrow-inverse-research/) |
| Long-horizon research trajectory | [`../.agent-tasks/TRAJECTORY.md`](../.agent-tasks/TRAJECTORY.md) |
| HPC safety rules | [`../.agents/skills/tacc-vista-hpc/SKILL.md`](../.agents/skills/tacc-vista-hpc/SKILL.md) |

## Architecture

- [`architecture/system-design.md`](architecture/system-design.md) — research system boundary and attack/detection design.
- [`architecture/vista-shared-terminal-mcp.md`](architecture/vista-shared-terminal-mcp.md) — separate Vista shared-terminal infrastructure design.

## Research

- [`research/findings.md`](research/findings.md) — earlier evidence and findings log; verify scope against the active inverse-recovery ledger.
- [`research/progress.md`](research/progress.md) — milestone-oriented progress report.
- [`research/datasets.md`](research/datasets.md) — dataset access, provenance, and consolidation notes.

## Operations

- [`operations/hpc-usage-guide.md`](operations/hpc-usage-guide.md) — comprehensive Vista operational guide.
- [`operations/hpc-quick-reference.md`](operations/hpc-quick-reference.md) — short command reference.
- [`operations/hpc-policy-reference-links.md`](operations/hpc-policy-reference-links.md) — official policy and reference links.

## Reports and archive

- [`archive/reports/narrative-report.md`](archive/reports/narrative-report.md) — historical narrative report; not the current research authority.
- [`archive/`](archive/) — superseded early research and proposal artifacts.
- [`../docs/DOCUMENTATION_REORGANIZATION_PROPOSAL.md`](DOCUMENTATION_REORGANIZATION_PROPOSAL.md) — migration rationale and remaining cleanup candidates.

## Maintenance rules

1. Keep root-level documents limited to entry points, codebase orientation, agent rules, and temporary handoff state.
2. Keep active execution state in `.agent-tasks/`, not in `docs/`.
3. Update this index when a document is moved or its authority changes.
4. Do not treat archived findings or historical reports as current benchmark evidence.
