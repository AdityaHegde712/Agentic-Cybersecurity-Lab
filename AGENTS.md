# Agent Directives & Operational Rules

<change_transparency>
### Change Transparency & Review Directive
Following any file creation, modification, or deletion, always append a concise post-action summary detailing what changed, why the change was made, and what it accomplishes. Keep this explanation directly below the completed action to preserve immediate execution while preventing blind operation. For comprehensive structured reporting, invoke the `report-changelog` skill.
</change_transparency>

### Large Result Artifacts

- Never manually open, paste, or load high-volume experiment-result files into agent context.
- Create or reuse repository-relative analysis scripts that read result files, emit compact aggregate reports, and save static plots under the corresponding `results/` directory.
- Inspect rendered plot images to assess behavior; report conclusions only at the aggregate/visual level.
- Treat raw result files as artifacts for scripts, not conversational context.
