<!-- repo-agent-bootstrap:file-kind=runbook -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Runbook

## Operating mode
1. Read `AGENTS.md`, `memory-bank/activeContext.md`, and the nearest path-specific instructions first.
2. Inventory commands, workflows, AI assets, and risky boundaries before generating new agent files.
3. Prefer original repo-specific content; vendor third-party assets only from pinned revisions with provenance and license notes.

## Maintenance workflow
1. Detect existing managed sections and compare them to freshly rendered output.
2. Replace only managed blocks; preserve everything else.
3. Re-run stack validation and surface any unresolved drift explicitly.

## Verification checklist
- The agent roster still matches the repository's major workflows.
- Commands in docs match current manifests or README guidance.
- Memory-bank files reflect current priorities rather than stale templates.
- Imported external assets remain pinned and attributable.
<!-- repo-agent-bootstrap:managed:end -->
