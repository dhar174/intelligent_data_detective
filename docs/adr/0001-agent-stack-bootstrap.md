<!-- repo-agent-bootstrap:file-kind=adr -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# ADR 0001: Bootstrap a hybrid agent stack

## Status
Accepted

## Context
intelligent_data_detective needs repository guidance that works across GitHub Copilot, Codex, and Claude without overloading every task with the same context.

## Decision
Adopt a layered stack with root guidance (`AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`), path-specific instructions, repository custom agents, lightweight repo skills, and `memory-bank/` state files. Maintenance runs update only managed sections.

## Consequences
### Positive
- Agents can recover repo context quickly and delegate to specialists.
- Repo-wide guidance stays concise while deeper knowledge moves into on-demand files.

### Negative
- The stack introduces more files that need occasional maintenance.
- Generated sections require clear markers so human edits remain safe.
<!-- repo-agent-bootstrap:managed:end -->
