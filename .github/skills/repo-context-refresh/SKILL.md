---
name: repo-context-refresh
description: 'Refresh repository context files, memory-bank state, and root guidance after architecture, workflow, or milestone changes.'
---

# Repo Context Refresh

Use this skill when a task changes project direction, subsystem boundaries, or contributor expectations and the repo's agent-facing docs need to catch up.

## Workflow
1. Read `AGENTS.md`, `docs/architecture.md`, and `memory-bank/activeContext.md`.
2. Inspect the touched source paths and validation changes.
3. Update only the docs that changed meaningfully: `memory-bank/`, `plans/`, `AGENTS.md`, `CLAUDE.md`, or path-specific instructions.
4. Keep repo-wide instructions tight; move deep detail into on-demand docs.
