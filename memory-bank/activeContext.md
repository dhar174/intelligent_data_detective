<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Active Context

## Current objective
The repo-agent-bootstrap skill has been run. The agent stack is freshly scaffolded and hand-tuned for IDD. Keep managed guidance aligned with the notebook codebase as it evolves.

## Why it matters now
All 27 cells, agent conventions (BaseNoExtrasModel, State reducers, @handle_tool_errors), and the supervisor-worker topology are now documented across AGENTS.md, memory-bank, and path instructions. Future edits to the notebook must keep these artefacts in sync.

## Current status
- Done:
  - Scaffolded 26 files with `repo-agent-bootstrap` (2026-04-20).
  - Hand-authored `.github/copilot-instructions.md` preserved (lines 1-93); managed block appended.
  - Deleted false-positive `frontend-experience-specialist.agent.md`.
  - Created `notebook-specialist.agent.md` with full IDD editing rules.
  - Customised `repo-planner.agent.md` for IDD orchestration focus.
  - Fixed `backend-python-specialist.agent.md` paths (no `src/` in this repo).
  - Fixed `backend.instructions.md` path references.
  - Created `.github/instructions/notebook.instructions.md`.
  - Enriched AGENTS.md, docs/architecture.md, and all memory-bank files.
- In progress:
  - Validator run pending.
- Blocked:
  - none

## Relevant files
- `IntelligentDataDetective_beta_v5.ipynb`
- `.github/copilot-instructions.md`
- `AGENTS.md`
- `memory-bank/systemPatterns.md`
- `idd_v4_state_graph.mmd`

## Next recommended steps
1. Run `python "${HOME}\.agents\skills\repo-agent-bootstrap\scripts\validate_agent_stack.py" --repo-root .`
2. Run full test suite: `python3 -m pytest -v`
3. If notebook cells are changed, re-verify the 27-cell count.
<!-- repo-agent-bootstrap:managed:end -->
