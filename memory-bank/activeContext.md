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

<!-- session-curated:start -->
## Session context — pipeline debugging (mid-session checkpoint)

### Current objective
Achieve a **production-quality clean run** of the full IDD v5 pipeline — zero recovery hops, all artifacts emitted, suitable for portfolio inclusion.

### High-water mark
- **Run 76 in flight.** `visualization` node passed cleanly. `viz_evaluator` has been silent ≥16 min — actively investigating whether it is looping or stuck on an LLM call.
- Earlier baselines: Run 73 finished `viz=True/report=True` but every agent hit `recursion_limit=160` (recovery path); Run 74 regressed on BR-7 (managed channels); Run 75 regressed on BR-8 (`InvalidUpdateError` on `structured_response` at cell 81).

### Open work
- **Wave 4 patches ready to apply** (already prepared as `_patch_notebook.py` sentinels):
  - `W2-EMERGENCY` — give `EMERGENCY_MSG` an outgoing edge (currently a dead-end).
  - `W2-BF2` — analyst fix.
  - `W2-BF6` — agent factory fix.
  - `W2-REC6` — fast-fail on unknown tool name.
- **Run 77** will use `python run_notebook_live.py --resume` (saves 12–15 min by skipping checkpointed nodes) **only if State schema is unchanged**; any new field/reducer invalidates `checkpoints.sqlite`.

### Active design decisions (do not regress)
- `recursion_limit = 160` (inner agents) / `400` (outer graph) — explicitly raised by user.
- `State.messages` is declared **directly** as `Annotated[list[AnyMessage], add_messages]`, **not** inherited from `langchain.agents.AgentState` / `langgraph.prebuilt.chat_agent_executor.AgentState` (BR-7 — those bases drag in the `remaining_steps` managed channel that LangGraph rejects in InputSchema).
- `structured_response: Annotated[Optional[BaseNoExtrasModel], _sr_reducer]` — the `_sr_reducer` is `lambda left, right: left if right is None and left is not None else right` (prefer-non-None last write). Required because main node + recovery shims both write concurrently and the default LastValue reducer raised `InvalidUpdateError` (BR-8).
<!-- session-curated:end -->

<!-- session-curated:2026-04-23:start -->
## 2026-04-23 — W9-SR-DROP unblocks supervisor graph; focus shifts to W10-PDF-DIAG

### Current focus
**W10-PDF-DIAG** — PDF artifact emission is the sole remaining cleanliness blocker after W9-SR-DROP. The supervisor graph now reaches FINAL with `viz=True report=True` cleanly (Run 87, ~21 min wall, 7/8 criteria GREEN).

### What changed (cell 22 of patched notebook)
- ✅ **W7-SR-ALIGN** — User `State.structured_response` annotation aligned exactly to langchain `AgentState`'s shape: `NotRequired[Annotated[Optional[Any], OmitFromSchema(input=True, output=False)]]`. Eliminated `_resolve_schema` set-iteration nondeterminism that intermittently failed cell 48 (`create_agent` calls) by colliding `_sr_reducer` BinaryOperatorAggregate with AgentState's LastValue channel for the same field.
- 🟡 **W8-VW-NOSR** (cell 57) — Removed `"structured_response": sr` from viz_worker wrapper return dict. Necessary but insufficient on its own (recovery shims also wrote sr).
- ✅ **W9-SR-DROP** (cell 22, breakthrough) — Completely removed `structured_response` from supervisor State. Each `create_agent` subgraph still owns its own sr via internal `AgentState[ResponseT]`; wrapper code reads it from `result["structured_response"]` (Python-dict from agent.invoke, NOT a langgraph supervisor channel). Static grep confirmed zero supervisor-state reads of `state["structured_response"]`. Without a supervisor channel, parallel writers (viz fan-out, recovery shims, agent subgraph propagation) silently drop instead of colliding at LastValue's "one write per step" contract.

### Run 87 result (post-W9)
7/8 GREEN. FINAL `viz=True report=True`. 0 recoveries, 0 W2-BA-finalhop, 0 tracebacks, 0 W4 negatives. All 11 graph stages traversed (initial_analysis → data_cleaner → analyst → visualization + viz_worker fan-out + viz_join → viz_evaluator → report_orchestrator + section_worker + report_join → report_packager → file_writer → FINAL). Native structured Pydantic outputs throughout (`CleaningMetadata`, `AnalysisInsights`, `ReportResults`, etc.). Wall ~21 min.

### Remaining gap — W10-PDF-DIAG
PDF artifact missing from `IDD_results/IDD_run_run_default_id-20260423-0951-6c29a6f7/`. HTML (244 KB) + MD written by file_writer LLM agent tools; PDF the lone gap. `W6-FW-PDF-FORCE` block IS present in cell 57 `file_writer_node` but **never executed at runtime** (zero `[W6-FW-PDF-FORCE]` log lines). `STAGE file_writer DONE` marker never logged — silent failure between `file_writer_agent.invoke()` return and the W6 try-block. Suspect `result["structured_response"]` KeyError or `assert isinstance(file_results, ListOfFiles)` failing because W9-SR-DROP causes the agent-subgraph→supervisor-state merge to drop sr.

### Recommended W10 patches
1. **W10-FW-RESULT-GUARD** — defensively handle missing key: `file_results = result.get("structured_response") or result.get("output") or result`; rebuild from tool calls if not `ListOfFiles`.
2. **W10b-FW-LOG** — add `print('STAGE file_writer post-invoke ok')` immediately after invoke to localize the silence.
3. **W10-PDF-DIRECT (alt)** — separate post-graph PDF-writer node that reads HTML from disk and renders via xhtml2pdf, independent of file_writer wrapper.

### Outstanding (non-blocking)
- `IDD_results/executed_20260423_055108.ipynb` is 0 bytes (runner `nbformat.write` flush bug → W11-NB-FLUSH).
- `outputs/` clutter (~50 redundant tool-marker files) → optional W12-OUTPUT-DEDUPE.
- `_diag_cmp.py`, `_diag2.py` left in repo root from earlier diag work.

### Hard constraints honored
- Did NOT modify `IntelligentDataDetective_beta_v5.ipynb` or `_patch_notebook.py` this session segment.
<!-- session-curated:2026-04-23:end -->

<!-- session-curated:2026-04-23-phase6:start -->
## 2026-04-23 (later) — Phase 6: structural CLEAN ✅, deliverable HOLLOW ❌ — pivot to content quality

### Current focus
**Phase 6 — content-quality / Potemkin-pipeline forensics.** The Wave 4/5 structural-cleanliness goal is *done* (Run 87 7/8, Run 88 8/8). The new — and now sole — goal is **producing a deliverable that a human would actually read**. The "8/8 CLEAN" success criterion is **OBSOLETE** and has been replaced by the new 12-criteria content bar (see decisionLog).

### Run 88 result — structurally CLEAN, semantically HOLLOW
8/8 GREEN structural pass. But the artifacts are a **Potemkin pipeline**:
- **Report text:** literally 356 chars, verbatim placeholder: `"Final report placeholder created. HTML, Markdown, and PDF artifacts prepared per instruction."`
- **Sections in report:** 0 (`Final report assembled from 0 sections`).
- **Visualizations:** 1 PNG (analyst recommended 3); embedded 5× in HTML with cosmetic hash variations.
- **PDF:** 1983 bytes — title page only.
- **Stub files in reports dir:** 25+ marker files (`*_ack`, `*_commit`, `*_ready`, `*_stub`, `*_trigger`, …) — `file_writer` LLM compensated for empty `written_sections` by spamming marker tools to "look busy".
- **Analyst output was rich and correct** (full `AnalysisInsights` with 3 distinct `VizSpec`s, correlation_insights, anomaly_insights). **The content was lost between analyst and report.**

### W7–W10 patches confirmed working (DO NOT regress)
- **W7-SR-ALIGN** (cell 22): user `State.structured_response: NotRequired[Annotated[Optional[Any], OmitFromSchema(input=True, output=False)]]`.
- **W8-VW-NOSR** (cell 57): viz_worker wrapper return no longer carries `structured_response`.
- **W9-SR-DROP** (cell 22): `structured_response` removed entirely from supervisor `State`.
- **W10-PDF-POST** (cell 83): post-graph deterministic xhtml2pdf converter; produces a valid PDF independent of `file_writer`. (Note: with current hollow HTML, the PDF is also hollow → still ≪ 30 KB target.)

### Wave 5 mode is ACTIVE
Fixes now go **directly into the source notebook** `IntelligentDataDetective_beta_v5.ipynb`. `_patch_notebook.py` becomes a historical changelog only. Backups for the cutover live at `_wave5_backup_20260423-091557/` (preserves SOURCE notebook + patcher + patched notebook).

### MyChatOpenai is LEGACY (per user, 2026-04-23)
Going forward, **use `ChatOpenAI` directly**, not `MyChatOpenai`. ⚠️ This **conflicts with `.github/agents/notebook-specialist.agent.md`**, which (in its managed block) still mandates `MyChatOpenai`. We are **not** editing the agent doc (managed), but the divergence is documented here and in `decisionLog.md`. New cells should use `ChatOpenAI`; existing `MyChatOpenai` call sites can be migrated opportunistically.

### Phase 6 plan (locked)
Source: `C:\Users\darf3\.copilot\session-state\6b170d0a-bbd7-4870-ba34-e2c0cb0d4241\plan.md`.

| Phase | Goal | GH Issue |
|---|---|---|
| **A** | Forensic deep-dive (no code changes) — RC1 viz fan-out drop, RC2 `report_section_worker` writes 0 sections, RC3 file_writer placeholder + stub spam, RC4 supervisor accepts hollow completion, **RC5** structured Pydantic outputs may not actually be persisted to State after W9-SR-DROP | #112 |
| B | Instrument-only telemetry run | #113 |
| C | Fix viz pipeline | #114 |
| D | Fix report pipeline (`Section.body` min-length validator, `written_sections` reducer audit) | #115 |
| E | Fix `file_writer` (rewrite prompt, restrict tool list to remove stub-marker tools, cap tool calls) | #116 |
| F | Tighten supervisor FINAL gate with content-validation preconditions | #117 |
| G | Validation against 12-criteria; expect 5–8 runs to converge | #118 |

**GitHub epic:** `dhar174/intelligent_data_detective#119`. Labels: `epic`, `phase-6`, `forensic`, `instrumentation`, `viz-pipeline`, `report-pipeline`, `file-writer`, `supervisor`, `validation`.

### Forensic agents in flight
- `forensic-pipeline` (explore) — completed; **results stranded** in cleared agent context. Re-harvest required.
- `forensic-sr-persistence` (explore) — completed; **results stranded**. Re-harvest required.
Both produced full forensic reports but the agent contexts cleared before harvest. Phase A continuation must either re-launch them or fall back to direct inspection.

### Hard constraints (Phase 6)
- Do **not** revert W7/W8/W9/W10 patches.
- Do **not** add `structured_response` back to supervisor `State`.
- Do **not** edit managed `.github/agents/*` blocks (e.g., notebook-specialist's MyChatOpenai rule).
- Wave 5 source-of-truth = the notebook itself; do not re-route fixes through `_patch_notebook.py`.

### Relevant files (Phase 6)
- `IntelligentDataDetective_beta_v5.ipynb` (Wave 5 direct-edit target; cells 22, 57, 83 are hot)
- `_wave5_backup_20260423-091557/` (cutover backup)
- `IDD_results/IDD_run_*-20260423-*/` (Run 87, Run 88 artifacts)
- Phase 6 plan: `C:\Users\darf3\.copilot\session-state\6b170d0a-bbd7-4870-ba34-e2c0cb0d4241\plan.md`

### Next recommended steps
1. **Phase A**: re-run forensic exploration of viz fan-out (RC1), report_section_worker writes (RC2), file_writer prompt + tool list (RC3), supervisor FINAL gate (RC4), and structured-output State persistence post-W9 (RC5).
2. **Phase B**: instrument-only telemetry run; capture per-stage counts of `written_sections`, `viz_specs`, `viz_results`.
3. Defer C–G until Phase A confirms each RC and Phase B gives a measured baseline.
<!-- session-curated:2026-04-23-phase6:end -->

