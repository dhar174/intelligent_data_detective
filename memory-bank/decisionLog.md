<!-- repo-agent-bootstrap:file-kind=memory-bank -->
# Decision Log

Architectural decisions and root-cause analyses for the IDD v5 pipeline. Append-only; cite sentinel IDs and run numbers.

---

## 2026-04-23 — ADR: Drop `structured_response` from supervisor State (W9-SR-DROP)

### Status
✅ Accepted and verified (Run 87).

### Context
Across Runs 75–86 the supervisor graph repeatedly failed with one of two errors that traced back to the same field, `State.structured_response`:

1. **Pre-W7 era (Runs 83–84):** `ValueError` at notebook cell 48 (the `create_agent(...)` calls) — only on some runs, others (Runs 81–82) passed unchanged.
2. **Post-W7 era (Runs 85–86):** `InvalidUpdateError: At key 'structured_response': Can receive only one value per step` during the `viz_worker` fan-out superstep.

### Root cause
`langchain.agents.factory._resolve_schemas` constructs the merged StateSchema/InputSchema/OutputSchema for `create_agent` by iterating `set({State, AgentState[ResponseT]})`. Python `set` iteration order is governed by PYTHONHASHSEED and therefore **nondeterministic across processes**. For any field name appearing on both schemas — `structured_response`, `messages`, `remaining_steps` — the "winning" channel definition depends on which class iterates last.

For `structured_response` specifically:
- Supervisor `State` declared `Annotated[Optional[BaseNoExtrasModel], _sr_reducer]` → langgraph `BinaryOperatorAggregate` channel.
- `AgentState[ResponseT]` declares `Annotated[ResponseT, OmitFromSchema(input=True, output=False)]` → langgraph `LastValue` channel.

`langgraph.channels.last_value._add_schema` (state.py:14–19) only tolerates a type mismatch between two schemas if the **second** channel is `LastValue`. With nondeterministic iteration, ~half of process starts produce StateSchema=LastValue + InputSchema=BinaryOperatorAggregate → `ValueError` at create_agent time. The other half pass (Runs 81–82 were lucky hash-seed rolls).

W7-SR-ALIGN fixed the cell-48 nondeterminism by aligning the user annotation to AgentState's exactly (both LastValue + same `OmitFromSchema` metadata → set merges to a single channel regardless of order). But this collapsed the supervisor sr channel to `LastValue`, so concurrent writes from (a) viz_worker wrapper, (b) W2-BA-finalhop recovery shim, (c) agent-subgraph propagation all collided in the same superstep → `InvalidUpdateError`.

W8-VW-NOSR removed writer (a) but writers (b) and (c) still triggered the error. Renaming-per-writer (Path A) was rejected as too invasive.

### Decision
**Path B — remove `structured_response` from the supervisor `State` TypedDict entirely.** Each `create_agent` subgraph still owns its own `structured_response` via internal `AgentState[ResponseT]`. Wrapper nodes read it from the Python-dict return of `agent.invoke()` (e.g., `result["structured_response"]`) — this is dict access, **not** a langgraph channel read. Static grep across the patched notebook confirmed zero supervisor-state reads of `state["structured_response"]` before adopting this decision.

### Consequences
- ✅ **Run 87 verified:** FINAL marker reached with viz=True report=True, 0 `InvalidUpdateError`, 0 recoveries, 0 W2-BA-finalhop hits, 0 W4 negatives, 0 tracebacks. Native structured Pydantic outputs throughout. ~21 min wall.
- ✅ Without a supervisor channel for sr, parallel writers silently drop instead of colliding at LastValue's "one write per step" contract — exactly the desired behaviour.
- ⚠️ Side effect: `file_writer_node`'s wrapper code expects `result["structured_response"]` to be a `ListOfFiles` Pydantic instance after agent.invoke. Post-W9, the agent-subgraph→supervisor-state merge can return a dict missing the key (or with a stale value), causing silent failure in the file_writer post-invoke path. **This is the W10-PDF-DIAG remaining gap** — `STAGE file_writer DONE` marker never logged, W6-FW-PDF-FORCE block unreached.

### Sentinels
- `W7-SR-ALIGN` — `_patch_notebook.py` ~L7668 (cell 22 State annotation aligned to AgentState). Required precondition.
- `W8-VW-NOSR` — `_patch_notebook.py` ~L7720 (cell 57 viz_worker wrapper drops sr from return dict). Necessary but insufficient.
- `W9-SR-DROP` — cell 22 `structured_response` field removed from `State` TypedDict. The breakthrough.

### Pattern (recorded in `systemPatterns.md`)
For `create_agent` compatibility with custom supervisor `State`, do not add custom-reducer fields whose names collide with `AgentState` fields (`structured_response`, `messages`, `remaining_steps`). Either omit the field from `State` (preferred for sr) or mirror AgentState's annotation exactly (only safe under single-writer guarantees).

### References
- `langchain.agents.factory._resolve_schemas` — set-iteration nondeterminism (root cause of cell-48 ValueError).
- `langgraph.channels.last_value._add_schema` (state.py:14–19) — type-mismatch tolerance rule (only when second channel is LastValue).
- Session files: `run85-86-summary.md`, `run87-summary.md`, `run87-postmortem.md`.

### Follow-up
- **W10-PDF-DIAG** (next session): harden `file_writer_node` extraction to defensively handle missing `structured_response` key (W10-FW-RESULT-GUARD), or add a separate post-graph PDF-writer node that reads HTML from disk (W10-PDF-DIRECT).

---

## 2026-04-23 (later) — ADR: Phase 6 pivot — content quality replaces structural cleanliness

### Status
✅ Accepted. Active acceptance bar.

### Context
Run 87 hit 7/8 structural gates and Run 88 hit 8/8 — but Run 88's deliverable is a **Potemkin pipeline**: 356-char placeholder report, 0 sections, 1 viz embedded 5× with cosmetic hash variation, 1983-byte title-only PDF, and 25+ stub-marker files (`*_ack`, `*_commit`, `*_ready`, `*_stub`, `*_trigger`, …) spammed by `file_writer` to compensate for empty `written_sections`. Analyst output was rich and correct; content was lost downstream.

### Decision
**Retire the "8/8 CLEAN" structural success criterion. Adopt a 12-criteria content-quality acceptance bar.** The first six criteria (the rest TBD as Phase G is enumerated):
1. Report HTML text-only ≥ 3000 chars
2. Report has ≥ 4 distinct sections
3. ≥ 3 distinct visualizations (different titles AND different image bytes)
4. PDF size ≥ 30 KB
5. 0 stub/marker files in reports dir
6. Report references actual analyst findings (correlation r values, anomaly notes appear verbatim)

A run that passes 8/8 structural gates but fails any content gate is **not** a passing run.

### Consequences
- All Phase 6 work (issues #112–#118 under epic #119) is judged against this bar.
- Existing W7–W10 patches stay in place; they are necessary but not sufficient.
- The previously-tracked "PDF artifact emission" subgoal collapses into criterion 4 (PDF ≥ 30 KB), which requires real HTML body content first.

---

## 2026-04-23 (later) — ADR: Wave 5 mode — direct-to-notebook edits; patcher → changelog

### Status
✅ Accepted.

### Context
Through Wave 4, all notebook fixes flowed through `_patch_notebook.py` as sentinel-bracketed blocks (~80 sentinels accumulated). With W7/W8/W9/W10 confirmed working and Phase 6 requiring rapid iterative content fixes, the patcher indirection adds latency without buying safety.

### Decision
**Wave 5: edit `IntelligentDataDetective_beta_v5.ipynb` directly.** `_patch_notebook.py` is preserved as a historical changelog only. Pre-cutover backups live at `_wave5_backup_20260423-091557/` (SOURCE notebook + patcher + patched notebook all preserved).

### Consequences
- Faster iteration on Phase 6 fixes.
- The patcher must NOT be re-introduced for new Wave 5 work; if a regression requires re-applying a Wave 4 sentinel, do it as a direct notebook edit and reference the original sentinel ID in the commit / decision log.
- Rollback for any Phase 6 cell change is the `_wave5_backup_20260423-091557/` snapshot.

---

## 2026-04-23 (later) — ADR: `ChatOpenAI` direct over `MyChatOpenai` (legacy)

### Status
✅ Accepted (per user, 2026-04-23). ⚠️ Conflicts with managed agent doc — see Consequences.

### Context
`MyChatOpenai` (Cell 5) was a custom wrapper around `ChatOpenAI` that historically handled o-series model quirks and the OpenAI Responses API. Per user, those quirks no longer require the wrapper layer.

### Decision
**Use `ChatOpenAI` directly going forward.** Existing `MyChatOpenai` call sites can be migrated opportunistically; new code must not introduce new `MyChatOpenai` references.

### Consequences
- ⚠️ This conflicts with `.github/agents/notebook-specialist.agent.md`, which (in its `<!-- repo-agent-bootstrap:managed -->` block) still mandates `MyChatOpenai`. **The managed block is not edited** (per maintenance contract). The divergence is documented here and in `memory-bank/activeContext.md`. When the managed block is next regenerated by `repo-agent-bootstrap`, the new `ChatOpenAI` rule should be re-asserted.
- `memory-bank/systemPatterns.md` managed block also still says "Always use MyChatOpenai" — same constraint applies.

---

## 2026-04-23 (later) — ADR: Zero stub/marker files in reports directory

### Status
✅ Accepted. Hard gate (criterion 5 of Phase 6 12-criteria).

### Context
Run 88's `file_writer` LLM compensated for an empty body by calling marker tools 25+ times, producing a clutter of files matching `*_ack`, `*_commit`, `*_ready`, `*_stub`, `*_trigger`. These have no downstream consumer.

### Decision
**Reports dir must contain 0 stub/marker files.** Implemented in Phase E by:
1. Removing stub-marker tools from `file_writer`'s tool list.
2. Capping total tool calls per `file_writer` invocation.
3. Rewriting the `file_writer` system prompt to forbid placeholder/marker emissions.

### Consequences
- Tools that other agents legitimately call as no-op markers (if any) must be re-scoped or renamed; `file_writer` must not have access.
- Validation: glob the reports dir post-run; any match against the stub patterns above fails the run.

---

## 2026-04-23 (later) — ADR: GitHub epic + 7 sub-issues for Phase 6

### Status
✅ Accepted.

### Context
Phase 6 spans seven distinct workstreams (forensic, instrumentation, viz pipeline, report pipeline, file_writer, supervisor gate, validation) with sequential dependencies (A → B → {C,D,E} → F → G).

### Decision
Tracked under **epic `dhar174/intelligent_data_detective#119`** with seven sub-issues:

| Phase | Issue | Scope |
|---|---|---|
| A | #112 | Forensic deep-dive (RC1–RC5, no code changes) |
| B | #113 | Instrument-only telemetry run |
| C | #114 | Fix viz pipeline |
| D | #115 | Fix report pipeline |
| E | #116 | Fix `file_writer` |
| F | #117 | Tighten supervisor FINAL gate |
| G | #118 | Validate against 12-criteria |

Labels: `epic`, `phase-6`, `forensic`, `instrumentation`, `viz-pipeline`, `report-pipeline`, `file-writer`, `supervisor`, `validation`.

### Consequences
- Cross-cutting fixes get filed against the closest sub-issue; epic #119 is the rollup.
- Phase A's RC5 (structured-output State persistence post-W9) is the newest concern and may re-open W9-SR-DROP design space if confirmed.

---

## 2026-04-23 (later) — Note: Wave 5 cutover backup location

`_wave5_backup_20260423-091557/` is the canonical pre-cutover snapshot (SOURCE `IntelligentDataDetective_beta_v5.ipynb` + `_patch_notebook.py` + the most-recent patched notebook). Use this as the rollback target for any Phase 6 notebook regression. Do not delete or compress without superseding-snapshot agreement.
