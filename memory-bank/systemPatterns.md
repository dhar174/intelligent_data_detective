<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# System Patterns

## High-level architecture
IDD uses a **supervisor-worker LangGraph** pattern. A single `State` TypedDict flows through all nodes. The supervisor routes work to specialised workers; workers return results and route back to the supervisor (or directly to an end node). All agents live in `IntelligentDataDetective_beta_v5.ipynb`.

```
__start__ → initial_analysis → supervisor
supervisor dispatches to:
  data_cleaner, analyst, visualization, viz_worker, viz_join, viz_evaluator,
  report_orchestrator, report_section_worker, report_join, report_packager, file_writer
supervisor or packager → __end__ (when report_done & report_ready & already_wrote)
supervisor → EMERGENCY_MSG → __end__  (on fatal error)
```
Authoritative topology: `idd_v4_state_graph.mmd`

## State reducers (Cell 7)
Fields on `State` carry custom reducer annotations. Never assign directly; use the declared semantics:
- `Annotated[T, keep_first]` — immutable after first set (e.g., original dataset path)
- `Annotated[list, operator.add]` — accumulate across steps (e.g., messages, artifact paths)
- `Annotated[list, _reduce_plan_keep_sorted]` — merge sorted plan items deduplicated by key

## Agent output models (Cell 7)
Every agent returns a Pydantic model extending `BaseNoExtrasModel`. Required base fields:
```python
reply_msg_to_supervisor: str
finished_this_task: bool
expect_reply: bool
```
`model_config = ConfigDict(extra="forbid")` — no extra fields allowed.

## Tool pattern (Cells 12–13)
```python
@handle_tool_errors
def my_tool(df_id: str, ...) -> tuple[str, dict]:
    validate_dataframe_exists(df_id)   # first, for any DataFrame-touching tool
    ...
    return result_str, artifact_dict
```
Tools are appended to typed lists (`data_cleaning_tools`, `analyst_tools`, etc.) at point of definition.

## DataFrameRegistry (Cell 8)
- Thread-safe LRU cache keyed by `df_id` (UUID string).
- `registry.register_dataframe(df, df_id)` — store a DataFrame.
- `registry.get_dataframe(df_id, load_if_not_exists=True)` — auto-reloads from CSV on miss.
- Never bypass the registry by passing DataFrames between functions directly.

## MyChatOpenai (Cell 5)
- Custom wrapper around `ChatOpenAI` that handles o-series model quirks and the OpenAI Responses API.
- **Always use `MyChatOpenai`** in the notebook — never `ChatOpenAI` directly.

## Memory namespaces
```
('memories', '<kind>')
```
Where `kind` ∈ `{conversation, analysis, cleaning, visualization, insights, errors}`. Per-kind TTL and limits configured in `memory_config.yaml`. Never hardcode values.

## File writes
All artifact writes go through `_resolve_artifact_path()` for path-traversal protection. Tools must not use `open()` directly.

## Extension points
- Add new cell: append after the relevant section, never reorder.
- Add new tool: apply `@handle_tool_errors`, call `validate_dataframe_exists`, append to tool list.
- Add new agent: add node to graph, add `memory_kinds` mapping in `memory_config.yaml`.
- Add new State field: choose reducer annotation, add to `State` TypedDict in Cell 7.
<!-- repo-agent-bootstrap:managed:end -->

<!-- session-curated:start -->
## Session-curated patterns (debugging knowledge from current run series)

### Structured response channel
```python
structured_response: Annotated[Optional[BaseNoExtrasModel], _sr_reducer]
_sr_reducer = lambda left, right: left if right is None and left is not None else right
```
**Why:** BR-8. Main agent node and the recovery shims both write to `structured_response` in the same superstep. The default LastValue reducer raised `InvalidUpdateError` on concurrent writes. `_sr_reducer` keeps the first non-None value when the incoming write is None, otherwise takes the new value (prefer-non-None last write).

### `State.messages` declaration
**Always declare directly:**
```python
messages: Annotated[list[AnyMessage], add_messages]
```
**Do NOT** inherit from `langchain.agents.AgentState` or `langgraph.prebuilt.chat_agent_executor.AgentState` — those bases drag in the `remaining_steps` *managed* channel which LangGraph rejects when the schema is used as an `InputSchema` (BR-7 regression).

### EMERGENCY_MSG contract
- Routed from `supervisor` when `next == "EMERGENCY_MSG"`.
- **Must have an outgoing edge** to `supervisor` or `__end__`. A bare-dict return without `Command(goto=...)` and no static edge in the graph = dead-end (currently the case; W2-EMERGENCY patch in Wave 4 fixes it).

### Schema tool names (RC-1)
`ToolStrategy(Schema)` registers the structured-output tool with `name = Schema.__name__`, **not** `respond`. Recovery code matching on `"respond"` will silently miss every emission. Per-agent expected tool names:

| Agent | Schema-tool name |
|---|---|
| data_cleaner | `CleaningMetadata` |
| initial_analyst | `InitialDescription` |
| analyst | `AnalysisInsights` |
| visualization | `VisualizationResults` |
| viz_evaluator | `VizFeedback` |
| report_orchestrator | `ReportOutline` |
| report_packager | `ReportResults` |

### RIP (`report_intermediate_progress`)
Historical "treadmill" tool — multi-cause loop bug (RC-2, 4 distinct causes). **Current state after W2-BB / W2-BC patches:**
- Removed from 5 worker tool lists; **only `init_analyst` retains it**.
- Loop-guard threshold lowered to `_rip_n >= 3`, keyed on `(thread_id, agent_name)`.
- Returns a `ToolMessage` with `status="error"` once threshold trips.
- **Future:** deprecate entirely.

### Recursion limits
- Inner agents: `recursion_limit = 160`
- Outer graph: `recursion_limit = 400`
- User explicitly raised these. **Do not lower** without an explicit user request.

### AgentState / supervisor-State channel-name collision (post-W9 rule)
**Pattern:** When using `langgraph.prebuilt.create_agent` with a custom supervisor `State`, do **NOT** add custom-reducer fields to `State` whose names collide with `AgentState` fields (`structured_response`, `messages`, `remaining_steps`).

**Why:** `langchain.agents.factory._resolve_schemas` builds the merged StateSchema/InputSchema/OutputSchema by iterating `set({State, AgentState[ResponseT]})`. Set iteration order varies with PYTHONHASHSEED, so for any colliding field the "winning" channel type is nondeterministic. langgraph's `_add_schema` only tolerates type mismatches when the second-iter channel is `LastValue` — adverse hash-seed rolls produce `ValueError` at `create_agent` time (e.g., cell 48). Even when iteration succeeds, forcing `LastValue` on the supervisor channel turns any concurrent write (viz fan-out, recovery shims, subgraph→parent propagation) into `InvalidUpdateError` ("Can receive only one value per step").

**Resolution rules:**
1. For sr specifically: omit `structured_response` from supervisor `State` entirely. Each create_agent subgraph keeps its own internal `AgentState[ResponseT].structured_response`; wrapper nodes read it via `result["structured_response"]` from the `agent.invoke()` return dict — that is a Python-dict access, NOT a graph channel read. (W9-SR-DROP, Run 87.)
2. If you must keep a colliding field on `State`, mirror AgentState's annotation **exactly** (`NotRequired[Annotated[Optional[Any], OmitFromSchema(input=True, output=False)]]`) so set-iteration produces the same channel either way (W7-SR-ALIGN). This avoids cell-48 ValueError but still leaves multi-writer LastValue collisions — only safe if you guarantee a single writer per superstep.
3. `messages` must be declared directly on `State` as `Annotated[list[AnyMessage], add_messages]` and `State` must NOT inherit from AgentState (BR-7).

**Diagnostic signature:** intermittent `ValueError` at create_agent / `InvalidUpdateError: At key 'structured_response': Can receive only one value per step` only on some runs.

### W2-* patcher series
All notebook changes flow through `_patch_notebook.py` as sentinel-bracketed blocks (~80 sentinels accumulated this session). Triage classifications (✅ WORKING / ⚠️ NEEDS-WORK / ❌ ABANDON) live in `patcher-audit.md` (session state). **Wave 5** will fold ✅ WORKING sentinels into the source notebook permanently and retire the corresponding patcher entries.
<!-- session-curated:end -->

<!-- session-curated:2026-04-23-phase6:start -->
## 2026-04-23 (later) — Phase 6 anti-patterns & open concerns

### Anti-pattern: Potemkin pipeline (Run 88)
**Symptom:** All structural / cleanliness gates pass (no `InvalidUpdateError`, no recoveries, no tracebacks, FINAL marker reached, viz=True report=True), but the deliverable is a hollow shell. Concretely in Run 88:
- Report HTML body = 356 chars (verbatim placeholder string).
- 0 sections written (`Final report assembled from 0 sections`).
- 1 viz embedded 5× with cosmetic hash variations (analyst recommended 3 distinct viz_specs).
- PDF = 1983 bytes (title page only).
- 25+ stub-marker files in reports dir (`*_ack`, `*_commit`, `*_ready`, `*_stub`, `*_trigger`).
- Analyst output upstream WAS rich and correct — content was lost between analyst and report.

**Root causes (under forensic confirmation, Phase 6 RC1–RC5):**
- **RC1** — viz fan-out: only 1 of N analyst-recommended viz_specs survives the worker → join → evaluator path.
- **RC2** — `report_section_worker` writes 0 entries to `written_sections`.
- **RC3** — `file_writer` LLM emits placeholder body and spams stub-marker tools to look busy.
- **RC4** — supervisor FINAL gate accepts "viz=True / report=True" booleans without validating content.
- **RC5** — structured Pydantic outputs may not actually be persisted to supervisor `State` after W9-SR-DROP; downstream nodes may be reading empty / stale State fields while the wrapper-dict path looks healthy.

**Detection:** structural gates alone are insufficient. Validate against the **Phase 6 12-criteria content bar** (see `decisionLog.md` ADR Phase 6 pivot): HTML text ≥ 3000 chars, ≥ 4 sections, ≥ 3 distinct visualizations (different titles AND different image bytes), PDF ≥ 30 KB, 0 stub files, report references actual analyst findings (correlation r values, anomaly notes verbatim).

**Mitigation pattern:** every node that produces a content artifact must (a) have a min-content validator on its Pydantic output (e.g., `Section.body` min-length), (b) raise (not silently return placeholder) when the validator fails, and (c) have a corresponding supervisor-gate precondition that checks the *content*, not just a boolean flag.

### Concern: Structured-output State persistence after W9-SR-DROP
**Pattern under investigation (Phase 6 RC5):** With `structured_response` removed from supervisor `State` (W9-SR-DROP), each `create_agent` subgraph still owns its own internal sr via `AgentState[ResponseT]`. Wrapper nodes read it from the Python-dict return of `agent.invoke()` — but **whatever the wrapper does next** to land that data in supervisor `State` (e.g., copying fields into `analysis_insights`, `viz_specs`, `written_sections`, `report_outline`) is the *only* persistence path. If that copy is missing, partial, or writes to a field with no reducer / a default-LastValue reducer that gets overwritten by a later worker, downstream nodes silently read empty data.

**Why this matters now:** Run 88's hollow report is consistent with `report_section_worker` reading an empty `report_outline` even though `report_orchestrator`'s subgraph successfully produced a rich `ReportOutline` Pydantic instance. Static grep for "what writes `report_outline` / `written_sections` / `viz_specs` to `State` post-W9" is the Phase A deliverable.

**Diagnostic checklist for any field formerly carried by the sr channel:**
1. Does the wrapper read `result["structured_response"]` post-invoke?
2. Does the wrapper explicitly assign each Pydantic-model field to a supervisor `State` field in its return dict?
3. Does that supervisor `State` field have an appropriate reducer (not default LastValue if multiple workers write)?
4. Does the consumer node read from supervisor `State`, or does it accidentally re-call `agent.invoke()` and re-parse?

### Pattern: Wave-5 direct-edit discipline
With `_patch_notebook.py` retired as a changelog-only artifact (per decisionLog ADR Wave 5), notebook edits go straight into `IntelligentDataDetective_beta_v5.ipynb`. Discipline rules:
- Cell additions/reorderings still forbidden (notebook 27-cell contract).
- Each Phase 6 edit references the originating GH sub-issue (#112–#118) in its commit message.
- Pre-cutover rollback target: `_wave5_backup_20260423-091557/`.
- Sentinel IDs from Wave 4 (W7, W8, W9, W10, …) remain valid identifiers in commit messages and decision-log entries even though no sentinel comments are added to the notebook anymore.

### Pattern: Content-validation supervisor precondition (Phase F target)
Future supervisor FINAL gate must check, in order:
1. `len(state["written_sections"]) >= 4`
2. `len({v.title for v in state["viz_results"]}) >= 3` AND distinct image-byte hashes
3. `len(state["report_html_body_text"]) >= 3000`
4. `os.path.getsize(state["pdf_path"]) >= 30_000`
5. `glob(reports_dir / "*_{ack,commit,ready,stub,trigger}*") == []`
6. Report body contains substrings from `state["analysis_insights"].correlation_insights` and `.anomaly_insights`.

If any fail, route to a remediation node (or fail loudly) — **not** to FINAL with `report=True`.

### Note on `MyChatOpenai` ↔ `ChatOpenAI` divergence
The managed block above (line 50–52) still says "Always use `MyChatOpenai`". As of 2026-04-23 the user has marked `MyChatOpenai` as **legacy** and prefers `ChatOpenAI` direct (see `decisionLog.md` ADR). The managed block is intentionally NOT edited. Treat the ADR as authoritative for new code; treat the managed block as the regenerable scaffold contract.
<!-- session-curated:2026-04-23-phase6:end -->

