<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# System Patterns

## High-level architecture
IDD uses a **supervisor-worker LangGraph** pattern. A single `State` TypedDict flows through all nodes. The supervisor routes work to specialised workers; workers return results and route back to the supervisor (or directly to an end node). Current runnable proof work lives in `IntelligentDataDetective_beta_v5_patched.ipynb`, regenerated from the source notebook by `_patch_notebook.py`.

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

## W14 visualization fan-in pattern
W14H fixed a parallel fan-in race where `visualization_results` could behave like a last-writer channel and preserve only one worker output. `viz_join` must union all available channels before evaluation:
- `viz_results`
- `visualization_results`
- `viz_paths`
- discovered PNG artifacts under run/artifact directories

The proof marker is `viz_join sent_count=3 received_count=3` followed by `viz_evaluator.start viz_tasks_count=3 viz_results_count=3`.

## W14 artifact completion pattern
Completion is not a boolean-only gate. A passing final run must produce canonical root `final_report.html`, `final_report.md`, and `final_report.pdf`; embed resolving visualization paths in root HTML/Markdown; avoid `.txt` marker/status files; and pass both validators (`validate_run.py` 12/12, `validate_artifact_quality.py` 9/9).

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
- Add notebook behavior: update `_patch_notebook.py`, regenerate the patched notebook, and validate.
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

<!-- session-curated:2026-10-12-cp1-corrective-pass:start -->
### Pattern: Lossless DataFrameRegistry Backing Persistence (CP1 Corrective Pass)
**Pattern:** Internal backing snapshots in `DataFrameRegistry` must strictly use `.pkl` for disk serialization on LRU cache eviction, while external files (`raw_path`) preserve user formats.
```python
# Internal backing files in self.data_dir:
filename = f"{safe_prefix}_{digest[:16]}.pkl"
# Saved via pd.to_pickle(df, path) and reloaded via pd.read_pickle(path)
```
**Why:** CSV serialization silently discards integer column name types (`0` stringified to `'0'`), timezone metadata on datetimes, MultiIndex column/row hierarchies, and nullable dtypes (`pd.NA`, `Int64`, `boolean`). Using `.pkl` provides 100% lossless representation fidelity across cache eviction while keeping unpickling strictly confined to verified paths within `self.data_dir` (`_get_owned_backing_path`).
<!-- session-curated:2026-10-12-cp1-corrective-pass:end -->

<!-- session-curated:2026-10-13-cp1-private-storage-security:start -->
### Pattern: Private Storage Boundary & Filesystem Security Discipline (CP1 Security Pass)
**Pattern:** Registry-owned snapshot storage must be strictly isolated per instance, permission-hardened, and validated against untrusted filesystem locations to protect pickle serialization:
```python
# 1. Default storage directory allocation:
tempfile.mkdtemp(prefix="idd_registry_", dir=str(WORKING_DIRECTORY))
# Enforced 0700 permissions (drwx------) on POSIX

# 2. Backing file writes via atomic staging:
tmp_path = path.parent / f".tmp_{uuid.uuid4().hex[:12]}_{path.name}"
# Mode 0600 (-rw-------) applied on POSIX before atomic os.replace(tmp_path, path)

# 3. Directory pre-validation on explicit data_dir:
# - Rejects filesystem root (path == path.parent)
# - Rejects non-directory files
# - On POSIX: rejects world-writable dirs (st.st_mode & stat.S_IWOTH)
# - On POSIX: rejects unowned dirs (st.st_uid != os.getuid())

# 4. Deserialization trust boundary check:
if path.suffix.lower() in (".pkl", ".pickle"):
    resolved = path.resolve()
    resolved_data_dir = self.data_dir.resolve()
    if not resolved.is_relative_to(resolved_data_dir) or resolved == resolved_data_dir:
        raise DataFrameRegistryError(...)
```
**Why:** Deserializing pickles from world-writable or multi-user shared directories without strict permission enforcement creates local privilege escalation and tampering risks. By creating isolated directories via `mkdtemp(0700)`, restricting snapshot permissions to `0600`, and enforcing containment within verified `self.data_dir`, the registry maintains a secure deserialization boundary. On Windows, security relies on NTFS DACL inheritance within `%TEMP%` / `%USERPROFILE%`.
<!-- session-curated:2026-10-13-cp1-private-storage-security:end -->

<!-- session-curated:2026-10-13-cp2-canonical-tools:start -->
### Pattern: Standardized Tool Error Contract & Exception Sanitization (CP2)
**Pattern:** Data tools return a standardized 4-field user-safe dictionary rather than raw strings:
```python
{
    "status": "error",
    "operation": func.__name__,
    "reason": "An unexpected data-processing failure occurred." | specific_reason,
    "action": "User-actionable remediation instructions",
}
```
**Why:** Standardizing the error dictionary matches notebook Cell 32 and provides actionable guidance for LLM agent reflection. Unexpected internal exceptions are logged with full traceback via `logger.exception()` but sanitized in the response to prevent leaking internal stack traces, system paths, or environment secrets to model prompts. Known domain exceptions (`FileNotFoundError`, `KeyError`, `ValueError`, `EmptyDataError`, `ParserError`, `DtypeWarning`) return distinct, actionable diagnostics.

### Pattern: Signature-Aware Argument Binding in Tool Decorator (CP2)
**Pattern:** Decorators inspecting parameters (such as `df_id` for DataFrame validation) must use Python's `inspect.signature` rather than positional indexing:
```python
func_sig = inspect.signature(func)
bound_args = func_sig.bind(*args, **kwargs)
bound_args.apply_defaults()

if "df_id" in bound_args.arguments:
    df_id = bound_args.arguments["df_id"]
    # validate registered DataFrame exists
```
**Why:** Blind `args[0]` checks (the root cause of Claim 7 and historical test failures) erroneously treat the first argument of any tool as a DataFrame identifier. When decorating functions without a `df_id` parameter (e.g. `func(message: str, count: int)`), passing `"hello"` fails because `"hello"` is not in the registry. Signature binding accurately extracts `df_id` regardless of positional index, keyword argument order, or parameter object attribute (`.df_id`), and cleanly skips DataFrame validation for functions that do not take a DataFrame ID.

### Pattern: Artifact Path Containment & Multi-Root Verification (CP2 / PR #149)
**Pattern:** All tool writes must use single-root containment, while report aggregators use multi-root read validation:
```python
# Single-root write containment (tools):
def _resolve_artifact_path(file_name: str, config=None, subdir=None, create_parents=True):
    base = _get_artifacts_base(config)
    path = (base / file_name).resolve()
    if not _is_subpath(path, base):
        raise ValueError("Path outside artifacts root")
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)  # Delayed mkdir only after validation
    return path

# Multi-root read verification (report/packager nodes):
def is_safe_artifact_path(candidate, allowed_roots, require_file=False):
    # Verifies candidate is within at least one authorized root and rejects URI schemes
```
**Why:** Single-root containment protects against path traversal attacks (`../`), absolute path escapes, and symlink escapes, ensuring tool artifacts are written strictly into the designated run directory. Delayed directory creation prevents malicious or malformed paths from creating directories before validation. Multi-root read verification allows downstream publication nodes to verify images across artifacts, working directories, and run directories without weakening write containment.

### Pattern: Exception Boundary, Diagnostic Integrity & RFC URI Rejection (CP2 Corrective Pass)
**Pattern:** Enclose pre-execution reflection in exception-safe boundaries and reject remote RFC URI schemes while honoring Windows paths:
```python
# 1. Closed decorator boundary:
@wraps(func)
def wrapper(*args, **kwargs):
    try:
        # Pre-execution: signature binding, parameter property access, DataFrame validation
        ...
        return func(*args, **kwargs)
    except Exception as e:
        # Unexpected descriptor / infrastructure exceptions logged locally with traceback
        # but sanitized to user-safe dictionary without leaking secret canaries
        return _tool_failure(func.__name__, e)

# 2. Strict DataFrame validation vs infrastructure error separation:
def _validate_dataframe_for_tool(df_id: str):
    # Distinguishes genuine missing/empty DataFrame from unexpected registry RuntimeError.
    # Registry errors re-raise to propagate cleanly to _tool_failure sanitization.

# 3. RFC URI scheme rejection:
_SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
def _is_rfc_uri(path_str: str) -> bool:
    # Rejects remote schemes (http:, https:, data:, etc.) or :// before path resolution
    # Safely distinguishes single-letter Windows drive letters (C:\ or C:/)
```
**Why:** 
1. If a caller passes an object whose attribute descriptor (like `.df_id`) raises `RuntimeError`, or if argument binding fails unexpectedly, raw exceptions could bypass `@handle_tool_errors`, crash the agent node, or leak confidential canaries in prompt context. Enclosing all wrapper operations in a unified `try...except` guarantees complete sanitization.
2. In `_validate_dataframe_for_tool`, treating an unexpected registry crash as a missing dataframe misinforms the agent to "check the dataframe ID". Propagating unexpected errors to `_tool_failure` yields a system error response with local diagnostic logging.
3. Rejecting RFC URI schemes before `resolve()` prevents remote network URLs from bypassing path normalization, while preserving absolute Windows local paths (`C:\...`).
<!-- session-curated:2026-10-13-cp2-canonical-tools:end -->




