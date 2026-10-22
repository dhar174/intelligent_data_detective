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

### Pattern: Exception Boundary, Diagnostic Integrity & Path Pre-Filter Security (CP2 Final Corrective Pass)
**Pattern:** Enclose pre-execution reflection in exception-safe boundaries, enforce exact 4-key error schema equality with canary secret leak prevention, and reject remote RFC URIs, single-letter schemes, drive-relative paths, and POSIX UNC notation:
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

# 3. Ambiguous URI, Drive-Relative & POSIX UNC Path Rejection:
_URI_SCHEME_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")

def _is_uri_scheme(path_str: str) -> bool:
    # Only unambiguous absolute Windows drive paths (^[A-Za-z]:[/\\\\]) are exempted.
    # Rejects remote schemes (http:, https:, data:, etc.), single-letter schemes (a:payload),
    # and Windows drive-relative paths (C:relative.png, D:folder\file.txt) on all platforms
    # before path resolution, without creating directories.

# In _resolve_artifact_path / is_safe_artifact_path:
if os.name != "nt" and (_WINDOWS_DRIVE_PATTERN.match(fn_stripped) or fn_stripped.startswith(r"\\")):
    raise ValueError(...)  # POSIX UNC and drive letter rejection before resolution

# 4. Diagnostic Probe Hardening (Claim 6):
def probe_claim_6():
    # Enforces exact 4-key schema equality:
    # set(res.keys()) == {'status', 'operation', 'reason', 'action'}
    # and serializes entire payload (json.dumps(res, default=str)) to verify
    # complete absence of synthetic canary secrets in all fields.
```
**Why:** 
1. If a caller passes an object whose attribute descriptor (like `.df_id`) raises `RuntimeError`, or if argument binding fails unexpectedly, raw exceptions could bypass `@handle_tool_errors`, crash the agent node, or leak confidential canaries in prompt context. Enclosing all wrapper operations in a unified `try...except` guarantees complete sanitization.
2. In `_validate_dataframe_for_tool`, treating an unexpected registry crash as a missing dataframe misinforms the agent to "check the dataframe ID". Propagating unexpected errors to `_tool_failure` yields a system error response with local diagnostic logging.
3. Rejecting ambiguous URI schemes and Windows drive-relative paths (`C:foo`) prevents remote URLs or unanchored drives from creating directories before validation. Rejecting UNC paths on POSIX (`\\\\server\\share`) prevents OS-dependent path traversal, while on Windows UNC paths are contained within authorized roots.
4. Serializing the entire dictionary payload in `probe_claim_6` guarantees that canary secrets do not leak into undeclared dictionary keys or metadata, avoiding false-green diagnostic certifications.
<!-- session-curated:2026-10-13-cp2-canonical-tools:end -->

<!-- session-curated:2026-10-14-cp3-canonical-state-graph:start -->
### Pattern: Canonical State, Reducer Discipline & Channel Invariants (CP3)
**Pattern:** Canonical State (`idd_state.py`) defines 70 fields (39 with custom reducers, 31 plain fields) with strict channel discipline:
```python
class State(TypedDict, total=False):
    # BR-7 Invariant: declared directly, never inheriting from AgentState
    messages: Annotated[list[AnyMessage], add_messages]
    
    # W9-SR-DROP Invariant: structured_response intentionally OMITTED from supervisor State
    
    # Turn accounting & boolean state
    _report_dispatched: Annotated[Optional[bool], bool_or]
    ...

# Safe boolean-OR reducer:
def bool_or(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    """Safe boolean-OR reducer: treats None as False, avoids TypeError."""
    return b if a is None else (a if b is None else bool(a | b))
```
**Why:**
1. **BR-7**: Inheriting `State` from LangChain's `AgentState` drags in the `remaining_steps` managed channel which LangGraph's `InputSchema` rejects at graph compilation. Direct declaration avoids managed channel collisions.
2. **W9-SR-DROP**: In `langgraph.prebuilt.create_agent`, each agent subgraph manages its own `AgentState[ResponseT]`. Colliding a supervisor custom-reducer channel with `AgentState.structured_response` causes `_resolve_schemas` set-iteration nondeterminism and `InvalidUpdateError` on concurrent writes across steps. Wrapper nodes read structured responses from the `agent.invoke()` Python dictionary return (`result["structured_response"]`) and map fields into dedicated supervisor State keys.
3. **Safe Reducers (`bool_or`)**: Naive bitwise OR (`a | b`) crashes with `TypeError` when either operand is `None` during intermediate supersteps. Checking `None` first ensures robust multi-agent state accumulation.

### Pattern: Canonical Graph Assembly & Offline Pass-Through Architecture (CP3)
**Pattern:** `idd_graph.py` encapsulates `build_graph()` to construct the authoritative 15-node, 33-edge StateGraph matching `validate_graph.py`:
```python
def build_graph(
    node_overrides: Optional[Dict[str, Callable]] = None,
    checkpointer: Any = None,
    store: Any = None,
    cache: Any = None,
    structural_only: bool = True,
    **kwargs: Any,
) -> CompiledStateGraph:
    ...
```
**Why:**
1. **Offline Testability**: Providing default lightweight pass-through nodes (`_make_pass_through_node`) allows compiling and verifying the exact 15-node graph topology without live LLM calls, external API keys, or side-effect SQLite database file creation (`checkpointer=None`).
2. **Barrier & Fan-In Preservation**: Preserves all structural graph barriers (`viz_join`, `report_join`, `EMERGENCY_MSG` fan-in edge back to `supervisor`), eliminating unreachable nodes and dead ends.
3. **Strict Layering DAG**: Establishes a clean unidirectional DAG (`idd_models` -> `idd_state` -> `idd_graph` -> `idd_core`), completely decoupling graph compilation from execution adapters.
<!-- session-curated:2026-10-14-cp3-canonical-state-graph:end -->

<!-- session-curated:2026-10-14-cp3-corrective-pass:start -->
### Pattern: Dual-Mode Graph Assembly & Fail-Closed Node Validation (CP3 Corrective Pass)
**Pattern:** `build_graph()` distinguishes structural compile/topology checks from executable pipelines:
```python
# In idd_graph.py:
def _make_pass_through_node(name: str) -> Callable[[State], dict]:
    def pass_through_node(state: State) -> dict:
        raise RuntimeError(
            f"Cannot execute placeholder node '{name}'. "
            "Graph was assembled with structural_only=True. "
            "To execute the graph, provide node_overrides and compile with structural_only=False."
        )
    pass_through_node.__name__ = f"{name}_node"
    return pass_through_node

def build_graph(
    node_overrides: Optional[Dict[str, Callable]] = None,
    ...,
    structural_only: bool = True,
    **kwargs: Any,
) -> CompiledStateGraph:
    ...
    if not structural_only:
        required_nodes = set(PRODUCTION_NODE_NAMES) - {"FINISH"}
        missing = required_nodes - set(overrides.keys())
        if missing:
            raise ValueError(f"Cannot build executable graph (structural_only=False): missing node override(s): {sorted(missing)}")
```
**Why:**
1. **Safety Boundary**: Structural compilation allows fast unit testing and topology validation without requiring mock implementations for all 14 nodes.
2. **Fail-Closed Execution**: If a test or runtime invokes a placeholder node, it immediately raises `RuntimeError` rather than silently passing through an empty dictionary.
3. **Executable Mode Validation**: When `structural_only=False`, the graph builder validates upfront that all required production/mock nodes are present, failing closed with `ValueError` on missing or unrecognized node overrides.

### Pattern: Fail-Closed Supervisor Routing & Self-Loop Rejection (CP3 Corrective Pass)
**Pattern:** `route_from_supervisor()` rigorously validates routing targets and rejects supervisor self-loops:
```python
def route_from_supervisor(state: State) -> str:
    # W13N-RT completion route:
    if bool(state.get("report_generator_complete")) and state.get("report_results"):
        return "FINISH" if bool(state.get("file_writer_complete")) else "file_writer"

    nxt = state.get("next")
    if not nxt:
        raise ValueError("Supervisor routing failed: destination 'next' is missing or empty in State.")

    allowed: Set[str] = {
        "initial_analysis", "data_cleaner", "analyst", "viz_worker", "viz_join",
        "viz_evaluator", "visualization", "report_orchestrator", "report_section_worker",
        "report_join", "report_packager", "file_writer", "FINISH", "EMERGENCY_MSG",
    }
    if nxt not in allowed:
        raise ValueError(f"Supervisor routing failed: destination {nxt!r} is invalid. Allowed destinations: {sorted(allowed)}")
    return nxt
```
**Why:**
1. **Self-Loop Prevention**: Removing `'supervisor'` from `allowed` prevents infinite self-routing loops and catastrophic `KeyError: 'supervisor'` during conditional edge resolution.
2. **Deterministic Error Halting**: Missing, empty, or unmapped worker destinations fail immediately with descriptive `ValueError` rather than propagating undefined states or falling back to a default node.

### Pattern: Pydantic v2 Required-Nullable Contract & Schema Parity (CP3 Corrective Pass)
**Pattern:** Strict field parity with Cell 7 schemas:
```python
# AgentMembers: strictly 2 fields (membership listing, not routing)
class AgentMembers(BaseModel):
    description: str = Field(default="Members of an agent list.")
    agent_type: Literal[...]

# Router: next_agent_metadata required in JSON schema but nullable
class Router(BaseNoExtrasModel):
    next: AgentId = Field(..., description="The next agent to route to.")
    next_agent_prompt: str = Field(..., description="Instructions for the next agent.")
    next_agent_metadata: Optional[NextAgentMetadata] = Field(
        ...,
        description="Metadata for the next agent.",
    )
```
**Why:**
1. **AgentMembers Separation of Concerns**: Routing is dynamic and determined by `Router` during execution. Embedding `next` on `AgentMembers` caused confusion between static worker definitions and dynamic routing decisions.
2. **Pydantic v2 Required-Nullable Serialization**: Using `Field(...)` with `Optional[T]` places `next_agent_metadata` in `schema['required']` while configuring `anyOf: [..., {'type': 'null'}]`. This requires the LLM or caller to explicitly evaluate metadata (passing `None` if absent) rather than omitting the field from JSON payloads.
<!-- session-curated:2026-10-14-cp3-corrective-pass:end -->

<!-- session-curated:2026-10-14-cp3-review-resolutions:start -->
### Pattern: Fan-Out Parent State Context Propagation & Dict Normalization (CP3 Review Resolution)
**Pattern:** In `idd_graph.py`, fan-out routing functions (`assign_viz_workers` and `dispatch_sections`) creating `Send` objects MUST forward parent state context alongside targeted items:
```python
# assign_viz_workers: forwards context aligned with notebook Cell 57 Fix AM-1
return [
    Send("viz_worker", {
        "task": task, "spec": spec,
        "cleaning_metadata": state.get("cleaning_metadata"),
        "analysis_insights": state.get("analysis_insights"),
        "available_df_ids": state.get("available_df_ids"),
        "_config": state.get("_config"),
        "messages": state.get("messages", []),
        "final_turn_msgs_list": state.get("final_turn_msgs_list", []),
    })
    for task, spec in zip(tasks, specs)
]

# dispatch_sections: normalizes models to dicts and injects section worker dependencies
return [
    Send("report_section_worker", {
        "section": s.model_dump() if hasattr(s, "model_dump") else s,
        "cleaning_metadata": state.get("cleaning_metadata"),
        "analysis_insights": state.get("analysis_insights"),
        "viz_results": state.get("viz_results"),
        "available_df_ids": state.get("available_df_ids"),
        "_config": state.get("_config"),
        "user_request": state.get("user_request"),
    })
    for s in sections
]
```
**Why:**
1. **Prevent State Starvation**: In LangGraph, `Send` payloads initialize the worker's input state. If parent channels (`cleaning_metadata`, `analysis_insights`) are omitted, worker guards trigger fail-safes (e.g., `section_worker` immediately aborting to `data_cleaner`), resulting in 0 completed tasks and supervisor loops.
2. **Dict Serialization Parity**: Subworkers expect `dict` structures for section inputs rather than raw Pydantic model objects, avoiding serialization errors across process boundaries.

### Pattern: LangChain Fallback Stubs in Compatibility Façade (CP3 Review Resolution)
**Pattern:** Façade modules (`idd_core.py`) re-exporting framework symbols must provide complete fallback stubs when third-party packages are absent (`HAS_LANGCHAIN = False`):
```python
if not HAS_LANGCHAIN:
    class HumanMessage: ...
    class AIMessage: ...
    class SystemMessage: ...
    class ToolMessage: ...
    class BaseMessage: ...
    AnyMessage = Any
    def add_messages(left, right): return (left or []) + (right or [])
    class MyChatOpenai: ...
```
**Why:**
Guarantees `from idd_core import *` never raises `AttributeError` in lightweight, offline, or test environments lacking LangChain installations.

### Pattern: Separation of AgentId and SupervisorRouteTarget with EMERGENCY_MSG (CP3 Review Resolution)
**Pattern:** Preserve `AgentId` in `idd_state.py` for exact 14-value member typing, while defining `SupervisorRouteTarget` in `idd_graph.py` for supervisor routing:
```python
# idd_state.py: 14 canonical agents
AgentId: TypeAlias = Literal[
    "supervisor", "initial_analysis", "data_cleaner", "analyst",
    "visualization", "viz_evaluator", "viz_worker", "viz_join",
    "report_orchestrator", "report_section_worker", "report_join",
    "report_packager", "file_writer", "FINISH",
]

# idd_graph.py: supervisor routing targets including emergency & terminal tokens
SupervisorRouteTarget: TypeAlias = Literal[
    "initial_analysis", "data_cleaner", "analyst", "visualization",
    "report_orchestrator", "file_writer", "viz_worker", "viz_join",
    "viz_evaluator", "report_section_worker", "report_join", "report_packager",
    "END", "FINISH", "EMERGENCY_MSG",
]

# route_from_supervisor normalizes 'END' to 'FINISH'
if nxt == "END":
    nxt = "FINISH"
```
**Why:**
1. **Contract Invariant**: Modifying `AgentId` breaks downstream test assertions and message models expecting 14 agent identifiers.
2. **Emergency Routing**: The supervisor must be able to route to `EMERGENCY_MSG` on unrecoverable failures.
3. **Terminal Normalization**: Both `"END"` and `"FINISH"` cleanly transition to the 15-node graph's terminal `FINISH` node.

### Pattern: Authoritative 15-Node, 33-Edge LangGraph Topology Defense (CP3 Review Resolution)
**Pattern:** Reject external suggestions to simplify or rewire edges that violate verified notebook architecture:
1. **`viz_evaluator` -> `supervisor`**: Under patch sentinel `W2-BR6` (Cell 60), `viz_evaluator` routes both `"Accepted"` and `"Revise"` back to `supervisor` for centralized orchestration.
2. **`report_orchestrator` -> `report_join`**: Direct static edge (`# ensure join waits for all`) acts as a mandatory barrier synchronization edge.
**Why:**
Eliminating these edges or rerouting directly between workers causes race conditions, circumvents supervisor state reconciliation, and violates `validate_graph.py` assertions (15 nodes, 33 edges, 21 conditional).
<!-- session-curated:2026-10-14-cp3-review-resolutions:end -->

<!-- session-curated:2026-10-15-cp4-notebook-integration:start -->
### Pattern: Canonical Module Notebook Execution & Provenance Bootstrap (CP4)
**Pattern:** The production notebook (`IntelligentDataDetective_beta_v5_patched.ipynb`) directly imports core functionality from canonical Python modules at repo root rather than defining duplicate classes:
- **Cell 4**: `idd_bootstrap.bootstrap_environment()` verifies system path, working directory, and validates cryptographic SHA-256 digests in `idd_manifest.json` against on-disk canonical modules.
- **Cell 16**: `from idd_models import Plan, BaseNoExtrasModel, ...; from idd_state import AnalysisConfig`
- **Cell 19**: `from idd_registry import DataFrameRegistry, get_global_registry, get_global_df_registry; global_df_registry = get_global_registry()`
- **Cell 22**: `from idd_state import State, VizWorkerState, keep_first, ...` (all 15 reducers)
- **Cell 32**: `from idd_tools import handle_tool_errors, validate_dataframe_exists, _resolve_artifact_path`
- **Cell 60**: `from idd_graph import build_graph; data_analysis_team_builder = build_graph(..., structural_only=False, node_cache_policies=CachePolicy(ttl=120))`

**Why:**
1. **Eliminate Split-Brain**: Prevents drift between tested Python modules and runnable notebook logic. Any improvement or bugfix in `idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, or `idd_graph.py` is immediately active in the notebook without dual-editing.
2. **Provenance & Tamper Detection**: `idd_bootstrap.py` checks that canonical files on disk match their recorded SHA-256 hashes in `idd_manifest.json`, failing loudly if an untracked change or file corruption occurs.

### Pattern: Automated Anti-Drift Parity Guard (`test_core_parity_guard.py`) (CP4)
**Pattern:** Continuous regression suite enforcing three layers of architectural anti-drift:
1. **Static AST Analysis**: Sweeps all 99 notebook cells to assert canonical import statements are present in Cells 4, 16, 19, 22, 32, 60, and verifies zero duplicate definitions of `Plan`, `State`, `DataFrameRegistry`, `handle_tool_errors`, `_resolve_artifact_path`, or `build_graph` across any other cell.
2. **Runtime Object Identity**: Asserts `Plan is idd_models.Plan`, `State is idd_state.State`, `DataFrameRegistry is idd_registry.DataFrameRegistry`, `global_df_registry is get_global_registry()`, `handle_tool_errors is idd_tools.handle_tool_errors`, and `build_graph is idd_graph.build_graph`.
3. **Behavioral Invariants**: Verifies Codex Findings 1 & 2 (positional non-`df_id` strings do not trigger registry lookup, directory traversal in subdirs is blocked before mkdir), cache reload, and 15-node executable graph compilation with cache policies.
4. **Negative Controls**: Tests synthetic redefinitions, manifest digest tampering, missing modules, and missing files to guarantee the guard fails closed.
<!-- session-curated:2026-10-15-cp4-notebook-integration:end -->

<!-- session-curated:2026-10-21-cp5-proof-runner:start -->
### Pattern: Fail-Closed Proof Runner & Out-of-Tree Child-Kernel Execution (CP5)
**Pattern:** Production and preparation verification runs must execute via `idd_proof.py` / `tools/proof_runner.py` targeting strictly isolated directories outside the source repository:
1. **Two-Stage Execution Protocol**:
   - `prepare`: Sets up the out-of-tree workspace, copies required datasets and the patched notebook, and executes preflight cells up to Cell 60 in an isolated child Jupyter kernel. Returns status `PREPARED_NO_PRODUCTION_EXECUTION` and `cp5_pass=False` with zero API credentials loaded and zero live spend ($0.00).
   - `run`: Executes production graph cells only when explicitly authorized and commanded.
2. **Fail-Closed Negative Controls**:
   - If an unexpected or mismatched kernel identity is detected (`KERNEL_IDENTITY_MISMATCH`), the runner immediately terminates with exit code `1`, preserving forensic partial evidence (`attempt.json`, execution logs) in the target directory without corrupting git or working tree state.
3. **Workspace Hygiene & Non-Contamination**:
   - Running in dedicated attempt directories outside the git worktree guarantees that run artifacts (CSVs, charts, PDFs, SQLite checkpoint databases, and logs) never dirty the git tree or trigger false-positive anti-drift detections.
<!-- session-curated:2026-10-21-cp5-proof-runner:end -->

<!-- session-curated:2026-10-22-cp5-sw-figure-order:start -->
### Pattern: Section Worker Figure Metadata Initialization Order (CP5-SW-FIGURE-ORDER)
**Pattern:** In Cell 57 (`section_worker`), figure metadata and `expected_viz_str` preparation must strictly precede `default_instruction` string interpolation:
```python
# Prepare expected visualization metadata FIRST
expected_viz_str = ""
if section.required_figures:
    figure_descs = []
    for fig_id in section.required_figures:
        # lookup and format figure title, desc, path...
        figure_descs.append(...)
    expected_viz_str = f"The following figures were generated: ...\n" + "\n".join(figure_descs)

# Construct instruction SECOND using populated expected_viz_str
default_instruction = f"""...
{expected_viz_str}
..."""
```
**Why:**
1. **UnboundLocalError Prevention**: In Python functions, referencing a variable in an f-string before its assignment causes Python to treat it as local throughout the scope, raising `UnboundLocalError: local variable 'expected_viz_str' referenced before assignment` when executed.
2. **Fail-Closed Patcher Invariant**: In `_patch_notebook.py`, the active section-scoped W13J-SW transformation uses required-anchor validation to ensure that figure-metadata preparation is relocated ahead of `default_instruction` without introducing syntax drift or altering downstream prompt contracts.
3. **Regression Prevention**: `tests/unit/test_section_worker.py` (8 tests) enforces this ordering contract by executing the actual compiled Cell 57 body with canonical models/messages and mock boundaries locally and inside a real Jupyter kernel (`GENERATED_CELL57_NO_KEY_KERNEL_PASS`).
<!-- session-curated:2026-10-22-cp5-sw-figure-order:end -->





