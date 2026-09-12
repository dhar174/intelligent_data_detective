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
