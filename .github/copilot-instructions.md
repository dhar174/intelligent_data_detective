# Intelligent Data Detective - GitHub Copilot Instructions

## Build, Test, and Lint

```bash
# Run all tests
python3 -m pytest -v
# Expected: 22/22 pass in test_intelligent_data_detective.py, 15/16 in test_error_handling_framework.py (1 known edge-case failure)

# Run a single test
python3 -m pytest test_intelligent_data_detective.py::TestDataFrameRegistry::test_cache_lru_eviction -v

# Format and lint
black test_intelligent_data_detective.py test_error_handling_framework.py
flake8 test_intelligent_data_detective.py --max-line-length=88 --extend-ignore=E203,E501
```

**Full workflow execution requires API keys and takes 6–25 minutes. Never cancel it.**

```bash
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"  # Optional – enables web search

export IDD_NOTEBOOK="IntelligentDataDetective_beta_v5_patched.ipynb"
export IDD_SAMPLE_DATASET="retail_orders"
python run_notebook_live.py

python validate_run.py --latest --log-path notebook_run_log.txt --window 180
python validate_artifact_quality.py --latest
```

## Architecture

The active W14 runnable notebook is `IntelligentDataDetective_beta_v5_patched.ipynb` (99 cells). Notebook behavior changes are made in `_patch_notebook.py`, which regenerates the patched notebook from `IntelligentDataDetective_beta_v5.ipynb`; do not hand-edit the generated patched notebook. The W14 completion baseline is `IDD_run_run_default_id-20260504-1338-b3079aea`, validated by `validate_run.py` 12/12 and `validate_artifact_quality.py` 9/9. Key source areas:

| Cell | Purpose |
|------|---------|
| Area | Purpose |
|------|---------|
| `_patch_notebook.py` | Durable patch source; regenerates the patched notebook |
| `IntelligentDataDetective_beta_v5_patched.ipynb` | Committed runnable W14 notebook |
| Early notebook cells | Environment setup, imports, `MyChatOpenai`, models, `State`, `DataFrameRegistry` |
| Tool cells | All tools + `@handle_tool_errors` decorator and per-agent tool lists |
| Agent/graph cells | Agent construction, supervisor routing, LangGraph graph wiring |
| Final cells | Dataset selection, execution entrypoint, report/artifact generation |

**Agent pipeline** (supervisor-worker pattern via LangGraph):

```
Supervisor → Initial Analysis → Data Cleaner → Analyst → Visualization
          → Report Orchestrator → Section Workers → Report Packager → File Writer
```

The `State` TypedDict is the shared state object passed through every node. Agents communicate back to the supervisor via `reply_msg_to_supervisor`, `finished_this_task`, and `expect_reply` fields on their response models.

## Key Conventions

### Pydantic models for agent responses
All agent output models extend `BaseNoExtrasModel` (`model_config = ConfigDict(extra="forbid")`), which requires three base fields: `reply_msg_to_supervisor: str`, `finished_this_task: bool`, `expect_reply: bool`. Omitting any of these breaks supervisor routing.

### State reducers
The `State` TypedDict uses custom reducers instead of plain annotations. Examples:
- `Annotated[Optional[AnalysisConfig], keep_first]` – first non-None value wins; never overwritten
- `Annotated[Optional[Plan], _reduce_plan_keep_sorted]` – merge-sorted plan steps
- `Annotated[Sequence[BaseMessage], operator.add]` – messages accumulate (standard LangGraph pattern)

Do not use plain field assignments for state fields that have these reducers, or state merges will silently behave incorrectly.

### Visualization fan-in
W14H intentionally rebuilds visualization fan-in in `viz_join` from `viz_results`, `visualization_results`, `viz_paths`, and discovered PNG artifacts before evaluation. Preserve this union behavior; relying only on the last-writer `visualization_results` channel can drop parallel worker outputs.

### Tool implementation pattern
Every tool in Cell 13 follows this signature and decorator:
```python
@handle_tool_errors
def my_tool(df_id: str, ...) -> tuple[str, dict]:
    validate_dataframe_exists(df_id)  # raises on invalid
    ...
    return result_message, artifact_dict
```
`@handle_tool_errors` catches exceptions and returns a standardised error string so the agent can recover. `validate_dataframe_exists()` must be the first call in any tool that touches a DataFrame.

### Tool list construction
The per-agent tool lists (`data_cleaning_tools`, `analyst_tools`, `visualization_tools`, etc.) are defined as empty lists early in Cell 13, then populated incrementally with `.append()` / `.extend()` throughout the cell. Tools appear after their function definitions. When adding a new tool, register it in the correct list at the point of definition.

### DataFrameRegistry
`DataFrameRegistry` (Cell 8) is the single source of truth for all DataFrames. It uses `threading.RLock` internally. DataFrames are referenced everywhere by a string `df_id` (UUID or custom). Use `registry.register_dataframe(df, df_id)` to add, `registry.get_dataframe(df_id, load_if_not_exists=True)` to retrieve, which will reload from the stored CSV path on a cache miss.

### MyChatOpenai
`MyChatOpenai` (Cell 5) overrides `_get_request_payload_mod` to handle o-series model quirks and the OpenAI Responses API. Use `MyChatOpenai` everywhere in the notebook instead of `ChatOpenAI` directly.

### Memory namespaces
Memory is stored under categorised namespaces: `('memories', '<kind>')` where kind ∈ `{conversation, analysis, cleaning, visualization, insights, errors}`. TTL and per-kind limits are driven by `memory_config.yaml`, not hardcoded. Each agent is mapped to a subset of kinds in that file. When adding a new agent, add its `memory_kinds` mapping there.

### File writes
All file-writing tools call `_resolve_artifact_path()` which validates that the target is within the artifacts directory (path-traversal protection). Never bypass this by writing directly with `open()`.

### Final artifact baseline
The current completion proof expects canonical root artifacts `final_report.html`, `final_report.md`, and `final_report.pdf`, at least three distinct visualizations, no `.txt` marker/status artifacts, no tiny placeholder files, no path-normalization warnings, and no recovery/final-hop/native-failure markers in the run log.

## Known Issues

- `test_error_handling_framework.py`: one test fails due to an edge case in function-signature introspection. This is pre-existing and has no runtime impact.
- `flake8` reports some whitespace warnings on test files; these are acceptable.

<!-- repo-agent-bootstrap:file-kind=copilot-instructions -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Repository-wide Copilot instructions

This repository uses Python and LangGraph, LangChain, pytest.

When making changes:
- prefer small, focused diffs
- preserve existing architecture unless the task explicitly changes it
- run the relevant validation commands before finishing
- update `memory-bank/activeContext.md` and `memory-bank/progress.md` when project state shifts

Important references:
- `AGENTS.md`
- `docs/architecture.md`
- `memory-bank/activeContext.md`
- `memory-bank/progress.md`

Do not:
- edit generated files casually
- introduce new dependencies without justification
- remove tests to avoid fixing failures
<!-- repo-agent-bootstrap:managed:end -->
