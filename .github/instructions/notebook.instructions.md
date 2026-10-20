---
applyTo: "**/*.ipynb"
---

<!-- repo-agent-bootstrap:file-kind=path-instructions -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Notebook editing instructions

These rules apply to notebook behavior changes for IDD v5.

Current workflow:
- Edit `_patch_notebook.py`.
- Regenerate `IntelligentDataDetective_beta_v5_patched.ipynb`.
- Run validation against the patched notebook.
- Do not hand-edit `IntelligentDataDetective_beta_v5_patched.ipynb`; it is a generated runnable artifact.

## Cell integrity
- **Never delete or reorder existing source-notebook cells.** The cell map is load-order dependent; renumbering breaks imports.
- The W14 patched notebook currently has 99 cells after regeneration. Confirm JSON validity and cell count after patcher changes.

## Canonical Python modules alignment
Under CP4, the production notebook imports its core components directly from canonical Python modules:
- **Cell 4**: `idd_bootstrap` (`bootstrap_environment()` - provenance check, environment, and manifest validation)
- **Cell 16**: `idd_models` (`BaseNoExtrasModel`, `Plan`, schemas) and `idd_state.AnalysisConfig`
- **Cell 19**: `idd_registry` (`DataFrameRegistry`, `global_df_registry = get_global_registry()`, `get_global_df_registry`)
- **Cell 22**: `idd_state` (`State`, `VizWorkerState`, and all 15 reducers)
- **Cell 32**: `idd_tools` (`handle_tool_errors`, `validate_dataframe_exists`, `_resolve_artifact_path`)
- **Cell 60**: `idd_graph` (`build_graph(..., structural_only=False)`)

**Zero duplicate redefinitions**: Notebook cells must never duplicate or shadow classes, functions, or schemas provided by canonical modules.

## Model usage
- Always use `MyChatOpenai` (Cell 5) — never `ChatOpenAI` directly.
- `MyChatOpenai` handles o-series model quirks and the OpenAI Responses API.

## Pydantic agent output models
- Every agent output model must extend `BaseNoExtrasModel` (Cell 16, imported from `idd_models`).
- Required base fields: `reply_msg_to_supervisor: str`, `finished_this_task: bool`, `expect_reply: bool`.
- Use `model_config = ConfigDict(extra="forbid")` — do not add `extra="allow"`.

## State reducers
- `State` fields are defined in `idd_state.State` (Cell 22) with canonical reducers (`keep_first`, `_reduce_plan_keep_sorted`, `operator.add`).
- Never use plain field assignment for reduced fields — use the reducer semantics or add a new field in `idd_state.py`.

## Tool pattern (Cell 32)
```python
@handle_tool_errors
def my_tool(df_id: str, ...) -> tuple[str, dict]:
    validate_dataframe_exists(df_id)   # must be first for DataFrame-touching tools
    ...
    return result_message, artifact_dict
```
- Tool error handling and artifact resolution are imported from `idd_tools` (`handle_tool_errors`, `validate_dataframe_exists`, `_resolve_artifact_path`).

## DataFrameRegistry (Cell 19)
- All DataFrames are stored by string `df_id` (UUID or custom).
- Register: `registry.register_dataframe(df, df_id)`.
- Retrieve: `registry.get_dataframe(df_id, load_if_not_exists=True)` — reloads from CSV on cache miss.
- Never bypass the registry by passing DataFrames directly between functions.

## Memory namespaces
- Namespaces: `('memories', '<kind>')`, kind ∈ `{conversation, analysis, cleaning, visualization, insights, errors}`.
- TTL and per-kind limits come from `memory_config.yaml` — do not hardcode them.
- When adding a new agent, add its `memory_kinds` mapping in `memory_config.yaml`.

## File writes
- All file-writing tools must call `_resolve_artifact_path()` for path-traversal protection.
- Never write directly with `open()` in tool code.

## Validation after any notebook edit
```bash
# Regenerate and confirm patched notebook validity
python _patch_notebook.py
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb

# Anti-drift guard & patcher integrity
python -m pytest tests/unit/test_core_parity_guard.py -v
python -m pytest tests/unit/test_patcher_integrity.py -q

# Error handling framework (all 16 tests must pass; no deselect)
python -m pytest test_error_handling_framework.py -v

# Core tests & validators
python -m pytest test_intelligent_data_detective.py -v
python -m pytest test_validate_run.py -q
```

For completion-impacting changes, run a full patched-notebook proof with `IDD_NOTEBOOK=IntelligentDataDetective_beta_v5_patched.ipynb` and validate with `validate_run.py` plus `validate_artifact_quality.py`.
<!-- repo-agent-bootstrap:managed:end -->
