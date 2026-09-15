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

## Model usage
- Always use `MyChatOpenai` (Cell 5) — never `ChatOpenAI` directly.
- `MyChatOpenai` handles o-series model quirks and the OpenAI Responses API.

## Pydantic agent output models
- Every agent output model must extend `BaseNoExtrasModel` (Cell 7).
- Required base fields: `reply_msg_to_supervisor: str`, `finished_this_task: bool`, `expect_reply: bool`.
- Use `model_config = ConfigDict(extra="forbid")` — do not add `extra="allow"`.

## State reducers
- `State` fields are annotated with custom reducers (`keep_first`, `_reduce_plan_keep_sorted`, `operator.add`).
- Never use plain field assignment for reduced fields — use the reducer semantics or add a new field.
- Adding a new State field: choose the right reducer and add it to Cell 7 in alphabetical order.

## Tool pattern (Cell 12/13)
```python
@handle_tool_errors
def my_tool(df_id: str, ...) -> tuple[str, dict]:
    validate_dataframe_exists(df_id)   # must be first for DataFrame-touching tools
    ...
    return result_message, artifact_dict
```
- Register the new tool in the correct list (`data_cleaning_tools`, `analyst_tools`, `visualization_tools`, etc.) immediately after its definition inside Cell 13.
- Tool lists are built incrementally with `.append()` / `.extend()`.

## DataFrameRegistry (Cell 8)
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
python3 _patch_notebook.py
python3 -c "import json; c=json.load(open('IntelligentDataDetective_beta_v5_patched.ipynb', encoding='utf-8'))['cells']; print(len(c),'cells')"
# Run unit tests (no API keys needed)
python3 -m pytest test_intelligent_data_detective.py -v
# Run validator smoke tests
python3 -m pytest test_validate_run.py -q
```

For completion-impacting changes, run a full patched-notebook proof with `IDD_NOTEBOOK=IntelligentDataDetective_beta_v5_patched.ipynb` and validate with `validate_run.py` plus `validate_artifact_quality.py`.
<!-- repo-agent-bootstrap:managed:end -->
