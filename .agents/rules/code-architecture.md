# Code Architecture & Invariants Rules

This document defines the non-negotiable architectural invariants and code conventions for `intelligent_data_detective`. All agents and contributors must strictly adhere to these rules.

---

## 1. The Notebook Generation Paradigm

1. **Source vs. Runnable Notebook**:
   - `IntelligentDataDetective_beta_v5_patched.ipynb` is the committed, runnable notebook (99 cells in the W14 completion baseline).
   - `_patch_notebook.py` is the **authoritative generator and patch source**.
   - `IntelligentDataDetective_beta_v5.ipynb` is the original source input for the patcher.
2. **Never Hand-Edit the Patched Notebook**:
   - Direct edits to `IntelligentDataDetective_beta_v5_patched.ipynb` will be overwritten when `_patch_notebook.py` is run.
   - All behavioral fixes, prompt improvements, and logic additions MUST be introduced via `_patch_notebook.py` sentinel blocks.
   - After updating `_patch_notebook.py`, regenerate and verify:
     ```powershell
     python _patch_notebook.py
     python -c "import json; cells=json.load(open('IntelligentDataDetective_beta_v5_patched.ipynb', encoding='utf-8'))['cells']; print(f'{len(cells)} cells OK')"
     ```
3. **Cell Structure Preservation**:
   - Never delete or reorder notebook cells. The 99-cell structure is an architectural invariant.

---

## 2. LangGraph State Management & Channel Discipline

1. **State Reducers (Cell 7 & Cell 22)**:
   - Fields on `State` carry custom reducer annotations. Never assign directly to reduced fields; use reducer semantics:
     - `Annotated[T, keep_first]`: Immutable after first assignment.
     - `Annotated[list, operator.add]`: Appending lists across worker steps.
     - `Annotated[list, _reduce_plan_keep_sorted]`: Merging sorted plan items deduplicated by step key.
2. **Channel Collision Avoidance (W9-SR-DROP & BR-7)**:
   - **`structured_response` must NOT be a channel on supervisor `State`.**
     - In `langgraph.prebuilt.create_agent`, each agent subgraph manages its own `AgentState[ResponseT]`.
     - Colliding a supervisor custom-reducer channel with `AgentState.structured_response` causes `_resolve_schemas` set-iteration nondeterminism and `InvalidUpdateError` on concurrent writes.
     - Wrapper nodes must read structured responses from the `agent.invoke()` Python dictionary return (`result["structured_response"]`) and map fields into dedicated supervisor State keys.
   - **`State.messages` must be declared directly** as `Annotated[list[AnyMessage], add_messages]`, NEVER inherited from LangChain's `AgentState` base (which drags in the `remaining_steps` managed channel rejected in `InputSchema`).
3. **Recursion Limits**:
   - Inner agents: `recursion_limit = 160`.
   - Outer graph: `recursion_limit = 400`.
   - Do not lower these limits without explicit instruction.
4. **EMERGENCY_MSG Contract**:
   - `EMERGENCY_MSG` must have an explicit outgoing edge to `__end__` or `supervisor`. A bare-dict return without a static edge creates a dead-end.

---

## 3. Pydantic Models & Structured Output Contracts

1. **`BaseNoExtrasModel` Enforcement**:
   - Every agent output model must inherit from `BaseNoExtrasModel`.
   - `model_config = ConfigDict(extra="forbid")` is mandatory.
   - Required base fields:
     ```python
     reply_msg_to_supervisor: str
     finished_this_task: bool
     expect_reply: bool
     ```
2. **Schema-Tool Naming Contract (RC-1)**:
   - `ToolStrategy(Schema)` registers the structured-output tool with `name = Schema.__name__`, **NOT** `"respond"`.
   - Never write fallback or extraction code matching on `"respond"`.

---

## 4. DataFrame Handling & LRU Caching

1. **`DataFrameRegistry` Invariant (Cell 8)**:
   - All datasets are referenced strictly by string `df_id` (UUID string).
   - Never pass raw `pd.DataFrame` objects across nodes or tool calls.
   - The registry is a thread-safe LRU cache that auto-reloads from disk on miss.
2. **Tool Function Contract (Cell 13)**:
   - Every tool touching a DataFrame MUST:
     1. Apply `@handle_tool_errors`.
     2. Invoke `validate_dataframe_exists(df_id)` as the first executable line.
     3. Retrieve DataFrame via `registry.get_dataframe(df_id)`.
     4. Register any transformed DataFrame under a new `df_id`.

---

## 5. File System & Artifact Integrity

1. **Path Resolution (`_resolve_artifact_path`)**:
   - All file writes must pass through `_resolve_artifact_path()` for path-traversal protection.
   - Tools must never call `open()` directly with arbitrary relative paths.
2. **Zero Stub/Marker Files**:
   - Section workers and packagers must NEVER emit small `.txt` status or marker files (e.g. `status.txt`, `final_note.txt`, `final_ready_note.txt`, `stop_file.txt`).
   - Deliverables consist strictly of canonical artifacts: `final_report.html`, `final_report.md`, `final_report.pdf`, cleaned CSVs, and visualization PNGs.
3. **No `src/` Directory**:
   - All Python modules in this repository live at the root. Do not introduce a `src/` hierarchy.
