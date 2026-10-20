# IDD Regression and Drift Report

This report tracks historical synchronization drift and issue resolution across the core representations of the codebase:
- `idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, `idd_graph.py`, and `idd_core.py` (Canonical Python Core Modules & Compatibility Façade)
- `_patch_notebook.py` (Authoritative Notebook Patcher)
- `IntelligentDataDetective_beta_v5_patched.ipynb` (Runnable 99-Cell W14 Patched Notebook)
- `IntelligentDataDetective_beta_v5.ipynb` (Source Notebook Input)

## Summary of Findings & CP1–CP4 Resolution

Historically (prior to Checkpoints 1–4, Issues #140, #144, #145, #162), `idd_core.py` was a static snapshot extracted from an earlier notebook version while bug fixes and architectural improvements landed only in `_patch_notebook.py`.

**Current Status (Post-CP4):** The split-brain drift between the notebook and Python modules has been **resolved**:
1. **Canonical Python Modules as Single Source of Truth:** Domain models (`idd_models.py`), `DataFrameRegistry` (`idd_registry.py`), the 70-field `State` schema and reducers (`idd_state.py`), `@handle_tool_errors` / `validate_dataframe_exists` / `_resolve_artifact_path` (`idd_tools.py`), and the 15-node, 33-edge LangGraph topology (`idd_graph.py`) are maintained in canonical modules and re-exported with 100% object identity by `idd_core.py`.
2. **Direct Notebook Wiring (CP4):** `_patch_notebook.py` wires Cells 4, 7, 10, 16, 19, 22, 32, 46, 48, 60, 94, and 96 of `IntelligentDataDetective_beta_v5_patched.ipynb` directly to the canonical modules and verifies SHA-256 module provenance at startup via `idd_bootstrap.py` and `idd_manifest.json`.
3. **Anti-Drift Enforcement:** `tests/unit/test_core_parity_guard.py` enforces 99-cell structure, zero symbol shadowing across all 99 cells, runtime object identity (`is`), and effective generated cell execution parity in CI.

## Detail: Historical Regressions and Current Status

### 1. Error Handling and Validation (Issues #20, #26, #144, #162)
- **Status:** **Resolved (CP2 & CP4)**
- **Details:** `idd_tools.py` (re-exported by `idd_core.py` and imported in Notebook Cell 32) implements signature-aware `@handle_tool_errors`, `validate_dataframe_exists(df_id)` backed by `get_global_registry()`, and pre-mkdir path-traversal validation in `_resolve_artifact_path()`. All 16 tests in `test_error_handling_framework.py` pass as a mandatory blocking CI gate.

### 2. Duplicate Functions Deduplication & Canonical Extraction (Issue #38, #140, #145)
- **Status:** **Resolved (CP1–CP4)**
- **Details:** Core models, reducers, registry helpers, tool decorators, and graph builders are deduplicated across the notebook and canonical modules. `tests/unit/test_core_parity_guard.py` enforces zero duplicate redefinitions across all 99 cells.

### 3. Memory Lifecycle and Adaptive Context (Issues #66, #72, #74)
- **Status:** **Resolved in `memory_enhancements.py` & Notebook**
- **Details:** The enhanced memory system (`retrieve_memories_with_ranking`, namespaces, and `MemoryPolicyEngine`) is maintained in `memory_enhancements.py` and wired into the patched notebook, verified by `test_memory_categorization.py`, `test_memory_integration.py`, and `test_memory_lifecycle.py`.

### 4. Phase 6/E: Fix `file_writer` Pipeline (Issue #116)
- **Status:** **Tracked (Tool-Surface Scoping)**
- **Details:** The W14 completion baseline validates deterministic HTML/Markdown/PDF artifact generation and final manifest verification (`validate_run.py` 12/12, `validate_artifact_quality.py` 9/9). Further restriction of `file_writer_tools` surface area remains tracked under Issue #116.

### 5. Phase 6/F: Tighten Supervisor Completion Gate (RC4) (Issue #117)
- **Status:** **Resolved in `idd_graph.py` & W14J Patch**
- **Details:** Both `idd_graph.py` (`route_to_writer` / `route_from_supervisor`) and Cell 60 (which imports them directly from `idd_graph`) enforce the W14J reachable completion state and artifact readiness gates.

### 6. LangSmith Tracing Environment (Issue #125)
- **Status:** **Resolved**
- **Details:** `run_notebook_live.py` loads the LangSmith/LangChain tracing variables and verifies their presence in the child kernel without exposing secrets.

## Conclusion and Maintenance Contract

1. **Never re-inline canonical logic into `_patch_notebook.py`:** Modify `idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, or `idd_graph.py`, update `idd_manifest.json` digests, and regenerate `IntelligentDataDetective_beta_v5_patched.ipynb` via `python _patch_notebook.py`.
2. **Enforce CI Parity Gates:** Keep `tests/unit/test_core_parity_guard.py`, `test_error_handling_framework.py` (16/16), `validate_notebook_integrity.py`, and `validate_graph.py` green on every change.
