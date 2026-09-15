# IDD Regression and Drift Report

This report analyzes closed issues and tracks whether their corresponding changes have successfully made it across the three core representations of the codebase:
- `IntelligentDataDetective_beta_v5.ipynb` (Source Notebook)
- `_patch_notebook.py` (Patcher)
- `idd_core.py` (Extracted core for testing)

## Summary of Findings

Overall, there is a severe sync drift between `IntelligentDataDetective_beta_v5.ipynb` / `_patch_notebook.py` and the extracted testing script `idd_core.py`. Many critical bug fixes, memory enhancements, and architectural shifts have only been applied to the notebooks (either directly or via the patcher), rendering `idd_core.py` obsolete and missing key features.

In addition, several of the requirements from Epic #119 (Phase 6 Content-Quality Investigation) were either partially implemented or missed altogether.

## Detail: Missing Implementations and Regressions

### 1. Error Handling and Validation (Issues #20, #26)
- **Status:** **Drifted / Missing from `idd_core.py`**
- **Details:** The `handle_tool_errors` decorator and DataFrame validation logic (`validate_dataframe_exists`) were added to the source notebook and `_patch_notebook.py` (lines ~12770) but were never synced to `idd_core.py`. This means tests running against `idd_core.py` are testing code that lacks the expected robust error boundaries.

### 2. Duplicate Functions Deduplication (Issue #38)
- **Status:** **Drifted / Out of Sync**
- **Details:** The issue required removing duplicate functions, specifically calling out `perform_normality_test` and `calculate_correlation_matrix`. These have been deduplicated in the source notebook (they only appear once as a function definition). However, `perform_normality_test` is completely missing from `idd_core.py`, highlighting the extraction drift.

### 3. Memory Lifecycle and Adaptive Context (Issues #66, #72, #74)
- **Status:** **Drifted / Missing from `idd_core.py`**
- **Details:** The enhanced memory system (e.g., `retrieve_memories_with_ranking`, namespaces, and `MemoryPolicyEngine`) is correctly reflected in the notebook and `memory_enhancements.py`. However, `idd_core.py` still relies on legacy structures and doesn't contain these critical updates. The patcher was not updated to bridge this gap in the core extraction.

### 4. Phase 6/E: Fix `file_writer` Pipeline (Issue #116)
- **Status:** **Incomplete / Regression**
- **Details:** The issue explicitly required restricting `file_writer_tools` to *only* keep HTML/MD/PDF write tools to stop the LLM from spamming markers. However, looking at the patched notebook and the source notebook, `list_visualizations` and `get_visualization` are still being appended to `file_writer_tools`. The tool list was not locked down as requested.

### 5. Phase 6/F: Tighten Supervisor Completion Gate (RC4) (Issue #117)
- **Status:** **Incomplete / Regression**
- **Details:** The requirement was to replace the simple boolean flag (`report_generator_complete`) check in the `supervisor_node` (or routing logic) with content validation: checking that `report_text` length >= 1000, `viz_paths` count >= analyst recommended, and `written_sections` count >= 4. This validation logic does not exist in the source notebook, the patcher, or the compiled state graph. The application still purely relies on `bool(state.get("report_generator_complete"))`.

### 6. LangSmith Tracing Environment (Issue #125)
- **Status:** **Partially Implemented**
- **Details:** The issue asks to make the `run_notebook_live.py` load LangSmith tracing variables. The variables are read in the runner, but it was noted in the issue that the `langsmith` CLI isn't installed properly in the environment ("langsmith command not found"), which may still prevent full tracing capability unless the environment definition itself is updated.

## Conclusion and Next Steps

The primary problem is architectural drift: `idd_core.py` is falling behind the notebook evolution. To fix these regressions:
1. Re-extract `idd_core.py` from the latest `IntelligentDataDetective_beta_v5_patched.ipynb` to bring tests back in line with reality.
2. Address the incomplete Epic #119 sub-issues:
    - Enforce the restricted `file_writer_tools` constraint (Issue #116).
    - Implement the strict content validation gate for final completion (Issue #117).
