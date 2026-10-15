# IDD Regression and Drift Report

This report analyzes issue history and current status, tracking whether corresponding changes have successfully made it across the three core representations of the codebase:
- `IntelligentDataDetective_beta_v5.ipynb` (Source Notebook)
- `_patch_notebook.py` (Patcher)
- `idd_core.py` (Extracted core for testing)

## Summary of Findings

Overall, there is a severe sync drift between `IntelligentDataDetective_beta_v5.ipynb` / `_patch_notebook.py` and the extracted testing script `idd_core.py`. Many critical bug fixes, memory enhancements, and architectural shifts have only been applied to the notebooks (either directly or via the patcher), rendering `idd_core.py` obsolete and missing key features.

In addition, several of the requirements from Epic #119 (Phase 6 Content-Quality Investigation) were either partially implemented or missed altogether.

## Detail: Missing Implementations and Regressions

### 1. Error Handling and Validation (Issues #20, #26)
- **Status:** **Drifted / Out of Sync**
- **Details:** `idd_core.py` defines both `validate_dataframe_exists` and `handle_tool_errors`, but these implementations lag the patcher's hardened behavior: validation retains the legacy direct-CSV reload path, and the decorator lacks the newer bound-argument and structured-error handling. Tests importing `idd_core.py` therefore do not exercise the latest notebook behavior.

### 2. Duplicate Functions Deduplication (Issue #38)
- **Status:** **Drifted / Out of Sync**
- **Details:** The issue required removing duplicate functions, specifically calling out `perform_normality_test` and `calculate_correlation_matrix`. These have been deduplicated in the source notebook (they only appear once as a function definition). However, `perform_normality_test` is completely missing from `idd_core.py`, highlighting the extraction drift.

### 3. Memory Lifecycle and Adaptive Context (Issues #66, #72, #74)
- **Status:** **Drifted / Missing from `idd_core.py`**
- **Details:** The enhanced memory system (e.g., `retrieve_memories_with_ranking`, namespaces, and `MemoryPolicyEngine`) is correctly reflected in the notebook and `memory_enhancements.py`. However, `idd_core.py` still relies on legacy structures and doesn't contain these critical updates. The patcher was not updated to bridge this gap in the core extraction.

### 4. Phase 6/E: Fix `file_writer` Pipeline (Issue #116)
- **Status:** **Open / Incomplete**
- **Details:** `file_writer_tools` still includes general-purpose and visualization tools instead of only the HTML/Markdown/PDF writers required by Issue #116 E2. The W14 proof validates one successful artifact set, but Issue #116 has been reopened because the required tool-surface constraint remains unimplemented.

### 5. Phase 6/F: Tighten Supervisor Completion Gate (RC4) (Issue #117)
- **Status:** **Superseded / Patch-to-Output Drift**
- **Details:** The source and committed patched graph do not contain the full Issue #117 final-completion gate, and the patched `route_from_supervisor` can still fast-forward on `report_generator_complete` plus `report_results`. However, `_patch_notebook.py` does contain W11 route-readiness logic and a W13N file-writer guard for at least 1,000 content characters and four sections. Issue #117 was closed as completed/superseded by the W14 proof, so document the stale generated output and remaining gate mismatch rather than claiming validation is absent everywhere.

### 6. LangSmith Tracing Environment (Issue #125)
- **Status:** **Resolved**
- **Details:** `run_notebook_live.py` loads the LangSmith/LangChain tracing variables and verifies their presence in the child kernel without exposing secrets. Issue #125 was closed after dashboard activity and SDK trace queries were confirmed; the optional CLI may still be unavailable locally, but that does not prevent SDK-based tracing.

## Conclusion and Next Steps

The primary problem is architectural drift: `idd_core.py` is falling behind the notebook evolution. To fix these regressions:
1. Re-extract `idd_core.py` from the latest `IntelligentDataDetective_beta_v5_patched.ipynb` to bring tests back in line with reality.
2. Address the incomplete Epic #119 sub-issues:
    - Enforce the restricted `file_writer_tools` constraint (Issue #116).
    - Implement the strict content validation gate for final completion (Issue #117).
