---
trigger: model_decision
description: "Validation criteria, test execution suites, notebook integrity checks, and release quality gates for code modifications and pipeline runs."
---

# Validation & Release Gates Rules

This document defines the strict validation criteria, test execution suites, and quality standards required for all code, notebook patches, and pipeline runs in `intelligent_data_detective`.

---

## 1. Offline Verification Suites (No API Keys Required)

Before submitting or committing any code modifications, all applicable offline test suites must execute cleanly. These suites align with current repository CI (`.github/workflows/copilot-setup-steps.yml`):

### Validator, Unit & Integration Suites
```powershell
python -m pytest test_validate_run.py tests/unit tests/integration -q
```
- Historical baseline context: 463 passed, 9 skipped (includes `tests/unit/test_graph_validation_safety.py` 12 tests).
- Validates agent messages, artifact path safety, BaseNoExtrasModel contracts, tool error handlers, patcher integrity, reducers, DataFrameRegistry caching, supervisor routing, and graph harness installation suppression safety.

### Core Pipeline & Memory Enhancement Suites
```powershell
python -m pytest test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py -v
```
- Historical baseline context: 77 passed (22 core + 55 memory).
- Validates DataFrame registry caching, reducer mechanics, tool error handlers, prompt formatters, Pydantic schemas, memory namespace TTLs, eviction policies, and semantic retrieval scoring.

### Prompt Template Formatting Suites
```powershell
python -m pytest test_prompt_formatting.py test_prompt_template_fixes.py -v
```
- Historical baseline context: 17 passed.
- Validates prompt bracket escaping, validator logic, and prompt template rendering.

### Notebook Integrity Suite
```powershell
python -m pytest test_validate_notebook_integrity.py -v
```
- Historical baseline context: 29 passed.
- Validates fail-closed notebook integrity checks across cell counts, real code-object compilation (ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), safe Python-bearing and non-Python cell magics handling, and diagnostic reporting.

### Anti-Drift Guard Suite
```powershell
python -m pytest tests/unit/test_core_parity_guard.py -v
```
- 22 passed.
- Validates 99 cells, canonical imports in Cells 4, 16, 19, 22, 32, 60, zero duplicate shadowing across all cells, runtime object identity (`is`), Codex Review Findings 1 & 2 behavioral parity, registry lifecycle, executable graph assembly, and fail-closed negative controls (synthetic redefinition detection and manifest tampering).

### Error Handling Framework Suite
```powershell
python -m pytest test_error_handling_framework.py -v
```
- **CI Parity Policy**: All 16 tests in this suite are strictly blocking in CI without `--deselect`.
- Signature-aware argument binding in `idd_tools.handle_tool_errors` cleanly handles parameter extraction for `df_id` and non-`df_id` functions, resolving the historical defect in `test_integration_with_different_function_signatures`. Zero failures or deselects permitted.

---

## 2. Notebook Integrity & Static Graph Verification

When modifying `_patch_notebook.py`:

```powershell
# 1. Regenerate patched notebook
python _patch_notebook.py

# 2. Fail-closed notebook structure and code-object compilation gate (exact 99 cells)
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb

# 3. Static graph reachability and syntax check (guarded against installer execution)
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
```
- **Standard**: 99 cells OK, 15 graph nodes, 0 unreachable nodes, 0 dead ends, 0 compilation errors.

---

## 3. Production Proof Twin Gates (Keyed Live Runs)

When full notebook execution is run (e.g. `python run_notebook_live.py`), the output must pass BOTH production proof validators:

### Gate 1: Execution Telemetry Gate (`validate_run.py`)
```powershell
python validate_run.py --latest --log-path notebook_run_log.txt --window 180
```
**Required Score: 12 / 12 PASS**

| Criterion | Requirement | Failure Implication |
| :--- | :--- | :--- |
| **C1** | FINAL marker in run log | Pipeline failed to finish execution |
| **C2** | Status flags: `viz=True` AND `report=True` | Graph aborted before report generation |
| **C3** | Native structured outputs (`InitialDescription`, `CleaningMetadata`, `AnalysisInsights`) | Agents fell back to unvalidated strings |
| **C4** | Full visualization fan-in (`viz_join sent_count == received_count`) | Parallel workers dropped by race condition |
| **C5** | Report orchestrator & section workers completed natively | Section outline or workers crashed |
| **C6** | Report packager completed natively with valid artifact manifest | Packaging aborted or fell back |
| **C7** | File writer manifest matches expected artifact roster | Output files missing or malformed |
| **C8** | Zero `recovered` nodes in validation window | Agent execution failed closed into recovery |
| **C9** | Zero `W2-BA-finalhop` emergency jumps | Execution bypassed intended graph edges |
| **C10** | Zero `path_normalized_missing` markers | Malformed file paths provided by agents |
| **C11** | Zero `GraphRecursionError` hits | Agent got stuck in loop before termination |
| **C12** | Zero Python `Traceback` hits in validation window | Unhandled exception occurred in node/tool |

### Gate 2: Artifact Quality & Content Gate (`validate_artifact_quality.py`)
```powershell
python validate_artifact_quality.py --latest
```
**Required Score: 9 / 9 PASS**

| Criterion | Requirement | Failure Implication |
| :--- | :--- | :--- |
| **Q1** | Standards-compliant, parseable PDF (pypdf/PyMuPDF) ≥5 pages, ≥30 KB | PDF is empty, pseudo-bytes, or title-only |
| **Q2** | HTML contains valid `<img>` tags resolving to real files | Charts broken in browser view |
| **Q3** | Markdown contains valid embedded image links resolving on disk | Visuals missing in Markdown view |
| **Q4** | At least 3 distinct visualizations with unique titles and image hashes | Cosmetic duplicate charts plotted |
| **Q5** | Non-ID-dominated visualizations | Trivial charts plotting index or row IDs |
| **Q6** | HTML text length ≥10,000 chars; includes analyst correlations & anomalies | Hollow/placeholder report deliverable |
| **Q7** | Low paragraph repetition across sections | Boilerplate repeated to inflate size |
| **Q8** | Zero stub or marker files in report directory | Looping agents left `.txt` marker debris |
| **Q9** | Canonical root artifacts exist (`final_report.html`, `.md`, `.pdf`) | Deliverables not in root discovery path |

---

## 4. Anti-Potemkin Forensics

Superficial completion without substantive analytical content is treated as a critical failure.
Always check:
1. Does the report body contain verbatim analyst correlation values ($r$) and anomaly records?
2. Are all generated visualization PNGs non-zero and distinct in content?
3. Did any agent create stub `.txt` files to simulate progress?
If any Potemkin markers are present, the run is rejected.
