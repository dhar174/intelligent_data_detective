# Validation & Release Gates Rules

This document defines the strict validation criteria, test execution suites, and quality standards required for all code, notebook patches, and pipeline runs in `intelligent_data_detective`.

---

## 1. Offline Verification Suites (No API Keys Required)

Before submitting or committing any code modifications, all applicable offline test suites must execute cleanly.

### Core Pipeline Unit Suite
```powershell
python -m pytest test_intelligent_data_detective.py -v
```
- **Standard**: 22 / 22 PASS.
- Validates DataFrame registry caching, reducer mechanics, tool error handlers, prompt formatters, and Pydantic schemas.

### Validator Unit Suite
```powershell
python -m pytest test_validate_run.py -q
```
- **Standard**: 8 / 8 PASS.
- Validates the rule logic of `validate_run.py` against known passing and failing synthetic log traces.

### Error Handling Framework Suite
```powershell
python -m pytest test_error_handling_framework.py -v
```
- **Standard**: 15 / 16 PASS (1 known edge-case failure acceptable per repository baseline).
- Validates decorator failure boundaries, retry policies, and fallback mechanics.

### Memory Lifecycle & Categorization Suite
```powershell
python -m pytest test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_adaptive_retrieval.py -v
```
- Validates memory namespace TTLs, eviction policies, and semantic retrieval scoring.

---

## 2. Notebook Integrity & Static Graph Verification

When modifying `_patch_notebook.py`:

```powershell
# 1. Regenerate patched notebook
python _patch_notebook.py

# 2. Verify cell count invariant (99 cells)
python -c "import json; cells=json.load(open('IntelligentDataDetective_beta_v5_patched.ipynb', encoding='utf-8'))['cells']; print(f'{len(cells)} cells OK')"

# 3. Static graph reachability and syntax check
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
