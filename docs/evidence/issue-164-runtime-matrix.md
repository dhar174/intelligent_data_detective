# Issue #164 Evidence Report: Real-Kernel Runtime Verification Matrix

- **Issue:** [#164 — Real-Kernel Runtime Verification](https://github.com/dhar174/intelligent_data_detective/issues/164)
- **Branch:** `antigravity/issue-164-real-kernel-validation`
- **Starting Baseline SHA (`main` post-CP4):** `08bdc2b9e61fe67423dcc4528895bedd25c6d209`
- **Patched Notebook (`IntelligentDataDetective_beta_v5_patched.ipynb`) SHA-256:** `15bf02e69e76e572675adeedafbcc00e690838a227f4edf0289065c23e5530da` (unchanged from CP4 baseline; 99 cells)
- **Manifest (`idd_manifest.json`) SHA-256:** `f132ae6583d15b01982558a0e79dfb01e1f44d0e8e1a25e8761ad3c12388a72b`
- **Bootstrap (`idd_bootstrap.py`) SHA-256:** `09dee57af507ced6be4309ad379392675f7448c0145de7b16fbefbf0fb7e4776`

---

## 1. Execution Matrix Summary

| Scenario | Platform / Runner | Python Version | Real Kernel Identity | SQLite Dependency Mode | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linux — SQLite Present** | GitHub Actions (`ubuntu-latest`, Ubuntu 24.04 `7.0.0-1017-azure`) | `3.12.15` (`actions/setup-python@v5`) | `idd-preflight` (`ipykernel==7.4.0`, `nbclient==0.11.0`, `jupyter_client==8.10.0`) | `langgraph-checkpoint-sqlite==3.1.2` installed (`SqliteSaver` active) | **PASS** (`12 passed in 45.8s`) |
| **Linux — SQLite Absent (Package Uninstalled)** | GitHub Actions (`ubuntu-latest`, Ubuntu 24.04 `7.0.0-1017-azure`) | `3.12.15` (`actions/setup-python@v5`) | `idd-preflight` (`ipykernel==7.4.0`, `nbclient==0.11.0`, `jupyter_client==8.10.0`) | `pip uninstall -y langgraph-checkpoint-sqlite` + `IDD_EXPECT_SQLITE_UNINSTALLED=1` (`MemorySaver` fallback) | **PASS** (`1 passed in 7.1s`) |
| **Windows (Local) — SQLite Present** | `Windows-11-10.0.26340-SP0` (native AMD64) | `3.12.2` (`MSC v.1937 64 bit (AMD64)`) | `py312-codex` (`ipykernel==7.2.0`, `nbclient==0.10.4`, `jupyter_client==8.8.0`) | `langgraph-checkpoint-sqlite==3.1.2` installed (`SqliteSaver` active) | **PASS** (`15 passed in 104.73s`) |
| **Windows (Local) — SQLite Absent (Package Uninstalled)** | `Windows-11-10.0.26340-SP0` (native AMD64) | `3.12.2` (`MSC v.1937 64 bit (AMD64)`) | `py312-codex` (`ipykernel==7.2.0`, `nbclient==0.10.4`, `jupyter_client==8.8.0`) | `pip uninstall -y langgraph-checkpoint-sqlite` + `IDD_EXPECT_SQLITE_UNINSTALLED=1` (`MemorySaver` fallback) | **PASS** (`1 passed in 15.11s`) |
| **Windows (Hosted CI) — SQLite Present** | GitHub Actions (`windows-latest`, Windows Server 2022 `10.0.20348`) | `3.12.10` (`actions/setup-python@v5`) | `idd-preflight` (`ipykernel==7.4.0`, `nbclient==0.11.0`, `jupyter_client==8.10.0`) | `langgraph-checkpoint-sqlite==3.1.2` installed (`SqliteSaver` active) | **PASS** (`12 passed`) |
| **Windows (Hosted CI) — SQLite Absent (Package Uninstalled)** | GitHub Actions (`windows-latest`, Windows Server 2022 `10.0.20348`) | `3.12.10` (`actions/setup-python@v5`) | `idd-preflight` (`ipykernel==7.4.0`, `nbclient==0.11.0`, `jupyter_client==8.10.0`) | `pip uninstall -y langgraph-checkpoint-sqlite` + `IDD_EXPECT_SQLITE_UNINSTALLED=1` (`MemorySaver` fallback) | **PASS** (`1 passed`) |
| **Google Colab Clone** | Cloned repository in Google Colab runtime | Python 3.12 (Colab runtime) | Colab Jupyter kernel | Both SQLite-present & SQLite-absent supported | **NOT EXECUTED** (no live Colab runtime available in this session; reproducible runbook provided in Section 7) |

---

## 2. Environment & Package Versions

### 2.1 Local Windows 11 Native Environment
- **OS:** `Windows-11-10.0.26340-SP0`
- **Python:** `3.12.2 (tags/v3.12.2:6abddd9, Feb 6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)]`
- **Kernelspec:** `py312-codex` (`C:\Users\darf3\AppData\Roaming\jupyter\kernels\py312-codex`, launching `C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe -m ipykernel_launcher -f {connection_file}`)
- **Key Packages:**
  - `ipykernel==7.2.0`
  - `jupyter_client==8.8.0`
  - `nbclient==0.10.4`
  - `nbformat==5.10.4`
  - `langchain==1.5.7`
  - `langchain_core==1.6.12`
  - `langchain-openai==1.3.1`
  - `langchain-experimental==0.4.1`
  - `langgraph==1.1.3`
  - `langgraph-checkpoint==4.3.0`
  - `langgraph-checkpoint-sqlite==3.1.2`
  - `pydantic==2.14.1`
  - `pandas==2.2.1`
  - `numpy==1.26.4`
  - `scipy==1.13.1`
  - `scikit-learn==1.4.1.post1`
  - `matplotlib==3.8.3`
  - `seaborn==0.13.2`
  - `pytest==8.4.2`

### 2.2 GitHub Actions Hosted CI (`ubuntu-latest` & `windows-latest`)
- **Linux Runner (`ubuntu-latest`):** `Linux runnervmxehtw 7.0.0-1017-azure #17~24.04.1-Ubuntu SMP PREEMPT x86_64 GNU/Linux`, **Python `3.12.15`**
- **Windows Runner (`windows-latest`):** `Windows-2022-10.0.20348-SP0 (AMD64)`, **Python `3.12.10`**
- **Kernelspec:** `idd-preflight` (`python -m ipykernel install --user --name idd-preflight --display-name "IDD scientific smoke"`)
- **Jupyter Execution Stack:** `nbclient==0.11.0`, `nbformat==5.11.1`, `jupyter-client==8.10.0`, `ipykernel==7.4.0`
- **Key Hosted CI Packages:** `langchain==1.4.6`, `langchain-core==1.6.12`, `langchain-community==0.4.2`, `langchain-openai==1.7.2`, `langchain-experimental==0.4.2`, `langgraph==1.2.16`, `langgraph-checkpoint==4.3.0`, `langgraph-checkpoint-sqlite==3.1.2`, `pydantic==2.14.1`, `pandas==3.0.6`, `numpy==2.5.4`, `scipy==1.18.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `pytest==9.1.1`

---

## 3. Real Notebook Cells Executed & Dependencies Mocked

All cells under test in `tests/smoke/test_real_kernel_notebook.py` are extracted directly at runtime from the committed `IntelligentDataDetective_beta_v5_patched.ipynb` via `load_patched_notebook()` and executed in an isolated `ipykernel` process via `nbclient.NotebookClient`.

### 3.1 Executed Notebook Cells (`IntelligentDataDetective_beta_v5_patched.ipynb`)

| Cell Index | Role in Notebook | What Executed in the Real Kernel |
| :---: | :--- | :--- |
| **Cell 1** | Runtime toggle | Sets `use_local_llm = False`. |
| **Cell 4** | Bootstrap & Manifest Verification | Verifies `idd_bootstrap.py` against `EXPECTED_BOOTSTRAP_SHA256`, calls `bootstrap_idd_environment()`, validates all 7 canonical modules against `idd_manifest.json`, imports `idd_core`, `idd_models`, `idd_registry`, `idd_state`, `idd_tools`, `idd_graph`, and creates `_TEMP_DIRECTORY` / `WORKING_DIRECTORY`. Executed with `IDD_SKIP_NOTEBOOK_INSTALLS=1` to prevent unpinned `pip install -U` commands from mutating the provisioned scientific environment. |
| **Cell 7** | Reducers & `AgentMembers` | Imports canonical `keep_first`, `_keep_last_or_clear`, `_reduce_plan_keep_sorted`, `_reduce_strs_resettable`, `_reduce_viz_results_resettable`, `_assert_sorted_completed_no_dups`, `_sr_reducer`, `AgentId`, `Supervisor`, `AgentOrSupervisor`, `AgentMembers` from `idd_state` and defines `_ensure_working_directory()`. |
| **Cell 10** | `MyChatOpenai` | Imports canonical `MyChatOpenai` from `idd_models` and validates `prompt_cache_key` filtering behavior. |
| **Cell 16** | Canonical Models & Schemas | Imports all 18 canonical `BaseNoExtrasModel` / `BaseModel` schemas (`BaseNoExtrasModel`, `CellIdentifier`, `GetDataParams`, `UpdateDataParams`, `SortDataParams`, `ImputeDataParams`, `OutlierDataParams`, `DropDuplicatesParams`, `ConvertDataTypeParams`, `RenameColumnsParams`, `SampleDataParams`, `StringOperationsParams`, `NormalizeDataParams`, `FilterRowsParams`, `GroupByAggregateParams`, `DateTimeOperationsParams`, `PivotTableParams`, `PlanStep`, `Plan`, `Router`, etc.) from `idd_models` and defines `DataSourceConfig`. |
| **Cell 19** | `DataFrameRegistry` | Imports canonical `DataFrameRegistry`, `RegistryContext`, `get_active_registry`, `get_default_registry`, `registry_context`, `registry`, and `register_dataframe_in_registry` from `idd_registry`. |
| **Cell 22** | Canonical `State` & `CLASS_TO_AGENT` | Imports the 70-field `State` `TypedDict`, `VizWorkerState`, `CLASS_TO_AGENT`, and all agent output schemas (`InitialDescription`, `CleaningMetadata`, `AnalysisInsights`, `DataVisualization`, `VisualizationResults`, `VizSpec`, `VizFeedback`, `SectionOutline`, `Section`, `ReportOutline`, `ReportResults`, `FileResult`, `SendAgentMessage`, `NextAgentMetadata`, `AnalysisConfig`, `default_an_config`) from `idd_state`. |
| **Cell 28** | Memory Lifecycle Engine | Loads `memory_config.yaml` and initializes `MemoryPolicyEngine` (`memory_enhancements.py`). |
| **Cell 31** | Auxiliary Non-DF Tools | Defines file, workspace, and memory helper tools required by Cell 32's tool list assembly. |
| **Cell 32** | Canonical `@tool` Functions | Imports all 23 canonical `@handle_tool_errors`-decorated DataFrame tools and `_resolve_artifact_path` / `validate_dataframe_exists` from `idd_tools`, plus `FileManagementToolkit` and `PythonAstREPLTool`. |
| **Cell 46** | Supervisor Factory | Executes the real `make_supervisor_node(llms, members, system_prompt)` factory, proving it binds canonical `idd_models.Router` via `ToolStrategy(Router)` without any local `class Router` redefinition. |
| **Cell 57** | Node Wrappers, Barriers & Emergency Routing | Executes the full Cell 57 source (`initial_analysis_node`, `data_cleaner_node`, `analyst_node`, `viz_join`, `report_join`, `report_packager_node`, `file_writer_node`, and `emergency_correspondence_node`) and verifies `emergency_correspondence_node` routes via canonical `CLASS_TO_AGENT`. |
| **Cell 60** | Canonical Graph Assembly | Executes `build_graph(...)` from `idd_graph`, compiling `data_detective_graph` (15 user nodes, 33 edges, 21 conditional edges) with either `SqliteSaver` (when `langgraph-checkpoint-sqlite` is installed) or `MemorySaver` fallback (when absent). |
| **Cell 94** | SQLite Migration Demo | Executes the real Cell 94 demo against an isolated `WORKING_DIRECTORY / "checkpoints_demo.sqlite"` database (closing `_cp_conn` in `finally`) when `SqliteSaver` is available, or logs the skip notice when absent, without rebinding `data_detective_graph`. |
| **Cell 96** | SQLite Restore Demo | Executes the real Cell 96 demo into `_restored_graph` (closing `_restore_conn` in `finally`) when `SqliteSaver` is available, or logs the skip notice when absent, preserving `data_detective_graph` identity. |

### 3.2 Mocked vs. Unmocked Components
- **Unmocked (100% Real):**
  - `idd_bootstrap.py` and `idd_manifest.json` SHA-256 verification.
  - All 6 canonical Python modules (`idd_models`, `idd_registry`, `idd_state`, `idd_tools`, `idd_graph`, `idd_core`).
  - `DataFrameRegistry` and `registry_context` scoped isolation, LRU eviction, and disk reload.
  - `@handle_tool_errors`, `validate_dataframe_exists`, `_resolve_artifact_path`, and real decorated tools (`get_Summary_Statistics`, `handle_Missing_Values`, `filter_Rows`, `group_By_Aggregate`).
  - Cell 46 `make_supervisor_node` closure creation and `ToolStrategy(Router)` schema binding.
  - Cell 57 `emergency_correspondence_node`, `viz_join`, and `report_join` barrier functions.
  - Cell 60 `build_graph`, `route_from_supervisor`, `route_to_writer`, `route_to_viz_workers` (`Send("viz_worker", ...)`), `assign_workers` (`Send("report_section_worker", ...)`), and `route_after_viz_eval`.
  - Cells 94 and 96 checkpoint lifecycle and connection cleanup.
- **Mocked ( Strictly to Avoid Paid External LLM/Embedding Calls):**
  - The 8 LLM variables (`big_picture_llm`, `router_llm`, `initial_analysis_llm`, `data_cleaner_llm`, `analyst_llm`, `viz_llm`, `report_llm`, `file_writer_llm`) are injected as `MagicMock(spec=MyChatOpenai)` in a minimal setup cell immediately before Cell 60 (replacing Cells 11 and 49–55 which instantiate paid OpenAI clients).
  - For the multi-hop `data_detective_graph.invoke(...)` transition test in `test_cells_60_94_96_graph_lifecycle_checkpointer_and_transitions_in_kernel`, the LLM-invoking worker node bodies (`initial_analysis_node`, `data_cleaner_node`, `analyst_node`, `visualization_orchestrator`, `viz_worker`, `viz_evaluator_node`, `report_orchestrator`, `section_worker`, `report_packager_node`, `file_writer_node`) and `update_memory_with_kind` are replaced with deterministic no-key stubs that return valid canonical Pydantic state updates (`AnalysisInsights`, `VizSpec`, `DataVisualization`, `VizFeedback`, `SectionOutline`, `ReportOutline`, `ReportResults`, `SendAgentMessage`), while the real Cell 57 `viz_join`, `report_join`, and `emergency_correspondence_node` functions and all Cell 60 `idd_graph` edges/routers execute unmodified.

---

## 4. Stage 4 Security, Adversarial Fail-Closed Startup & Negative Controls

All 7 Stage 4 security scenarios and 3 negative controls were executed inside real `ipykernel` subprocesses (`TestRealKernelSecurityAndNegativeControls` in `tests/smoke/test_real_kernel_notebook.py`):

1. **Missing canonical file (`test_stage4_1_missing_canonical_file_fails_closed_before_later_cells`):**
   - Deletes `idd_tools.py` in an isolated sandbox repo and executes Cell 4 followed by a sentinel cell.
   - **Result:** Cell 4 raises `FileNotFoundError` before any canonical module is imported (`sys.modules` contains zero `idd_*` modules) and the downstream sentinel cell never executes.
2. **Tampered manifest digest (`test_stage4_2_tampered_manifest_digest_rejects_before_module_import`):**
   - Corrupts the `idd_models.py` SHA-256 digest in `idd_manifest.json` while planting a poison sentinel inside `idd_models.py`.
   - **Result:** Cell 4 raises `RuntimeError: Canonical module integrity check failed` before importing `idd_models.py`; neither the poison marker nor the downstream sentinel cell ever executes.
3. **Modified `idd_bootstrap.py` (`test_stage4_3_modified_idd_bootstrap_rejected_before_import`):**
   - Appends a poison sentinel to `idd_bootstrap.py` without updating Cell 4's `EXPECTED_BOOTSTRAP_SHA256`.
   - **Result:** Cell 4's pre-import SHA-256 guard raises `RuntimeError: Bootstrap script integrity check failed`, `idd_bootstrap` is absent from `sys.modules`, and the poison code in `idd_bootstrap.py` never executes.
4. **Already-imported canonical module from an unrelated checkout (`test_stage4_4_already_imported_canonical_module_from_unrelated_checkout_rejected`):**
   - Pre-imports `idd_models` from a foreign directory before running Cell 4.
   - **Result:** Cell 4 raises `RuntimeError: Canonical module 'idd_models' is already loaded from an unexpected location`, and the downstream sentinel cell never executes.
5. **Competing module locations on `sys.path` (`test_stage4_5_competing_module_locations_on_sys_path`):**
   - Pre-populates `sys.path[0]` with a competing directory containing a poison `idd_models.py`.
   - **Result:** Cell 4 promotes the verified `REPO_ROOT` to `sys.path[0]`, imports the authentic `REPO_ROOT / "idd_models.py"`, and never executes the competing poison module.
6. **Missing cloned repository in a Colab-style environment (`test_stage4_6_missing_cloned_repository_in_colab_style_environment_fails_closed`):**
   - Simulates `google.colab` in `sys.modules` inside an empty working directory lacking `idd_bootstrap.py` and `idd_manifest.json`.
   - **Result:** Cell 4 fails closed with `RuntimeError: Could not locate IDD repository root containing 'idd_bootstrap.py' and 'idd_manifest.json'`, and subsequent cells never execute.
7. **Corrupted or unexpected patcher source anchors (`test_stage4_7_corrupted_or_unexpected_patcher_source_anchors_fail_closed_in_kernel`):**
   - Executes `_patch_notebook.verify_cp4_input_cell()` inside the real kernel against tampered inputs (injectedrogue `class ForgedModel` in Cell 16 and missing `class State` anchor in Cell 22).
   - **Result:** Fails closed with `RuntimeError` on both unexpected and missing AST symbols.
8. **Deliberate In-Memory Notebook Mutation Negative Controls (`test_negative_controls_deliberately_broken_notebook_cells_fail_kernel_verification`):**
   - **Control A:** Injects a local `class Router(BaseNoExtrasModel)` shadow into Cell 46 in memory -> kernel verification fails with `AssertionError: Negative control failed: Router was shadowed in Cell 46!`.
   - **Control B:** Mutates Cell 96 in memory to rebind `data_detective_graph = _restored_graph` -> kernel verification fails with `AssertionError: Negative control failed: Cell 96 overwrote data_detective_graph!`.
   - **Control C:** Mutates Cell 16 in memory to shadow `PlanStep` -> static parity check fails with `AssertionError`.

---

## 5. Clean-Clone & CI Environment Defects Found & Resolved

1. **Missing Tracked Trajectory Fixtures in Clean Clones / Worktrees:**
   - During Stage 1 baseline verification in a fresh git worktree (`git worktree add`), running `python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb` failed in Cell 2 because `.gitignore` contained `tests/trajectory/fixtures/*.csv` and `tests/trajectory/fixtures/*.json`, preventing `tests/trajectory/fixtures/sample_dirty.csv` (220 bytes) and `tests/trajectory/fixtures/idd_test_dataset.json` (1,053 bytes) from being tracked in git.
   - **Fix:** Added `!tests/trajectory/fixtures/sample_dirty.csv` and `!tests/trajectory/fixtures/idd_test_dataset.json` exceptions to `.gitignore` and committed both small deterministic fixture files.
2. **Missing Cell 7 Top-Level Imports (`kagglehub`, `langmem`, `langchain-community`) in CI When `IDD_SKIP_NOTEBOOK_INSTALLS=1`:**
   - Executing the full unmodified Cell 7 in a real kernel on a clean GitHub Actions runner uncovered that `.github/workflows/copilot-setup-steps.yml` previously omitted `kagglehub`, `langmem`, and `langchain-community` from its dependency installation step (previous unit tests had mocked those modules or executed only selective AST slices of Cell 7).
   - **Fix:** Added `kagglehub`, `langmem`, `langchain-community`, and `langgraph-checkpoint-sqlite` to `.github/workflows/copilot-setup-steps.yml` and `.github/workflows/real-kernel-windows.yml`, and ordered `nbformat` installation prior to `validate_graph.py`.
3. **Production Pipeline Impact:** Zero changes to `_patch_notebook.py`, `IntelligentDataDetective_beta_v5_patched.ipynb`, or any canonical `idd_*.py` module.

---

## 6. Test Suite Execution Results & Skip Explanations

### 6.1 Local Windows 11 Execution Commands & Results

| Suite / Command | Result | Notes |
| :--- | :--- | :--- |
| `python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v` | **PASS** (99 cells, 42 code cells compiled) | 0 warnings, 0 errors |
| `python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb` | **PASS** (15 nodes, 33 edges, 21 conditional, 70 State fields) | Compile cell index 60 |
| `python -m pytest tests/unit/test_core_parity_guard.py -v` | **38 passed** in 10.51s | 0 skipped, 0 failed |
| `python -m pytest test_error_handling_framework.py -v` | **16 passed** in 1.21s | Mandatory 16/16 blocking gate |
| `$env:IDD_RUN_KERNEL_SMOKE="1"; $env:IDD_KERNEL_NAME="py312-codex"; python -m pytest tests/smoke/ -v` | **15 passed** in 104.73s | 12 in `test_real_kernel_notebook.py` + 3 in `test_scientific_kernel.py`; 0 skipped |
| `python -m pip uninstall -y langgraph-checkpoint-sqlite` + `$env:IDD_EXPECT_SQLITE_UNINSTALLED="1"; python -m pytest tests/smoke/test_real_kernel_notebook.py -k "sqlite_absent" -v` | **1 passed** in 15.11s | Verified genuine environment-level removal of `langgraph-checkpoint-sqlite` |
| `python -m pytest test_validate_run.py tests/unit tests/integration -q` | **689 passed, 9 skipped** in 36.16s | All 9 skips explained below |
| `python -m pytest test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_prompt_formatting.py test_prompt_template_fixes.py test_validate_notebook_integrity.py -v` | **127 passed** in 6.93s | 0 skipped, 0 failed |

### 6.2 Explanation of Every Skip in Existing Unit Suites
Zero tests are skipped in `tests/smoke/test_real_kernel_notebook.py` or `tests/smoke/test_scientific_kernel.py` when `IDD_RUN_KERNEL_SMOKE=1`. In the broader unit suite (`test_validate_run.py tests/unit tests/integration`), the 9 skips on Windows (and 3 skips on Linux) are:
1. `test_validate_run.py:29` (1 skip): `Run 88 fixture missing` — optional local archival directory check; synthetic run validator tests in the same file execute and pass.
2. `tests/unit/test_registry.py` lines `1165, 1173, 1185, 1195, 1270, 1285` (6 skips on Windows only; **0 skips on Linux**): `POSIX permission bits test` / `POSIX ownership test` — guarded by `os.name == "nt"` because Windows NTFS does not use POSIX `0o700` mode bits or `os.getuid()`. All 6 tests execute and pass on the Linux (`ubuntu-latest`) CI runner (`695 passed, 3 skipped`).
3. `tests/unit/test_registry.py` lines `109, 1461` (2 skips): `pyarrow not installed` — optional Arrow extension dtype checks when `pyarrow` is not installed; canonical pickle backing storage tests execute and pass.

### 6.3 Hosted GitHub Actions CI Runs (PR [#169](https://github.com/dhar174/intelligent_data_detective/pull/169))
- **Windows Hosted CI (`.github/workflows/real-kernel-windows.yml`, `windows-latest`, Python 3.12.10):**
  - [Run 39486968503 / Job 118849774945](https://github.com/dhar174/intelligent_data_detective/actions/runs/39486968503/job/118849774945) — **PASS** (`38 passed` anti-drift guard, `16 passed` error handling, `3 passed` scientific preflight kernel smoke, `12 passed` real-kernel notebook validation with SQLite present, `1 passed` real-kernel notebook validation with SQLite uninstalled).
- **Linux Hosted CI (`.github/workflows/copilot-setup-steps.yml`, `ubuntu-latest`, Python 3.12.15):**
  - [Run 39487032430 / Job 118850001634](https://github.com/dhar174/intelligent_data_detective/actions/runs/39487032430/job/118850001634) (and final post-format verification run on PR #169) — **PASS** (`695 passed, 3 skipped` unit/integration, `3 passed` scientific preflight kernel smoke, `12 passed` real-kernel notebook validation with SQLite present, `1 passed` real-kernel notebook validation with SQLite uninstalled, `105 passed` root regression, `17 passed` prompt templates, `16 passed` error handling, `black --check` and `flake8` clean).

---

## 7. Reproducible Google Colab Clone Validation Procedure (`NOT EXECUTED`)

> **Status:** `NOT EXECUTED` — No live Google Colab runtime was available during this engineering session. Per Issue #164 instructions, local simulation (`test_stage4_6_missing_cloned_repository_in_colab_style_environment_fails_closed`) is reported only under Stage 4 fail-closed startup verification and is **not** claimed as a substitute for genuine Google Colab execution.

To execute this validation in a genuine Google Colab runtime (Python 3.12):

```bash
# Cell 1 in Google Colab: Clone the repository and switch to the validation branch
!git clone https://github.com/dhar174/intelligent_data_detective.git /content/intelligent_data_detective
%cd /content/intelligent_data_detective
!git checkout antigravity/issue-164-real-kernel-validation

# Cell 2 in Google Colab: Install dependencies and register kernelspec
!python -m pip install -q \
  langchain langchain-core langchain-community langchain-openai langchain_experimental \
  langgraph langgraph-checkpoint-sqlite langmem kagglehub \
  pandas numpy scipy scikit-learn matplotlib seaborn pydantic \
  python-dotenv pyyaml tiktoken openpyxl tavily-python chromadb joblib typing_extensions \
  xhtml2pdf pypdf pymupdf pillow markdown bleach \
  pytest nbclient==0.11.0 nbformat==5.11.1 jupyter-client==8.10.0 ipykernel==7.4.0
!python -m ipykernel install --user --name idd-preflight --display-name "IDD scientific smoke"

# Cell 3 in Google Colab: Run real-kernel smoke validation (SQLite present)
!IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight python -m pytest tests/smoke/ -v -s

# Cell 4 in Google Colab: Run real-kernel smoke validation (SQLite absent - package uninstalled)
!python -m pip uninstall -y langgraph-checkpoint-sqlite
!IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight IDD_EXPECT_SQLITE_UNINSTALLED=1 \
  python -m pytest tests/smoke/test_real_kernel_notebook.py -k "sqlite_absent" -v -s
```

---

## 8. Remaining Limitations

1. **Google Colab Live Runtime:** Cloned-repository execution on a hosted Google Colab VM remains `NOT EXECUTED` in automated CI because GitHub Actions cannot provision proprietary Google Colab runtimes without external credentials; the reproducible 4-cell procedure in Section 7 is ready for manual execution in Colab. Standalone single-file `.ipynb` Colab acquisition without a cloned repository is out of scope for Issue #164 and tracked separately under Issue #166.
2. **No Paid LLM / Network Calls:** By design for Issue #164, LLM client instantiation Cells 11 and 49–55 and live end-to-end OpenAI inference (Cells 62+) are not executed in the no-key harness; live trajectory proof remains governed by `run_notebook_live.py`, `validate_run.py` (12/12), and `validate_artifact_quality.py` (9/9).

