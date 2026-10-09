# IDD Core Parity Inventory & Effective-Runtime Baseline

**Document**: `docs/architecture/idd-core-parity-inventory.md`  
**Purpose**: Architectural drift inventory, symbol-level runtime bindings, and empirical baseline evidence for Issue #150 / #140  
**Status**: DRAFT FOR CP0 REVIEW (Execution stopped for review gate)  
**Date**: October 2026  
**Auditor / Steward**: IDD Repo Coordinator & Subagent Specialist Team (`notebook-patch-specialist`, `pydantic-contract-guardian`, `langgraph-topology-architect`)

---

## 1. Pinned Starting Baseline & Environment

### 1.1 Git Provenance
- **Repository**: `dhar174/intelligent_data_detective`
- **Starting HEAD Commit**: `f28a2b0479ad5dd729d00f3a46aacece15b2554b` (`Merge pull request #154 from dhar174/antigravity/issue-152-preflight`)
- **Remote `origin/main`**: `f28a2b0479ad5dd729d00f3a46aacece15b2554b`
- **Execution Branch**: `antigravity/issue-150-core-parity-baseline`
- **Historical Planning Reference**: Planning snapshot `f4a749e25a34b95d6c4bd02e397ac793a732d95b` (October 6, 2026) has advanced cleanly to `f28a2b0` via PR #154. Working tree is clean.

### 1.2 Python Runtime & Installed Dependencies
- **Interpreter**: Python 3.12.2 (tags/v3.12.2:6abddd9, MSC v.1937 64-bit AMD64) on Windows 11
- **LangChain / LangGraph Stack**:
  - `langchain`: `1.4.3`
  - `langchain-core`: `1.6.7`
  - `langchain-openai`: `1.6.7`
  - `langchain-experimental`: `0.4.2`
  - `langgraph`: `1.2.14`
- **Scientific Stack**:
  - `pandas`: `2.2.1`
  - `numpy`: `1.26.4`
  - `scipy`: `1.13.1`
  - `scikit-learn`: `1.9.1`
  - `matplotlib`: `3.8.3`
  - `seaborn`: `0.13.2`
- **Data & Serialization**:
  - `pydantic`: `2.13.5`
  - `openpyxl`: `3.1.5`
  - `chromadb`: `1.5.9`
  - `joblib`: `1.6.0`
  - `tiktoken`: `0.14.0`
- **Validation & Tooling**:
  - `pytest`: `8.4.2`
  - `bleach`: `6.4.0`
  - `nbclient`: `0.10.4`
  - `nbformat`: `5.10.4`
  - `ipykernel`: `7.2.0`

### 1.3 Measured No-Key Test Baseline (Actual Execution)
All suites executed directly against `f28a2b0`:

| Test Suite / Command | Exit Code | Result | Details & Skips |
| :--- | :---: | :--- | :--- |
| `python -m pytest test_validate_run.py tests/unit tests/integration -q -rs` | 0 | **463 passed, 9 skipped** (28.20s) | 7 skips due directly to missing `idd_core` symbols (`build_graph`, `AGENT_MEMBERS`, `options`, `next`); 1 skip due to missing Run 88 fixture; 1 skip due to optional `pyarrow`. |
| `python -m pytest test_prompt_formatting.py test_prompt_template_fixes.py -q` | 0 | **17 passed** (3.01s) | All prompt formatting and placeholder tests green. |
| `python -m pytest test_intelligent_data_detective.py -q` | 0 | **22 passed** (1.00s) | Core 22 tests green. |
| `python -m pytest test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py -q` | 0 | **88 passed** (4.45s) | Memory lifecycle and notebook integrity validator green. |
| `python -m pytest test_error_handling_framework.py -q` | 1 | **15 passed, 1 failed** (1.54s) | 1 failure is known isolated edge-case `test_integration_with_different_function_signatures` (reproduced below). With `--deselect`, 15/15 pass. |
| `python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v` | 0 | **99/99 cells validated** (42 code cells compiled) | Zero compilation errors; valid structure. |
| `python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb` | 0 | **Compiled OK** (15 nodes, 33 edges) | 0 unreachable nodes; 0 dead-end nodes. |

### 1.4 Notebook Regeneration Determinism Check
Repeated regeneration was measured using `python _patch_notebook.py`:
- Committed baseline SHA256 (`IntelligentDataDetective_beta_v5_patched.ipynb`):  
  `AB72B7C4E83AF1F5C1F34B71DE4D542204E84B2702DCEE2902AB86D48460679C`
- Pass 1 Regeneration SHA256:  
  `AB72B7C4E83AF1F5C1F34B71DE4D542204E84B2702DCEE2902AB86D48460679C`
- Pass 2 Regeneration SHA256:  
  `AB72B7C4E83AF1F5C1F34B71DE4D542204E84B2702DCEE2902AB86D48460679C`
- Git working copy drift: **0 bytes (clean)**. Exactly 99 cells preserved.

---

## 2. Inventory of Maintained & Generated Representations

The repository currently maintains or generates five distinct code representations of IDD runtime logic:

| Representation File | Kind & Size | Role & Execution Status | Maintenance Channel |
| :--- | :--- | :--- | :--- |
| **`idd_core.py`** | Standalone Python module (52,417 bytes, 1,226 lines) | **Unit test substrate**. Sanitized pure-Python extract created so `tests/unit` and `tests/integration` can run without notebook execution. Deliberately omits `State` and `build_graph`. Contains desynchronized bug fixes. | Hand-edited; partially desynchronized from notebook. |
| **`IntelligentDataDetective_beta_v5.ipynb`** | Source Jupyter Notebook (1,851,765 bytes, 98 cells) | **Original authoring source**. Unpatched raw notebook containing legacy syntax, historical markdown notes, and base cells. | Direct authoring target before W14. |
| **`_patch_notebook.py`** | Notebook Patcher Engine (772,603 bytes, 14,846 lines) | **Authoritative compiler & generator**. Ingests `IntelligentDataDetective_beta_v5.ipynb`, applies ~70 surgical patch blocks (sentinels), and emits the 99-cell runnable patched notebook. | Authoritative patch source for all notebook fixes. |
| **`IntelligentDataDetective_beta_v5_patched.ipynb`** | Committed Runnable Notebook (1,792,945 bytes, 99 cells) | **Production execution target**. Runnable end-to-end multi-agent pipeline used by `run_notebook_live.py`, `validate_run.py`, and `validate_artifact_quality.py`. | Generated exclusively via `_patch_notebook.py`. |
| **`intelligentdatadetective_beta_v5.py`** | Textual Export Script (950,439 bytes, 19,088 lines) | **Non-runnable text artifact**. Direct `nbconvert` text dump of the notebook. Retains IPython shell magics (e.g. `!pip show` at line 688) and late `__future__` imports. **Not an importable module**. | Re-exported artifact; selected snippets compiled in tests. |

---

## 3. Comprehensive Parity & Symbol Ownership Matrix

Every overlapping symbol across the 5 representations has been audited for locations, effective runtime binding, consumers, and proposed canonical owner:

### 3.1 Pydantic Models & Data Contracts

| Symbol | Locations & Line Anchors | Effective Runtime Binding | Consumers | Differences & Drift | Proposed Canonical Owner | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`BaseNoExtrasModel`** | `idd_core.py:390`<br>`v5.py:698`<br>Patched NB: Cell 16:L2 | `BaseNoExtrasModel`<br>(`extra="forbid"`) | All agent output models | Consistent base contract: requires `reply_msg_to_supervisor`, `finished_this_task`, `expect_reply`. | `idd_runtime.models` | Shared Production Logic |
| **`PlanStep`** | `idd_core.py:605`<br>`v5.py:925`<br>Patched NB: Cell 16:L229 | `PlanStep`<br>(extends `BaseNoExtrasModel`) | `Plan`, `CompletedStepsAndTasks` | Identical fields (`step_number`, `step_name`, `step_description`, `is_step_complete`, `plan_version`). Note: Informal docs refer to this as "Step", but no `Step` class exists. | `idd_runtime.models` | Shared Production Logic |
| **`Plan`** | `idd_core.py:613`<br>`v5.py:931`<br>Patched NB: Cell 16:L235 | `Plan`<br>(with `_counter` / `_next` lock) | `_reduce_plan_keep_sorted`, `CompletedStepsAndTasks` | **Drift**: `idd_core` uses `_counter: ClassVar[itertools.count]`; notebook uses `_next = itertools.count(1).__next__`. Both overwrite user `plan_version` on instantiation. | `idd_runtime.models` | Shared Production Logic |
| **`CompletedStepsAndTasks`** | `idd_core.py:655`<br>`v5.py:970`<br>Patched NB: Cell 16:L285 | `CompletedStepsAndTasks` | Supervisor, progress accounting | **CRITICAL BUG IN NOTEBOOK (RC-2)**: `idd_core.py:686` returns sorted `dedup_list`. Notebook Cell 16:L310 sorts `dedup_list` but returns unsorted `list(seen.values())`, crashing if input is unsorted! | `idd_runtime.models` | Shared Production Logic |
| **`CleaningMetadata`** | `idd_core.py:436`<br>`v5.py:734`<br>Patched NB: Cell 16:L42 | `CleaningMetadata` | `data_cleaner_node`, `supervisor` | Synced. Formerly claimed missing in #140/#145, but present in current `idd_core.py`. | `idd_runtime.models` | Shared Production Logic |
| **`AnalysisInsights`** | `idd_core.py:467`<br>`v5.py:765`<br>Patched NB: Cell 16:L73 | `AnalysisInsights` | `analyst_node`, `visualization` | Synced. Contains `recommended_visualizations: List[VizSpec]`. | `idd_runtime.models` | Shared Production Logic |
| **`VizSpec`** | `idd_core.py:449`<br>`v5.py:747`<br>Patched NB: Cell 16:L55 | `VizSpec` | `AnalysisInsights`, `viz_worker` | Synced. Inherits `BaseNoExtrasModel`. | `idd_runtime.models` | Shared Production Logic |
| **`SectionOutline`** | `idd_core.py:742`<br>`v5.py:1085`<br>Patched NB: Cell 16:L372 | `SectionOutline` | `report_orchestrator`, `section_worker` | Synced. | `idd_runtime.models` | Shared Production Logic |
| **`Section`** | `idd_core.py:732`<br>`v5.py:1075`<br>Patched NB: Cell 16:L362 | `Section` | `report_section_worker`, `report_join` | Synced. Requires `min_length=100` on content. | `idd_runtime.models` | Shared Production Logic |
| **`ReportOutline`** | `idd_core.py:753`<br>`v5.py:1096`<br>Patched NB: Cell 16:L383 | `ReportOutline` | `report_orchestrator`, `report_packager` | Synced. Subclasses `SectionOutline`. | `idd_runtime.models` | Shared Production Logic |
| **`ReportResults`** | `idd_core.py:513`<br>`v5.py:811`<br>Patched NB: Cell 16:L119 | `ReportResults` | `report_packager`, `file_writer` | Synced. Paths for PDF, HTML, Markdown. | `idd_runtime.models` | Shared Production Logic |
| **`ListOfFiles`** | `idd_core.py:547`<br>`v5.py:845`<br>Patched NB: Cell 16:L153 | `ListOfFiles` | `file_writer_node`, final manifest | Synced. | `idd_runtime.models` | Shared Production Logic |
| **`Router`** | Patched NB: Cell 46:L1072<br>`v5.py:12260` | Function-local in `make_supervisor_node` | Supervisor LLM structured output | **MISSING from `idd_core.py`**. Trapped in closure. Must be promoted to top-level model. | `idd_runtime.models` | Shared Production Logic |

### 3.2 State Schemas & Reducers

| Symbol | Locations & Line Anchors | Effective Runtime Binding | Consumers | Differences & Drift | Proposed Canonical Owner | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`State`** | Patched NB: Cell 22:L70<br>`v5.py:1380`<br>`idd_core.py:1037` (omitted) | 70-field TypedDict with reducers | All 15 graph nodes, supervisor | **MISSING from `idd_core.py`** by explicit design comment ("incompatible with unit test imports"). Enforces W9-SR-DROP and BR-7. | `idd_runtime.state` | Shared Production Logic |
| **`_reduce_plan_keep_sorted`** | `idd_core.py:1022`<br>Patched NB: Cell 22:L32 | Reducer merging plans by step number | `State.current_plan` | Synced logic. Combines steps and deduplicates by `step_number` (last-wins). | `idd_runtime.reducers` | Shared Production Logic |
| **`keep_first`** | `idd_core.py:87`<br>Patched NB: Cell 7:L240 | Reducer preserving first non-None | State path channels (`artifacts_path`, etc.) | Synced. | `idd_runtime.reducers` | Shared Production Logic |
| **`_sr_reducer`** | Patched NB: Cell 22:L50<br>`idd_core.py` (absent) | Last-write-wins prefer non-None | `State.report_results` | Exists in notebook Cell 22 (W2-BR8c); absent from `idd_core.py`. | `idd_runtime.reducers` | Shared Production Logic |
| **`VizWorkerState`** | Patched NB: Cell 22:L150<br>`idd_core.py` (absent) | TypedDict for `Send("viz_worker")` | `viz_worker` fan-out | Missing from `idd_core.py`. | `idd_runtime.state` | Shared Production Logic |

### 3.3 DataFrame Management & Tool Decorators

| Symbol | Locations & Line Anchors | Effective Runtime Binding | Consumers | Differences & Drift | Proposed Canonical Owner | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`DataFrameRegistry`** | `idd_core.py:865`<br>Patched NB: Cell 19:L3 | Thread-safe LRU registry | All data-touching tools | **Drift**: Notebook Cell 19 includes `_read_df` supporting `.csv`, `.parquet`, `.pkl`, `.json`; `idd_core.py` has older reload logic. | `idd_runtime.registry` | Shared Production Logic |
| **`validate_dataframe_exists`** | `idd_core.py:1144`<br>Patched NB: Cell 32:L28 | Checks registry & reloads | All data-touching tools | **Drift**: `idd_core.py:1157` hardcodes `pd.read_csv`, failing on pickle cache misses. Notebook delegates to registry. | `idd_runtime.tools` | Shared Production Logic |
| **`handle_tool_errors`** | `idd_core.py:1169`<br>Patched NB: Cell 32:L67 | Error handling decorator | All tool functions | **CRITICAL DRIFT**: `idd_core` returns raw string `f"Error: {e}"`. Patched notebook returns structured dict `_tool_error(operation, reason, action)`. `tests/unit/test_handle_tool_errors.py` tests string! | `idd_runtime.tools` | Shared Production Logic |
| **`_resolve_artifact_path`** | `idd_core.py:1105`<br>Patched NB: Cell 32:L647 & Cell 57:L1000 | Path containment & env precedence | Visualization, File Writer | **Drift**: `idd_core` uses basic `_is_subpath`. Patched notebook (PR #149) enforces multi-root allowed roots (`_allowed_roots`, `_fw_allowed_roots`) and canonical report paths. | `idd_runtime.artifacts` | Shared Production Logic |
| **`delete_rows` & `_build_query_view`** | Patched NB: Cell 32:L269, L300<br>`v5.py:3820`<br>`idd_core.py` (absent) | Tool with pandas 3.0.6 collision guard | Data cleaner agent | Missing from `idd_core.py`. `_build_query_view` fixes Issue #147. | `idd_runtime.tools` | Shared Production Logic |

### 3.4 Graph Topology & Routing

| Symbol | Locations & Line Anchors | Effective Runtime Binding | Consumers | Differences & Drift | Proposed Canonical Owner | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`build_graph`** | Patched NB: Cell 60:L11<br>`idd_core.py` (absent) | Compiles 15-node StateGraph | Integration tests, live runner | **MISSING from `idd_core.py`**. Causes 4 tests in `tests/integration/test_graph_compile.py` to skip! | `idd_runtime.graph` | Shared Production Logic |
| **`AGENT_MEMBERS` / `AgentMembers`** | `idd_core.py:347` (`AgentMembers` class)<br>Patched NB: Cell 46:L10 | List of agent strings / literals | Supervisor routing, routers | `idd_core.py` defines class `AgentMembers` without `AGENT_MEMBERS` constant, causing `tests/integration/test_routing.py` skips. | `idd_runtime.constants` | Shared Production Logic |
| **`make_supervisor_node`** | Patched NB: Cell 46:L54<br>`idd_core.py` (absent) | Supervisor node factory | LangGraph supervisor node | Missing from `idd_core.py`. | `idd_runtime.supervisor` | Shared Production Logic |
| **`initial_analysis_node`, `data_cleaner_node`, etc. (11 wrapper nodes)** | Patched NB: Cell 57<br>`idd_core.py` (absent) | Safe invoke wrapper nodes | LangGraph graph assembly | 11 node wrappers and 3 join barriers exist only in the notebook. Missing from `idd_core.py`. | `idd_runtime.nodes` | Shared Production Logic |

---

## 4. Empirical Drift Reproduction Evidence

All 12 claims were tested directly using `python tools/diagnostics/reproduce_drift.py`. Below is the exact observed evidence:

### Claim 1: Plan Counter Sharing & Concurrency
- **Implementation Locator**: `idd_core.py:619-642` (`Plan._counter: ClassVar[itertools.count]`) vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 16:L245` (`Plan._next = itertools.count(1).__next__`).
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_1)
- **Expected**: Each independent plan has an independent version unless explicitly linked.
- **Observed**: `Plan 1 version: 1, Plan 2 version: 2`. Across 5 concurrent threads, all generated versions were strictly unique (`[3, 4, 5, 6, 7]`).
- **Status**: **REPRODUCED**. Plan versions are driven by a class-level in-memory counter. Instantiating any Plan increments the global counter across instances and threads.

### Claim 2: Plan-Version Synchronization Overwrites Explicit Input
- **Implementation Locator**: `idd_core.py:635-642` (`_sync_steps_and_assert_increasing`)
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_2)
- **Expected**: `Plan(plan_version=42, ...)` preserves version 42.
- **Observed**: Requested `plan_version=42`; Resulting `plan_version=8`.
- **Status**: **REPRODUCED**. `self._ver_assigned` defaults to `False` on instantiation, causing the after-validator to unconditionally overwrite the caller's explicit version with the class counter.

### Claim 3: Completed-Step Sorting Divergence (Split-Brain Fix RC-2)
- **Implementation Locators**:  
  - Fixed: `idd_core.py:5-6, 683-686` (`return dedup_list`)
  - Buggy: `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 16:L305-L310` (`return list(seen.values())`)
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_3)
- **Expected**: If steps `[3, 1, 2]` are passed to `CompletedStepsAndTasks`, they should either both sort or both reject.
- **Observed**: `idd_core.py` auto-sorts to `[1, 2, 3]`. The patched notebook sorts local variable `dedup_list` but discards it, returning `list(seen.values())` in original insertion order `[3, 1, 2]`, which crashes `_assert_sorted_completed_no_dups` with `ValueError: completed_steps must be sorted ascending by step_number`.
- **Status**: **REPRODUCED**. Severe split-brain bug: the fix was applied to `idd_core.py` and textual export `v5.py`, but never added to `_patch_notebook.py`.

### Claim 4: Duplicate Step Numbers Rejection
- **Implementation Locator**: `idd_core.py:630-633, 649-652`
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_4)
- **Expected**: `Plan` rejects duplicate step numbers.
- **Observed**: `Plan` with steps `[1, 1]` raised `ValidationError: plan_steps must be strictly increasing by step_number, got [1, 1]`.
- **Status**: **REPRODUCED**. `Plan` strictly rejects duplicate step numbers via increasing sequence check.

### Claim 5: Validation Context / Subset Behavior
- **Implementation Locator**: `idd_core.py:665-667, 695-702`
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_5)
- **Expected**: When validation context `{"plan": plan}` is provided, steps outside the plan are rejected.
- **Observed**: Passing an unplanned step with context raised `ValueError: Completed step ... is not present in the supplied Plan`. Without context, the subset check is bypassed.
- **Status**: **REPRODUCED**. Validation context enforces plan subset integrity.

### Claim 6: Structured Tool Errors Divergence
- **Implementation Locators**:  
  - `idd_core.py:1169-1215` (returns raw string)
  - `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 32:L7-L140` (returns structured dict)
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_6)
- **Expected**: Identical error contracts across tests and runtime.
- **Observed**: `idd_core` returns `str` (`"Error: Column or key ''missing_col'' not found"`). The notebook returns structured dictionary (`{"status": "error", "operation": "failing_core_tool", "reason": "...", "action": "..."}`). `tests/unit/test_handle_tool_errors.py` explicitly tests string returns!
- **Status**: **REPRODUCED**. Critical fracture between test suite and production runtime.

### Claim 7: Signature-Aware `df_id` Extraction
- **Implementation Locator**: `idd_core.py:1176-1186`
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_7)
- **Expected**: Tool with signature `func(operation_name: str, df_id: str)` extracts `df_id` correctly.
- **Observed**: `custom_tool("clean", "valid_df_id")` fails with `Error: DataFrame with ID 'clean' not found or is invalid`. `args[0]` was blindly assumed to be `df_id`.
- **Status**: **REPRODUCED**. This is the exact root cause of the non-blocking failure in `test_error_handling_framework.py:347`.

### Claim 8: Registry Reload Formats (Non-CSV Cache Miss)
- **Implementation Locators**:  
  - `idd_core.py:1154-1162` (hardcoded `pd.read_csv`)
  - `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 19:L80-L100` (`_read_df` multi-format)
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_8)
- **Expected**: Pickle/parquet cache miss is reloaded from disk.
- **Observed**: `validate_dataframe_exists("pkl_df")` returned `False` in `idd_core` because `pd.read_csv` failed on pickle. Notebook Cell 19 supports `.pkl`, `.parquet`, `.json`, `.csv`.
- **Status**: **REPRODUCED**. `idd_core` lacks multi-format reload capability.

### Claim 9: Artifact-Root Precedence & Strict Containment
- **Implementation Locators**:  
  - `idd_core.py:1075-1132` (legacy single-base path resolution)
  - `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 57:L1000-L1500` (PR #149 strict containment)
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_9)
- **Expected**: Strict containment against allowed roots list (`_allowed_roots`, `_fw_allowed_roots`).
- **Observed**: Notebook Cell 57 implements PR #149 multi-root containment (`_artifact_root`, `_working_dir`, `_run_root`), whereas `idd_core.py` only resolves against a single base directory.
- **Status**: **REPRODUCED**. Path resolution in `idd_core.py` lacks PR #149 security hardening.

### Claim 10: LLM-Adapter Identity
- **Implementation Locators**: `idd_core.py:38-70` vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 46, 48`.
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_10)
- **Expected**: Uniform model adapter contracts.
- **Observed**: `idd_core.py` defines `MyChatOpenai`, while notebook activeContext (2026-04-23) mandates using `ChatOpenAI` directly.
- **Status**: **REPRODUCED**. Divergent model adapter conventions.

### Claim 11: Skipped Production State / Routing / Graph Checks
- **Implementation Locators**:  
  - `tests/integration/test_graph_compile.py:54, 58, 64, 68`
  - `tests/integration/test_routing.py:43, 52, 68`
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_11)
- **Expected**: Integration tests compile the real production graph and verify routes.
- **Observed**: `idd_core.build_graph` is False; `idd_core.AGENT_MEMBERS` is False. 7 integration tests skip silently!
- **Status**: **REPRODUCED**. Integration test suite is hollow because `idd_core` does not expose graph construction.

### Claim 12: Integer/String Column-Label Regression (#147) vs PR #149
- **Implementation Locator**: `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 32:L269-L295` (`_build_query_view`)
- **Command**: `python tools/diagnostics/reproduce_drift.py` (run_claim_12)
- **Expected**: Check whether querying integer column `0` with backticks raises `UndefinedVariableError`.
- **Observed**: Query `qv.query('`0` >= 20')` on DataFrame `{0: [10, 20, 30], "name": ["a", "b", "c"]}` succeeded cleanly, returning rows 1 and 2.
- **Status**: **NOT REPRODUCED (FIXED ON MAIN)**. PR #149's `_build_query_view` projection correctly aliased integer column `0` to string `'0'`, resolving the regression. Issue #147 is administratively open on GitHub, but verified fixed in current production code.

---

## 5. Summary of Confirmed vs Disproven Claims

| Claim ID | Claim Description | Status | Verdict & Notes |
| :--- | :--- | :---: | :--- |
| **Claim 1** | Plan counter shared & thread-safe | **CONFIRMED** | Uses class-level `itertools.count(1)` with `threading.Lock()`. |
| **Claim 2** | Plan-version synchronization overwrites input | **CONFIRMED** | `_ver_assigned=False` overwrites caller `plan_version`. |
| **Claim 3** | Completed-step sorting divergence | **CONFIRMED** | Fatal split-brain: `idd_core` fixed; notebook Cell 16 retains unsorted bug. |
| **Claim 4** | Duplicate step numbers rejected | **CONFIRMED** | `Plan` strictly rejects duplicate step numbers. |
| **Claim 5** | Validation context enforces plan subset | **CONFIRMED** | Ghost steps rejected when `{"plan": plan}` context is provided. |
| **Claim 6** | Structured tool error contract divergence | **CONFIRMED** | `idd_core` returns string; notebook Cell 32 returns structured dict. |
| **Claim 7** | `args[0]` blindly assumed to be `df_id` | **CONFIRMED** | Breaks multi-argument tools where `df_id` is not first. |
| **Claim 8** | Registry reload fails on non-CSV | **CONFIRMED** | `idd_core` hardcodes `pd.read_csv`; notebook has multi-format reader. |
| **Claim 9** | Artifact-root containment drift | **CONFIRMED** | PR #149 multi-root containment present only in notebook Cell 57. |
| **Claim 10** | LLM adapter identity drift | **CONFIRMED** | `MyChatOpenai` legacy wrapper vs direct `ChatOpenAI`. |
| **Claim 11** | Integration tests skip real graph checks | **CONFIRMED** | 7 tests skip due to missing `build_graph` and `AGENT_MEMBERS`. |
| **Claim 12** | Integer column query fails (#147) | **DISPROVEN / FIXED** | Proven fixed by PR #149's `_build_query_view`. Administratively open only. |
| **Claim 13** | Core pipeline models missing from `idd_core` | **DISPROVEN / STALE** | `CleaningMetadata`, `AnalysisInsights`, `VizSpec`, etc. are present in `idd_core.py`. |
| **Claim 14** | `Plan.total_steps` exists | **DISPROVEN** | Field does not exist and is forbidden by `extra="forbid"`. |

---
*Inventory artifact prepared for Checkpoint 0 review per Issue #150 specifications.*
