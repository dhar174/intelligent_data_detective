# IDD Core Parity Inventory & Effective-Runtime Baseline

**Document**: `docs/architecture/idd-core-parity-inventory.md`  
**Purpose**: Architectural drift inventory, symbol-level runtime bindings, and empirical baseline evidence for Issue #150 / #140  
**References**: Refs #150, Refs #140, Refs #144, Refs #145  
**Status**: DRAFT FOR CP0 ARCHITECTURAL RE-REVIEW (Execution stopped for review gate)  
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
| `python -m pytest test_validate_run.py tests/unit tests/integration -q -rs` | 0 | **463 passed, 9 skipped** (28.20s) | 7 skips due directly to missing `idd_core` symbols and integration test expectations (`build_graph`, `AGENT_MEMBERS`, `options`, `next`); 1 skip due to missing Run 88 fixture; 1 skip due to optional `pyarrow`. |
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

Every overlapping symbol across the 5 representations has been audited for locations, effective runtime binding, consumers, and proposed canonical owner under the root-level modular architecture:

### 3.1 Pydantic Models & Data Contracts

| Symbol | Locations & Line Anchors | Effective Runtime Binding | Consumers | Differences & Drift | Proposed Canonical Owner | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`BaseNoExtrasModel`** | `idd_core.py:390`<br>`v5.py:698`<br>Patched NB: Cell 16:L2 | `BaseNoExtrasModel`<br>(`extra="forbid"`) | All agent output models | Consistent base contract: requires `reply_msg_to_supervisor`, `finished_this_task`, `expect_reply`. | `idd_models.py` | Shared Production Logic |
| **`PlanStep`** | `idd_core.py:605`<br>`v5.py:925`<br>Patched NB: Cell 16:L229 | `PlanStep`<br>(extends `BaseNoExtrasModel`) | `Plan`, `CompletedStepsAndTasks` | Identical fields (`step_number`, `step_name`, `step_description`, `is_step_complete`, `plan_version`). Note: Informal docs refer to this as "Step", but no `Step` class exists. | `idd_models.py` | Shared Production Logic |
| **`Plan`** | `idd_core.py:613`<br>`v5.py:931`<br>Patched NB: Cell 16:L235 | `Plan`<br>(with `_counter` / `_next` lock) | `_reduce_plan_keep_sorted`, `CompletedStepsAndTasks` | **Drift**: `idd_core` uses `_counter: ClassVar[itertools.count]`; notebook uses `_next = itertools.count(1).__next__`. Both overwrite user `plan_version` on instantiation. | `idd_models.py` | Shared Production Logic |
| **`CompletedStepsAndTasks`** | `idd_core.py:655`<br>`v5.py:970`<br>Patched NB: Cell 16:L284 | `CompletedStepsAndTasks` | Supervisor, progress accounting | **CRITICAL SPLIT-BRAIN (RC-2)**: `idd_core.py:686` returns sorted `dedup_list`. Notebook Cell 16:L317 sorts `dedup_list` but returns unsorted `list(seen.values())`, crashing on unsorted completed steps! | `idd_models.py` | Shared Production Logic |
| **`CleaningMetadata`** | `idd_core.py:436`<br>`v5.py:734`<br>Patched NB: Cell 16:L19 | `CleaningMetadata` | `data_cleaner_node`, `supervisor` | Synced. Present in both `idd_core.py` and notebook. | `idd_models.py` | Shared Production Logic |
| **`AnalysisInsights`** | `idd_core.py:467`<br>`v5.py:765`<br>Patched NB: Cell 16:L47 | `AnalysisInsights` | `analyst_node`, `visualization` | Synced. Contains `recommended_visualizations: List[VizSpec]`. | `idd_models.py` | Shared Production Logic |
| **`VizSpec`** | `idd_core.py:449`<br>`v5.py:747`<br>Patched NB: Cell 16:L30 | `VizSpec` | `AnalysisInsights`, `viz_worker` | Synced. Inherits `BaseNoExtrasModel`. | `idd_models.py` | Shared Production Logic |
| **`SectionOutline`** | `idd_core.py:742`<br>`v5.py:1296`<br>Patched NB: Cell 19:L233 | `SectionOutline` | `report_orchestrator`, `section_worker` | Synced. Present in `idd_core.py` and notebook Cell 19. | `idd_models.py` | Shared Production Logic |
| **`Section`** | `idd_core.py:732`<br>`v5.py:1287`<br>Patched NB: Cell 19:L224 | `Section` | `report_section_worker`, `report_join` | Synced fields (`name`, `section_num`, `description`, `goals`, `data_signals`, `expected_figures`, `content`). **Neither implementation enforces `min_length=100`**; `content` is an unconstrained string field in both. (Historical line 1075 locator was inside `_norm_path`). | `idd_models.py` | Shared Production Logic |
| **`ReportOutline`** | `idd_core.py:753`<br>`v5.py:1307`<br>Patched NB: Cell 19:L244 | `ReportOutline` | `report_orchestrator`, `report_packager` | Synced. Subclasses `SectionOutline`. | `idd_models.py` | Shared Production Logic |
| **`ReportResults`** | `idd_core.py:513`<br>`v5.py:811`<br>Patched NB: Cell 16:L94 | `ReportResults` | `report_packager`, `file_writer` | Synced. Paths for PDF, HTML, Markdown. | `idd_models.py` | Shared Production Logic |
| **`ListOfFiles`** | `idd_core.py:547`<br>`v5.py:845`<br>Patched NB: Cell 16:L181 | `ListOfFiles` | `file_writer_node`, final manifest | Synced. | `idd_models.py` | Shared Production Logic |
| **`Router`** | Patched NB: Cell 46:L1072<br>`v5.py:12260` | Function-local in `make_supervisor_node` | Supervisor LLM structured output | **MISSING from `idd_core.py`**. Trapped in closure. Must be promoted to top-level model. | `idd_models.py` | Shared Production Logic |

### 3.2 State Schemas & Reducers

| Symbol | Locations & Line Anchors | Effective Runtime Binding | Consumers | Differences & Drift | Proposed Canonical Owner | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`State`** | Patched NB: Cell 22:L70<br>`v5.py:1380`<br>`idd_core.py:1037` (omitted) | 70-field TypedDict with reducers | All 15 graph nodes, supervisor | **MISSING from `idd_core.py`** by explicit design comment ("incompatible with unit test imports"). Enforces W9-SR-DROP and BR-7. | `idd_state.py` | Shared Production Logic |
| **`_reduce_plan_keep_sorted`** | `idd_core.py:1022`<br>Patched NB: Cell 22:L32 | Reducer merging plans by step number | `State.current_plan` | Synced logic. Combines steps and deduplicates by `step_number` (last-wins). | `idd_state.py` | Shared Production Logic |
| **`keep_first`** | `idd_core.py:87`<br>Patched NB: Cell 7:L240 | Reducer preserving first non-None | State path channels (`artifacts_path`, etc.) | Synced. | `idd_state.py` | Shared Production Logic |
| **`_sr_reducer`** | Patched NB: Cell 22:L50<br>`idd_core.py` (absent) | Last-write-wins prefer non-None | `State.report_results` | Exists in notebook Cell 22 (W2-BR8c); absent from `idd_core.py`. | `idd_state.py` | Shared Production Logic |
| **`VizWorkerState`** | Patched NB: Cell 22:L150<br>`idd_core.py` (absent) | TypedDict for `Send("viz_worker")` | `viz_worker` fan-out | Missing from `idd_core.py`. | `idd_state.py` | Shared Production Logic |

### 3.3 DataFrame Management & Tool Decorators

| Symbol | Locations & Line Anchors | Effective Runtime Binding | Consumers | Differences & Drift | Proposed Canonical Owner | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`DataFrameRegistry`** | `idd_core.py:773`<br>Patched NB: Cell 19:L3<br>`v5.py:1004` | Thread-safe LRU registry | All data-touching tools | **Drift**: Notebook Cell 19 includes `_read_df` supporting `.csv`, `.parquet`, `.pkl`, `.json`. `idd_core.py:773` has `_read_df`, but `validate_dataframe_exists` hardcodes `pd.read_csv`. | `idd_registry.py` | Shared Production Logic |
| **`validate_dataframe_exists`** | `idd_core.py:1144`<br>Patched NB: Cell 32:L28 | Checks registry & reloads | All data-touching tools | **Drift**: `idd_core.py:1157` hardcodes `pd.read_csv`, failing on cache eviction of `.pkl`, `.json`, `.parquet`. | `idd_tools.py` | Shared Production Logic |
| **`handle_tool_errors`** | `idd_core.py:1169`<br>Patched NB: Cell 32:L67 | Error handling decorator | All tool functions | **CRITICAL DRIFT**: `idd_core` returns raw string `f"Error: {e}"`. Patched notebook returns structured dict `_tool_error(operation, reason, action)`. Neither is signature-aware, causing failure on non-first positional arguments. | `idd_tools.py` | Shared Production Logic |
| **`_resolve_artifact_path`** | `idd_core.py:1105`<br>Patched NB: Cell 32:L647 & Cell 57:L1000 | Path containment & env precedence | Visualization, File Writer | **Drift**: `idd_core` uses single-root `_is_subpath`. Patched notebook (PR #149) enforces multi-root allowed roots (`_allowed_roots`, `_fw_allowed_roots`) and canonical report paths. | `idd_tools.py` | Shared Production Logic |
| **`delete_rows` & `_build_query_view`** | Patched NB: Cell 32:L269, L300<br>`v5.py:3521`<br>`idd_core.py` (absent) | Tool with pandas query collision guard | Data cleaner agent | Missing from `idd_core.py`. `_build_query_view` resolves integer column labels (#147). | `idd_tools.py` | Shared Production Logic |

### 3.4 Graph Topology & Routing

| Symbol | Locations & Line Anchors | Effective Runtime Binding | Consumers | Differences & Drift | Proposed Canonical Owner | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`build_graph`** | Patched NB: Cell 60:L11 (as `data_analysis_team_builder`)<br>`idd_core.py` (absent) | Compiles 15-node StateGraph | Integration tests, live runner | **MISSING from `idd_core.py`**. Causes 4 tests in `tests/integration/test_graph_compile.py` to skip. | `idd_graph.py` | Shared Production Logic |
| **`AGENT_MEMBERS` / `AgentMembers`** | `idd_core.py:174` (`AgentMembers` class)<br>Patched NB: Cell 46:L10 | List of agent strings / literals | Supervisor routing, routers | `idd_core.py:174` defines class `AgentMembers` (with field `agent_type`, not `next`). Lacks `AGENT_MEMBERS` list constant. | `idd_graph.py` | Shared Production Logic |
| **`make_supervisor_node`** | Patched NB: Cell 46:L54<br>`idd_core.py` (absent) | Supervisor node factory | LangGraph supervisor node | Missing from `idd_core.py`. | `idd_graph.py` | Shared Production Logic |
| **`initial_analysis_node`, `data_cleaner_node`, etc. (11 wrapper nodes)** | Patched NB: Cell 57<br>`idd_core.py` (absent) | Safe invoke wrapper nodes | LangGraph graph assembly | 11 node wrappers and 3 join barriers exist only in the notebook. Missing from `idd_core.py`. | `idd_graph.py` | Shared Production Logic |

---

## 4. Empirical Drift Reproduction Evidence

All 12 claims were tested directly using [`tools/diagnostics/reproduce_drift.py`](../../tools/diagnostics/reproduce_drift.py) under Python 3.12.2 without network or API key dependencies:

### Claim 1: Plan Counter Sharing & Concurrency
- **Locators**: `idd_core.py:613–642` (`Plan._counter: ClassVar[itertools.count]`) vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 16:L235–265` (`Plan._next = itertools.count(1).__next__`).
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 1`
- **Expected**: Globally monotonic, thread-safe version allocation across Plan instances.
- **Observed**: Created versions across 5 concurrent threads: `[1, 2, 3, 4, 5]`. All unique: `True`. Thread safety verified with `threading.Lock()`.
- **Classification**: **OBSERVED INVARIANT** (Evidence: DYNAMIC, Defect: NO).
- **Analysis**: Class-level shared counter is an intentional healthy contract (the intended correction in Issue #140 to the older per-instance counter bug where every plan restarted at version 1). It is not a defect that independent plans share a monotonically increasing counter.

### Claim 2: Plan-Version Synchronization Overwrites Explicit Input
- **Locators**: `idd_core.py:635–642` (`_sync_steps_and_assert_increasing`) vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 16:L250–265`.
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 2`
- **Expected**: `Plan(plan_version=42, ...)` preserves version 42 during explicit construction or deserialization.
- **Observed**: Caller requested `plan_version=42`; resulting `plan.plan_version=6` (Overwritten: `True`).
- **Classification**: **REPRODUCED** (Evidence: DYNAMIC, Defect: YES).
- **Analysis**: `_ver_assigned` defaults to `False` on instantiation, causing the after-validator to unconditionally overwrite the caller's explicit version with the class counter.

### Claim 3: Completed-Step Sorting Divergence (Split-Brain Fix RC-2)
- **Locators**:  
  - Fixed: `idd_core.py:683–686` (`return dedup_list`)
  - Buggy: `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 16:L314–325` (`return list(seen.values())`)
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 3`
- **Expected**: When unsorted steps `[3, 1, 2]` arrive, `CompletedStepsAndTasks` should sort them ascending to `[1, 2, 3]`.
- **Observed**: `idd_core.py` accepted `[3, 1, 2]` and returned sorted `[1, 2, 3]`. Notebook Cell 16 executed with `[3, 1, 2]` CRASHED with `ValidationError: completed_steps must be sorted ascending by step_number`.
- **Classification**: **REPRODUCED** (Evidence: DYNAMIC, Defect: YES).
- **Analysis**: Severe split-brain bug: the fix was applied to `idd_core.py:686` (`return dedup_list`), but never ported to `_patch_notebook.py`. The notebook sorts `dedup_list` locally but discards it, returning `list(seen.values())` in original insertion order `[3, 1, 2]`, which crashes the subsequent `_assert_sorted_completed_no_dups` validator.

### Claim 4: Duplicate Numeric Completed-Step IDs in CompletedStepsAndTasks
- **Locators**: `idd_core.py:670–680` vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 16:L298–317`.
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 4`
- **Expected**: Duplicate numeric step numbers (e.g. `step_number=2` on two distinct steps with different names) should be rejected in `CompletedStepsAndTasks`.
- **Observed**: `idd_core` rejected duplicate `step_number=2` with `ValidationError: Duplicate step_number 2 in completed_steps`. Notebook Cell 16 accepted duplicate `step_number=2` with differing names and returned `[(2, 'X'), (2, 'Y')]` because the notebook deduplicates on full `Triplet` `(step_number, step_name, step_description)` rather than numeric `step_number`.
- **Classification**: **REPRODUCED** (Evidence: DYNAMIC, Defect: YES).
- **Analysis**: Contrast between contracts: `idd_core` enforces numeric uniqueness on completed steps; notebook Cell 16 permits duplicate numeric IDs as long as names/descriptions differ. (Separately: `Plan.plan_steps` in both implementations rejects duplicate step numbers via strictly increasing sequence validation).

### Claim 5: Validation Context / Subset Enforcement
- **Locators**: `idd_core.py:695–702` vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 16:L328–335`.
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 5`
- **Expected**: When validation context `{"plan": plan}` is provided, steps outside the plan are rejected. Without context, validation permits unconstrained completed steps.
- **Observed**: Dynamic test on `idd_core`: without context accepted=True; with context rejecting unplanned step 3=True. Static inspection of notebook Cell 16: implements identical `info.context` subset check (`get("plan")`).
- **Classification**: **OBSERVED INVARIANT** (Evidence: BOTH, Defect: NO).
- **Analysis**: Validation context subset enforcement is an intentional, healthy contract: subset checking is conditional on the context Plan being provided.

### Claim 6: Structured Tool Error Schema Divergence
- **Locators**: `idd_core.py:1169–1215` vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 32:L7–25`.
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 6`
- **Expected**: Tool errors must return standardized dictionary: `{"status": "error", "operation": str, "reason": str, "action": str}` with sanitized internal exceptions.
- **Observed**: Dynamic test on `idd_core`: returns raw string `str: "Error: Column or key ''missing_col'' not found"`. Static inspection of notebook Cell 32: defines structured dictionary `{"status": "error", "operation": ..., "reason": ..., "action": ...}` and `_tool_failure` exception sanitization.
- **Classification**: **REPRODUCED** (Evidence: BOTH, Defect: YES).
- **Analysis**: Fracture between test suite and production runtime. `tests/unit/test_handle_tool_errors.py` explicitly tests string returns, while `tests/unit/test_tool_error_handling.py` and production notebook tools rely on the structured dictionary schema.

### Claim 7: Signature-Aware `df_id` Extraction
- **Locators**: `idd_core.py:1176–1186` vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 32:L90–100` vs `test_error_handling_framework.py:324–355`.
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 7`
- **Expected**: Tool decorator must inspect function signature to bind `df_id`; functions without `df_id` must not have their first argument treated as a DataFrame ID.
- **Observed**: Calling `tool_without_df_id("hello", 42)` returned `"Error: DataFrame with ID 'hello' not found or is invalid"`. `args[0]` was blindly assumed to be `df_id`.
- **Classification**: **REPRODUCED** (Evidence: DYNAMIC, Defect: YES).
- **Analysis**: Both `idd_core` and notebook Cell 32 currently check `if args and isinstance(args[0], str): df_id = args[0]`. This breaks functions where `df_id` is not the first parameter and functions with non-df_id string parameters (the exact root cause of the isolated failure in `test_error_handling_framework.py`).

### Claim 8: Registry Reload Formats and `validate_dataframe_exists` Eviction Fallback
- **Locators**: `idd_core.py:803–814, 1154–1162` vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 19:L70–95, Cell 32:L50–60`.
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 8`
- **Expected**: Both `get_dataframe(..., load_if_not_exists=True)` and `validate_dataframe_exists` must support all registered formats upon cache eviction.
- **Observed**: `get_dataframe(load_if_not_exists=True)` succeeded across `.csv`, `.pkl`, `.json`. However, `validate_dataframe_exists` FAILED on evicted `.pkl` and `.json` DataFrames (`False`) because it hardcodes `pd.read_csv(raw_path)`.
- **Classification**: **REPRODUCED** (Evidence: DYNAMIC, Defect: YES).
- **Analysis**: `validate_dataframe_exists` bypasses `registry._read_df` and calls `pd.read_csv` directly, causing silent validation failure on cache-evicted pickle, parquet, and JSON DataFrames.

### Claim 9: Artifact-Root Containment & PR #149 Multi-Root Protections
- **Locators**: `idd_core.py:1105–1135` vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 32:L647–680, Cell 57:L1000–1030`.
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 9`
- **Expected**: Strict path containment within allowed roots; path traversal attempts (`../`) and absolute escapes must be rejected.
- **Observed**: `idd_core._resolve_artifact_path` dynamically verified: safe path resolved (`True`), path traversal `../../outside.txt` blocked (`ValueError`). Notebook Cell 57 statically verified: enforces PR #149 multi-root containment list `_allowed_roots = [_artifact_root, _working_dir, _run_root]`.
- **Classification**: **REPRODUCED** (Evidence: BOTH, Defect: YES).
- **Analysis**: `idd_core` uses single-root containment; notebook Cell 57 enforces PR #149's strict multi-root containment.

### Claim 10: LLM Adapter Identity (MyChatOpenai vs ChatOpenAI)
- **Locators**: `idd_core.py:341–370` vs `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 10:L130–170` vs `AGENTS.md:79` vs `tests/unit/test_mychatopenai.py:1–50`.
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 10`
- **Expected**: `MyChatOpenai` is the documented production model adapter subclassing `ChatOpenAI` for OpenAI o-series and custom Responses payload support.
- **Observed**: `idd_core` defines `MyChatOpenai` (`True`). Notebook Cell 10 defines `MyChatOpenai` (`True`). `AGENTS.md` explicitly documents: "MyChatOpenai: use everywhere in the notebook instead of ChatOpenAI". `tests/unit/test_mychatopenai.py` tests payload mutations.
- **Classification**: **OBSERVED INVARIANT** (Evidence: STATIC, Defect: NO).
- **Analysis**: `MyChatOpenai` is an active production contract and intentional subclass of `ChatOpenAI`, not an obsolete or drifting duplicate.

### Claim 11: Skipped Integration Tests Audit (Reconciling Root Causes)
- **Locators**: `tests/integration/test_graph_compile.py:15–51` vs `tests/integration/test_routing.py:16–72`.
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 11`
- **Expected**: Integration tests should execute meaningful assertions without live API keys against actual 15-node production topology and Router contracts.
- **Observed**: Static inspection: `test_graph_compile` skips 4 tests on missing `OPENAI_API_KEY` fixture (`True`) and asserts legacy node `report_generator` (`True`). `test_routing` skips 3 tests looking for lowercase `core.options` (`True`) and `AgentMembers.next` (`True`).
- **Classification**: **STATIC EVIDENCE ONLY** (Evidence: STATIC, Defect: YES).
- **Analysis**: The 7 integration tests skip due to a combination of: (1) requiring `OPENAI_API_KEY` fixture in graph tests, (2) expecting legacy node `report_generator` instead of 15-node topology, and (3) expecting router field `next` on member model `AgentMembers`.

### Claim 12: Integer Column Label Regression (#147) & delete_rows Collision Verification
- **Locators**: `IntelligentDataDetective_beta_v5_patched.ipynb:Cell 32:L269–355` vs `_patch_notebook.py:13495–13575`.
- **Command**: `python tools/diagnostics/reproduce_drift.py --claim 12`
- **Expected**: `delete_rows` must query integer column labels without `UndefinedVariableError`, resolve integer/string collisions, and properly execute in-place, non-in-place, and error-handling paths.
- **Observed**: Extracted actual production `delete_rows` tool and `_build_query_view` directly from Notebook Cell 32 into a controlled harness with an in-memory registry. Case A (integer 0 alone query `` `0` >= 20 ``): passed, 2 rows deleted. Case B (int/str collision): passed, string column '0' queried and 2 rows deleted without error. Case C (in-place mutation): passed, 2 rows deleted and registry updated. Case D (non-in-place operation): passed, returned JSON and registry source untouched. Case E (invalid query): passed, returned structured error dictionary and source untouched. All 5 cases passed.
- **Classification**: **OBSERVED INVARIANT** (Evidence: DYNAMIC, Defect: NO for tool implementation).
- **Analysis**: Actual production `delete_rows` tool and `_build_query_view` extracted directly from notebook Cell 32 are verified working under controlled conditions. Note: full multi-agent pipeline integration remains unverified without a live run; Issue #147 must remain open administratively until live pipeline validation.

---

## 5. Summary of Empirical Findings & Disproven Historical Claims

### 5.1 Drift Claims Classification Summary

| Claim ID | Claim Description | Classification | Defect? | Evidence Type | Key Finding |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Claim 1** | Plan counter shared & monotonic | **OBSERVED INVARIANT** | NO | DYNAMIC | Class-level shared counter is an intentional healthy contract (fixing per-instance reset). |
| **Claim 2** | Plan-version overwrite | **REPRODUCED** | YES | DYNAMIC | Caller-supplied `plan_version` is overwritten because `_ver_assigned` defaults to `False`. |
| **Claim 3** | Completed-step sorting split-brain | **REPRODUCED** | YES | DYNAMIC | `idd_core` returns sorted `dedup_list`; notebook Cell 16 crashes on unsorted input. |
| **Claim 4** | Duplicate numeric step IDs | **REPRODUCED** | YES | DYNAMIC | `idd_core` rejects duplicate numeric step numbers; notebook Cell 16 allows them if names differ. |
| **Claim 5** | Validation context subset check | **OBSERVED INVARIANT** | NO | BOTH | Subset enforcement is intentionally conditional on context Plan being passed. |
| **Claim 6** | Structured tool error schema | **REPRODUCED** | YES | BOTH | `idd_core` returns plain strings; notebook Cell 32 returns structured dicts with sanitization. |
| **Claim 7** | Signature-unaware `df_id` check | **REPRODUCED** | YES | DYNAMIC | Blind `args[0]` inspection breaks tools where `df_id` is 2nd argument or non-df_id functions. |
| **Claim 8** | Registry reload vs validate_exists | **REPRODUCED** | YES | DYNAMIC | `validate_dataframe_exists` fails on evicted `.pkl`/`.json` by hardcoding `pd.read_csv`. |
| **Claim 9** | Artifact-root containment | **REPRODUCED** | YES | BOTH | `idd_core` uses single root; notebook Cell 57 enforces PR #149 multi-root containment. |
| **Claim 10** | LLM adapter identity | **OBSERVED INVARIANT** | NO | STATIC | `MyChatOpenai` is active production contract in Cell 10 and `idd_core`, mandated by `AGENTS.md`. |
| **Claim 11** | Skipped integration tests | **STATIC EVIDENCE ONLY** | YES | STATIC | 7 tests skip due to `OPENAI_API_KEY` fixture, legacy `report_generator` node, and router schema. |
| **Claim 12** | Integer column queries (#147) | **OBSERVED INVARIANT** | NO | DYNAMIC | Actual production `delete_rows` tool verified working across 5 cases; full pipeline integration unverified. |

### 5.2 Disproven & Superseded Claims from Historical Reviews
1. **`Section` content constraint (`min_length=100`)**: DISPROVEN. Neither `idd_core.py:732` nor notebook Cell 19:L224 enforces `min_length=100`. Both define `content: str = Field(..., description="Content of the section")`.
2. **`Section` in textual export at line 1075**: DISPROVEN. Line 1075 of `intelligentdatadetective_beta_v5.py` is inside `DataFrameRegistry._norm_path`. `Section` is at line 1287.
3. **`idd_core.AgentMembers` at line 347**: DISPROVEN. `AgentMembers` is defined at `idd_core.py:174`.
4. **`idd_core.DataFrameRegistry` at line 865**: DISPROVEN. `DataFrameRegistry` begins at `idd_core.py:773`.
5. **`Plan.total_steps` and `Step`**: DISPROVEN. `total_steps` never existed (and is forbidden by `extra="forbid"`). The class name is `PlanStep`.
6. **Core models missing from `idd_core.py`**: DISPROVEN. `CleaningMetadata`, `AnalysisInsights`, `VizSpec`, `ReportResults`, `ListOfFiles`, `Section`, `SectionOutline` are present in `idd_core.py`.
