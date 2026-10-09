# IDD Canonical Core Migration Plan

**Document**: `docs/plans/idd-canonical-core-migration.md`  
**Purpose**: Architectural migration design, modular package boundaries, anti-drift guard, and incremental execution slices for Issue #140  
**References**: Refs #150, Refs #140, Refs #144, Refs #145  
**Status**: DRAFT FOR CP0 REVIEW (Execution stopped for review gate)  
**Date**: October 2026  
**Auditor / Steward**: IDD Repo Coordinator & Subagent Specialist Team (`notebook-patch-specialist`, `pydantic-contract-guardian`, `langgraph-topology-architect`)

---

## 1. Architectural Context & Objectives

### 1.1 The Split-Brain Problem
As established in [`docs/architecture/idd-core-parity-inventory.md`](file:///c:/Users/darf3/Documents/intelligent_data_detective/docs/architecture/idd-core-parity-inventory.md), the IDD codebase currently maintains five overlapping representations of its core runtime logic:
1. `idd_core.py` (hand-edited unit test substrate; missing `State` and `build_graph()`; contains desynchronized bug fixes).
2. `IntelligentDataDetective_beta_v5.ipynb` (authoring source notebook; contains legacy code).
3. `_patch_notebook.py` (authoritative patch engine with ~70 surgical patch blocks).
4. `IntelligentDataDetective_beta_v5_patched.ipynb` (committed runnable 99-cell production notebook).
5. `intelligentdatadetective_beta_v5.py` (non-runnable, non-importable `nbconvert` text export containing shell magics).

This split-brain architecture has caused severe empirical drift:
- **Bug Fix Desynchronization (RC-2)**: The fix for completed step sorting was committed to `idd_core.py:686` (`return dedup_list`), but never applied to `_patch_notebook.py` / notebook Cell 16:L310 (`return list(seen.values())`), leaving the production notebook vulnerable to crashing on unsorted completed steps.
- **Contract Drift**: `idd_core.py` tool error handlers return plain strings, whereas notebook Cell 32 returns structured dictionaries `{"status": "error", "error": ...}`.
- **Missing Capabilities**: `idd_core.py` hardcodes `pd.read_csv` on registry disk reload (failing on `.pkl`, `.parquet`, `.json`), and uses single-root artifact containment rather than PR #149's multi-root containment (`_allowed_roots`).
- **Skipped Integration Tests**: 7 integration tests in `tests/integration/test_graph_compile.py` and `tests/integration/test_routing.py` are unconditionally skipped because `idd_core.py` omits `build_graph()`, `AGENT_MEMBERS`, `options`, and `next`.

### 1.2 Core Migration Objectives
1. **Single Source of Truth**: Establish modular, pure-Python canonical modules at the repository root as the sole authority for models, registry, state, tools, and graph topology.
2. **Notebook Re-Export / Clean Import**: Have `_patch_notebook.py` inject thin imports from canonical modules into notebook cells, eliminating duplicated logic and preserving byte-for-byte regeneration determinism.
3. **Preserve Production Invariants**:
   - Strictly preserve the **99-cell notebook structure** (no deletions, no insertions, no reordering).
   - Strictly preserve **`BaseNoExtrasModel`** (`extra="forbid"`, `reply_msg_to_supervisor`, `finished_this_task`, `expect_reply`).
   - Strictly preserve **W9-SR-DROP** (no `structured_response` channel on supervisor `State`; extraction inside node wrappers).
   - Strictly preserve **BR-7** (`State.messages` declared as `Annotated[list[AnyMessage], add_messages]`, never inheriting from LangChain's `AgentState`).
   - Strictly preserve **fan-in barriers** (`viz_worker -> viz_join`, `report_section_worker -> report_join`) and **emergency escape edges** (`EMERGENCY_MSG -> supervisor`).
   - Strictly preserve **PR #149 multi-root artifact containment** and **PR #130 protections**.
   - Zero opportunistic changes to model adapters, LLM models, prompt texts, or graph topology.
4. **Automated Anti-Drift Guard**: Introduce `tests/unit/test_core_parity_guard.py` to enforce that notebook cells and canonical modules remain 100% synchronized via AST analysis.
5. **Real Integration Coverage**: Un-skip all 7 integration tests in `tests/integration/test_graph_compile.py` and `tests/integration/test_routing.py` by providing full canonical graph compilation.
6. **Unified Guidance Alignment (Issue #144)**: Update `AGENTS.md`, `GEMINI.md`, `.agents/`, and documentation in the same implementation PR.

---

## 2. Modular Architecture & Module Boundaries

In accordance with repo conventions ("No `src/` directory: all Python files live at the root"), canonical runtime logic will be organized into focused root Python modules with clear single responsibilities, while maintaining `idd_core.py` as a backwards-compatible façade.

```
REPO ROOT
├── idd_models.py           # Canonical Pydantic models (BaseNoExtrasModel, Plan, VizSpec, etc.)
├── idd_registry.py         # Thread-safe DataFrameRegistry with LRU caching & multi-format reload
├── idd_state.py            # 70-field LangGraph State schema & custom state reducers
├── idd_tools.py            # Tool definitions, signature-aware @handle_tool_errors, artifact resolution
├── idd_graph.py            # 15-node, 33-edge LangGraph construction (build_graph, AGENT_MEMBERS)
├── idd_core.py             # Backwards-compatibility façade re-exporting all canonical symbols
├── _patch_notebook.py      # Authoritative patcher wiring canonical modules into the 99-cell notebook
├── IntelligentDataDetective_beta_v5_patched.ipynb # 99-cell runnable production notebook
└── tests/
    ├── unit/test_core_parity_guard.py  # AST anti-drift verification test
    └── integration/                    # Un-skipped graph compilation & routing tests
```

### 2.1 Module Boundary Specifications

#### 1. `idd_models.py`
- **Scope**: All domain models, structured output contracts, and planning schemas.
- **Contents**:
  - `BaseNoExtrasModel`: Base class with `model_config = ConfigDict(extra="forbid")` and mandatory supervisor communication fields (`reply_msg_to_supervisor: str`, `finished_this_task: bool`, `expect_reply: bool`).
  - `PlanStep`: Single plan step model (`step_number`, `step_name`, `step_description`, `is_step_complete`, `plan_version`).
  - `Plan`: Counter-managed plan container with monotonic version synchronization and duplicate-step validation.
  - `CompletedStepsAndTasks`: Step progress tracking with deterministic sorting fix (RC-2: `return dedup_list`).
  - `CleaningMetadata`: Dataset cleaning statistics and transformations.
  - `AnalysisInsights` & `VizSpec`: Analytical findings, correlations, and recommended charts.
  - `SectionOutline`, `Section`, `ReportOutline`, `ReportResults`, `ListOfFiles`: Reporting pipeline models.
  - `Router`: Supervisor routing decision model.
- **Dependencies**: `pydantic`, `typing`, `itertools`, `threading`. (Zero LangGraph or LangChain runtime dependencies).

#### 2. `idd_registry.py`
- **Scope**: Canonical `DataFrameRegistry` implementation.
- **Contents**:
  - `DataFrameRegistry`: Thread-safe LRU cache with disk-spill and auto-reload capabilities.
  - `_read_df`: Multi-format file reader supporting `.csv`, `.parquet`, `.pkl`, `.feather`, `.json`, `.xlsx`.
  - Global singleton instance access (`DataFrameRegistry.get_instance()` or `default_registry`) with explicit reset API (`registry.clear()`, `registry.reset_instance()`) for test isolation.
- **Dependencies**: `pandas`, `threading`, `pathlib`, `os`.

#### 3. `idd_state.py`
- **Scope**: Complete LangGraph `State` definition and custom channel reducers.
- **Contents**:
  - Custom reducers:
    - `keep_first(a, b)`: Immutability reducer retaining the first non-None value.
    - `_reduce_plan_keep_sorted(current, update)`: Plan merging and deduplication preserving sorted step order.
    - `operator.add`: List concatenation reducer for logs, errors, and message history.
  - `State` class: Complete 70-field TypedDict matching notebook Cell 22.
    - `messages: Annotated[list[AnyMessage], add_messages]` (BR-7 compliance).
    - **No `structured_response` channel on `State`** (W9-SR-DROP compliance).
- **Dependencies**: `typing`, `typing_extensions`, `operator`, `langchain_core.messages`, `langgraph.graph.message`.

#### 4. `idd_tools.py`
- **Scope**: Tool error decorator, artifact path resolution, and analysis/viz tools.
- **Contents**:
  - `@handle_tool_errors`: Decorator providing:
    - Signature-aware `df_id` parameter extraction (inspecting `inspect.signature(func).parameters` rather than blind `args[0]`).
    - Standardized structured error return: `{"status": "error", "error": f"{e.__class__.__name__}: {str(e)}", "traceback": ...}` matching notebook Cell 32.
  - `validate_dataframe_exists(df_id)`: Guard ensuring referenced DataFrame exists in `DataFrameRegistry`.
  - `_resolve_artifact_path(path)`: Secure path traversal protection enforcing PR #149's multi-root containment (`_allowed_roots: list[Path] = [ARTIFACTS_DIR, REPORT_DIR, ...]`).
  - Analysis, cleaning, visualization, and report generation tool functions.
- **Dependencies**: `functools`, `inspect`, `traceback`, `pathlib`, `idd_registry`, `idd_models`.

#### 5. `idd_graph.py`
- **Scope**: 15-node, 33-edge supervisor-worker state graph construction and topology definitions.
- **Contents**:
  - `AGENT_MEMBERS`: List of worker agents (`["initial_analysis", "data_cleaner", "analyst", "visualization", "report_orchestrator", "report_section_worker", "report_packager", "file_writer", "viz_worker", "viz_evaluator"]`).
  - `AGENT_OPTIONS`: Routing options for supervisor (`AGENT_MEMBERS + ["FINISH", "EMERGENCY_MSG"]`).
  - Agent subgraph wrappers: Node functions executing inner agents with `recursion_limit = 160`, reading structured responses from `result["structured_response"]` (W9-SR-DROP), and returning state updates.
  - Fan-in barriers: `viz_join` and `report_join` synchronization logic.
  - `build_graph()`: Function assembling and compiling the `StateGraph` with `recursion_limit = 400`, 15 nodes, and 33 edges.
- **Dependencies**: `langgraph`, `langchain_core`, `idd_state`, `idd_models`, `idd_tools`.

#### 6. `idd_core.py` (Backwards-Compatibility Façade)
- **Scope**: Unified public import façade preserving 100% backward compatibility for all existing unit tests and callers.
- **Implementation**:
  ```python
  """idd_core.py - Canonical façade re-exporting IDD runtime components."""
  from idd_models import (
      BaseNoExtrasModel, PlanStep, Plan, CompletedStepsAndTasks,
      CleaningMetadata, AnalysisInsights, VizSpec, SectionOutline,
      Section, ReportOutline, ReportResults, ListOfFiles, Router,
  )
  from idd_registry import DataFrameRegistry
  from idd_state import State, keep_first, _reduce_plan_keep_sorted
  from idd_tools import handle_tool_errors, validate_dataframe_exists, _resolve_artifact_path
  from idd_graph import build_graph, AGENT_MEMBERS, AGENT_OPTIONS

  __all__ = [
      "BaseNoExtrasModel", "PlanStep", "Plan", "CompletedStepsAndTasks",
      "CleaningMetadata", "AnalysisInsights", "VizSpec", "SectionOutline",
      "Section", "ReportOutline", "ReportResults", "ListOfFiles", "Router",
      "DataFrameRegistry", "State", "keep_first", "_reduce_plan_keep_sorted",
      "handle_tool_errors", "validate_dataframe_exists", "_resolve_artifact_path",
      "build_graph", "AGENT_MEMBERS", "AGENT_OPTIONS",
  ]
  ```

---

## 3. Explicit Shared-Registry Ownership

### 3.1 Singleton vs. Test Isolation Semantics
Currently, `DataFrameRegistry` is instantiated ad-hoc in multiple places (in `idd_core.py`, in tests, and in notebook Cell 19). This creates potential state leakage during test runs and test-to-production inconsistencies.

**Canonical Ownership Contract**:
1. **Module Singleton**: `idd_registry.py` exposes `registry = DataFrameRegistry()`.
2. **Explicit Reset API**:
   ```python
   class DataFrameRegistry:
       ...
       def clear(self) -> None:
           """Evict all cached DataFrames and clear disk spills."""
           with self._lock:
               self._cache.clear()
               self._metadata.clear()
               # Clean temporary cache directory if applicable
   ```
3. **Pytest Fixture (`conftest.py`)**:
   ```python
   @pytest.fixture(autouse=True)
   def reset_dataframe_registry():
       """Ensure clean registry state before each test."""
       from idd_registry import registry
       registry.clear()
       yield
       registry.clear()
   ```
4. **Registry Invariant**: Raw `pd.DataFrame` objects must never be passed across nodes or tools. All operations strictly reference data via UUID string `df_id`.

---

## 4. Import & Bootstrap Strategy (Local & Colab Environments)

The IDD architecture requires seamless execution in two primary environments:
1. **Local Workstation / CI**: Repo root is in `sys.path`. Canonical Python files live directly at repo root.
2. **Google Colab**: The notebook may be executed in an ephemeral Colab instance where the repo might be cloned or files downloaded.

### 4.1 Bootstrap Strategy in the Notebook
To preserve the **strict 99-cell invariant** and ensure robust execution everywhere:
- **Cells 1–6 (Setup & Environment)**:
  - Cell 2 already handles environment bootstrapping and pip installation.
  - A standardized sys.path bootstrap is added:
    ```python
    import sys
    from pathlib import Path
    REPO_ROOT = Path.cwd()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    ```
- **Cells 16, 19, 22, 32, 60 (Domain Logic Cells)**:
  - Rather than defining 500–1,000 lines of duplicated Python classes in notebook cells, `_patch_notebook.py` injects clean canonical imports and re-exports:
    - **Cell 16 (Models)**: `from idd_models import BaseNoExtrasModel, Plan, PlanStep, CompletedStepsAndTasks, ...`
    - **Cell 19 (Registry)**: `from idd_registry import DataFrameRegistry, registry`
    - **Cell 22 (State)**: `from idd_state import State, keep_first, _reduce_plan_keep_sorted`
    - **Cell 32 (Tools & Decorator)**: `from idd_tools import handle_tool_errors, validate_dataframe_exists, _resolve_artifact_path, ...`
    - **Cell 60 (Graph Assembly)**: `from idd_graph import build_graph, AGENT_MEMBERS, AGENT_OPTIONS; app = build_graph()`
- **Cell ID & Metadata Preservation**:
  - Cell count remains exactly 99.
  - All existing cell IDs, execution order, and markdown instructional content remain unchanged.

---

## 5. Behavioral Bug Fixes vs. Relocation Separation

To maintain strict engineering rigor, behavioral fixes must be explicitly isolated from code relocation:

| Issue / Claim | Kind | Migration Scope | Rationale & Behavioral Change |
| :--- | :--- | :--- | :--- |
| **Claim 3 / RC-2** (Completed Steps Sorting) | **Behavioral Bug Fix** | `idd_models.CompletedStepsAndTasks` | Change notebook return from unsorted `list(seen.values())` to sorted `dedup_list`. Prevents crashes on unsorted completed steps. |
| **Claim 2** (Plan Version Overwrite) | **Behavioral Bug Fix** | `idd_models.Plan` | In `Plan.__init__`, check if `plan_version` was explicitly supplied by caller; if so, retain it; otherwise assign from counter. |
| **Claim 6** (Structured Tool Errors) | **Behavioral Bug Fix** | `idd_tools.handle_tool_errors` | Align `idd_core` with notebook Cell 32: return standardized dictionary `{"status": "error", "error": f"{cls}: {msg}", ...}` instead of bare string. |
| **Claim 7** (Signature-Aware `df_id`) | **Behavioral Bug Fix** | `idd_tools.handle_tool_errors` | Use `inspect.signature` to locate the `df_id` parameter by name or position, rather than assuming `args[0]`. Fixes edge-case failure in `test_error_handling_framework.py`. |
| **Claim 8** (Multi-Format Disk Reload) | **Behavioral Bug Fix** | `idd_registry.DataFrameRegistry` | Port notebook Cell 19 multi-format `_read_df` (`.csv`, `.parquet`, `.pkl`, `.feather`, `.json`, `.xlsx`) to `idd_registry.py`. |
| **Claim 9** (Multi-Root Containment) | **Behavioral Bug Fix** | `idd_tools._resolve_artifact_path` | Port notebook Cell 57 / PR #149 `_allowed_roots` multi-root validation into `idd_tools.py`. |
| **State & Graph Relocation** | **Pure Relocation** | `idd_state.py`, `idd_graph.py` | Move 70-field `State` and 15-node `build_graph()` into canonical modules without modifying their topology, channels, or reducers. |

---

## 6. Anti-Drift Guard Specification (`test_core_parity_guard.py`)

To ensure that future PRs never re-introduce drift between canonical modules and the notebook, a dedicated no-key unit test suite `tests/unit/test_core_parity_guard.py` will be created.

### 6.1 Guard Verification Checks
1. **Model Contract Parity**:
   - Asserts that all models in `idd_models` subclass `BaseNoExtrasModel` and enforce `extra="forbid"`.
   - Asserts that `CompletedStepsAndTasks.completed_steps` always returns strictly ascending sorted steps.
2. **Notebook Cell Import Verification**:
   - Parses `IntelligentDataDetective_beta_v5_patched.ipynb` via `nbformat`.
   - Asserts that Cells 16, 19, 22, 32, and 60 import their symbols from `idd_models`, `idd_registry`, `idd_state`, `idd_tools`, and `idd_graph`.
   - Asserts that no duplicated class definitions exist in those notebook cells.
3. **State Channel Invariant Check**:
   - Asserts that `State` defines `messages: Annotated[list[AnyMessage], add_messages]`.
   - Asserts that `State` has **no `structured_response` channel** (W9-SR-DROP enforcement).
4. **Topology Parity Check**:
   - Asserts that `build_graph()` produces exactly 15 nodes and 33 edges matching `validate_graph.py`.
   - Asserts that `EMERGENCY_MSG` has an outgoing edge to `supervisor`.

---

## 7. Control-Plane & Agent Guidance Alignment (Issue #144 in Same Implementation PR)

In accordance with user authorization, Issue #144 instructions and control-plane guidance must be updated in the **SAME implementation PR** as Issue #140:

### 7.1 Documents & Rules to Update
1. **`AGENTS.md`**:
   - Update Repo Map: document `idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, `idd_graph.py`, and `idd_core.py` as the canonical core.
   - Update Engineering Conventions: state that production logic is defined in root modules, while notebook cells import them.
2. **`GEMINI.md`**:
   - Update architecture diagrams and subagent roster descriptions.
3. **`.agents/rules/code-architecture.md`**:
   - Update Section 1 (Notebook Generation Paradigm): explain the relationship between canonical modules, `_patch_notebook.py`, and the generated notebook.
4. **`.agents/rules/validation-and-gates.md`**:
   - Add `test_core_parity_guard.py` to the mandatory no-key test gate commands.
5. **`.github/agents/` and `.github/instructions/`**:
   - Synchronize subagent instruction prompts with canonical module imports.
6. **`README.md`**:
   - Update architecture section to describe the modular Python core and its relationship to the interactive notebook.

---

## 8. Incremental Implementation Slices & Rollback Gates

To ensure fail-closed verification, the migration is decomposed into six bounded slices:

```
[Slice 0: CP0 Baseline & Plan] ──> [Slice 1: Models & Registry] ──> [Slice 2: Tools & Errors]
                                                                            │
                                                                            ▼
[Slice 5: Keyed Live Proof]   <── [Slice 4: Notebook & Docs]    <── [Slice 3: State & Graph]
```

### Slice 0: Baseline & Diagnostics (CP0 - Current Status)
- **Scope**: Native Project #7 setup (#151); empirical drift reproduction suite (`tools/diagnostics/reproduce_drift.py`); parity inventory (`docs/architecture/idd-core-parity-inventory.md`); migration plan (`docs/plans/idd-canonical-core-migration.md`).
- **Gates**: No-key baseline measured (463 passed, 9 skipped); 0 bytes production churn; draft PR opened.
- **Review Gate**: **STOP FOR CP0 REVIEW**.

### Slice 1: Models & Registry Parity (CP1)
- **Scope**:
  - Implement `idd_models.py` with `BaseNoExtrasModel`, `Plan`, `PlanStep`, `CompletedStepsAndTasks`, and reporting models. Apply Fix RC-2 (sorting fix) and Claim 2 (version fix).
  - Implement `idd_registry.py` with multi-format `_read_df` and `clear()` API.
  - Update `idd_core.py` to re-export from `idd_models` and `idd_registry`.
- **Validation Gate**:
  - `python -m pytest tests/unit/test_plan.py tests/unit/test_models.py tests/unit/test_registry.py -v`
  - All existing unit tests pass; Claims 1, 2, 3, 4, 5, 8 verified green in unit test suite.
- **Rollback Criteria**: Any failure in Pydantic schema validation or registry caching rolls back Slice 1.

### Slice 2: Tools & Error Handling Hardening (CP2)
- **Scope**:
  - Implement `idd_tools.py` with signature-aware `@handle_tool_errors`, structured error dictionary return, and multi-root `_resolve_artifact_path`.
  - Update `idd_core.py` to re-export from `idd_tools`.
- **Validation Gate**:
  - `python -m pytest test_error_handling_framework.py -v` passes **16/16** (resolving the previously isolated failure `test_integration_with_different_function_signatures`).
  - Claims 6, 7, 9 verified green.
- **Rollback Criteria**: Any breakage in tool execution or path resolution rolls back Slice 2.

### Slice 3: State, Graph & Integration Un-skipping (CP3)
- **Scope**:
  - Implement `idd_state.py` with 70-field `State` and reducers (`keep_first`, `_reduce_plan_keep_sorted`).
  - Implement `idd_graph.py` with `build_graph()`, `AGENT_MEMBERS`, node wrappers with `structured_response` extraction (W9-SR-DROP), recursion limits, and escape edges.
  - Update `idd_core.py` to re-export `State`, `build_graph`, `AGENT_MEMBERS`.
  - Un-skip the 7 integration tests in `tests/integration/test_graph_compile.py` and `tests/integration/test_routing.py`.
- **Validation Gate**:
  - `python -m pytest tests/integration/test_graph_compile.py tests/integration/test_routing.py -v` passes 100% (0 skips).
  - Overall suite: `python -m pytest test_validate_run.py tests/unit tests/integration -q` passes **470 passed, 2 skipped** (remaining skips: Run 88 fixture, pyarrow).
- **Rollback Criteria**: Any graph compilation error, dead-end node, or schema collision rolls back Slice 3.

### Slice 4: Notebook Wiring, Anti-Drift Guard & Issue #144 Guidance (CP4)
- **Scope**:
  - Update `_patch_notebook.py` sentinels to inject imports from canonical modules into Cells 16, 19, 22, 32, and 60.
  - Regenerate notebook: `python _patch_notebook.py`.
  - Verify notebook: `validate_notebook_integrity.py` (99 cells) and `validate_graph.py`.
  - Implement `tests/unit/test_core_parity_guard.py`.
  - Update agent guidance, rules, and docs as specified in Issue #144 (`AGENTS.md`, `GEMINI.md`, `.agents/`, `README.md`).
- **Validation Gate**:
  - Full no-key test suite passes cleanly:
    - `python -m pytest test_validate_run.py tests/unit tests/integration -q`
    - `python -m pytest tests/unit/test_core_parity_guard.py -v`
    - `python -m pytest test_intelligent_data_detective.py -v`
    - `python -m pytest test_prompt_formatting.py test_prompt_template_fixes.py -v`
    - `python -m pytest test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py -v`
    - `python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb`
    - `python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb`
- **Rollback Criteria**: Notebook cell count != 99 or any anti-drift guard assertion failure.

### Slice 5: Keyed Proof & Production Verification Gate (CP5)
- **Scope**:
  - When live paid-model testing is authorized by the user:
  - Execute end-to-end run: `python run_notebook_live.py`.
  - Enforce twin gates:
    - `python validate_run.py --latest --log-path notebook_run_log.txt --window 180` (12/12 production criteria).
    - `python validate_artifact_quality.py --latest` (9/9 artifact quality criteria).
  - Verify zero recovery markers, zero final-hop warnings, zero path-normalization warnings, and complete final artifacts (`final_report.html`, `.pdf`, `.md`, charts, cleaned CSVs).

---

## 9. Unresolved Decisions & Review Questions for CP0 Review

Before proceeding from CP0 to CP1 implementation, the following architectural choices are submitted for review:

1. **Façade vs. Direct Imports**:
   - *Proposal*: All existing tests continue importing from `idd_core` via re-exports, while new tests import from specific modules (`idd_models`, `idd_state`, etc.). Notebook imports directly from specific modules.
   - *Alternative*: Refactor all existing tests to import directly from specific modules. (Rejected as too broad for CP1).
2. **Shared Registry Instance**:
   - *Proposal*: Expose a module-level singleton `registry` with an explicit `registry.clear()` fixture in `conftest.py`.
   - *Alternative*: Require dependency-injecting the registry instance into all tools. (Rejected: breaks LangChain tool invocation signatures).
3. **Issue #147 Administrative Resolution**:
   - *Finding*: Issue #147 (integer/string column label queries) is confirmed fixed on `main` via PR #149's `_build_query_view` projection.
   - *Recommendation*: Close Issue #147 administratively during CP1 review without additional code changes.
