# IDD Canonical Core Migration Plan

**Document**: `docs/plans/idd-canonical-core-migration.md`  
**Purpose**: Architectural migration design, modular boundaries, anti-drift guard, and incremental execution slices for Issue #140  
**References**: Refs #150, Refs #140, Refs #144, Refs #145, Refs #147, Refs #155  
**Status**: DRAFT FOR CP0 ARCHITECTURAL RE-REVIEW (Execution stopped for review gate)  
**Date**: October 2026  
**Auditor / Steward**: IDD Repo Coordinator & Subagent Specialist Team (`notebook-patch-specialist`, `pydantic-contract-guardian`, `langgraph-topology-architect`)

---

## 1. Architectural Context & Objectives

### 1.1 The Split-Brain Problem
As established in [`docs/architecture/idd-core-parity-inventory.md`](../architecture/idd-core-parity-inventory.md), the IDD codebase currently maintains five overlapping representations of its core runtime logic:
1. `idd_core.py` (hand-edited unit test substrate; missing `State` and `build_graph()`; contains desynchronized bug fixes).
2. `IntelligentDataDetective_beta_v5.ipynb` (authoring source notebook; contains legacy code).
3. `_patch_notebook.py` (authoritative patch engine with ~70 surgical patch blocks).
4. `IntelligentDataDetective_beta_v5_patched.ipynb` (committed runnable 99-cell production notebook).
5. `intelligentdatadetective_beta_v5.py` (non-runnable, non-importable `nbconvert` text export containing shell magics).

Empirical findings from [`tools/diagnostics/reproduce_drift.py`](../../tools/diagnostics/reproduce_drift.py) establish critical fractures:
- **Completed-Step Sorting Split-Brain (Claim 3 / RC-2)**: `idd_core.py:686` returns sorted `dedup_list` on incoming `[3, 1, 2]`. Notebook Cell 16:L317 sorts `dedup_list` locally but discards it, executing `return list(seen.values())`, crashing `CompletedStepsAndTasks` with `ValidationError: completed_steps must be sorted ascending by step_number`.
- **Duplicate Completed Steps Contract (Claim 4)**: `idd_core` rejects duplicate numeric `step_number`s even with different names; notebook Cell 16 deduplicates on full `Triplet` `(step_number, step_name, step_description)`, permitting duplicate numeric step numbers if names differ.
- **Tool Error Contract Drift (Claim 6)**: `idd_core.py` returns plain strings (`"Error: ..."`), whereas notebook Cell 32 defines the production structured dictionary schema `{"status": "error", "operation": ..., "reason": ..., "action": ...}` and sanitizes unexpected exceptions via `_tool_failure`.
- **Signature-Unaware `df_id` Extraction (Claim 7)**: Both decorators blindly inspect `args[0]`, treating non-df_id strings as DataFrame IDs (the root cause of the isolated failure in `test_error_handling_framework.py`).
- **Registry Eviction Fallback Defect (Claim 8)**: `validate_dataframe_exists` hardcodes `pd.read_csv`, silently failing on cache-evicted `.pkl`, `.parquet`, and `.json` DataFrames.
- **Skipped Integration Tests (Claim 11)**: All 4 graph compilation tests and 3 routing tests skip unconditionally due to `OPENAI_API_KEY` requirements, legacy node names (`report_generator`), and router schema mismatches.

### 1.2 Core Migration Objectives
1. **Single Source of Truth**: Establish modular root-level Python modules as the sole authority for models, registry, state, tools, and graph topology.
2. **Notebook Re-Export / Clean Import**: Have `_patch_notebook.py` inject thin imports from canonical modules into notebook cells, preserving byte-for-byte regeneration determinism.
3. **Preserve Production Invariants**:
   - Strictly preserve the **99-cell notebook structure** (no deletions, no insertions, no reordering).
   - Strictly preserve **`BaseNoExtrasModel`** (`extra="forbid"`, `reply_msg_to_supervisor`, `finished_this_task`, `expect_reply`).
   - Strictly preserve **W9-SR-DROP** (no `structured_response` channel on supervisor `State`; extraction inside node wrappers).
   - Strictly preserve **BR-7** (`State.messages` declared as `Annotated[list[AnyMessage], add_messages]`, never inheriting from LangChain's `AgentState`).
   - Strictly preserve **fan-in barriers** (`viz_worker -> viz_join`, `report_section_worker -> report_join`) and **emergency escape edges** (`EMERGENCY_MSG -> supervisor`).
   - Strictly preserve **PR #149 multi-root artifact containment** (`_allowed_roots`) and **PR #130 protections**.
   - Zero opportunistic changes to model adapters (`MyChatOpenai`), LLM models, prompt texts, or graph topology.
4. **Automated Anti-Drift Guard**: Introduce `tests/unit/test_core_parity_guard.py` enforcing AST and runtime symbol identity.
5. **Real No-Key Integration Coverage**: Modernize the 7 integration tests to compile the actual 15-node topology without requiring live API credentials.
6. **Unified Guidance Alignment (Issue #144)**: Synchronize `AGENTS.md`, `GEMINI.md`, `.agents/`, and documentation in the same implementation PR.

---

## 2. Modular Architecture & Module Boundaries

In accordance with repository conventions ("No `src/` directory: all Python files live at the root"), canonical runtime logic is organized into focused root Python modules with clear boundaries, while maintaining `idd_core.py` as a backwards-compatible façade:

```
REPO ROOT
├── idd_models.py           # Canonical Pydantic models (BaseNoExtrasModel, Plan, VizSpec, Section, Router)
├── idd_registry.py         # Thread-safe DataFrameRegistry with LRU caching, multi-format reload, reset API
├── idd_state.py            # 70-field LangGraph State schema & custom state reducers (keep_first, plan reducer)
├── idd_tools.py            # Signature-aware @handle_tool_errors, production dict error schema, path resolution
├── idd_graph.py            # 15-node, 33-edge LangGraph factory (build_graph, AGENT_MEMBERS, AGENT_OPTIONS)
├── idd_core.py             # Backwards-compatibility façade re-exporting all canonical symbols
├── _patch_notebook.py      # Authoritative patcher wiring canonical modules into the 99-cell notebook
├── IntelligentDataDetective_beta_v5_patched.ipynb # 99-cell runnable production notebook
└── tests/
    ├── unit/test_core_parity_guard.py  # AST anti-drift verification test
    └── integration/                    # Modernized no-key graph compilation & routing tests
```

### 2.1 Module Boundary Specifications

#### 1. `idd_models.py`
- **Scope**: All domain models, structured output contracts, and planning schemas.
- **Contents**:
  - `BaseNoExtrasModel`: Base class with `model_config = ConfigDict(extra="forbid")` and mandatory supervisor communication fields (`reply_msg_to_supervisor: str`, `finished_this_task: bool`, `expect_reply: bool`).
  - `PlanStep`: Single plan step model (`step_number`, `step_name`, `step_description`, `is_step_complete`, `plan_version`).
  - `Plan`: Counter-managed plan container with monotonic version allocation and duplicate-step validation. Preserves caller-supplied `plan_version` when explicitly provided.
  - `CompletedStepsAndTasks`: Step progress tracking with deterministic sorting fix (RC-2: `return dedup_list`). Rejects duplicate numeric step numbers.
  - `CleaningMetadata`: Dataset cleaning statistics and transformations.
  - `AnalysisInsights` & `VizSpec`: Analytical findings, correlations, and recommended charts.
  - `SectionOutline`, `Section`, `ReportOutline`, `ReportResults`, `ListOfFiles`: Reporting pipeline models.
  - `Router`: Supervisor routing decision model with `next: Literal[...]` field.
- **Dependencies**: `pydantic`, `typing`, `itertools`, `threading`. (Zero LangChain or LangGraph dependencies).

#### 2. `idd_registry.py`
- **Scope**: Canonical `DataFrameRegistry` implementation and global accessors.
- **Contents**:
  - `DataFrameRegistry`: Thread-safe LRU cache with attributes `self.registry`, `self.cache`, `self.df_id_to_raw_path`, `self.capacity`, `self._lock`.
  - `_read_df`: Multi-format file reader supporting `.csv`, `.parquet`, `.pkl`, `.feather`, `.json`, `.xlsx`.
  - `clear()`: Method evicting in-memory cache and registry entries without deleting non-owned user data.
  - Global registry accessors: `get_global_registry()`, `set_global_registry(reg)`.
- **Dependencies**: `pandas`, `threading`, `pathlib`, `os`.

#### 3. `idd_state.py`
- **Scope**: Complete LangGraph `State` definition and custom channel reducers.
- **Contents**:
  - Custom reducers: `keep_first`, `_reduce_plan_keep_sorted`, `_sr_reducer`, `operator.add`.
  - `State`: Complete 70-field TypedDict matching notebook Cell 22.
    - `messages: Annotated[list[AnyMessage], add_messages]` (BR-7 compliance).
    - **No `structured_response` channel on `State`** (W9-SR-DROP compliance).
  - `VizWorkerState`: TypedDict for visualization fan-out.
- **Dependencies**: `typing`, `typing_extensions`, `operator`, `langchain_core.messages`, `langgraph.graph.message`.

#### 4. `idd_tools.py`
- **Scope**: Tool error decorator, artifact path resolution, and analysis/viz tools.
- **Contents**:
  - `_tool_error(operation, reason, action) -> dict`: Returns `{"status": "error", "operation": operation, "reason": reason, "action": action}`.
  - `_tool_failure(operation, action, exc) -> dict`: Sanitizes unexpected failures while logging exception.
  - `@handle_tool_errors`: Decorator providing:
    - Signature-aware `df_id` binding (inspecting `inspect.signature(func)`).
    - Preserves non-df_id functions without treating `args[0]` as a DataFrame ID.
    - Returns production structured error dictionary.
  - `validate_dataframe_exists(df_id)`: Validates existence and delegates reload to `registry._read_df` (fixing Claim 8).
  - `_resolve_artifact_path(path)`: Secure path traversal protection enforcing PR #149's multi-root containment (`_allowed_roots`).
- **Dependencies**: `functools`, `inspect`, `logging`, `traceback`, `pathlib`, `idd_registry`, `idd_models`.

#### 5. `idd_graph.py`
- **Scope**: 15-node, 33-edge supervisor-worker state graph construction and topology definitions.
- **Contents**:
  - `AGENT_MEMBERS`: Worker agent names list (`["initial_analysis", "data_cleaner", "analyst", "visualization", "report_orchestrator", "report_section_worker", "report_packager", "file_writer", "viz_worker", "viz_evaluator"]`).
  - `AGENT_OPTIONS`: Routing options (`AGENT_MEMBERS + ["FINISH", "EMERGENCY_MSG"]`).
  - `options`: Lowercase alias for backward compatibility with existing tests.
  - `build_graph(llm_factory=None)`: Graph factory supporting no-key stub construction.
  - Agent subgraph wrappers: Node functions extracting `result["structured_response"]` (W9-SR-DROP).
  - Fan-in barriers: `viz_join` and `report_join`.
- **Dependencies**: `langgraph`, `langchain_core`, `idd_state`, `idd_models`, `idd_tools`.

#### 6. `idd_core.py` (Compatibility Façade)
- **Scope**: Backward-compatibility façade re-exporting all symbols from root modules.
- **Dynamic Registry Read Delegation & Migration Contract**:
  ```python
  """idd_core.py - Compatibility façade."""
  import idd_registry
  from idd_models import *
  from idd_registry import (
      DataFrameRegistry,
      get_global_registry,
      set_global_registry,
      override_global_registry,
  )
  from idd_state import State, keep_first, _reduce_plan_keep_sorted
  from idd_tools import (
      handle_tool_errors,
      validate_dataframe_exists,
      _resolve_artifact_path,
      _tool_error,
      _tool_failure,
  )
  from idd_graph import build_graph, AGENT_MEMBERS, AGENT_OPTIONS

  # Module-level read delegation via PEP 562
  def __getattr__(name: str):
      if name == "global_df_registry":
          return idd_registry.get_global_registry()
      if name == "options":
          return AGENT_OPTIONS
      raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
  ```
  > [!IMPORTANT]
  > **Python PEP 562 Module-Assignment Reality**:
  > Python permits module-level `__getattr__` for missing attribute reads, but does **not** support assignment interception through an ordinary module-level `__setattr__` function. Assigning `idd_core.global_df_registry = replacement` inserts `global_df_registry` directly into `idd_core.__dict__`, shadowing `__getattr__` without notifying `idd_registry` or updating canonical tools.
  > Therefore, direct module-attribute reassignment is **explicitly unsupported** for transparent interception. Callers modifying the global registry must migrate to the explicit API: `idd_registry.set_global_registry(...)` or the context-managed override fixture `idd_registry.override_global_registry(...)`.

---

## 3. Explicit Shared-Registry Ownership & Test Isolation Contract

### 3.1 Real Attribute Structure of `DataFrameRegistry`
`DataFrameRegistry` maintains the following actual attributes:
- `self.registry: Dict[str, dict]`: Metadata dictionary mapping `df_id` to `{"df": pd.DataFrame | None, "raw_path": str}`.
- `self.cache: OrderedDict[str, pd.DataFrame]`: In-memory LRU cache limited by `self.capacity`.
- `self.df_id_to_raw_path: Dict[str, str]`: Mapping of `df_id` to filesystem source path.
- `self.capacity: int`: LRU eviction limit (default 20).
- `self._lock: threading.RLock`: Thread-safety reentrant lock.

### 3.2 Reset Contract (`clear()`)
A registry reset must clear transient in-memory state while **never deleting non-owned user data or published report artifacts**:
```python
def clear(self) -> None:
    """Evict all cached DataFrames and clear in-memory registry mappings."""
    with self._lock:
        self.cache.clear()
        self.registry.clear()
        self.df_id_to_raw_path.clear()
```

### 3.3 Authoritative Ownership & Concurrent-Safe Override Architecture
`idd_registry.py` is the single source of truth for registry state. To provide true concurrent test isolation and prevent thread-interleaving corruption, the canonical API uses `contextvars.ContextVar` for execution-scoped overrides combined with a thread-safe process default:

```python
# In idd_registry.py:
import contextlib
import contextvars
import threading
from typing import Optional

_default_registry: DataFrameRegistry = DataFrameRegistry(capacity=20)
_default_registry_lock = threading.RLock()
_registry_override: contextvars.ContextVar[Optional[DataFrameRegistry]] = (
    contextvars.ContextVar("idd_registry_override", default=None)
)

def get_global_registry() -> DataFrameRegistry:
    """Return the active DataFrameRegistry instance.
    
    Resolution order:
    1. Context-local override if active in the caller's Context (thread or task).
    2. Otherwise, the process-default registry protected by RLock.
    """
    override = _registry_override.get()
    if override is not None:
        return override
    with _default_registry_lock:
        return _default_registry

def set_global_registry(registry: DataFrameRegistry) -> None:
    """Explicitly replace the process-default DataFrameRegistry instance."""
    global _default_registry
    if not isinstance(registry, DataFrameRegistry):
        raise TypeError("registry must be an instance of DataFrameRegistry")
    with _default_registry_lock:
        _default_registry = registry

@contextlib.contextmanager
def override_global_registry(registry: DataFrameRegistry):
    """Context-local manager for test and run isolation.
    
    Sets the override in the caller's ContextVar execution context via ContextVar.set(),
    yielding the registry, and guarantees exact LIFO restoration using ContextVar.reset(token)
    in finally.
    
    Concurrency & Isolation Invariants:
    - Thread & Task Isolation: Overrides are bound to the caller's execution Context.
      Concurrent threads or asyncio tasks with independent contexts do NOT observe
      or interfere with each other's overrides.
    - Race-Free Restoration: Uses ContextVar token-based reset. Interleaving threads
      or tasks cannot overwrite or corrupt each other's prior registry pointers.
    - Zero Process-Default Mutation: An override never modifies _default_registry.
      The process default remains pristine throughout test execution.
    - Nested Overrides: Nested override blocks within the same context restore in
      strict LIFO order via distinct ContextVar tokens.
    - Child Propagation: Asyncio tasks spawned within an active override inherit a
      context copy, while new threads default to the process default unless context
      is copied explicitly via contextvars.copy_context().
    """
    if not isinstance(registry, DataFrameRegistry):
        raise TypeError("registry must be an instance of DataFrameRegistry")
    token = _registry_override.set(registry)
    try:
        yield registry
    finally:
        _registry_override.reset(token)
```

### 3.4 Invariant Guarantees & Atomic CP1 Caller Migration
1. **Dynamic Resolution Invariant**: Canonical tools in `idd_tools.py` obtain the active registry dynamically via `get_global_registry()`. Tools must NEVER capture a stale instance at module import time.
2. **Façade Compatibility Contract**:
   - **Ordinary Reads** (`idd_core.global_df_registry.get_dataframe(...)`): Supported transparently via `idd_core.__getattr__`, which dynamically delegates to `idd_registry.get_global_registry()`.
   - **In-place Mutations** (`global_df_registry.register_dataframe(...)`, `.clear()`): Supported transparently on whichever registry is active in the caller's context.
   - **Elimination of Legacy Direct Assignment**:
     - Direct attribute assignment on the module (`idd_core.global_df_registry = replacement`) is unsupported under PEP 562 because setting an attribute creates a module `__dict__` entry that shadows `__getattr__`, severing tools from test overrides.
3. **Atomic Caller Migration in CP1 (Slice 1)**:
   - Rather than postponing caller updates to CP2, all test fixtures and callers performing direct assignment:
     - `tests/conftest.py:54` (`core.global_df_registry = fresh`)
     - `tests/unit/test_handle_tool_errors.py:57` (`idd_core.global_df_registry = small_reg`)
     **MUST be migrated atomically in CP1** to `with override_global_registry(...)`.
   - This ensures that when the `idd_core` façade is introduced in CP1, zero tests rely on module attribute shadowing, and unit tests prove actual façade/canonical identity from day one.
4. **Required Future Parity Tests (CP1 Gate Deliverables)**:
   - *Test A (Façade & Canonical Identity)*: `idd_core.global_df_registry is idd_registry.get_global_registry()`.
   - *Test B (Tool Dynamic Observation)*: Tools calling `get_global_registry()` immediately observe context override.
   - *Test C (Nested Overrides & Token Reset)*: Nested blocks restore outer context in strict LIFO order.
   - *Test D (Exception Safety)*: Exceptions raised inside override blocks guarantee context reset in `finally`.
   - *Test E (Concurrent Thread Isolation)*: Overlapping overrides in concurrent threads do not leak across threads or corrupt the process default.
   - *Test F (Asyncio Task Isolation)*: Asyncio tasks inherit context copy but mutations do not leak back to parent.
   - *Test G (Process Default Updates)*: `set_global_registry()` updates process default under lock when no override is active.
   - *Test H (User Data Preservation)*: `clear()` evicts LRU cache and tracking maps without deleting user files.

---

## 4. Structured Error Contract & Exception Sanitization

### 4.1 Production Error Schema
The migration strictly preserves the production error dictionary schema established in notebook Cell 32 and asserted in `tests/unit/test_tool_error_handling.py`:
```python
def _tool_error(operation: str, reason: str, action: str) -> dict:
    return {
        "status": "error",
        "operation": operation,
        "reason": reason,
        "action": action,
    }
```

### 4.2 Exception Sanitization
Internal exception tracebacks are logged locally but never exposed in agent-visible tool return payloads:
```python
def _tool_failure(operation: str, action: str, exc: Exception | None = None) -> dict:
    if exc is not None:
        logging.exception("%s failed: %s", operation, exc)
    return _tool_error(
        operation,
        "An unexpected data-processing failure occurred.",
        action,
    )
```

### 4.3 Test Migration for String-Asserting Tests
`tests/unit/test_handle_tool_errors.py` currently asserts string returns (`assert isinstance(result, str)`).
During Slice 2, these assertions will be updated to assert dictionary structure:
```python
# Before (testing old core string):
assert "Error: DataFrame with ID" in result
# After (testing canonical production dict):
assert result["status"] == "error"
assert "not found" in result["reason"]
```

---

## 5. Modernizing the Seven Integration Tests (No-Key Execution)

### 5.1 Root Causes of Existing Skips
1. `tests/integration/test_graph_compile.py` (4 skips):
   - Fixture `api_key` skips whenever `OPENAI_API_KEY` is unset.
   - Fixture `compiled_graph` calls `core.build_graph(api_key=api_key)`.
   - Test `test_required_nodes_present` asserts legacy node `"report_generator"`, which was decomposed into 5 reporting nodes.
2. `tests/integration/test_routing.py` (3 skips):
   - Test 1 checks `core.AGENT_MEMBERS` or `core.members` (absent). Calculates `unknown` routes but does not assert them.
   - Test 2 looks for lowercase `core.options` (plan previously had uppercase `AGENT_OPTIONS`).
   - Test 3 expects `AgentMembers.next`, but `AgentMembers` in `idd_core.py` has field `agent_type`.

### 5.2 No-Key Testable Graph Factory Strategy
`idd_graph.build_graph(llm_factory=None)` supports dependency injection:
- In production: defaults to `MyChatOpenai()`.
- In tests: accepts a mock/stub LLM factory (`FakeListChatModel` or `MagicMock`) that constructs nodes without contacting OpenAI APIs or requiring an API key.
- Node set asserted:
  `{"supervisor", "initial_analysis", "data_cleaner", "analyst", "visualization", "viz_worker", "viz_join", "viz_evaluator", "report_orchestrator", "report_section_worker", "report_join", "report_packager", "file_writer", "EMERGENCY_MSG", "FINISH"}`.
- Unknown route assertion: `assert not unknown_routes`.
- Router model: `Router` (with field `next: Literal[...]`) exported for routing schema validation.

---

## 6. Real Local and Colab Bootstrap Contract

The IDD architecture supports three execution environments while preserving the **strict 99-cell invariant**:

### 6.1 Execution Modes & Fail-Closed Provenance Contract
1. **Local Workstation / CI**:
   - Repository root is in `sys.path`. Canonical modules (`idd_models.py`, `idd_registry.py`, etc.) are imported directly.
2. **Colab with Cloned Repository**:
   - Notebook Cell 2 executes repository setup:
     ```python
     import os, sys
     from pathlib import Path
     repo_path = Path.cwd()
     if (repo_path / "idd_models.py").exists():
         if str(repo_path) not in sys.path:
             sys.path.insert(0, str(repo_path))
     ```
3. **Colab / Ephemeral Environment Provenance Specification (Pre-CP4 Design)**:
   - To guarantee version integrity without assuming a pre-existing repository clone, the module acquisition architecture must enforce:
     a. **Pinned Version Manifest**: A machine-readable manifest (`idd_manifest.json`) specifying expected SHA-256 digests for all canonical files (`idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, `idd_graph.py`, `idd_core.py`).
     b. **Cryptographic Verification**: Hashes are verified before modules are added to `sys.path` or imported.
     c. **Fail-Closed Behavior**: If any canonical module is missing, modified, or has a mismatched digest, Cell 2 raises an explicit error and aborts notebook initialization.
     d. **No Moving Target**: Silent downloading of unpinned `main` code is strictly prohibited.
     e. **Distribution Mechanism Decision**: The exact acquisition vehicle (e.g. pinned commit tarball/shallow clone vs GitHub release asset vs pre-built wheel) is an explicit pre-CP4 design decision (marked **REQUIRES DECISION** in ADR).
   - Preserves the strict 99-cell notebook invariant.

---

## 7. Anti-Drift Guard Specification (`test_core_parity_guard.py`)

A dedicated no-key test suite `tests/unit/test_core_parity_guard.py` will enforce bidirectional synchronization:

### 7.1 Static AST Checks
1. **Import Verification**: Verifies Cells 16, 19, 22, 32, 60 import canonical symbols from root modules.
2. **No Subsequent Shadowing / Redefinition**: Parses subsequent cells to ensure no cell silently redefines canonical classes (e.g. `Plan`, `CompletedStepsAndTasks`, `State`, `handle_tool_errors`).
3. **99-Cell Invariant**: Enforces `len(cells) == 99`.

### 7.2 Bounded Runtime Identity Probes
Executes compiled cell objects in an isolated test harness without paid LLM calls to assert:
- `id(notebook_Plan) == id(idd_models.Plan)`.
- `id(notebook_State) == id(idd_state.State)`.
- `id(notebook_handle_tool_errors) == id(idd_tools.handle_tool_errors)`.

---

## 8. Control-Plane & Agent Guidance Alignment (Issue #144 in Same Implementation PR)

In accordance with user authorization, Issue #144 surfaces must be updated in the **SAME implementation PR** as Issue #140:

### 8.1 Changed-File Map for Control-Plane Surfaces
- `AGENTS.md`: Update repo map to document `idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, `idd_graph.py`, `idd_core.py`.
- `GEMINI.md`: Update architecture overview and subagent coordination protocols.
- `.agents/rules/code-architecture.md`: Document canonical root modules as single source of truth.
- `.agents/rules/validation-and-gates.md`: Add `test_core_parity_guard.py` to mandatory gate commands.
- `.agents/agents/idd-repo-coordinator/agent.md`: Update coordinator role and module ownership.
- `.agents/agents/notebook-patch-specialist/agent.md`: Update patcher guidance on canonical module imports.
- `.github/copilot-instructions.md`: Synchronize Copilot instructions with canonical modules.
- `.github/instructions/backend.instructions.md`: Update backend module guidelines.
- `.github/instructions/notebook.instructions.md`: Update notebook import guidelines.
- `README.md`: Update architecture and development documentation.

---

## 9. Incremental Implementation Slices & Rollback Gates

```
[Slice 0: CP0 Baseline & Plan] ──> [Slice 1: Models & Registry] ──> [Slice 2: Tools & Errors]
                                                                            │
                                                                            ▼
[Slice 5: Keyed Live Proof]   <── [Slice 4: Notebook & Docs]    <── [Slice 3: State & Graph]
```

### Slice 0: Baseline & Diagnostics (CP0 - Current Status)
- **Scope**: Native Project #7 setup (#151); empirical diagnostic suite [`tools/diagnostics/reproduce_drift.py`](../../tools/diagnostics/reproduce_drift.py); AST scanner [`tools/diagnostics/probe_notebook_cells.py`](../../tools/diagnostics/probe_notebook_cells.py); parity inventory [`docs/architecture/idd-core-parity-inventory.md`](../architecture/idd-core-parity-inventory.md); migration plan [`docs/plans/idd-canonical-core-migration.md`](idd-canonical-core-migration.md).
- **Gates**: No-key baseline measured (463 passed, 9 skipped); 0 bytes production churn; draft PR #155 opened.
- **Review Gate**: **STOP FOR CP0 ARCHITECTURAL RE-REVIEW**.

### Slice 1: Models & Registry Parity (CP1)
- **Scope**:
  - Implement `idd_models.py` with `BaseNoExtrasModel`, `PlanStep`, `Plan`, `CompletedStepsAndTasks`, `Section`, `Router`.
  - Fix RC-2 sorting split-brain (`CompletedStepsAndTasks` returning sorted `dedup_list`).
  - Fix Claim 2 (`Plan` preserving caller-supplied `plan_version`).
  - Fix Claim 4 (reject duplicate numeric step numbers in `CompletedStepsAndTasks`).
  - Implement `idd_registry.py` with `_read_df`, `clear()`, `get_global_registry`, `set_global_registry`, and `override_global_registry` (`ContextVar` design).
  - Update `idd_core.py` to re-export from `idd_models` and `idd_registry` with dynamic `__getattr__` routing.
  - **Atomic Caller & Fixture Migration**: Atomically migrate all direct-assignment test fixtures (`tests/conftest.py:54` and `tests/unit/test_handle_tool_errors.py:57`) in the same CP1 PR to `with override_global_registry(...)`.
- **Validation Gate**:
  - `python -m pytest tests/unit/test_models.py tests/unit/test_registry.py -v` *(Note: `tests/unit/test_dataframe_registry.py` is a proposed future test deliverable in CP2)*
  - Tests A through H verified green (including concurrent thread isolation, nested token reset, and façade/canonical identity).
  - Claims 1, 2, 3, 4, 5, 8 verified green in unit test suite.
- **Rollback Criteria**: Any regression in Pydantic validation, registry caching, or thread isolation rolls back Slice 1.

### Slice 2: Tools & Error Handling Hardening (CP2)
- **Scope**:
  - Implement `idd_tools.py` with signature-aware `@handle_tool_errors`, production structured dictionary schema `{"status": "error", "operation": ..., "reason": ..., "action": ...}`, `_tool_failure` sanitization, and multi-root `_resolve_artifact_path`.
  - Fix `validate_dataframe_exists` to use `registry._read_df` (fixing Claim 8).
  - Update `idd_core.py` to re-export from `idd_tools`.
  - Update `tests/unit/test_handle_tool_errors.py` to assert dictionary schema.
- **Validation Gate**:
  - `python -m pytest test_error_handling_framework.py -v` passes **16/16** (resolving `test_integration_with_different_function_signatures`).
  - Claims 6, 7, 9 verified green.
- **Rollback Criteria**: Any breakage in tool execution or path resolution rolls back Slice 2.

### Slice 3: State, Graph & Integration Modernization (CP3)
- **Scope**:
  - Implement `idd_state.py` with 70-field `State` and custom reducers.
  - Implement `idd_graph.py` with `build_graph(llm_factory=None)`, `AGENT_MEMBERS`, `AGENT_OPTIONS`, `options`.
  - Modernize `tests/integration/test_graph_compile.py` and `tests/integration/test_routing.py` to run no-key against 15-node topology.
  - Update `idd_core.py` to re-export from `idd_state` and `idd_graph`.
- **Validation Gate**:
  - `python -m pytest tests/integration/test_graph_compile.py tests/integration/test_routing.py -v` passes 100% (0 skips).
  - Overall suite passes **470 passed, 2 skipped** (remaining skips: Run 88 fixture, pyarrow).
- **Rollback Criteria**: Any graph compilation error or dead-end node rolls back Slice 3.

### Slice 4: Notebook Wiring, Anti-Drift Guard & Issue #144 Guidance (CP4)
- **Scope**:
  - Update `_patch_notebook.py` sentinels to inject imports from canonical modules into Cells 16, 19, 22, 32, 60.
  - Regenerate notebook: `python _patch_notebook.py`.
  - Implement `tests/unit/test_core_parity_guard.py` (future test).
  - Update agent guidance, rules, and docs (Issue #144 changed-file map).
- **Validation Gate**:
  - Full no-key test suite passes cleanly:
    - `python -m pytest test_validate_run.py tests/unit tests/integration -q`
    - `python -m pytest tests/unit/test_core_parity_guard.py -v`
    - `python -m pytest test_intelligent_data_detective.py -v`
    - `python -m pytest test_prompt_formatting.py test_prompt_template_fixes.py -v`
    - `python -m pytest test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py -v`
    - `python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb`
    - `python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb`
- **Rollback Criteria**: Notebook cell count != 99 or any anti-drift guard failure.

### Slice 5: Keyed Proof & Production Verification Gate (CP5)
- **Scope**:
  - When live paid-model testing is authorized by the user:
  - Execute end-to-end run: `python run_notebook_live.py`.
  - Enforce twin gates:
    - `python validate_run.py --latest --log-path notebook_run_log.txt --window 180` (12/12 production criteria).
    - `python validate_artifact_quality.py --latest` (9/9 artifact quality criteria).

---

## 10. Architectural Decision Record (ADR)

| Decision Item | Status | Decision & Rationale | Tradeoffs & Alternatives |
| :--- | :---: | :--- | :--- |
| **1. Module Architecture** | **PROPOSED** | Root-level Python modules (`idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, `idd_graph.py`, `idd_core.py`). | Avoids creating a `src/` directory (strictly forbidden by repo rules) and prevents package-nesting import complexities. |
| **2. Registry Ownership & Overrides** | **PROPOSED** | Canonical registry in `idd_registry.py` with `contextvars.ContextVar`-based `override_global_registry` and thread-safe process default (`_default_registry`). Atomic migration of direct-assignment fixtures (`conftest.py`, `test_handle_tool_errors.py`) in CP1. | Process-global pointer swapping fails on concurrent thread interleaving. ContextVar token reset provides true thread and task isolation. Direct module assignment is eliminated in CP1 rather than delayed to CP2. |
| **3. Plan Version Preservation** | **PROPOSED** | Monotonic class counter assigns version by default, but caller-supplied `plan_version` is preserved when explicitly provided. | Enables clean deserialization and state restoration while maintaining globally monotonic auto-versioning for new plans. |
| **4. Colab Module Distribution** | **REQUIRES DECISION** | Fail-closed manifest and digest verification contract specified for Cell 2. Rejects silent downloading of unpinned `main` code. | Exact module acquisition mechanism (shallow clone vs release asset vs pinned curl) deferred to pre-CP4 decision. |
| **5. Façade Compatibility** | **PROPOSED** | Keep `idd_core.py` as a 100% backwards-compatible re-export façade for symbols and reads. | Existing unit tests run without modification; direct attribute assignment callers migrate to explicit override context manager. |
| **6. No-Key Graph Construction** | **PROPOSED** | Dependency-injected `build_graph(llm_factory=None)` supporting stub LLMs for testing. | Enables 100% of graph topology and routing assertions to run without requiring an OpenAI API key. |
| **7. Administrative Resolution of #147** | **REQUIRES DECISION** | Keep Issue #147 open administratively until live multi-agent proof confirms full pipeline integration. | Tool-level `delete_rows` is verified working under PR #149, but full pipeline verification requires a live run (CP5). |

---
*Migration plan prepared for Checkpoint 0 review per Issue #150 specifications.*
