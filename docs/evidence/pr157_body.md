## Summary
Implements Checkpoint 1 (CP1) and resolves all P1/P2 findings from independent review per authoritative Issue #156 (Parent: #140, Companion: #144, #145).

### 1. Canonical Models (`idd_models.py`)
- **`BaseNoExtrasModel`**: Mandatory base contract (`extra='forbid'`, `reply_msg_to_supervisor`, `finished_this_task`, `expect_reply`).
- **`Plan` Lifecycle Contract**:
  - Normal construction (`Plan(...)` / `model_validate(data)`) allocates fresh monotonic version under lock, even if draft input supplied `plan_version=1`.
  - Restoration entrypoint (`Plan.from_persisted_snapshot(data)` / `model_validate(data, context={'restore': True})`) preserves stored snapshot version (e.g. 42).
  - Allocator high-water mark policy: advances under lock on restored version so subsequent new plans receive versions > restored.
  - `Plan.reset_counter(start=1)` for deterministic test isolation.
  - Step versions synchronized to parent in both creation and restoration paths.
  - `Plan._counter` iterator-compliant wrapper for backward compatibility.
  - Revalidation idempotence: `Plan.model_validate(existing_plan)` preserves existing version without advancing the allocator.
- **State Reducer Lifecycle (`_reduce_plan_keep_sorted`)**:
  - Restores merged state using `Plan.from_persisted_snapshot(merged)`, preserving winning plan version without allocating new version numbers during LangGraph state reduction.
- **`CompletedStepsAndTasks`**: RC-2 ascending deduplication return and duplicate step number rejection.

### 2. Canonical DataFrameRegistry (`idd_registry.py`)
- **Thread Safety**: `threading.RLock` protection for cache, registry, and path mappings.
- **Storage & Path Isolation**:
  - Bounded in-memory LRU caching (`capacity`).
  - Instance-isolated default persistence directory (`WORKING_DIRECTORY / f"idd_registry_{uuid.uuid4().hex[:12]}"`), ensuring independent registry instances never collide or share backing paths.
  - Re-registration updates: registering updated data under an existing `df_id` updates backing files on disk, ensuring cache-evicted reloads serve fresh data.
  - Optional `data_dir` constructor argument for explicit storage directory configuration.
- **Multi-Format Persistence & Reload**: Supports `.csv`, `.parquet`, `.pkl`, `.json`.
- **Filesystem Safety**: `clear()` resets in-memory cache and tracking maps, but never deletes underlying files on disk.
- **ContextVar Override Engine**:
  - `get_global_registry()` resolves active context override or thread-safe process default.
  - `set_global_registry(registry)` safely updates process default under lock.
  - `override_global_registry(registry)` context manager using `ContextVar.set()` and token-based reset in `finally`.
  - Guarantees strict LIFO nesting, thread isolation, and asyncio task inheritance.

### 3. CP0 Diagnostic Alignment (`reproduce_drift.py` & `test_cp0_diagnostics.py`)
- **Claim 2 Refinement**: Evaluates approved NEW vs. RESTORE lifecycle contracts, high-water mark advancement, step synchronization, and revalidation idempotence. Reports `OBSERVED INVARIANT` (`is_defect=False`) on healthy code and dynamically reproduces regressions when simulated.
- **Regression Tests**: Added `test_claim_2_approved_lifecycle_returns_observed_invariant`, `test_claim_2_simulated_broken_restore_returns_reproduced`, and `test_claim_2_simulated_broken_new_plan_allocation_returns_reproduced`.

### 4. Atomic Compatibility Migration (`idd_core.py`)
- Re-exports all canonical symbols from `idd_models` and `idd_registry`.
- Exposes backward-compatible `global_df_registry` facade via module-level `__getattr__` delegating dynamically to `get_global_registry()`.
- Replaces internal direct lookups in `validate_dataframe_exists` and `get_global_df_registry` with `get_global_registry()`.
- Updated header docstring explaining transitional role as compatibility façade while planning models and registry live in dedicated canonical modules.
- Preserves legacy string tool-error return contracts and boolean `validate_dataframe_exists` return contracts until CP2.

### 5. Direct Assignment Callers Migration
- `tests/conftest.py`: Migrated `global_registry_reset` fixture from direct module assignment to `with core.override_global_registry(fresh): yield fresh`.
- `tests/unit/test_handle_tool_errors.py`: Migrated eviction tests to `with override_global_registry(small_reg):`.
- `tools/diagnostics/reproduce_drift.py`: Migrated probe 7 and probe 8 to `with idd_core.override_global_registry(reg):`.

### 6. Verification Results
- **Unit & Integration Suite**: 487 passed, 9 skipped in 18.25s (`test_validate_run.py`, `tests/unit`, `tests/integration`).
- **Models Test Suite**: 29 passed in 2.36s (`tests/unit/test_models.py`).
- **Registry Test Suite**: 32 passed, 1 skipped in 2.55s (`tests/unit/test_registry.py`).
- **CP0 Diagnostics Suite**: 14 passed in 3.45s (`tools/diagnostics/test_cp0_diagnostics.py`).
- **Drift Probes**: 12/12 evaluated cleanly (`tools/diagnostics/reproduce_drift.py --json`: 6 reproduced, 5 observed invariants, 1 static only, 0 blocked).
- **Core & Memory Suites**: 110 passed in 3.44s (`test_intelligent_data_detective.py`, `test_memory_categorization.py`, `test_memory_integration.py`, `test_memory_lifecycle.py`, `test_validate_notebook_integrity.py`).
- **Prompt Suites**: 17 passed in 2.25s (`test_prompt_formatting.py`, `test_prompt_template_fixes.py`).
- **Error Handling Suite**: 15 passed, 1 deselected in 0.70s (`test_error_handling_framework.py`).
- **Notebook Integrity & State Graph**:
  - `validate_notebook_integrity.py`: PASSED (99 cells, all code cells compiled cleanly).
  - `validate_graph.py`: PASSED (15 nodes, 33 edges, 0 dead-ends, 70 state fields).
- **Notebook Churn**: 0 lines modified in `IntelligentDataDetective_beta_v5_patched.ipynb`, `IntelligentDataDetective_beta_v5.ipynb`, or `_patch_notebook.py`.
- **Lint**: flake8 clean (0 errors) on canonical modules and unit tests.

Refs #156
Refs #140, #144, #145
