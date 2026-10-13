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
- **Explicit Source Copy-on-Write Integrity (Initial & Subsequent)**:
  - Both initial registration and subsequent updates of an in-memory DataFrame with an explicit source path preserve the external user file byte-for-byte unchanged while writing updated/transformed working data to registry-owned backing storage in `self.data_dir`.
  - Cache evictions reload from the registry-owned backing path (`reload_path`), guaranteeing data freshness without corrupting external inputs or discarding in-memory transformations.
  - Atomic staged writes (`.tmp` + `os.replace`) ensure registry state is never corrupted if persistence fails.
  - Explicit accessors: `get_source_path_from_id()` and `get_backing_path_from_id()`, with legacy `get_raw_path_from_id()` returning source path if explicit, else reload path.
- **Lossless Backing Persistence & Representation Fidelity**:
  - Uses `.pkl` snapshots for internal registry-owned backing storage in `self.data_dir` (`_safe_backing_filename`), guaranteeing exact round-trip fidelity for mixed column label types (`0: int` vs `"0": str`), timezone-aware datetimes (`datetime64[ns, UTC]`), nontrivial indexes (MultiIndex, custom named indexes), and nullable/categorical dtypes across cache eviction and reload.
  - External user-supplied files (`raw_path`) retain their caller-specified format (.csv, .json, .parquet) and remain byte-for-byte untouched.
- **Directory Containment & Unicode Safety**:
  - Logical `df_id` values are decoupled from physical filenames via deterministic SHA-256 digest + alphanumeric sanitization (`_safe_backing_filename`), preventing path injection.
  - Uses `errors='surrogatepass'` encoding so distinct Python strings containing Unicode surrogate code points (e.g. `\ud800` vs `\udc00`) never collapse to identical digest filenames.
  - Enforces strict directory containment (`_get_owned_backing_path`), raising `DataFrameRegistryError` if a path escapes `self.data_dir`.
  - Traversal strings (`../`, `../../`), subdirectories (`folder/child`), Windows absolute/relative paths (`C:\temp`, `..\outside`), and root paths (`/absolute/path`) are fully contained within owned storage.
- **Multi-Format Persistence & Reload**: Supports `.csv`, `.parquet`, `.pkl`, `.json`.
- **Filesystem Safety**: `clear()` resets in-memory cache and tracking maps, but never deletes underlying files on disk.
- **ContextVar Override Engine**:
  - `get_global_registry()` resolves active context override or thread-safe process default.
  - `set_global_registry(registry)` safely updates process default under lock.
  - `override_global_registry(registry)` context manager using `ContextVar.set()` and token-based reset in `finally`.
  - Guarantees strict LIFO nesting, thread isolation, and asyncio task inheritance.

### 3. Compatibility Bridge & Legacy Validation (`idd_core.py`)
- **Authoritative Reload Delegation**: `validate_dataframe_exists()` now delegates directly to `reg.get_dataframe(df_id, load_if_not_exists=True)` instead of manually reading stale source paths with `pd.read_csv()` and overwriting working backing representations.
- **Contract Preservation**: Preserves the legacy boolean contract (`True`/`False`), supports all registered reload formats (.csv, .pkl, .json, .parquet), and prevents decorated tools (`@handle_tool_errors`) from silently rolling back working DataFrames on LRU cache eviction.
- **Symbol Re-exports**: Re-exports all canonical symbols from `idd_models` and `idd_registry`.
- **Dynamic Facade**: Exposes backward-compatible `global_df_registry` facade via module-level `__getattr__` delegating dynamically to `get_global_registry()`.

### 4. CP0 Diagnostic Alignment (`reproduce_drift.py` & `test_cp0_diagnostics.py`)
- **Claim 2 Dynamic High-Water Mark**: Evaluates approved NEW vs. RESTORE lifecycle contracts. Dynamically sets `target_restore_v = v2 + 100`, guaranteeing it strictly exceeds pre-restoration allocator state regardless of test order. Verifies allocator advances > `target_restore_v`, step synchronization, and revalidation idempotence. Reports `OBSERVED INVARIANT` (`is_defect=False`) on healthy code.
- **Claim 8 Format Support Resolution**: Dynamically verifies that `validate_dataframe_exists` and `get_dataframe(load_if_not_exists=True)` succeed across all registered formats (.csv, .pkl, .json) upon cache eviction, resolving the historical defect and reporting `OBSERVED INVARIANT` (`is_defect=False`).
- **Regression Tests**: Added simulated regression tests for broken restore, broken new-plan allocation, broken high-water advancement (Claim 2), and broken non-CSV validation fallback (Claim 8) to `test_cp0_diagnostics.py`.

### 5. Verification Results
- **Unit & Integration Suite**: 528 passed, 9 skipped in 27.48s (`test_validate_run.py`, `tests/unit`, `tests/integration`).
- **Models Test Suite**: 29 passed in 2.36s (`tests/unit/test_models.py`).
- **Registry Test Suite**: 70 passed, 1 skipped in 4.33s (`tests/unit/test_registry.py` including 10 lossless persistence tests in `TestLosslessDataFramePersistence`).
- **Tool Error Handling Suite**: 24 passed in 3.65s (`tests/unit/test_handle_tool_errors.py`).
- **CP0 Diagnostics Suite**: 17 passed in 25.02s (`tools/diagnostics/test_cp0_diagnostics.py`).
- **Drift Probes**: 12/12 evaluated cleanly (`tools/diagnostics/reproduce_drift.py --json`: 5 reproduced, 6 observed invariants, 1 static only, 0 blocked).
- **Core & Memory Suites**: 110 passed in 5.81s (`test_intelligent_data_detective.py`, `test_memory_categorization.py`, `test_memory_integration.py`, `test_memory_lifecycle.py`, `test_validate_notebook_integrity.py`).
- **Prompt Suites**: 17 passed in 2.72s (`test_prompt_formatting.py`, `test_prompt_template_fixes.py`).
- **Error Handling Suite**: 15 passed, 1 deselected in 0.88s (`test_error_handling_framework.py`).
- **Notebook Integrity & State Graph**:
  - `validate_notebook_integrity.py`: PASSED (99 cells, all code cells compiled cleanly).
  - `validate_graph.py`: PASSED (15 nodes, 33 edges, 0 dead-ends, 70 state fields).
- **Notebook Churn**: 0 lines modified in `IntelligentDataDetective_beta_v5_patched.ipynb`, `IntelligentDataDetective_beta_v5.ipynb`, or `_patch_notebook.py`.
- **Lint**: flake8 clean (0 errors) on canonical modules and unit tests.

Refs #156
Refs #140, #144, #145
