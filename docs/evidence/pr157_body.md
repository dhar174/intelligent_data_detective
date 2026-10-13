## Summary
Implements Checkpoint 1 (CP1) and resolves all P1/P2 findings from independent review per authoritative Issue #156 (Parent: #140, Companion: #144, #145).

### 1. Canonical Models (`idd_models.py`)
- **`BaseNoExtrasModel`**: Mandatory base contract (`extra='forbid'`, `reply_msg_to_supervisor`, `finished_this_task`, `expect_reply`).
- **`Plan` Lifecycle Contract**:
  - Normal construction (`Plan(...)` / `model_validate(data)`) allocates fresh monotonic version under lock, even if draft input supplied `plan_version=1`.
  - Restoration entrypoint (`Plan.from_persisted_snapshot(data)` / `model_validate(data, context={'restore': True})`) preserves stored snapshot version (e.g. 42). Fully revalidates existing `Plan` instances via `data.model_dump()` through the RESTORE lifecycle, preventing mutated invalid state (e.g. duplicate step numbers, out-of-order steps, corrupted schema fields) from bypassing validation, while synchronizing step versions and leaving the original input object unmutated.
  - Allocator high-water mark policy: advances under lock on restored version only after successful validation so subsequent new plans receive versions > restored, without advancing on failed restoration.
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
- **Private Storage Security Boundary & Permissions**:
  - Secure default directory creation uses `tempfile.mkdtemp(prefix="idd_registry_", dir=str(WORKING_DIRECTORY))` with explicit `0700` (`drwx------`) directory permissions on POSIX, eliminating shared/world-accessible defaults and umask race conditions.
  - Configured directory validation (`_validate_and_secure_data_dir`) rejects filesystem volume roots (`/`, `C:\`), non-directory files, and on POSIX any existing directory that is **group-writable (`stat.S_IWGRP`) or world-writable (`stat.S_IWOTH`)**, regardless of matching UID/GID, plus directories not owned by the current UID (`st.st_uid != os.getuid()`). Owner-only (`0700`) directories remain accepted.
  - **Secure staging writes**: `_write_df()` creates the staging file with `os.open(O_WRONLY | O_CREAT | O_EXCL [| O_NOFOLLOW])` and mode `0600` (normalised with `fchmod`, independent of umask) **before any bytes are serialized**, writes through the open descriptor, then atomically `os.replace()`s it over the target. Failures remove the staging file and leave any existing snapshot untouched. User-facing export methods are unchanged.
  - **Fail-closed pickle deserialization**:
    - *Internal snapshots*: each entry records `reload_is_owned_snapshot`. A `.pkl`/`.pickle` reload of an owned snapshot proceeds only if the recorded path is absolute and resolves (following symlinks) to exactly the digest-derived backing file for that `df_id` inside the validated `self.data_dir`; the backing entry must not be a symlink, `data_dir` must still resolve to its validated location, and on POSIX the snapshot must be owned by the current UID and not group/world-writable. Otherwise `DataFrameRegistryError` is raised **before `pd.read_pickle()`**.
    - *External pickle imports*: `register_dataframe(df=None, raw_path="*.pkl|*.pickle")` is **rejected by default**. Callers must opt in explicitly with `trust_external_pickle=True`; the decision is recorded on the entry (`trusted_external_pickle`), the path is never labelled registry-owned, and later reloads are only allowed from that same trusted source path.
    - A DataFrame written to a *new* external `.pkl` path is reloaded from a registry-owned snapshot, never from the externally written file.
    - CSV, JSON, and Parquet imports are unaffected.
  - Windows security model: POSIX mode bits are not meaningful on NTFS; confidentiality of the storage directory and staging files relies on NTFS DACLs inherited from user `%TEMP%` / `%USERPROFILE%` (or administrator-configured ACLs for custom `data_dir`), documented explicitly in class contracts.
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
- **Claim 8 Format Support Resolution**: Dynamically verifies that `validate_dataframe_exists` and `get_dataframe(load_if_not_exists=True)` succeed across all registered external source formats (.csv, .pkl, .json) upon cache eviction using `df=None` and explicit `trust_external_pickle=True` for the temporary fixture, verifying that reload paths identify the source files rather than private snapshots and testing unmasked reader execution. Reports `OBSERVED INVARIANT` (`is_defect=False`).
- **Regression Tests**: Added simulated regression tests for broken restore, broken new-plan allocation, broken high-water advancement (Claim 2), broken CSV reader detection, broken JSON reader detection, silent registration failure detection, untrusted pickle rejection, and non-CSV validation fallback (Claim 8) to `test_cp0_diagnostics.py`.

### 5. Verification Results
- **Unit & Integration Suite**: 550 passed, 16 skipped in 19.92s (`test_validate_run.py`, `tests/unit`, `tests/integration`).
- **Models Test Suite**: 35 passed in 2.47s (`tests/unit/test_models.py`, including 6 new tests in `TestPlanInstanceRestorationValidation` covering valid instance restoration, nested version sync, duplicate step rejection, schema violations, roundtrip equivalence, and allocator safety).
- **Registry Test Suite**: 94 tests total (`tests/unit/test_registry.py`):
  - On Windows: 86 passed, 8 skipped (6 POSIX permission/ownership tests, 2 optional pyarrow).
  - On Linux (POSIX in WSL): 94 passed, 0 skipped in 5.90s.
  - Covers group-writable directory rejection (mode 0770 rejected, mode 0700 accepted), fail-closed pickle deserialization (boundary confinement, traversal, symlink escapes, external untrusted pickle rejection, trusted opt-in imports, CSV/JSON/Parquet compatibility), and secure staging permissions (mode 0600 during serialization, cleanup on failure).
- **Tool Error Handling Suite**: 24 passed in 2.28s (`tests/unit/test_handle_tool_errors.py`).
- **CP0 Diagnostics Suite**: 21 passed in 4.13s (`tools/diagnostics/test_cp0_diagnostics.py`).
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
