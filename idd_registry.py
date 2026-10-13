"""
idd_registry.py — Canonical DataFrameRegistry & ContextVar Override Engine.

Home of the thread-safe LRU DataFrameRegistry and context-isolated registry management:
- DataFrameRegistry (LRU cache, multi-format reload, disk persistence, user file safety)
- DataFrameRegistryError
- get_global_registry() -> active context override or thread-safe process default
- set_global_registry(registry) -> update process default under lock
- override_global_registry(registry) -> contextlib manager using ContextVar token reset

Architecture: Checkpoint 1 (Issue #156 / Refs #140, #144, #145).
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import os
import re
import stat
import tempfile
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Default Working Directory Configuration
# ---------------------------------------------------------------------------

WORKING_DIRECTORY = Path(
    os.environ.get("IDD_WORKING_DIR", tempfile.gettempdir())
).resolve()


# ---------------------------------------------------------------------------
# Registry Exception Class
# ---------------------------------------------------------------------------

class DataFrameRegistryError(Exception):
    """Exception raised for errors in the DataFrameRegistry."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return self.message

    def to_dict(self) -> Dict[str, str]:
        return {"error": self.message}


# ---------------------------------------------------------------------------
# Canonical DataFrameRegistry
# ---------------------------------------------------------------------------

class DataFrameRegistry:
    """Thread-safe LRU cache and persistent registry for DataFrames.

    Invariants & Storage Ownership Model:
    1. Single Source of Truth: All datasets are referenced strictly by string df_id.
    2. LRU In-Memory Caching: Bounded by `capacity`; evictions remain loadable from disk.
    3. Multi-Format Reload: Supports .csv, .parquet, .pkl, .pickle, .json.
    4. Per-Instance Storage Isolation & Secure Private Storage:
       - Default storage (`data_dir` is None): Allocates an exclusively owned private
         directory via `tempfile.mkdtemp(prefix="idd_registry_", dir=str(WORKING_DIRECTORY))`
         with restrictive 0700 permissions (drwx------) on POSIX. On Windows, relies on NTFS
         DACL inheritance within %TEMP%, ensuring independent registries registering identical
         df_ids never collide and private snapshots cannot be accessed by unauthorized local users.
       - Configured storage (`data_dir` is provided): Validated against untrusted locations.
         Rejects filesystem volume roots, non-directories, and on POSIX systems, rejects directories
         that are world-writable (S_IWOTH) or not owned by the current process UID.
         On Windows, filesystem security relies on NTFS Access Control Lists (ACLs); when configuring
         custom multi-user storage, administrators must ensure NTFS DACLs exclude unauthorized principals.
    5. Storage Ownership & Copy-on-Write Model:
       - External source provenance (`source_path`): Original user-supplied file path;
         NEVER modified or overwritten on DataFrame updates.
       - Registry-owned backing storage (`reload_path`): Bounded persisted representation
         in self.data_dir used to restore working DataFrames after cache eviction.
         Persisted losslessly as `.pkl` snapshots within trusted owned storage to ensure
         exact round-trip fidelity for column label types, non-default indexes, timezone
         metadata, and nullable/categorical dtypes. When a DataFrame is updated, it is persisted
         to registry-owned storage, leaving external source files byte-for-byte unchanged.
       - Published artifacts: Output documents/charts created via _resolve_artifact_path().
    6. Filesystem Security, Containment & Permission Discipline:
       - Logical `df_id` strings are decoupled from physical filenames using deterministic
         cryptographic digests and path sanitation.
       - Automatically generated backing paths are strictly verified to reside within
         self.data_dir, preventing path traversal across POSIX and Windows conventions.
       - Staging files and backing snapshots are created with restrictive permissions
         (mode 0600 on POSIX where supported) to guarantee confidentiality and integrity.
    7. File Safety Invariant: `clear()` resets in-memory cache and tracking maps, but NEVER
       deletes user files or published artifacts from disk.
    """

    def __init__(
        self,
        capacity: int = 20,
        data_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.df_id_to_raw_path: Dict[str, str] = {}
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.capacity: int = capacity
        self.data_dir: Path = self._validate_and_secure_data_dir(data_dir)

    def _validate_and_secure_data_dir(
        self, data_dir: Optional[Union[str, Path]]
    ) -> Path:
        """Establish or validate a secure, private storage directory.

        Security Policy:
        1. Default storage (data_dir is None):
           Creates an exclusively owned private directory via `tempfile.mkdtemp`
           with permissions 0700 (drwx------) on POSIX. On Windows, relies on NTFS
           DACL inheritance within %TEMP% restricting access to the current user principal.
        2. Explicitly configured storage (data_dir is provided):
           - Rejects filesystem volume roots (e.g. '/' or 'C:\\').
           - If directory does not exist, creates it and enforces 0700 permissions on POSIX.
           - If directory exists, verifies it is a directory. On POSIX, validates that:
             a. It is owned by the current process UID (rejects unowned/shared dirs).
             b. It is NOT world-writable (rejects world-writable dirs like shared /tmp).
             c. If group-writable, it belongs to the current process GID.
           - On Windows, verifies the directory is valid and not a volume root. Documents
             trust assumptions: filesystem security relies on NTFS ACLs protecting the caller's
             configured location.
        """
        if data_dir is None:
            WORKING_DIRECTORY.mkdir(parents=True, exist_ok=True)
            created = tempfile.mkdtemp(prefix="idd_registry_", dir=str(WORKING_DIRECTORY))
            resolved = Path(created).resolve()
            if os.name == "posix":
                try:
                    os.chmod(resolved, 0o700)
                except OSError:
                    pass
            self._is_owned_storage = True
            return resolved

        path = self._norm_path(data_dir)
        self._is_owned_storage = False

        # Reject filesystem root (e.g. "/" on POSIX, "C:\" on Windows)
        if path == path.parent:
            raise DataFrameRegistryError(
                f"Configured data_dir cannot be a filesystem root: '{path}'"
            )

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            if os.name == "posix":
                try:
                    os.chmod(path, 0o700)
                except OSError:
                    pass
        else:
            if not path.is_dir():
                raise DataFrameRegistryError(
                    f"Configured data_dir is not a directory: '{path}'"
                )
            if os.name == "posix":
                st = path.stat()
                # 1. Ownership check: Must be owned by current user
                current_uid = os.getuid()
                if st.st_uid != current_uid:
                    raise DataFrameRegistryError(
                        f"Unsafe data_dir: directory '{path}' is owned by UID {st.st_uid}, "
                        f"expected current UID {current_uid}. Refusing to trust shared directory."
                    )
                # 2. World-writable check: Must NOT be world-writable
                if st.st_mode & stat.S_IWOTH:
                    raise DataFrameRegistryError(
                        f"Unsafe data_dir: directory '{path}' is world-writable (mode {oct(st.st_mode)}). "
                        f"Refusing to trust shared writable directory for pickle storage."
                    )
                # 3. Group-writable check: If group-writable, must match current GID
                if (st.st_mode & stat.S_IWGRP) and hasattr(os, "getgid"):
                    current_gid = os.getgid()
                    if st.st_gid != current_gid:
                        raise DataFrameRegistryError(
                            f"Unsafe data_dir: directory '{path}' is group-writable by GID {st.st_gid}, "
                            f"expected current GID {current_gid}. Refusing to trust shared writable directory."
                        )

        return path

    def _norm_path(self, p: Union[str, Path]) -> Path:
        """Resolve and expand path."""
        return Path(p).expanduser().resolve() if isinstance(p, (str, Path)) else Path(p)

    def _safe_backing_filename(self, df_id: str) -> str:
        """Derive a safe, deterministic, collision-resistant filename within owned storage.

        Decouples user-controlled string df_id from physical filenames to prevent path
        traversal escapes across POSIX and Windows conventions. Uses surrogatepass encoding
        so distinct Python strings containing surrogate characters do not collapse into
        the same digest.

        Uses the `.pkl` extension for internal registry-owned backing snapshots in self.data_dir
        so that cache eviction and subsequent reload losslessly preserve all pandas DataFrame
        types, index structures, timezone metadata, and nullable dtypes across round-trips.
        Internal backing storage is strictly contained within the registry's private data directory;
        user-supplied external files continue to use their respective external formats.
        """
        digest = hashlib.sha256(df_id.encode("utf-8", errors="surrogatepass")).hexdigest()
        safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", df_id)[:24].strip("_")
        if safe_prefix:
            return f"{safe_prefix}_{digest[:16]}.pkl"
        return f"df_{digest[:24]}.pkl"

    def _get_owned_backing_path(self, df_id: str) -> Path:
        """Return and validate a strictly contained backing file path in self.data_dir."""
        filename = self._safe_backing_filename(df_id)
        candidate = (self.data_dir / filename).resolve()
        data_dir_resolved = self.data_dir.resolve()
        try:
            candidate.relative_to(data_dir_resolved)
        except ValueError:
            raise DataFrameRegistryError(
                f"Path containment violation: backing path '{candidate}' escapes authorized directory '{data_dir_resolved}'"
            )
        if candidate == data_dir_resolved:
            raise DataFrameRegistryError(
                f"Path containment violation: candidate cannot be directory itself: '{candidate}'"
            )
        return candidate

    def _write_df(self, df: pd.DataFrame, path: Path) -> bool:
        """Write DataFrame to disk atomically based on file extension."""
        tmp_path: Optional[Path] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.parent / f".tmp_{uuid.uuid4().hex[:12]}_{path.name}"
            suf = path.suffix.lower()
            if suf == ".csv":
                df.to_csv(tmp_path, index=False)
            elif suf == ".parquet":
                df.to_parquet(tmp_path, index=False)
            elif suf in (".pkl", ".pickle"):
                df.to_pickle(tmp_path)
            elif suf == ".json":
                df.to_json(tmp_path, orient="records")
            else:
                df.to_csv(tmp_path, index=False)
            if os.name == "posix":
                try:
                    os.chmod(tmp_path, 0o600)
                except OSError:
                    pass
            os.replace(tmp_path, path)
            return True
        except Exception as e:
            print(f"Error writing DataFrame to {path}: {e}")
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            return False

    def _read_df(self, path: Path) -> pd.DataFrame:
        """Read DataFrame from disk based on file extension."""
        suf = path.suffix.lower()
        if suf == ".csv":
            return pd.read_csv(path)
        if suf == ".parquet":
            return pd.read_parquet(path)
        if suf in (".pkl", ".pickle"):
            return pd.read_pickle(path)
        if suf == ".json":
            return pd.read_json(path, orient="records")
        return pd.read_csv(path)

    def _touch_cache(self, df_id: str, df: pd.DataFrame) -> None:
        """Update LRU order; evict least recently used if exceeding capacity."""
        self.cache[df_id] = df
        self.cache.move_to_end(df_id)
        if len(self.cache) > self.capacity:
            evicted_id, _ = self.cache.popitem(last=False)
            if evicted_id in self.registry:
                self.registry[evicted_id]["df"] = None

    def write_dataframe_to_csv_file(self, df: pd.DataFrame, file_path: str) -> bool:
        with self._lock:
            try:
                df.to_csv(file_path, index=False)
                return True
            except Exception as e:
                print(f"Error writing DataFrame to {file_path}: {e}")
                return False

    def write_dataframe_to_parquet_file(self, df: pd.DataFrame, file_path: str) -> bool:
        with self._lock:
            try:
                df.to_parquet(file_path, index=False)
                return True
            except Exception as e:
                print(f"Error writing DataFrame to {file_path}: {e}")
                return False

    def write_dataframe_to_pickle_file(self, df: pd.DataFrame, file_path: str) -> bool:
        with self._lock:
            try:
                df.to_pickle(file_path)
                return True
            except Exception as e:
                print(f"Error writing DataFrame to {file_path}: {e}")
                return False

    def write_dataframe_to_json_file(self, df: pd.DataFrame, file_path: str) -> bool:
        with self._lock:
            try:
                df.to_json(file_path, orient="records")
                return True
            except Exception as e:
                print(f"Error writing DataFrame to {file_path}: {e}")
                return False

    def write_dataframe_to_file(self, df: pd.DataFrame, file_path: str) -> bool:
        with self._lock:
            return self._write_df(df, self._norm_path(file_path))

    def register_dataframe(
        self,
        df: Optional[pd.DataFrame] = None,
        df_id: Optional[str] = None,
        raw_path: str = "",
    ) -> Optional[str]:
        """Register a DataFrame or a reference to a disk file."""
        with self._lock:
            if df_id is None:
                df_id = str(uuid.uuid4())

            if df is None and not raw_path:
                print("Either df or raw_path must be provided")
                return None

            # ---------------------------------------------------------------
            # 1. UPDATE EXISTING ENTRY (Re-registration)
            # ---------------------------------------------------------------
            if df_id in self.registry:
                entry = self.registry[df_id]
                if df is not None:
                    # An updated DataFrame is being registered!
                    # Preserve external source file; write to registry-owned backing storage.
                    backing_path = self._get_owned_backing_path(df_id)
                    if not self._write_df(df, backing_path):
                        print(f"Error: failed to persist updated DataFrame '{df_id}' to backing storage")
                        return None

                    # Backing write succeeded; update metadata and in-memory cache atomically
                    entry["df"] = df
                    entry["reload_path"] = str(backing_path)
                    if raw_path:
                        norm_user_path = str(self._norm_path(raw_path))
                        entry["source_path"] = norm_user_path
                        entry["raw_path"] = norm_user_path
                        self.df_id_to_raw_path[df_id] = norm_user_path
                    elif not entry.get("source_path"):
                        entry["raw_path"] = str(backing_path)
                        self.df_id_to_raw_path[df_id] = str(backing_path)

                    self._touch_cache(df_id, df)
                    return df_id

                # df is None: re-registering an explicit source path to reload
                if raw_path:
                    norm_path = self._norm_path(raw_path)
                    if not norm_path.exists():
                        print(f"File {raw_path} does not exist")
                        return None
                    try:
                        loaded = self._read_df(norm_path)
                    except Exception as e:
                        print(f"Error loading DataFrame from {norm_path}: {e}")
                        return None
                    entry["df"] = loaded
                    entry["source_path"] = str(norm_path)
                    entry["reload_path"] = str(norm_path)
                    entry["raw_path"] = str(norm_path)
                    self.df_id_to_raw_path[df_id] = str(norm_path)
                    self._touch_cache(df_id, loaded)
                    return df_id

                return df_id

            # ---------------------------------------------------------------
            # 2. INITIAL REGISTRATION (New entry)
            # ---------------------------------------------------------------
            if raw_path:
                norm_path = self._norm_path(raw_path)
                if df is None:
                    if not norm_path.exists():
                        print(f"File {raw_path} does not exist")
                        return None
                    try:
                        df = self._read_df(norm_path)
                    except Exception as e:
                        print(f"Error loading DataFrame from {norm_path}: {e}")
                        return None
                    if df is None:
                        return None
                    self.registry[df_id] = {
                        "df": df,
                        "source_path": str(norm_path),
                        "reload_path": str(norm_path),
                        "raw_path": str(norm_path),
                    }
                    self.df_id_to_raw_path[df_id] = str(norm_path)
                    self._touch_cache(df_id, df)
                    return df_id
                else:
                    # df is provided with an explicit raw_path
                    if norm_path.exists() and norm_path.is_file():
                        # The source file exists on disk as external user provenance.
                        # Preserve external source file byte-for-byte unchanged.
                        # Persist supplied working DataFrame to registry-owned backing storage.
                        backing_path = self._get_owned_backing_path(df_id)
                        if not self._write_df(df, backing_path):
                            print(f"Error: failed to persist initial DataFrame '{df_id}' to backing storage")
                            return None
                        self.registry[df_id] = {
                            "df": df,
                            "source_path": str(norm_path),
                            "reload_path": str(backing_path),
                            "raw_path": str(norm_path),
                        }
                        self.df_id_to_raw_path[df_id] = str(norm_path)
                        self._touch_cache(df_id, df)
                        return df_id
                    elif not norm_path.exists():
                        # Explicit new file path supplied; persist df to that path
                        if not self._write_df(df, norm_path):
                            return None
                        self.registry[df_id] = {
                            "df": df,
                            "source_path": str(norm_path),
                            "reload_path": str(norm_path),
                            "raw_path": str(norm_path),
                        }
                        self.df_id_to_raw_path[df_id] = str(norm_path)
                        self._touch_cache(df_id, df)
                        return df_id
                    else:
                        print(f"Path {raw_path} exists but is not a regular file")
                        return None

            # Auto-generated backing path
            if df is None:
                return None

            backing_path = self._get_owned_backing_path(df_id)
            if not self._write_df(df, backing_path):
                return None

            self.registry[df_id] = {
                "df": df,
                "source_path": None,
                "reload_path": str(backing_path),
                "raw_path": str(backing_path),
            }
            self.df_id_to_raw_path[df_id] = str(backing_path)
            self._touch_cache(df_id, df)
            return df_id

    def get_dataframe(
        self, df_id: str, load_if_not_exists: bool = False
    ) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame by df_id, optionally reloading from disk if evicted."""
        with self._lock:
            if df_id in self.cache:
                self.cache.move_to_end(df_id)
                return self.cache[df_id]

            info = self.registry.get(df_id)
            if not info:
                return None

            df = info.get("df")
            if df is not None:
                self._touch_cache(df_id, df)
                return df

            if load_if_not_exists:
                reload_p = info.get("reload_path") or info.get("raw_path")
                if not reload_p:
                    return None
                path = self._norm_path(str(reload_p))
                if path.suffix.lower() in (".pkl", ".pickle"):
                    resolved_path = path.resolve()
                    resolved_data_dir = self.data_dir.resolve()
                    if resolved_path.is_relative_to(resolved_data_dir):
                        if resolved_path == resolved_data_dir:
                            raise DataFrameRegistryError(
                                f"Security violation: reload path cannot be storage directory itself: '{resolved_path}'"
                            )
                try:
                    loaded = self._read_df(path)
                except FileNotFoundError:
                    return None
                except Exception as e:
                    print(f"Error loading DataFrame from {path}: {e}")
                    return None
                self.registry[df_id]["df"] = loaded
                self._touch_cache(df_id, loaded)
                return loaded

            return None

    def remove_dataframe(self, df_id: str) -> None:
        """Remove a DataFrame from in-memory cache and registration."""
        with self._lock:
            self.registry.pop(df_id, None)
            self.cache.pop(df_id, None)
            self.df_id_to_raw_path.pop(df_id, None)

    def get_raw_path_from_id(self, df_id: str) -> Optional[str]:
        """Get the disk path associated with a registered DataFrame ID.

        Preserves public source-path provenance compatibility: returns the external
        source path if one was provided upon registration, or the auto-generated
        backing path if none was provided.
        """
        with self._lock:
            info = self.registry.get(df_id)
            if info:
                return info.get("source_path") or info.get("raw_path") or info.get("reload_path")
            return self.df_id_to_raw_path.get(df_id)

    def get_source_path_from_id(self, df_id: str) -> Optional[str]:
        """Get the original external source path associated with a DataFrame ID, if one was provided."""
        with self._lock:
            info = self.registry.get(df_id)
            if info:
                return info.get("source_path")
            return None

    def get_backing_path_from_id(self, df_id: str) -> Optional[str]:
        """Get the registry-owned reload backing path for a DataFrame ID."""
        with self._lock:
            info = self.registry.get(df_id)
            if info:
                return info.get("reload_path")
            return None

    def get_id_from_raw_path(self, raw_path: str) -> Optional[str]:
        """Look up the DataFrame ID registered for a given disk path (source or reload path)."""
        with self._lock:
            target = str(self._norm_path(raw_path))
            for df_id, info in self.registry.items():
                sp = info.get("source_path")
                rp = info.get("reload_path")
                raw_p = info.get("raw_path")
                if sp and str(self._norm_path(sp)) == target:
                    return df_id
                if rp and str(self._norm_path(rp)) == target:
                    return df_id
                if raw_p and str(self._norm_path(raw_p)) == target:
                    return df_id
            for df_id, path in self.df_id_to_raw_path.items():
                if str(self._norm_path(path)) == target:
                    return df_id
            return None

    def has_df(self, df_id: str) -> bool:
        """Check if df_id is registered."""
        with self._lock:
            return df_id in self.registry

    def ids(self) -> List[str]:
        """Return list of all registered DataFrame IDs."""
        with self._lock:
            return list(self.registry.keys())

    def size(self) -> int:
        """Return number of registered DataFrames."""
        with self._lock:
            return len(self.registry)

    def clear(self) -> None:
        """Evict all entries from in-memory cache and tracking maps.

        CRITICAL SAFETY GUARANTEE: Does NOT delete underlying files on disk.
        User data and previously saved artifacts are preserved.
        """
        with self._lock:
            self.registry.clear()
            self.cache.clear()
            self.df_id_to_raw_path.clear()


# ---------------------------------------------------------------------------
# ContextVar-based Concurrent Override Management
# ---------------------------------------------------------------------------

_default_registry: DataFrameRegistry = DataFrameRegistry(capacity=20)
_default_registry_lock: threading.RLock = threading.RLock()
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
