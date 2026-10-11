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
import os
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
    4. Per-Instance Storage Isolation: When data_dir is not explicitly specified,
       each DataFrameRegistry instance allocates an isolated storage directory
       (WORKING_DIRECTORY / f"idd_registry_{instance_id}") so independent registries
       registering identical df_ids never collide or overwrite each other's backing files.
    5. Storage Ownership Separation:
       - Registry-owned backing files: Auto-generated in self.data_dir for in-memory frames.
       - User-supplied source files: Referenced by explicit raw_path; never modified or relocated.
       - Published artifacts: Output documents/charts created via _resolve_artifact_path().
    6. File Safety Invariant: `clear()` resets in-memory cache and tracking maps, but NEVER
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
        if data_dir is not None:
            self.data_dir: Path = self._norm_path(data_dir)
            self._is_owned_storage: bool = False
        else:
            instance_id = uuid.uuid4().hex[:12]
            self.data_dir = (WORKING_DIRECTORY / f"idd_registry_{instance_id}").resolve()
            self._is_owned_storage = True
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _norm_path(self, p: Union[str, Path]) -> Path:
        """Resolve and expand path."""
        return Path(p).expanduser().resolve() if isinstance(p, (str, Path)) else Path(p)

    def _write_df(self, df: pd.DataFrame, path: Path) -> bool:
        """Write DataFrame to disk based on file extension."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            suf = path.suffix.lower()
            if suf == ".csv":
                df.to_csv(path, index=False)
            elif suf == ".parquet":
                df.to_parquet(path, index=False)
            elif suf in (".pkl", ".pickle"):
                df.to_pickle(path)
            elif suf == ".json":
                df.to_json(path, orient="records")
            else:
                df.to_csv(path, index=False)
            return True
        except Exception as e:
            print(f"Error writing DataFrame to {path}: {e}")
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
            path = self._norm_path(raw_path) if raw_path else None

            if df_id in self.registry:
                self.registry[df_id]["df"] = df
                if raw_path and path is not None:
                    self.registry[df_id]["raw_path"] = str(path)
                    self.df_id_to_raw_path[df_id] = str(path)
                elif df is not None:
                    try:
                        default_path = Path(self.data_dir) / f"{df_id}.csv"
                        self._write_df(df, default_path)
                        self.df_id_to_raw_path[df_id] = str(default_path)
                        self.registry[df_id]["raw_path"] = str(default_path)
                    except Exception as e:
                        print(f"Warning: could not persist in-memory DataFrame {df_id}: {e}")
                if df is not None:
                    self._touch_cache(df_id, df)
                return df_id

            if df is None and not raw_path:
                print("Either df or raw_path must be provided")
                return None

            is_auto_path = not bool(raw_path)
            if not raw_path:
                raw_path = str((self.data_dir / f"{df_id}.csv").resolve())

            path = self._norm_path(raw_path)

            if df is None and not path.exists():
                print("Either provide a DataFrame or a valid raw_path")
                return None

            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            if df is not None:
                # If auto-generated path, always write to ensure fresh backing representation.
                # If user-supplied raw_path, write if file doesn't already exist.
                if is_auto_path or not path.exists() or not path.is_file():
                    if not self._write_df(df, path):
                        return None

            if df is None:
                try:
                    df = self._read_df(path)
                except Exception as e:
                    print(f"Error loading DataFrame from {path}: {e}")
                    return None

            if df is None and raw_path is not None and not os.path.exists(raw_path):
                print(f"File {raw_path} does not exist")
                return None

            self.registry[df_id] = {"df": df, "raw_path": str(raw_path)}
            self.df_id_to_raw_path[df_id] = str(raw_path)
            if df is not None:
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
                raw_p = info.get("raw_path")
                if not raw_p:
                    return None
                path = self._norm_path(str(raw_p))
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
        """Get the disk path associated with a registered DataFrame ID."""
        with self._lock:
            return self.df_id_to_raw_path.get(df_id)

    def get_id_from_raw_path(self, raw_path: str) -> Optional[str]:
        """Look up the DataFrame ID registered for a given disk path."""
        with self._lock:
            target = str(self._norm_path(raw_path))
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
