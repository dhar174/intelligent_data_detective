"""
idd_tools.py - Canonical Tool Error Handling, Validation, and Artifact Security.

Part of Checkpoint 2 (CP2) under Issue #140 and Issue #158.
Authoritative source for structured tool errors, signature-aware @handle_tool_errors,
canonical validate_dataframe_exists, and PR #149 artifact path containment.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import re
import tempfile
from pathlib import Path as PathlibPath
from typing import Any, Callable, Optional, Sequence, Union

import pandas as pd

from idd_registry import get_global_registry

logger = logging.getLogger(__name__)

# RFC-compliant URI scheme detection: ^[A-Za-z][A-Za-z0-9+.-]*:
_URI_SCHEME_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")
# Windows drive-letter path pattern: ^[A-Za-z]:[/\\]
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")

# Default working directory fallback
_DEFAULT_WORKING_DIR = PathlibPath(tempfile.mkdtemp(prefix="idd_tools_"))
WORKING_DIRECTORY = PathlibPath(
    os.environ.get("IDD_WORKING_DIR", str(_DEFAULT_WORKING_DIR))
)


# ---------------------------------------------------------------------------
# Structured Tool Error Response Contract
# ---------------------------------------------------------------------------


def _tool_error(
    operation: str,
    reason: str,
    action: str,
) -> dict[str, str]:
    """Return the standardized 4-field user-safe error response used by data tools."""
    return {
        "status": "error",
        "operation": operation,
        "reason": reason,
        "action": action,
    }


def _tool_failure(
    operation: str,
    action: str,
    exc: Exception | None = None,
) -> dict[str, str]:
    """Log diagnostic details locally while keeping unexpected failure messages user-safe."""
    if exc is not None:
        logger.exception("%s failed: %s", operation, exc)
    return _tool_error(
        operation,
        "An unexpected data-processing failure occurred.",
        action,
    )


# ---------------------------------------------------------------------------
# Canonical DataFrame Validation
# ---------------------------------------------------------------------------


def validate_dataframe_exists(df_id: str) -> bool:
    """Validate the existence and non-empty status of a DataFrame in the registry.

    Delegates to the authoritative registry loader get_dataframe(load_if_not_exists=True),
    properly supporting all registered source formats (.csv, .pkl, .parquet, .json)
    across cache evictions without hardcoding pandas file readers or modifying
    source provenance.

    Args:
        df_id: The ID of the DataFrame to validate.

    Returns:
        bool: True if DataFrame exists and is valid (non-empty), False otherwise.
    """
    if not isinstance(df_id, str) or not df_id.strip():
        return False
    try:
        registry = get_global_registry()
        df = registry.get_dataframe(df_id, load_if_not_exists=True)
        return df is not None and not df.empty
    except Exception:
        return False


def _validate_dataframe_for_tool(
    operation: str, df_id: Any
) -> Optional[dict[str, str]]:
    """Validate DataFrame existence in registry strictly for tool execution.

    Distinguishes genuine validation failures (missing/empty DataFrame, invalid ID)
    from unexpected infrastructure/registry exceptions (e.g. RuntimeError).
    Genuine validation failures return a structured _tool_error dictionary.
    Unexpected infrastructure exceptions propagate to caller so they can be
    sanitized and logged via _tool_failure.
    """
    if not isinstance(df_id, str) or not df_id.strip():
        return _tool_error(
            operation,
            "DataFrame ID must be a non-empty string.",
            "Provide the ID of a registered, non-empty DataFrame.",
        )
    registry = get_global_registry()
    df = registry.get_dataframe(df_id, load_if_not_exists=True)
    if df is None or df.empty:
        return _tool_error(
            operation,
            f"DataFrame '{df_id}' was not found or is empty.",
            "Provide the ID of a registered, non-empty DataFrame.",
        )
    return None


# ---------------------------------------------------------------------------
# Signature-Aware @handle_tool_errors Decorator
# ---------------------------------------------------------------------------


def handle_tool_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for consistent error handling and DataFrame validation across tool functions.

    Features:
    - Signature-aware argument binding using inspect.signature.
    - Accurately binds declared 'df_id' parameter regardless of positional or keyword order.
    - Extracts .df_id attribute from bound parameter objects.
    - Preserves optional/defaulted df_id parameter semantics (omitted/None bypasses validation).
    - Entire execution path (binding, extraction, validation, invocation) within exception boundary.
    - Emits structured error dictionary for argument binding mismatches.
    - Preserves function name, docstring, annotations, and inspectable signature.
    - Standardizes known exceptions (FileNotFoundError, KeyError, ValueError, pandas errors).
    - Sanitizes unexpected exceptions via _tool_failure with local diagnostic logging.
    """
    try:
        func_signature = inspect.signature(func)
    except (TypeError, ValueError):
        func_signature = None

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            # 1. Signature-aware argument binding
            bound_args = None
            if func_signature is not None:
                try:
                    bound_args = func_signature.bind(*args, **kwargs)
                except TypeError as exc:
                    return _tool_error(
                        func.__name__,
                        f"Invalid arguments: {exc}",
                        "Check the operation parameters and try again.",
                    )

            # 2. Signature-aware df_id extraction
            df_id = None
            df_id_supplied = False

            if bound_args is not None and "df_id" in bound_args.arguments:
                val = bound_args.arguments["df_id"]
                param = func_signature.parameters.get("df_id")
                # Optional df_id with default None: if explicitly passed None, treat as no df_id supplied
                if (
                    param is not None
                    and param.default is not inspect.Parameter.empty
                    and param.default is None
                    and val is None
                ):
                    df_id_supplied = False
                else:
                    df_id = val
                    df_id_supplied = True
            elif "df_id" in kwargs:
                df_id = kwargs["df_id"]
                df_id_supplied = True
            else:
                # Check for parameter object containing .df_id
                target_obj = None
                if args:
                    target_obj = args[0]
                elif bound_args is not None:
                    for arg_val in bound_args.arguments.values():
                        target_obj = arg_val
                        break

                if target_obj is not None:
                    # In Python, hasattr(o, name) can propagate non-AttributeError exceptions
                    # from descriptor/property getters. Since wrapper is enclosed in a try/except,
                    # any unexpected exception will be safely caught and sanitized by _tool_failure.
                    if hasattr(target_obj, "df_id"):
                        df_id = getattr(target_obj, "df_id")
                        df_id_supplied = True

            # 3. Validate DataFrame for declared or supplied df_id
            if df_id_supplied:
                validation_err = _validate_dataframe_for_tool(func.__name__, df_id)
                if validation_err is not None:
                    return validation_err

            # 4. Invoke wrapped function with standardized error handling
            return func(*args, **kwargs)

        except FileNotFoundError:
            return _tool_error(
                func.__name__,
                "The dataset or required file could not be found.",
                "Check the DataFrame ID and registered source path.",
            )
        except KeyError:
            return _tool_error(
                func.__name__,
                "A requested column or key does not exist.",
                "Check the available column names and try again.",
            )
        except pd.errors.EmptyDataError:
            return _tool_error(
                func.__name__,
                "The dataset contains no rows.",
                "Provide a non-empty DataFrame.",
            )
        except pd.errors.ParserError:
            return _tool_error(
                func.__name__,
                "The dataset could not be parsed.",
                "Check the source file format and reload the dataset.",
            )
        except pd.errors.DtypeWarning:
            return _tool_error(
                func.__name__,
                "The data types are incompatible with this operation.",
                "Use a column with a compatible data type.",
            )
        except ValueError:
            return _tool_error(
                func.__name__,
                "One or more inputs are invalid.",
                "Check the operation parameters and try again.",
            )
        except Exception as exc:
            return _tool_failure(
                func.__name__,
                "Retry the operation; if it continues, check the dataset and logs.",
                exc=exc,
            )

    return wrapper


# ---------------------------------------------------------------------------
# Artifact Path Resolution & PR #149 Security Containment
# ---------------------------------------------------------------------------


def _is_subpath(path: PathlibPath, parent: PathlibPath) -> bool:
    """Return True if path resolves strictly within parent directory."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, Exception):
        return False


def _get_artifacts_base(
    config: Optional[Any] = None, *, create: bool = False
) -> PathlibPath:
    """Resolve artifacts root following strict precedence.

    Resolution order:
    1. config.configurable.runtime.artifacts_dir (RunnableConfig or dict)
    2. Global RUNTIME.artifacts_dir (if defined in sys.modules["__main__"])
    3. os.environ["IDD_ARTIFACTS_DIR"] (if set and non-empty)
    4. WORKING_DIRECTORY / "artifacts" fallback
    """
    # 1. Configurable runtime.artifacts_dir
    try:
        cfg = getattr(config, "configurable", None)
        if cfg is None and isinstance(config, dict):
            cfg = config.get("configurable", {})
        if isinstance(cfg, dict):
            runtime = cfg.get("runtime")
        else:
            runtime = getattr(cfg, "runtime", None)

        if runtime is not None:
            art_dir = getattr(runtime, "artifacts_dir", None)
            if art_dir is None and isinstance(runtime, dict):
                art_dir = runtime.get("artifacts_dir")
            if art_dir:
                base = PathlibPath(art_dir).resolve()
                if create:
                    base.mkdir(parents=True, exist_ok=True)
                return base
    except Exception:
        pass

    # 2. Global RUNTIME object
    try:
        import sys

        main_mod = sys.modules.get("__main__")
        rt = getattr(main_mod, "RUNTIME", None)
        if rt is not None and getattr(rt, "artifacts_dir", None):
            base = PathlibPath(rt.artifacts_dir).resolve()
            if create:
                base.mkdir(parents=True, exist_ok=True)
            return base
    except Exception:
        pass

    # 3. Environment variable override
    env_artifacts = os.environ.get("IDD_ARTIFACTS_DIR")
    if env_artifacts and env_artifacts.strip():
        base = PathlibPath(env_artifacts.strip()).resolve()
        if create:
            base.mkdir(parents=True, exist_ok=True)
        return base

    # 4. Working directory fallback
    base = (PathlibPath(WORKING_DIRECTORY) / "artifacts").resolve()
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def _is_uri_scheme(path_str: str) -> bool:
    """Return True if path_str begins with an RFC-compliant URI scheme or is an

    ambiguous single-character URI / drive-relative path.
    Only unambiguous absolute Windows drive paths (^[A-Za-z]:[/\\]) are exempted.
    """
    if not path_str or not isinstance(path_str, str):
        return False
    s = path_str.strip()
    if not s:
        return False
    if "://" in s:
        return True
    if s.lower().startswith("data:"):
        return True
    match = _URI_SCHEME_PATTERN.match(s)
    if match:
        scheme = match.group(1)
        if len(scheme) == 1:
            rest = s[2:]
            if rest.startswith(("\\", "/")):
                return False
            return True
        return True
    return False


def _resolve_artifact_path(
    file_name: str,
    config: Optional[Any] = None,
    subdir: Optional[str] = None,
    create_parents: bool = True,
) -> PathlibPath:
    """Resolve and validate a file path strictly within the artifacts directory.

    Guarantees:
    - Path traversal attacks (../) and absolute path escapes raise ValueError.
    - Symlink escapes raise ValueError.
    - URI schemes (javascript:, data:, file:, etc.) raise ValueError.
    - Subdirectory escapes raise ValueError.
    - Targeting the artifacts root directory directly raises ValueError.
    - Directories are created ONLY after successful validation (never on rejected paths).

    Args:
        file_name: Target relative or absolute file name.
        config: Optional RunnableConfig or configuration dict.
        subdir: Optional subdirectory under artifacts root.
        create_parents: Whether to create parent directories if valid.

    Returns:
        PathlibPath: Validated resolved path within authorized root.

    Raises:
        ValueError: If file_name is invalid or path escapes authorized root.
    """
    if not file_name or not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name must be a non-empty string.")

    fn_stripped = file_name.strip()
    if _is_uri_scheme(fn_stripped):
        raise ValueError(
            f"Refusing to resolve URI scheme as artifact path: {file_name}"
        )
    if os.name != "nt" and (
        _WINDOWS_DRIVE_PATTERN.match(fn_stripped) or fn_stripped.startswith(r"\\")
    ):
        raise ValueError(f"Refusing to access path outside artifacts root: {file_name}")

    base = _get_artifacts_base(config, create=False)
    if subdir:
        if not isinstance(subdir, str) or not subdir.strip():
            raise ValueError("subdir must be a non-empty string.")
        candidate_base = (base / subdir).resolve()
        if not _is_subpath(candidate_base, base):
            raise ValueError(f"Subdirectory escapes artifacts root: {subdir}")
        base = candidate_base

    candidate = PathlibPath(file_name)
    path = (
        (base / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )

    if not _is_subpath(path, base):
        raise ValueError(f"Refusing to access path outside artifacts root: {path}")

    if path == base:
        raise ValueError(
            "file_name cannot resolve to the artifacts root directory itself."
        )

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path


def is_safe_artifact_path(
    candidate: Union[str, PathlibPath],
    allowed_roots: Sequence[Union[str, PathlibPath]],
    *,
    require_file: bool = False,
) -> bool:
    """Verify that candidate path resolves strictly within at least one authorized root.

    Used by downstream reporting and publication nodes to verify multi-root image
    containment (_allowed_roots = [_artifact_root, _working_dir, _run_root]) without
    weakening single-root containment for tool artifacts.
    """
    if not candidate:
        return False
    try:
        cand_str = str(candidate).strip()
        if _is_uri_scheme(cand_str):
            return False
        if os.name != "nt" and (
            _WINDOWS_DRIVE_PATTERN.match(cand_str) or cand_str.startswith(r"\\")
        ):
            return False

        cand_obj = PathlibPath(candidate)
        for root in allowed_roots:
            if not root:
                continue
            root_path = PathlibPath(root).resolve()
            cand_path = (
                (root_path / cand_obj).resolve()
                if not cand_obj.is_absolute()
                else cand_obj.resolve()
            )
            if _is_subpath(cand_path, root_path):
                if require_file and (
                    not cand_path.is_file() or cand_path.stat().st_size <= 0
                ):
                    continue
                return True
        return False
    except Exception:
        return False
