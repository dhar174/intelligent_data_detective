"""
tests/unit/test_canonical_tools.py - Comprehensive Unit Tests for CP2 Canonical Tools.

Validates:
1. Structured tool error schema (4 mandatory fields, sanitization).
2. Signature-aware @handle_tool_errors (df_id first, later, kwarg, kw-only, absent, param object).
3. Canonical validate_dataframe_exists (registry delegation, evicted lossless reload, ContextVar).
4. Secure artifact path resolution (precedence, traversal, symlink escapes, deferred mkdir).
5. Symbol identity with idd_core compatibility façade.
"""

import inspect
import sys
import types
from typing import Optional
import pandas as pd
import pytest

import idd_core
import idd_tools
from idd_registry import DataFrameRegistry, override_global_registry


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Structured Error Response Schema & Sanitization
# ---------------------------------------------------------------------------


class TestStructuredErrorResponses:
    def test_tool_error_schema_fields(self):
        err = idd_tools._tool_error(
            operation="filter_data",
            reason="Column missing",
            action="Check column list",
        )
        assert isinstance(err, dict)
        assert set(err.keys()) == {"status", "operation", "reason", "action"}
        assert err["status"] == "error"
        assert err["operation"] == "filter_data"
        assert err["reason"] == "Column missing"
        assert err["action"] == "Check column list"

    def test_tool_failure_sanitizes_unexpected_exception(self, caplog):
        with caplog.at_level("ERROR"):
            err = idd_tools._tool_failure(
                operation="deep_compute",
                action="Retry the operation",
                exc=RuntimeError("internal secret connection failure: password=123"),
            )
        assert err["status"] == "error"
        assert err["operation"] == "deep_compute"
        assert err["reason"] == "An unexpected data-processing failure occurred."
        assert err["action"] == "Retry the operation"
        # Secret exception message must not leak in returned dict
        assert "password" not in str(err)
        # But diagnostic details must be logged locally
        assert any("deep_compute failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 2. Signature-Aware @handle_tool_errors
# ---------------------------------------------------------------------------


class TestSignatureAwareDecorator:
    def test_first_positional_df_id(self):
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):
            df = pd.DataFrame({"a": [1, 2]})
            reg.register_dataframe(df, "df_first")

            @idd_tools.handle_tool_errors
            def tool_pos(df_id: str, factor: int) -> int:
                return factor * 2

            assert tool_pos("df_first", 3) == 6

            # Error case
            err = tool_pos("nonexistent_df", 3)
            assert isinstance(err, dict)
            assert err["status"] == "error"
            assert "nonexistent_df" in err["reason"]

    def test_later_positional_df_id(self):
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):
            df = pd.DataFrame({"a": [1, 2]})
            reg.register_dataframe(df, "df_later")

            @idd_tools.handle_tool_errors
            def tool_later(prefix: str, count: int, df_id: str) -> str:
                return f"{prefix}_{count}_{df_id}"

            assert tool_later("item", 5, "df_later") == "item_5_df_later"

            err = tool_later("item", 5, "missing_later")
            assert isinstance(err, dict)
            assert err["status"] == "error"
            assert "missing_later" in err["reason"]

    def test_keyword_df_id(self):
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):
            df = pd.DataFrame({"a": [1, 2]})
            reg.register_dataframe(df, "df_kw")

            @idd_tools.handle_tool_errors
            def tool_kw(x: int, df_id: str = "default") -> str:
                return f"{x}_{df_id}"

            assert tool_kw(10, df_id="df_kw") == "10_df_kw"

    def test_keyword_only_df_id(self):
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):
            df = pd.DataFrame({"a": [1, 2]})
            reg.register_dataframe(df, "df_kwonly")

            @idd_tools.handle_tool_errors
            def tool_kwonly(value: float, *, df_id: str) -> str:
                return f"{value}_{df_id}"

            assert tool_kwonly(3.14, df_id="df_kwonly") == "3.14_df_kwonly"

            err = tool_kwonly(3.14, df_id="bad_kwonly")
            assert isinstance(err, dict)
            assert err["status"] == "error"

    def test_function_without_df_id_executes_normally_with_string_args(self):
        """Claim 7 resolution: String arguments in functions without df_id must NOT trigger validation."""

        @idd_tools.handle_tool_errors
        def echo_message(message: str, count: int) -> str:
            return f"{message}: {count}"

        # 'hello' is a string at args[0], but function has no df_id parameter!
        result = echo_message("hello", 42)
        assert result == "hello: 42"

    def test_parameter_object_with_df_id_attribute(self):
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):
            df = pd.DataFrame({"a": [1, 2]})
            reg.register_dataframe(df, "df_obj")

            class RequestObject:
                def __init__(self, df_id: str, task: str):
                    self.df_id = df_id
                    self.task = task

            @idd_tools.handle_tool_errors
            def process_request(req: RequestObject) -> str:
                return f"processed {req.task} on {req.df_id}"

            valid_req = RequestObject("df_obj", "clean")
            assert process_request(valid_req) == "processed clean on df_obj"

            bad_req = RequestObject("missing_obj", "clean")
            err = process_request(bad_req)
            assert isinstance(err, dict)
            assert err["status"] == "error"
            assert "missing_obj" in err["reason"]

    def test_parameter_object_descriptor_exception_sanitized_and_logged(self, caplog):
        """Test A, B, C: Descriptor raising unexpected exception produces sanitized error and logs locally."""

        class MaliciousRequest:
            @property
            def df_id(self):
                raise RuntimeError("internal secret connection key: 0xDEADBEEF")

        @idd_tools.handle_tool_errors
        def process_request(req: MaliciousRequest) -> str:
            return "ok"

        with caplog.at_level("ERROR"):
            result = process_request(MaliciousRequest())

        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["reason"] == "An unexpected data-processing failure occurred."
        assert "0xDEADBEEF" not in str(result)
        assert any("0xDEADBEEF" in r.message for r in caplog.records)

    def test_registry_runtime_error_sanitized_not_misclassified(
        self, monkeypatch, caplog
    ):
        """Test D: Unexpected registry failure produces sanitized unexpected-error, not missing-data."""
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):

            def broken_get_dataframe(df_id, load_if_not_exists=True):
                raise RuntimeError("registry disk failure: sector corrupted")

            monkeypatch.setattr(reg, "get_dataframe", broken_get_dataframe)

            @idd_tools.handle_tool_errors
            def inspect_tool(df_id: str) -> str:
                return "ok"

            with caplog.at_level("ERROR"):
                result = inspect_tool("test_df")

            assert isinstance(result, dict)
            assert result["status"] == "error"
            assert result["reason"] == "An unexpected data-processing failure occurred."
            # Must NOT be misclassified as missing/empty DataFrame
            assert "was not found" not in result["reason"]
            assert "sector corrupted" not in str(result)
            assert any("sector corrupted" in r.message for r in caplog.records)

    def test_missing_dataframe_produces_missing_data_error(self):
        """Test E: A genuinely missing DataFrame produces the correct missing-data response."""
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):

            @idd_tools.handle_tool_errors
            def inspect_tool(df_id: str) -> str:
                return "ok"

            result = inspect_tool("definitely_not_registered_df")
            assert isinstance(result, dict)
            assert result["status"] == "error"
            assert "definitely_not_registered_df" in result["reason"]
            assert "was not found or is empty" in result["reason"]

    def test_empty_dataframe_produces_validation_error(self):
        """Test F: An empty DataFrame produces the correct validation response."""
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):
            empty_df = pd.DataFrame()
            reg.register_dataframe(empty_df, "empty_registered_df")

            @idd_tools.handle_tool_errors
            def inspect_tool(df_id: str) -> str:
                return "ok"

            result = inspect_tool("empty_registered_df")
            assert isinstance(result, dict)
            assert result["status"] == "error"
            assert "empty_registered_df" in result["reason"]
            assert "was not found or is empty" in result["reason"]

    def test_optional_defaulted_df_id_semantics(self):
        """Test I: Optional/defaulted df_id parameter semantics."""
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):
            df = pd.DataFrame({"a": [1, 2]})
            reg.register_dataframe(df, "valid_opt_df")

            @idd_tools.handle_tool_errors
            def tool_with_optional_df(message: str, df_id: Optional[str] = None) -> str:
                return f"{message}:{df_id}"

            # Omitted optional df_id: executes normally without validation
            assert tool_with_optional_df("hello") == "hello:None"

            # Explicit positional None: executes normally without validation
            assert tool_with_optional_df("hello", None) == "hello:None"

            # Explicit keyword None: executes normally without validation
            assert tool_with_optional_df("hello", df_id=None) == "hello:None"

            # Explicit valid df_id: validated and passes
            assert (
                tool_with_optional_df("hello", df_id="valid_opt_df")
                == "hello:valid_opt_df"
            )

            # Explicit invalid df_id: validated and returns structured error
            err_missing = tool_with_optional_df("hello", df_id="nonexistent_opt_df")
            assert isinstance(err_missing, dict)
            assert err_missing["status"] == "error"
            assert "nonexistent_opt_df" in err_missing["reason"]

            # Explicit non-string df_id: validated and returns structured error
            err_type = tool_with_optional_df("hello", df_id=999)
            assert isinstance(err_type, dict)
            assert err_type["status"] == "error"
            assert "DataFrame ID must be a non-empty string." in err_type["reason"]

    def test_empty_or_non_string_df_id_fails_gracefully(self):
        @idd_tools.handle_tool_errors
        def my_tool(df_id: str) -> str:
            return "ok"

        err_empty = my_tool("")
        assert isinstance(err_empty, dict)
        assert err_empty["status"] == "error"
        assert err_empty["reason"] == "DataFrame ID must be a non-empty string."

        err_none = my_tool(None)
        assert isinstance(err_none, dict)
        assert err_none["status"] == "error"
        assert err_none["reason"] == "DataFrame ID must be a non-empty string."

        err_num = my_tool(12345)
        assert isinstance(err_num, dict)
        assert err_num["status"] == "error"
        assert err_num["reason"] == "DataFrame ID must be a non-empty string."

    def test_invalid_argument_counts_produce_structured_error(self):
        @idd_tools.handle_tool_errors
        def binary_op(a: int, b: int) -> int:
            return a + b

        # Missing required argument
        err_missing = binary_op(10)
        assert isinstance(err_missing, dict)
        assert err_missing["status"] == "error"
        assert "Invalid arguments" in err_missing["reason"]

        # Too many arguments
        err_extra = binary_op(10, 20, 30)
        assert isinstance(err_extra, dict)
        assert err_extra["status"] == "error"
        assert "Invalid arguments" in err_extra["reason"]

    def test_metadata_and_signature_preservation(self):
        @idd_tools.handle_tool_errors
        def documented_function(df_id: str, factor: int = 1) -> str:
            """This is a documented tool."""
            return f"{df_id}:{factor}"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented tool."
        sig = inspect.signature(documented_function)
        assert "df_id" in sig.parameters
        assert "factor" in sig.parameters
        assert sig.parameters["factor"].default == 1

    def test_known_exception_categorization(self):
        @idd_tools.handle_tool_errors
        def raise_fnf():
            raise FileNotFoundError("missing.csv")

        @idd_tools.handle_tool_errors
        def raise_ke():
            raise KeyError("missing_col")

        @idd_tools.handle_tool_errors
        def raise_ve():
            raise ValueError("bad parameter")

        @idd_tools.handle_tool_errors
        def raise_empty():
            raise pd.errors.EmptyDataError("empty data")

        assert (
            raise_fnf()["reason"] == "The dataset or required file could not be found."
        )
        assert raise_ke()["reason"] == "A requested column or key does not exist."
        assert raise_ve()["reason"] == "One or more inputs are invalid."
        assert raise_empty()["reason"] == "The dataset contains no rows."


# ---------------------------------------------------------------------------
# 3. Canonical validate_dataframe_exists
# ---------------------------------------------------------------------------


class TestValidateDataFrameExistsCanonical:
    def test_valid_in_memory_dataframe(self):
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):
            df = pd.DataFrame({"col": [1, 2, 3]})
            df_id = reg.register_dataframe(df, "valid_in_mem")
            assert idd_tools.validate_dataframe_exists(df_id) is True

    def test_empty_dataframe_returns_false(self):
        reg = DataFrameRegistry(capacity=5)
        with override_global_registry(reg):
            df = pd.DataFrame()
            df_id = reg.register_dataframe(df, "empty_df")
            assert idd_tools.validate_dataframe_exists(df_id) is False

    def test_unknown_df_id_returns_false(self):
        assert idd_tools.validate_dataframe_exists("unregistered_xyz") is False
        assert idd_tools.validate_dataframe_exists("") is False
        assert idd_tools.validate_dataframe_exists(None) is False

    def test_evicted_multi_format_reload(self, tmp_path):
        """Claim 8: Evicted non-CSV frames reload losslessly via get_dataframe."""
        reg = DataFrameRegistry(capacity=1)
        with override_global_registry(reg):
            # 1. Pickle format
            pkl_file = tmp_path / "data.pkl"
            df_pkl = pd.DataFrame({"col": [10, 20]})
            df_pkl.to_pickle(pkl_file)
            reg.register_dataframe(df_pkl, "df_pkl", raw_path=str(pkl_file))

            # Evict df_pkl by registering a second frame
            reg.register_dataframe(pd.DataFrame({"x": [1]}), "evictor")
            assert reg.registry["df_pkl"]["df"] is None

            # Validation must succeed through get_dataframe loader
            assert idd_tools.validate_dataframe_exists("df_pkl") is True


# ---------------------------------------------------------------------------
# 4. Artifact Path Resolution & Containment
# ---------------------------------------------------------------------------


class TestArtifactPathResolution:
    def test_valid_relative_path_and_creation(self, tmp_path):
        cfg = types.SimpleNamespace(
            configurable={
                "runtime": types.SimpleNamespace(artifacts_dir=str(tmp_path / "art"))
            }
        )
        resolved = idd_tools._resolve_artifact_path("report.html", config=cfg)
        assert resolved == (tmp_path / "art" / "report.html").resolve()
        assert resolved.parent.exists()

    def test_environment_override_precedence(self, tmp_path, monkeypatch):
        art_dir = tmp_path / "env_artifacts"
        monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(art_dir))
        resolved = idd_tools._resolve_artifact_path("chart.png", config=None)
        assert resolved == (art_dir / "chart.png").resolve()

    def test_working_directory_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("IDD_ARTIFACTS_DIR", raising=False)
        monkeypatch.setattr(idd_tools, "WORKING_DIRECTORY", tmp_path / "workdir")
        resolved = idd_tools._resolve_artifact_path("output.csv", config=None)
        assert resolved == (tmp_path / "workdir" / "artifacts" / "output.csv").resolve()

    def test_path_traversal_double_dot_rejected_without_mkdir(self, tmp_path):
        base_dir = tmp_path / "art_base"
        cfg = types.SimpleNamespace(
            configurable={"runtime": types.SimpleNamespace(artifacts_dir=str(base_dir))}
        )
        with pytest.raises(ValueError, match="(?i)outside|refus"):
            idd_tools._resolve_artifact_path(
                "../../escape.txt", config=cfg, subdir="sub_uncreated"
            )

        # sub_uncreated directory must NOT have been created on disk
        assert not (base_dir / "sub_uncreated").exists()

    def test_absolute_escape_rejected(self, tmp_path):
        base_dir = tmp_path / "art_base"
        cfg = types.SimpleNamespace(
            configurable={"runtime": types.SimpleNamespace(artifacts_dir=str(base_dir))}
        )
        outside = str(tmp_path / "outside.txt")
        with pytest.raises(ValueError, match="(?i)outside|refus"):
            idd_tools._resolve_artifact_path(outside, config=cfg)

    def test_symlink_escape_rejected(self, tmp_path):
        outside_dir = tmp_path / "secret_outside"
        outside_dir.mkdir()
        secret_file = outside_dir / "secret.png"
        secret_file.write_bytes(b"secret")

        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        cfg = types.SimpleNamespace(
            configurable={"runtime": types.SimpleNamespace(artifacts_dir=str(art_dir))}
        )

        symlink_file = art_dir / "symlink_secret.png"
        try:
            symlink_file.symlink_to(secret_file)
            with pytest.raises(ValueError, match="(?i)outside|refus"):
                idd_tools._resolve_artifact_path(str(symlink_file), config=cfg)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported in this environment")

    def test_multi_root_report_containment_helper(self, tmp_path):
        art_root = tmp_path / "artifacts"
        work_root = tmp_path / "workspace"
        art_root.mkdir()
        work_root.mkdir()

        chart_in_art = art_root / "chart.png"
        chart_in_art.write_bytes(b"chart")

        chart_in_work = work_root / "viz.png"
        chart_in_work.write_bytes(b"viz")

        outside_file = tmp_path / "outside.png"
        outside_file.write_bytes(b"outside")

        allowed_roots = [art_root, work_root]
        assert idd_tools.is_safe_artifact_path(chart_in_art, allowed_roots) is True
        assert idd_tools.is_safe_artifact_path(chart_in_work, allowed_roots) is True
        assert idd_tools.is_safe_artifact_path(outside_file, allowed_roots) is False

    def test_uri_schemes_rejected_in_is_safe_artifact_path(self, tmp_path):
        """Reject all recognized RFC URI schemes before filesystem resolution."""
        allowed_roots = [tmp_path]
        schemes_to_test = [
            "javascript:alert(1)",
            "vbscript:execute",
            "mailto:user@example.com",
            "ftp:example.com/file",
            "data:image/png;base64,AAAA",
            "file:///etc/passwd",
            "custom+scheme:payload",
            "http://example.com/image.png",
            "https://example.com/image.png",
            "a:payload",
            "x:javascript",
            "C:relative.png",
        ]
        for uri in schemes_to_test:
            assert (
                idd_tools.is_safe_artifact_path(uri, allowed_roots) is False
            ), f"Failed to reject: {uri}"

    def test_uri_schemes_rejected_in_resolve_artifact_path(self):
        """Reject URI schemes in _resolve_artifact_path with ValueError."""
        with pytest.raises(ValueError, match="Refusing to resolve URI scheme"):
            idd_tools._resolve_artifact_path("javascript:alert(1)")

        with pytest.raises(ValueError, match="Refusing to resolve URI scheme"):
            idd_tools._resolve_artifact_path("http://evil.com/malicious.png")

        with pytest.raises(ValueError, match="Refusing to resolve URI scheme"):
            idd_tools._resolve_artifact_path("file:///etc/passwd")

        with pytest.raises(ValueError, match="Refusing to resolve URI scheme"):
            idd_tools._resolve_artifact_path("a:payload")

        with pytest.raises(ValueError, match="Refusing to resolve URI scheme"):
            idd_tools._resolve_artifact_path("x:javascript")

    def test_drive_relative_paths_rejected_without_directory_creation(self, tmp_path):
        """Verify Windows drive-relative paths are rejected as ambiguous URI schemes without creating dirs."""
        art_dir = tmp_path / "artifacts"
        cfg = types.SimpleNamespace(
            configurable={"runtime": types.SimpleNamespace(artifacts_dir=str(art_dir))}
        )
        drive_relative_cases = [
            "C:relative.png",
            r"D:folder\file.txt",
            "Z:secret.txt",
        ]
        for path_str in drive_relative_cases:
            assert idd_tools.is_safe_artifact_path(path_str, [art_dir]) is False
            with pytest.raises(ValueError, match="Refusing to resolve URI scheme"):
                idd_tools._resolve_artifact_path(path_str, config=cfg)

        # Confirm no directories were prematurely created
        assert not (art_dir / "folder").exists()

    def test_unc_paths_rejected_on_posix_and_windows(self, tmp_path, monkeypatch):
        """Verify UNC paths are rejected outside root on Windows and as non-POSIX paths on POSIX."""
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        cfg = types.SimpleNamespace(
            configurable={"runtime": types.SimpleNamespace(artifacts_dir=str(art_dir))}
        )
        unc_path = r"\\server\share\file.png"

        # On current platform, UNC path is not inside tmp_path / artifacts
        assert idd_tools.is_safe_artifact_path(unc_path, [art_dir]) is False
        with pytest.raises(
            ValueError, match="Refusing to access path outside artifacts root"
        ):
            idd_tools._resolve_artifact_path(unc_path, config=cfg)

        # On simulated POSIX platform, UNC prefix \\ is explicitly rejected
        monkeypatch.setattr(idd_tools.os, "name", "posix")
        assert idd_tools.is_safe_artifact_path(unc_path, [art_dir]) is False
        with pytest.raises(
            ValueError, match="Refusing to access path outside artifacts root"
        ):
            idd_tools._resolve_artifact_path(unc_path, config=cfg)

    def test_legitimate_relative_paths_resolve_safely(self, tmp_path):
        """Verify legitimate relative paths with subdirectories resolve strictly inside root."""
        art_dir = tmp_path / "artifacts"
        cfg = types.SimpleNamespace(
            configurable={"runtime": types.SimpleNamespace(artifacts_dir=str(art_dir))}
        )
        for rel_path in [
            "chart.png",
            "reports/report.html",
            "visualizations/figure_01.png",
        ]:
            resolved = idd_tools._resolve_artifact_path(rel_path, config=cfg)
            assert resolved == (art_dir / rel_path).resolve()
            assert resolved.parent.exists()

    def test_windows_drive_letter_paths(self, tmp_path):
        """Verify Windows drive-letter paths are distinguished from URI schemes and checked for containment."""
        art_root = tmp_path / "artifacts"
        art_root.mkdir()
        valid_file = art_root / "chart.png"
        valid_file.write_bytes(b"chart")

        allowed_roots = [art_root]
        # Valid path within allowed root succeeds
        assert idd_tools.is_safe_artifact_path(str(valid_file), allowed_roots) is True
        # Legitimate Windows drive path outside allowed root is rejected by containment, not false scheme match
        outside_drive_path = "Z:/unauthorized/outside.png"
        assert (
            idd_tools.is_safe_artifact_path(outside_drive_path, allowed_roots) is False
        )
        with pytest.raises(
            ValueError, match="Refusing to access path outside artifacts root"
        ):
            idd_tools._resolve_artifact_path(outside_drive_path)

    def test_global_runtime_artifact_precedence(self, tmp_path, monkeypatch):
        """Verify __main__.RUNTIME.artifacts_dir takes precedence over env var and fallback."""
        global_art = tmp_path / "global_runtime_artifacts"
        env_art = tmp_path / "env_artifacts"
        monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(env_art))

        main_mod = sys.modules["__main__"]
        orig_runtime = getattr(main_mod, "RUNTIME", None)
        try:
            main_mod.RUNTIME = types.SimpleNamespace(artifacts_dir=str(global_art))
            resolved = idd_tools._resolve_artifact_path(
                "global_report.html", config=None
            )
            assert resolved == (global_art / "global_report.html").resolve()
        finally:
            if orig_runtime is not None:
                main_mod.RUNTIME = orig_runtime
            else:
                delattr(main_mod, "RUNTIME")


# ---------------------------------------------------------------------------
# 5. Compatibility Façade Identity
# ---------------------------------------------------------------------------


class TestCoreFacadeIdentity:
    def test_facade_re_exports_canonical_symbols(self):
        assert idd_core.handle_tool_errors is idd_tools.handle_tool_errors
        assert idd_core.validate_dataframe_exists is idd_tools.validate_dataframe_exists
        assert idd_core._resolve_artifact_path is idd_tools._resolve_artifact_path
        assert idd_core._tool_error is idd_tools._tool_error
        assert idd_core._tool_failure is idd_tools._tool_failure
        assert idd_core._is_subpath is idd_tools._is_subpath
        assert idd_core._get_artifacts_base is idd_tools._get_artifacts_base
        assert idd_core.is_safe_artifact_path is idd_tools.is_safe_artifact_path
        assert idd_core.WORKING_DIRECTORY is idd_tools.WORKING_DIRECTORY
