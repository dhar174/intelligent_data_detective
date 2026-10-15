"""
Unit tests for validate_dataframe_exists and handle_tool_errors in idd_core.
Uses global_registry_reset (NOT autouse) to isolate global_df_registry state.
"""
from pathlib import Path
import pandas as pd

import idd_core
from idd_core import (
    DataFrameRegistry,
    handle_tool_errors,
    override_global_registry,
    validate_dataframe_exists,
)

BASE = dict(reply_msg_to_supervisor="test", finished_this_task=True, expect_reply=False)


# ---------------------------------------------------------------------------
# validate_dataframe_exists
# ---------------------------------------------------------------------------

class TestValidateDataframeExists:
    def test_empty_string_df_id_returns_false(self, global_registry_reset):
        assert validate_dataframe_exists("") is False

    def test_none_df_id_returns_false(self, global_registry_reset):
        # The function signature expects str, but should handle None gracefully
        assert validate_dataframe_exists(None) is False  # type: ignore[arg-type]

    def test_unregistered_id_returns_false(self, global_registry_reset):
        assert validate_dataframe_exists("nonexistent_df_id_xyz") is False

    def test_registered_nonempty_df_in_memory_returns_true(self, global_registry_reset, sample_df, tmp_path):
        csv_path = tmp_path / "sample.csv"
        sample_df.to_csv(csv_path, index=False)
        df_id = idd_core.global_df_registry.register_dataframe(
            sample_df, "mem_df", raw_path=str(csv_path)
        )
        assert validate_dataframe_exists(df_id) is True

    def test_registered_empty_df_returns_false(self, global_registry_reset, tmp_path):
        empty_df = pd.DataFrame()
        csv_path = tmp_path / "empty.csv"
        empty_df.to_csv(csv_path, index=False)
        df_id = idd_core.global_df_registry.register_dataframe(
            empty_df, "empty_df", raw_path=str(csv_path)
        )
        assert validate_dataframe_exists(df_id) is False

    def test_raw_path_exists_on_disk_reloads_from_csv(self, global_registry_reset, sample_df, tmp_path):
        """If df is not in memory cache but raw_path has valid CSV, function reloads and returns True."""
        csv_path = tmp_path / "onDisk.csv"
        sample_df.to_csv(csv_path, index=False)

        small_reg = DataFrameRegistry(capacity=2)
        with override_global_registry(small_reg):
            # Register target df
            target_id = small_reg.register_dataframe(sample_df, "target_df", raw_path=str(csv_path))
            assert target_id is not None

            # Evict target by adding 2 more dfs
            for i in range(2):
                extra = pd.DataFrame({"val": [i, i + 1]})
                small_reg.register_dataframe(extra, f"evict_{i}")

            # target_df should now be evicted from in-memory cache
            assert "target_df" not in small_reg.cache

            # But raw_path is on disk — should reload
            result = validate_dataframe_exists("target_df")
            assert result is True

    def test_raw_path_registered_but_file_missing_returns_false(self, global_registry_reset, tmp_path):
        csv_path = tmp_path / "missing.csv"
        small_reg = DataFrameRegistry(capacity=2)
        with override_global_registry(small_reg):
            sample = pd.DataFrame({"a": [1, 2]})
            sample.to_csv(csv_path, index=False)
            small_reg.register_dataframe(sample, "ghost_df", raw_path=str(csv_path))

            # Evict the df from cache
            for i in range(2):
                extra = pd.DataFrame({"x": [i]})
                small_reg.register_dataframe(extra, f"evict2_{i}")

            # Delete the source file and owned backing file so disk-fallback fails
            csv_path.unlink()
            backing = small_reg.get_backing_path_from_id("ghost_df")
            if backing and Path(backing).exists():
                Path(backing).unlink()
            assert validate_dataframe_exists("ghost_df") is False

    def test_validation_preserves_updated_working_data_after_eviction(self, tmp_path):
        """Test A: validate_dataframe_exists must never overwrite working data with stale source on eviction."""
        src_file = tmp_path / "user_source.csv"
        df_v1 = pd.DataFrame({"value": [1, 2, 3]})
        df_v1.to_csv(src_file, index=False)
        original_bytes = src_file.read_bytes()

        reg = DataFrameRegistry(capacity=1)
        with override_global_registry(reg):
            reg.register_dataframe(df_v1, "dataset_1", raw_path=str(src_file))

            # Update working data
            df_v2 = pd.DataFrame({"value": [100, 200, 300]})
            reg.register_dataframe(df_v2, "dataset_1", raw_path=str(src_file))

            # Evict from in-memory cache
            reg.register_dataframe(pd.DataFrame({"dummy": [0]}), "evictor")
            assert reg.registry["dataset_1"]["df"] is None

            # Legacy validation must use authoritative reload, NOT restore stale source
            assert validate_dataframe_exists("dataset_1") is True

            # Working DataFrame must still contain updated values
            working_df = reg.get_dataframe("dataset_1", load_if_not_exists=True)
            assert working_df is not None
            assert list(working_df["value"]) == [100, 200, 300]

            # Owned backing file must contain updated data
            backing_path = reg.get_backing_path_from_id("dataset_1")
            assert backing_path is not None
            backed_df = pd.read_pickle(backing_path)
            assert list(backed_df["value"]) == [100, 200, 300]

            # Original source file must remain byte-for-byte unchanged
            assert src_file.read_bytes() == original_bytes

    def test_decorator_cannot_revert_dataframe_to_stale_source(self, tmp_path):
        """Test B: @handle_tool_errors validation step must not revert working data to stale source."""
        src_file = tmp_path / "user_source.csv"
        df_v1 = pd.DataFrame({"value": [1, 2, 3]})
        df_v1.to_csv(src_file, index=False)
        original_bytes = src_file.read_bytes()

        reg = DataFrameRegistry(capacity=1)
        with override_global_registry(reg):
            reg.register_dataframe(df_v1, "dataset_1", raw_path=str(src_file))

            df_v2 = pd.DataFrame({"value": [100, 200, 300]})
            reg.register_dataframe(df_v2, "dataset_1", raw_path=str(src_file))

            # Evict dataset_1
            reg.register_dataframe(pd.DataFrame({"dummy": [0]}), "evictor")
            assert reg.registry["dataset_1"]["df"] is None

            @handle_tool_errors
            def inspect_data(df_id: str) -> str:
                df = reg.get_dataframe(df_id)
                return f"sum={df['value'].sum()}"

            result = inspect_data("dataset_1")
            assert result == "sum=600"

            working_df = reg.get_dataframe("dataset_1")
            assert list(working_df["value"]) == [100, 200, 300]
            assert src_file.read_bytes() == original_bytes

    def test_validation_supports_evicted_non_csv_formats(self, tmp_path):
        """Test D: validate_dataframe_exists succeeds on evicted non-CSV formats (.pkl, .json)."""
        reg = DataFrameRegistry(capacity=1)
        with override_global_registry(reg):
            # Pickle format
            pkl_file = tmp_path / "data.pkl"
            df_pkl = pd.DataFrame({"col": [10, 20]})
            df_pkl.to_pickle(pkl_file)
            reg.register_dataframe(df_pkl, "df_pkl", raw_path=str(pkl_file))

            # JSON format
            json_file = tmp_path / "data.json"
            df_json = pd.DataFrame({"col": [30, 40]})
            df_json.to_json(json_file, orient="records")
            reg.register_dataframe(df_json, "df_json", raw_path=str(json_file))

            # Both should have been evicted by the small capacity
            assert validate_dataframe_exists("df_pkl") is True
            assert validate_dataframe_exists("df_json") is True


# ---------------------------------------------------------------------------
# handle_tool_errors: df_id extraction paths
# ---------------------------------------------------------------------------

class TestHandleToolErrorsDfIdExtraction:
    def test_positional_str_df_id_not_found_returns_error(self, global_registry_reset):
        @handle_tool_errors
        def my_tool(df_id: str):
            return "ok"

        result = my_tool("nonexistent_df_xyz")
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["operation"] == "my_tool"
        assert "nonexistent_df_xyz" in result["reason"]
        assert "action" in result

    def test_kwargs_df_id_not_found_returns_error(self, global_registry_reset):
        @handle_tool_errors
        def my_tool(df_id: str):
            return "ok"

        result = my_tool(df_id="nonexistent_via_kwargs")
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["operation"] == "my_tool"
        assert "nonexistent_via_kwargs" in result["reason"]
        assert "action" in result

    def test_object_with_df_id_attr_not_found_returns_error(self, global_registry_reset):
        class Req:
            def __init__(self, df_id):
                self.df_id = df_id

        @handle_tool_errors
        def my_tool(req):
            return "ok"

        result = my_tool(Req("bad_df_id"))
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["operation"] == "my_tool"
        assert "bad_df_id" in result["reason"]
        assert "action" in result

    def test_registered_df_calls_wrapped_function(self, global_registry_reset, sample_df, tmp_path):
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)
        df_id = idd_core.global_df_registry.register_dataframe(
            sample_df, "good_df", raw_path=str(csv_path)
        )

        @handle_tool_errors
        def my_tool(df_id: str):
            return f"processed:{df_id}"

        result = my_tool(df_id)
        assert result == f"processed:{df_id}"

    def test_no_df_id_in_args_calls_wrapped_function(self, global_registry_reset):
        @handle_tool_errors
        def no_df_tool(x: int):
            return x * 2

        result = no_df_tool(5)
        assert result == 10

    def test_first_arg_non_string_non_obj_skips_validation(self, global_registry_reset):
        @handle_tool_errors
        def numeric_tool(n: int):
            return n + 1

        result = numeric_tool(42)
        assert result == 43


# ---------------------------------------------------------------------------
# handle_tool_errors: exception handling
# ---------------------------------------------------------------------------

class TestHandleToolErrorsExceptions:
    def test_value_error_returns_error_string(self):
        @handle_tool_errors
        def bad_tool():
            raise ValueError("invalid configuration")

        result = bad_tool()
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["operation"] == "bad_tool"
        assert result["reason"] == "One or more inputs are invalid."
        assert "action" in result

    def test_file_not_found_returns_error_string(self):
        @handle_tool_errors
        def fnf_tool():
            raise FileNotFoundError("data.csv")

        result = fnf_tool()
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["operation"] == "fnf_tool"
        assert result["reason"] == "The dataset or required file could not be found."
        assert "action" in result

    def test_key_error_returns_error_string(self):
        @handle_tool_errors
        def key_tool():
            raise KeyError("column_x")

        result = key_tool()
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["operation"] == "key_tool"
        assert result["reason"] == "A requested column or key does not exist."
        assert "action" in result

    def test_empty_data_error_returns_error_string(self):
        @handle_tool_errors
        def empty_tool():
            raise pd.errors.EmptyDataError()

        result = empty_tool()
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["operation"] == "empty_tool"
        assert result["reason"] == "The dataset contains no rows."
        assert "action" in result

    def test_generic_exception_returns_error_string(self):
        @handle_tool_errors
        def generic_tool():
            raise RuntimeError("something went wrong")

        result = generic_tool()
        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert result["operation"] == "generic_tool"
        assert result["reason"] == "An unexpected data-processing failure occurred."
        assert "Retry the operation" in result["action"]


# ---------------------------------------------------------------------------
# handle_tool_errors: decorator mechanics
# ---------------------------------------------------------------------------

class TestHandleToolErrorsDecorator:
    def test_preserves_wrapped_function_name(self):
        @handle_tool_errors
        def my_named_function(x):
            return x

        assert my_named_function.__name__ == "my_named_function"

    def test_preserves_wrapped_function_docstring(self):
        @handle_tool_errors
        def documented_tool(x):
            """This tool does something."""
            return x

        assert documented_tool.__doc__ == "This tool does something."

    def test_return_value_passed_through_on_success(self, global_registry_reset, sample_df, tmp_path):
        csv_path = tmp_path / "pass.csv"
        sample_df.to_csv(csv_path, index=False)
        df_id = idd_core.global_df_registry.register_dataframe(
            sample_df, "pass_df", raw_path=str(csv_path)
        )

        @handle_tool_errors
        def returns_dict(df_id: str):
            return {"key": "value", "df": df_id}

        result = returns_dict(df_id)
        assert isinstance(result, dict)
        assert result["key"] == "value"
