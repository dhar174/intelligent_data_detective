import logging
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import pytest


class Registry:
    def __init__(self):
        self.frames = {}
        self.paths = {}
        self.raise_on_get = False
        self.register_count = 0

    def register_dataframe(self, df, df_id, raw_path=""):
        self.register_count += 1
        self.frames[df_id] = df
        self.paths[df_id] = raw_path
        return df_id

    def get_dataframe(self, df_id, load_if_not_exists=False):
        if self.raise_on_get:
            raise RuntimeError("registry unavailable")
        df = self.frames.get(df_id)
        if df is not None or not load_if_not_exists:
            return df

        raw_path = self.paths.get(df_id)
        if not raw_path:
            return None

        raw_path = Path(raw_path)
        if not raw_path.exists():
            return None

        if raw_path.suffix == ".pkl":
            reader = pd.read_pickle
        elif raw_path.suffix == ".json":
            reader = lambda path: pd.read_json(path, orient="records")
        else:
            reader = pd.read_csv
        try:
            df = reader(raw_path)
        except Exception:
            logging.exception("Error loading DataFrame from %s", raw_path)
            return None
        self.frames[df_id] = df
        return df

    def get_raw_path_from_id(self, df_id):
        return self.paths.get(df_id)


def _load_tools(registry):
    source = (Path(__file__).resolve().parents[2] / "intelligentdatadetective_beta_v5.py").read_text('utf-8')
    start = source.index("# Error Handling and Validation Framework")
    end = source.index("data_cleaning_tools = [", start)
    namespace = {
        "Dict": Dict,
        "List": List,
        "Union": Union,
        "Optional": Optional,
        "os": __import__("os"),
        "logging": logging,
        "inspect": inspect,
        "functools": __import__("functools"),
        "pd": pd,
        "global_df_registry": registry,
        "tool": lambda *args, **kwargs: (lambda fn: fn),
        "cap_output": lambda *args, **kwargs: (lambda fn: fn),
    }
    exec(source[start:end], namespace)
    return namespace


def _assert_error(result, operation):
    assert result["status"] == "error"
    assert result["operation"] == operation
    assert result["reason"]
    assert result["action"]


def test_cleaning_tools_validate_and_preserve_failed_mutations():
    registry = Registry()
    original = pd.DataFrame({"value": [1.0, None, 3.0], "name": ["a", "b", "c"]})
    registry.register_dataframe(original, "df")
    tools = _load_tools(registry)

    missing = tools["drop_column"]("missing", "value")
    _assert_error(missing, "drop_column")
    assert registry.get_dataframe("df").equals(original)

    empty_df_id = tools["drop_column"]("", "value")
    _assert_error(empty_df_id, "drop_column")
    assert empty_df_id["reason"] == "DataFrame ID must be a non-empty string."
    assert registry.get_dataframe("df").equals(original)

    non_string_df_id = tools["drop_column"](123, "value")
    _assert_error(non_string_df_id, "drop_column")
    assert non_string_df_id["reason"] == "DataFrame ID must be a non-empty string."
    assert registry.get_dataframe("df").equals(original)

    invalid_query = tools["delete_rows"]("df", ["unknown > 1"])
    _assert_error(invalid_query, "delete_rows")
    assert registry.get_dataframe("df").equals(original)

    malformed_selector = tools["delete_rows"]("df", {"valid": ["value > 1"], "bad": 123})
    _assert_error(malformed_selector, "delete_rows")
    assert registry.get_dataframe("df").equals(original)

    invalid_signature = tools["drop_column"]("df", "value", "extra")
    _assert_error(invalid_signature, "drop_column")
    assert "too many positional arguments" in invalid_signature["reason"]
    assert registry.get_dataframe("df").equals(original)

    non_numeric = tools["fill_missing_median"]("df", "name")
    _assert_error(non_numeric, "fill_missing_median")
    assert registry.get_dataframe("df").equals(original)


def test_median_rejects_all_null_and_success_updates_registry():
    registry = Registry()
    registry.register_dataframe(
        pd.DataFrame({"empty": [None, None], "value": [1.0, None]}), "df"
    )
    tools = _load_tools(registry)

    result = tools["fill_missing_median"]("df", "empty")
    _assert_error(result, "fill_missing_median")
    assert registry.get_dataframe("df")["empty"].isna().all()

    success = tools["fill_missing_median"]("df", "value")
    assert "filled with median" in success
    assert registry.get_dataframe("df")["value"].tolist() == [1.0, 1.0]


def test_unexpected_tool_failure_is_safe_and_logged(caplog):
    registry = Registry()
    registry.register_dataframe(pd.DataFrame({"value": [1]}), "df")
    tools = _load_tools(registry)

    @tools["handle_tool_errors"]
    def broken(df_id):
        raise RuntimeError("internal details")

    with caplog.at_level(logging.ERROR):
        result = broken("df")

    _assert_error(result, "broken")
    assert result["reason"] == "An unexpected data-processing failure occurred."
    assert "internal details" not in result["reason"]
    assert "internal details" in caplog.text


def test_validation_reload_supports_non_csv_backing_files(tmp_path):
    registry = Registry()
    pd.DataFrame({"value": [1.0, 2.0], "name": ["a", "b"]}).to_pickle(
        tmp_path / "frame.pkl"
    )
    registry.register_dataframe(None, "df", str(tmp_path / "frame.pkl"))
    tools = _load_tools(registry)

    result = tools["drop_column"]("df", "value")

    assert result == "Column dropped successfully. New columns: name"
    assert list(registry.get_dataframe("df").columns) == ["name"]


def test_unexpected_validation_failure_uses_tool_failure(caplog):
    registry = Registry()
    registry.register_dataframe(pd.DataFrame({"value": [1.0]}), "df")
    registry.raise_on_get = True
    tools = _load_tools(registry)

    with caplog.at_level(logging.ERROR):
        result = tools["drop_column"]("df", "value")

    _assert_error(result, "drop_column")
    assert result["reason"] == "An unexpected data-processing failure occurred."
    assert "registry unavailable" in caplog.text


def test_unreadable_backing_file_logs_and_returns_missing_error(tmp_path, caplog):
    registry = Registry()
    broken_path = tmp_path / "broken.pkl"
    broken_path.write_text("not a pickle", encoding="utf-8")
    registry.register_dataframe(None, "df", str(broken_path))
    tools = _load_tools(registry)

    with caplog.at_level(logging.ERROR):
        result = tools["drop_column"]("df", "value")

    _assert_error(result, "drop_column")
    assert result["reason"] == "DataFrame 'df' was not found or is empty."
    assert "Error loading DataFrame from" in caplog.text


def test_drop_and_delete_rows_success_paths_and_no_match():
    registry = Registry()
    registry.register_dataframe(
        pd.DataFrame({0: [10, 20, 30], "value": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]}),
        "df",
    )
    tools = _load_tools(registry)

    drop_success = tools["drop_column"]("df", "value")
    assert drop_success == "Column dropped successfully. New columns: 0, name"
    dropped_df = registry.get_dataframe("df")
    assert list(dropped_df.columns) == [0, "name"]

    non_inplace = tools["delete_rows"]("df", ["`0` >= 20"], inplace=False)
    assert non_inplace == dropped_df.loc[[1, 2]].to_json()
    assert registry.get_dataframe("df").equals(dropped_df)

    delete_success = tools["delete_rows"]("df", ["`0` >= 20"])
    assert delete_success == "2 rows deleted successfully."
    assert registry.get_dataframe("df")[0].tolist() == [10]

    no_match = tools["delete_rows"]("df", ["`0` > 999"])
    assert no_match == "No rows match the provided condition(s): ['`0` > 999']"
    assert registry.get_dataframe("df")[0].tolist() == [10]


@pytest.mark.parametrize("reverse_columns", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_delete_rows_string_binding_is_order_independent(
    reverse_columns, inplace
):
    original = pd.DataFrame(
        [[100, 0], [0, 100]],
        columns=["0", 0],
        index=["string_column_match", "integer_column_match"],
    )
    if reverse_columns:
        original = original.iloc[:, [1, 0]].copy()

    before = original.copy(deep=True)
    registry = Registry()
    registry.register_dataframe(original, "df")
    init_registrations = registry.register_count
    tools = _load_tools(registry)

    result = tools["delete_rows"]("df", ["`0` >= 20"], inplace=inplace)

    if inplace:
        assert result == "1 rows deleted successfully."
        pd.testing.assert_frame_equal(
            registry.get_dataframe("df"),
            before.loc[["integer_column_match"]],
        )
        assert registry.register_count == init_registrations + 1
    else:
        assert result == before.loc[["string_column_match"]].to_json()
        pd.testing.assert_frame_equal(registry.get_dataframe("df"), before)
        assert registry.register_count == init_registrations

    pd.testing.assert_frame_equal(original, before)


def test_delete_rows_numeric_compatibility_preserves_integer_label():
    original = pd.DataFrame(
        [[10, "alice"], [25, "bob"]],
        columns=[0, "name"],
        index=["r1", "r2"],
    )
    before = original.copy(deep=True)
    registry = Registry()
    registry.register_dataframe(original, "df")
    init_registrations = registry.register_count
    tools = _load_tools(registry)

    # Preview does not mutate registry or frame
    preview = tools["delete_rows"]("df", ["`0` >= 20"], inplace=False)
    assert preview == before.loc[["r2"]].to_json()
    assert registry.register_count == init_registrations
    pd.testing.assert_frame_equal(registry.get_dataframe("df"), before)

    # Inplace deletion removes matched row and preserves integer 0 label type
    result = tools["delete_rows"]("df", ["`0` >= 20"], inplace=True)
    assert result == "1 rows deleted successfully."
    assert registry.register_count == init_registrations + 1
    stored = registry.get_dataframe("df")
    pd.testing.assert_frame_equal(stored, before.loc[["r1"]])
    assert list(stored.columns) == [0, "name"]
    assert type(stored.columns[0]) is int
    pd.testing.assert_frame_equal(original, before)


def test_delete_rows_unrelated_query_works_with_colliding_columns():
    original = pd.DataFrame(
        [[100, 0, 50], [0, 100, 10]],
        columns=["0", 0, "value"],
        index=["row_keep", "row_drop"],
    )
    before = original.copy(deep=True)
    registry = Registry()
    registry.register_dataframe(original, "df")
    init_registrations = registry.register_count
    tools = _load_tools(registry)

    result = tools["delete_rows"]("df", ["value < 20"], inplace=True)
    assert result == "1 rows deleted successfully."
    assert registry.register_count == init_registrations + 1
    stored = registry.get_dataframe("df")
    pd.testing.assert_frame_equal(stored, before.loc[["row_keep"]])
    assert list(stored.columns) == ["0", 0, "value"]
    assert [type(c) for c in stored.columns] == [str, int, str]
    pd.testing.assert_frame_equal(original, before)


def test_delete_rows_string_only_ordinary_and_spaced_columns():
    original = pd.DataFrame(
        [["active", 30], ["inactive", 15], ["pending", 50]],
        columns=["account status", "score"],
        index=["user1", "user2", "user3"],
    )
    before = original.copy(deep=True)
    registry = Registry()
    registry.register_dataframe(original, "df")
    init_registrations = registry.register_count
    tools = _load_tools(registry)

    preview = tools["delete_rows"](
        "df", ["`account status` == 'inactive' and score < 20"], inplace=False
    )
    assert preview == before.loc[["user2"]].to_json()
    assert registry.register_count == init_registrations
    pd.testing.assert_frame_equal(registry.get_dataframe("df"), before)

    result = tools["delete_rows"](
        "df", ["`account status` == 'inactive' and score < 20"], inplace=True
    )
    assert result == "1 rows deleted successfully."
    assert registry.register_count == init_registrations + 1
    stored = registry.get_dataframe("df")
    pd.testing.assert_frame_equal(stored, before.loc[["user1", "user3"]])
    pd.testing.assert_frame_equal(original, before)


def test_delete_rows_no_match_collision_precedence():
    # String "0" has values [0, 0]; integer 0 has values [100, 100].
    # Query `0` >= 20 matches 0 rows in string "0", but matches 2 rows in integer 0.
    # String column precedence requires no-match, not reinterpretation or deletion.
    original = pd.DataFrame(
        [[0, 100], [0, 100]],
        columns=["0", 0],
        index=["row_a", "row_b"],
    )
    before = original.copy(deep=True)
    registry = Registry()
    registry.register_dataframe(original, "df")
    init_registrations = registry.register_count
    tools = _load_tools(registry)

    result = tools["delete_rows"]("df", ["`0` >= 20"], inplace=True)
    assert result == "No rows match the provided condition(s): ['`0` >= 20']"
    assert registry.register_count == init_registrations
    pd.testing.assert_frame_equal(registry.get_dataframe("df"), before)
    pd.testing.assert_frame_equal(original, before)


def test_delete_rows_missing_column_and_malformed_queries():
    original = pd.DataFrame(
        [[100, 0], [0, 100]],
        columns=["0", 0],
        index=["r1", "r2"],
    )
    before = original.copy(deep=True)
    registry = Registry()
    registry.register_dataframe(original, "df")
    init_registrations = registry.register_count
    tools = _load_tools(registry)

    # Missing column query returns structured error without mutating
    missing_result = tools["delete_rows"]("df", ["nonexistent_col > 1"])
    _assert_error(missing_result, "delete_rows")
    assert "UndefinedVariableError" in missing_result["reason"]
    assert registry.register_count == init_registrations
    pd.testing.assert_frame_equal(registry.get_dataframe("df"), before)

    # Malformed query syntax returns structured error without mutating
    syntax_result = tools["delete_rows"]("df", ["> > >"])
    _assert_error(syntax_result, "delete_rows")
    assert "SyntaxError" in syntax_result["reason"]
    assert registry.register_count == init_registrations
    pd.testing.assert_frame_equal(registry.get_dataframe("df"), before)
    pd.testing.assert_frame_equal(original, before)


def test_delete_rows_ambiguous_compatibility_alias():
    # Two integer columns with label 0 -> alias "0" is ambiguous.
    # Neither integer column should win silently; unrelated string column remains usable.
    original = pd.DataFrame(
        [[10, 20, 5], [30, 40, 50]],
        columns=[0, 0, "valid_col"],
        index=["r1", "r2"],
    )
    before = original.copy(deep=True)
    registry = Registry()
    registry.register_dataframe(original, "df")
    init_registrations = registry.register_count
    tools = _load_tools(registry)

    # Querying the ambiguous alias "0" returns structured error without mutating
    ambiguous_result = tools["delete_rows"]("df", ["`0` >= 20"])
    _assert_error(ambiguous_result, "delete_rows")
    assert "UndefinedVariableError" in ambiguous_result["reason"]
    assert registry.register_count == init_registrations
    pd.testing.assert_frame_equal(registry.get_dataframe("df"), before)

    # Unrelated valid string column remains queryable and mutable
    valid_result = tools["delete_rows"]("df", ["valid_col > 20"], inplace=True)
    assert valid_result == "1 rows deleted successfully."
    assert registry.register_count == init_registrations + 1
    stored = registry.get_dataframe("df")
    pd.testing.assert_frame_equal(stored, before.loc[["r1"]])
    pd.testing.assert_frame_equal(original, before)
