import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


class Registry:
    def __init__(self):
        self.frames = {}
        self.paths = {}

    def register_dataframe(self, df, df_id, raw_path=""):
        self.frames[df_id] = df
        self.paths[df_id] = raw_path
        return df_id

    def get_dataframe(self, df_id):
        return self.frames.get(df_id)

    def get_raw_path_from_id(self, df_id):
        return self.paths.get(df_id)


def _load_tools(registry):
    source = Path("intelligentdatadetective_beta_v5.py").read_text()
    start = source.index("# Error Handling and Validation Framework")
    end = source.index("data_cleaning_tools = [", start)
    namespace = {
        "Dict": Dict,
        "List": List,
        "Union": Union,
        "Optional": Optional,
        "os": __import__("os"),
        "logging": logging,
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

    invalid_query = tools["delete_rows"]("df", ["unknown > 1"])
    _assert_error(invalid_query, "delete_rows")
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
