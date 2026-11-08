"""Generated datetime tool contracts, with no model/provider execution."""

import socket

from langchain_core.messages import ToolMessage
import pandas as pd
import pytest

from idd_registry import DataFrameRegistry, override_global_registry
from tests.unit.test_native_completion import load_native


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Datetime regression must remain offline")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _fixture(tmp_path, monkeypatch, current_api=True, values=None):
    ns = load_native(tmp_path, middleware=True)
    registry = DataFrameRegistry(data_dir=tmp_path / "date-registry")
    registry.register_dataframe(
        pd.DataFrame(
            {
                "when": values or ["2025-01-01 00:00:00", "2025-01-02 00:00:00"],
                "value": [10, 20],
            }
        ),
        "date-fixture",
    )
    if current_api:
        native = pd.to_datetime

        def current_signature(arg, *, errors="raise", format=None, utc=False):
            # Exercise the actual current API's rejection of removed keywords on
            # older provisioned Pandas too. Valid parsing delegates unchanged.
            return native(arg, errors=errors, format=format, utc=utc)

        monkeypatch.setattr(pd, "to_datetime", current_signature)
    return ns, registry


def _invoke(ns, registry, **options):
    with override_global_registry(registry):
        result = ns["convert_data_types"].invoke(
            {
                "type": "tool_call",
                "name": "convert_data_types",
                "id": "date-call",
                "args": {
                    "df_id": "date-fixture",
                    "column_types": {"when": "datetime64[ns]"},
                    **options,
                },
            }
        )
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "date-call"
    return result


@pytest.mark.parametrize("current_api", [False, True])
@pytest.mark.parametrize("explicit_format", [False, True])
def test_datetime_conversion_retains_values_with_supported_api(
    tmp_path, monkeypatch, current_api, explicit_format
):
    values = ["01/02/2025", "02/02/2025"] if explicit_format else None
    ns, registry = _fixture(tmp_path, monkeypatch, current_api, values)
    options = {"datetime_formats": {"when": "%d/%m/%Y"}} if explicit_format else {}
    result = _invoke(ns, registry, **options)
    assert result.artifact["results"][0]["status"] == "ok", result.artifact
    frame = registry.get_dataframe("date-fixture")
    assert pd.api.types.is_datetime64_any_dtype(frame["when"])
    start = "2025-02-01" if explicit_format else "2025-01-01"
    assert frame["when"].tolist() == list(pd.date_range(start, periods=2))
    assert frame["value"].tolist() == [10, 20]


def test_datetime_coercion_policy_remains_native(tmp_path, monkeypatch):
    ns, registry = _fixture(
        tmp_path, monkeypatch, values=["2025-01-01 00:00:00", "bad-date"]
    )
    result = _invoke(ns, registry, errors="coerce")
    assert result.artifact["results"][0]["status"] == "ok"
    frame = registry.get_dataframe("date-fixture")
    assert frame["when"].iloc[0] == pd.Timestamp("2025-01-01")
    assert pd.isna(frame["when"].iloc[1])


def test_datetime_parse_failure_remains_an_error(tmp_path, monkeypatch):
    ns, registry = _fixture(
        tmp_path, monkeypatch, values=["2025-01-01 00:00:00", "bad-date"]
    )
    before = registry.get_dataframe("date-fixture").copy()
    result = _invoke(ns, registry, errors="raise")
    assert result.artifact["results"][0]["status"] == "error"
    assert "bad-date" in result.artifact["results"][0]["detail"]
    assert registry.get_dataframe("date-fixture").equals(before)
    assert "1 issue(s)" in result.content


def test_datetime_dry_run_does_not_replace_frame(tmp_path, monkeypatch):
    ns, registry = _fixture(tmp_path, monkeypatch)
    before = registry.get_dataframe("date-fixture")
    result = _invoke(ns, registry, dry_run=True)
    assert result.artifact["action"] == "convert_types_dry_run"
    assert result.artifact["results"][0]["status"] == "ok"
    assert registry.get_dataframe("date-fixture") is before


def test_datetime_missing_dataframe_remains_failure(tmp_path, monkeypatch):
    ns, registry = _fixture(tmp_path, monkeypatch)
    registry.remove_dataframe("date-fixture")
    result = _invoke(ns, registry)
    assert result.artifact["error"]
    assert result.content.startswith("Error")
