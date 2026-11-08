"""Generated-function figure provenance checks; no providers or real credentials."""

from copy import deepcopy
import json
import io
import textwrap
import ast
from pathlib import Path
import socket

import pytest
from langchain_core.messages import ToolMessage
from langchain_openai.chat_models.base import _construct_responses_api_input

from idd_state import VisualizationResults
from tests.unit.test_native_completion import (
    ControlledChatModel,
    FLAGS,
    load_native,
    worker_state,
)
from tests.unit.test_viz_join import figures, load_functions, state_for


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*a, **k):
        raise AssertionError("Network forbidden")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setenv("IDD_ALLOW_RECOVERY", "0")
    for name in ("OPENAI_API_KEY", "TAVILY_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def plot_namespace(base, monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from idd_registry import DataFrameRegistry

    ns = load_native(base, middleware=True)
    registry = DataFrameRegistry(data_dir=base / "registry")
    registry.register_dataframe(
        pd.DataFrame(
            {
                "gross": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "net": [9.0, 18.0, 25.0, 36.0, 41.0, 54.0],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        ),
        "retail-fixture",
    )
    ns.update(
        get_global_df_registry=lambda: registry,
        global_df_registry=registry,
        plt=plt,
        sns=sns,
        _HAS_SNS=True,
        nan=float("nan"),
        io=io,
        textwrap=textwrap,
        ast=ast,
    )
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(base / "artifacts"))
    return ns


PLOTS = [
    ("create_histogram", {"columns": ["net"]}),
    ("create_scatter_plot", {"x": "gross", "y": "net"}),
    ("create_box_plot", {"values": "net", "group": "group"}),
    ("create_violin_plot", {"values": "net", "group": "group"}),
]


@pytest.mark.parametrize("name,args", PLOTS)
def test_actual_plot_path_reaches_native_model_payload(
    tmp_path, monkeypatch, name, args
):
    ns = plot_namespace(tmp_path, monkeypatch)
    result = ns[name].invoke(
        {
            "type": "tool_call",
            "id": "plot-call",
            "name": name,
            "args": {"df_id": "retail-fixture", **args},
        }
    )
    assert not str(result.content).startswith("Failed"), result.content
    path = result.artifact["image_path"]
    assert Path(path).is_file()
    wire = _construct_responses_api_input([result])
    assert json.loads(wire[0]["output"])["image_path"] == path
    assert "image_base64" not in str(wire) and "image_bytes" not in str(wire)


def test_plot_failure_is_not_advertised_as_saved(tmp_path, monkeypatch):
    ns = plot_namespace(tmp_path, monkeypatch)
    result = ns["create_scatter_plot"].invoke(
        {
            "type": "tool_call",
            "id": "plot-failure",
            "name": "create_scatter_plot",
            "args": {"df_id": "retail-fixture", "x": "missing_column", "y": "net"},
        }
    )
    assert str(result.content).startswith("Failed to generate scatter plot:")
    assert not result.artifact.get("image_path")


@pytest.mark.parametrize("damage", ["missing", "empty", "corrupt", "outside", "stale"])
def test_worker_rejects_false_asset_before_memory_or_publication(tmp_path, damage):
    ns = load_native(tmp_path, middleware=True)
    item = figures(tmp_path)[0]
    path = Path(item.path)
    if damage == "missing":
        item.path = str(path.with_name("never-created.png"))
    elif damage == "empty":
        path.write_bytes(b"")
    elif damage == "corrupt":
        path.write_bytes(b"not a PNG")
    else:
        item = figures(tmp_path / "old-run")[0]
        current = tmp_path / "current-run"
        current.mkdir()
        ns["RUNTIME"].run_dir = current
    model = ControlledChatModel(
        response_name="VisualizationResults",
        payload=VisualizationResults(**FLAGS, visualizations=[item]).model_dump(),
    )
    ns["visualization_orchestrator_llm"] = model
    ns["visualization_agent"] = ns["create_visualization_agent"](["retail-fixture"])
    memory = []
    ns["update_memory_with_kind"] = lambda *a, **k: memory.append(k)
    state = worker_state(tmp_path)
    for index, message in enumerate(state["messages"]):
        message.id = f"checkpoint-message-{index}"
    before = deepcopy(state)
    with pytest.raises(ValueError, match="(?i)visualization.*(asset|figure|run)"):
        ns["viz_worker"](state)
    assert memory == []
    assert state == before


@pytest.mark.parametrize("shape", ["model", "dict", "envelope", "path_only"])
@pytest.mark.parametrize("damage", ["missing", "empty", "corrupt"])
def test_join_validates_missing_asset_before_deduplication(tmp_path, shape, damage):
    item = figures(tmp_path)[0]
    if damage == "missing":
        item.path = str(tmp_path / "never-created.png")
    else:
        Path(item.path).write_bytes(b"" if damage == "empty" else b"corrupt PNG")
    state = state_for([])
    if shape == "model":
        state["visualization_results"] = VisualizationResults(
            **FLAGS, visualizations=[item]
        )
    elif shape == "dict":
        state["viz_results"] = [item.model_dump()]
    elif shape == "envelope":
        state["viz_results"] = [
            VisualizationResults(**FLAGS, visualizations=[item]).model_dump()
        ]
    state["viz_paths"] = [item.path]
    with pytest.raises(ValueError):
        load_functions(tmp_path)["viz_join"](state)


@pytest.mark.parametrize("collision", ["id", "path"])
def test_join_rejects_conflicting_identity_bindings(tmp_path, collision):
    items = figures(tmp_path)
    other = (
        items[1].model_copy(update={"visualization_id": items[0].visualization_id})
        if collision == "id"
        else items[0].model_copy(update={"visualization_id": "different-owner"})
    )
    state = state_for([items[0], other])
    with pytest.raises(ValueError, match="(?i)conflicting"):
        load_functions(tmp_path)["viz_join"](state)


@pytest.mark.parametrize(
    "path",
    [
        "file:///tmp/chart.png",
        "../escape.png",
        "data:image/png;base64,AAA",
        "https://example.com/chart.png",
    ],
)
def test_join_rejects_unsafe_paths_without_creating_directories(tmp_path, path):
    item = figures(tmp_path)[0].model_copy(update={"path": path})
    state = state_for([item])
    state["artifacts_path"] = str(tmp_path / "not-created")
    with pytest.raises(ValueError):
        load_functions(tmp_path)["viz_join"](state)
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("envelope", [False, True])
def test_join_rejects_unfinished_duplicate_before_deduplication(tmp_path, envelope):
    item = figures(tmp_path)[0]
    state = state_for([item])
    if envelope:
        state["viz_results"] = [
            VisualizationResults(
                **{**FLAGS, "finished_this_task": False}, visualizations=[item]
            )
        ]
    else:
        state["viz_results"] = [
            item,
            item.model_copy(update={"finished_this_task": False}),
        ]
    with pytest.raises(ValueError, match="unfinished"):
        load_functions(tmp_path)["viz_join"](state)


def test_real_fallback_path_used_after_failed_custom_plot(tmp_path, monkeypatch):
    ns = plot_namespace(tmp_path, monkeypatch)
    missing = str(tmp_path / "intended-but-never-created.png")

    class FallbackModel(ControlledChatModel):
        def _generate(self, messages, **kwargs):
            outputs = [
                m
                for m in messages
                if isinstance(m, ToolMessage) and m.name == "create_scatter_plot"
            ]
            if outputs:
                visible = json.loads(str(outputs[-1].content))
                path = visible["image_path"]
                item = figures(tmp_path)[0].model_copy(
                    update={"path": path, "visualization_id": "assigned-spec-id"}
                )
                self.payload = VisualizationResults(
                    **FLAGS, visualizations=[item]
                ).model_dump()
            return super()._generate(messages, **kwargs)

    model = FallbackModel(
        response_name="VisualizationResults",
        payload={},
        scripted_responses=[
            {
                "name": "python_repl_tool",
                "args": {
                    "code": "raise NameError('controlled plotting failure')",
                    "df_id": "retail-fixture",
                },
            },
            {
                "name": "create_scatter_plot",
                "args": {"df_id": "retail-fixture", "x": "gross", "y": "net"},
            },
        ],
    )
    ns["visualization_orchestrator_llm"] = model
    ns["visualization_agent"] = ns["create_visualization_agent"](["retail-fixture"])
    out = ns["viz_worker"](worker_state(tmp_path))
    record = out["viz_results"][0]
    assert record["visualization_id"] == "assigned-spec-id"
    assert record["path"] != missing and Path(record["path"]).is_file()
    assert len(model._requests) == 3
    assert any("actual saved path" in str(m.content) for m in model._requests[0])
    assert any(
        f"Owned current-run directory: {tmp_path}" in str(m.content)
        for m in model._requests[0]
    )
    assert any(
        "controlled plotting failure" in str(m.content)
        for m in model._requests[-1]
        if isinstance(m, ToolMessage)
    )
