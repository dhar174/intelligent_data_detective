"""Generated-function figure provenance checks; no providers or real credentials."""

from copy import deepcopy
from dataclasses import replace
import json
import io
import textwrap
import ast
import base64
from pathlib import Path
import socket
import re

import pytest
from langchain_core.messages import ToolMessage
from langchain_openai.chat_models.base import _construct_responses_api_input

from idd_state import DataVisualization, VisualizationResults, VizSpec
from tests.unit.test_native_completion import (
    ControlledChatModel,
    FLAGS,
    load_native,
    worker_state,
)
from tests.unit.test_viz_join import (
    apply_join,
    figures,
    load_functions,
    production_runtime,
    state_for,
)


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
    ns["RUNTIME"] = production_runtime(base / "artifacts" / "run-fixture")
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
    monkeypatch.delenv("IDD_ARTIFACTS_DIR", raising=False)
    return ns


PLOTS = [
    ("create_histogram", {"columns": ["net"]}),
    ("create_scatter_plot", {"x": "gross", "y": "net"}),
    ("create_box_plot", {"values": "net", "group": "group"}),
    ("create_violin_plot", {"values": "net", "group": "group"}),
    (
        "create_correlation_heatmap",
        {"columns": ["gross", "net"], "figsize": [6.0, 4.0]},
    ),
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
    assert Path(path).parent == ns["RUNTIME"].viz_dir
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
        ns["RUNTIME"] = production_runtime(current)
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
    expected = (
        "outside the current run"
        if damage in ("outside", "stale")
        else "not a readable PNG" if damage == "corrupt" else "missing or invalid"
    )
    with pytest.raises(ValueError, match=expected):
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
    assert type(ns["RUNTIME"]).__name__ == "RuntimeCtx"
    assert not hasattr(ns["RUNTIME"], "run_dir")
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
        f"Owned current-run directory: {ns['RUNTIME'].artifacts_dir}" in str(m.content)
        for m in model._requests[0]
    )
    assert any(
        "controlled plotting failure" in str(m.content)
        for m in model._requests[-1]
        if isinstance(m, ToolMessage)
    )


@pytest.mark.parametrize("path_only", [False, True])
@pytest.mark.parametrize("with_base", [False, True])
def test_real_runtime_owned_png_is_accepted(tmp_path, path_only, with_base):
    root = tmp_path / "artifacts" / "run-fixture"
    ns = load_functions(root)
    ns["WORKING_DIRECTORY"] = tmp_path
    assert type(ns["RUNTIME"]).__name__ == "RuntimeCtx"
    assert not hasattr(ns["RUNTIME"], "run_dir")
    items = figures(root)
    state = state_for([] if path_only else items)
    state["viz_paths"] = [str(Path(item.path).relative_to(root)) for item in items]
    if with_base:
        state["artifacts_path"] = str(root)
    else:
        state.pop("artifacts_path", None)
    result = apply_join(state, ns["viz_join"](state))
    actual = result["visualization_results"].visualizations
    assert len(actual) == len(items)
    assert {Path(item.path).resolve() for item in actual} == {
        Path(item.path) for item in items
    }
    if not path_only:
        assert actual == items


def test_missing_real_runtime_artifact_root_fails_closed(tmp_path):
    ns = load_functions(tmp_path)
    ns["RUNTIME"] = replace(ns["RUNTIME"], artifacts_dir=None)
    with pytest.raises(ValueError, match="no current run"):
        ns["_cp5_validated_figure_path"](figures(tmp_path)[0].path, {})


def test_worker_missing_runtime_root_fails_before_model_call(tmp_path):
    ns = load_native(tmp_path, middleware=True)
    ns["RUNTIME"] = replace(ns["RUNTIME"], artifacts_dir=None)
    model = ControlledChatModel(response_name="VisualizationResults", payload={})
    ns["visualization_orchestrator_llm"] = model
    ns["visualization_agent"] = ns["create_visualization_agent"](["retail-fixture"])
    memory = []
    ns["update_memory_with_kind"] = lambda *a, **k: memory.append(k)
    with pytest.raises(ValueError, match="no current run"):
        ns["viz_worker"](worker_state(tmp_path))
    assert model._requests == []
    assert memory == []


def test_real_runtime_rejects_existing_older_run_png(tmp_path):
    ns = load_functions(tmp_path / "current-run")
    older = figures(tmp_path / "older-run")[0]
    with pytest.raises(ValueError, match="outside the current run"):
        ns["_cp5_validated_figure_path"](older.path, {})


HEATMAP_SELECTIONS = [
    (["net", "gross"], "none", ["net", "gross"]),
    ([1, 0], "none", ["net", "gross"]),
    (None, "none", ["gross", "net"]),
    ("all", "none", ["gross", "net"]),
    (["net", "gross"], "alphabetical", ["gross", "net"]),
]


@pytest.mark.parametrize("columns,order,expected", HEATMAP_SELECTIONS)
@pytest.mark.parametrize("seaborn", [False, True])
@pytest.mark.parametrize("return_bytes", [False, True])
def test_heatmap_saves_actual_matrix_columns_and_payload(
    tmp_path, monkeypatch, columns, order, expected, seaborn, return_bytes
):
    import numpy as np
    from matplotlib.axes import Axes
    from PIL import Image

    ns = plot_namespace(tmp_path, monkeypatch)
    ns["_HAS_SNS"] = seaborn
    assert "cols" not in ns
    assert not hasattr(ns["RUNTIME"], "run_dir")
    observed = []
    if seaborn:
        original = ns["sns"].heatmap

        def capture(matrix, **kwargs):
            observed.append(matrix.to_numpy(copy=True))
            return original(matrix, **kwargs)

        monkeypatch.setattr(ns["sns"], "heatmap", capture)
    else:
        original = Axes.imshow

        def capture(axis, matrix, *args, **kwargs):
            observed.append(np.array(matrix, copy=True))
            return original(axis, matrix, *args, **kwargs)

        monkeypatch.setattr(Axes, "imshow", capture)
    result = ns["create_correlation_heatmap"].invoke(
        {
            "type": "tool_call",
            "id": "heatmap-matrix",
            "name": "create_correlation_heatmap",
            "args": {
                "df_id": "retail-fixture",
                "columns": columns,
                "order": order,
                "return_bytes": return_bytes,
                "figsize": [6.0, 4.0],
            },
        }
    )
    assert not str(result.content).startswith("Failed"), result.content
    artifact = result.artifact
    path = Path(artifact["image_path"])
    assert path.parent == ns["RUNTIME"].viz_dir
    assert re.fullmatch(
        rf"retail-fixture__correlation_heatmap__{'_'.join(expected)}__[0-9a-f]{{8}}\.png",
        path.name,
    )
    assert artifact["columns"] == expected and artifact["n_cols"] == len(expected)
    assert artifact["plot_type"] == "correlation_heatmap"
    expected_matrix = (
        ns["get_global_df_registry"]().get_dataframe("retail-fixture")[expected].corr()
    )
    assert len(observed) == 1
    np.testing.assert_allclose(observed[0], expected_matrix.to_numpy())
    with Image.open(path) as image:
        assert image.format == "PNG" and min(image.size) > 0
        image.verify()
    encoded = (
        artifact["image_bytes"]
        if return_bytes
        else base64.b64decode(artifact["image_base64"])
    )
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.format == "PNG"
        image.verify()
    wire = _construct_responses_api_input([result])
    assert json.loads(wire[0]["output"])["image_path"] == str(path)
    assert "image_bytes" not in str(wire) and "image_base64" not in str(wire)
    assert not (tmp_path / "figures").exists()


@pytest.mark.parametrize("failure", ["missing_df", "one_numeric", "empty", "save"])
def test_heatmap_failures_never_advertise_success(tmp_path, monkeypatch, failure):
    import pandas as pd
    from matplotlib.figure import Figure

    ns = plot_namespace(tmp_path, monkeypatch)
    df_id = "retail-fixture"
    if failure == "missing_df":
        df_id = "not-registered"
    elif failure == "one_numeric":
        ns["get_global_df_registry"]().register_dataframe(
            pd.DataFrame({"number": [1, 2], "text": ["a", "b"]}), df_id
        )
    elif failure == "empty":
        ns["get_global_df_registry"]().register_dataframe(
            pd.DataFrame({"a": [float("nan")], "b": [float("nan")]}), df_id
        )
    else:

        def fail_save(*args, **kwargs):
            raise OSError("controlled heatmap save failure")

        monkeypatch.setattr(Figure, "savefig", fail_save)
    result = ns["create_correlation_heatmap"].invoke(
        {
            "type": "tool_call",
            "id": "heatmap-error",
            "name": "create_correlation_heatmap",
            "args": {"df_id": df_id, "figsize": [6.0, 4.0]},
        }
    )
    assert "Correlation heatmap generated." not in str(result.content)
    assert not result.artifact.get("image_path")
    assert not list(ns["RUNTIME"].artifacts_dir.rglob("*.png"))
    if failure == "save":
        assert "controlled heatmap save failure" in str(result.content)


def test_heatmap_native_worker_and_join_preserve_actual_path_and_id(
    tmp_path, monkeypatch
):
    ns = plot_namespace(tmp_path, monkeypatch)
    spec = VizSpec(
        **FLAGS,
        title="Retail correlation",
        viz_type="auto",
        df_id="retail-fixture",
        viz_instructions="Create a correlation heatmap for gross and net.",
        viz_id="assigned-heatmap-id",
        columns=["gross", "net"],
        x=None,
        y=None,
        hue=None,
        bins=None,
        agg=None,
        query=None,
        description="Numeric correlations",
        limit=None,
        style="whitegrid",
    )

    class HeatmapModel(ControlledChatModel):
        def _generate(self, messages, **kwargs):
            outputs = [
                m
                for m in messages
                if isinstance(m, ToolMessage) and m.name == "create_correlation_heatmap"
            ]
            if outputs:
                visible = json.loads(str(outputs[-1].content))
                item = DataVisualization(
                    **FLAGS,
                    path=visible["image_path"],
                    visualization_id=spec.viz_id,
                    visualization_type="correlation_heatmap",
                    visualization_title=spec.title,
                    visualization_description="Pearson correlation of registered gross and net columns.",
                    visualization_style="coolwarm",
                )
                self.payload = VisualizationResults(
                    **FLAGS, visualizations=[item]
                ).model_dump()
            return super()._generate(messages, **kwargs)

    model = HeatmapModel(
        response_name="VisualizationResults",
        payload={},
        scripted_responses=[
            {
                "name": "create_correlation_heatmap",
                "args": {
                    "df_id": "retail-fixture",
                    "columns": ["gross", "net"],
                    "figsize": [6.0, 4.0],
                },
            }
        ],
    )
    ns["visualization_orchestrator_llm"] = model
    ns["visualization_agent"] = ns["create_visualization_agent"](["retail-fixture"])
    state = worker_state(tmp_path)
    state.update(
        viz_spec=spec, viz_specs=[spec], individual_viz_task=spec.viz_instructions
    )
    memory = []
    ns["update_memory_with_kind"] = lambda *a, **k: memory.append(k)
    out = ns["viz_worker"](state)
    assert len(memory) == 1
    joined = apply_join({**state, **out}, ns["viz_join"]({**state, **out}))
    item = joined["visualization_results"].visualizations[0]
    assert item.visualization_id == spec.viz_id
    assert Path(item.path).parent == ns["RUNTIME"].viz_dir
    assert len(model._requests) == 2
    assert model._emitted == ["create_correlation_heatmap", "VisualizationResults"]
    assert any(
        f"Assigned visualization_id: {spec.viz_id}" in str(m.content)
        for m in model._requests[0]
    )
    assert joined["visualization_complete"] is True
