"""Offline CP5 fan-in regressions executing the generated Cell 57 functions.

Figures and reports here are temporary test fixtures, never production proof.
"""

import ast
from datetime import datetime
import json
import logging
from pathlib import Path
import socket
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from PIL import Image, ImageDraw
import pytest

from idd_state import DataVisualization, State, VisualizationResults
from idd_state import _reduce_viz_results_resettable

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = "IntelligentDataDetective_beta_v5_patched.ipynb"
FLAGS = dict(
    reply_msg_to_supervisor="Done", finished_this_task=True, expect_reply=False
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError(
            "Provider/network access forbidden in offline fan-in tests"
        )

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def notebook():
    return json.loads((ROOT / NOTEBOOK).read_text(encoding="utf-8"))


def load_functions(base, names=("viz_join", "viz_evaluator_node")):
    source = "".join(notebook()["cells"][57]["source"])
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in nodes} == set(names)
    ns = dict(
        State=State,
        Command=Command,
        DataVisualization=DataVisualization,
        VisualizationResults=VisualizationResults,
        datetime=datetime,
        AIMessage=AIMessage,
        HumanMessage=HumanMessage,
        PathlibPath=Path,
        WORKING_DIRECTORY=base,
        RUNTIME=SimpleNamespace(run_dir=base, artifacts_dir=base / "artifacts"),
        ALIASES={},
        sample_prompt_text="Offline retail fixture",
        in_memory_store=None,
        get_store=lambda: None,
        update_memory_with_kind=lambda *a, **k: None,
        _pl_logger=logging.getLogger(__name__),
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<generated Cell 57>", "exec"),
        ns,
    )
    return ns


def figures(base):
    directory = base / "visualizations"
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(4):
        path = directory / f"owned_chart_{i}.png"  # Deliberately differs from ID.
        image = Image.new("RGB", (240, 160), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 30 + i * 10, 120 + i * 20, 140), fill=(40, 90, 160))
        image.save(path)
        results.append(
            DataVisualization(
                **FLAGS,
                path=str(path),
                visualization_id=f"fig-retail-{i}",
                visualization_title=f"Retail metric {i}",
                visualization_type="bar",
                visualization_style="whitegrid",
                visualization_description=f"Controlled retail series {i}; four distinct observations.",
            )
        )
    return results


def state_for(items):
    return dict(
        _config={},
        visualization_results=VisualizationResults(**FLAGS, visualizations=items),
        viz_results=[],
        viz_paths=[],
        viz_tasks=[],
    )


def apply_join(state, update):
    # Exercise LangGraph's real reduced channel with the ordered node writes.
    # A dict alone would collapse the clear and append into just the last value.
    if isinstance(update, Command):
        assert not update.goto
        writes = list(update.update)
    else:
        writes = list(update.items())
    channel = BinaryOperatorAggregate(list, _reduce_viz_results_resettable)
    channel.update([state.get("viz_results") or []])
    channel.update([value for key, value in writes if key == "viz_results"])
    return {**state, **dict(writes), "viz_results": channel.get()}


@pytest.mark.parametrize(
    "shape", ["pydantic", "dict", "list", "mixed_list", "jsonplus"]
)
def test_structured_inventory_survives_without_tasks_or_disk_scan(
    tmp_path, monkeypatch, shape
):
    items = figures(tmp_path)
    state = state_for(items)
    payload = state["visualization_results"]
    if shape == "dict":
        payload = payload.model_dump(mode="json")
    elif shape == "list":
        payload = items
    elif shape == "mixed_list":
        payload = [v.model_dump() if i % 2 else v for i, v in enumerate(items)]
    elif shape == "jsonplus":
        serde = JsonPlusSerializer()
        kind, data = serde.dumps_typed(payload.model_dump(mode="json"))
        payload = serde.loads_typed((kind, data))
        assert isinstance(payload, dict)
        assert all(isinstance(v, dict) for v in payload["visualizations"])
    state["visualization_results"] = payload

    def forbidden_scan(*a, **k):
        raise AssertionError("Structured results must not need disk discovery")

    monkeypatch.setattr(Path, "rglob", forbidden_scan)
    update = apply_join(state, load_functions(tmp_path)["viz_join"](state))
    assert update["visualization_results"].visualizations == items
    assert update["visualization_complete"] is True
    joined = update
    assert [DataVisualization.model_validate(v) for v in joined["viz_results"]] == items


def test_union_deduplicates_ids_and_paths_preserving_real_metadata(tmp_path):
    items = figures(tmp_path)
    state = state_for(items[1:])
    state["visualization_results"] = state["visualization_results"].model_dump()
    state["viz_results"] = [items[0].model_dump(), items[1].model_dump()]
    state["viz_paths"] = [v.path for v in items] + [
        str(Path(items[0].path).parent / "." / Path(items[0].path).name)
    ]
    update = apply_join(state, load_functions(tmp_path)["viz_join"](state))
    assert update["visualization_results"].visualizations == items
    joined = update
    assert [DataVisualization.model_validate(v) for v in joined["viz_results"]] == items


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"visualizations": [{}], **FLAGS},
        {"visualizations": "wrong", **FLAGS},
        [None],
        [dict(path="missing.png")],
    ],
)
def test_malformed_payload_fails_instead_of_empty_success(tmp_path, payload):
    state = state_for([])
    state["visualization_results"] = payload
    with pytest.raises(ValueError):
        load_functions(tmp_path)["viz_join"](state)


@pytest.mark.parametrize("payload", [None, [], {"visualizations": [], **FLAGS}])
def test_missing_required_figures_cannot_be_completed_from_stale_pngs(
    tmp_path, payload
):
    figures(tmp_path / "historical")
    state = state_for([])
    state.update(visualization_results=payload, viz_tasks=["required"] * 4)
    with pytest.raises(ValueError, match="(?i)(missing|expected|figure)"):
        load_functions(tmp_path)["viz_join"](state)


def test_no_inventory_no_tasks_does_not_claim_completion(tmp_path):
    state = state_for([])
    update = apply_join(state, load_functions(tmp_path)["viz_join"](state))
    assert update["visualization_complete"] is False
    assert update["last_agent_finished_this_task"] is False
    assert update["visualization_results"].finished_this_task is False


def test_no_task_evaluation_preserves_inventory_without_claiming_quality(tmp_path):
    items = figures(tmp_path)
    state = state_for(items)
    ns = load_functions(tmp_path)
    joined = apply_join(state, ns["viz_join"](state))
    update = ns["viz_evaluator_node"](joined)
    assert "visualization_results" not in update
    assert "viz_results" not in update
    assert joined["visualization_results"].visualizations == items
    assert update["viz_grade"] == "not_evaluated"
    assert update["last_agent_finished_this_task"] is False
    assert "skipping" in update["viz_feedback"]


@pytest.mark.parametrize("serialized", [False, True])
def test_worker_snapshot_containers_in_viz_results(tmp_path, serialized):
    items = figures(tmp_path)
    state = state_for(items)
    snapshots = [VisualizationResults(**FLAGS, visualizations=[v]) for v in items]
    state["viz_results"] = (
        [v.model_dump() for v in snapshots] if serialized else snapshots
    )
    update = apply_join(state, load_functions(tmp_path)["viz_join"](state))
    assert update["visualization_results"].visualizations == items
    joined = update
    assert [DataVisualization.model_validate(v) for v in joined["viz_results"]] == items
    assert all(
        isinstance(v, dict) and "visualizations" not in v for v in joined["viz_results"]
    )


def test_explicit_owned_path_only_channel(tmp_path):
    items = figures(tmp_path)
    state = state_for([])
    state["viz_paths"] = [items[0].path]
    update = apply_join(state, load_functions(tmp_path)["viz_join"](state))
    joined = update["visualization_results"].visualizations
    assert len(joined) == 1
    assert joined[0].path == items[0].path


def test_explicit_unrelated_path_is_rejected(tmp_path):
    items = figures(tmp_path / "old_attempt")
    current = tmp_path / "current_attempt"
    current.mkdir()
    state = state_for([])
    state["viz_paths"] = [items[0].path]
    with pytest.raises(ValueError, match="outside the current run"):
        load_functions(current)["viz_join"](state)


def test_unfinished_required_metadata_is_not_success(tmp_path):
    items = figures(tmp_path)
    items[0].finished_this_task = False
    with pytest.raises(ValueError, match="unfinished"):
        load_functions(tmp_path)["viz_join"](state_for(items))


def test_repeated_join_does_not_append_duplicate_reporting_items(tmp_path):
    items = figures(tmp_path)
    ns = load_functions(tmp_path)
    state = state_for(items)
    joined = apply_join(state, ns["viz_join"](state))
    again = apply_join(joined, ns["viz_join"](joined))
    assert again["viz_results"] == joined["viz_results"]
    assert again["visualization_results"].visualizations == items


def test_task_evaluation_does_not_duplicate_reporting_channel(tmp_path):
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from idd_state import VizFeedback

    items = figures(tmp_path)
    state = state_for(items)
    state.update(
        viz_tasks=[v.visualization_title for v in items],
        viz_specs=[],
        messages=[AIMessage(content="Evaluate the four figures.")],
    )
    ns = load_functions(
        tmp_path, ("viz_join", "viz_evaluator_node", "_safe_viz_evaluator_invoke")
    )
    calls = []

    def evaluate(payload, config):
        calls.append(payload)
        assert payload["viz_results"] == items
        return {
            "structured_response": VizFeedback(
                **FLAGS,
                grade="acceptable",
                feedback="Controlled assessment: all four charts have readable labels.",
                redo_list=[],
            ),
            "messages": [AIMessage(content="Controlled evaluation complete.")],
        }

    ns.update(
        VizFeedback=VizFeedback,
        ChatPromptTemplate=ChatPromptTemplate,
        MessagesPlaceholder=MessagesPlaceholder,
        enhanced_retrieve_mem=lambda s: "",
        get_global_df_registry=lambda: None,
        viz_evaluator_prompt_template=ChatPromptTemplate.from_messages(
            [("system", "Evaluate figures.")]
        ),
        viz_evaluator_agent=SimpleNamespace(invoke=evaluate),
    )
    joined = apply_join(state, ns["viz_join"](state))
    evaluated = apply_join(joined, ns["viz_evaluator_node"](joined))
    assert len(calls) == 1
    assert evaluated["viz_grade"] == "acceptable"
    assert [
        DataVisualization.model_validate(v) for v in evaluated["viz_results"]
    ] == items


def test_previously_completed_inventory_cannot_be_replaced_by_empty_result(tmp_path):
    state = state_for([])
    state["visualization_complete"] = True
    with pytest.raises(ValueError, match="(?i)(missing|lost|figure)"):
        load_functions(tmp_path)["viz_join"](state)


def test_relative_artifact_metadata_and_absolute_path_channel_are_one_figure(tmp_path):
    items = figures(tmp_path)
    relative = [
        v.model_copy(update={"path": str(Path(v.path).relative_to(tmp_path))})
        for v in items
    ]
    state = state_for(relative)
    state.update(artifacts_path=str(tmp_path), viz_paths=[v.path for v in items])
    update = apply_join(state, load_functions(tmp_path)["viz_join"](state))
    assert update["visualization_results"].visualizations == relative


def test_duplicate_id_or_equivalent_path_does_not_inflate_inventory(tmp_path):
    items = figures(tmp_path)
    state = state_for(items)
    state["visualization_results"] = items + [
        items[0].model_copy(update={"path": items[1].path}),
        items[1].model_copy(
            update={"visualization_id": "another-id-for-the-same-file"}
        ),
    ]
    update = apply_join(state, load_functions(tmp_path)["viz_join"](state))
    assert update["visualization_results"].visualizations == items
