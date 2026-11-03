"""SQLite-backed canonical revision path with controlled, offline model responses."""

import socket

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from idd_graph import build_graph
from idd_state import VizFeedback, VizSpec, VisualizationResults
from tests.unit.test_cp5_report_grounding import generated_supervisor
from tests.unit.test_native_completion import (
    ControlledChatModel,
    FLAGS,
    evaluator_state,
    load_native,
)
from tests.unit.test_viz_join import figures


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden in revision graph test")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setenv("IDD_ALLOW_RECOVERY", "0")


def test_revision_regenerates_before_acceptable_report_and_restores_sqlite(tmp_path):
    ns = load_native(tmp_path, middleware=True)
    state = evaluator_state(tmp_path)
    old = list(state["visualization_results"].visualizations)
    revised = figures(tmp_path / "revised")
    tasks = [v.visualization_title for v in revised]
    specs = [
        VizSpec(
            **FLAGS,
            title=v.visualization_title,
            viz_type="bar",
            df_id="retail-fixture",
            viz_instructions=v.visualization_title,
            viz_id=v.visualization_id,
            columns=["region", "revenue"],
            x="region",
            y="revenue",
            hue=None,
            bins=None,
            agg="sum",
            query=None,
            description=v.visualization_description,
            limit=20,
            style=v.visualization_style,
        )
        for v in revised
    ]
    thread = {"configurable": {"thread_id": "revision-checkpoint"}}
    state.update(
        _config=thread,
        viz_tasks=tasks,
        viz_specs=specs,
        viz_paths=[v.path for v in old],
        initial_analysis_complete=True,
        data_cleaning_complete=True,
        analyst_complete=True,
        visualization_complete=True,
        report_generator_complete=False,
        last_agent_id="viz_join",
    )

    class RevisionModel(ControlledChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            revise = len(self._requests) == 0
            self.payload = VizFeedback(
                **FLAGS,
                grade="revise" if revise else "acceptable",
                feedback=(
                    "Correct the misleading axis."
                    if revise
                    else "The revised figures are clear."
                ),
                redo_list=["fig-retail-0"] if revise else [],
            ).model_dump()
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

    evaluator = RevisionModel(response_name="VizFeedback", payload={})
    ns["viz_evaluator_llm"] = evaluator
    ns["viz_evaluator_agent"] = ns["create_viz_evaluator_agent"]()
    worker = ControlledChatModel(
        response_name="VisualizationResults",
        payload={},
        task_payloads={
            v.visualization_title: VisualizationResults(
                **FLAGS, visualizations=[v]
            ).model_dump()
            for v in revised
        },
    )
    ns["visualization_orchestrator_llm"] = worker
    ns["visualization_agent"] = ns["create_visualization_agent"](["retail-fixture"])
    seen = []

    def analyst(current):
        seen.append(current["next_agent_prompt"])
        return {
            "viz_tasks": tasks,
            "viz_specs": specs,
            "last_agent_id": "analyst",
            "analyst_complete": True,
        }

    def visualization(current):
        assert current["viz_results"] == []
        assert current["visualization_results"] is None
        assert current["viz_paths"] == []
        return {
            "viz_tasks": tasks,
            "viz_specs": specs,
            "last_agent_id": "visualization_orchestrator",
        }

    overrides = {
        "supervisor": generated_supervisor(),
        "analyst": analyst,
        "visualization": visualization,
        "viz_worker": ns["viz_worker"],
        "viz_join": ns["viz_join"],
        "viz_evaluator": ns["viz_evaluator_node"],
    }
    db = tmp_path / "revision.sqlite"
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(node_overrides=overrides, checkpointer=saver)
        graph.update_state(thread, state, as_node="viz_join")
        result = graph.invoke(None, thread, interrupt_before=["report_orchestrator"])
        assert graph.get_state(thread).next == ("report_orchestrator",)
        assert result["viz_grade"] == "acceptable"
        assert result["visualization_results"].visualizations == revised
        assert result["viz_paths"] == [v.path for v in revised]
        assert all(v.path not in result["viz_paths"] for v in old)
        assert len(evaluator._requests) == 2 and len(worker._requests) == len(revised)
        assert len(seen) == 1 and "Correct the misleading axis." in seen[0]
        assert "fig-retail-0" in seen[0]
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(node_overrides=overrides, checkpointer=saver)
        restored = graph.get_state(thread)
        assert restored.next == ("report_orchestrator",)
        assert restored.values["visualization_results"].visualizations == revised
        assert restored.values["viz_grade"] == "acceptable"


def test_exhausted_retries_stop_the_canonical_graph_without_a_report(tmp_path):
    items = figures(tmp_path)
    state = {
        "initial_analysis_complete": True,
        "data_cleaning_complete": True,
        "analyst_complete": True,
        "visualization_complete": False,
        "report_generator_complete": False,
        "last_agent_id": "analyst",
        "viz_grade": None,
        "viz_results": [item.model_dump() for item in items],
        "visualization_results": VisualizationResults(**FLAGS, visualizations=items),
        "viz_paths": [item.path for item in items],
        "_viz_retry_count": 4,
    }
    calls = []

    def unexpected(current):
        calls.append(current)
        return {}

    overrides = {
        "supervisor": generated_supervisor(),
        "visualization": unexpected,
        "viz_evaluator": unexpected,
        "report_orchestrator": unexpected,
    }
    cfg = {"configurable": {"thread_id": "exhausted-retries"}}
    db = tmp_path / "exhausted.sqlite"
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(node_overrides=overrides, checkpointer=saver)
        graph.update_state(cfg, state, as_node="analyst")
        with pytest.raises(RuntimeError, match=r"(?i)visualization.*retr.*exhaust"):
            graph.invoke(None, cfg)
        assert calls == []
        pending = graph.get_state(cfg)
        assert pending.next == ("supervisor",)
        assert pending.values["visualization_complete"] is False
        assert pending.values["viz_grade"] is None
        assert pending.values["_viz_retry_count"] == 4
        assert pending.values["visualization_results"].visualizations == items
        assert pending.values["viz_paths"] == [item.path for item in items]
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(node_overrides=overrides, checkpointer=saver)
        restored = graph.get_state(cfg)
        assert restored.next == ("supervisor",)
        assert restored.values["visualization_complete"] is False
        assert restored.values["viz_grade"] is None
