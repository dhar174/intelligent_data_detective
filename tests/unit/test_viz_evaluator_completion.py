"""Offline regressions for the generated native evaluator."""

import socket

import pytest
from langchain_core.messages import HumanMessage

from idd_state import VizFeedback
from tests.unit.test_native_completion import (
    ControlledChatModel,
    FLAGS,
    evaluator_state,
    load_native,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setenv("IDD_ALLOW_RECOVERY", "0")


def invoke(base, *, grade="acceptable", finished=True, revision=0):
    ns = load_native(base, middleware=True)
    state = evaluator_state(base)
    state["messages"] = [HumanMessage(content="prior history")]
    state["user_prompt"] = "UNIQUE USER REQUEST"
    state["next_agent_prompt"] = "UNIQUE EVALUATION REQUEST"
    state["cleaned_dataset_description"] = "UNIQUE DATASET"
    state["analysis_insights"] = "UNIQUE INSIGHTS"
    state["viz_revise_count"] = revision
    response = VizFeedback(
        **{**FLAGS, "finished_this_task": finished},
        grade=grade,
        feedback=(
            "The axis is misleading" if grade == "revise" else "Metadata is suitable"
        ),
        redo_list=["Retail metric 0"] if grade == "revise" else [],
    )
    model = ControlledChatModel(
        response_name="VizFeedback", payload=response.model_dump()
    )
    ns["viz_evaluator_llm"] = model
    ns["viz_evaluator_agent"] = ns["create_viz_evaluator_agent"]()
    return ns, state, model, response


def test_actual_model_receives_fresh_context(tmp_path):
    ns, state, model, response = invoke(tmp_path)
    out = ns["viz_evaluator_node"](state)
    content = "\n".join(str(m.content) for m in model._requests[0])
    for value in (
        "UNIQUE USER REQUEST",
        "UNIQUE EVALUATION REQUEST",
        "UNIQUE DATASET",
        "UNIQUE INSIGHTS",
        "fig-retail-0",
    ):
        assert value in content
    assert "{visualization_results}" not in content
    assert out["viz_eval_result"] == response


def test_revision_does_not_mutate_input_inventory(tmp_path):
    ns, state, model, response = invoke(tmp_path, grade="revise")
    original = list(state["visualization_results"].visualizations)
    out = ns["viz_evaluator_node"](state)
    assert state["visualization_results"].visualizations == original
    assert out["viz_grade"] == "revise" and out["next"] == "analyst"
    assert out["viz_eval_result"] == response


def test_exhausted_revision_is_not_approved(tmp_path):
    ns, state, _, _ = invoke(tmp_path, grade="revise", revision=2)
    with pytest.raises(ValueError, match="(?i)revision.*budget"):
        ns["viz_evaluator_node"](state)


def test_unfinished_feedback_cannot_approve(tmp_path):
    ns, state, _, _ = invoke(tmp_path, finished=False)
    with pytest.raises(ValueError, match="(?i)unfinished"):
        ns["viz_evaluator_node"](state)


def test_inventory_lookup_exposes_metadata_to_model(tmp_path):
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class LookupModel(ControlledChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self._requests.append(messages)
            name, args = (
                ("list_visualizations", {})
                if len(self._requests) == 1
                else (self.response_name, self.payload)
            )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": name,
                                    "args": args,
                                    "id": f"call-{len(self._requests)}",
                                }
                            ],
                        )
                    )
                ]
            )

    ns, state, _, response = invoke(tmp_path)
    model = LookupModel(response_name="VizFeedback", payload=response.model_dump())
    ns["viz_evaluator_llm"] = model
    ns["viz_evaluator_agent"] = ns["create_viz_evaluator_agent"]()
    ns["viz_evaluator_node"](state)
    message = next(
        m
        for m in model._requests[1]
        if isinstance(m, ToolMessage) and m.name == "list_visualizations"
    )
    assert "fig-retail-0" in str(message.content)
    assert message.artifact["items"]


def test_requested_figure_preview_reaches_responses_serialization(tmp_path):
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_openai.chat_models.base import _construct_responses_api_input

    class PreviewModel(ControlledChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self._requests.append(messages)
            name, args = (
                (
                    "get_visualization",
                    {"visualization_id": "fig-retail-0", "include_preview": True},
                )
                if len(self._requests) == 1
                else (self.response_name, self.payload)
            )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": name,
                                    "args": args,
                                    "id": f"call-{len(self._requests)}",
                                }
                            ],
                        )
                    )
                ]
            )

    ns, state, _, response = invoke(tmp_path)
    model = PreviewModel(response_name="VizFeedback", payload=response.model_dump())
    ns["viz_evaluator_llm"] = model
    ns["viz_evaluator_agent"] = ns["create_viz_evaluator_agent"]()
    ns["viz_evaluator_node"](state)
    message = next(
        m
        for m in model._requests[1]
        if isinstance(m, ToolMessage) and m.name == "get_visualization"
    )
    wire = _construct_responses_api_input([message])[0]["output"]
    assert any(block["type"] == "input_image" for block in wire)
    assert "fig-retail-0" in str(wire)
    assert message.artifact["item"]["visualization_id"] == "fig-retail-0"


def test_unchanged_repeated_inventory_fails_promptly(tmp_path):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class LoopModel(ControlledChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self._requests.append(messages)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "list_visualizations",
                                    "args": {},
                                    "id": f"loop-{len(self._requests)}",
                                }
                            ],
                        )
                    )
                ]
            )

    ns, state, _, _ = invoke(tmp_path)
    model = LoopModel(response_name="VizFeedback", payload={})
    ns["viz_evaluator_llm"] = model
    ns["viz_evaluator_agent"] = ns["create_viz_evaluator_agent"]()
    with pytest.raises(ValueError, match="(?i)repeated.*inventory"):
        ns["viz_evaluator_node"](state)
    assert len(model._requests) <= 3


def test_three_figures_complete_natively(tmp_path):
    ns, state, model, response = invoke(tmp_path)
    state["visualization_results"].visualizations = state[
        "visualization_results"
    ].visualizations[:3]
    state["viz_results"] = state["viz_results"][:3]
    state["viz_tasks"] = state["viz_tasks"][:3]
    out = ns["viz_evaluator_node"](state)
    assert out["viz_eval_result"] == response
    assert len(out["viz_results"]) == 3
    assert len(model._requests) == 1


def test_missing_metadata_is_not_approved(tmp_path):
    ns, state, _, _ = invoke(tmp_path)
    state["visualization_results"].visualizations[0].visualization_description = ""
    with pytest.raises(ValueError, match="(?i)metadata"):
        ns["viz_evaluator_node"](state)


def test_distinct_inventory_queries_are_allowed(tmp_path):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class PagingModel(ControlledChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self._requests.append(messages)
            n = len(self._requests)
            name, args = (
                ("list_visualizations", {"start": n - 1, "limit": 1})
                if n <= 3
                else (self.response_name, self.payload)
            )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {"name": name, "args": args, "id": f"page-{n}"}
                            ],
                        )
                    )
                ]
            )

    ns, state, _, response = invoke(tmp_path)
    model = PagingModel(response_name="VizFeedback", payload=response.model_dump())
    ns["viz_evaluator_llm"] = model
    ns["viz_evaluator_agent"] = ns["create_viz_evaluator_agent"]()
    out = ns["viz_evaluator_node"](state)
    assert out["viz_eval_result"] == response
    assert len(model._requests) == 4


def test_supervisor_emergency_instruction_reaches_model(tmp_path):
    from idd_state import SendAgentMessage

    ns, state, model, _ = invoke(tmp_path)
    state["supervisor_to_agent_msgs"] = [
        SendAgentMessage(
            **FLAGS,
            recipient="viz_evaluator",
            message="UNIQUE EMERGENCY INSTRUCTION",
            delivery_status=False,
            agent_obj_needs_recreated_bool=False,
            is_message_critical=True,
            immediate_emergency_reroute_to_recipient=True,
        )
    ]
    ns["viz_evaluator_node"](state)
    assert any(
        "UNIQUE EMERGENCY INSTRUCTION" in str(m.content) for m in model._requests[0]
    )


def test_native_invalid_feedback_exhausts_without_approval(tmp_path):
    from langgraph.errors import GraphRecursionError

    ns, state, _, response = invoke(tmp_path)
    payload = response.model_dump()
    payload["grade"] = "invalid"
    model = ControlledChatModel(response_name="VizFeedback", payload=payload)
    ns["viz_evaluator_llm"] = model
    ns["viz_evaluator_agent"] = ns["create_viz_evaluator_agent"]()
    with pytest.raises(GraphRecursionError):
        ns["viz_evaluator_node"](state)


def test_partial_inventory_cannot_be_approved(tmp_path):
    from idd_state import VisualizationResults

    ns, state, _, _ = invoke(tmp_path)
    state["visualization_results"] = VisualizationResults(
        **FLAGS, visualizations=state["visualization_results"].visualizations[:1]
    )
    state["viz_results"] = state["viz_results"][:1]
    with pytest.raises(ValueError, match="(?i)metadata"):
        ns["viz_evaluator_node"](state)


def test_revision_routes_through_canonical_sqlite_graph(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from idd_graph import build_graph

    ns, state, _, response = invoke(tmp_path, grade="revise")
    ids = [v.visualization_id for v in state["visualization_results"].visualizations]
    config = {"configurable": {"thread_id": "offline-revision"}}
    state["_config"] = config
    with SqliteSaver.from_conn_string(str(tmp_path / "checkpoint.sqlite")) as saver:
        graph = build_graph(
            node_overrides={
                "viz_join": ns["viz_join"],
                "viz_evaluator": ns["viz_evaluator_node"],
            },
            checkpointer=saver,
        )
        graph.update_state(config, state, as_node="viz_worker")
        result = graph.invoke(None, config, interrupt_before=["supervisor"])
        assert result["next"] == "analyst"
        assert result["viz_eval_result"] == response
        assert [
            v.visualization_id for v in result["visualization_results"].visualizations
        ] == ids
        graph.update_state(config, {"next": result["next"]}, as_node="supervisor")
        assert graph.get_state(config).next == ("analyst",)


def test_completed_native_feedback_is_recorded_in_visualization_memory(tmp_path):
    ns, state, _, response = invoke(tmp_path)
    records = []
    ns["update_memory_with_kind"] = lambda *args, **kwargs: records.append(
        kwargs["text"]
    )
    ns["viz_evaluator_node"](state)
    assert len(records) == 1
    assert response.feedback in records[0]
    assert "fig-retail-0" in records[0]
