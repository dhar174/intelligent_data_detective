"""Offline regressions for native analyst completion and revision context."""

import json
import socket
import subprocess

from PIL import Image
import pytest
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite

import idd_state
import tests.unit.test_native_completion as native_completion
from tests.unit.test_native_completion import (
    ControlledChatModel,
    FLAGS,
    analyst_figures,
    analysis_insights,
    analyst_state,
    initial_description,
    load_analyst,
)

PINNED_BASE = "88bac1c4c0b5687ceaee9cc01e9514270380214f"


@pytest.fixture(autouse=True)
def offline_analyst(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("External access forbidden in analyst revision tests")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setenv("IDD_ALLOW_RECOVERY", "0")
    for name in (
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _make_agent(ns, model, state):
    ns["analyst_llm"] = model
    ns["analyst_agent"] = ns["create_analyst_agent"](
        initial_description(), state["available_df_ids"]
    )


def _tool_name(tool):
    if isinstance(tool, dict):
        return tool.get("name")
    return getattr(tool, "name", None)


def _contains_context_value(context, value):
    return value in context or json.dumps(value, ensure_ascii=False)[1:-1] in context


def test_analyst_initial_completion_uses_native_schema_without_respond(tmp_path):
    ns = load_analyst(tmp_path)
    expected = analysis_insights()
    model = ControlledChatModel(
        response_name="AnalysisInsights",
        payload=expected.model_dump(),
        follow_instructions=True,
    )
    state = analyst_state(tmp_path)
    _make_agent(ns, model, state)

    update = ns["analyst_node"](state)

    assert update["analysis_insights"] == expected
    assert update["analyst_complete"] is True
    assert update["last_agent_finished_this_task"] is True
    assert model._emitted == ["AnalysisInsights"]
    assert len(model._requests) == 1
    bound_names = {
        name for binding in model._bindings for name in map(_tool_name, binding)
    }
    assert model._binding_kwargs
    assert any(call.get("tool_choice") == "any" for call in model._binding_kwargs)
    assert "AnalysisInsights" in bound_names
    assert "respond" not in bound_names
    assert {"get_column_names", "list_visualizations", "get_visualization"}.issubset(
        bound_names
    )
    request_text = "\n".join(str(message.content) for message in model._requests[0])
    assert "call the `respond` tool" not in request_text


def test_analyst_revision_receives_real_feedback_png_inventory_and_supervisor_instruction(
    tmp_path,
):
    ns = load_analyst(tmp_path)
    expected = analysis_insights()
    instruction = (
        "Revise the missing regional revenue bar chart and the revenue versus "
        "order-value scatter plot. Apply the evaluator's requested fixes."
    )
    existing_figures = analyst_figures(tmp_path)
    feedback = idd_state.VizFeedback(
        **FLAGS,
        grade="revise",
        feedback=(
            "The regional revenue bar chart is missing the North category. "
            "The revenue versus order-value scatter plot needs a trend line."
        ),
        redo_list=[
            "Bar chart — Mean revenue by region (PNG); include the North category.",
            "Verify the scatter plot's fitted trend line and annotate unusual points.",
        ],
    )
    state = analyst_state(
        tmp_path,
        supervisor_instruction=instruction,
        viz_feedback=feedback.feedback,
        viz_eval_result=feedback,
        figures=existing_figures,
        prior_insights=analysis_insights(),
    )
    memory_note = (
        "Earlier evidence note: North revenue was highest in the clean sample."
    )
    ns["enhanced_retrieve_mem"] = lambda _: memory_note
    assert isinstance(state["viz_eval_result"], idd_state.VizFeedback)
    assert state["viz_feedback"] == feedback.feedback
    assert state["analysis_config"] is idd_state.default_an_config
    assert ns["analyst_df_id"] in ns["analyst_registry"].registry
    assert len(list((tmp_path / "analyst-visualizations").glob("*.png"))) == 2
    for item in existing_figures:
        with Image.open(item.path) as image:
            image.verify()
        assert item.visualization_id in item.model_dump_json()
        assert item.visualization_title in item.model_dump_json()
        assert item.visualization_type in item.model_dump_json()

    model = ControlledChatModel(
        response_name="AnalysisInsights",
        payload=expected.model_dump(),
        required_context=[
            instruction,
            feedback.feedback,
            memory_note,
            *feedback.redo_list,
            *(item.path for item in existing_figures),
            *(item.visualization_title for item in existing_figures),
        ],
    )
    _make_agent(ns, model, state)

    update = ns["analyst_node"](state)

    assert update["analysis_insights"] == expected
    request = model._requests[0]
    runtime_contexts = [
        message for message in request if message.name == "runtime_context"
    ]
    assert runtime_contexts
    context = str(runtime_contexts[-1].content)
    assert instruction in context
    assert feedback.feedback in context
    assert feedback.redo_list[0] in context and feedback.redo_list[1] in context
    assert all(_contains_context_value(context, item.path) for item in existing_figures)
    assert all(item.visualization_title in context for item in existing_figures)


def test_analyst_native_unfinished_output_fails_with_integrity_error(tmp_path):
    ns = load_analyst(tmp_path)
    payload = analysis_insights().model_dump()
    payload["finished_this_task"] = False
    model = ControlledChatModel(response_name="AnalysisInsights", payload=payload)
    state = analyst_state(tmp_path)
    _make_agent(ns, model, state)

    with pytest.raises((ValueError, RuntimeError), match=r"^analyst integrity:"):
        ns["analyst_node"](state)

    assert model._emitted == ["AnalysisInsights"]


def test_analyst_malformed_native_payload_preserves_structured_output_error(tmp_path):
    ns = load_analyst(tmp_path)
    payload = analysis_insights().model_dump()
    del payload["correlation_insights"]
    model = ControlledChatModel(response_name="AnalysisInsights", payload=payload)
    state = analyst_state(tmp_path)
    _make_agent(ns, model, state)

    with pytest.raises((StructuredOutputValidationError, GraphRecursionError)):
        ns["analyst_node"](state)

    assert model._emitted
    assert set(model._emitted) == {"AnalysisInsights"}


def test_analyst_revision_without_native_response_cannot_reuse_prior_completion(
    tmp_path,
):
    ns = load_analyst(tmp_path)
    expected = analysis_insights()
    model = ControlledChatModel(
        response_name="AnalysisInsights",
        payload=expected.model_dump(),
        prose_on_calls=list(range(2, 162)),
        prose_response="I reviewed the feedback and will revise the analysis.",
    )
    initial = analyst_state(tmp_path)
    initial["_config"] = {"configurable": {"thread_id": "analyst-revision-same-thread"}}
    _make_agent(ns, model, initial)
    first_update = ns["analyst_node"](initial)
    assert first_update["analysis_insights"] == expected

    instruction = "Reassess the North region and explain the scatter plot relationship."
    revision = analyst_state(
        tmp_path,
        supervisor_instruction=instruction,
        history=[
            *initial["messages"],
            *first_update["messages"],
            HumanMessage(
                content="Retained analysis note: North had the highest revenue."
            ),
            AIMessage(content=instruction, name="supervisor"),
        ],
    )
    revision["_config"] = initial["_config"]
    revision["analysis_insights"] = first_update["analysis_insights"]
    revision["analyst_complete"] = True

    with pytest.raises(
        (ValueError, GraphRecursionError, StructuredOutputValidationError)
    ) as caught:
        ns["analyst_node"](revision)

    if isinstance(caught.value, ValueError):
        assert str(caught.value).startswith("analyst integrity:")
    assert len(model._requests) >= 2
    assert all(response is None for response in model._emitted[1:])


def test_analyst_retains_history_and_emergency_without_mutating_caller_state(tmp_path):
    ns = load_analyst(tmp_path)
    expected = analysis_insights()
    instruction = "Recompute the North region finding now."
    emergency_text = "Emergency instruction: confirm the North region comparison."
    history = [
        HumanMessage(content="Original request: compare revenue by region."),
        AIMessage(
            content="Retained history: North revenue was 120 in the sample.",
            name="analyst",
        ),
        AIMessage(content=instruction, name="supervisor"),
    ]
    state = analyst_state(
        tmp_path,
        supervisor_instruction=instruction,
        history=history,
    )
    emergency = idd_state.SendAgentMessage(
        **FLAGS,
        recipient="analyst",
        message=emergency_text,
        delivery_status=False,
        agent_obj_needs_recreated_bool=True,
        is_message_critical=True,
        immediate_emergency_reroute_to_recipient=True,
    )
    unrelated_pending = idd_state.SendAgentMessage(
        **FLAGS,
        recipient="visualization",
        message="Keep the visualization assignment pending.",
        delivery_status=False,
        agent_obj_needs_recreated_bool=False,
        is_message_critical=False,
        immediate_emergency_reroute_to_recipient=False,
    )
    state["emergency_reroute"] = "analyst"
    original_supervisor_messages = [emergency, unrelated_pending]
    state["supervisor_to_agent_msgs"] = original_supervisor_messages
    original_messages = state["messages"]
    original_message_values = list(original_messages)
    model = ControlledChatModel(
        response_name="AnalysisInsights",
        payload=expected.model_dump(),
        required_context=[instruction, emergency_text],
    )
    _make_agent(ns, model, state)

    update = ns["analyst_node"](state)

    assert update["analysis_insights"] == expected
    request_text = "\n".join(str(message.content) for message in model._requests[-1])
    assert "Retained history: North revenue was 120 in the sample." in request_text
    assert state["messages"] is original_messages
    assert state["messages"] == original_message_values
    assert state["supervisor_to_agent_msgs"][0] is emergency
    assert emergency.delivery_status is False
    assert state["supervisor_to_agent_msgs"] is original_supervisor_messages
    assert unrelated_pending.delivery_status is False

    delivery_update = update["supervisor_to_agent_msgs"]
    assert isinstance(delivery_update, Overwrite)
    assert len(delivery_update.value) == 2
    assert delivery_update.value[0].delivery_status is True
    assert delivery_update.value[1].delivery_status is False
    assert delivery_update.value[0] is not emergency
    assert delivery_update.value[1] is not unrelated_pending

    graph = StateGraph(idd_state.State)
    graph.add_node(
        "deliver",
        lambda _: {"supervisor_to_agent_msgs": delivery_update},
    )
    graph.add_edge(START, "deliver")
    graph.add_edge("deliver", END)
    reduced = graph.compile().invoke(
        {"supervisor_to_agent_msgs": original_supervisor_messages}
    )
    assert len(reduced["supervisor_to_agent_msgs"]) == 2
    assert [item.message for item in reduced["supervisor_to_agent_msgs"]] == [
        emergency.message,
        unrelated_pending.message,
    ]
    assert reduced["supervisor_to_agent_msgs"] == delivery_update.value
    assert emergency.delivery_status is False
    assert unrelated_pending.delivery_status is False


def _lookup(name, **args):
    return {"name": name, "args": args}


def _lookup_tool_messages(model_request):
    return [message for message in model_request if isinstance(message, ToolMessage)]


def _load_pinned_base_notebook(monkeypatch):
    source = subprocess.run(
        [
            "git",
            "show",
            f"{PINNED_BASE}:IntelligentDataDetective_beta_v5_patched.ipynb",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    pinned = json.loads(source)
    monkeypatch.setattr(native_completion, "notebook", lambda: pinned)
    return pinned


@pytest.mark.parametrize("pattern", ["identical", "alternating"])
def test_pinned_base_nonempty_visualization_retrievals_reach_recursion_limit(
    tmp_path, monkeypatch, pattern
):
    pinned = _load_pinned_base_notebook(monkeypatch)
    assert len(pinned["cells"]) == 99
    ns = load_analyst(tmp_path)
    items = analyst_figures(tmp_path)
    state = analyst_state(tmp_path, figures=items)
    # The base analyst node does not pass the visualization inventory through its
    # child state, so the real tool reads the same nonempty PNG inventory from disk.
    artifacts_dir = str(tmp_path / "analyst-visualizations")
    if pattern == "identical":
        calls = [
            _lookup(
                "list_visualizations",
                start=0,
                limit=50,
                artifacts_dir=artifacts_dir,
            )
        ] * 200
    else:
        sequence = [
            _lookup(
                "list_visualizations",
                start=0,
                limit=50,
                artifacts_dir=artifacts_dir,
            ),
            _lookup(
                "get_visualization", path=items[0].path, artifacts_dir=artifacts_dir
            ),
            _lookup(
                "list_visualizations",
                start=1,
                limit=1,
                artifacts_dir=artifacts_dir,
            ),
            _lookup(
                "get_visualization", path=items[1].path, artifacts_dir=artifacts_dir
            ),
        ]
        calls = [sequence[index % len(sequence)] for index in range(200)]
    model = ControlledChatModel(
        response_name="AnalysisInsights",
        payload=analysis_insights().model_dump(),
        scripted_responses=calls,
    )
    _make_agent(ns, model, state)

    with pytest.raises(GraphRecursionError) as caught:
        ns["analyst_node"](state)

    assert "160" in str(caught.value)
    # Each model/tool cycle consumes four graph steps, so the 160-step cap
    # produces 40 actual model requests in this native agent.
    assert len(model._requests) == 40
    assert len(model._emitted) == 40
    assert model._emitted[:4] == [call["name"] for call in calls[:4]]
    tool_contents = [
        str(message.content)
        for request in model._requests
        for message in _lookup_tool_messages(request)
    ]
    assert tool_contents
    assert any(
        _contains_context_value(content, items[0].path) for content in tool_contents
    ), tool_contents[:3]
    assert any(
        _contains_context_value(content, items[1].path) for content in tool_contents
    ), tool_contents[:3]


@pytest.mark.parametrize("recovery_mode", ["0", "1"])
def test_four_identical_visualization_lookups_fail_before_another_model_call(
    tmp_path, monkeypatch, recovery_mode
):
    monkeypatch.setenv("IDD_ALLOW_RECOVERY", recovery_mode)
    ns = load_analyst(tmp_path)
    items = analyst_figures(tmp_path)
    state = analyst_state(tmp_path, figures=items)
    expected = analysis_insights()
    same_list = _lookup("list_visualizations", start=0, limit=1)
    model = ControlledChatModel(
        response_name="AnalysisInsights",
        payload=expected.model_dump(),
        scripted_responses=[same_list] * 4,
    )
    _make_agent(ns, model, state)

    with pytest.raises(
        ValueError,
        match=r"^analyst integrity: repeated unchanged visualization retrieval",
    ):
        ns["analyst_node"](state)

    assert len(model._requests) == 4
    prior_results = _lookup_tool_messages(model._requests[-1])
    assert len(prior_results) == 3
    assert len({str(message.content) for message in prior_results}) == 1


def test_alternating_known_visualization_results_fail_after_three_redundant_results(
    tmp_path,
):
    ns = load_analyst(tmp_path)
    items = analyst_figures(tmp_path)
    state = analyst_state(tmp_path, figures=items)
    expected = analysis_insights()
    list_first = _lookup("list_visualizations", start=0, limit=1)
    get_first = _lookup("get_visualization", visualization_id=items[0].visualization_id)
    model = ControlledChatModel(
        response_name="AnalysisInsights",
        payload=expected.model_dump(),
        scripted_responses=[list_first, get_first, list_first, get_first, list_first],
    )
    _make_agent(ns, model, state)

    with pytest.raises(
        ValueError,
        match=r"^analyst integrity: repeated unchanged visualization retrieval",
    ):
        ns["analyst_node"](state)

    assert len(model._requests) == 5
    tool_names = [
        message.name
        for message in _lookup_tool_messages(model._requests[-1])
        if message.name in {"list_visualizations", "get_visualization"}
    ]
    assert tool_names == ["list_visualizations", "get_visualization"] * 2


def test_successful_dataframe_tool_resets_repeated_visualization_lookup_streak(
    tmp_path,
):
    ns = load_analyst(tmp_path)
    items = analyst_figures(tmp_path)
    state = analyst_state(tmp_path, figures=items)
    expected = analysis_insights()
    same_list = _lookup("list_visualizations", start=0, limit=1)
    model = ControlledChatModel(
        response_name="AnalysisInsights",
        payload=expected.model_dump(),
        scripted_responses=[
            same_list,
            same_list,
            same_list,
            _lookup("get_column_names", df_id="retail-fixture"),
            same_list,
            same_list,
            same_list,
        ],
    )
    _make_agent(ns, model, state)

    with pytest.raises(
        ValueError,
        match=r"^analyst integrity: repeated unchanged visualization retrieval",
    ):
        ns["analyst_node"](state)

    assert len(model._requests) == 7
    names = [
        message.name
        for message in _lookup_tool_messages(model._requests[-1])
        if message.name
    ]
    assert names == [
        "list_visualizations",
        "list_visualizations",
        "list_visualizations",
        "get_column_names",
        "list_visualizations",
        "list_visualizations",
    ]


def test_unknown_respond_tool_never_creates_a_fabricated_completion(tmp_path):
    ns = load_analyst(tmp_path)
    model = ControlledChatModel(
        response_name="AnalysisInsights",
        payload=analysis_insights().model_dump(),
        scripted_responses=[_lookup("respond")],
        prose_on_calls=list(range(2, 162)),
        prose_response="I will continue reviewing the task.",
    )
    state = analyst_state(tmp_path)
    _make_agent(ns, model, state)

    with pytest.raises(
        (ValueError, GraphRecursionError, StructuredOutputValidationError)
    ):
        ns["analyst_node"](state)

    assert model._emitted[0] == "respond"
    assert all(response is None for response in model._emitted[1:])


def test_distinct_and_preview_visualization_retrievals_are_legitimate_progress(
    tmp_path,
):
    ns = load_analyst(tmp_path)
    items = analyst_figures(tmp_path)
    state = analyst_state(tmp_path, figures=items)
    expected = analysis_insights()
    model = ControlledChatModel(
        response_name="AnalysisInsights",
        payload=expected.model_dump(),
        scripted_responses=[
            _lookup("list_visualizations", start=0, limit=1),
            _lookup("get_visualization", visualization_id=items[0].visualization_id),
            _lookup("list_visualizations", start=1, limit=1),
            _lookup("get_visualization", visualization_id=items[1].visualization_id),
            _lookup(
                "get_visualization",
                visualization_id=items[0].visualization_id,
                include_preview=True,
            ),
            _lookup("AnalysisInsights", **expected.model_dump()),
        ],
    )
    _make_agent(ns, model, state)

    update = ns["analyst_node"](state)

    assert update["analysis_insights"] == expected
    assert len(model._requests) == 6
    final_request_text = "\n".join(
        str(message.content) for message in model._requests[-1]
    )
    assert items[0].visualization_id in final_request_text
    assert items[1].visualization_id in final_request_text
    assert "data:image/png;base64," in final_request_text
    prior_results = _lookup_tool_messages(model._requests[-1])
    paged_lists = [
        message for message in prior_results if message.name == "list_visualizations"
    ]
    assert len(paged_lists) == 2
    assert str(paged_lists[0].content) != str(paged_lists[1].content)
    assert items[0].visualization_id in str(paged_lists[0].content)
    assert items[1].visualization_id in str(paged_lists[1].content)
