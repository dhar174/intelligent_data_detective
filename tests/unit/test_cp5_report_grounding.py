"""Offline regressions for CP5 report context, figure provenance and routing.

The generated notebook functions and native create_agent/ToolStrategy wiring are
real. Only the model and memory boundaries are controlled.
"""

import ast
import logging
import os
import socket

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from idd_state import ReportOutline, ReportResults, Section, State, VizFeedback
from tests.unit.test_native_completion import (
    ControlledChatModel,
    FLAGS,
    execute_definitions,
    load_native,
)
from tests.unit.test_section_worker import state as section_state_fixture
from tests.unit.test_viz_join import notebook


@pytest.fixture(autouse=True)
def no_provider_access(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden in report grounding regressions")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setenv("IDD_ALLOW_RECOVERY", "0")


@pytest.fixture
def report_state(tmp_path):
    state = section_state_fixture.__wrapped__(tmp_path)
    state["user_prompt"] = "UNIQUE REQUEST: explain retail revenue"
    state["cleaned_dataset_description"] = "UNIQUE CLEANED DATA"
    state["next_agent_prompt"] = "UNIQUE OUTLINE REQUEST"
    state["messages"] = [HumanMessage(content="UNIQUE RETAINED HISTORY")]
    return state


def report_agent(tmp_path, role, response, model=None):
    ns = load_native(tmp_path, middleware=True)
    execute_definitions(ns, 25, {"report_generator_prompt_template"})
    ns["critical_complex_summary_llm"] = ns["summary_llm"]
    execute_definitions(
        ns, 44, {"prehook_critical_complex", "create_report_generator_agent"}
    )
    execute_definitions(
        ns,
        57,
        {
            "_cp5_report_inventory",
            "_cp5_ground_report_figures",
            "_cp5_report_runtime_context",
            "report_orchestrator",
            "section_worker",
        },
    )
    # Generated section worker retrieves memory from the graph store.
    ns["get_store"] = lambda: ns["in_memory_store"]
    model = model or ControlledChatModel(
        response_name=type(response).__name__, payload=response.model_dump()
    )
    for name in (
        "report_orchestrator_llm",
        "report_section_worker_llm",
        "report_packager_llm",
    ):
        ns[name] = model
    agent = ns["create_report_generator_agent"](
        df_ids=["retail_orders"], rg_agent_task=role
    )
    ns[
        {
            "outline": "report_generator_agent",
            "section": "report_section_agent",
            "package": "report_packager_agent",
        }[role]
    ] = agent
    return ns, model


def generated_supervisor():
    source = "".join(notebook()["cells"][46]["source"])
    supervisor = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "supervisor_node"
    )
    ns = dict(
        State=State,
        RunnableConfig=RunnableConfig,
        Command=Command,
        os=os,
        _pl_logger=logging.getLogger(__name__),
    )
    exec(
        compile(
            ast.Module(body=[supervisor], type_ignores=[]),
            "<generated Cell 46>",
            "exec",
        ),
        ns,
    )
    return ns["supervisor_node"]


def test_native_outline_model_receives_current_inventory_and_request(
    tmp_path, report_state
):
    report_state["visualization_results"] = {
        "visualizations": [v.model_dump() for v in report_state["viz_results"]]
    }
    report_state["viz_results"] = []  # Exercise the envelope channel alone.
    ns, model = report_agent(tmp_path, "outline", report_state["report_outline"])
    memory_queries = []
    ns["enhanced_retrieve_mem"] = (
        lambda state: memory_queries.append(state) or "UNIQUE MEMORY"
    )
    output = ns["report_orchestrator"](report_state)
    assert isinstance(output["report_outline"], ReportOutline)
    messages = model._requests[0]
    context = "\n".join(str(m.content) for m in messages)
    for value in (
        "UNIQUE REQUEST",
        "UNIQUE CLEANED DATA",
        "UNIQUE OUTLINE REQUEST",
        "UNIQUE MEMORY",
        "fig-revenue",
        "fig-discount",
    ):
        assert value in context
    assert any(m.name == "runtime_context" for m in messages)
    assert "call the `respond` tool" not in context
    assert len(memory_queries) == 1
    names = [getattr(t, "name", getattr(t, "__name__", "")) for t in model._bindings[0]]
    assert {"list_visualizations", "get_visualization", "ReportOutline"}.issubset(names)


@pytest.mark.parametrize("serialized", [False, True])
def test_envelope_only_outline_reaches_native_section_worker(
    tmp_path, report_state, serialized
):
    from idd_graph import dispatch_sections
    from idd_state import VisualizationResults

    figures = list(report_state["viz_results"])
    envelope = VisualizationResults(**FLAGS, visualizations=figures)
    report_state["visualization_results"] = (
        envelope.model_dump() if serialized else envelope
    )
    report_state["viz_results"] = []
    outline_ns, _ = report_agent(tmp_path, "outline", report_state["report_outline"])
    report_state["report_outline"] = outline_ns["report_orchestrator"](report_state)[
        "report_outline"
    ]
    sends = dispatch_sections(report_state)
    assert len(sends) == 1
    section = report_state["report_outline"].sections[0]
    response = Section(
        **FLAGS,
        name=section.name,
        section_num=section.section_num,
        description=section.description,
        goals=section.goals,
        data_signals=[],
        expected_figures=section.expected_figures,
        content="Revenue varies by channel."
    )
    worker_ns, model = report_agent(tmp_path, "section", response)
    result = worker_ns["section_worker"](sends[0].arg)
    assert len(model._requests) == 1
    assert result["sections"][0].expected_figures == figures
    assert report_state["viz_results"] == []


def test_report_grounding_is_available_from_canonical_modules(tmp_path, report_state):
    import idd_tools
    import idd_core

    collect = getattr(idd_tools, "report_figure_inventory", None)
    ground = getattr(idd_tools, "ground_report_figures", None)
    assert callable(collect) and callable(ground)
    assert idd_core.report_figure_inventory is collect
    assert idd_core.ground_report_figures is ground
    inventory = collect(report_state, working_directory=tmp_path)
    original = report_state["report_outline"].model_dump()
    result = ground(inventory, report_state["report_outline"])
    assert result == report_state["report_outline"]
    assert result is not report_state["report_outline"]
    assert report_state["report_outline"].model_dump() == original


def test_outline_inventory_tool_can_be_used_before_native_completion(
    tmp_path, report_state
):
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class LookupModel(ControlledChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if self._requests:
                return super()._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
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
                                    "id": "inventory",
                                }
                            ],
                        )
                    )
                ]
            )

    model = LookupModel(
        response_name="ReportOutline",
        payload=report_state["report_outline"].model_dump(),
    )
    report_state["visualization_results"] = {
        "visualizations": [v.model_dump() for v in report_state["viz_results"]]
    }
    report_state["viz_results"] = []
    ns, _ = report_agent(
        tmp_path, "outline", report_state["report_outline"], model=model
    )
    assert isinstance(
        ns["report_orchestrator"](report_state)["report_outline"], ReportOutline
    )
    inventory = next(
        m
        for m in model._requests[1]
        if isinstance(m, ToolMessage) and m.name == "list_visualizations"
    )
    assert "fig-revenue" in str(inventory.content) and "fig-discount" in str(
        inventory.content
    )
    assert inventory.artifact["items"]


@pytest.mark.parametrize("invalid", ["missing", "invented", "mismatch"])
def test_native_outline_rejects_ungrounded_figure_references(
    tmp_path, report_state, invalid
):
    outline = report_state["report_outline"].model_copy(deep=True)
    figure = outline.sections[0].expected_figures[0]
    if invalid == "missing":
        figure.path = figure.visualization_id = ""
    elif invalid == "invented":
        figure.visualization_id, figure.path = "V1", "path/to/figure.png"
    else:
        figure.path = outline.sections[0].expected_figures[1].path
    ns, _ = report_agent(tmp_path, "outline", outline)
    with pytest.raises(ValueError, match="(?i)(figure|inventory)"):
        ns["report_orchestrator"](report_state)


@pytest.mark.parametrize("assigned", [False, True])
@pytest.mark.parametrize("corrected", [True, False])
def test_native_section_blank_figure_gets_one_paired_correction(
    tmp_path, report_state, assigned, corrected
):
    section = report_state["section"]
    figure = section.expected_figures[0]
    blank = figure.model_copy(update={"visualization_id": "", "path": ""})
    if not assigned:
        section = section.model_copy(update={"expected_figures": []})
        report_state["section"] = section
    good = Section(
        **FLAGS,
        name=section.name,
        section_num=section.section_num,
        description=section.description,
        goals=section.goals,
        data_signals=[],
        expected_figures=section.expected_figures,
        content="Revenue varies by region." if assigned else "Methods and caveats.",
    )
    bad = good.model_copy(update={"expected_figures": [blank]})
    model = ControlledChatModel(
        response_name="Section",
        payload=(good if corrected else bad).model_dump(),
        scripted_responses=[{"name": "Section", "args": bad.model_dump()}],
    )
    ns, _ = report_agent(tmp_path, "section", good, model=model)
    if corrected:
        output = ns["section_worker"](report_state)
        assert output["sections"] == [good]
        assert output["last_agent_finished_this_task"] is True
    else:
        with pytest.raises(ValueError, match="No further correction round"):
            ns["section_worker"](report_state)
    assert len(model._requests) == 2
    errors = [m for m in model._requests[1] if isinstance(m, ToolMessage)]
    assert len(errors) == 1
    error = errors[0]
    assert error.name == "Section"
    assert error.tool_call_id == "controlled-call-1"
    assert error.status == "error"
    assert (
        "exact assigned figure IDs and paths" if assigned else "expected_figures=[]"
    ) in error.content
    assert section.expected_figures == good.expected_figures


def test_native_section_rejects_fabricated_figure(tmp_path, report_state):
    sec = report_state["section"]
    fake = sec.expected_figures[0].model_copy(
        update={"visualization_id": "V1", "path": "path/to/figure.png"}
    )
    response = Section(
        **FLAGS,
        name=sec.name,
        section_num=sec.section_num,
        description=sec.description,
        goals=sec.goals,
        data_signals=[],
        expected_figures=[fake],
        content="A section with a fabricated figure."
    )
    ns, _ = report_agent(tmp_path, "section", response)
    with pytest.raises(ValueError, match="(?i)(figure|inventory)"):
        ns["section_worker"](report_state)


@pytest.mark.parametrize("field", ["path", "visualization_id"])
def test_unique_id_or_path_is_reconciled_without_mutating_outline(
    tmp_path, report_state, field
):
    outline = report_state["report_outline"].model_copy(deep=True)
    setattr(outline.sections[0].expected_figures[0], field, "")
    ns, _ = report_agent(tmp_path, "outline", outline)
    result = ns["report_orchestrator"](report_state)["report_outline"]
    assert result.sections[0].expected_figures == report_state["viz_results"]
    assert getattr(outline.sections[0].expected_figures[0], field) == ""


def test_native_outline_rejects_ambiguous_inventory(tmp_path, report_state):
    figure = report_state["viz_results"][0]
    report_state["viz_results"].append(figure.model_copy(update={"path": "other.png"}))
    ns, _ = report_agent(tmp_path, "outline", report_state["report_outline"])
    with pytest.raises(ValueError, match="(?i)ambiguous"):
        ns["report_orchestrator"](report_state)


def test_packager_real_model_receives_draft_and_inventory_without_next_prompt(
    tmp_path, report_state
):
    from tests.unit.test_patcher_integrity import _compile_report_packager_node

    response = ReportResults(
        **FLAGS, markdown_report_path="", html_report_path="", pdf_report_path=""
    )
    ns, model = report_agent(tmp_path, "package", response)
    packager = _compile_report_packager_node(notebook(), tmp_path)
    report_state.update(
        next_agent_prompt=None, sections=[], written_sections=["UNIQUE REPORT DRAFT"]
    )
    real_invoke = ns["_safe_report_packager_invoke"]
    seen = []
    memory_queries = []

    class StopAfterNativeCompletion(Exception):
        pass

    def invoke(agent, inputs, config=None):
        seen.append(real_invoke(agent, inputs, config=config))
        raise StopAfterNativeCompletion()

    packager.__globals__.update(
        report_packager_agent=ns["report_packager_agent"],
        _safe_report_packager_invoke=invoke,
        enhanced_retrieve_mem=lambda state: memory_queries.append(state)
        or "UNIQUE MEMORY",
    )
    with pytest.raises(StopAfterNativeCompletion):
        packager(report_state)
    assert seen[0]["structured_response"] == response
    context = "\n".join(str(m.content) for m in model._requests[0])
    for value in (
        "UNIQUE REQUEST",
        "UNIQUE CLEANED DATA",
        "UNIQUE REPORT DRAFT",
        "UNIQUE MEMORY",
        "fig-revenue",
        "fig-discount",
    ):
        assert value in context
    assert any(m.name == "runtime_context" for m in model._requests[0])
    assert "call the `respond` tool" not in context
    assert len(memory_queries) == 2  # The pre-existing packager lookups only.


def test_supervisor_routes_real_revision_to_analyst_then_visualization():
    from langgraph.channels.binop import BinaryOperatorAggregate
    from idd_state import _reduce_viz_results_resettable
    import operator

    feedback = VizFeedback(
        **FLAGS,
        grade="revise",
        feedback="Correct the misleading axis.",
        redo_list=["fig-revenue"]
    )
    state = dict(
        initial_analysis_complete=True,
        data_cleaning_complete=True,
        analyst_complete=True,
        visualization_complete=True,
        report_generator_complete=False,
        last_agent_id="viz_evaluator",
        viz_grade="revise",
        viz_feedback=feedback.feedback,
        viz_eval_result=feedback,
        viz_tasks=["Rebuild the plot"],
        viz_specs=[{"title": "Revenue"}],
        next="analyst",
    )
    supervisor = generated_supervisor()
    first = supervisor(state, {})
    assert first.goto == "analyst"
    assert feedback.feedback in first.update["next_agent_prompt"]
    assert "fig-revenue" in first.update["next_agent_prompt"]
    state.update(first.update)
    state["last_agent_id"] = "analyst"
    second = supervisor(state, {})
    assert second.goto == "visualization"
    assert feedback.feedback in second.update["next_agent_prompt"]
    results = BinaryOperatorAggregate(list, _reduce_viz_results_resettable)
    results.update([["old figure"]])
    results.update([second.update["viz_results"]])
    paths = BinaryOperatorAggregate(list, operator.add)
    paths.update([["old.png"]])
    paths.update([second.update["viz_paths"]])
    assert results.get() == [] and paths.get() == []
    assert second.update["visualization_results"] is None


def test_supervisor_rejects_revision_without_specs():
    state = dict(
        initial_analysis_complete=True,
        data_cleaning_complete=True,
        analyst_complete=True,
        visualization_complete=True,
        last_agent_id="analyst",
        viz_grade="revise",
        viz_tasks=[],
        viz_specs=[],
    )
    with pytest.raises(ValueError, match="(?i)no revised"):
        generated_supervisor()(state, {})


def test_revision_uses_feedback_and_clears_paths_even_if_completion_flag_is_false():
    state = dict(
        initial_analysis_complete=True,
        data_cleaning_complete=True,
        analyst_complete=True,
        visualization_complete=False,
        last_agent_id="analyst",
        viz_grade="revise",
        viz_feedback="Fix the axis",
        viz_tasks=["Fix chart"],
        viz_specs=[{"title": "Chart"}],
    )
    result = generated_supervisor()(state, {})
    assert result.goto == "visualization"
    assert "Fix the axis" in result.update["next_agent_prompt"]
    assert result.update["viz_results"] is None
    assert "viz_paths" in result.update


def test_supervisor_routes_acceptable_evaluation_to_report():
    state = dict(
        initial_analysis_complete=True,
        data_cleaning_complete=True,
        analyst_complete=True,
        visualization_complete=True,
        report_generator_complete=False,
        last_agent_id="viz_evaluator",
        viz_grade="acceptable",
        next="report_orchestrator",
    )
    assert generated_supervisor()(state, {}).goto == "report_orchestrator"


@pytest.mark.parametrize("retry_count", [4, 5])
@pytest.mark.parametrize("recovery", ["0", "1"])
def test_supervisor_retry_exhaustion_fails_without_approving(
    monkeypatch, retry_count, recovery
):
    monkeypatch.setenv("IDD_ALLOW_RECOVERY", recovery)
    state = dict(
        initial_analysis_complete=True,
        data_cleaning_complete=True,
        analyst_complete=True,
        visualization_complete=False,
        report_generator_complete=False,
        last_agent_id="analyst",
        viz_grade=None,
        _viz_retry_count=retry_count,
    )
    before = dict(state)
    with pytest.raises(RuntimeError, match=r"(?i)visualization.*retr.*exhaust"):
        generated_supervisor()(state, {})
    assert state == before


@pytest.mark.parametrize("retry_count", [0, 1, 3])
def test_supervisor_retry_budget_still_dispatches_visualization(retry_count):
    state = dict(
        initial_analysis_complete=True,
        data_cleaning_complete=True,
        analyst_complete=True,
        visualization_complete=False,
        last_agent_id="analyst",
        _viz_retry_count=retry_count,
    )
    result = generated_supervisor()(state, {})
    assert result.goto == "visualization"
    assert result.update["_viz_retry_count"] == retry_count + 1
    assert not result.update.get("visualization_complete")


def test_supervisor_completed_evaluation_is_not_blocked_by_retry_count():
    state = dict(
        initial_analysis_complete=True,
        data_cleaning_complete=True,
        analyst_complete=True,
        visualization_complete=True,
        last_agent_id="viz_evaluator",
        viz_grade="acceptable",
        _viz_retry_count=4,
    )
    assert generated_supervisor()(state, {}).goto == "report_orchestrator"


def test_supervisor_revision_feedback_takes_precedence_over_retry_exhaustion():
    state = dict(
        initial_analysis_complete=True,
        data_cleaning_complete=True,
        analyst_complete=True,
        visualization_complete=False,
        last_agent_id="viz_evaluator",
        viz_grade="revise",
        viz_feedback="Correct the axis.",
        _viz_retry_count=4,
    )
    result = generated_supervisor()(state, {})
    assert result.goto == "analyst"
    assert "Correct the axis." in result.update["next_agent_prompt"]
