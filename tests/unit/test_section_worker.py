"""No-key regressions executing section_worker from generated notebook Cell 57.

Only the model and memory boundaries are substituted. The function body, routing,
messages, and Pydantic input/output contracts are the application's real ones.
"""

import ast
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
import pytest

from idd_graph import PRODUCTION_NODE_NAMES, build_graph, dispatch_sections
from idd_state import (
    AnalysisInsights,
    CleaningMetadata,
    DataVisualization,
    ReportOutline,
    Section,
    SectionOutline,
    SendAgentMessage,
    State,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_NAME = "IntelligentDataDetective_beta_v5_patched.ipynb"
RESPONSE_FIELDS = dict(
    reply_msg_to_supervisor="Complete", finished_this_task=True, expect_reply=False
)
FIGURE_HEADING = (
    "The following figures are expected to be included in the report, "
    "and can be found at the corresponding paths:\n"
)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("section worker regression must never contact providers")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def load_worker(notebook, tmp_path):
    source = "".join(notebook["cells"][57]["source"])
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "section_worker"
    ]
    assert len(functions) == 1
    calls = []
    store = InMemoryStore()  # No embedding index or provider.

    def invoke(payload, config):
        # default_instruction is a local; observe it without rewriting the worker.
        # W13J renders a compact prompt separately, which must remain unchanged.
        worker_locals = sys._getframe(1).f_locals
        calls.append(
            SimpleNamespace(
                payload=payload,
                config=config,
                instruction=worker_locals["default_instruction"],
                user_t=worker_locals["user_t"],
                figures=worker_locals["expected_viz_str"],
            )
        )
        outline = payload["section"]
        result = Section(
            **RESPONSE_FIELDS,
            name=outline.name,
            section_num=outline.section_num,
            description=outline.description,
            goals=outline.goals,
            data_signals=list(outline.data_signals_needed),
            expected_figures=outline.expected_figures,
            content="Revenue varies by channel; correlation r = -0.1607.",
        )
        return {
            "structured_response": result,
            "messages": [AIMessage(content=result.content)],
        }

    namespace = {
        cls.__name__: cls
        for cls in (
            State,
            Section,
            SectionOutline,
            CleaningMetadata,
            AnalysisInsights,
            SendAgentMessage,
            Command,
            AIMessage,
            HumanMessage,
            ChatPromptTemplate,
            MessagesPlaceholder,
        )
    }
    namespace.update(
        sample_prompt_text="Analyze retail orders.",
        report_generator_tools=[],
        DEFAULT_TOOLING_GUIDELINES="Use registered DataFrame IDs.",
        report_generator_prompt_template=ChatPromptTemplate.from_messages(
            [("system", "{report_task}"), MessagesPlaceholder("messages")]
        ),
        report_section_agent=SimpleNamespace(invoke=invoke),
        get_store=lambda: store,
        in_memory_store=store,
        update_memory_with_kind=lambda *args, **kwargs: None,
        WORKING_DIRECTORY=tmp_path,
    )
    # Keep original line numbers for actionable Cell 57 tracebacks.
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "<Cell 57>", "exec"),
        namespace,
    )
    return namespace["section_worker"], calls


@pytest.fixture
def worker(tmp_path):
    notebook = json.loads((REPO_ROOT / NOTEBOOK_NAME).read_text(encoding="utf-8"))
    return load_worker(notebook, tmp_path)


@pytest.fixture
def state(tmp_path):
    figures = [
        DataVisualization(
            **RESPONSE_FIELDS,
            path=str(tmp_path / filename),
            visualization_id=figure_id,
            visualization_type=kind,
            visualization_description=description,
            visualization_style="whitegrid",
            visualization_title=title,
        )
        for filename, figure_id, kind, description, title in [
            ("revenue.png", "fig-revenue", "bar", "Revenue by channel.", "Revenue"),
            (
                "discount.png",
                "fig-discount",
                "scatter",
                "Discount vs revenue.",
                "Discount",
            ),
        ]
    ]
    section = SectionOutline(
        **RESPONSE_FIELDS,
        name="Revenue drivers",
        section_num=1,
        description="Explain channel revenue and discount effects.",
        goals=["Compare revenue drivers"],
        data_signals_needed={"net_revenue": "Revenue after discounts"},
        data_signals_available=["retail_orders"],
        expected_figures=figures,
        word_target=300,
    )
    return {
        "section": section,
        "report_outline": ReportOutline(
            **section.model_dump(), title="Retail report", sections=[section]
        ),
        "cleaning_metadata": CleaningMetadata(
            **RESPONSE_FIELDS,
            steps_taken=["Removed duplicate orders"],
            data_description_after_cleaning="180 cleaned retail orders.",
        ),
        "analysis_insights": AnalysisInsights(
            **RESPONSE_FIELDS,
            summary="Online is the largest revenue channel.",
            correlation_insights="Discount and revenue: r = -0.1607.",
            anomaly_insights="High-discount orders warrant review.",
            recommended_visualizations=[],
            recommended_next_steps=["Review discount policy"],
        ),
        "viz_results": figures,
        "available_df_ids": ["retail_orders"],
        "user_prompt": "Analyze retail revenue drivers.",
        "messages": [AIMessage(content="Write the revenue section.")],
        "final_turn_msgs_list": [],
        "next_agent_prompt": None,
        "_config": {"configurable": {"thread_id": "section-no-key"}},
    }


def figure_line(figure):
    return (
        f"Title: {figure.visualization_title} : Type: {figure.visualization_type} "
        f"Description: {figure.visualization_description} : Path: {figure.path} "
        f"with id : {figure.visualization_id}\n"
    )


def test_multiple_figures_reach_model_with_exact_metadata(worker, state):
    run, calls = worker
    result = run(state)
    assert len(calls) == 1
    call = calls[0]
    expected = FIGURE_HEADING + "".join(map(figure_line, state["viz_results"]))
    assert call.figures == expected
    assert expected in call.instruction
    assert f"Figures to reference: {expected}" in call.payload["messages"][0].content
    for figure in state["viz_results"]:
        assert call.instruction.count(figure_line(figure)) == 1
        assert call.payload["messages"][0].content.count(figure_line(figure)) == 1
    assert call.payload["section"] is state["section"]
    assert result["section_complete"] is True
    assert result["sections"][0].expected_figures == state["viz_results"]
    assert "r = -0.1607" in result["written_sections"][0]


def test_without_figures_reaches_model(worker, state):
    state["section"] = state["section"].model_copy(update={"expected_figures": []})
    state["viz_results"] = []
    run, calls = worker
    assert run(state)["section_complete"] is True
    assert calls[0].figures == FIGURE_HEADING
    assert "Title:" not in calls[0].instruction
    assert calls[0].payload["section"].expected_figures == []


@pytest.mark.parametrize("prompt", ["Prioritize channel comparisons", ""])
def test_supervisor_instruction_composition_and_compact_prompt_unchanged(
    worker, state, prompt
):
    run, calls = worker
    run(state)
    original = calls[-1]
    state["next_agent_prompt"] = prompt
    run(state)
    supplied = calls[-1]
    assert supplied.instruction == f"{prompt}. {original.instruction}"
    assert original.instruction.startswith(original.user_t)
    # W13J already uses its section-only prompt and clears next_agent_prompt.
    # Preserve that effective behavior as well as the local prefix composition.
    assert supplied.payload["messages"] == original.payload["messages"]
    assert supplied.payload["next_agent_prompt"] is None


def test_canonical_graph_dispatch_reaches_generated_worker(worker, state):
    run, calls = worker
    sends = dispatch_sections(state)
    assert len(sends) == 1 and sends[0].node == "report_section_worker"
    assert isinstance(sends[0].arg["section"], dict)
    overrides = {name: lambda state: {} for name in PRODUCTION_NODE_NAMES}
    overrides["supervisor"] = lambda state: {"next": "report_orchestrator"}
    overrides["report_section_worker"] = run
    graph = build_graph(node_overrides=overrides, structural_only=False)
    result = graph.invoke(state, interrupt_after=["report_section_worker"])
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call.payload["section"], SectionOutline)
    for key in (
        "cleaning_metadata",
        "analysis_insights",
        "viz_results",
        "available_df_ids",
    ):
        assert call.payload[key] == state[key]
    assert call.config == state["_config"]
    # The canonical State intentionally filters worker-only section_complete.
    assert result["sections"][0].name == state["section"].name
    assert result["last_agent_finished_this_task"] is True


@pytest.mark.parametrize(
    "missing, destination",
    [("cleaning_metadata", "data_cleaner"), ("analysis_insights", "analyst")],
)
def test_missing_context_preserves_rerouting_without_model_call(
    worker, state, missing, destination
):
    run, calls = worker
    state[missing] = None
    result = run(state)
    assert isinstance(result, Command)
    assert result.goto == destination
    assert calls == []


def test_fresh_regeneration_is_idempotent_and_worker_executable(tmp_path, state):
    # Generate outside the checkout; the patcher always starts from the source.
    source_name = "IntelligentDataDetective_beta_v5.ipynb"
    (tmp_path / source_name).write_bytes((REPO_ROOT / source_name).read_bytes())
    committed = (REPO_ROOT / NOTEBOOK_NAME).read_bytes()
    for _ in range(2):
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "_patch_notebook.py")],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
        generated = (tmp_path / NOTEBOOK_NAME).read_bytes()
        assert generated == committed
        notebook = json.loads(generated)
        assert len(notebook["cells"]) == 99
        run, calls = load_worker(notebook, tmp_path)
        assert run(state)["section_complete"] is True
        assert len(calls) == 1
