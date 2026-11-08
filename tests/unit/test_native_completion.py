"""Offline #177 regressions against generated functions and real inner agents.

Only chat/summary models and embedding-backed memory are controlled boundaries.
No notebook model configuration, credential loader or live runner is executed.
"""

import ast
import base64
import builtins
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
import glob
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time
import textwrap
from threading import Event
from types import SimpleNamespace
import typing
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    ModelResponse,
    before_model,
    wrap_model_call,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from PIL import Image, ImageDraw
from pydantic import Field, PrivateAttr, ValidationError
import pytest

import idd_core
import idd_state
from idd_registry import DataFrameRegistry, get_global_df_registry
from idd_registry import override_global_registry
from tests.unit.test_viz_join import FLAGS, figures, notebook, production_runtime

WRAPPERS = [
    ("initial_analysis", "InitialDescription", "initial_analyst_llm", "IA"),
    ("data_cleaner", "CleaningMetadata", "data_cleaner_llm", "DC"),
    ("analyst", "AnalysisInsights", "analyst_llm", "AN"),
    ("visualization", "VisualizationResults", "visualization_orchestrator_llm", "VZ"),
    ("viz_evaluator", "VizFeedback", "viz_evaluator_llm", "VE"),
    ("report_orchestrator", "ReportOutline", "report_orchestrator_llm", "RO"),
    ("report_packager", "ReportResults", "report_packager_llm", "RP"),
]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("External access forbidden in native completion tests")

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


def execute_definitions(ns, cell, names=None):
    source = "".join(notebook()["cells"][cell]["source"])
    nodes = ast.parse(source).body
    if names is not None:
        nodes = [
            n
            for n in nodes
            if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
            or (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
            )
        ]
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), f"<generated Cell {cell}>", "exec"
        ),
        ns,
    )


def load_native(base, *, middleware=False):
    ns = {**vars(typing), **vars(idd_core), **vars(idd_state)}
    # Explicit imports required by the unchanged generated function bodies.
    ns.update(
        datetime=datetime,
        glob=glob,
        json=json,
        os=os,
        re=re,
        shutil=shutil,
        uuid=uuid,
        ast=ast,
        io=io,
        textwrap=textwrap,
        builtins=builtins,
    )
    ns.update(base64=base64, hashlib=hashlib)
    for value in (
        create_agent,
        AgentState,
        ModelRequest,
        ModelResponse,
        before_model,
        wrap_model_call,
        ToolStrategy,
        AIMessage,
        BaseMessage,
        ChatMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
        ChatPromptTemplate,
        MessagesPlaceholder,
        InMemorySaver,
        Runtime,
        Command,
        get_global_df_registry,
    ):
        ns[value.__name__] = value
    ns.update(
        WORKING_DIRECTORY=base,
        RUNTIME=production_runtime(base),
        PathlibPath=Path,
        in_memory_store=InMemoryStore(),
        get_store=lambda: None,
        enhanced_retrieve_mem=lambda state: "No stored memories",
        update_memory_with_kind=lambda *a, **k: None,
        _pl_logger=logging.getLogger(__name__),
        ALIASES={},
        sample_prompt_text="Offline retail fixture",
        use_local_llm=False,
        USE_STRICT_JSON_SCHEMA_FINAL_HOP=False,
        data_cleaning_tools=[],
        visualization_tools=[],
        _cp5_wrap_model_call=wrap_model_call,
    )
    execute_definitions(
        ns,
        25,
        {
            "DEFAULT_TOOLING_GUIDELINES",
            "analyst_prompt_template_main",
            "data_cleaner_prompt_template",
            "visualization_prompt_template",
            "viz_evaluator_prompt_template",
        },
    )
    if middleware:
        from langchain.tools import tool, ToolRuntime
        from langchain_experimental.tools import PythonAstREPLTool
        from langgraph.prebuilt import InjectedState, InjectedStore
        from langmem import create_manage_memory_tool, create_search_memory_tool
        from matplotlib.figure import Figure
        import numpy as np
        import pandas as pd

        ns.update(
            tool=tool,
            ToolRuntime=ToolRuntime,
            PythonAstREPLTool=PythonAstREPLTool,
            InjectedState=InjectedState,
            InjectedStore=InjectedStore,
            create_manage_memory_tool=create_manage_memory_tool,
            create_search_memory_tool=create_search_memory_tool,
            Figure=Figure,
            pd=pd,
            np=np,
            InMemoryStore=InMemoryStore,
            contextmanager=contextmanager,
        )
        # Real capper, decorated business tools, catalogs and memory-tool wiring.
        # The in-memory store has no embedding index; no business tools are stubbed.
        execute_definitions(ns, 31)
        execute_definitions(ns, 32)
        execute_definitions(ns, 42)
        # Actual pre-model summarization middleware; its model is also offline.
        ns.update(CONTEXT_HEADROOM=512, MAX_CONTEXT=32000)
        execute_definitions(ns, 43)
        summary = GenericFakeChatModel(
            messages=iter([AIMessage(content="Fixture context")] * 200)
        )
        ns.update(
            summary_llm=summary,
            quick_summary_llm=summary,
            complex_summary_llm=summary,
            critical_complex_summary_llm=summary,
        )
        execute_definitions(
            ns,
            44,
            {
                "prehook",
                "prehook_quick",
                "prehook_complex",
                "prehook_critical_complex",
            },
        )
    execute_definitions(
        ns,
        44,
        {
            "_make_unknown_tool_guard",
            "create_analyst_agent",
            "create_data_cleaner_agent",
            "create_visualization_agent",
            "create_viz_evaluator_agent",
        },
    )
    execute_definitions(
        ns,
        54,
        {
            "save_viz_for_state",
            "_resolve_artifacts_root",
            "_safe_copy",
            "_ensure_list_str",
        },
    )
    execute_definitions(
        ns,
        57,
        {
            *(f"_safe_{r}_invoke" for r, *_ in WRAPPERS),
            "data_cleaner_node",
            "viz_worker",
            "viz_join",
            "_cp5_current_run_root",
            "_cp5_validated_figure_path",
            "_cp5_cleaner_messages",
            "_cp5_validate_cleaner_completion",
            "viz_evaluator_node",
        },
    )
    return ns


def load_analyst(base):
    """Load the real generated analyst factory, middleware, and node offline."""
    ns = load_native(base, middleware=True)
    ns["default_an_config"] = idd_state.default_an_config
    registry = DataFrameRegistry(data_dir=base / "analyst-registry")
    df_id = "retail-fixture"
    dataframe = ns["pd"].DataFrame(
        {
            "region": ["North", "South", "East", "West"],
            "revenue": [120.0, 95.0, 110.0, 102.0],
            "order_value": [50.0, 42.0, 47.0, 45.0],
        }
    )
    assert registry.register_dataframe(df=dataframe, df_id=df_id) == df_id
    assert df_id in registry.registry
    ns.update(
        analyst_registry=registry,
        analyst_df_id=df_id,
        get_global_df_registry=lambda: registry,
        global_df_registry=registry,
    )
    for cell, base_names in (
        (44, {"create_analyst_agent", "prehook_critical_complex"}),
        (57, {"_safe_analyst_invoke", "analyst_node"}),
    ):
        source = "".join(notebook()["cells"][cell]["source"])
        names = set(base_names)
        for node in ast.parse(source).body:
            targets = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_cp5_analyst"):
                    names.add(node.name)
            elif isinstance(node, ast.Assign):
                targets = [
                    target for target in node.targets if isinstance(target, ast.Name)
                ]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
            names.update(
                target.id for target in targets if target.id.startswith("_cp5_analyst")
            )
        execute_definitions(ns, cell, names)
    return ns


class ControlledChatModel(BaseChatModel):
    """Real create_agent/ToolStrategy execution, scripted model boundary only."""

    response_name: str
    payload: dict
    follow_instructions: bool = False
    task_payloads: dict[str, dict] = Field(default_factory=dict)
    required_context: list[str] = Field(default_factory=list)
    scripted_responses: list[dict] = Field(default_factory=list)
    prose_on_calls: list[int] = Field(default_factory=list)
    prose_response: str = "The requested analysis is complete."
    _requests: list = PrivateAttr(default_factory=list)
    _bindings: list = PrivateAttr(default_factory=list)
    _binding_kwargs: list = PrivateAttr(default_factory=list)
    _script_index: int = PrivateAttr(default=0)
    _emitted: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-native-completion"

    def bind_tools(self, tools, **kwargs):
        self._bindings.append(tools)
        self._binding_kwargs.append(dict(kwargs))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._requests.append(messages)
        name = self.response_name
        payload = self.payload
        if self.required_context:
            contexts = [m for m in messages if m.name == "runtime_context"]
            assert (
                contexts
            ), "Native analyst must receive a named runtime_context message"
            context = str(contexts[-1].content)
            for expected in self.required_context:
                encoded = json.dumps(expected, ensure_ascii=False)[1:-1]
                assert (
                    expected in context or encoded in context
                ), f"Missing assigned runtime context: {expected}"
        if self.task_payloads:
            # Inner-agent history may include earlier sibling tasks. Follow the
            # newest explicit runtime task, as the production instructions require.
            contexts = [m for m in messages if m.name == "runtime_context"]
            assert contexts, "Native worker must retain its rendered runtime context"
            context = str(contexts[-1].content)
            matches = [v for k, v in self.task_payloads.items() if k in context]
            assert (
                len(matches) == 1
            ), "Native worker must receive its exact assigned task"
            payload = matches[0]
            assert all(
                v["visualization_id"] in context for v in payload["visualizations"]
            ), "The native model must receive the assigned figure IDs"
        if len(self._requests) in self.prose_on_calls:
            self._emitted.append(None)
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content=self.prose_response))
                ]
            )
        if self.scripted_responses and self._script_index < len(
            self.scripted_responses
        ):
            scripted = self.scripted_responses[self._script_index]
            self._script_index += 1
            if "content" in scripted:
                self._emitted.append(None)
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(content=str(scripted["content"]))
                        )
                    ]
                )
            name = scripted["name"]
            payload = scripted.get("args", {})
        if self.follow_instructions and any(
            "call the `respond` tool" in str(m.content) for m in messages
        ):
            name = "respond"
        self._emitted.append(name)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": name,
                                "args": payload,
                                "id": f"controlled-call-{len(self._requests)}",
                            }
                        ],
                    )
                )
            ]
        )


def cleaner_result():
    return idd_state.CleaningMetadata(
        **FLAGS,
        steps_taken=["Checked missing values and duplicate rows"],
        data_description_after_cleaning="Clean controlled retail fixture",
    )


def initial_description():
    return idd_state.InitialDescription(
        **FLAGS,
        dataset_description="Controlled retail fixture",
        data_sample="region,revenue\nNorth,120",
        notes="No issues",
    )


def analysis_insights():
    return idd_state.AnalysisInsights(
        **FLAGS,
        summary="North region revenue leads the controlled retail fixture.",
        correlation_insights="Revenue increases with order value in the sample.",
        anomaly_insights="No outlier is supported by this small controlled sample.",
        recommended_visualizations=[
            idd_state.VizSpec(
                **FLAGS,
                title="Revenue by region",
                viz_type="bar",
                df_id="retail-fixture",
                viz_instructions="Plot total revenue by region.",
                viz_id="retail-region-revenue",
                columns=["region", "revenue"],
                x="region",
                y="revenue",
                hue=None,
                bins=None,
                agg="sum",
                query=None,
                description="Total revenue for each region.",
                limit=None,
                style="whitegrid",
            ),
            idd_state.VizSpec(
                **FLAGS,
                title="Revenue versus order value",
                viz_type="scatter",
                df_id="retail-fixture",
                viz_instructions=(
                    "Plot order value against revenue with a fitted trend line."
                ),
                viz_id="retail-revenue-order-scatter",
                columns=["order_value", "revenue"],
                x="order_value",
                y="revenue",
                hue=None,
                bins=None,
                agg=None,
                query=None,
                description="Relationship between order value and revenue.",
                limit=None,
                style="whitegrid",
            ),
        ],
        recommended_next_steps=["Check regional revenue after the revised chart."],
    )


def analyst_state(
    base,
    *,
    supervisor_instruction=None,
    viz_feedback=None,
    viz_eval_result=None,
    figures=None,
    prior_insights=None,
    history=None,
):
    state = worker_state(base)
    instruction = supervisor_instruction or (
        "Analyze the cleaned retail fixture and report evidence-backed findings."
    )
    state.update(
        user_prompt="Analyze revenue by region and order value.",
        messages=history
        or [
            HumanMessage(content="Analyze revenue by region and order value."),
            AIMessage(content=instruction, name="supervisor"),
        ],
        final_turn_msgs_list=[AIMessage(content=instruction, name="supervisor")],
        next_agent_prompt=instruction,
        analysis_config=idd_state.default_an_config,
        viz_feedback=(
            viz_feedback.feedback
            if isinstance(viz_feedback, idd_state.VizFeedback)
            else viz_feedback
        ),
        viz_grade=(
            viz_eval_result.grade
            if viz_eval_result is not None
            else getattr(viz_feedback, "grade", None)
        ),
        viz_eval_result=viz_eval_result
        or (viz_feedback if isinstance(viz_feedback, idd_state.VizFeedback) else None),
        visualization_results=(
            idd_state.VisualizationResults(**FLAGS, visualizations=figures)
            if figures is not None
            else None
        ),
        viz_results=[item.model_dump() for item in figures or []],
        viz_paths=[item.path for item in figures or []],
        _count_=0,
    )
    if prior_insights is not None:
        state["analysis_insights"] = prior_insights
        state["viz_specs"] = prior_insights.recommended_visualizations
        state["analyst_complete"] = prior_insights.finished_this_task
    return state


def analyst_figures(base):
    """Create exactly two valid PNGs with canonical figure metadata."""
    directory = base / "analyst-visualizations"
    directory.mkdir(parents=True, exist_ok=True)
    items = []
    for index, (title, viz_type) in enumerate(
        [
            ("Regional revenue by region", "bar"),
            ("Revenue versus order value", "scatter"),
        ]
    ):
        path = directory / f"analyst_fixture_{index}.png"
        image = Image.new("RGB", (240, 160), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 25 + index * 10, 180, 135), fill=(40, 90, 160))
        image.save(path, format="PNG")
        items.append(
            idd_state.DataVisualization(
                **FLAGS,
                path=str(path),
                visualization_id=f"analyst-figure-{index}",
                visualization_type=viz_type,
                visualization_description=f"Controlled valid {viz_type} PNG fixture.",
                visualization_style="whitegrid",
                visualization_title=title,
            )
        )
    return items


def worker_state(base):
    return dict(
        _config={"configurable": {"thread_id": "native-worker-fixture"}},
        user_prompt="Plot the controlled retail fixture",
        messages=[HumanMessage(content="Plot the controlled retail fixture")],
        available_df_ids=["retail-fixture"],
        individual_viz_task="Plot the controlled retail fixture",
        cleaning_metadata=cleaner_result(),
        initial_description=initial_description(),
        viz_specs=[],
        viz_results=[],
        viz_paths=[],
        viz_tasks=[],
        artifacts_path=str(base / "artifacts"),
        run_id="native-fixture",
        final_turn_msgs_list=[],
    )


def test_viz_worker_delivers_complete_spec_and_explicit_python_binding(
    tmp_path, monkeypatch
):
    from tests.unit.test_figure_provenance import plot_namespace, apply_join

    ns = plot_namespace(tmp_path, monkeypatch)
    spec = idd_state.VizSpec(
        **FLAGS,
        viz_id="net-distribution",
        title="Net distribution",
        viz_type="histogram",
        df_id="retail-fixture",
        columns=["net"],
        x="net",
        bins=3,
        y=None,
        hue=None,
        agg=None,
        query=None,
        description="Net distribution",
        limit=None,
        style="matplotlib",
        viz_instructions="Plot the distribution with the specified bins.",
    )
    path = ns["RUNTIME"].artifacts_dir / "native-distribution.png"
    code = (
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\nax.hist(df['net'], bins=3)\n"
        "ax.set(xlabel='Net revenue', ylabel='Orders', title='Net distribution')\n"
        f"fig.savefig({str(path)!r})\nplt.close(fig)\nprint({str(path)!r})"
    )
    item = idd_state.DataVisualization(
        **FLAGS,
        path=str(path),
        visualization_id=spec.viz_id,
        visualization_title=spec.title,
        visualization_type="histogram",
        visualization_style="matplotlib",
        visualization_description="Three bins of registered net revenue.",
    )
    model = ControlledChatModel(
        response_name="VisualizationResults",
        payload=idd_state.VisualizationResults(
            **FLAGS, visualizations=[item]
        ).model_dump(),
        required_context=[
            json.dumps(spec.model_dump(mode="json"), sort_keys=True),
            "Every Python call starts with a fresh namespace",
            "Pass df_id explicitly on each data call",
        ],
        scripted_responses=[
            {"name": "python_repl_tool", "args": {"code": code, "df_id": spec.df_id}}
        ],
    )
    ns["visualization_orchestrator_llm"] = model
    ns["visualization_agent"] = ns["create_visualization_agent"]([spec.df_id])
    state = worker_state(tmp_path)
    state.update(
        viz_spec=spec, viz_specs=[spec], individual_viz_task=spec.viz_instructions
    )
    update = ns["viz_worker"](state)
    joined = apply_join({**state, **update}, ns["viz_join"]({**state, **update}))
    assert joined["visualization_complete"] is True
    assert (
        joined["visualization_results"].visualizations[0].visualization_id
        == spec.viz_id
    )
    with Image.open(path) as image:
        image.verify()
    assert model._emitted == ["python_repl_tool", "VisualizationResults"]
    tool = next(
        t for t in model._bindings[0] if getattr(t, "name", None) == "python_repl_tool"
    )
    assert "fresh namespace" in tool.description
    assert (
        "df_id" in tool.description
        and "not Python variable aliases" in tool.description
    )


def test_viz_repeated_python_errors_stop_before_recursion(tmp_path):
    ns, _ = _repl_scope_fixture(tmp_path)
    model = ControlledChatModel(
        response_name="VisualizationResults",
        payload={},
        scripted_responses=[{"name": "python_repl_tool", "args": {"code": "print(df)"}}]
        * 90,
    )
    ns["visualization_orchestrator_llm"] = model
    ns["visualization_agent"] = ns["create_visualization_agent"](["scope_a"])
    with pytest.raises(
        ValueError, match="Visualization repeated unchanged Python errors"
    ):
        ns["viz_worker"](worker_state(tmp_path))
    assert len(model._requests) == 4
    assert "VisualizationResults" not in model._emitted


def test_viz_python_error_correction_and_new_assignment_share_native_thread(tmp_path):
    ns, _ = _repl_scope_fixture(tmp_path)
    result = idd_state.VisualizationResults(
        **FLAGS, visualizations=figures(tmp_path)
    ).model_dump()
    bad = {"name": "python_repl_tool", "args": {"code": "print(df)"}}
    good = {
        "name": "python_repl_tool",
        "args": {"code": "print(df['value'].sum())", "df_id": "scope_a"},
    }
    complete = {"name": "VisualizationResults", "args": result}
    model = ControlledChatModel(
        response_name="VisualizationResults",
        payload=result,
        scripted_responses=[bad, bad, bad, good, complete] * 2,
    )
    ns["visualization_orchestrator_llm"] = model
    ns["visualization_agent"] = ns["create_visualization_agent"](["scope_a"])
    state = worker_state(tmp_path)
    original = list(state["messages"])
    for assignment in ("First assignment", "Second assignment"):
        state["individual_viz_task"] = assignment
        assert ns["viz_worker"](state)["viz_results"]
    assert state["messages"] == original
    assert model._emitted == ["python_repl_tool"] * 4 + ["VisualizationResults"] + [
        "python_repl_tool"
    ] * 4 + ["VisualizationResults"]
    assert len(model._requests) == 10


def test_viz_successful_calculation_resets_error_streak(tmp_path):
    ns, _ = _repl_scope_fixture(tmp_path)
    result = idd_state.VisualizationResults(
        **FLAGS, visualizations=figures(tmp_path)
    ).model_dump()
    bad = {"name": "python_repl_tool", "args": {"code": "print(df)"}}
    good = {
        "name": "python_repl_tool",
        "args": {"code": "print(df['value'].sum())", "df_id": "scope_a"},
    }
    model = ControlledChatModel(
        response_name="VisualizationResults",
        payload=result,
        scripted_responses=[bad, bad, bad, good, bad, bad, bad, good],
    )
    ns["visualization_orchestrator_llm"] = model
    ns["visualization_agent"] = ns["create_visualization_agent"](["scope_a"])
    assert ns["viz_worker"](worker_state(tmp_path))["viz_results"]
    assert len(model._requests) == 9


@pytest.mark.parametrize("repeat", [False, True])
def test_viz_distinguishes_new_and_alternating_python_errors(tmp_path, repeat):
    ns, _ = _repl_scope_fixture(tmp_path)
    result = idd_state.VisualizationResults(
        **FLAGS, visualizations=figures(tmp_path)
    ).model_dump()
    codes = (
        ["print(missing_a)", "print(missing_b)"] * 4
        if repeat
        else [f"print(missing_{i})" for i in range(5)]
    )
    model = ControlledChatModel(
        response_name="VisualizationResults",
        payload=result,
        scripted_responses=[
            {"name": "python_repl_tool", "args": {"code": code}} for code in codes
        ],
    )
    ns["visualization_orchestrator_llm"] = model
    ns["visualization_agent"] = ns["create_visualization_agent"](["scope_a"])
    if repeat:
        with pytest.raises(
            ValueError, match="Visualization repeated unchanged Python errors"
        ):
            ns["viz_worker"](worker_state(tmp_path))
        assert len(model._requests) == 5
        assert "VisualizationResults" not in model._emitted
    else:
        assert ns["viz_worker"](worker_state(tmp_path))["viz_results"]
        assert len(model._requests) == 6


def evaluator_state(base):
    state = worker_state(base)
    items = figures(base)
    state.update(
        visualization_results=idd_state.VisualizationResults(
            **FLAGS, visualizations=items
        ),
        viz_results=[v.model_dump() for v in items],
        viz_tasks=[v.visualization_title for v in items],
    )
    return state


@pytest.mark.parametrize("flag", [None, "0"])
def test_evaluator_node_cannot_force_approve_a_strict_native_error(
    tmp_path, monkeypatch, flag
):
    ns = load_native(tmp_path)
    if flag is None:
        monkeypatch.delenv("IDD_ALLOW_RECOVERY", raising=False)
    error = GraphRecursionError("controlled evaluator exhaustion")

    def native(*a, **k):
        raise error

    ns["viz_evaluator_agent"] = SimpleNamespace(invoke=native)
    with pytest.raises(GraphRecursionError) as caught:
        ns["viz_evaluator_node"](evaluator_state(tmp_path))
    assert caught.value is error


@pytest.mark.parametrize("role,schema,llm,tag", WRAPPERS)
@pytest.mark.parametrize("flag", [None, "0", "false"])
@pytest.mark.parametrize("kind", ["recursion", "ordinary", "transient"])
def test_strict_native_error_preserves_exception_without_recovery(
    tmp_path,
    monkeypatch,
    role,
    schema,
    llm,
    tag,
    flag,
    kind,
):
    ns = load_native(tmp_path)
    if flag is None:
        monkeypatch.delenv("IDD_ALLOW_RECOVERY", raising=False)
    else:
        monkeypatch.setenv("IDD_ALLOW_RECOVERY", flag)
    error = (
        GraphRecursionError("native exhaustion")
        if kind == "recursion"
        else RuntimeError(
            "503 overloaded" if kind == "transient" else "native task error"
        )
    )
    attempts, forbidden = [], []

    def native(inputs, config):
        attempts.append(config)
        raise error

    def recovery(*a, **k):
        forbidden.append("final-hop")
        # Return a controlled object so baseline recovery cannot hide the defect.
        return SimpleNamespace(invoke=lambda *a, **k: object())

    def sleep(*a):
        forbidden.append("retry sleep")

    ns[llm] = SimpleNamespace(with_structured_output=recovery)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(Path, "rglob", lambda *a, **k: forbidden.append("scan"))
    with pytest.raises(type(error)) as caught:
        ns[f"_safe_{role}_invoke"](
            SimpleNamespace(invoke=native),
            {"messages": [HumanMessage(content="Task")]},
            config={"configurable": {"thread_id": "strict-error-fixture"}},
        )
    assert caught.value is error
    assert len(attempts) == 1
    assert attempts[0]["configurable"]["thread_id"] == "strict-error-fixture"
    assert attempts[0]["recursion_limit"] == 160
    assert forbidden == []


@pytest.mark.parametrize("role,schema,llm,tag", WRAPPERS)
def test_explicit_recovery_retains_finalhop_and_prohibited_marker(
    tmp_path,
    monkeypatch,
    capsys,
    role,
    schema,
    llm,
    tag,
):
    ns = load_native(tmp_path)
    monkeypatch.setenv("IDD_ALLOW_RECOVERY", "1")
    value, called = object(), []

    def native(*a, **k):
        raise GraphRecursionError("controlled legacy failure")

    def recover(model):
        called.append(model)
        return SimpleNamespace(invoke=lambda *a: value)

    ns[llm] = SimpleNamespace(with_structured_output=recover)
    result = ns[f"_safe_{role}_invoke"](
        SimpleNamespace(invoke=native), {"messages": []}
    )
    assert result["structured_response"] is value
    assert called == [getattr(idd_state, schema)]
    output = capsys.readouterr().out
    assert f"W2-BA-finalhop[{tag}] succeeded" in output
    from validate_run import RECOVERY_RE

    assert RECOVERY_RE.search(output)


@pytest.mark.parametrize("role,schema,llm,tag", WRAPPERS)
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupts_never_recover(tmp_path, role, schema, llm, tag, interrupt):
    ns = load_native(tmp_path)
    error = interrupt("controlled interruption")

    def native(*a, **k):
        raise error

    with pytest.raises(interrupt) as caught:
        ns[f"_safe_{role}_invoke"](SimpleNamespace(invoke=native), {"messages": []})
    assert caught.value is error


@pytest.mark.parametrize("role,schema,llm,tag", WRAPPERS)
def test_native_success_returns_original_result(tmp_path, role, schema, llm, tag):
    ns = load_native(tmp_path)
    value = {"messages": [], "structured_response": object()}
    assert (
        ns[f"_safe_{role}_invoke"](
            SimpleNamespace(invoke=lambda *a, **k: value),
            {"messages": []},
        )
        is value
    )


@pytest.mark.parametrize("role", ["data_cleaner", "visualization"])
def test_runtime_context_survives_wrapper_without_an_extra_system_message(
    tmp_path, role
):
    ns = load_native(tmp_path)
    observed = []

    def native(payload, config):
        observed.extend(payload["messages"])
        return {"messages": []}

    context = "Owned runtime context: df=retail-fixture; task=regional revenue; schema=canonical"
    ns[f"_safe_{role}_invoke"](
        SimpleNamespace(invoke=native),
        {
            "messages": [
                SystemMessage(content=context),
                HumanMessage(content="Finish"),
            ],
        },
    )
    assert any(context in str(m.content) for m in observed)
    assert isinstance(observed[0], HumanMessage)
    assert not any(isinstance(m, SystemMessage) for m in observed)


@pytest.mark.parametrize("role", ["data_cleaner", "visualization"])
def test_real_agent_factory_terminates_with_declared_response_tool(tmp_path, role):
    ns = load_native(tmp_path, middleware=True)
    payload = (
        cleaner_result()
        if role == "data_cleaner"
        else idd_state.VisualizationResults(
            **FLAGS,
            visualizations=figures(tmp_path),
        )
    )
    model = ControlledChatModel(
        response_name=type(payload).__name__, payload=payload.model_dump()
    )
    ns[
        (
            "data_cleaner_llm"
            if role == "data_cleaner"
            else "visualization_orchestrator_llm"
        )
    ] = model
    agent = (
        ns["create_data_cleaner_agent"](initial_description(), ["retail-fixture"])
        if role == "data_cleaner"
        else ns["create_visualization_agent"](["retail-fixture"])
    )
    result = ns[f"_safe_{role}_invoke"](
        agent,
        {
            "messages": [
                SystemMessage(content="Runtime fixture context"),
                HumanMessage(content="Finish"),
            ],
        },
        config={"configurable": {"thread_id": f"native-success-{role}"}},
    )
    assert result["structured_response"] == payload
    assert len(model._requests) == 1
    content = "\n".join(str(m.content) for m in model._requests[0])
    assert "Runtime fixture context" in content
    assert "{visualization_task}" not in content
    assert "{output_format}" not in content


def test_cleaner_node_uses_the_actual_native_response_tool(tmp_path):
    ns = load_native(tmp_path, middleware=True)
    model = ControlledChatModel(
        response_name="CleaningMetadata",
        payload=cleaner_result().model_dump(),
        follow_instructions=True,
    )
    ns["data_cleaner_llm"] = model
    ns["data_cleaner_agent"] = ns["create_data_cleaner_agent"](
        initial_description(), ["retail-fixture"]
    )
    # Observe the real node's native request without executing embedding storage.
    state = worker_state(tmp_path)
    state["cleaning_metadata"] = None
    result = ns["data_cleaner_node"](state)
    assert result["cleaning_metadata"] == cleaner_result()
    assert len(model._requests) == 1
    content = "\n".join(str(m.content) for m in model._requests[0])
    assert "CleaningMetadata" in content
    assert "call the `respond` tool" not in content


_CLEANING_MEMORY = ("memories", "cleaning")
_CLEANING_MEMORY_ID = "00000000-0000-0000-0000-000000000001"


def _cleaner_memory_fixture(tmp_path, *calls):
    ns = load_native(tmp_path, middleware=True)
    model = ControlledChatModel(
        response_name="CleaningMetadata",
        payload=cleaner_result().model_dump(),
        scripted_responses=list(calls),
    )
    ns["data_cleaner_llm"] = model
    ns["data_cleaner_agent"] = ns["create_data_cleaner_agent"](
        initial_description(), ["retail-fixture"]
    )
    state = worker_state(tmp_path)
    state["cleaning_metadata"] = None
    return ns, model, state


def _memory_call(**args):
    return {"name": "manage_memory", "args": args}


def _memory_errors(request):
    return [
        m
        for m in request
        if isinstance(m, ToolMessage)
        and m.name == "manage_memory"
        and m.status == "error"
    ]


@pytest.mark.parametrize(
    ("args", "guidance"),
    [
        ({"action": "update", "content": "Revised"}, "search_memory"),
        ({"action": "delete", "id": None}, "search_memory"),
        ({"action": "create", "id": _CLEANING_MEMORY_ID}, "Omit id for create"),
    ],
)
def test_cleaner_memory_id_rejection_is_paired_and_leaves_store_untouched(
    tmp_path, args, guidance
):
    ns, model, state = _cleaner_memory_fixture(tmp_path, _memory_call(**args))
    store = ns["in_memory_store"]
    store.put(_CLEANING_MEMORY, _CLEANING_MEMORY_ID, {"content": "Original"})

    result = ns["data_cleaner_node"](state)

    assert result["cleaning_metadata"] == cleaner_result()
    assert model._emitted == ["manage_memory", "CleaningMetadata"]
    (error,) = _memory_errors(model._requests[1])
    assert error.tool_call_id == "controlled-call-1"
    assert guidance in error.content
    assert [(m.key, m.value) for m in store.search(_CLEANING_MEMORY)] == [
        (_CLEANING_MEMORY_ID, {"content": "Original"})
    ]
    bound_tool = next(
        tool
        for tool in model._bindings[0]
        if getattr(tool, "name", None) == "manage_memory"
    )
    original = next(
        tool for tool in ns["data_cleaning_tools"] if tool.name == bound_tool.name
    )
    assert "search_memory" in bound_tool.description
    assert "Omit id for create" in bound_tool.description
    assert bound_tool.args_schema is original.args_schema
    assert bound_tool.description != original.description


def test_cleaner_memory_can_correct_id_and_complete(tmp_path):
    ns, model, state = _cleaner_memory_fixture(
        tmp_path,
        _memory_call(action="update", content="Revised"),
        _memory_call(action="update", content="Revised", id=_CLEANING_MEMORY_ID),
    )
    store = ns["in_memory_store"]
    store.put(_CLEANING_MEMORY, _CLEANING_MEMORY_ID, {"content": "Original"})

    result = ns["data_cleaner_node"](state)

    assert result["cleaning_metadata"] == cleaner_result()
    assert model._emitted == ["manage_memory", "manage_memory", "CleaningMetadata"]
    assert len(_memory_errors(model._requests[1])) == 1
    assert store.get(_CLEANING_MEMORY, _CLEANING_MEMORY_ID).value == {
        "content": "Revised"
    }


def test_cleaner_memory_parallel_rejections_count_as_one_round(tmp_path, monkeypatch):
    generate = ControlledChatModel._generate

    def generate_parallel(model, *args, **kwargs):
        result = generate(model, *args, **kwargs)
        if len(model._requests) == 1:
            result.generations[0].message.tool_calls.append(
                {
                    "name": "manage_memory",
                    "args": {"action": "update"},
                    "id": "parallel-call",
                    "type": "tool_call",
                }
            )
        return result

    monkeypatch.setattr(ControlledChatModel, "_generate", generate_parallel)
    ns, model, state = _cleaner_memory_fixture(tmp_path, _memory_call(action="delete"))

    result = ns["data_cleaner_node"](state)

    assert result["cleaning_metadata"] == cleaner_result()
    assert {m.tool_call_id for m in _memory_errors(model._requests[1])} == {
        "controlled-call-1",
        "parallel-call",
    }
    assert not ns["in_memory_store"].search(_CLEANING_MEMORY)


def test_cleaner_memory_repeated_contract_error_stops_assignment(tmp_path):
    ns, model, state = _cleaner_memory_fixture(
        tmp_path, _memory_call(action="delete"), _memory_call(action="update")
    )

    with pytest.raises(ValueError, match="manage_memory.*after.*correction"):
        ns["data_cleaner_node"](state)

    assert model._emitted == ["manage_memory", "manage_memory"]
    assert len(_memory_errors(model._requests[1])) == 1
    assert state["cleaning_metadata"] is None
    assert not ns["in_memory_store"].search(_CLEANING_MEMORY)


def test_cleaner_memory_correction_resets_for_next_assignment_on_same_thread(tmp_path):
    completion = {"name": "CleaningMetadata", "args": cleaner_result().model_dump()}
    ns, model, state = _cleaner_memory_fixture(
        tmp_path,
        _memory_call(action="delete"),
        completion,
        _memory_call(action="delete"),
    )

    first = ns["data_cleaner_node"](state)
    second_state = worker_state(tmp_path)
    second_state.update(cleaning_metadata=None, user_prompt="Clean a second assignment")
    assert state["_config"] == second_state["_config"]
    second = ns["data_cleaner_node"](second_state)

    assert first["cleaning_metadata"] == second["cleaning_metadata"] == cleaner_result()
    assert [m.tool_call_id for m in _memory_errors(model._requests[2])] == [
        "controlled-call-1"
    ]
    assert [m.tool_call_id for m in _memory_errors(model._requests[3])] == [
        "controlled-call-1",
        "controlled-call-3",
    ]


@pytest.mark.parametrize("action", ["create", "update", "delete"])
def test_cleaner_memory_valid_operations_use_real_store(tmp_path, action):
    args = {"content": "Revised"}
    if action != "create":
        args.update(action=action, id=_CLEANING_MEMORY_ID)
    ns, model, state = _cleaner_memory_fixture(tmp_path, _memory_call(**args))
    store = ns["in_memory_store"]
    if action != "create":
        store.put(_CLEANING_MEMORY, _CLEANING_MEMORY_ID, {"content": "Original"})

    result = ns["data_cleaner_node"](state)

    assert result["cleaning_metadata"] == cleaner_result()
    assert model._emitted == ["manage_memory", "CleaningMetadata"]
    assert not _memory_errors(model._requests[1])
    items = store.search(_CLEANING_MEMORY)
    assert [m.value for m in items] == (
        [] if action == "delete" else [{"content": "Revised"}]
    )
    if action == "update":
        assert items[0].key == _CLEANING_MEMORY_ID


def test_cleaner_memory_keeps_native_validation_for_malformed_uuid(tmp_path):
    ns, model, state = _cleaner_memory_fixture(
        tmp_path, _memory_call(action="create", id="malformed")
    )

    result = ns["data_cleaner_node"](state)

    assert result["cleaning_metadata"] == cleaner_result()
    (error,) = _memory_errors(model._requests[1])
    assert error.tool_call_id == "controlled-call-1"
    assert "uuid" in error.content.lower()
    assert not ns["in_memory_store"].search(_CLEANING_MEMORY)


def test_cleaner_memory_propagates_unexpected_store_exception(tmp_path, monkeypatch):
    ns, model, state = _cleaner_memory_fixture(
        tmp_path, _memory_call(content="Revised")
    )
    failure = ValueError("Unexpected store failure")

    def fail_put(*args, **kwargs):
        raise failure

    monkeypatch.setattr(type(ns["in_memory_store"]), "put", fail_put)
    with pytest.raises(ValueError, match="Unexpected store failure") as error:
        ns["data_cleaner_node"](state)

    assert error.value is failure
    assert model._emitted == ["manage_memory"]
    assert state["cleaning_metadata"] is None


class _HistorySummarySpy:
    def __init__(self, response="Older observations", failure=None):
        self.requests = []
        self.response = response
        self.failure = failure

    def bind(self, *, tools, tool_choice, max_tokens):
        assert tools == [] and tool_choice == "none" and max_tokens == 32
        return self

    def invoke(self, messages):
        self.requests.append(messages)
        if self.failure:
            raise self.failure
        return AIMessage(content=self.response)


def _hosted_cleaner_history(tmp_path, summary, *, budget=300, calls=()):
    ns = load_native(tmp_path, middleware=True)
    ns.update(MAX_CONTEXT=160, CONTEXT_HEADROOM=20)
    ns["prehook_quick"] = ns["make_hosted_history_middleware"](
        summary, model_budget_tokens=budget, max_summary_tokens=32
    )
    model = ControlledChatModel(
        response_name="CleaningMetadata",
        payload=cleaner_result().model_dump(),
        scripted_responses=list(calls),
    )
    ns["data_cleaner_llm"] = model
    agent = ns["create_data_cleaner_agent"](initial_description(), ["retail-fixture"])
    return ns, model, agent


def _history_exchange(prefix, *results):
    ids = [f"{prefix}-{i}" for i in range(len(results))]
    return [
        AIMessage(
            id=prefix,
            content="",
            tool_calls=[
                {"name": "search_memory", "args": {"query": "retail"}, "id": call_id}
                for call_id in ids
            ],
        ),
        *[
            ToolMessage(id=call_id, tool_call_id=call_id, content=content)
            for call_id, content in zip(ids, results)
        ],
    ]


def test_hosted_history_under_budget_passes_messages_without_summarizing(tmp_path):
    summary = _HistorySummarySpy()
    _, model, agent = _hosted_cleaner_history(tmp_path, summary)
    messages = [
        HumanMessage(id="user", content="Clean this data"),
        *_history_exchange("parallel", "First result", "Second result"),
    ]
    result = agent.invoke(
        {"messages": messages}, config={"configurable": {"thread_id": "under-budget"}}
    )

    assert result["structured_response"] == cleaner_result()
    system_prompt, *effective = model._requests[0]
    assert isinstance(system_prompt, SystemMessage)
    assert effective == messages
    assert summary.requests == []


def test_hosted_history_default_cleaner_hook_trims_before_model_call(tmp_path):
    ns, model, _ = _cleaner_memory_fixture(tmp_path)
    agent = ns["data_cleaner_agent"]
    messages = [
        AIMessage(id="stale", content="Earlier observation " * 40_000),
        HumanMessage(id="current", content="Clean this data"),
    ]
    assert ns["count_tokens_approximately"](messages) > 180_000
    result = agent.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": "default-budget"}},
    )

    assert result["structured_response"] == cleaner_result()
    assert all(m.id != "stale" for m in model._requests[0])
    assert any(m.name == "history_summary" for m in model._requests[0])


def test_hosted_history_reduces_requests_and_keeps_checkpointed_parallel_pairs(
    tmp_path,
):
    summary = _HistorySummarySpy()
    ns, model, agent = _hosted_cleaner_history(tmp_path, summary)
    config = {"configurable": {"thread_id": "reused-history"}}
    first = HumanMessage(id="old-user", content="Earlier assignment")
    agent.invoke({"messages": [first]}, config=config)
    messages = [
        *_history_exchange("old-tool", "TOOL PAYLOAD " * 300, "second old result"),
        AIMessage(
            id="old-observations",
            content=[
                {"type": "reasoning", "text": "PRIVATE REASONING BLOCK"},
                {"type": "text", "text": "Earlier observation " * 160},
            ],
        ),
        HumanMessage(
            id="current-context", name="runtime_context", content="Clean context"
        ),
        HumanMessage(id="current-user", content="Current user request"),
        AIMessage(
            id="current-supervisor", name="supervisor", content="Current assignment"
        ),
        *_history_exchange("latest-tool", "Latest first", "Latest second"),
    ]
    result = agent.invoke({"messages": messages}, config=config)

    assert result["structured_response"] == cleaner_result()
    _, *effective = model._requests[1]
    assert ns["count_tokens_approximately"](effective) <= 300
    assert effective[0].name == "history_summary"
    assert effective[1].id == messages[4].id
    assert str(effective[1].content).startswith(
        "Clean context\n\nObserved cleaner progress:\n"
    )
    assert messages[4].content == "Clean context"
    assert effective[2:] == messages[5:]
    assert len(summary.requests) == 1
    prompt = summary.requests[0]
    assert ns["count_tokens_approximately"](prompt) <= 160 - 20 - 32
    assert "Earlier observation" in prompt[-1].content
    assert "TOOL PAYLOAD" not in prompt[-1].content
    assert "PRIVATE REASONING BLOCK" not in prompt[-1].content
    checkpoint = agent.get_state(config).values
    assert first in checkpoint["messages"]
    assert all(message in checkpoint["messages"] for message in messages)
    assert all(m.name != "history_summary" for m in checkpoint["messages"])
    assert "llm_input_messages" not in checkpoint


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        (
            _history_exchange("latest", "protected result " * 200),
            "protected tool exchange",
        ),
        (
            _history_exchange("missing", "first result " * 200, "missing result")[:-1],
            "incomplete tool exchange",
        ),
        ([ToolMessage(tool_call_id="orphan", content="orphan " * 200)], "orphaned"),
    ],
)
def test_hosted_history_rejects_unsafe_pressured_requests(tmp_path, messages, error):
    summary = _HistorySummarySpy()
    _, model, agent = _hosted_cleaner_history(tmp_path, summary)
    with pytest.raises(ValueError, match=error):
        agent.invoke(
            {"messages": messages}, config={"configurable": {"thread_id": "unsafe"}}
        )

    assert summary.requests == []
    assert model._requests == []


def test_hosted_history_omits_an_oversized_summary(tmp_path):
    summary = _HistorySummarySpy(response="Oversized summary " * 200)
    _, model, agent = _hosted_cleaner_history(tmp_path, summary)
    latest = HumanMessage(content="Current assignment")
    result = agent.invoke(
        {"messages": [AIMessage(content="Earlier observation " * 200), latest]},
        config={"configurable": {"thread_id": "oversized-summary"}},
    )

    assert result["structured_response"] == cleaner_result()
    assert len(summary.requests) == 1
    assert model._requests[0][1:] == [latest]


def test_hosted_history_propagates_summary_failure_before_model_call(tmp_path):
    failure = RuntimeError("Scripted summary failure")
    summary = _HistorySummarySpy(failure=failure)
    _, model, agent = _hosted_cleaner_history(tmp_path, summary)
    with pytest.raises(RuntimeError, match="Scripted summary failure") as caught:
        agent.invoke(
            {
                "messages": [
                    AIMessage(content="Earlier observation " * 200),
                    HumanMessage(content="Clean"),
                ]
            },
            config={"configurable": {"thread_id": "summary-error"}},
        )

    assert caught.value is failure
    assert model._requests == []


def test_hosted_history_keeps_the_cleaner_id_correction_under_pressure(tmp_path):
    summary = _HistorySummarySpy()
    _, model, agent = _hosted_cleaner_history(
        tmp_path,
        summary,
        calls=(
            _memory_call(action="update", content="Revised"),
            _memory_call(action="update", content="Revised", id=_CLEANING_MEMORY_ID),
        ),
    )
    result = agent.invoke(
        {
            "messages": [
                AIMessage(id="stale", content="Earlier observation " * 200),
                HumanMessage(
                    name="runtime_context", content="Clean with current memory"
                ),
            ]
        },
        config={"configurable": {"thread_id": "correction-with-trim"}},
    )

    assert result["structured_response"] == cleaner_result()
    assert model._emitted == ["manage_memory", "manage_memory", "CleaningMetadata"]
    assert [m.tool_call_id for m in _memory_errors(model._requests[1])] == [
        "controlled-call-1"
    ]
    assert all(not any(m.id == "stale" for m in request) for request in model._requests)


@pytest.mark.parametrize("role", ["data_cleaner", "visualization"])
def test_real_agent_nonterminating_tool_fails_strictly(tmp_path, role):
    ns = load_native(tmp_path, middleware=True)
    model = ControlledChatModel(response_name="respond", payload={})
    ns[
        (
            "data_cleaner_llm"
            if role == "data_cleaner"
            else "visualization_orchestrator_llm"
        )
    ] = model
    agent = (
        ns["create_data_cleaner_agent"](initial_description(), [])
        if role == "data_cleaner"
        else ns["create_visualization_agent"]([])
    )
    with pytest.raises(GraphRecursionError):
        ns[f"_safe_{role}_invoke"](
            agent,
            {"messages": [HumanMessage(content="Finish")]},
            config={"configurable": {"thread_id": f"native-loop-{role}"}},
        )
    assert 1 < len(model._requests) < 160


@pytest.mark.parametrize("shape", ["model", "dict", "single"])
def test_native_worker_publishes_figure_deltas_without_replaying_history(
    tmp_path, shape
):
    ns = load_native(tmp_path, middleware=True)
    items = figures(tmp_path)
    envelope = idd_state.VisualizationResults(**FLAGS, visualizations=items[1:])
    state = worker_state(tmp_path)
    state["viz_results"] = [items[0].model_dump()]
    state["viz_paths"] = [items[0].path]
    if shape == "model":
        model = ControlledChatModel(
            response_name="VisualizationResults", payload=envelope.model_dump()
        )
        ns["visualization_orchestrator_llm"] = model
        ns["visualization_agent"] = ns["create_visualization_agent"](["retail-fixture"])
    else:
        payload = envelope.model_dump() if shape == "dict" else items[1]
        ns["visualization_agent"] = SimpleNamespace(
            invoke=lambda *a, **k: {
                "messages": [AIMessage(content="Native fixture complete")],
                "structured_response": payload,
            }
        )
    update = ns["viz_worker"](state)
    expected = items[1:] if shape != "single" else [items[1]]
    assert [
        idd_state.DataVisualization.model_validate(v) for v in update["viz_results"]
    ] == expected
    assert update["viz_paths"] == [v.path for v in expected]
    assert "structured_response" not in update
    assert state["viz_results"] == [items[0].model_dump()]
    assert all(v.visualization_id != items[0].visualization_id for v in expected)


@pytest.mark.parametrize("damage", ["empty", "malformed", "unfinished", "missing"])
def test_native_worker_rejects_invalid_required_completion(tmp_path, damage):
    ns = load_native(tmp_path)
    result = idd_state.VisualizationResults(
        **FLAGS, visualizations=figures(tmp_path)
    ).model_dump()
    if damage == "empty":
        result["visualizations"] = []
    elif damage == "malformed":
        del result["visualizations"][0]["visualization_title"]
    elif damage == "unfinished":
        result["finished_this_task"] = False
    else:
        result = None
    ns["visualization_agent"] = SimpleNamespace(
        invoke=lambda *a, **k: {
            "messages": [AIMessage(content="Incomplete fixture")],
            "structured_response": result,
        }
    )
    with pytest.raises((ValueError, ValidationError)):
        ns["viz_worker"](worker_state(tmp_path))


def _repl_scope_fixture(base):
    import pandas as pd

    ns = load_native(base, middleware=True)
    registry = DataFrameRegistry(data_dir=base / "scope-registry")
    registry.register_dataframe(pd.DataFrame({"value": [10.0, 20.0]}), "scope_a")
    registry.register_dataframe(pd.DataFrame({"value": [100.0, 200.0]}), "scope_b")
    ns.update(get_global_df_registry=lambda: registry, global_df_registry=registry)
    ns["ambient_frame"] = pd.DataFrame({"unrelated": [999.0]})
    return ns, registry


def _invoke_scope(ns, code, df_id="scope_a", call_id="scope-call"):
    return ns["python_repl_tool"].invoke(
        {
            "type": "tool_call",
            "id": call_id,
            "name": "python_repl_tool",
            "args": {"code": code, "df_id": df_id},
        }
    )


@pytest.mark.parametrize(
    "code,expected",
    [
        ("base = 5\ndef compute():\n    return base + 1\nprint(compute())", "6"),
        (
            "frame = df.copy()\nprint(frame['value'].apply(lambda value: value + frame['value'].iloc[0]).tolist())",
            "[20.0, 30.0]",
        ),
        ("offset = 3\nprint(sum(value + offset for value in [1, 2]))", "9"),
    ],
)
def test_repl_same_call_nested_scopes_resolve_assignments(tmp_path, code, expected):
    ns, _ = _repl_scope_fixture(tmp_path)
    result = _invoke_scope(ns, code)
    assert str(result.content).splitlines()[0] == expected
    assert "NameError" not in str(result.content)


def test_repl_fresh_calls_do_not_retain_variables(tmp_path):
    ns, _ = _repl_scope_fixture(tmp_path)
    assert "41" in str(
        _invoke_scope(ns, "call_private = 41\nprint(call_private)").content
    )
    result = _invoke_scope(ns, "print(call_private)")
    assert "NameError" in str(result.content)


def test_repl_registers_only_authored_dataframe_bindings(tmp_path):
    ns, registry = _repl_scope_fixture(tmp_path)
    result = _invoke_scope(
        ns, "frame = df[['value']].copy()\nprint(frame['value'].sum())"
    )
    assert result.artifact["newly_registered_dfs"] == ["frame"]
    assert registry.get_dataframe("frame")["value"].tolist() == [10.0, 20.0]
    assert registry.get_dataframe("ambient_frame") is None


def test_repl_dataframe_rebinding_persists_under_input_id(tmp_path):
    ns, registry = _repl_scope_fixture(tmp_path)
    result = _invoke_scope(
        ns, "df = df.assign(value=df['value'] + 7)\nprint(df['value'].sum())"
    )
    assert "44.0" in str(result.content)
    assert registry.get_dataframe("scope_a")["value"].tolist() == [17.0, 27.0]
    assert not result.artifact.get("newly_registered_dfs")


def test_repl_real_code_errors_remain_errors(tmp_path):
    ns, registry = _repl_scope_fixture(tmp_path)
    result = _invoke_scope(ns, "raise RuntimeError('controlled-repl-failure')")
    assert "RuntimeError" in str(result.content) and "controlled-repl-failure" in str(
        result.content
    )
    assert registry.get_dataframe("scope_a")["value"].tolist() == [10.0, 20.0]


def test_repl_sandbox_still_rejects_outside_writes(tmp_path):
    ns, _ = _repl_scope_fixture(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
    result = _invoke_scope(
        ns, f"open({str(outside.as_posix())!r}, 'w').write('forbidden')"
    )
    assert not outside.exists()
    assert "Error" in str(result.content) or "outside" in str(result.content).lower()


def test_repl_parallel_calls_cannot_cross_dataframe_or_stdout(tmp_path, monkeypatch):
    ns, registry = _repl_scope_fixture(tmp_path)
    original_stdout = sys.stdout
    first_inside, release_first, second_started, second_inside = (
        Event(),
        Event(),
        Event(),
        Event(),
    )
    cls = ns["PythonAstREPLTool"]
    original = cls.invoke

    def intercept(self, value, *args, **kwargs):
        if "FIRST_SCOPE" in value["query"]:
            first_inside.set()
            assert release_first.wait(8)
        else:
            second_inside.set()
        return original(self, value, *args, **kwargs)

    monkeypatch.setattr(cls, "invoke", intercept)

    def second_call():
        second_started.set()
        return _invoke_scope(
            ns, "print('SECOND_SCOPE', float(df['value'].iloc[0]))", "scope_b", "second"
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            _invoke_scope,
            ns,
            "print('FIRST_SCOPE', float(df['value'].iloc[0]))",
            "scope_a",
            "first",
        )
        assert first_inside.wait(8)
        second = executor.submit(second_call)
        try:
            assert second_started.wait(8)
            overlapped = second_inside.wait(0.5)
        finally:
            release_first.set()
        first_result = first.result(timeout=8)
        second_result = second.result(timeout=8)
    assert not overlapped, "REPL namespace and stdout capture must execute atomically"
    assert sys.stdout is original_stdout
    assert "FIRST_SCOPE 10.0" in str(first_result.content)
    assert "SECOND_SCOPE" not in str(first_result.content)
    assert "SECOND_SCOPE 100.0" in str(second_result.content)
    assert registry.get_dataframe("scope_a")["value"].tolist() == [10.0, 20.0]
    assert registry.get_dataframe("scope_b")["value"].tolist() == [100.0, 200.0]


def _grounded_cleaner_fixture(base, *calls, payload=None):
    import pandas as pd

    ns, model, state = _cleaner_memory_fixture(base, *calls)
    registry = DataFrameRegistry(data_dir=base / "cleaner-registry")
    registry.register_dataframe(
        pd.DataFrame(
            {"value": [10.0, 20.0, 20.0, None], "group": ["a", "b", "b", "c"]}
        ),
        "retail-fixture",
    )
    state.update(
        user_prompt="ORIGINAL_REQUEST: preserve useful observations and explain caveats",
        next_agent_prompt="CURRENT_ASSIGNMENT: handle missing values and exact duplicates",
    )
    if payload is not None:
        model.payload = payload.model_dump()
    return ns, model, state, registry


def _cleaner_context(request):
    return "\n".join(str(m.content) for m in request if isinstance(m, HumanMessage))


def test_cleaner_current_assignment_and_observed_quality_reach_model(tmp_path):
    ns, model, state, registry = _grounded_cleaner_fixture(tmp_path)
    with override_global_registry(registry):
        ns["data_cleaner_node"](state)
    context = _cleaner_context(model._requests[0])
    assert "ORIGINAL_REQUEST" in context and "CURRENT_ASSIGNMENT" in context
    assert '"duplicate_rows": 1' in context and '"value": 1' in context
    assert "never claim an action" in context.lower()
    assert "once as CSV using export_dataframe" in context
    assert "clip outliers" not in context.lower()
    assert "best-effort values" not in context


def test_cleaner_progress_reflects_real_mutation_and_export_is_not_cleaning(tmp_path):
    ns, model, state, registry = _grounded_cleaner_fixture(
        tmp_path,
        {
            "name": "fill_missing_median",
            "args": {"df_id": "retail-fixture", "column_name": "value"},
        },
        {"name": "detect_and_remove_duplicates", "args": {"df_id": "retail-fixture"}},
    )
    with override_global_registry(registry):
        result = ns["data_cleaner_node"](state)
    assert result["data_cleaning_complete"] is True
    context = _cleaner_context(model._requests[-1])
    assert '"duplicate_rows": 0' in context and '"value": 0' in context
    assert (
        "fill_missing_median" in context and "detect_and_remove_duplicates" in context
    )
    assert "Exporting a CSV does not clean" in context
    assert len(registry.get_dataframe("retail-fixture")) == 3
    assert all(not m.name == "cleaner_progress" for m in result["messages"])
    assert all(
        "Observed cleaner progress:" not in str(m.content)
        for m in result["messages"]
        if isinstance(m, HumanMessage)
    )


@pytest.mark.parametrize("existing", [False, True])
def test_cleaner_unfinished_native_output_never_publishes_completion(
    tmp_path, existing
):
    incomplete = cleaner_result().model_copy(update={"finished_this_task": False})
    ns, model, state, registry = _grounded_cleaner_fixture(tmp_path, payload=incomplete)
    if existing:
        state.update(cleaning_metadata=incomplete, data_cleaning_complete=True)
    with override_global_registry(registry), pytest.raises(
        ValueError, match="Cleaner.*unfinished"
    ):
        ns["data_cleaner_node"](state)
    assert (
        state.get("cleaning_metadata") is incomplete
        if existing
        else state.get("cleaning_metadata") is None
    )


def test_cleaner_redundant_schema_loop_fails_before_recursion(tmp_path):
    call = {"name": "get_dataframe_schema", "args": {"df_id": "retail-fixture"}}
    ns, model, state, registry = _grounded_cleaner_fixture(tmp_path, *([call] * 8))
    with override_global_registry(registry), pytest.raises(
        ValueError, match="Cleaner.*no progress"
    ):
        ns["data_cleaner_node"](state)
    assert model._emitted == ["get_dataframe_schema"] * 4
    assert state["cleaning_metadata"] is None


def test_cleaner_memory_only_loop_does_not_count_as_cleaning_progress(tmp_path):
    ns, model, state, registry = _grounded_cleaner_fixture(
        tmp_path, *[_memory_call(content=f"Different memory {i}") for i in range(8)]
    )
    with override_global_registry(registry), pytest.raises(
        ValueError, match="Cleaner.*memory-only"
    ):
        ns["data_cleaner_node"](state)
    assert len(model._emitted) == 3
    assert state["cleaning_metadata"] is None


def test_cleaner_changed_quality_results_reset_retrieval_streak(tmp_path):
    profile = {"name": "check_missing_values", "args": {"df_id": "retail-fixture"}}
    mutation = {
        "name": "fill_missing_median",
        "args": {"df_id": "retail-fixture", "column_name": "value"},
    }
    ns, model, state, registry = _grounded_cleaner_fixture(
        tmp_path, *([profile] * 3), mutation, *([profile] * 3)
    )
    with override_global_registry(registry):
        result = ns["data_cleaner_node"](state)
    assert result["data_cleaning_complete"] is True
    assert len(model._requests) == 8


def test_cleaner_progress_bound_resets_for_new_assignment_on_same_thread(tmp_path):
    profile = {"name": "get_dataframe_schema", "args": {"df_id": "retail-fixture"}}
    completion = {"name": "CleaningMetadata", "args": cleaner_result().model_dump()}
    ns, model, state, registry = _grounded_cleaner_fixture(
        tmp_path, *([profile] * 4), profile, completion
    )
    with override_global_registry(registry):
        with pytest.raises(ValueError, match="Cleaner.*no progress"):
            ns["data_cleaner_node"](state)
        next_state = dict(
            state, user_prompt="A fresh assignment", cleaning_metadata=None
        )
        result = ns["data_cleaner_node"](next_state)
    assert result["data_cleaning_complete"] is True
    assert model._emitted[-2:] == ["get_dataframe_schema", "CleaningMetadata"]


def test_cleaner_export_only_keeps_dirty_quality_visible(tmp_path):
    ns, model, state, registry = _grounded_cleaner_fixture(
        tmp_path,
        {
            "name": "export_dataframe",
            "args": {
                "df_id": "retail-fixture",
                "file_name": "still-dirty",
                "file_format": "csv",
            },
        },
    )
    with override_global_registry(registry):
        ns["data_cleaner_node"](state)
    result = next(
        m
        for m in model._requests[1]
        if isinstance(m, ToolMessage) and m.name == "export_dataframe"
    )
    assert not str(result.content).startswith(("Error", "Failed"))
    assert list(tmp_path.rglob("still-dirty.csv"))
    context = _cleaner_context(model._requests[1])
    assert '"duplicate_rows": 1' in context and '"value": 1' in context
    assert "Exporting a CSV does not clean" in context


def test_cleaner_structured_tool_error_cannot_reset_progress_bound(tmp_path):
    profile = {"name": "check_missing_values", "args": {"df_id": "retail-fixture"}}
    invalid = {
        "name": "fill_missing_median",
        "args": {"df_id": "retail-fixture", "column_name": "not_a_column"},
    }
    ns, model, state, registry = _grounded_cleaner_fixture(
        tmp_path, *([profile] * 3), invalid, profile
    )
    with override_global_registry(registry), pytest.raises(
        ValueError, match="Cleaner.*no progress"
    ):
        ns["data_cleaner_node"](state)
    assert len(model._emitted) == 5
    errors = [
        m
        for m in model._requests[-1]
        if isinstance(m, ToolMessage) and m.name == "fill_missing_median"
    ]
    assert json.loads(errors[-1].content)["status"] == "error"


_VIZ_MEMORY = ("memories", "visualization")


def _viz_memory_fixture(base, *calls):
    ns = load_native(base, middleware=True)
    payload = idd_state.VisualizationResults(**FLAGS, visualizations=figures(base)[:1])
    model = ControlledChatModel(
        response_name="VisualizationResults",
        payload=payload.model_dump(),
        scripted_responses=list(calls),
    )
    ns["visualization_orchestrator_llm"] = model
    ns["visualization_agent"] = ns["create_visualization_agent"](["retail-fixture"])
    state = worker_state(base)
    return ns, model, state


@pytest.mark.parametrize(
    "args",
    [
        {"action": "update", "content": "Revised"},
        {"action": "delete"},
        {"action": "create", "id": _CLEANING_MEMORY_ID, "content": "New"},
    ],
)
def test_viz_memory_contract_rejection_is_paired_and_does_not_mutate_store(
    tmp_path, args
):
    ns, model, state = _viz_memory_fixture(tmp_path, _memory_call(**args))
    store = ns["in_memory_store"]
    store.put(_VIZ_MEMORY, _CLEANING_MEMORY_ID, {"content": "Original"})
    update = ns["viz_worker"](state)
    assert len(update["viz_results"]) == 1
    assert model._emitted == ["manage_memory", "VisualizationResults"]
    (error,) = _memory_errors(model._requests[1])
    assert error.tool_call_id == "controlled-call-1"
    assert "Omit id for create" in error.content and "search_memory" in error.content
    assert store.get(_VIZ_MEMORY, _CLEANING_MEMORY_ID).value == {"content": "Original"}
    assert len(store.search(_VIZ_MEMORY)) == 1


def test_viz_memory_binding_preserves_native_enum_uuid_and_flat_arguments(tmp_path):
    ns, model, state = _viz_memory_fixture(tmp_path)
    ns["viz_worker"](state)
    original = ns["_viz_mm_raw"]
    catalog = next(t for t in ns["visualization_tools"] if t.name == "manage_memory")
    bound = next(
        t for t in model._bindings[0] if getattr(t, "name", None) == "manage_memory"
    )
    assert catalog is original
    assert bound.args_schema is original.args_schema
    schema = bound.args_schema.model_json_schema()["properties"]
    assert set(schema) == {"content", "action", "id"}
    assert schema["action"]["default"] == "create"
    assert set(schema["action"]["enum"]) == {"create", "update", "delete"}
    assert any(x.get("format") == "uuid" for x in schema["id"]["anyOf"])
    assert "Omit id for create" in bound.description
    assert catalog.description != bound.description


def test_viz_memory_corrected_update_completes_natively(tmp_path):
    ns, model, state = _viz_memory_fixture(
        tmp_path,
        _memory_call(action="update", content="Revised"),
        _memory_call(action="update", content="Revised", id=_CLEANING_MEMORY_ID),
    )
    store = ns["in_memory_store"]
    store.put(_VIZ_MEMORY, _CLEANING_MEMORY_ID, {"content": "Original"})
    assert len(ns["viz_worker"](state)["viz_results"]) == 1
    assert store.get(_VIZ_MEMORY, _CLEANING_MEMORY_ID).value == {"content": "Revised"}
    assert model._emitted == ["manage_memory", "manage_memory", "VisualizationResults"]


def test_viz_memory_repeated_contract_violation_fails_closed(tmp_path):
    ns, model, state = _viz_memory_fixture(
        tmp_path, _memory_call(action="delete"), _memory_call(action="update")
    )
    with pytest.raises(
        ValueError, match="Visualization manage_memory.*after.*correction"
    ):
        ns["viz_worker"](state)
    assert model._emitted == ["manage_memory", "manage_memory"]
    assert state["viz_results"] == []
    assert not ns["in_memory_store"].search(_VIZ_MEMORY)


@pytest.mark.parametrize("action", ["create", "update", "delete"])
def test_viz_memory_valid_calls_delegate_to_real_langmem(tmp_path, action):
    args = {"content": "Revised"}
    if action != "create":
        args.update(action=action, id=_CLEANING_MEMORY_ID)
    ns, model, state = _viz_memory_fixture(tmp_path, _memory_call(**args))
    store = ns["in_memory_store"]
    if action != "create":
        store.put(_VIZ_MEMORY, _CLEANING_MEMORY_ID, {"content": "Original"})
    assert len(ns["viz_worker"](state)["viz_results"]) == 1
    assert not _memory_errors(model._requests[1])
    assert [x.value for x in store.search(_VIZ_MEMORY)] == (
        [] if action == "delete" else [{"content": "Revised"}]
    )


@pytest.mark.parametrize(
    "args,term",
    [
        ({"action": "create", "id": "malformed", "content": "New"}, "uuid"),
        ({"action": "remember", "content": "New"}, "input should be"),
    ],
)
def test_viz_memory_other_malformed_arguments_keep_native_validation(
    tmp_path, args, term
):
    ns, model, state = _viz_memory_fixture(tmp_path, _memory_call(**args))
    ns["viz_worker"](state)
    (error,) = _memory_errors(model._requests[1])
    assert term in error.content.lower()
    assert error.artifact != "idd_visualization_memory_id_contract"
    assert not ns["in_memory_store"].search(_VIZ_MEMORY)


def test_viz_memory_unexpected_store_failure_propagates(tmp_path, monkeypatch):
    ns, model, state = _viz_memory_fixture(tmp_path, _memory_call(content="New"))
    failure = RuntimeError("Unexpected visualization store failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(type(ns["in_memory_store"]), "put", fail)
    with pytest.raises(RuntimeError) as caught:
        ns["viz_worker"](state)
    assert caught.value is failure
    assert model._emitted == ["manage_memory"]
    assert state["viz_results"] == []


def test_viz_memory_parallel_rejections_are_one_correction_round(tmp_path, monkeypatch):
    generate = ControlledChatModel._generate

    def parallel(model, *args, **kwargs):
        result = generate(model, *args, **kwargs)
        if len(model._requests) == 1:
            result.generations[0].message.tool_calls.append(
                {
                    "name": "manage_memory",
                    "args": {"action": "update"},
                    "id": "parallel-memory",
                    "type": "tool_call",
                }
            )
        return result

    monkeypatch.setattr(ControlledChatModel, "_generate", parallel)
    ns, model, state = _viz_memory_fixture(tmp_path, _memory_call(action="delete"))
    assert len(ns["viz_worker"](state)["viz_results"]) == 1
    assert {m.tool_call_id for m in _memory_errors(model._requests[1])} == {
        "controlled-call-1",
        "parallel-memory",
    }
    assert not ns["in_memory_store"].search(_VIZ_MEMORY)


def test_viz_memory_correction_resets_for_new_assignment_on_same_thread(tmp_path):
    ns, model, state = _viz_memory_fixture(tmp_path)
    completion = {"name": "VisualizationResults", "args": model.payload}
    model.scripted_responses = [
        _memory_call(action="delete"),
        completion,
        _memory_call(action="update"),
    ]
    config = {"configurable": {"thread_id": "viz-memory-reused-thread"}}
    for assignment in ["first", "second"]:
        result = ns["_safe_visualization_invoke"](
            ns["visualization_agent"],
            {
                "messages": [
                    HumanMessage(
                        name="runtime_context", content=assignment, id=assignment
                    )
                ]
            },
            config=config,
        )
        assert isinstance(result["structured_response"], idd_state.VisualizationResults)
    assert model._emitted == [
        "manage_memory",
        "VisualizationResults",
        "manage_memory",
        "VisualizationResults",
    ]
    assert [m.tool_call_id for m in _memory_errors(model._requests[-1])] == [
        "controlled-call-1",
        "controlled-call-3",
    ]


class ParallelCleanerModel(ControlledChatModel):
    parallel_calls: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not self._requests:
            self._requests.append(messages)
            self._emitted.extend(call["name"] for call in self.parallel_calls)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content="", tool_calls=self.parallel_calls)
                    )
                ]
            )
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.mark.parametrize("second", ["impute", "convert"])
def test_cleaner_parallel_mutations_preserve_both_updates(
    tmp_path, monkeypatch, second
):
    from threading import Barrier, BrokenBarrierError
    import pandas as pd

    ns = load_native(tmp_path, middleware=True)
    registry = DataFrameRegistry(data_dir=tmp_path / "parallel-registry")
    registry.register_dataframe(
        pd.DataFrame(
            {
                "discount_rate": [0.1, None, 0.3],
                "satisfaction_score": [7.0, None, 9.0],
                "order_date": [
                    "2025-01-01 00:00:00",
                    "2025-01-02 00:00:00",
                    "2025-01-03 00:00:00",
                ],
            }
        ),
        "retail-fixture",
    )
    original = registry.get_dataframe("retail-fixture")
    rendezvous = Barrier(2, timeout=1)
    real_copy = pd.DataFrame.copy

    def overlapping_copy(frame, *args, **kwargs):
        copied = real_copy(frame, *args, **kwargs)
        if frame is original:
            # Force two baseline reads to overlap. A serialized invocation times
            # out alone, then the next invocation reads the already updated frame.
            try:
                rendezvous.wait()
            except BrokenBarrierError:
                pass
        return copied

    monkeypatch.setattr(pd.DataFrame, "copy", overlapping_copy)
    calls = [
        {
            "name": "fill_missing_median",
            "args": {"df_id": "retail-fixture", "column_name": "discount_rate"},
            "id": "parallel-discount",
        },
        {
            "name": (
                "fill_missing_median" if second == "impute" else "convert_data_types"
            ),
            "args": (
                {"df_id": "retail-fixture", "column_name": "satisfaction_score"}
                if second == "impute"
                else {
                    "df_id": "retail-fixture",
                    "column_types": {"order_date": "datetime64[ns]"},
                }
            ),
            "id": "parallel-second",
        },
    ]
    model = ParallelCleanerModel(
        response_name="CleaningMetadata",
        payload=cleaner_result().model_dump(),
        parallel_calls=calls,
    )
    ns["data_cleaner_llm"] = model
    ns["data_cleaner_agent"] = ns["create_data_cleaner_agent"](
        initial_description(), ["retail-fixture"]
    )
    state = worker_state(tmp_path)
    state["cleaning_metadata"] = None
    with override_global_registry(registry):
        result = ns["data_cleaner_node"](state)
    frame = registry.get_dataframe("retail-fixture")
    assert frame["discount_rate"].isna().sum() == 0
    if second == "impute":
        assert frame["satisfaction_score"].isna().sum() == 0
    else:
        assert pd.api.types.is_datetime64_any_dtype(frame["order_date"])
        assert frame["order_date"].tolist() == list(
            pd.date_range("2025-01-01", periods=3)
        )
    tool_results = [m for m in model._requests[-1] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in tool_results} == {c["id"] for c in calls}
    assert all(m.status == "success" for m in tool_results)
    assert len(model._requests) == 2
    assert result["cleaning_metadata"] == cleaner_result()
    assert result["data_cleaning_complete"] is True


def test_cleaner_tool_serialization_releases_after_native_exception(tmp_path):
    from langchain.tools import tool
    import pandas as pd

    ns = load_native(tmp_path, middleware=True)
    registry = DataFrameRegistry(data_dir=tmp_path / "exception-registry")
    registry.register_dataframe(
        pd.DataFrame({"value": [1.0, None, 3.0]}), "retail-fixture"
    )

    @tool
    def cleaner_failure() -> str:
        """Raise an unexpected native tool exception."""
        raise RuntimeError("cleaner serialization canary")

    ns["data_cleaning_tools"] = [*ns["data_cleaning_tools"], cleaner_failure]
    model = ParallelCleanerModel(
        response_name="CleaningMetadata",
        payload=cleaner_result().model_dump(),
        parallel_calls=[{"name": "cleaner_failure", "args": {}, "id": "failure-call"}],
    )
    ns["data_cleaner_llm"] = model
    ns["data_cleaner_agent"] = ns["create_data_cleaner_agent"](
        initial_description(), ["retail-fixture"]
    )
    state = worker_state(tmp_path)
    state["cleaning_metadata"] = None
    with override_global_registry(registry), pytest.raises(
        RuntimeError, match="serialization canary"
    ):
        ns["data_cleaner_node"](state)
    assert state["cleaning_metadata"] is None
    assert not state.get("data_cleaning_complete")
    # Reuse the same factory/lock after the failing ToolNode worker has exited.
    # A new native thread avoids carrying an unfinished tool exchange forward.
    model._requests.clear()
    model.parallel_calls = [
        {
            "name": "fill_missing_median",
            "args": {"df_id": "retail-fixture", "column_name": "value"},
            "id": "corrected-call",
        }
    ]
    state["_config"] = {"configurable": {"thread_id": "cleaner-after-exception"}}
    with override_global_registry(registry):
        result = ns["data_cleaner_node"](state)
    assert registry.get_dataframe("retail-fixture")["value"].tolist() == [1.0, 2.0, 3.0]
    assert result["cleaning_metadata"] == cleaner_result()
    assert result["data_cleaning_complete"] is True
