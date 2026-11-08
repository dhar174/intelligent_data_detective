"""Offline #177 regressions against generated functions and real inner agents.

Only chat/summary models and embedding-backed memory are controlled boundaries.
No notebook model configuration, credential loader or live runner is executed.
"""

import ast
import base64
import builtins
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
import time
import textwrap
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
    assert effective[1:] == messages[4:]
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
