"""Offline #177 regressions against generated functions and real inner agents.

Only chat/summary models and embedding-backed memory are controlled boundaries.
No notebook model configuration, credential loader or live runner is executed.
"""

import ast
from contextlib import contextmanager
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import shutil
import socket
import time
from types import SimpleNamespace
import typing
import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model
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
from pydantic import Field, PrivateAttr, ValidationError
import pytest

import idd_core
import idd_state
from idd_registry import get_global_df_registry
from tests.unit.test_viz_join import FLAGS, figures, notebook

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
    ns.update(datetime=datetime, json=json, os=os, re=re, shutil=shutil, uuid=uuid)
    for value in (
        create_agent,
        AgentState,
        before_model,
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
        RUNTIME=SimpleNamespace(run_dir=base, artifacts_dir=base / "artifacts"),
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
    )
    execute_definitions(
        ns,
        25,
        {
            "DEFAULT_TOOLING_GUIDELINES",
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
            summary_llm=summary, quick_summary_llm=summary, complex_summary_llm=summary
        )
        execute_definitions(
            ns,
            44,
            {"prehook", "prehook_quick", "prehook_complex"},
        )
    execute_definitions(
        ns,
        44,
        {
            "_make_unknown_tool_guard",
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
            "viz_evaluator_node",
        },
    )
    return ns


class ControlledChatModel(BaseChatModel):
    """Real create_agent/ToolStrategy execution, scripted model boundary only."""

    response_name: str
    payload: dict
    follow_instructions: bool = False
    task_payloads: dict[str, dict] = Field(default_factory=dict)
    _requests: list = PrivateAttr(default_factory=list)
    _bindings: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-native-completion"

    def bind_tools(self, tools, **kwargs):
        self._bindings.append(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._requests.append(messages)
        name = self.response_name
        payload = self.payload
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
        if self.follow_instructions and any(
            "call the `respond` tool" in str(m.content) for m in messages
        ):
            name = "respond"
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
