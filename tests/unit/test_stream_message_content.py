"""Offline regressions for the generated Cell 75 stream display."""

import ast
from collections import defaultdict
from collections.abc import Mapping
import json
import logging
from pathlib import Path
import re
import socket
import textwrap
import time
import typing

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    RemoveMessage,
    ToolMessage,
    ToolMessageChunk,
)
import pytest

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "IntelligentDataDetective_beta_v5_patched.ipynb"
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External access forbidden in offline stream tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def stream():
    """Execute the generated display code without executing notebook setup cells."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {
        **vars(typing),
        "Mapping": Mapping,
        "_SENTINEL": object(),
        "re": re,
        "textwrap": textwrap,
        "defaultdict": defaultdict,
        "AIMessage": AIMessage,
        "AIMessageChunk": AIMessageChunk,
        "RemoveMessage": RemoveMessage,
        "ToolMessage": ToolMessage,
        "ToolMessageChunk": ToolMessageChunk,
    }
    for cell, wanted in (
        (40, {"getnestedattr"}),
        (
            75,
            {
                "_clean_stream_text",
                "derive_label",
                "content_to_text",
                "id_key",
                "StreamState",
                "ensure_header",
                "print_incremental_chunk",
                "print_full_text",
                "print_ai_message_chunk",
                "print_ai_message_complete",
                "print_tool_message",
                "handle_updates",
                "handle_messages",
                "process_stream_event",
                "_log_stage_start",
                "_log_stage_end",
                "stream_graph_output",
            },
        ),
    ):
        source = "".join(notebook["cells"][cell]["source"])
        tree = ast.parse(source)
        selected = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
        ]
        assert {node.name for node in selected} == wanted
        module = ast.Module(body=selected, type_ignores=[])
        exec(compile(module, f"<generated Cell {cell}>", "exec"), namespace)

    namespace.update(
        names_to_ids={},
        _ANSI_RE=re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]"),
        STREAM_MODES={"messages", "updates", "values", "custom", "debug", "events"},
        stream_state=namespace["StreamState"](),
        received_steps=[],
        _pl_logger=logging.getLogger(__name__),
        _pl_time=time,
        _pl_stage_times={},
        _pl_logged_nodes=set(),
    )
    return namespace


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("A plain response.", "A plain response."),
        ("", ""),
        (" \t ", ""),
        ([], ""),
        (
            [{"type": "text", "text": "A structured response."}],
            "A structured response.",
        ),
        (
            [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Reasoning. "}],
                },
                {"type": "text", "text": "A final answer."},
            ],
            "Reasoning. A final answer.",
        ),
        (
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.invalid/chart.png"},
                }
            ],
            "",
        ),
    ],
)
def test_complete_message_prints_supported_content_without_crashing(
    stream, capsys, content, expected
):
    message = AIMessage(content=content, id="message-1", name="viz_evaluator")
    stream["print_ai_message_complete"](message, "message-1", "viz_evaluator")
    printed = capsys.readouterr().out
    assert printed == "\n[viz_evaluator][viz_evaluator]\n" + expected


@pytest.mark.parametrize("message_kind", ["ai", "tool"])
def test_structured_message_event_reaches_next_update(stream, capsys, message_kind):
    content = [{"type": "text", "text": "Figure evidence received."}]
    if message_kind == "ai":
        content.insert(
            0,
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Checked figures. "}],
            },
        )
        message = AIMessage(content=content, id="event-message", name="viz_evaluator")
    else:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,c3ludGhldGlj"},
            }
        )
        message = ToolMessage(
            content=content,
            id="event-message",
            tool_call_id="call-1",
            name="get_visualization",
        )

    events = [
        ((), "messages", (message, {"langgraph_node": "viz_evaluator"})),
        ((), "updates", {"viz_evaluator": {"next": "report_orchestrator"}}),
    ]

    class OfflineGraph:
        def stream(self, initial_state, *, stream_mode, config, subgraphs):
            assert stream_mode == ["messages", "updates"]
            assert subgraphs is True
            yield from events

    stream["stream_graph_output"](OfflineGraph(), {}, {}, thread_id="offline")
    output = capsys.readouterr().out
    assert "Figure evidence received." in output
    if message_kind == "ai":
        assert "Checked figures." in output
    assert "Next Agent: report_orchestrator" in output
    assert "Streaming error" not in output
    assert "Stream complete" in output
    assert [entry[0] for entry in stream["received_steps"]] == events


def test_streamed_chunk_and_final_structured_message_do_not_duplicate_text(
    stream, capsys
):
    chunk = AIMessageChunk(content="Partial", id="shared-id", name="viz_evaluator")
    complete = AIMessage(
        content=[{"type": "text", "text": "Partial and final."}],
        id="shared-id",
        name="viz_evaluator",
    )
    for message in (chunk, complete):
        stream["process_stream_event"](
            ((), "messages", (message, {"langgraph_node": "viz_evaluator"}))
        )
    output = capsys.readouterr().out
    assert output.count("Partial and final.") == 1


def test_real_langgraph_stream_continues_past_structured_message(stream, capsys):
    from langgraph.graph import END, START, StateGraph, add_messages

    class State(typing.TypedDict):
        messages: typing.Annotated[list, add_messages]
        next: str

    def evaluate(state):
        return {
            "messages": [
                AIMessage(
                    content=[
                        {
                            "type": "reasoning",
                            "summary": [
                                {"type": "summary_text", "text": "Figures checked. "}
                            ],
                        },
                        {"type": "text", "text": "Evaluation complete."},
                    ],
                    id="evaluation-id",
                    name="viz_evaluator",
                )
            ],
            "next": "report_orchestrator",
        }

    def report(state):
        return {
            "messages": [
                AIMessage(
                    content="Report stage reached.",
                    id="report-id",
                    name="report_orchestrator",
                )
            ]
        }

    builder = StateGraph(State)
    builder.add_node("viz_evaluator", evaluate)
    builder.add_node("report_orchestrator", report)
    builder.add_edge(START, "viz_evaluator")
    builder.add_edge("viz_evaluator", "report_orchestrator")
    builder.add_edge("report_orchestrator", END)
    graph = builder.compile()

    stream["stream_graph_output"](graph, {"messages": []}, {}, thread_id="offline")
    output = capsys.readouterr().out
    assert "Figures checked." in output
    assert "Evaluation complete." in output
    assert "Report stage reached." in output
    assert "Streaming error" not in output
    assert "Stream complete" in output


def test_unrelated_stream_failure_still_reports_an_error(stream, capsys):
    class OfflineGraph:
        def stream(self, initial_state, *, stream_mode, config, subgraphs):
            yield (
                (),
                "messages",
                (
                    AIMessage(
                        content=[{"type": "text", "text": "Before failure."}],
                        id="message-1",
                        name="viz_evaluator",
                    ),
                    {},
                ),
            )
            raise RuntimeError("offline sentinel failure")

    stream["stream_graph_output"](OfflineGraph(), {}, {}, thread_id="offline")
    output = capsys.readouterr().out
    assert "Before failure." in output
    assert "Streaming error: offline sentinel failure" in output
    assert "Stream complete" not in output
