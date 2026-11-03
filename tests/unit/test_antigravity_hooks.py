"""Exercise the hook subprocess boundary without an installed memory service."""

import json
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / ".agents" / "hooks.json"


@pytest.fixture
def memory_home(tmp_path):
    home = tmp_path / "memory installation"
    hooks = home / "hooks"
    hooks.mkdir(parents=True)
    # Stand-ins for the external mem0ry4ai process protocol, using real I/O.
    for name in ("session_start", "capture"):
        (hooks / f"{name}.py").write_text(
            "import json, pathlib, sys\n"
            "data = json.load(sys.stdin)\n"
            f"pathlib.Path(__file__).with_name('{name}.received').write_text(json.dumps(data))\n"
            "print('Recovered project decision')\n",
            encoding="utf-8",
        )
    return home


def run_hook(event, memory_home, *, raw_input=None, **fields):
    assert CONFIG.is_file(), "IDD has no registered Antigravity lifecycle hooks"
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    handlers = config["idd-memory-lifecycle"][event]
    assert len(handlers) == 1
    # Antigravity's non-tool events take flat handlers, not matcher/hooks wrappers.
    handler = handlers[0]
    assert handler["type"] == "command"
    command = shlex.split(handler["command"])
    assert command[0] == "python"
    command[0] = sys.executable
    payload = {
        "conversationId": "test-conversation",
        "workspacePaths": [str(ROOT.parent / "other-project"), str(ROOT)],
        "transcriptPath": str(memory_home / "transcript.jsonl"),
        "artifactDirectoryPath": str(memory_home / "artifacts"),
        "modelName": "test-model",
        **fields,
    }
    wire_input = json.dumps(payload, ensure_ascii=False) if raw_input is None else raw_input
    if isinstance(wire_input, str):
        wire_input = wire_input.encode("utf-8")
    result = subprocess.run(
        command, input=wire_input, capture_output=True,
        cwd=ROOT / ".agents", timeout=10,
        env={**os.environ, "MEM0RY4AI_HOME": str(memory_home)},
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), result.stderr.decode("utf-8", errors="replace")


def test_first_invocation_recovers_context_for_idd_in_multi_root_workspace(memory_home):
    output, _ = run_hook("PreInvocation", memory_home, invocationNum=0)
    messages = "\n".join(step["ephemeralMessage"] for step in output["injectSteps"])
    assert "Recovered project decision" in messages
    assert "memory-scout" in messages
    received = json.loads((memory_home / "hooks/session_start.received").read_text())
    assert received["cwd"] == str(ROOT)
    assert received["session_id"] == "test-conversation"
    assert received["source"] == "antigravity"
    assert received["hook_event_name"] == "SessionStart"


def test_later_invocations_do_not_repeat_recovery(memory_home):
    output, _ = run_hook("PreInvocation", memory_home, invocationNum=4)
    assert output == {"injectSteps": []}
    assert not (memory_home / "hooks/session_start.received").exists()


def test_missing_memory_installation_is_visible_and_keeps_local_recovery(tmp_path):
    output, stderr = run_hook("PreInvocation", tmp_path, invocationNum=0)
    messages = "\n".join(step["ephemeralMessage"] for step in output["injectSteps"])
    assert "memory-bank/activeContext.md" in messages
    assert "unavailable" in messages.lower()
    assert "session_start.py" in stderr


def test_failed_memory_recovery_does_not_inject_child_stdout(memory_home):
    (memory_home / "hooks/session_start.py").write_text(
        "import sys\nprint('BROKEN_CONTEXT')\nsys.exit(7)\n"
    )
    output, stderr = run_hook("PreInvocation", memory_home, invocationNum=0)
    assert "BROKEN_CONTEXT" not in json.dumps(output)
    assert "unavailable" in json.dumps(output).lower()
    assert "7" in stderr


def test_idle_stop_captures_transcript_with_antigravity_metadata(memory_home):
    output, _ = run_hook(
        "Stop", memory_home, fullyIdle=True, terminationReason="model_stop", executionNum=1
    )
    assert output == {"decision": ""}
    received = json.loads((memory_home / "hooks/capture.received").read_text())
    assert received["cwd"] == str(ROOT)
    assert received["session_id"] == "test-conversation"
    assert received["transcript_path"] == str(memory_home / "transcript.jsonl")
    assert received["hook_event_name"] == "SessionEnd"
    assert received["source"] == "antigravity"
    assert received["fully_idle"] is True


def test_stop_with_background_work_does_not_capture_partial_closeout(memory_home):
    output, _ = run_hook("Stop", memory_home, fullyIdle=False, terminationReason="model_stop")
    assert output == {"decision": ""}
    assert not (memory_home / "hooks/capture.received").exists()


def test_capture_failure_is_visible_without_trapping_agent_in_stop_loop(memory_home):
    (memory_home / "hooks/capture.py").write_text("raise SystemExit(9)\n")
    output, stderr = run_hook("Stop", memory_home, fullyIdle=True)
    assert output == {"decision": ""}
    assert "9" in stderr


@pytest.mark.parametrize("raw_input", [
    "not JSON", "[]", b'{"path": "\xff"}',
    '{"invocationNum":0,"fullyIdle":true}'.encode("utf-16"),
    '{"invocationNum":0,"fullyIdle":true}'.encode("utf-32"),
], ids=["malformed-json", "non-object", "invalid-utf8", "utf16", "utf32"])
@pytest.mark.parametrize("event,expected", [
    ("PreInvocation", {"injectSteps": []}), ("Stop", {"decision": ""}),
])
def test_bad_input_returns_valid_hook_json_without_running_backend(memory_home, raw_input, event, expected):
    output, stderr = run_hook(event, memory_home, raw_input=raw_input)
    assert output == expected
    assert "invalid hook payload" in stderr
    assert not list((memory_home / "hooks").glob("*.received"))


def test_backend_timeout_is_bounded_and_injects_fallback(memory_home, monkeypatch, capsys):
    path = ROOT / ".agents/hooks/idd_memory.py"
    spec = importlib.util.spec_from_file_location("idd_memory_hook", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setenv("MEM0RY4AI_HOME", str(memory_home))

    def timeout(command, **kwargs):
        assert 0 < kwargs["timeout"] < 30
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(adapter.subprocess, "run", timeout)
    output = adapter.handle("start", {"invocationNum": 0})
    assert "unavailable" in json.dumps(output).lower()
    assert "TimeoutExpired" in capsys.readouterr().err


def test_non_ascii_memory_survives_non_utf8_launcher(memory_home, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    (memory_home / "hooks/session_start.py").write_text(
        "print('\\u5317\\u4eac')\n", encoding="utf-8"
    )
    output, _ = run_hook("PreInvocation", memory_home, invocationNum=0)
    messages = "\n".join(step["ephemeralMessage"] for step in output["injectSteps"])
    assert "北京" in messages


@pytest.mark.parametrize("launcher_encoding", ["ascii", "cp1252"])
@pytest.mark.parametrize("transcript_folder", ["café", "会话"])
@pytest.mark.parametrize("event,backend", [
    ("PreInvocation", "session_start"), ("Stop", "capture"),
])
def test_utf8_transcript_path_survives_non_utf8_stdin(
    memory_home, monkeypatch, launcher_encoding, transcript_folder, event, backend
):
    monkeypatch.setenv("PYTHONIOENCODING", launcher_encoding)
    transcript_path = str(memory_home / transcript_folder / "transcript.jsonl")
    output, _ = run_hook(
        event, memory_home, invocationNum=0, fullyIdle=True,
        transcriptPath=transcript_path,
    )
    received_path = memory_home / "hooks" / f"{backend}.received"
    assert received_path.is_file(), "Valid UTF-8 payload must reach the memory backend"
    received = json.loads(received_path.read_text(encoding="utf-8"))
    assert received["transcript_path"] == transcript_path
    if event == "PreInvocation":
        assert any("Recovered project decision" in step["ephemeralMessage"]
                   for step in output["injectSteps"])
    else:
        assert output == {"decision": ""}
