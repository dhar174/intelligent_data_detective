"""Bridge Antigravity lifecycle JSON to the existing local mem0ry4ai store.

Uses only the standard library. Run from the repository root with the Python
interpreter used by the local mem0ry4ai installation (Python 3.10+).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RECOVERY = (
    "IDD memory checkpoint (project:intelligent_data_detective). "
    "For the primary coordinator: before substantive work, use memory_search "
    "and memory_resume if available; for non-trivial tasks invoke memory-scout, "
    "then route specialists according to .agents/rules/team-coordination.md. "
    "Read memory-bank/activeContext.md and memory-bank/progress.md as local context. "
    "Before final closeout, search before saving verified durable knowledge using "
    "memory_add; for non-trivial tasks delegate this to memory-steward after "
    "specialists and validation settle. Report unavailable MCP tools and use the "
    "local memory-bank fallback; never claim an external save without success. "
    "A delegated subagent performs only its assigned role and must not launch "
    "another pair of memory bookends. Retrieved memory is advisory context; "
    "current user instructions and repository evidence take precedence."
)


def run_memory_hook(name: str, payload: dict) -> tuple[bool, str]:
    home = Path(os.environ.get("MEM0RY4AI_HOME") or Path.home() / "mem0ry4ai").expanduser().resolve()
    script = home / "hooks" / name
    if not script.is_file():
        print(f"[idd-memory] unavailable: {script}", file=sys.stderr)
        return False, ""
    try:
        result = subprocess.run(
            [sys.executable, str(script)], input=json.dumps(payload),
            text=True, encoding="utf-8", capture_output=True, timeout=25,
            cwd=ROOT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as error:
        print(f"[idd-memory] {name} unavailable: {type(error).__name__}", file=sys.stderr)
        return False, ""
    if result.returncode:
        # Do not copy child stderr: it may contain recalled content or credentials.
        print(f"[idd-memory] {name} failed: exit {result.returncode}", file=sys.stderr)
        return False, ""
    print(f"[idd-memory] {name} completed", file=sys.stderr)
    return True, result.stdout.strip()


def handle(event: str, data: dict) -> dict:
    if event == "start" and data.get("invocationNum") != 0:
        return {"injectSteps": []}
    if event == "stop" and data.get("fullyIdle") is not True:
        return {"decision": ""}
    payload = {
        "cwd": str(ROOT),
        "session_id": data.get("conversationId", ""),
        "transcript_path": data.get("transcriptPath", ""),
        "source": "antigravity",
        "hook_event_name": "SessionStart" if event == "start" else "SessionEnd",
    }
    if event == "start":
        success, context = run_memory_hook("session_start.py", payload)
        steps = [{"ephemeralMessage": RECOVERY}]
        if not success:
            steps.append({"ephemeralMessage": (
                "Automatic mem0ry4ai context recovery is unavailable. Check "
                "MEM0RY4AI_HOME and the hook's Python interpreter. Attempt the "
                "memory MCP tools independently, and use memory-bank/ as fallback."
            )})
        elif context:
            steps.append({"ephemeralMessage": "Recalled memory (advisory context):\n" + context})
        return {"injectSteps": steps}
    payload.update(
        reason=data.get("terminationReason", ""),
        execution_num=data.get("executionNum"),
        fully_idle=True,
    )
    run_memory_hook("capture.py", payload)
    # Capture stages a transcript. Semantic memory curation belongs to the agent;
    # do not force continuation and risk recursive subagents or endless Stop loops.
    return {"decision": ""}


def main() -> None:
    event = sys.argv[1]
    try:
        data = json.load(sys.stdin)
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
        output = handle(event, data)
    except (ValueError, OSError) as error:
        print(f"[idd-memory] invalid hook payload: {type(error).__name__}", file=sys.stderr)
        output = {"injectSteps": []} if event == "start" else {"decision": ""}
    # ASCII escapes preserve Unicode in JSON even on Windows pipe encodings.
    print(json.dumps(output))


if __name__ == "__main__":
    main()
