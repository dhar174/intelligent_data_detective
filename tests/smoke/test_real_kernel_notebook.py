"""
Issue #164: Real-Kernel No-Key Runtime Verification Suite.

Executes actual code cells extracted from the committed regenerated notebook
`IntelligentDataDetective_beta_v5_patched.ipynb` inside a real Jupyter kernel
(`ipykernel`) via `nbclient.NotebookClient` and `nbformat`.

Covers:
1. Stage 2 Real-Kernel Execution:
   - Cell 4: bootstrap, manifest SHA-256 verification, install suppression, and canonical import provenance.
   - Cells 7, 10, 16, 19, 22: effective canonical definitions, WORKING_DIRECTORY sync, and object identity (`is`).
   - Cell 32: representative real decorated DataFrame tools under `override_global_registry`, non-df_id tools,
     sanitized structured error responses, and fail-closed `_resolve_artifact_path` traversal protection.
   - Cell 46: `make_supervisor_node` factory and canonical `idd_graph.Router` closure binding.
   - Cell 57: `emergency_correspondence_node` and `CLASS_TO_AGENT` routing with CP4-CELL57-EMERGENCY-FILTER.
   - Cell 60 + Cells 94 & 96: real canonical graph assembly (70 State fields, 15 nodes, 33 edges, 21 conditional edges),
     representative no-key graph transitions (including `viz_join`, `report_join`, and `EMERGENCY_MSG`),
     checkpoint demo isolation (`checkpoints_demo.sqlite`), production graph preservation, and connection cleanup.
2. Stage 3 SQLite Matrix:
   - SQLite present (`langgraph.checkpoint.sqlite.SqliteSaver`).
   - SQLite absent (`langgraph.checkpoint.memory.MemorySaver` fallback).
3. Stage 4 Security, Adversarial Fail-Closed Startup & Negative Controls:
   - Missing canonical files.
   - Tampered manifest digests (proving tampered module is never imported or executed).
   - Modified `idd_bootstrap.py` (proving tampered bootstrap is rejected before import/execution).
   - Pre-loaded canonical module in `sys.modules` from an unrelated checkout.
   - Competing module locations on `sys.path`.
   - Missing cloned repository in a Colab-style environment.
   - Corrupted or unexpected patcher source anchors (`verify_cp4_input_cell`, `replace_required`).
   - Deliberate in-memory notebook cell mutations demonstrating test failure (negative controls).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import ipykernel  # noqa: F401
import jupyter_client
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import nbformat
import pytest

from tests.unit.test_core_parity_guard import (
    REPO_ROOT,
    _create_repo_sandbox,
    assert_notebook_static_parity,
    load_patched_notebook,
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


PROBE_PREFIX = "__IDD_KERNEL_PROBE_JSON__:"

# Minimal no-key LLM and runtime config setup injected immediately prior to Cell 60.
# Cell 1, 4, 7, 10, 16, 19, 22, 28, 31, 32, 46, 57, 60, 94, and 96 run unmodified.
NO_KEY_CELL60_SETUP_SOURCE = """
from unittest.mock import MagicMock
from langgraph.store.memory import InMemoryStore
from langgraph.cache.memory import InMemoryCache

for _llm_name in [
    "big_picture_llm",
    "router_llm",
    "reply_llm",
    "plan_llm",
    "replan_llm",
    "progress_llm",
    "todo_llm",
    "low_reasoning_llm",
]:
    globals()[_llm_name] = MagicMock(name=_llm_name)

sample_prompt_text = "Analyze dataset quality, clean anomalies, and summarize key findings."
in_memory_store = InMemoryStore()
InMemoryCache = InMemoryCache
thread_id = "issue-164-kernel-thread"
run_config = {"configurable": {"thread_id": thread_id}}
"""

KERNEL_CLEANUP_SOURCE = """
import sqlite3 as _sqlite3

for _conn_name in ("conn", "_cp_conn", "_restore_conn"):
    _conn_obj = globals().get(_conn_name)
    if isinstance(_conn_obj, _sqlite3.Connection):
        try:
            _conn_obj.close()
        except Exception:
            pass

_temp_dir_obj = globals().get("_TEMP_DIRECTORY")
if _temp_dir_obj is not None and hasattr(_temp_dir_obj, "cleanup"):
    try:
        _temp_dir_obj.cleanup()
    except Exception:
        pass
"""


def resolve_real_kernel_name() -> str:
    """Resolve and validate the real Jupyter kernelspec to execute against."""
    requested = os.environ.get("IDD_KERNEL_NAME", "").strip()
    ksm = jupyter_client.kernelspec.KernelSpecManager()
    available = ksm.find_kernel_specs()
    kernel_name = requested or (
        "python3" if "python3" in available else next(iter(available))
    )
    spec = ksm.get_kernel_spec(kernel_name)
    assert spec is not None and spec.argv, f"Invalid kernelspec for {kernel_name!r}"
    return kernel_name


@contextlib.contextmanager
def _scoped_environ(
    overrides: Optional[Dict[str, Optional[str]]] = None,
) -> Iterator[None]:
    """Temporarily set environment variables so the spawned ipykernel inherits them."""
    effective: Dict[str, Optional[str]] = {
        "IDD_SKIP_NOTEBOOK_INSTALLS": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    if overrides:
        effective.update(overrides)

    saved: Dict[str, Optional[str]] = {}
    for key, val in effective.items():
        saved[key] = os.environ.get(key)
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
    try:
        yield
    finally:
        for key, old_val in saved.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


CellSpec = Union[int, Tuple[str, str]]

# Checkout artifacts that real notebook cells could create or mutate when the kernel's
# working directory is the repository root (e.g. Cell 60 opens relative `checkpoints.sqlite`).
_CHECKOUT_GUARDED_PATTERNS = ("checkpoints*.sqlite*",)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_checkout_state(root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Capture top-level file names plus digests of guarded checkpoint databases."""
    guarded: Dict[str, str] = {}
    for pattern in _CHECKOUT_GUARDED_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file():
                guarded[path.name] = _sha256_file(path)
    top_level = sorted(p.name for p in root.iterdir() if p.is_file())
    return {"guarded": guarded, "top_level_files": top_level}


def assert_checkout_unchanged(
    before: Dict[str, Any], root: Path = REPO_ROOT, context: str = ""
) -> None:
    """Fail if a kernel run created, removed or modified checkout files."""
    after = snapshot_checkout_state(root)
    assert after["guarded"] == before["guarded"], (
        f"Kernel run mutated checkout checkpoint databases {context}: "
        f"before={before['guarded']} after={after['guarded']}"
    )
    new_files = sorted(set(after["top_level_files"]) - set(before["top_level_files"]))
    removed = sorted(set(before["top_level_files"]) - set(after["top_level_files"]))
    assert not new_files and not removed, (
        f"Kernel run changed top-level checkout files {context}: "
        f"new={new_files} removed={removed}"
    )


def _enter_work_dir_source(work_dir: Path) -> str:
    return (
        "import os as _idd_os\n"
        f"_idd_os.makedirs({str(work_dir)!r}, exist_ok=True)\n"
        f"_idd_os.chdir({str(work_dir)!r})\n"
    )


def execute_cells_in_real_kernel(
    cell_specs: Sequence[CellSpec],
    *,
    notebook_dict: Optional[Dict[str, Any]] = None,
    cwd: Path = REPO_ROOT,
    work_dir: Optional[Path] = None,
    env_overrides: Optional[Dict[str, Optional[str]]] = None,
    timeout: int = 180,
) -> Tuple[nbformat.NotebookNode, Dict[str, Any], List[str]]:
    """
    Build a notebook from actual committed cells in `IntelligentDataDetective_beta_v5_patched.ipynb`
    (plus any explicit probe/setup cells) and execute it in a real `ipykernel` via `NotebookClient`.

    - The kernel starts in `cwd` so generated Cell 4 can locate the verified checkout.
    - When `work_dir` is given, the kernel `chdir`s into it immediately after Cell 4 so relative
      writes from later cells (e.g. Cell 60 `checkpoints.sqlite`) land in a disposable directory.
      Executing Cell 60 without `work_dir` is refused.
    - Connection/tempdir cleanup always runs in a harness-level `finally`, even when a cell raises.
    - The checkout is snapshotted before and verified after every run (success or failure).

    Returns `(executed_nb, probe_payloads_by_label, stream_texts_by_cell)`.
    """
    if 60 in cell_specs and work_dir is None:
        raise ValueError(
            "Executing Cell 60 requires an isolated work_dir; it opens a relative checkpoints.sqlite."
        )

    source_nb = notebook_dict if notebook_dict is not None else load_patched_notebook()
    nb = nbformat.v4.new_notebook()
    labels: List[str] = []

    def _append(label: str, src: str, meta_key: str, meta_val: Any) -> None:
        cell_node = nbformat.v4.new_code_cell(source=src)
        cell_node.metadata[meta_key] = meta_val
        nb.cells.append(cell_node)
        labels.append(label)

    for spec in cell_specs:
        if isinstance(spec, int):
            raw_cell = source_nb["cells"][spec]
            assert (
                raw_cell.get("cell_type") == "code"
            ), f"Cell {spec} is not a code cell"
            src = raw_cell.get("source", [])
            src = "".join(src) if isinstance(src, list) else src
            _append(f"cell_{spec}", src, "idd_cell_index", spec)
            if spec == 4 and work_dir is not None:
                _append(
                    "enter_isolated_work_dir",
                    _enter_work_dir_source(work_dir),
                    "idd_probe_label",
                    "enter_isolated_work_dir",
                )
        else:
            label, src = spec
            _append(label, src, "idd_probe_label", label)

    kernel_name = resolve_real_kernel_name()
    checkout_before = snapshot_checkout_state()
    try:
        with _scoped_environ(env_overrides):
            client = NotebookClient(
                nb,
                timeout=timeout,
                kernel_name=kernel_name,
                resources={"metadata": {"path": str(cwd)}},
            )
            client.reset_execution_trackers()
            with client.setup_kernel():
                try:
                    for index, cell in enumerate(list(nb.cells)):
                        client.execute_cell(cell, index)
                finally:
                    # Harness-level cleanup: runs even after an expected cell failure.
                    cleanup_cell = nbformat.v4.new_code_cell(
                        source=KERNEL_CLEANUP_SOURCE
                    )
                    cleanup_cell.metadata["tags"] = ["raises-exception"]
                    cleanup_cell.metadata["idd_probe_label"] = "harness_cleanup"
                    nb.cells.append(cleanup_cell)
                    labels.append("harness_cleanup")
                    client.execute_cell(cleanup_cell, len(nb.cells) - 1)
            executed_nb = nb
    finally:
        assert_checkout_unchanged(
            checkout_before, context=f"(cells={list(cell_specs)})"
        )

    probes: Dict[str, Any] = {}
    cell_streams: List[str] = []
    for idx, cell in enumerate(executed_nb.cells):
        chunks: List[str] = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text = out.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                chunks.append(text)
                for line in text.splitlines():
                    if line.startswith(PROBE_PREFIX):
                        payload = json.loads(line[len(PROBE_PREFIX) :])
                        probes[labels[idx]] = payload
        cell_streams.append("".join(chunks))

    return executed_nb, probes, cell_streams


def stream_for(executed_nb: nbformat.NotebookNode, streams: List[str], key: str) -> str:
    """Return the stream text of the executed cell with notebook index or probe label `key`."""
    for idx, cell in enumerate(executed_nb.cells):
        meta = cell.get("metadata", {})
        if (
            f"cell_{meta.get('idd_cell_index')}" == key
            or meta.get("idd_probe_label") == key
        ):
            return streams[idx]
    raise KeyError(key)


def _sqlite_schema(path: Path) -> List[Tuple[str, str]]:
    conn = sqlite3.connect(str(path))
    try:
        return sorted(
            conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name").fetchall()
        )
    finally:
        conn.close()


@pytest.fixture
def checkout_checkpoint_canary() -> Iterator[Path]:
    """
    Seed a canary `checkpoints.sqlite` in the checkout root (outside every per-test work_dir)
    — or reuse a pre-existing developer database untouched — and assert after the test that
    its bytes and schema are unchanged and no checkpoint/WAL/SHM files appeared.
    """
    canary = REPO_ROOT / "checkpoints.sqlite"
    created = False
    if not canary.exists():
        conn = sqlite3.connect(str(canary))
        try:
            conn.execute("CREATE TABLE reviewer_canary (marker TEXT NOT NULL)")
            conn.execute("INSERT INTO reviewer_canary VALUES ('issue-164')")
            conn.commit()
        finally:
            conn.close()
        created = True
    before_digest = _sha256_file(canary)
    before_schema = _sqlite_schema(canary)
    before_siblings = sorted(p.name for p in REPO_ROOT.glob("checkpoints*"))
    try:
        yield canary
    finally:
        try:
            assert (
                _sha256_file(canary) == before_digest
            ), "Checkout checkpoints.sqlite bytes changed"
            assert (
                _sqlite_schema(canary) == before_schema
            ), "Checkout checkpoints.sqlite schema changed"
            after_siblings = sorted(p.name for p in REPO_ROOT.glob("checkpoints*"))
            assert (
                after_siblings == before_siblings
            ), f"New checkpoint/WAL/SHM files in checkout: {before_siblings} -> {after_siblings}"
        finally:
            if created:
                canary.unlink()


# ===========================================================================
# Stage 2 & Stage 3: Real-Kernel Execution of Committed Notebook Cells
# ===========================================================================


class TestRealKernelNotebookExecution:
    """Verify actual generated notebook cells execute in a real Jupyter kernel."""

    def test_committed_notebook_static_parity_precheck(self):
        """Ensure the committed 99-cell notebook satisfies static CP4 parity before kernel run."""
        nb = load_patched_notebook()
        assert_notebook_static_parity(nb)

    def test_cells_4_through_57_canonical_provenance_tools_supervisor_and_emergency_in_kernel(
        self, tmp_path: Path
    ):
        """
        Execute actual Cells 1, 4, 7, 10, 16, 19, 22, 28, 31, 32, 46, and 57 in a real Jupyter kernel.

        Verifies:
        - Cell 4: real ipykernel shell, install suppression, manifest SHA-256 verification,
          canonical module origins, and `sys.path[0]` repo root promotion.
        - Cells 7, 10, 16, 19, 22: object identity (`is`) for all canonical classes, functions,
          reducers, `AgentMembers`, `MyChatOpenai`, `DataFrameRegistry`, `State` (70 fields,
          W9-SR-DROP invariant), and `CLASS_TO_AGENT`, plus `WORKING_DIRECTORY` synchronization.
        - Cell 32: real decorated DataFrame tools (`drop_column`, `get_dataframe_schema`,
          `assess_data_quality`) under `override_global_registry`, non-`df_id` positional string
          tool execution, structured sanitized error return outside override, and fail-closed
          `_resolve_artifact_path` traversal rejection prior to directory creation.
        - Cell 46: `make_supervisor_node` binds canonical `idd_graph.Router`.
        - Cell 57: `emergency_correspondence_node` routes via canonical `CLASS_TO_AGENT`
          with `CP4-CELL57-EMERGENCY-FILTER` intact.
        """
        probe_code = f"""
import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest

import idd_bootstrap
import idd_core
import idd_graph
import idd_models
import idd_registry
import idd_state
import idd_tools

repo_root = Path({str(REPO_ROOT)!r}).resolve()

# 1. Verify real ipykernel shell & Cell 4 provenance
shell = get_ipython()
assert shell is not None
assert "ZMQInteractiveShell" in type(shell).__name__
assert is_colab is False
assert Path(sys.path[0]).resolve() == repo_root

for mod_stem in idd_bootstrap.CANONICAL_MODULE_STEMS:
    mod_obj = sys.modules[mod_stem]
    assert Path(mod_obj.__file__).resolve() == (repo_root / f"{{mod_stem}}.py").resolve()

# 2. Verify Cells 7, 10, 16, 19, 22 canonical object identity (`is`)
wd = WORKING_DIRECTORY.resolve()
assert idd_tools.WORKING_DIRECTORY.resolve() == wd
assert idd_registry.WORKING_DIRECTORY.resolve() == wd
assert idd_core.WORKING_DIRECTORY.resolve() == wd

assert AgentMembers is idd_state.AgentMembers
assert MyChatOpenai is idd_core.MyChatOpenai
assert BaseNoExtrasModel is idd_models.BaseNoExtrasModel
assert Plan is idd_models.Plan
assert PlanStep is idd_models.PlanStep
assert AnalysisConfig is idd_state.AnalysisConfig
assert DataFrameRegistry is idd_registry.DataFrameRegistry
assert get_global_registry is idd_registry.get_global_registry
assert get_global_df_registry is idd_registry.get_global_df_registry
assert global_df_registry is idd_registry.global_df_registry
assert get_global_df_registry() is global_df_registry
assert State is idd_state.State
assert VizWorkerState is idd_state.VizWorkerState
assert CLASS_TO_AGENT is idd_state.CLASS_TO_AGENT
assert CLASS_TO_AGENT is idd_core.CLASS_TO_AGENT
assert keep_first is idd_state.keep_first
assert bool_or is idd_state.bool_or
assert dict_merge_shallow is idd_state.dict_merge_shallow
assert _reduce_plan_keep_sorted is idd_state._reduce_plan_keep_sorted

assert len(State.__annotations__) == 70
assert "structured_response" not in State.__annotations__

# Verify MyChatOpenai strips prompt_cache_key
adapter = MyChatOpenai(model="gpt-4o", api_key="test-key")
payload = adapter._get_request_payload_mod("test", prompt_cache_key="cache-123")
assert "prompt_cache_key" not in payload

# 3. Verify Cell 32 canonical tool identity, scoped registry isolation & artifact path containment
assert handle_tool_errors is idd_tools.handle_tool_errors
assert validate_dataframe_exists is idd_tools.validate_dataframe_exists
assert _resolve_artifact_path is idd_tools._resolve_artifact_path
assert is_safe_artifact_path is idd_tools.is_safe_artifact_path

scoped_dir = Path({str(tmp_path / "scoped_reg")!r})
scoped_reg = DataFrameRegistry(data_dir=scoped_dir)
test_df = pd.DataFrame({{"keep_col": [10, 20, 30], "drop_me": [1, 2, 3]}})
scoped_df_id = scoped_reg.register_dataframe(test_df, "kernel_scoped_df")

with idd_registry.override_global_registry(scoped_reg):
    assert get_global_df_registry() is scoped_reg
    drop_fn = getattr(drop_column, "func", drop_column)
    drop_msg = drop_fn(df_id=scoped_df_id, column_name="drop_me")
    assert "Column dropped successfully" in drop_msg
    assert list(scoped_reg.get_dataframe(scoped_df_id).columns) == ["keep_col"]

    schema_fn = getattr(get_dataframe_schema, "func", get_dataframe_schema)
    schema_text, schema_artifact = schema_fn(df_id=scoped_df_id)
    assert "keep_col" in schema_text
    assert isinstance(schema_artifact, dict)

# Outside override_global_registry, scoped_df_id must fail closed with sanitized structured error
outside_err = getattr(drop_column, "func", drop_column)(df_id=scoped_df_id, column_name="keep_col")
assert isinstance(outside_err, dict)
assert outside_err.get("status") == "error"
assert "Traceback" not in json.dumps(outside_err)

# Non-df_id tool with positional string argument
@handle_tool_errors
def _non_df_metric_tool(metric_name: str, factor: int = 2):
    return {{"metric": metric_name, "value": 21 * factor}}

non_df_res = _non_df_metric_tool("accuracy", 3)
assert non_df_res == {{"metric": "accuracy", "value": 63}}

# Artifact path containment & pre-mkdir traversal rejection
default_art = _resolve_artifact_path("kernel_check.md")
assert default_art.parent.resolve() == (wd / "artifacts").resolve()
assert idd_tools._is_subpath(default_art, wd)

canary_escape = Path({str(tmp_path / "escaped_dir")!r})
assert not canary_escape.exists()
try:
    _resolve_artifact_path(
        "plot.png",
        config={{"configurable": {{"runtime": {{"artifacts_dir": {str(tmp_path / "safe_artifacts")!r}}}}}}},
        subdir=f"../{{canary_escape.name}}",
    )
    raise AssertionError("Expected ValueError on subdir traversal")
except ValueError as exc:
    assert "escapes artifacts root" in str(exc)
assert not canary_escape.exists()

# 4. Verify Cell 46 supervisor factory binds canonical idd_graph.Router
dummy_llms = [MagicMock() for _ in range(8)]
sup_fn = make_supervisor_node(
    dummy_llms,
    ["initial_analysis", "data_cleaner", "analyst", "file_writer", "visualization", "report_orchestrator"],
    "Analyze dataset",
)
inner_sup = [
    c.cell_contents
    for c in (sup_fn.__closure__ or ())
    if callable(c.cell_contents) and getattr(c.cell_contents, "__name__", "") == "supervisor_node"
]
assert len(inner_sup) == 1
sup_closure_vars = inspect.getclosurevars(inner_sup[0])
assert sup_closure_vars.nonlocals.get("Router") is idd_graph.Router

# 5. Verify Cell 57 emergency_correspondence_node routes via CLASS_TO_AGENT
insights_obj = AnalysisInsights(
    summary="Summary",
    correlation_insights="None",
    anomaly_insights="None",
    recommended_visualizations=[],
    recommended_next_steps=["Next"],
    reply_msg_to_supervisor="Need clarification on imputation",
    finished_this_task=False,
    expect_reply=True,
)
pending_msg = SendAgentMessage(
    recipient="analyst",
    message="Use median imputation.",
    delivery_status=False,
    agent_obj_needs_recreated_bool=False,
    is_message_critical=False,
    immediate_emergency_reroute_to_recipient=False,
    reply_msg_to_supervisor="Sent reply",
    finished_this_task=True,
    expect_reply=False,
)
em_cmd = emergency_correspondence_node({{
    "supervisor_to_agent_msgs": [pending_msg],
    "analysis_insights": insights_obj,
    "last_created_obj": "analysis_insights",
    "next": "EMERGENCY_MSG",
}})
assert em_cmd.goto == "analyst"
assert em_cmd.update["messages"].content == "Use median imputation."

print({PROBE_PREFIX!r} + json.dumps({{
    "status": "ok",
    "kernel_shell": type(shell).__name__,
    "python_executable": sys.executable,
    "state_fields": len(State.__annotations__),
    "emergency_goto": em_cmd.goto,
}}))
"""

        cell_specs: List[CellSpec] = [
            1,
            4,
            7,
            10,
            16,
            19,
            22,
            28,
            31,
            32,
            46,
            57,
            ("probe_cells_4_to_57", probe_code),
        ]
        executed_nb, probes, streams = execute_cells_in_real_kernel(
            cell_specs, work_dir=tmp_path / "kernel_cwd"
        )

        # Verify Cell 4 output markers
        cell4_out = stream_for(executed_nb, streams, "cell_4")
        assert "Skipping in-notebook dependency installation" in cell4_out
        assert "Canonical modules verified against idd_manifest.json" in cell4_out

        probe_res = probes.get("probe_cells_4_to_57")
        assert probe_res is not None, f"Missing probe output. Streams: {streams}"
        assert probe_res["status"] == "ok"
        assert probe_res["state_fields"] == 70
        assert probe_res["emergency_goto"] == "analyst"

    @pytest.mark.parametrize(
        "sqlite_available", [True, False], ids=["sqlite_present", "sqlite_absent"]
    )
    def test_cells_60_94_96_graph_lifecycle_checkpointer_and_transitions_in_kernel(
        self, tmp_path: Path, sqlite_available: bool, checkout_checkpoint_canary
    ):
        """Positive graph lifecycle scenario (see `run_graph_lifecycle_scenario`)."""
        res, streams_by_key = self.run_graph_lifecycle_scenario(
            tmp_path, sqlite_available
        )
        assert res["status"] == "ok"
        assert res["node_count"] == 15
        assert res["edge_count"] == 33
        assert res["conditional_edge_count"] == 21
        assert res["seeded_checkpoints"] >= 10
        if sqlite_available:
            assert res["checkpointer_type"] == "SqliteSaver"
            assert res["migrated_checkpoints"] == res["seeded_checkpoints"]
            assert res["restored_checkpoints"] == res["seeded_checkpoints"]
            assert (tmp_path / "checkpoints.sqlite").exists()
            assert (
                f"copied {res['seeded_checkpoints']} checkpoint(s)"
                in streams_by_key["cell_94"]
            )
            assert (
                f"{res['seeded_checkpoints']} checkpoint(s); next="
                in streams_by_key["cell_96"]
            )
        else:
            assert res["checkpointer_type"] in ("MemorySaver", "InMemorySaver")
            assert (
                "SqliteSaver unavailable; checkpoint migration demonstration skipped."
                in streams_by_key["cell_94"]
            )
            assert (
                "SqliteSaver unavailable; checkpoint restore demonstration skipped."
                in streams_by_key["cell_96"]
            )
        for key in ("cell_94", "cell_96"):
            assert "WARNING" not in streams_by_key[key], streams_by_key[key]

    def test_cell94_lossless_pending_writes_deterministic_order_and_delta_replay(
        self, tmp_path: Path, checkout_checkpoint_canary
    ):
        """
        Codex P2 regression: prove Cell 94 migrate_thread preserves task ordering
        and delta channel replay when tasks have reversed task_id vs task_path ordering.

        Scenario:
        1. Disposable thread with parent checkpoint:
           - task 'z-task', path 'a-path', value ['FIRST']
           - task 'a-task', path 'z-path', value ['SECOND']
        2. Child checkpoint linked to parent, omitting 'progress_reports' from channel_values.
        3. Generated Cell 94 migrate_thread into isolated destination DB.
        4. Assert get_delta_channel_history returns ['FIRST'], ['SECOND'] for both src and dst.
        5. Assert parent CheckpointTuple pending_writes match source exactly in order.
        """
        deterministic_probe = f"""
import json
from pathlib import Path as PathlibPath
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver as _DetSqliteSaver

_det_thread = "deterministic-order-test-thread"
_det_db_path = str((WORKING_DIRECTORY / "det_dest.sqlite").resolve())

# 1. Store parent checkpoint with reversed task-ID vs task-path order:
# 'z-task' with 'a-path' must sort FIRST by (task_path, task_id, idx)
# 'a-task' with 'z-path' must sort SECOND by (task_path, task_id, idx)
_src_cp = data_detective_graph.checkpointer
_cfg_parent = {{"configurable": {{"thread_id": _det_thread, "checkpoint_ns": "", "checkpoint_id": "det-parent"}}}}
_cp_parent = empty_checkpoint()
_saved_parent = _src_cp.put(_cfg_parent, _cp_parent, {{}}, {{}})
_src_cp.put_writes(_saved_parent, [("progress_reports", ["FIRST"])], task_id="z-task", task_path="a-path")
_src_cp.put_writes(_saved_parent, [("progress_reports", ["SECOND"])], task_id="a-task", task_path="z-path")

# 2. Store child linked to parent, leaving progress_reports absent from stored channel_values
_cfg_child = {{
    "configurable": {{
        "thread_id": _det_thread,
        "checkpoint_ns": "",
        "checkpoint_id": _saved_parent["configurable"]["checkpoint_id"],
    }}
}}
_cp_child = empty_checkpoint()
_saved_child = _src_cp.put(_cfg_child, _cp_child, {{}}, {{}})

# 3. Migrate using generated migrate_thread into isolated destination
with _DetSqliteSaver.from_conn_string(_det_db_path) as _dst_cp:
    _copied = migrate_thread(_det_thread, _dst_cp, full_history=True)
    assert _copied == 2, f"Expected 2 checkpoints copied, got {{_copied}}"

    # 4. Assert delta channel history matches source exactly: ['FIRST'], ['SECOND']
    _src_delta = _src_cp.get_delta_channel_history(config=_saved_child, channels=["progress_reports"])
    _dst_delta = _dst_cp.get_delta_channel_history(config=_saved_child, channels=["progress_reports"])
    assert _src_delta == _dst_delta, f"Delta mismatch: src={{_src_delta}} vs dst={{_dst_delta}}"
    _src_writes = _src_delta["progress_reports"]["writes"]
    assert [w[2] for w in _src_writes] == [["FIRST"], ["SECOND"]], _src_writes
    assert [w[0] for w in _src_writes] == ["z-task", "a-task"], _src_writes

    # 5. Assert parent CheckpointTuple pending_writes match source exactly in order
    _src_tups = list(_src_cp.list({{"configurable": {{"thread_id": _det_thread}}}}))
    _dst_tups = list(_dst_cp.list({{"configurable": {{"thread_id": _det_thread}}}}))
    assert len(_dst_tups) == len(_src_tups) == 2
    for _st, _dt in zip(_src_tups, _dst_tups):
        assert _dt.pending_writes == _st.pending_writes, f"PW mismatch: {{_dt.pending_writes}} vs {{_st.pending_writes}}"

print({PROBE_PREFIX!r} + json.dumps({{
    "status": "ok",
    "copied": _copied,
    "delta_writes_order": [w[0] for w in _src_writes],
}}))
"""

        deterministic_nodes_setup = f"""
{NO_KEY_CELL60_SETUP_SOURCE}
import os
import idd_tools
import idd_registry
import idd_core

WORKING_DIRECTORY = PathlibPath({str(tmp_path)!r})
idd_tools.WORKING_DIRECTORY = WORKING_DIRECTORY
idd_registry.WORKING_DIRECTORY = WORKING_DIRECTORY
idd_core.WORKING_DIRECTORY = WORKING_DIRECTORY
os.chdir(str(WORKING_DIRECTORY))

initial_analysis_node = lambda state: {{}}
data_cleaner_node = lambda state: {{}}
analyst_node = lambda state: {{}}
visualization_orchestrator = lambda state: {{}}
viz_worker = lambda state: {{}}
viz_evaluator_node = lambda state: {{}}
report_orchestrator = lambda state: {{}}
section_worker = lambda state: {{}}
report_packager_node = lambda state: {{}}
file_writer_node = lambda state: {{}}
"""
        cell_specs: List[CellSpec] = [
            1,
            4,
            7,
            10,
            16,
            19,
            22,
            28,
            31,
            32,
            46,
            57,
            ("setup_cell60_no_key_stubs", deterministic_nodes_setup),
            60,
            94,
            ("deterministic_pending_writes_probe", deterministic_probe),
        ]
        executed_nb, probes, streams = execute_cells_in_real_kernel(
            cell_specs, work_dir=tmp_path / "kernel_cwd"
        )
        probe_res = probes.get("deterministic_pending_writes_probe")
        assert probe_res is not None, f"Missing probe output. Streams: {streams}"
        assert probe_res["status"] == "ok"
        assert probe_res["copied"] == 2
        assert probe_res["delta_writes_order"] == ["z-task", "a-task"]

    @staticmethod
    def run_graph_lifecycle_scenario(
        tmp_path: Path,
        sqlite_available: bool,
        notebook_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Execute actual Cells 1, 4, 7, 10, 16, 19, 22, 28, 31, 32, 46, 57, 60, 94, and 96
        in a real Jupyter kernel under both SQLite-present and SQLite-absent modes.

        Verifies:
        - Cell 60 compiles `data_detective_graph` via `idd_graph.build_graph` with `structural_only=False`.
        - Graph topology has 15 user nodes, 33 edges, 21 conditional edges, and 70 State fields.
        - BEFORE Cell 94, the production graph is invoked under the exact `run_config` thread so the
          late cells operate on non-empty checkpoint history (including mid-run pending routing).
        - SQLite-present: Cell 94 migrates every seeded checkpoint into `checkpoints_demo.sqlite`
          (history ids, parents, `next`, task names and latest values compared against an
          independent SqliteSaver read); Cell 96 reads restored state/history while its connection
          is open and then closes it; production history is unchanged and production execution
          still works afterwards.
        - SQLite-absent (import blocked in-kernel, or package genuinely uninstalled): Cell 60 falls
          back to `MemorySaver`, Cells 94/96 skip, no demo DB is created, production history kept.
        - Representative no-key transitions on the real compiled graph:
          `initial_analysis -> supervisor -> data_cleaner -> supervisor -> analyst -> supervisor
          -> EMERGENCY_MSG (real Cell 57 node) -> analyst -> supervisor -> visualization
          -> viz_worker -> viz_join (real Cell 57 barrier) -> viz_evaluator -> supervisor
          -> report_orchestrator -> report_section_worker -> report_join (real Cell 57 barrier)
          -> report_packager -> file_writer -> supervisor -> FINISH -> __end__`.

        NOTE: in this scenario the supervisor node is a SCRIPTED replacement (the real Cell 46
        factory is called, but its returned callable is not used for routing). Real supervisor
        runtime/Router invocation is covered by `test_cell46_real_supervisor_runtime_routing_in_kernel`.
        """
        sqlite_blocker_code = """
import builtins
import importlib.abc
import importlib.util
import os
import sys

if os.environ.get("IDD_EXPECT_SQLITE_UNINSTALLED") == "1":
    assert (
        importlib.util.find_spec("langgraph.checkpoint.sqlite") is None
    ), "Expected langgraph-checkpoint-sqlite to be uninstalled from the environment!"
else:
    class _BlockSqliteCheckpointerFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "langgraph.checkpoint.sqlite" or fullname.startswith("langgraph.checkpoint.sqlite."):
                raise ModuleNotFoundError("Simulated absence of langgraph-checkpoint-sqlite for Issue #164")
            return None

    for _k in list(sys.modules):
        if _k == "langgraph.checkpoint.sqlite" or _k.startswith("langgraph.checkpoint.sqlite."):
            sys.modules.pop(_k, None)

    sys.meta_path.insert(0, _BlockSqliteCheckpointerFinder())
"""

        # Before Cell 60 runs, we keep the real Cell 57 `viz_join`, `report_join`, and
        # `emergency_correspondence_node` functions intact and provide deterministic no-key
        # node stubs for the LLM-calling nodes so we can both compile and invoke the real
        # Cell 60 `data_detective_graph` in the kernel without external API calls.
        deterministic_nodes_setup = f"""
{NO_KEY_CELL60_SETUP_SOURCE}
import os
import idd_tools
import idd_registry
import idd_core

WORKING_DIRECTORY = PathlibPath({str(tmp_path)!r})
idd_tools.WORKING_DIRECTORY = WORKING_DIRECTORY
idd_registry.WORKING_DIRECTORY = WORKING_DIRECTORY
idd_core.WORKING_DIRECTORY = WORKING_DIRECTORY
# Isolate Cell 60's relative 'checkpoints.sqlite' inside per-test WORKING_DIRECTORY
os.chdir(str(WORKING_DIRECTORY))

_TRANSITION_LOG = []

# Preserve real Cell 57 barrier & emergency nodes
_real_viz_join = viz_join
_real_report_join = report_join
_real_emergency_node = emergency_correspondence_node

def _logged_viz_join(state):
    _TRANSITION_LOG.append("viz_join")
    return _real_viz_join(state)

def _logged_report_join(state):
    _TRANSITION_LOG.append("report_join")
    return _real_report_join(state)

def _logged_emergency_node(state):
    _TRANSITION_LOG.append("EMERGENCY_MSG")
    return _real_emergency_node(state)

viz_join = _logged_viz_join
report_join = _logged_report_join
emergency_correspondence_node = _logged_emergency_node

_SUPERVISOR_SEQUENCE = [
    ("data_cleaner", "Clean the dataset"),
    ("analyst", "Run statistical analysis"),
    ("EMERGENCY_MSG", "Deliver clarification to analyst"),
    ("visualization", "Create visualizations"),
    ("report_orchestrator", "Draft report sections"),
    ("FINISH", "All tasks complete"),
]
_sup_step = {{"idx": 0}}

_orig_make_supervisor_node = globals()["make_supervisor_node"]

def make_supervisor_node(llms, members, prompt):
    # Verify real Cell 46 factory still binds canonical Router before returning deterministic test coordinator
    _real_sup = _orig_make_supervisor_node(llms, members, prompt)
    assert callable(_real_sup)
    def _deterministic_supervisor(state):
        idx = _sup_step["idx"]
        _sup_step["idx"] = idx + 1
        target, next_prompt = _SUPISOR_NEXT = _SUPERVISOR_SEQUENCE[min(idx, len(_SUPERVISOR_SEQUENCE) - 1)]
        _TRANSITION_LOG.append(f"supervisor->{{target}}")
        update = {{
            "next": target,
            "next_agent_prompt": next_prompt,
        }}
        if target == "EMERGENCY_MSG":
            update["supervisor_to_agent_msgs"] = [
                SendAgentMessage(
                    recipient="analyst",
                    message="Proceed with robust median imputation.",
                    delivery_status=False,
                    agent_obj_needs_recreated_bool=False,
                    is_message_critical=False,
                    immediate_emergency_reroute_to_recipient=False,
                    reply_msg_to_supervisor="Clarification queued",
                    finished_this_task=True,
                    expect_reply=False,
                )
            ]
        return update
    return _deterministic_supervisor

globals()["make_supervisor_node"] = make_supervisor_node

def initial_analysis_node(state):
    _TRANSITION_LOG.append("initial_analysis")
    return {{"initial_analysis_complete": True, "last_created_obj": "initial_description"}}

def data_cleaner_node(state):
    _TRANSITION_LOG.append("data_cleaner")
    return {{"data_cleaning_complete": True, "last_created_obj": "cleaning_metadata"}}

_analyst_calls = {{"count": 0}}
def analyst_node(state):
    _analyst_calls["count"] += 1
    _TRANSITION_LOG.append(f"analyst#{{_analyst_calls['count']}}")
    if _analyst_calls["count"] == 1:
        insights = AnalysisInsights(
            summary="Preliminary stats",
            correlation_insights="Strong correlation",
            anomaly_insights="Outliers detected",
            recommended_visualizations=[],
            recommended_next_steps=["Confirm imputation"],
            reply_msg_to_supervisor="Confirm median imputation?",
            finished_this_task=False,
            expect_reply=True,
        )
        return {{"analysis_insights": insights, "last_created_obj": "analysis_insights"}}
    return {{"analyst_complete": True, "last_created_obj": "analysis_insights"}}

def visualization_orchestrator(state):
    _TRANSITION_LOG.append("visualization")
    spec = VizSpec(
        title="Category Distribution",
        viz_type="bar",
        df_id="df_test_1",
        viz_instructions="Plot bar chart of categories",
        viz_id="viz_1",
        columns=["category"],
        x="category",
        y=None,
        hue=None,
        bins=None,
        agg=None,
        query=None,
        description="Bar chart of categories",
        limit=None,
        style="seaborn-v0_8-whitegrid",
        reply_msg_to_supervisor="Queued 1 chart",
        finished_this_task=True,
        expect_reply=False,
    )
    return {{"viz_tasks": ["Plot category distribution"], "viz_specs": [spec]}}

def viz_worker(state):
    _TRANSITION_LOG.append("viz_worker")
    dv = DataVisualization(
        path=str(WORKING_DIRECTORY / "chart1.png"),
        visualization_id="viz_1",
        visualization_type="bar",
        visualization_description="Balanced distribution",
        visualization_style="seaborn-v0_8-whitegrid",
        visualization_title="Category Distribution",
        reply_msg_to_supervisor="Chart created",
        finished_this_task=True,
        expect_reply=False,
    )
    return {{"viz_results": [dv]}}

def viz_evaluator_node(state):
    _TRANSITION_LOG.append("viz_evaluator")
    fb = VizFeedback(
        grade="acceptable",
        feedback="All charts passed quality checks",
        redo_list=[],
        reply_msg_to_supervisor="Charts approved",
        finished_this_task=True,
        expect_reply=False,
    )
    return {{"viz_feedback": fb, "viz_grade": "acceptable", "visualization_complete": True}}

def report_orchestrator(state):
    _TRANSITION_LOG.append("report_orchestrator")
    sec = SectionOutline(
        name="Executive Summary",
        section_num=1,
        description="Summarize key findings",
        goals=["Highlight top trends"],
        data_signals_needed={{"category": "counts"}},
        data_signals_available=["category"],
        expected_figures=[],
        word_target=300,
        reply_msg_to_supervisor="Section defined",
        finished_this_task=True,
        expect_reply=False,
    )
    outline = ReportOutline(
        title="IDD Real-Kernel Verification Report",
        name="Full Report",
        section_num=0,
        description="Comprehensive data report",
        goals=["Complete report"],
        data_signals_needed={{}},
        data_signals_available=[],
        expected_figures=[],
        word_target=600,
        sections=[sec],
        reply_msg_to_supervisor="Outline ready",
        finished_this_task=True,
        expect_reply=False,
    )
    return {{"report_outline": outline}}

def section_worker(state):
    _TRANSITION_LOG.append("report_section_worker")
    return {{"written_sections": ["# Executive Summary\\\\nAll invariants verified."]}}

def report_packager_node(state):
    _TRANSITION_LOG.append("report_packager")
    rep = ReportResults(
        pdf_report_path=str(WORKING_DIRECTORY / "final_report.pdf"),
        html_report_path=str(WORKING_DIRECTORY / "final_report.html"),
        markdown_report_path=str(WORKING_DIRECTORY / "final_report.md"),
        reply_msg_to_supervisor="Report packaged",
        finished_this_task=True,
        expect_reply=False,
    )
    return {{
        "report_results": rep,
        "report_generator_complete": True,
        "last_created_obj": "report_results",
    }}

def file_writer_node(state):
    _TRANSITION_LOG.append("file_writer")
    return {{"file_writer_complete": True, "last_created_obj": "list_of_files"}}

# Stub memory helper called inside Cell 57 viz_join so no embedding model call is needed
update_memory_with_kind = lambda *a, **kw: None
RUNTIME = MagicMock(run_dir=str(WORKING_DIRECTORY), artifacts_dir=str(WORKING_DIRECTORY))
"""

        post_cell60_capture = """
_cell60_graph_ref = data_detective_graph
_cell60_builder_ref = data_analysis_team_builder
_cell60_checkpointer_type = type(checkpointer).__name__
"""

        # Seed non-empty production history under the exact `run_config` thread used by
        # Cells 94/96 so migration/restore operate on real saved state, not empty history.
        seed_history_code = """
def _history_signature(graph, cfg):
    sig = []
    tups = list(graph.checkpointer.list(cfg))
    snaps = list(graph.get_state_history(cfg))
    assert len(tups) == len(snaps), f"Tuples vs snapshots count mismatch: {len(tups)} vs {len(snaps)}"
    for tup, snap in zip(tups, snaps):
        sig.append({
            "checkpoint_id": tup.config["configurable"]["checkpoint_id"],
            "parent_id": ((tup.parent_config or {}).get("configurable") or {}).get("checkpoint_id"),
            "next": list(snap.next),
            "step": (tup.metadata or {}).get("step"),
            "source": (tup.metadata or {}).get("source"),
            "tasks": sorted(t.name for t in snap.tasks),
            "pending_writes": list(tup.pending_writes or []),
        })
    return sig

_seed_cfg = {**run_config, "recursion_limit": 80}
_seed_final_state = data_detective_graph.invoke(
    {"user_prompt": "Analyze dataset (seed run)", "messages": [], "_config": _seed_cfg},
    config=_seed_cfg,
)
assert _seed_final_state.get("file_writer_complete") is True
_SEED_TRANSITIONS = list(_TRANSITION_LOG)
_src_history_before = _history_signature(data_detective_graph, run_config)
_src_latest_values = dict(data_detective_graph.get_state(run_config).values)
assert len(_src_history_before) >= 10, _src_history_before
assert any(h["next"] for h in _src_history_before), "Expected pending routing mid-history"
assert any(h["pending_writes"] for h in _src_history_before), "Expected pending writes in seeded history"

# Reset scripted stubs so the post-restore production invocation replays the full route.
_sup_step["idx"] = 0
_analyst_calls["count"] = 0
_TRANSITION_LOG.clear()
"""

        verify_and_invoke_code = f"""
import json
from langgraph.checkpoint.memory import MemorySaver
import idd_graph

assert build_graph is idd_graph.build_graph
assert route_from_supervisor is idd_graph.route_from_supervisor
assert route_to_writer is idd_graph.route_to_writer
assert write_output_to_file is idd_graph.write_output_to_file

# 1. Verify single production graph lifecycle preserved across Cells 60, 94, and 96
assert data_detective_graph is _cell60_graph_ref, "Production data_detective_graph was rebound after Cell 60!"
assert data_analysis_team_builder is _cell60_builder_ref
assert not data_detective_graph.is_structural_only

# Production history for the seeded thread is unchanged by the late cells.
_src_history_after = _history_signature(data_detective_graph, run_config)
assert _src_history_after == _src_history_before, "Late cells mutated production checkpoint history!"

_migrated_count_out = None
_restored_count_out = None
if {sqlite_available!r}:
    import sqlite3 as _verify_sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver as _VerifySqliteSaver

    assert _cell60_checkpointer_type == "SqliteSaver"
    _demo_db = WORKING_DIRECTORY / "checkpoints_demo.sqlite"
    assert _demo_db.exists()

    # Cell 94: every seeded checkpoint migrated.
    assert _migrated_checkpoint_count == len(_src_history_before), (
        _migrated_checkpoint_count, len(_src_history_before)
    )
    _migrated_count_out = _migrated_checkpoint_count

    # Independent read of the demo DB through a fresh real SqliteSaver connection.
    with _VerifySqliteSaver.from_conn_string(str(_demo_db)) as _verify_cp:
        _verify_graph = data_analysis_team_builder.compile(checkpointer=_verify_cp)
        _dst_history = _history_signature(_verify_graph, run_config)
        _dst_latest_values = dict(_verify_graph.get_state(run_config).values)

        # Order-sensitive replay assertion on migrated checkpoint delta channel history
        _channels_with_writes = set()
        for _h in _src_history_before:
            for _pw in _h["pending_writes"]:
                _channels_with_writes.add(_pw[1])
        assert _channels_with_writes, "Expected pending writes in seeded history"
        _src_delta = data_detective_graph.checkpointer.get_delta_channel_history(
            config=run_config, channels=list(_channels_with_writes)
        )
        _dst_delta = _verify_cp.get_delta_channel_history(
            config=run_config, channels=list(_channels_with_writes)
        )
        assert _dst_delta == _src_delta, f"Delta channel history mismatch: {{_dst_delta}} vs {{_src_delta}}"

    assert _dst_history == _src_history_before, "Migrated history differs from source history"
    assert set(_dst_latest_values) == set(_src_latest_values)
    for _k, _v in _src_latest_values.items():
        assert _dst_latest_values[_k] == _v, f"Migrated value mismatch for {{_k!r}}"

    # Cell 96: restored state was read while its connection was valid.
    assert _restored_graph is not data_detective_graph
    assert _restored_history_ids == [h["checkpoint_id"] for h in _src_history_before]
    assert _restored_next == tuple(_src_history_before[0]["next"])
    assert dict(_restored_snapshot.values) == _src_latest_values
    _restored_count_out = len(_restored_history_ids)

    # Cell 96 closed its demo connection; the production connection stays open and usable.
    try:
        _restore_conn.execute("SELECT 1")
        raise AssertionError("Cell 96 left its restore connection open")
    except _verify_sqlite3.ProgrammingError:
        pass
    _prod_rows = _cp_conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    assert _prod_rows >= len(_src_history_before)
else:
    assert isinstance(checkpointer, MemorySaver)
    assert _cell60_checkpointer_type in ("MemorySaver", "InMemorySaver")
    assert "_migrated_checkpoint_count" not in globals()
    assert "_restored_graph" not in globals()
    assert not (WORKING_DIRECTORY / "checkpoints_demo.sqlite").exists()

# 2. Verify graph topology (15 user nodes, 33 edges, 21 conditional edges)
drawable = data_detective_graph.get_graph()
user_nodes = sorted(n for n in drawable.nodes if n not in ("__start__", "__end__"))
expected_nodes = sorted([
    "initial_analysis",
    "supervisor",
    "data_cleaner",
    "analyst",
    "visualization",
    "viz_worker",
    "viz_join",
    "viz_evaluator",
    "report_orchestrator",
    "report_section_worker",
    "report_join",
    "report_packager",
    "file_writer",
    "EMERGENCY_MSG",
    "FINISH",
])
assert user_nodes == expected_nodes, f"Node mismatch: {{user_nodes}} vs {{expected_nodes}}"
total_edges = len(drawable.edges)
conditional_edges = sum(1 for e in drawable.edges if getattr(e, "conditional", False))
assert total_edges == 33, f"Expected 33 edges, got {{total_edges}}"
assert conditional_edges == 21, f"Expected 21 conditional edges, got {{conditional_edges}}"

# 3. Production execution still works AFTER Cells 94/96 (distinct prompt avoids node-cache hits)
invoke_cfg = {{
    "configurable": {{"thread_id": "issue-164-e2e-transition-thread"}},
    "recursion_limit": 80,
}}
final_state = data_detective_graph.invoke(
    {{"user_prompt": "Analyze dataset (post-restore run)", "messages": [], "_config": invoke_cfg}},
    config=invoke_cfg,
)

required_milestones = [
    "initial_analysis",
    "supervisor->data_cleaner",
    "data_cleaner",
    "supervisor->analyst",
    "analyst#1",
    "supervisor->EMERGENCY_MSG",
    "EMERGENCY_MSG",
    "analyst#2",
    "supervisor->visualization",
    "visualization",
    "viz_worker",
    "viz_join",
    "viz_evaluator",
    "supervisor->report_orchestrator",
    "report_orchestrator",
    "report_section_worker",
    "report_join",
    "report_packager",
    "file_writer",
]
for milestone in required_milestones:
    assert milestone in _TRANSITION_LOG, f"Missing milestone {{milestone!r}} in {{_TRANSITION_LOG}}"
    assert milestone in _SEED_TRANSITIONS, f"Seed run missing milestone {{milestone!r}}"

assert final_state.get("file_writer_complete") is True
assert final_state.get("visualization_complete") is True
assert final_state.get("report_generator_complete") is True
assert "All invariants verified." in (final_state.get("report_draft") or "")
assert final_state.get("report_results") is not None
assert final_state["report_results"].markdown_report_path.endswith("final_report.md")
assert len(_history_signature(data_detective_graph, invoke_cfg)) >= 10
assert _history_signature(data_detective_graph, run_config) == _src_history_before

print({PROBE_PREFIX!r} + json.dumps({{
    "status": "ok",
    "sqlite_available": {sqlite_available!r},
    "checkpointer_type": _cell60_checkpointer_type,
    "node_count": len(user_nodes),
    "edge_count": total_edges,
    "conditional_edge_count": conditional_edges,
    "seeded_checkpoints": len(_src_history_before),
    "migrated_checkpoints": _migrated_count_out,
    "restored_checkpoints": _restored_count_out,
    "transitions": _TRANSITION_LOG,
}}))
"""

        cell_specs: List[CellSpec] = [
            1,
            4,
            7,
            10,
            16,
            19,
            22,
            28,
            31,
            32,
            46,
            57,
        ]
        if not sqlite_available:
            cell_specs.append(("block_sqlite_checkpointer", sqlite_blocker_code))
        cell_specs.extend(
            [
                ("setup_cell60_no_key_stubs", deterministic_nodes_setup),
                60,
                ("capture_cell60_refs", post_cell60_capture),
                ("seed_run_config_history", seed_history_code),
                94,
                96,
                ("verify_and_invoke_graph", verify_and_invoke_code),
            ]
        )

        executed_nb, probes, streams = execute_cells_in_real_kernel(
            cell_specs,
            notebook_dict=notebook_dict,
            cwd=REPO_ROOT,
            work_dir=tmp_path / "kernel_cwd",
        )
        res = probes.get("verify_and_invoke_graph")
        assert res is not None, f"Missing graph verification probe. Streams: {streams}"
        streams_by_key = {
            key: stream_for(executed_nb, streams, key)
            for key in ("cell_60", "cell_94", "cell_96", "seed_run_config_history")
        }
        return res, streams_by_key

    @staticmethod
    def run_real_supervisor_scenario(
        tmp_path: Path, notebook_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke the REAL callable returned by generated Cell 46 `make_supervisor_node`
        inside a real LangGraph runtime (so `get_store()`/config plumbing is genuine),
        with only the chat-model boundary stubbed.

        - `_KernelStructuredChatModel.with_structured_output(schema, **kw)` records the exact
          schema object and kwargs the supervisor binds, and returns a deterministic
          schema-valid response (Plan / ToDoList / Router). Unknown schemas raise.
        - Model-driven path: no shortcut applies, so the supervisor runs planning, to-do and
          Router routing; the stub Router chooses `analyst`, which differs from the state-based
          `_routing_parse_fallback` choice (`initial_analysis`), so a broken Router binding
          or masked routing failure cannot pass.
        - Deterministic shortcut path (`initial_analysis_complete=True` -> `data_cleaner`) and
          the W4 turn-cap path (`FINISH`) exercise the real wrapper/body without model calls.
        - Mocked: chat models (8 role slots) and Cell 37 memory/embedding helpers
          (`enhanced_mem_text`, `update_memory_with_kind`, `query_embed_func`), which require
          OpenAI embeddings. Everything else (Cells 1-46 definitions, prompts from Cell 28,
          canonical Router/Plan/State, Command construction, W4 wrapper) is real.
        """
        supervisor_probe = f"""
import contextlib
import io
import json
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
import idd_graph

_STRUCTURED_BINDINGS = []
_STRUCTURED_INVOCATIONS = []
_MEMORY_WRITES = []

def _kernel_plan():
    return Plan(
        plan_title="Kernel Plan",
        plan_summary="Analyze the dataset after the initial description.",
        plan_steps=[
            PlanStep(
                step_number=1,
                step_name="Analyze",
                step_description="Run statistical analysis",
                is_step_complete=False,
                plan_version=1,
                reply_msg_to_supervisor="",
                finished_this_task=False,
                expect_reply=False,
            )
        ],
        plan_version=1,
        reply_msg_to_supervisor="",
        finished_this_task=False,
        expect_reply=False,
    )

def _kernel_todo():
    return ToDoList(
        to_do_list=["Run statistical analysis"],
        reply_msg_to_supervisor="",
        finished_this_task=False,
        expect_reply=False,
    )

def _kernel_router():
    return idd_graph.Router(
        next="analyst",
        next_agent_prompt="KERNEL_ROUTER_PROMPT: analyze the dataset",
        next_agent_metadata=None,
        reply_msg_to_supervisor="",
        finished_this_task=True,
        expect_reply=False,
    )

_SCHEMA_RESPONSES = [(Plan, _kernel_plan), (ToDoList, _kernel_todo), (idd_graph.Router, _kernel_router)]

class _KernelStructuredChatModel:
    \"\"\"No-key stub for the chat-model boundary used by the real Cell 46 supervisor.\"\"\"

    def __init__(self, role):
        self.role = role

    def with_structured_output(self, schema, **kwargs):
        _STRUCTURED_BINDINGS.append({{"role": self.role, "schema": schema, "kwargs": dict(kwargs)}})
        factory = next((f for s, f in _SCHEMA_RESPONSES if s is schema), None)

        def _respond(prompt_value):
            _STRUCTURED_INVOCATIONS.append({{"role": self.role, "schema": getattr(schema, "__name__", repr(schema))}})
            if factory is None:
                raise RuntimeError(f"Unexpected structured schema for role {{self.role}}: {{schema!r}}")
            return factory()

        return RunnableLambda(_respond)

_ROLES = ["big_picture", "router", "reply", "plan", "replan", "progress", "todo", "conversational"]
_kernel_llms = [_KernelStructuredChatModel(r) for r in _ROLES]

# Cell 37 memory/embedding helpers need OpenAI embeddings; stub at that boundary only.
enhanced_mem_text = lambda *a, **kw: ""
query_embed_func = lambda *a, **kw: [0.0]
def update_memory_with_kind(state, config, kind, store, text=""):
    _MEMORY_WRITES.append(kind)
in_memory_store = InMemoryStore()

real_supervisor = make_supervisor_node(
    _kernel_llms,
    ["initial_analysis", "data_cleaner", "analyst", "file_writer", "visualization", "report_orchestrator"],
    "Analyze dataset",
)
assert callable(real_supervisor)
assert getattr(real_supervisor, "__wrapped__", None) is not None, "W4 wrapper missing"

_CAPTURED = []
def _probe_node(state, config):
    _CAPTURED.append(real_supervisor(state, config))
    return {{}}

_g = StateGraph(State)
_g.add_node("probe", _probe_node)
_g.add_edge(START, "probe")
_g.add_edge("probe", END)
_probe_graph = _g.compile(store=InMemoryStore())

def _run(state):
    _CAPTURED.clear()
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        _probe_graph.invoke(state)
    assert len(_CAPTURED) == 1
    return _CAPTURED[0], _buf.getvalue()

_base_plan = Plan(
    plan_title="Existing Plan",
    plan_summary="Existing plan summary",
    plan_steps=[],
    plan_version=0,
    reply_msg_to_supervisor="",
    finished_this_task=False,
    expect_reply=False,
)

# A. Model-driven Router path (no shortcut applies).
_STRUCTURED_BINDINGS.clear(); _STRUCTURED_INVOCATIONS.clear(); _MEMORY_WRITES.clear()
cmd_model, out_model = _run({{
    "user_prompt": "Analyze dataset",
    "messages": [HumanMessage(content="Initial analysis finished.", name="initial_analysis")],
    "last_agent_id": "initial_analysis",
    "last_created_obj": "kernel_none",
    "current_plan": _base_plan,
}})
assert isinstance(cmd_model, Command), type(cmd_model)
assert cmd_model.goto == "analyst", (cmd_model.goto, out_model)
assert cmd_model.update["next"] == "analyst"
assert cmd_model.update["next_agent_prompt"].startswith("KERNEL_ROUTER_PROMPT")
assert cmd_model.update["_supervisor_turn_count"] == 1
assert cmd_model.update["current_plan"].plan_title == "Kernel Plan"
assert "ROUTING FALLBACK" not in out_model, out_model
assert "WARNING" not in out_model and "ERROR" not in out_model, out_model

_router_bindings = [b for b in _STRUCTURED_BINDINGS if b["role"] == "router"]
assert len(_router_bindings) == 1, _STRUCTURED_BINDINGS
assert _router_bindings[0]["schema"] is idd_graph.Router
assert _router_bindings[0]["kwargs"] == {{"strict": True, "method": "json_schema"}}
_plan_bindings = [b for b in _STRUCTURED_BINDINGS if b["role"] == "plan"]
assert len(_plan_bindings) == 1 and _plan_bindings[0]["schema"] is Plan
_todo_bindings = [b for b in _STRUCTURED_BINDINGS if b["role"] == "todo"]
assert len(_todo_bindings) == 1 and _todo_bindings[0]["schema"] is ToDoList
assert [i["schema"] for i in _STRUCTURED_INVOCATIONS] == ["Plan", "ToDoList", "Router"], _STRUCTURED_INVOCATIONS
assert _MEMORY_WRITES == ["plans", "todos", "routes"], _MEMORY_WRITES
_model_path_invocations = [i["schema"] for i in _STRUCTURED_INVOCATIONS]

# B. Deterministic shortcut path: no model invocation.
_STRUCTURED_INVOCATIONS.clear()
cmd_short, _ = _run({{
    "user_prompt": "Analyze dataset",
    "messages": [],
    "initial_analysis_complete": True,
}})
assert cmd_short.goto == "data_cleaner"
assert cmd_short.update["next"] == "data_cleaner"
assert cmd_short.update["_supervisor_turn_count"] == 1
assert _STRUCTURED_INVOCATIONS == []

# C. W4 global turn cap forces FINISH.
cmd_cap, out_cap = _run({{"user_prompt": "Analyze dataset", "messages": [], "_supervisor_turn_count": 30}})
assert cmd_cap.goto == "FINISH"
assert "[W4-SUPLIMIT]" in out_cap

print({PROBE_PREFIX!r} + json.dumps({{
    "status": "ok",
    "model_goto": cmd_model.goto,
    "shortcut_goto": cmd_short.goto,
    "cap_goto": cmd_cap.goto,
    "router_schema": f"{{idd_graph.Router.__module__}}.{{idd_graph.Router.__name__}}",
    "structured_invocations": _model_path_invocations,
}}))
"""
        _, probes, streams = execute_cells_in_real_kernel(
            [
                1,
                4,
                7,
                10,
                16,
                19,
                22,
                28,
                31,
                32,
                46,
                ("real_supervisor_probe", supervisor_probe),
            ],
            notebook_dict=notebook_dict,
            work_dir=tmp_path / "kernel_cwd",
        )
        res = probes.get("real_supervisor_probe")
        assert res is not None, f"Missing supervisor probe output. Streams: {streams}"
        return res

    def test_cell46_real_supervisor_runtime_routing_in_kernel(self, tmp_path: Path):
        """Real Cell 46 supervisor executes and routes via canonical Router (Codex P1-1)."""
        res = self.run_real_supervisor_scenario(tmp_path)
        assert res["status"] == "ok"
        assert res["model_goto"] == "analyst"
        assert res["shortcut_goto"] == "data_cleaner"
        assert res["cap_goto"] == "FINISH"


# ===========================================================================
# Stage 4: Security, Adversarial Fail-Closed Startup & Negative Controls
# ===========================================================================


class TestRealKernelSecurityAndNegativeControls:
    """
    Prove in real Jupyter kernels that startup fails closed and rejected code is
    NEVER subsequently imported or executed across all 7 Stage 4 scenarios, plus
    deliberate notebook mutation negative controls.
    """

    def test_stage4_1_missing_canonical_file_fails_closed_before_later_cells(
        self, tmp_path: Path
    ):
        """1. Missing canonical file aborts Cell 4 and prevents subsequent cell execution."""
        sandbox = _create_repo_sandbox(tmp_path)
        (sandbox / "idd_models.py").unlink()
        canary_file = tmp_path / "post_cell4_canary.txt"

        post_cell = f"""
from pathlib import Path
Path({str(canary_file)!r}).write_text("SHOULD_NOT_RUN", encoding="utf-8")
"""
        with pytest.raises(CellExecutionError) as exc_info:
            execute_cells_in_real_kernel(
                [1, 4, ("post_canary", post_cell)],
                cwd=sandbox,
            )
        assert "FileNotFoundError" in str(exc_info.value)
        assert "Canonical module 'idd_models.py' missing" in str(exc_info.value)
        assert (
            not canary_file.exists()
        ), "Subsequent cell executed after missing canonical file!"

    def test_stage4_2_tampered_manifest_digest_rejects_before_module_import(
        self, tmp_path: Path
    ):
        """2. Tampered manifest digest aborts Cell 4 and never imports or executes the tampered module."""
        sandbox = _create_repo_sandbox(tmp_path)
        tampered_flag = tmp_path / "tampered_module_executed.flag"
        post_canary = tmp_path / "post_cell4_canary.flag"

        # Append malicious payload to idd_models.py AND tamper the manifest digest
        with open(sandbox / "idd_models.py", "a", encoding="utf-8") as f:
            f.write(
                f"\nimport pathlib\npathlib.Path({str(tampered_flag)!r}).write_text('EXECUTED', encoding='utf-8')\n"
            )
        manifest_path = sandbox / "idd_manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["files"]["idd_models.py"] = "0" * 64
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        post_cell = f"""
import idd_models
from pathlib import Path
Path({str(post_canary)!r}).write_text("SHOULD_NOT_RUN", encoding="utf-8")
"""
        with pytest.raises(CellExecutionError) as exc_info:
            execute_cells_in_real_kernel(
                [1, 4, ("post_import", post_cell)],
                cwd=sandbox,
            )
        assert "RuntimeError" in str(exc_info.value)
        assert "Digest mismatch for 'idd_models.py'" in str(exc_info.value)
        assert (
            not tampered_flag.exists()
        ), "Tampered idd_models.py was imported/executed!"
        assert (
            not post_canary.exists()
        ), "Subsequent cell executed after manifest mismatch!"

    def test_stage4_3_modified_idd_bootstrap_rejected_before_import(
        self, tmp_path: Path
    ):
        """3. Modified idd_bootstrap.py is caught by Cell 4 pre-import SHA-256 check and never imported."""
        sandbox = _create_repo_sandbox(tmp_path)
        bootstrap_flag = tmp_path / "tampered_bootstrap_executed.flag"
        (sandbox / "idd_bootstrap.py").write_text(
            f"import pathlib\npathlib.Path({str(bootstrap_flag)!r}).write_text('PWNED', encoding='utf-8')\n",
            encoding="utf-8",
        )

        with pytest.raises(CellExecutionError) as exc_info:
            execute_cells_in_real_kernel(
                [1, 4],
                cwd=sandbox,
            )
        assert "RuntimeError" in str(exc_info.value)
        assert (
            "CP4 Bootstrap Integrity Error: idd_bootstrap.py SHA-256 mismatch"
            in str(exc_info.value)
        )
        assert (
            not bootstrap_flag.exists()
        ), "Modified idd_bootstrap.py was executed instead of rejected pre-import!"

    def test_stage4_4_already_imported_canonical_module_from_unrelated_checkout_rejected(
        self, tmp_path: Path
    ):
        """4. Pre-loaded canonical module in sys.modules from an unrelated checkout fails closed in Cell 4."""
        foreign_dir = tmp_path / "unrelated_checkout"
        foreign_dir.mkdir(parents=True, exist_ok=True)
        foreign_file = foreign_dir / "idd_models.py"
        foreign_file.write_text("# unrelated checkout\n", encoding="utf-8")
        canary_file = tmp_path / "after_foreign_module.flag"

        pre_inject = f"""
import sys, types
_foreign = types.ModuleType("idd_models")
_foreign.__file__ = {str(foreign_file)!r}
sys.modules["idd_models"] = _foreign
"""
        post_cell = f"""
from pathlib import Path
Path({str(canary_file)!r}).write_text("SHOULD_NOT_RUN", encoding="utf-8")
"""
        with pytest.raises(CellExecutionError) as exc_info:
            execute_cells_in_real_kernel(
                [("inject_foreign_mod", pre_inject), 1, 4, ("post_canary", post_cell)],
                cwd=REPO_ROOT,
            )
        assert "RuntimeError" in str(exc_info.value)
        assert "Loaded module 'idd_models' origin" in str(exc_info.value)
        assert not canary_file.exists()

    def test_stage4_5_competing_module_locations_on_sys_path(self, tmp_path: Path):
        """
        5. Competing module locations on sys.path:
           - Part A: When competing directory is at sys.path[0], Cell 4 promotes verified repo_root
             ahead of competing path so poisoned competing module is never imported or executed.
           - Part B: If a meta_path finder resolves a canonical module spec to a competing checkout,
             `verify_module_origins` fails closed with RuntimeError before importing the module.
        """
        competing_dir = tmp_path / "competing_sys_path"
        competing_dir.mkdir(parents=True, exist_ok=True)
        poison_flag = tmp_path / "competing_module_ran.flag"
        competing_models = competing_dir / "idd_models.py"
        competing_models.write_text(
            f"import pathlib\npathlib.Path({str(poison_flag)!r}).write_text('RAN', encoding='utf-8')\n"
            "raise RuntimeError('Competing idd_models executed!')\n",
            encoding="utf-8",
        )

        # Part A: competing_dir placed at sys.path[0] before Cell 4
        prepend_competing = f"""
import sys
sys.path.insert(0, {str(competing_dir)!r})
"""
        verify_canonical_loaded = f"""
import json
from pathlib import Path
import idd_models
assert Path(idd_models.__file__).resolve() == (Path({str(REPO_ROOT)!r}).resolve() / "idd_models.py").resolve()
print({PROBE_PREFIX!r} + json.dumps({{"sys_path_0": str(Path(sys.path[0]).resolve())}}))
"""
        _, probes, _ = execute_cells_in_real_kernel(
            [
                ("prepend_competing", prepend_competing),
                1,
                4,
                16,
                ("verify_canonical", verify_canonical_loaded),
            ],
            cwd=REPO_ROOT,
        )
        assert (
            Path(probes["verify_canonical"]["sys_path_0"]).resolve()
            == REPO_ROOT.resolve()
        )
        assert not poison_flag.exists(), "Competing module on sys.path was executed!"

        # Part B: MetaPathFinder resolves `idd_models` spec to competing_models -> fails closed before import
        hijack_finder = f"""
import importlib.abc
import importlib.machinery
import sys

class _CompetingSpecFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "idd_models":
            return importlib.machinery.ModuleSpec(fullname, None, origin={str(competing_models)!r})
        return None

sys.meta_path.insert(0, _CompetingSpecFinder())
"""
        with pytest.raises(CellExecutionError) as exc_info:
            execute_cells_in_real_kernel(
                [("hijack_finder", hijack_finder), 1, 4, 16],
                cwd=REPO_ROOT,
            )
        assert "RuntimeError" in str(exc_info.value)
        assert (
            "Import spec origin for canonical module 'idd_models' resolved to"
            in str(exc_info.value)
        )
        assert (
            not poison_flag.exists()
        ), "Competing module executed despite origin mismatch!"

    def test_stage4_6_missing_cloned_repository_in_colab_style_environment_fails_closed(
        self, tmp_path: Path
    ):
        """6. Missing cloned repository in a Colab-style environment fails closed in Cell 4 and idd_bootstrap."""
        empty_colab_cwd = tmp_path / "colab_empty_workspace"
        empty_colab_cwd.mkdir(parents=True, exist_ok=True)
        canary_file = tmp_path / "colab_post_cell4.flag"

        simulate_colab_no_clone = f"""
import sys, types
import pytest

# Stub google.colab.userdata so Cell 4 key lookup and idd_bootstrap Colab detection work
google_mod = types.ModuleType("google")
colab_mod = types.ModuleType("google.colab")
userdata_mod = types.ModuleType("google.colab.userdata")
userdata_mod.get = lambda key: "dummy-colab-key"
colab_mod.userdata = userdata_mod
google_mod.colab = colab_mod
sys.modules["google"] = google_mod
sys.modules["google.colab"] = colab_mod
sys.modules["google.colab.userdata"] = userdata_mod

# Verify idd_bootstrap.bootstrap_environment fails closed when running in Colab without a cloned repository
sys.path.insert(0, {str(REPO_ROOT)!r})
import idd_bootstrap
sys.path.remove({str(REPO_ROOT)!r})
with pytest.raises(RuntimeError, match="Running in Google Colab without an explicitly cloned repository"):
    idd_bootstrap.bootstrap_environment(start_dir={str(empty_colab_cwd)!r})
sys.modules.pop("idd_bootstrap", None)

# Make `'google.colab' in str(get_ipython())` evaluate to True for Cell 4
_orig_get_ipython = get_ipython
def get_ipython():
    return "<google.colab._shell.Shell object>"
"""
        post_cell = f"""
from pathlib import Path
Path({str(canary_file)!r}).write_text("SHOULD_NOT_RUN", encoding="utf-8")
"""
        with pytest.raises(CellExecutionError) as exc_info:
            execute_cells_in_real_kernel(
                [
                    ("simulate_colab", simulate_colab_no_clone),
                    1,
                    4,
                    ("post_canary", post_cell),
                ],
                cwd=empty_colab_cwd,
            )
        assert "RuntimeError" in str(exc_info.value)
        assert (
            "CP4 Bootstrap Integrity Error: idd_manifest.json not found in current or parent directories."
            in str(exc_info.value)
        )
        assert (
            not canary_file.exists()
        ), "Subsequent cell ran despite missing Colab clone!"

    def test_stage4_7_corrupted_or_unexpected_patcher_source_anchors_fail_closed_in_kernel(
        self,
    ):
        """7. Corrupted or unexpected patcher source anchors fail closed with RequiredPatchError in kernel."""
        patcher_probe = f"""
import json
import pytest
import _patch_notebook
from _patch_notebook import RequiredPatchError, replace_required, replace_required_regex, verify_cp4_input_cell

valid_cell10 = "class MyChatOpenai:\\n    pass\\ndef _construct_responses_api_payload():\\n    pass\\n"

# 1. Unexpected class drift
with pytest.raises(RequiredPatchError, match="input class inventory drifted"):
    verify_cp4_input_cell(
        valid_cell10 + "\\nclass RogueAdapter:\\n    pass\\n",
        cell_idx=10,
        patch_id="KERNEL-ANCHOR-CLASS",
        expected_classes={{"MyChatOpenai"}},
        expected_functions={{"_construct_responses_api_payload"}},
    )

# 2. Missing required structural anchor token
with pytest.raises(RequiredPatchError, match="missing required input structural anchor"):
    verify_cp4_input_cell(
        valid_cell10,
        cell_idx=10,
        patch_id="KERNEL-ANCHOR-TOKEN",
        expected_classes={{"MyChatOpenai"}},
        expected_functions={{"_construct_responses_api_payload"}},
        required_anchors=("# NON_EXISTENT_ANCHOR_SENTINEL",),
    )

# 3. Syntax corruption in input cell
with pytest.raises(RequiredPatchError, match="CP4 input cell syntax error"):
    verify_cp4_input_cell(
        "def broken_syntax(:\\n",
        cell_idx=10,
        patch_id="KERNEL-SYNTAX",
    )

# 4. Corrupted replace_required & replace_required_regex anchors
with pytest.raises(RequiredPatchError, match="required anchor count was 0"):
    replace_required("cell body", "missing_anchor", "new_body", patch_id="KERNEL-REQ-0")

with pytest.raises(RequiredPatchError, match="required structural anchor count was 2"):
    replace_required_regex("dup dup", r"dup", "single", patch_id="KERNEL-REGEX-2")

print({PROBE_PREFIX!r} + json.dumps({{"patcher_checks": 5}}))
"""
        _, probes, _ = execute_cells_in_real_kernel(
            [1, 4, ("patcher_probe", patcher_probe)]
        )
        assert probes["patcher_probe"]["patcher_checks"] == 5

    def test_negative_controls_deliberately_broken_notebook_cells_fail_kernel_verification(
        self, tmp_path: Path, checkout_checkpoint_canary
    ):
        """
        Include meaningful negative controls that deliberately introduce broken notebook
        behavior in-memory and demonstrate that real-kernel verification fails closed.

        Runs under `checkout_checkpoint_canary`: the expected-failure Cell 60/94/96 path must
        leave a pre-existing checkout `checkpoints.sqlite` byte-for-byte unchanged.
        """
        base_nb = load_patched_notebook()

        # Negative Control A: Deliberately mutate Cell 46 to define a local `class Router`
        # shadowing `idd_graph.Router` inside `make_supervisor_node`.
        nb_bad_router = copy.deepcopy(base_nb)
        src46 = "".join(nb_bad_router["cells"][46]["source"])
        mutated46 = src46.replace(
            "from idd_graph import Router",
            "class Router(BaseNoExtrasModel):\n        next: str = 'FINISH'\n",
        )
        assert mutated46 != src46, "Negative control A mutation anchor not found"
        nb_bad_router["cells"][46]["source"] = [mutated46]
        router_check_cell = """
import inspect
from unittest.mock import MagicMock
import idd_graph

fn = make_supervisor_node([MagicMock() for _ in range(8)], ["analyst"], "Prompt")
inner = [
    c.cell_contents
    for c in (fn.__closure__ or ())
    if callable(c.cell_contents) and getattr(c.cell_contents, "__name__", "") == "supervisor_node"
][0]
assert inspect.getclosurevars(inner).nonlocals.get("Router") is idd_graph.Router, (
    "Negative control caught shadowed Router in Cell 46!"
)
"""
        with pytest.raises(CellExecutionError) as exc_router:
            execute_cells_in_real_kernel(
                [1, 4, 7, 10, 16, 19, 22, 28, 46, ("check_router", router_check_cell)],
                notebook_dict=nb_bad_router,
                work_dir=tmp_path / "control_a_cwd",
            )
        assert "Negative control caught shadowed Router in Cell 46!" in str(
            exc_router.value
        )

        # Negative Control B: Deliberately mutate Cell 96 to overwrite `data_detective_graph`
        nb_bad_cell96 = copy.deepcopy(base_nb)
        src96 = "".join(nb_bad_cell96["cells"][96]["source"])
        nb_bad_cell96["cells"][96]["source"] = [
            src96 + "\ndata_detective_graph = object()\n"
        ]
        setup_min_60 = f"""
{NO_KEY_CELL60_SETUP_SOURCE}
for _n in [
    "initial_analysis_node", "data_cleaner_node", "analyst_node",
    "visualization_orchestrator", "viz_worker", "viz_join", "viz_evaluator_node",
    "report_orchestrator", "section_worker", "report_join", "report_packager_node",
    "file_writer_node", "emergency_correspondence_node"
]:
    globals()[_n] = lambda state: {{}}
"""
        capture_ref = "_orig_graph = data_detective_graph"
        assert_preserved = """
assert data_detective_graph is _orig_graph, "Negative control caught Cell 96 overwriting data_detective_graph!"
"""
        control_b_cwd = tmp_path / "control_b_cwd"
        with pytest.raises(CellExecutionError) as exc_cell96:
            execute_cells_in_real_kernel(
                [
                    1,
                    4,
                    7,
                    10,
                    16,
                    19,
                    22,
                    28,
                    46,
                    ("setup_min_60", setup_min_60),
                    60,
                    ("capture_ref", capture_ref),
                    94,
                    96,
                    ("assert_preserved", assert_preserved),
                ],
                notebook_dict=nb_bad_cell96,
                work_dir=control_b_cwd,
            )
        assert (
            "Negative control caught Cell 96 overwriting data_detective_graph!"
            in str(exc_cell96.value)
        )
        # Cell 60's relative checkpoint DB landed in the disposable work_dir, not the checkout.
        assert (control_b_cwd / "checkpoints.sqlite").exists()

    def test_harness_refuses_cell60_without_isolated_work_dir(self):
        """Cell 60 opens a relative checkpoints.sqlite; the harness must refuse checkout cwd."""
        with pytest.raises(ValueError, match="requires an isolated work_dir"):
            execute_cells_in_real_kernel([1, 4, 60])

    @pytest.mark.parametrize(
        "mutation",
        ["raise_at_supervisor_entry", "router_binding_replaced_with_plan"],
    )
    def test_negative_controls_real_supervisor_mutations_fail(
        self, tmp_path: Path, mutation: str
    ):
        """Codex P1-1 required regression: broken Cell 46 supervisor must fail the runtime gate."""
        nb = copy.deepcopy(load_patched_notebook())
        src46 = "".join(nb["cells"][46]["source"])
        if mutation == "raise_at_supervisor_entry":
            anchor = "    def supervisor_node(state: State, config: RunnableConfig):\n"
            replacement = (
                anchor
                + "        raise RuntimeError('REVIEW_MUTATION_SUPERVISOR_BODY_NEVER_WORKS')\n"
            )
            expected = "REVIEW_MUTATION_SUPERVISOR_BODY_NEVER_WORKS"
        else:
            anchor = (
                'routing_llm = updated_routing_prompt | models["router"]'
                ".with_structured_output(Router,"
            )
            replacement = (
                'routing_llm = updated_routing_prompt | models["router"]'
                ".with_structured_output(Plan,"
            )
            # The masked parse failure falls back to state routing (initial_analysis).
            expected = "ROUTING FALLBACK"
        assert src46.count(anchor) == 1, f"Mutation anchor for {mutation} not unique"
        nb["cells"][46]["source"] = [src46.replace(anchor, replacement)]
        with pytest.raises(CellExecutionError) as exc_info:
            TestRealKernelNotebookExecution.run_real_supervisor_scenario(
                tmp_path, notebook_dict=nb
            )
        assert expected in str(exc_info.value), str(exc_info.value)[-2000:]

    @pytest.mark.parametrize(
        "mutation",
        [
            "migration_raises_and_is_swallowed",
            "noop_migrate_thread",
            "restore_compiled_without_checkpointer",
            "replace_put_writes_with_pass",
        ],
    )
    def test_negative_controls_checkpoint_migration_and_restore_mutations_fail(
        self, tmp_path: Path, mutation: str, checkout_checkpoint_canary
    ):
        """Codex P1-2 and P2 required regression: broken Cell 94/96 must fail the seeded lifecycle gate."""
        nb = copy.deepcopy(load_patched_notebook())
        if mutation == "migration_raises_and_is_swallowed":
            idx = 94
            anchor = (
                "        _migrated_checkpoint_count = migrate_thread(\n"
                '            run_config["configurable"]["thread_id"], dst_cp, full_history=True\n'
                "        )  # preserves time-travel history\n"
            )
            replacement = (
                "        try:\n"
                "            raise RuntimeError('REVIEW_MUTATION_MIGRATION_FAILED')\n"
                "        except Exception as _mig_err:\n"
                "            print(f'WARNING migrate_thread outer error: {_mig_err}')\n"
                "            _migrated_checkpoint_count = 0\n"
            )
        elif mutation == "noop_migrate_thread":
            idx = 94
            anchor = "        src_cp = src_graph.checkpointer\n"
            replacement = "        return 0\n" + anchor
        elif mutation == "replace_put_writes_with_pass":
            idx = 94
            anchor = "dst_cp.put_writes(saved_cfg, task_writes, task_id, task_path=tp)"
            replacement = "pass  # REVIEW_MUTATION_DROP_PENDING_WRITES"
        else:
            idx = 96
            anchor = "checkpointer=_restore_cp,"
            replacement = "checkpointer=None,"
        src = "".join(nb["cells"][idx]["source"])
        assert src.count(anchor) == 1, f"Mutation anchor for {mutation} not unique"
        nb["cells"][idx]["source"] = [src.replace(anchor, replacement)]
        with pytest.raises(CellExecutionError) as exc_info:
            TestRealKernelNotebookExecution.run_graph_lifecycle_scenario(
                tmp_path, sqlite_available=True, notebook_dict=nb
            )
        message = re.sub(r"\x1b\[[0-9;]*m", "", str(exc_info.value))
        if mutation == "replace_put_writes_with_pass":
            assert "Migrated history differs from source history" in message, message[-2000:]
        elif idx == 94:
            # Seeded-history gate: zero migrated checkpoints vs N seeded.
            assert re.search(r"AssertionError: \(0, \d+\)", message), message[-2000:]
        else:
            assert "No checkpointer set" in message, message[-2000:]
