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
import json
import os
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


def execute_cells_in_real_kernel(
    cell_specs: Sequence[CellSpec],
    *,
    notebook_dict: Optional[Dict[str, Any]] = None,
    cwd: Path = REPO_ROOT,
    env_overrides: Optional[Dict[str, Optional[str]]] = None,
    timeout: int = 180,
) -> Tuple[nbformat.NotebookNode, Dict[str, Any], List[str]]:
    """
    Build a notebook from actual committed cells in `IntelligentDataDetective_beta_v5_patched.ipynb`
    (plus any explicit probe/setup cells) and execute it in a real `ipykernel` via `NotebookClient`.

    Returns `(executed_nb, probe_payloads_by_label, stream_texts_by_cell)`.
    """
    source_nb = notebook_dict if notebook_dict is not None else load_patched_notebook()
    nb = nbformat.v4.new_notebook()
    labels: List[str] = []

    for spec in cell_specs:
        if isinstance(spec, int):
            raw_cell = source_nb["cells"][spec]
            assert (
                raw_cell.get("cell_type") == "code"
            ), f"Cell {spec} is not a code cell"
            src = "".join(raw_cell.get("source", []))
            cell_node = nbformat.v4.new_code_cell(source=src)
            cell_node.metadata["idd_cell_index"] = spec
            nb.cells.append(cell_node)
            labels.append(f"cell_{spec}")
        else:
            label, src = spec
            cell_node = nbformat.v4.new_code_cell(source=src)
            cell_node.metadata["idd_probe_label"] = label
            nb.cells.append(cell_node)
            labels.append(label)

    kernel_name = resolve_real_kernel_name()
    with _scoped_environ(env_overrides):
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(cwd)}},
        )
        executed_nb = client.execute()

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
            ("cleanup", KERNEL_CLEANUP_SOURCE),
        ]
        _, probes, streams = execute_cells_in_real_kernel(cell_specs)

        # Verify Cell 4 output markers
        cell4_out = streams[1]
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
        self, tmp_path: Path, sqlite_available: bool
    ):
        """
        Execute actual Cells 1, 4, 7, 10, 16, 19, 22, 28, 31, 32, 46, 57, 60, 94, and 96
        in a real Jupyter kernel under both SQLite-present and SQLite-absent modes, then
        execute representative no-key graph transitions across the compiled graph.

        Verifies:
        - Cell 60 compiles `data_detective_graph` via `idd_graph.build_graph` with `structural_only=False`.
        - Graph topology has 15 user nodes, 33 edges, 21 conditional edges, and 70 State fields.
        - SQLite-present mode binds `SqliteSaver`, migrates checkpoints to `checkpoints_demo.sqlite`
          in Cell 94, compiles `_restored_graph` in Cell 96, and preserves `data_detective_graph`.
        - SQLite-absent mode (with `langgraph.checkpoint.sqlite` blocked inside the kernel)
          falls back cleanly to `MemorySaver` in Cell 60 and skips SQLite demo migration/restore
          in Cells 94 and 96 while preserving `data_detective_graph`.
        - Representative no-key transitions execute on the real compiled graph, exercising:
          `initial_analysis -> supervisor -> data_cleaner -> supervisor -> analyst -> supervisor
          -> EMERGENCY_MSG (real Cell 57 node) -> analyst -> supervisor -> visualization
          -> viz_worker -> viz_join (real Cell 57 barrier) -> viz_evaluator -> supervisor
          -> report_orchestrator -> report_section_worker -> report_join (real Cell 57 barrier)
          -> report_packager -> file_writer -> supervisor -> FINISH -> __end__`.
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

if {sqlite_available!r}:
    assert _cell60_checkpointer_type == "SqliteSaver"
    assert (WORKING_DIRECTORY / "checkpoints_demo.sqlite").exists()
    assert "_restored_graph" in globals()
    assert _restored_graph is not data_detective_graph
else:
    assert isinstance(checkpointer, MemorySaver)
    assert _cell60_checkpointer_type in ("MemorySaver", "InMemorySaver")

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

# 3. Execute representative no-key multi-hop workflow on the compiled graph
invoke_cfg = {{
    "configurable": {{"thread_id": "issue-164-e2e-transition-thread"}},
    "recursion_limit": 80,
}}
final_state = data_detective_graph.invoke(
    {{"user_prompt": "Analyze dataset", "messages": [], "_config": invoke_cfg}},
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

assert final_state.get("file_writer_complete") is True
assert final_state.get("visualization_complete") is True
assert final_state.get("report_generator_complete") is True
assert "All invariants verified." in (final_state.get("report_draft") or "")
assert final_state.get("report_results") is not None
assert final_state["report_results"].markdown_report_path.endswith("final_report.md")

print({PROBE_PREFIX!r} + json.dumps({{
    "status": "ok",
    "sqlite_available": {sqlite_available!r},
    "checkpointer_type": _cell60_checkpointer_type,
    "node_count": len(user_nodes),
    "edge_count": total_edges,
    "conditional_edge_count": conditional_edges,
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
                94,
                96,
                ("verify_and_invoke_graph", verify_and_invoke_code),
                ("cleanup", KERNEL_CLEANUP_SOURCE),
            ]
        )

        _, probes, streams = execute_cells_in_real_kernel(cell_specs, cwd=REPO_ROOT)
        res = probes.get("verify_and_invoke_graph")
        assert res is not None, f"Missing graph verification probe. Streams: {streams}"
        assert res["status"] == "ok"
        assert res["node_count"] == 15
        assert res["edge_count"] == 33
        assert res["conditional_edge_count"] == 21
        if sqlite_available:
            assert res["checkpointer_type"] == "SqliteSaver"
            assert (tmp_path / "checkpoints.sqlite").exists()
        else:
            assert res["checkpointer_type"] in ("MemorySaver", "InMemorySaver")
            combined_streams = "\n".join(streams)
            assert (
                "SqliteSaver unavailable; checkpoint migration demonstration skipped."
                in combined_streams
            )
            assert (
                "SqliteSaver unavailable; checkpoint restore demonstration skipped."
                in combined_streams
            )


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
        self, tmp_path: Path
    ):
        """
        Include meaningful negative controls that deliberately introduce broken notebook
        behavior in-memory and demonstrate that real-kernel verification fails closed.
        """
        base_nb = load_patched_notebook()

        # Negative Control A: Deliberately mutate Cell 46 to define a local `class Router`
        # shadowing `idd_graph.Router` inside `make_supervisor_node`.
        nb_bad_router = copy.deepcopy(base_nb)
        src46 = "".join(nb_bad_router["cells"][46]["source"])
        nb_bad_router["cells"][46]["source"] = [
            src46.replace(
                "from idd_graph import Router",
                "class Router(BaseNoExtrasModel):\n        next: str = 'FINISH'\n",
            )
        ]
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
                    ("check_router", router_check_cell),
                    ("cleanup", KERNEL_CLEANUP_SOURCE),
                ],
                notebook_dict=nb_bad_router,
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
                    ("cleanup", KERNEL_CLEANUP_SOURCE),
                ],
                notebook_dict=nb_bad_cell96,
            )
        assert (
            "Negative control caught Cell 96 overwriting data_detective_graph!"
            in str(exc_cell96.value)
        )
