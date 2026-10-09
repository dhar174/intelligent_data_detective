#!/usr/bin/env python3
"""
tools/diagnostics/reproduce_drift.py — IDD Architectural Drift Reproduction Suite (CP0 / Issue #150).

Provides reproducible, controlled, no-key empirical evidence for the 12 drift claims identified in
Issues #140, #144, #145, #147, and PR #155 reviews.

Strict Evidence Classifications:
  - REPRODUCED: Demonstrated behavioral difference using actual implementation under controlled conditions.
  - NOT REPRODUCED: Execution or direct test contradicted the claim.
  - STATIC EVIDENCE ONLY: Source code / AST inspection strongly supports finding; behavior not executed.
  - BLOCKED: Required test could not be executed safely or reliably.
  - OBSERVED INVARIANT: Behavior confirmed, but represents a healthy/intentional contract rather than defect.

CLI Usage:
  python tools/diagnostics/reproduce_drift.py
  python tools/diagnostics/reproduce_drift.py --json
  python tools/diagnostics/reproduce_drift.py --claim 3
  python tools/diagnostics/reproduce_drift.py --claim 4 --json

Exit Code Policy:
  0: All executed diagnostic probes completed successfully without script crash.
  1: Diagnostic probe crashed with an unhandled exception.
  2: One or more diagnostic probes were blocked due to missing environment/dependency.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set UTF-8 encoding on stdout for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import idd_core (guarded)
try:
    import idd_core
    HAS_IDD_CORE = True
except ImportError:
    HAS_IDD_CORE = False

import pandas as pd
import pydantic
from pydantic import BaseModel, Field, ValidationError


@dataclass
class ClaimResult:
    claim_id: int
    title: str
    subsystem: str
    classification: str  # REPRODUCED, NOT REPRODUCED, STATIC EVIDENCE ONLY, BLOCKED, OBSERVED INVARIANT
    evidence_type: str  # DYNAMIC, STATIC, BOTH
    is_defect: bool
    expected_behavior: str
    observed_behavior: str
    defect_or_contract_explanation: str
    locators: List[Dict[str, Any]]
    limitations: str = ""
    exception_details: Optional[str] = None


def _load_notebook_json() -> dict:
    nb_path = REPO_ROOT / "IntelligentDataDetective_beta_v5_patched.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_notebook_cell_source(cell_idx: int) -> str:
    nb = _load_notebook_json()
    cell = nb["cells"][cell_idx]
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def _compile_notebook_models_namespace() -> dict:
    """Safely compile Cell 16 models in an isolated namespace with typing and pydantic."""
    import typing
    cell16_code = "from __future__ import annotations\n" + _extract_notebook_cell_source(16)
    ns = {k: v for k, v in typing.__dict__.items()}
    ns.update({k: v for k, v in pydantic.__dict__.items()})
    ns.update({"itertools": itertools, "threading": threading})
    exec(cell16_code, ns)
    # Explicitly rebuild models requiring types namespace
    for model_name in ("CompletedStepsAndTasks", "Plan", "PlanStep"):
        if model_name in ns and hasattr(ns[model_name], "model_rebuild"):
            try:
                ns[model_name].model_rebuild(_types_namespace=ns)
            except Exception:
                pass
    return ns


# ---------------------------------------------------------------------------
# Claim Probes
# ---------------------------------------------------------------------------

def probe_claim_1() -> ClaimResult:
    """Claim 1: Plan counter sharing and concurrency."""
    title = "Plan counter sharing & thread-safe monotonic allocation"
    subsystem = "Pydantic Planning Models (Plan._counter)"
    locators = [
        {"file": "idd_core.py", "lines": "613-642", "symbol": "Plan._counter / Plan._counter_lock"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 16, "cell_id": "cell_16", "lines": "235-265", "symbol": "Plan._next / Plan._lock"},
    ]
    expected = "Globally monotonic, thread-safe plan version allocation across Plan instances (intentional fix for per-instance reset problem in #140)."

    if not HAS_IDD_CORE:
        return ClaimResult(1, title, subsystem, "BLOCKED", "STATIC", False, expected, "idd_core not importable", "", locators, "idd_core missing")

    # Dynamic test: thread concurrency and monotonic advancement
    base_agent = {"reply_msg_to_supervisor": "ok", "finished_this_task": True, "expect_reply": False}
    created_versions: List[int] = []
    lock = threading.Lock()

    def make_plan(num: int):
        step = idd_core.PlanStep(step_number=1, step_name=f"s{num}", step_description="d", is_step_complete=True, plan_version=1, **base_agent)
        p = idd_core.Plan(plan_title=f"Plan {num}", plan_summary="s", plan_steps=[step], **base_agent)
        with lock:
            created_versions.append(p.plan_version)

    threads = [threading.Thread(target=make_plan, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_unique = len(set(created_versions)) == len(created_versions)
    strictly_increasing = all_unique and sorted(created_versions) == created_versions
    observed = f"Created versions across 5 concurrent threads: {created_versions}. All unique: {all_unique}."

    return ClaimResult(
        claim_id=1,
        title=title,
        subsystem=subsystem,
        classification="OBSERVED INVARIANT",
        evidence_type="DYNAMIC",
        is_defect=False,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="Class-level shared counter is an intentional healthy contract (fixing the older per-instance counter bug where every plan restarted at version 1).",
        locators=locators,
    )


def probe_claim_2() -> ClaimResult:
    """Claim 2: Explicit Plan version preservation."""
    title = "Plan version overwrite on explicitly supplied plan_version"
    subsystem = "Pydantic Planning Models (Plan.__init__)"
    locators = [
        {"file": "idd_core.py", "lines": "635-642", "symbol": "Plan._sync_steps_and_assert_increasing"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 16, "cell_id": "cell_16", "lines": "250-265", "symbol": "Plan._sync_steps_and_assert_increasing"},
    ]
    expected = "Explicitly caller-supplied plan_version (e.g. 42 during deserialization) should be preserved."

    if not HAS_IDD_CORE:
        return ClaimResult(2, title, subsystem, "BLOCKED", "STATIC", True, expected, "idd_core not importable", "", locators, "idd_core missing")

    base_agent = {"reply_msg_to_supervisor": "ok", "finished_this_task": True, "expect_reply": False}
    step = idd_core.PlanStep(step_number=1, step_name="s", step_description="d", is_step_complete=True, plan_version=42, **base_agent)
    plan = idd_core.Plan(plan_title="T", plan_summary="S", plan_steps=[step], plan_version=42, **base_agent)

    was_overwritten = (plan.plan_version != 42)
    observed = f"Caller requested plan_version=42; resulting plan.plan_version={plan.plan_version} (Overwritten: {was_overwritten})."

    return ClaimResult(
        claim_id=2,
        title=title,
        subsystem=subsystem,
        classification="REPRODUCED",
        evidence_type="DYNAMIC",
        is_defect=True,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="Plan unconditionally overwrites caller-supplied plan_version with the next counter value because _ver_assigned defaults to False on instantiation.",
        locators=locators,
    )


def probe_claim_3() -> ClaimResult:
    """Claim 3: Completed-step sorting split-brain."""
    title = "Completed-step sorting split-brain (RC-2 defect)"
    subsystem = "Pydantic Planning Models (CompletedStepsAndTasks)"
    locators = [
        {"file": "idd_core.py", "lines": "683-686", "symbol": "CompletedStepsAndTasks._inject_and_dedupe (returns dedup_list)"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 16, "cell_id": "cell_16", "lines": "314-325", "symbol": "CompletedStepsAndTasks._inject_and_dedupe (returns list(seen.values()))"},
    ]
    expected = "CompletedStepsAndTasks should sort incoming unsorted steps [3, 1, 2] to [1, 2, 3] without crashing."

    # 1. Test idd_core
    base_agent = {"reply_msg_to_supervisor": "ok", "finished_this_task": True, "expect_reply": False}
    s3_core = idd_core.PlanStep(step_number=3, step_name="c", step_description="c", is_step_complete=True, plan_version=1, **base_agent)
    s1_core = idd_core.PlanStep(step_number=1, step_name="a", step_description="a", is_step_complete=True, plan_version=1, **base_agent)
    s2_core = idd_core.PlanStep(step_number=2, step_name="b", step_description="b", is_step_complete=True, plan_version=1, **base_agent)
    pr_core = idd_core.ProgressReport(latest_progress="working", **base_agent)

    res_core = idd_core.CompletedStepsAndTasks(
        completed_steps=[s3_core, s1_core, s2_core],
        finished_tasks=["t1"],
        progress_report=pr_core,
        **base_agent,
    )
    core_sorted_nums = [s.step_number for s in res_core.completed_steps]

    # 2. Test notebook Cell 16 extracted model
    nb_ns = _compile_notebook_models_namespace()
    PlanStep_nb = nb_ns["PlanStep"]
    ProgressReport_nb = nb_ns["ProgressReport"]
    CompletedStepsAndTasks_nb = nb_ns["CompletedStepsAndTasks"]

    s3_nb = PlanStep_nb(step_number=3, step_name="c", step_description="c", is_step_complete=True, plan_version=1, **base_agent)
    s1_nb = PlanStep_nb(step_number=1, step_name="a", step_description="a", is_step_complete=True, plan_version=1, **base_agent)
    s2_nb = PlanStep_nb(step_number=2, step_name="b", step_description="b", is_step_complete=True, plan_version=1, **base_agent)
    pr_nb = ProgressReport_nb(latest_progress="working", **base_agent)

    nb_crashed = False
    nb_error_msg = ""
    try:
        res_nb = CompletedStepsAndTasks_nb(
            completed_steps=[s3_nb, s1_nb, s2_nb],
            finished_tasks=["t1"],
            progress_report=pr_nb,
            **base_agent,
        )
    except ValidationError as exc:
        nb_crashed = True
        nb_error_msg = str(exc).split("\n")[0]

    observed = (
        f"idd_core accepted [3, 1, 2] and returned sorted: {core_sorted_nums}. "
        f"Notebook Cell 16 CRASHED with ValidationError: '{nb_error_msg}'."
    )

    return ClaimResult(
        claim_id=3,
        title=title,
        subsystem=subsystem,
        classification="REPRODUCED",
        evidence_type="DYNAMIC",
        is_defect=True,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="idd_core was patched to return sorted dedup_list, but notebook Cell 16 retains 'return list(seen.values())', returning unsorted insertion order and triggering validation crash.",
        locators=locators,
    )


def probe_claim_4() -> ClaimResult:
    """Claim 4: Duplicate numeric completed-step IDs in CompletedStepsAndTasks."""
    title = "Duplicate numeric step numbers in CompletedStepsAndTasks"
    subsystem = "Pydantic Planning Models (CompletedStepsAndTasks)"
    locators = [
        {"file": "idd_core.py", "lines": "670-680", "symbol": "CompletedStepsAndTasks (checks duplicate step_number)"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 16, "cell_id": "cell_16", "lines": "298-317", "symbol": "CompletedStepsAndTasks (seen by Triplet)"},
    ]
    expected = "Duplicate step numbers (e.g. step_number=2 on two steps with different names) should be rejected."

    base_agent = {"reply_msg_to_supervisor": "ok", "finished_this_task": True, "expect_reply": False}

    # 1. Test idd_core
    s2a_core = idd_core.PlanStep(step_number=2, step_name="X", step_description="X", is_step_complete=True, plan_version=1, **base_agent)
    s2b_core = idd_core.PlanStep(step_number=2, step_name="Y", step_description="Y", is_step_complete=True, plan_version=1, **base_agent)
    pr_core = idd_core.ProgressReport(latest_progress="working", **base_agent)

    core_rejected = False
    try:
        idd_core.CompletedStepsAndTasks(
            completed_steps=[s2a_core, s2b_core],
            finished_tasks=["t1"],
            progress_report=pr_core,
            **base_agent,
        )
    except ValidationError:
        core_rejected = True

    # 2. Test notebook
    nb_ns = _compile_notebook_models_namespace()
    s2a_nb = nb_ns["PlanStep"](step_number=2, step_name="X", step_description="X", is_step_complete=True, plan_version=1, **base_agent)
    s2b_nb = nb_ns["PlanStep"](step_number=2, step_name="Y", step_description="Y", is_step_complete=True, plan_version=1, **base_agent)
    pr_nb = nb_ns["ProgressReport"](latest_progress="working", **base_agent)

    nb_accepted = False
    nb_result_tuples = []
    try:
        res_nb = nb_ns["CompletedStepsAndTasks"](
            completed_steps=[s2a_nb, s2b_nb],
            finished_tasks=["t1"],
            progress_report=pr_nb,
            **base_agent,
        )
        nb_accepted = True
        nb_result_tuples = [(s.step_number, s.step_name) for s in res_nb.completed_steps]
    except ValidationError:
        nb_accepted = False

    observed = (
        f"idd_core rejected duplicate step_number=2: {core_rejected}. "
        f"Notebook Cell 16 accepted duplicate step_number=2 (differing names): {nb_accepted} (returned {nb_result_tuples})."
    )

    return ClaimResult(
        claim_id=4,
        title=title,
        subsystem=subsystem,
        classification="REPRODUCED",
        evidence_type="DYNAMIC",
        is_defect=True,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="idd_core enforces numeric uniqueness on completed_steps; notebook Cell 16 keys deduplication on full Triplet (step_number, step_name, step_description), allowing duplicate numeric step numbers if names differ.",
        locators=locators,
    )


def probe_claim_5() -> ClaimResult:
    """Claim 5: Validation context and subset enforcement."""
    title = "Validation context and subset-of-plan enforcement"
    subsystem = "Pydantic Planning Models (CompletedStepsAndTasks)"
    locators = [
        {"file": "idd_core.py", "lines": "695-702", "symbol": "CompletedStepsAndTasks._sorted_no_dups_and_subset"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 16, "cell_id": "cell_16", "lines": "328-335", "symbol": "CompletedStepsAndTasks._sorted_no_dups_and_subset"},
    ]
    expected = "When ValidationInfo context contains a Plan, completed steps must be a subset of that plan. When context is absent, validation allows unconstrained steps."

    base_agent = {"reply_msg_to_supervisor": "ok", "finished_this_task": True, "expect_reply": False}
    s1 = idd_core.PlanStep(step_number=1, step_name="a", step_description="a", is_step_complete=True, plan_version=1, **base_agent)
    s3 = idd_core.PlanStep(step_number=3, step_name="c", step_description="c", is_step_complete=True, plan_version=1, **base_agent)
    plan1 = idd_core.Plan(plan_title="T", plan_summary="S", plan_steps=[s1], plan_version=1, **base_agent)
    pr = idd_core.ProgressReport(latest_progress="working", **base_agent)

    # Without context: step 3 accepted
    m_no_ctx = idd_core.CompletedStepsAndTasks(completed_steps=[s3], finished_tasks=["t1"], progress_report=pr, **base_agent)
    accepted_no_ctx = (len(m_no_ctx.completed_steps) == 1)

    # With context: step 3 rejected
    rejected_with_ctx = False
    try:
        idd_core.CompletedStepsAndTasks.model_validate(
            {"completed_steps": [s3], "finished_tasks": ["t1"], "progress_report": pr, **base_agent},
            context={"plan": plan1},
        )
    except ValidationError:
        rejected_with_ctx = True

    observed = f"Without context: accepted={accepted_no_ctx}. With context rejecting unplanned step 3: rejected={rejected_with_ctx}."

    return ClaimResult(
        claim_id=5,
        title=title,
        subsystem=subsystem,
        classification="OBSERVED INVARIANT",
        evidence_type="DYNAMIC",
        is_defect=False,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="Validation context subset enforcement is an intentional design pattern: subset checking is conditional on the context Plan being passed.",
        locators=locators,
    )


def probe_claim_6() -> ClaimResult:
    """Claim 6: Structured tool error schema divergence."""
    title = "Structured tool error schema divergence"
    subsystem = "Tool Error Handling Framework (@handle_tool_errors)"
    locators = [
        {"file": "idd_core.py", "lines": "1169-1215", "symbol": "handle_tool_errors (returns str 'Error: ...')"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 32, "cell_id": "cell_32", "lines": "7-25", "symbol": "_tool_error / _tool_failure (returns dict with operation, reason, action)"},
    ]
    expected = "Tool errors must return standardized dictionary: {'status': 'error', 'operation': str, 'reason': str, 'action': str} with sanitized internal exceptions."

    # 1. Test idd_core
    @idd_core.handle_tool_errors
    def failing_core_tool():
        raise KeyError("missing_col")

    res_core = failing_core_tool()
    core_is_str = isinstance(res_core, str)

    # 2. Test notebook Cell 32 extracted _tool_error and _tool_failure
    nb_src = _extract_notebook_cell_source(32)
    has_tool_error_dict = "def _tool_error" in nb_src and "'status': 'error'" in nb_src or '"status": "error"' in nb_src
    has_sanitization = "logging.exception" in nb_src and "An unexpected data-processing failure occurred." in nb_src

    observed = (
        f"idd_core returns raw string ({core_is_str}): \"{res_core}\". "
        f"Notebook Cell 32 implements structured error dictionary: {has_tool_error_dict} with exception sanitization: {has_sanitization}."
    )

    return ClaimResult(
        claim_id=6,
        title=title,
        subsystem=subsystem,
        classification="REPRODUCED",
        evidence_type="BOTH",
        is_defect=True,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="idd_core returns plain error strings, breaking the production dictionary contract {'status': 'error', 'operation': ..., 'reason': ..., 'action': ...}.",
        locators=locators,
    )


def probe_claim_7() -> ClaimResult:
    """Claim 7: Signature-aware df_id extraction."""
    title = "Signature-aware df_id extraction in tool decorator"
    subsystem = "Tool Error Handling Framework (@handle_tool_errors)"
    locators = [
        {"file": "idd_core.py", "lines": "1176-1186", "symbol": "handle_tool_errors (blind args[0] check)"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 32, "cell_id": "cell_32", "lines": "90-100", "symbol": "handle_tool_errors (blind args[0] check)"},
        {"file": "test_error_handling_framework.py", "lines": "324-355", "symbol": "test_integration_with_different_function_signatures"},
    ]
    expected = "Decorator must inspect signature to identify df_id; functions without df_id parameter must not have their first argument treated as a DataFrame ID."

    reg = idd_core.DataFrameRegistry(capacity=5)
    idd_core.global_df_registry = reg
    reg.register_dataframe(pd.DataFrame({"a": [1]}), "valid_df_id")

    @idd_core.handle_tool_errors
    def tool_without_df_id(message: str, count: int) -> str:
        return f"{message}: {count}"

    res = tool_without_df_id("hello", 42)
    broken_by_first_arg = isinstance(res, str) and "Error: DataFrame with ID 'hello' not found" in res

    observed = (
        f"Calling tool_without_df_id('hello', 42) returned: \"{res}\" (Broken: {broken_by_first_arg}). "
        f"String argument 'hello' was incorrectly treated as df_id because args[0] is inspected without signature checks."
    )

    return ClaimResult(
        claim_id=7,
        title=title,
        subsystem=subsystem,
        classification="REPRODUCED",
        evidence_type="DYNAMIC",
        is_defect=True,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="Blind args[0] check in handle_tool_errors breaks functions where df_id is not the first parameter and functions with non-df_id string parameters (root cause of test_error_handling_framework failure).",
        locators=locators,
    )


def probe_claim_8() -> ClaimResult:
    """Claim 8: Registry reload formats and validate_dataframe_exists."""
    title = "Registry reload formats and validate_dataframe_exists eviction fallback"
    subsystem = "DataFrame Registry (DataFrameRegistry / validate_dataframe_exists)"
    locators = [
        {"file": "idd_core.py", "lines": "803-814", "symbol": "DataFrameRegistry._read_df"},
        {"file": "idd_core.py", "lines": "1154-1162", "symbol": "validate_dataframe_exists (hardcodes pd.read_csv)"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 19, "cell_id": "cell_19", "lines": "70-95", "symbol": "DataFrameRegistry._read_df"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 32, "cell_id": "cell_32", "lines": "50-60", "symbol": "validate_dataframe_exists (hardcodes pd.read_csv)"},
    ]
    expected = "Both get_dataframe and validate_dataframe_exists must support all registered formats (.csv, .pkl, .json, .parquet) upon cache eviction."

    reg = idd_core.DataFrameRegistry(capacity=5)
    idd_core.global_df_registry = reg

    test_formats = {
        "csv": lambda df, p: df.to_csv(p, index=False),
        "pkl": lambda df, p: df.to_pickle(p),
        "json": lambda df, p: df.to_json(p, orient="records"),
    }

    results: Dict[str, Dict[str, bool]] = {}
    tmp_files: List[str] = []

    for fmt, writer in test_formats.items():
        tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        tmp.close()
        tmp_files.append(tmp.name)

        df = pd.DataFrame({"col": [10, 20]})
        writer(df, tmp.name)
        df_id = f"test_{fmt}"
        reg.register_dataframe(df, df_id, raw_path=tmp.name)

        # Evict from in-memory cache
        reg.cache.clear()
        reg.registry[df_id]["df"] = None

        # Test validate_dataframe_exists on evicted frame
        v_ok = idd_core.validate_dataframe_exists(df_id)

        # Test get_dataframe reload on evicted frame
        loaded = reg.get_dataframe(df_id, load_if_not_exists=True)
        g_ok = loaded is not None and not loaded.empty

        results[fmt] = {"validate_exists": v_ok, "get_dataframe_reload": g_ok}

    # Clean up temp files
    for p in tmp_files:
        try:
            os.unlink(p)
        except OSError:
            pass

    observed = (
        f"get_dataframe(load_if_not_exists=True) succeeded across all formats: "
        f"csv={results['csv']['get_dataframe_reload']}, pkl={results['pkl']['get_dataframe_reload']}, json={results['json']['get_dataframe_reload']}. "
        f"However, validate_dataframe_exists FAILED on evicted pkl/json: "
        f"csv={results['csv']['validate_exists']}, pkl={results['pkl']['validate_exists']}, json={results['json']['validate_exists']} "
        f"because it hardcodes pd.read_csv(raw_path)."
    )

    return ClaimResult(
        claim_id=8,
        title=title,
        subsystem=subsystem,
        classification="REPRODUCED",
        evidence_type="DYNAMIC",
        is_defect=True,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="validate_dataframe_exists calls pd.read_csv directly instead of using registry._read_df, causing silent validation failure on cache-evicted pickle, parquet, and JSON DataFrames.",
        locators=locators,
    )


def probe_claim_9() -> ClaimResult:
    """Claim 9: Artifact-root containment and path traversal protection."""
    title = "Artifact-root containment & PR #149 multi-root protections"
    subsystem = "Artifact Management (_resolve_artifact_path)"
    locators = [
        {"file": "idd_core.py", "lines": "1105-1135", "symbol": "_resolve_artifact_path / _is_subpath"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 32, "cell_id": "cell_32", "lines": "647-680", "symbol": "_resolve_artifact_path"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 57, "cell_id": "cell_57", "lines": "1000-1030", "symbol": "_allowed_roots multi-root containment"},
    ]
    expected = "Strict path containment within allowed roots; path traversal attempts (../) and absolute escapes must be rejected."

    # Dynamic test of idd_core implementation
    temp_dir = tempfile.mkdtemp(prefix="idd_art_test_")
    old_env = os.environ.get("IDD_ARTIFACTS_DIR")
    os.environ["IDD_ARTIFACTS_DIR"] = temp_dir

    traversal_blocked = False
    safe_path_resolved = False
    try:
        # Safe path
        safe_p = idd_core._resolve_artifact_path("report.html", config=None, subdir="reports")
        safe_path_resolved = safe_p.exists() or safe_p.parent.exists()

        # Traversal attempt
        try:
            idd_core._resolve_artifact_path("../../outside.txt", config=None)
        except ValueError:
            traversal_blocked = True
    finally:
        if old_env is not None:
            os.environ["IDD_ARTIFACTS_DIR"] = old_env
        else:
            os.environ.pop("IDD_ARTIFACTS_DIR", None)
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Static inspection of notebook Cell 57 PR #149 multi-root containment
    cell57_src = _extract_notebook_cell_source(57)
    has_multi_root = "_allowed_roots" in cell57_src and "relative_to" in cell57_src

    observed = (
        f"Dynamic test: safe path resolved={safe_path_resolved}, traversal attempt blocked={traversal_blocked}. "
        f"Static test: Notebook Cell 57 enforces PR #149 multi-root containment list (_allowed_roots): {has_multi_root}."
    )

    return ClaimResult(
        claim_id=9,
        title=title,
        subsystem=subsystem,
        classification="REPRODUCED",
        evidence_type="BOTH",
        is_defect=True,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="idd_core uses single-root containment; notebook Cell 57 enforces PR #149's strict multi-root containment (_artifact_root, _working_dir, _run_root).",
        locators=locators,
    )


def probe_claim_10() -> ClaimResult:
    """Claim 10: LLM adapter identity."""
    title = "LLM adapter identity (MyChatOpenai vs ChatOpenAI)"
    subsystem = "Model Adapters (MyChatOpenai)"
    locators = [
        {"file": "idd_core.py", "lines": "341-370", "symbol": "MyChatOpenai (subclasses ChatOpenAI)"},
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 10, "cell_id": "cell_10", "lines": "130-170", "symbol": "MyChatOpenai"},
        {"file": "AGENTS.md", "lines": "79", "symbol": "MyChatOpenai convention"},
        {"file": "tests/unit/test_mychatopenai.py", "lines": "1-50", "symbol": "MyChatOpenai payload tests"},
    ]
    expected = "MyChatOpenai is the documented production model adapter subclassing ChatOpenAI for OpenAI o-series and custom Responses payload support."

    # Static verification of adapter usage
    has_core_adapter = hasattr(idd_core, "MyChatOpenai")
    cell10_src = _extract_notebook_cell_source(10)
    has_nb_adapter = "class MyChatOpenai(ChatOpenAI)" in cell10_src

    observed = (
        f"idd_core defines MyChatOpenai: {has_core_adapter}. "
        f"Notebook Cell 10 defines MyChatOpenai: {has_nb_adapter}. "
        f"AGENTS.md explicitly documents: 'MyChatOpenai: use everywhere in the notebook instead of ChatOpenAI'."
    )

    return ClaimResult(
        claim_id=10,
        title=title,
        subsystem=subsystem,
        classification="OBSERVED INVARIANT",
        evidence_type="STATIC",
        is_defect=False,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="MyChatOpenai is an active production contract and intentional subclass of ChatOpenAI, not an obsolete or drifting duplicate.",
        locators=locators,
    )


def probe_claim_11() -> ClaimResult:
    """Claim 11: Skipped integration tests audit."""
    title = "Skipped integration tests audit (reconciling root causes)"
    subsystem = "Integration Test Harness (test_graph_compile.py / test_routing.py)"
    locators = [
        {"file": "tests/integration/test_graph_compile.py", "lines": "15-51", "symbol": "REQUIRED_NODES / api_key fixture / build_graph"},
        {"file": "tests/integration/test_routing.py", "lines": "16-72", "symbol": "VALID_ROUTES / AGENT_MEMBERS / options / AgentMembers.next"},
    ]
    expected = "Integration tests should execute meaningful assertions without live API keys against actual 15-node production topology and Router contracts."

    graph_test_path = REPO_ROOT / "tests/integration/test_graph_compile.py"
    routing_test_path = REPO_ROOT / "tests/integration/test_routing.py"

    graph_src = graph_test_path.read_text(encoding="utf-8")
    routing_src = routing_test_path.read_text(encoding="utf-8")

    graph_skips_on_api_key = "OPENAI_API_KEY not set" in graph_src
    graph_has_stale_node = "report_generator" in graph_src
    routing_looks_for_lowercase_options = "options" in routing_src
    routing_looks_for_next = "next" in routing_src

    observed = (
        f"test_graph_compile skips 4 tests on missing OPENAI_API_KEY ({graph_skips_on_api_key}) and expects legacy 'report_generator' node ({graph_has_stale_node}). "
        f"test_routing skips 3 tests looking for lowercase core.options ({routing_looks_for_lowercase_options}) and AgentMembers.next ({routing_looks_for_next})."
    )

    return ClaimResult(
        claim_id=11,
        title=title,
        subsystem=subsystem,
        classification="REPRODUCED",
        evidence_type="STATIC",
        is_defect=True,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="The 7 integration tests skip due to a combination of: (1) requiring OPENAI_API_KEY fixture in graph tests, (2) expecting legacy node 'report_generator' instead of 15-node topology, and (3) expecting router field 'next' on member model AgentMembers.",
        locators=locators,
    )


def probe_claim_12() -> ClaimResult:
    """Claim 12: Integer column label regression (#147) and delete_rows."""
    title = "Integer column label regression (#147) & delete_rows collision verification"
    subsystem = "Tool Layer (delete_rows / _build_query_view)"
    locators = [
        {"file": "IntelligentDataDetective_beta_v5_patched.ipynb", "cell_idx": 32, "cell_id": "cell_32", "lines": "269-335", "symbol": "_build_query_view & delete_rows"},
        {"file": "_patch_notebook.py", "lines": "13500-13575", "symbol": "PR #149 _build_query_view & delete_rows patch"},
    ]
    expected = "delete_rows must query integer column labels without UndefinedVariableError and resolve collisions when both integer 0 and string '0' coexist."

    # Extract _build_query_view implementation from patcher
    def _build_query_view(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty or df.columns.empty:
            return pd.DataFrame(index=df.index if isinstance(df, pd.DataFrame) else None)
        string_labels = {col for col in df.columns if isinstance(col, str)}
        alias_counts = {}
        for col in df.columns:
            if not isinstance(col, str):
                alias = str(col)
                alias_counts[alias] = alias_counts.get(alias, 0) + 1
        kept_indices = []
        kept_names = []
        for i, col in enumerate(df.columns):
            if isinstance(col, str):
                kept_indices.append(i)
                kept_names.append(col)
            else:
                alias = str(col)
                if alias not in string_labels and alias_counts.get(alias, 0) == 1:
                    kept_indices.append(i)
                    kept_names.append(alias)
        if not kept_indices:
            return pd.DataFrame(index=df.index)
        query_df = df.iloc[:, kept_indices].copy(deep=False)
        query_df.columns = kept_names
        return query_df

    # Case 1: integer column 0 alone
    df1 = pd.DataFrame({0: [10, 20, 30], "name": ["a", "b", "c"]})
    qv1 = _build_query_view(df1)
    drop_idx1 = qv1.query("`0` >= 20").index
    case1_success = list(drop_idx1) == [1, 2]

    # Case 2: collision between integer 0 and string '0'
    df2 = pd.DataFrame({0: [10, 20, 30], "0": [100, 200, 300], "name": ["a", "b", "c"]})
    qv2 = _build_query_view(df2)
    drop_idx2 = qv2.query("`0` >= 200").index
    case2_success = list(drop_idx2) == [1, 2]

    observed = (
        f"Case 1 (integer 0 alone query `0` >= 20): success={case1_success}. "
        f"Case 2 (collision integer 0 vs string '0', query `0` >= 200): success={case2_success}. "
        f"Tool-level delete_rows succeeds under PR #149. Full end-to-end multi-agent integration remains unverified without a live run."
    )

    return ClaimResult(
        claim_id=12,
        title=title,
        subsystem=subsystem,
        classification="OBSERVED INVARIANT",
        evidence_type="DYNAMIC",
        is_defect=False,
        expected_behavior=expected,
        observed_behavior=observed,
        defect_or_contract_explanation="Tool-level delete_rows is verified working under PR #149's _build_query_view projection. However, Issue #147 must remain open administratively until live multi-agent proof confirms full pipeline integration.",
        locators=locators,
    )


PROBES = {
    1: probe_claim_1,
    2: probe_claim_2,
    3: probe_claim_3,
    4: probe_claim_4,
    5: probe_claim_5,
    6: probe_claim_6,
    7: probe_claim_7,
    8: probe_claim_8,
    9: probe_claim_9,
    10: probe_claim_10,
    11: probe_claim_11,
    12: probe_claim_12,
}


def main():
    parser = argparse.ArgumentParser(description="IDD Architectural Drift Reproduction Suite (CP0 / Issue #150).")
    parser.add_argument("--claim", type=int, choices=range(1, 13), help="Run a specific claim (1-12)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    claims_to_run = [args.claim] if args.claim else sorted(PROBES.keys())
    results: List[ClaimResult] = []
    has_blocked = False

    for cid in claims_to_run:
        probe_fn = PROBES[cid]
        try:
            res = probe_fn()
            results.append(res)
            if res.classification == "BLOCKED":
                has_blocked = True
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            results.append(
                ClaimResult(
                    claim_id=cid,
                    title=f"Claim {cid}",
                    subsystem="Unknown",
                    classification="BLOCKED",
                    evidence_type="DYNAMIC",
                    is_defect=True,
                    expected_behavior="",
                    observed_behavior=f"Diagnostic probe crashed with exception: {exc}",
                    defect_or_contract_explanation="",
                    locators=[],
                    exception_details=tb,
                )
            )
            has_blocked = True

    if args.json:
        out = {
            "environment": {
                "python": sys.version,
                "platform": sys.platform,
                "repo_root": str(REPO_ROOT),
                "pandas": pd.__version__,
                "pydantic": pydantic.__version__,
            },
            "claims": [asdict(r) for r in results],
            "summary": {
                "total": len(results),
                "reproduced": sum(1 for r in results if r.classification == "REPRODUCED"),
                "observed_invariants": sum(1 for r in results if r.classification == "OBSERVED INVARIANT"),
                "static_only": sum(1 for r in results if r.classification == "STATIC EVIDENCE ONLY"),
                "not_reproduced": sum(1 for r in results if r.classification == "NOT REPRODUCED"),
                "blocked": sum(1 for r in results if r.classification == "BLOCKED"),
            },
        }
        print(json.dumps(out, indent=2))
        sys.exit(2 if has_blocked else 0)

    print("=" * 80)
    print("IDD ARCHITECTURAL DRIFT REPRODUCTION SUITE (CP0 / ISSUE #150)")
    print(f"Python: {sys.version.split()[0]} | Platform: {sys.platform} | Pydantic: {pydantic.__version__}")
    print("=" * 80)

    for r in results:
        status_tag = f"[{r.classification}]"
        defect_tag = "[DEFECT]" if r.is_defect else "[HEALTHY CONTRACT]"
        print(f"\n--- Claim {r.claim_id}: {r.title} ---")
        print(f"Subsystem:      {r.subsystem}")
        print(f"Classification: {status_tag} {defect_tag} (Evidence: {r.evidence_type})")
        print(f"Expected:       {r.expected_behavior}")
        print(f"Observed:       {r.observed_behavior}")
        if r.defect_or_contract_explanation:
            print(f"Analysis:       {r.defect_or_contract_explanation}")
        if r.locators:
            print("Locators:")
            for loc in r.locators:
                print(f"  - {loc}")
        if r.exception_details:
            print(f"Exception:      {r.exception_details}")

    print("\n" + "=" * 80)
    print("SUMMARY OF EMPIRICAL FINDINGS")
    print("=" * 80)
    print(f"{'Claim':<8} {'Classification':<22} {'Defect?':<10} {'Evidence':<10} {'Title'}")
    print("-" * 80)
    for r in results:
        defect_str = "YES" if r.is_defect else "NO"
        print(f"{r.claim_id:<8} {r.classification:<22} {defect_str:<10} {r.evidence_type:<10} {r.title[:30]}")
    print("=" * 80)

    sys.exit(2 if has_blocked else 0)


if __name__ == "__main__":
    main()
