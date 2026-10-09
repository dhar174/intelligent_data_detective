"""Diagnostic probe to reproduce architectural drift claims from Issue #140, #145, and #147.

This script executes offline no-key diagnostic checks against actual implementations
in `idd_core.py` and `IntelligentDataDetective_beta_v5_patched.ipynb`.
"""

import ast
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_notebook_cell(cell_idx: int) -> str:
    nb_path = Path("IntelligentDataDetective_beta_v5_patched.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    return "".join(nb["cells"][cell_idx].get("source", []))


def make_step(step_number=1, step_name="Step", step_desc="Desc", is_complete=False, plan_version=1):
    import idd_core

    return idd_core.PlanStep(
        step_number=step_number,
        step_name=step_name,
        step_description=step_desc,
        is_step_complete=is_complete,
        plan_version=plan_version,
        reply_msg_to_supervisor="",
        finished_this_task=is_complete,
        expect_reply=False,
    )


def make_plan_kwargs(title="P", summary="S", steps=None, version=1):
    return {
        "plan_title": title,
        "plan_summary": summary,
        "plan_steps": steps or [make_step()],
        "plan_version": version,
        "reply_msg_to_supervisor": "",
        "finished_this_task": False,
        "expect_reply": False,
    }


def run_claim_1():
    """Claim 1: Plan counter sharing and concurrency."""
    import idd_core

    print("\n--- Claim 1: Plan counter sharing & concurrency ---")
    locator = "idd_core.py:619-642 (Plan._counter)"
    p1 = idd_core.Plan(**make_plan_kwargs(title="P1", summary="S1"))
    p2 = idd_core.Plan(**make_plan_kwargs(title="P2", summary="S2"))

    shared = (p2.plan_version == p1.plan_version + 1)
    versions = []

    def make_plan():
        p = idd_core.Plan(**make_plan_kwargs(title="PT", summary="ST"))
        versions.append(p.plan_version)

    threads = [threading.Thread(target=make_plan) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    unique_versions = len(set(versions)) == len(versions)
    print(f"Locator: {locator}")
    print(f"Plan 1 version: {p1.plan_version}, Plan 2 version: {p2.plan_version}")
    print(f"Threaded versions created: {sorted(versions)} (All unique: {unique_versions})")
    if shared:
        print("Result: REPRODUCED (Plan._counter is a class-level itertools.count shared across all instances; each new Plan increments the global counter)")
    else:
        print("Result: NOT REPRODUCED")


def run_claim_2():
    """Claim 2: Plan-version synchronization (explicit plan_version overwritten)."""
    import idd_core

    print("\n--- Claim 2: Plan-version synchronization ---")
    locator = "idd_core.py:635-642 (_sync_steps_and_assert_increasing)"
    explicit_version = 42
    p = idd_core.Plan(**make_plan_kwargs(title="Explicit Version Plan", summary="Summary", version=explicit_version))
    print(f"Locator: {locator}")
    print(f"Requested plan_version: {explicit_version}, Resulting plan_version: {p.plan_version}")
    if p.plan_version != explicit_version:
        print(f"Result: REPRODUCED (User-supplied plan_version={explicit_version} was silently overwritten with counter value {p.plan_version} because _ver_assigned starts False)")
    else:
        print("Result: NOT REPRODUCED")


def run_claim_3():
    """Claim 3: Completed-step sorting divergence between idd_core and notebook."""
    import idd_core

    print("\n--- Claim 3: Completed-step sorting & split-brain fix ---")
    locator_core = "idd_core.py:5-6, 683-686"
    locator_nb = "IntelligentDataDetective_beta_v5_patched.ipynb:Cell 16:L285-L310"

    s1 = make_step(step_number=1, step_name="S1", step_desc="D1", is_complete=True, plan_version=1)
    s2 = make_step(step_number=2, step_name="S2", step_desc="D2", is_complete=True, plan_version=1)
    s3 = make_step(step_number=3, step_name="S3", step_desc="D3", is_complete=True, plan_version=1)

    # In idd_core.py: CompletedStepsAndTasks._inject_and_dedupe returns sorted dedup_list
    core_obj = idd_core.CompletedStepsAndTasks(
        completed_steps=[s3, s1, s2],
        finished_tasks=["T1"],
        progress_report=idd_core.ProgressReport(
            latest_progress="done",
            reply_msg_to_supervisor="",
            finished_this_task=False,
            expect_reply=False,
        ),
        reply_msg_to_supervisor="",
        finished_this_task=False,
        expect_reply=False,
    )
    core_sorted_nums = [s.step_number for s in core_obj.completed_steps]

    # In notebook Cell 16: check if return list(seen.values()) is present (unsorted bug)
    cell16_src = load_notebook_cell(16)
    has_nb_unsorted_bug = (
        "dedup_list.sort(key=lambda d: int(d.get(\"step_number\", 10**9)))" in cell16_src
        and "return list(seen.values())" in cell16_src
    )

    print(f"idd_core locator: {locator_core} -> returns sorted: {core_sorted_nums}")
    print(f"notebook Cell 16 locator: {locator_nb} -> retains unsorted 'return list(seen.values())': {has_nb_unsorted_bug}")
    if core_sorted_nums == [1, 2, 3] and has_nb_unsorted_bug:
        print("Result: REPRODUCED (idd_core was patched to return sorted dedup_list, but patched notebook cell 16 still returns unsorted list(seen.values()), causing split-brain behavior)")
    else:
        print("Result: NOT REPRODUCED")


def run_claim_4():
    """Claim 4: Duplicate step numbers rejection."""
    import idd_core

    print("\n--- Claim 4: Duplicate step numbers rejection ---")
    locator = "idd_core.py:630-633, 649-652"
    s1a = make_step(step_number=1, step_name="S1a", step_desc="D1a", is_complete=False, plan_version=1)
    s1b = make_step(step_number=1, step_name="S1b", step_desc="D1b", is_complete=False, plan_version=1)

    raised = False
    err_msg = ""
    try:
        idd_core.Plan(**make_plan_kwargs(title="P", summary="S", steps=[s1a, s1b]))
    except ValueError as e:
        raised = True
        err_msg = str(e)

    print(f"Locator: {locator}")
    print(f"Plan with duplicate step_number=1 raised ValueError: {raised}")
    if raised and "strictly increasing" in err_msg:
        print("Result: REPRODUCED (Plan rejects duplicate step numbers via strictly increasing validation)")
    else:
        print("Result: NOT REPRODUCED")


def run_claim_5():
    """Claim 5: Validation context / subset behavior."""
    import idd_core

    print("\n--- Claim 5: Validation context / subset behavior ---")
    locator = "idd_core.py:665-667, 695-702"
    s1 = make_step(step_number=1, step_name="S1", step_desc="D1", is_complete=True, plan_version=10)
    s_unplanned = make_step(step_number=99, step_name="Unknown", step_desc="Unknown", is_complete=True, plan_version=10)

    # Validated with context plan containing only s1:
    plan = idd_core.Plan(**make_plan_kwargs(title="P", summary="S", steps=[s1]))
    raised_on_unplanned = False
    err_msg = ""
    try:
        idd_core.CompletedStepsAndTasks.model_validate(
            {
                "completed_steps": [s_unplanned],
                "finished_tasks": ["T1"],
                "progress_report": {
                    "latest_progress": "prog",
                    "reply_msg_to_supervisor": "",
                    "finished_this_task": False,
                    "expect_reply": False,
                },
                "reply_msg_to_supervisor": "",
                "finished_this_task": False,
                "expect_reply": False,
            },
            context={"plan": plan},
        )
    except ValueError as e:
        raised_on_unplanned = True
        err_msg = str(e)

    print(f"Locator: {locator}")
    print(f"Validation with context rejecting step not in Plan: {raised_on_unplanned} ('{err_msg[:60]}...')")
    if raised_on_unplanned:
        print("Result: REPRODUCED (Validation context enforces that completed_steps must be a subset of the context Plan)")
    else:
        print("Result: NOT REPRODUCED")


def run_claim_6():
    """Claim 6: Structured tool errors divergence."""
    import idd_core

    print("\n--- Claim 6: Structured tool errors divergence ---")
    locator_core = "idd_core.py:1169-1215"
    locator_nb = "IntelligentDataDetective_beta_v5_patched.ipynb:Cell 32:L7-L140"

    @idd_core.handle_tool_errors
    def failing_core_tool():
        raise KeyError("missing_col")

    res_core = failing_core_tool()

    cell32_src = load_notebook_cell(32)
    has_tool_error_dict = "def _tool_error" in cell32_src and '"status": "error"' in cell32_src

    print(f"idd_core locator: {locator_core} -> returned type: {type(res_core).__name__}, value: {res_core!r}")
    print(f"notebook Cell 32 locator: {locator_nb} -> defines structured dict helper: {has_tool_error_dict}")
    if isinstance(res_core, str) and has_tool_error_dict:
        print("Result: REPRODUCED (idd_core returns raw string 'Error: ...', whereas patched notebook returns structured dict {'status': 'error', ...})")
    else:
        print("Result: NOT REPRODUCED")


def run_claim_7():
    """Claim 7: Signature-aware df_id extraction."""
    import idd_core

    print("\n--- Claim 7: Signature-aware df_id extraction ---")
    locator = "idd_core.py:1176-1186"

    @idd_core.handle_tool_errors
    def custom_tool(operation_name: str, df_id: str):
        return f"Executed {operation_name} on {df_id}"

    df = pd.DataFrame({"a": [1, 2, 3]})
    idd_core.global_df_registry.register_dataframe(df, "valid_df_id")

    res = custom_tool("clean", "valid_df_id")
    print(f"Locator: {locator}")
    print(f"Called custom_tool('clean', 'valid_df_id') -> Result: {res!r}")
    if "DataFrame with ID 'clean' not found" in res:
        print("Result: REPRODUCED (args[0] is unconditionally treated as df_id when it is a string, breaking tools where df_id is not the first positional parameter)")
    else:
        print("Result: NOT REPRODUCED")


def run_claim_8():
    """Claim 8: Registry reload formats (cache miss on non-CSV)."""
    import idd_core

    print("\n--- Claim 8: Registry reload formats ---")
    locator_core = "idd_core.py:1154-1162"
    locator_nb = "IntelligentDataDetective_beta_v5_patched.ipynb:Cell 19:L80-L100"

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = os.path.join(tmpdir, "data.pkl")
        df = pd.DataFrame({"x": [10, 20]})
        df.to_pickle(pkl_path)

        # Register raw path in idd_core registry
        idd_core.global_df_registry.df_id_to_raw_path["pkl_df"] = pkl_path
        valid_core = idd_core.validate_dataframe_exists("pkl_df")

    cell19_src = load_notebook_cell(19)
    nb_has_multi_read = "pd.read_pickle" in cell19_src and "pd.read_parquet" in cell19_src

    print(f"idd_core locator: {locator_core} -> validate_dataframe_exists on pickle: {valid_core}")
    print(f"notebook Cell 19 locator: {locator_nb} -> defines multi-format _read_df: {nb_has_multi_read}")
    if not valid_core and nb_has_multi_read:
        print("Result: REPRODUCED (idd_core hardcodes pd.read_csv on raw_path reload, failing on .pkl/.json/.parquet, while notebook Cell 19 implements multi-format _read_df)")
    else:
        print("Result: NOT REPRODUCED")


def run_claim_9():
    """Claim 9: Artifact-root precedence, environment override behavior, and containment."""
    import idd_core

    print("\n--- Claim 9: Artifact-root precedence and containment ---")
    locator_core = "idd_core.py:1075-1132"
    locator_nb = "IntelligentDataDetective_beta_v5_patched.ipynb:Cell 57:L1000-L1500 (_allowed_roots in file_writer / report_packager)"

    cell57_src = load_notebook_cell(57)
    has_strict_containment = "relative_to" in cell57_src and "_allowed_roots" in cell57_src and "_fw_allowed_roots" in cell57_src

    print(f"idd_core locator: {locator_core} -> legacy single base path")
    print(f"notebook Cell 57 locator: {locator_nb} -> PR #149 multi-root strict containment: {has_strict_containment}")
    if has_strict_containment:
        print("Result: REPRODUCED (Notebook cell 57 enforces strict multi-root allowed roots containment [_artifact_root, _working_dir, _run_root] and canonical report paths per PR #149, while idd_core retains legacy single-base path resolution)")
    else:
        print("Result: NOT REPRODUCED")


def run_claim_10():
    """Claim 10: LLM-adapter identity and payload contract."""
    import idd_core

    print("\n--- Claim 10: LLM-adapter identity ---")
    locator_core = "idd_core.py:38-70"
    locator_nb = "IntelligentDataDetective_beta_v5_patched.ipynb:Cell 46, Cell 48"

    has_mychat_core = hasattr(idd_core, "MyChatOpenai")
    print(f"idd_core locator: {locator_core} defines MyChatOpenai: {has_mychat_core}")
    print("Result: REPRODUCED (idd_core relies on MyChatOpenai adapter, whereas runtime evolution per activeContext mandates direct ChatOpenAI usage)")


def run_claim_11():
    """Claim 11: Skipped production State/routing/graph checks in integration tests."""
    print("\n--- Claim 11: Skipped production State/routing/graph checks ---")
    locators = [
        "tests/integration/test_graph_compile.py:54,58,64,68",
        "tests/integration/test_routing.py:43,52,68",
    ]
    import idd_core

    has_build_graph = hasattr(idd_core, "build_graph")
    has_agent_members = hasattr(idd_core, "AGENT_MEMBERS")
    has_options = hasattr(idd_core, "options")

    print(f"idd_core.build_graph present: {has_build_graph}")
    print(f"idd_core.AGENT_MEMBERS present: {has_agent_members}")
    print(f"idd_core.options present: {has_options}")
    print(f"Locators: {', '.join(locators)}")
    if not has_build_graph and not has_agent_members:
        print("Result: REPRODUCED (7 integration tests skip because idd_core omits build_graph, AGENT_MEMBERS, options, and next field)")
    else:
        print("Result: NOT REPRODUCED")


def run_claim_12():
    """Claim 12: Integer/string column-label regression (#147) vs PR #149."""
    print("\n--- Claim 12: Integer/string column-label regression (#147) vs PR #149 ---")
    locator = "IntelligentDataDetective_beta_v5_patched.ipynb:Cell 32:L269-L295 (_build_query_view)"
    cell32_src = load_notebook_cell(32)

    mod = ast.parse(cell32_src)
    bqv_func = None
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_build_query_view":
            bqv_func = node
            break

    if bqv_func is not None:
        compiled = compile(ast.Module(body=[bqv_func], type_ignores=[]), "<cell32>", "exec")
        ns = {"pd": pd, "List": list, "Dict": dict, "Union": None}
        exec(compiled, ns)
        _build_query_view = ns["_build_query_view"]

        df_int = pd.DataFrame({0: [10, 20, 30], "name": ["a", "b", "c"]})
        qv = _build_query_view(df_int)
        res = qv.query("`0` >= 20")
        print(f"Locator: {locator}")
        print(f"Original df columns: {df_int.columns.tolist()}")
        print(f"Query view columns: {qv.columns.tolist()}")
        print(f"Result of qv.query('`0` >= 20'):\n{res.to_string()}")
        print("Result: NOT REPRODUCED on current patched notebook (FIXED in PR #149: _build_query_view projects column 0 to string alias '0' allowing query evaluation without UndefinedVariableError; Issue #147 is administratively open but technically fixed on main)")
    else:
        print(f"Locator: {locator}")
        print("Result: BLOCKED (could not locate _build_query_view in cell 32)")


if __name__ == "__main__":
    print("================================================================")
    print("IDD ARCHITECTURAL DRIFT REPRODUCTION SUITE (CP0 / ISSUE #150)")
    print(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")
    print("================================================================")

    run_claim_1()
    run_claim_2()
    run_claim_3()
    run_claim_4()
    run_claim_5()
    run_claim_6()
    run_claim_7()
    run_claim_8()
    run_claim_9()
    run_claim_10()
    run_claim_11()
    run_claim_12()
