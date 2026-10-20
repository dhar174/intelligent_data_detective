"""
tests/unit/test_core_parity_guard.py - Anti-Drift Guard for Canonical Notebook Parity.

Enforces:
1. Static AST validation:
   - Exactly 99 notebook cells in IntelligentDataDetective_beta_v5_patched.ipynb.
   - Canonical imports in Cells 4, 7, 10, 16, 19, 22, 32, 46, 48, 60, 94, and 96.
   - Zero shadowing / zero duplicate redefinitions / zero illegal rebinding across all 99 cells
     for canonical protected classes, functions, reducers, and routing helpers.
2. Runtime symbol identity (resilient to missing optional notebook packages like IPython):
   - Evaluated notebook cell symbols are identical object references (is) to canonical modules.
   - global_df_registry is the canonical DataFrameRegistry instance and get_global_df_registry
     is the canonical scoped accessor.
   - CLASS_TO_AGENT in Cell 22 is identical to idd_state.CLASS_TO_AGENT and idd_core.CLASS_TO_AGENT.
3. Effective generated cell execution & behavioral parity:
   - Actual Cell 4 execution fails closed on tampered manifest, missing module, foreign cached
     module in sys.modules, and promotes repo_root to sys.path[0] ahead of competing paths.
   - Actual Cell 7 + Cell 32 default artifact-root parity (WORKING_DIRECTORY synchronized).
   - Actual Cell 32 decorated tool execution under override_global_registry.
   - Actual Cell 46 make_supervisor_node binds canonical idd_graph.Router.
   - Actual Cell 57 emergency_correspondence_node routes via CLASS_TO_AGENT without NameError.
   - Actual full Cell 60 + Cell 94 + Cell 96 execution in both SQLite-present and SQLite-absent modes.
4. Patcher input-drift preconditions & guard mutation controls:
   - verify_cp4_input_cell fails closed with RequiredPatchError on unexpected symbols or missing anchors.
   - Guard-level mutation tests clone the 99-cell notebook and verify the guard itself raises AssertionError.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import json
import logging
import os
import shutil
import sqlite3
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple
from unittest.mock import MagicMock

import pandas as pd
import pytest

import _patch_notebook
import idd_bootstrap
import idd_core
import idd_graph
import idd_models
import idd_registry
import idd_state
import idd_tools

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHED_NOTEBOOK = REPO_ROOT / "IntelligentDataDetective_beta_v5_patched.ipynb"
MANIFEST_FILE = REPO_ROOT / "idd_manifest.json"

PROTECTED_CLASSES = {
    "Plan",
    "State",
    "VizWorkerState",
    "DataFrameRegistry",
    "Router",
    "AgentMembers",
    "MyChatOpenai",
    "BaseNoExtrasModel",
}
PROTECTED_FUNCTIONS = {
    # Tool decorators & path security
    "handle_tool_errors",
    "_resolve_artifact_path",
    "validate_dataframe_exists",
    "_get_artifacts_base",
    "_is_subpath",
    "is_safe_artifact_path",
    "_tool_error",
    "_tool_failure",
    # Registry accessors
    "get_global_registry",
    "get_global_df_registry",
    "override_global_registry",
    "set_global_registry",
    # Graph & routing (Cell 60)
    "build_graph",
    "route_from_supervisor",
    "route_to_writer",
    "write_output_to_file",
    # Reducers
    "keep_first",
    "bool_or",
    "dict_merge_shallow",
    "merge_lists",
    "merge_unique",
    "merge_int_sum",
    "merge_dicts",
    "merge_dict",
    "any_true",
    "last_wins",
    "_sr_reducer",
    "_reduce_strs_resettable",
    "_reduce_viz_results_resettable",
    "_keep_last_or_clear",
    "_reduce_plan_keep_sorted",
    # LLM payload helper
    "_construct_responses_api_payload",
}
PROTECTED_SYMBOLS = PROTECTED_CLASSES | PROTECTED_FUNCTIONS


def load_patched_notebook() -> Dict[str, Any]:
    assert PATCHED_NOTEBOOK.is_file(), f"Patched notebook not found: {PATCHED_NOTEBOOK}"
    with open(PATCHED_NOTEBOOK, "r", encoding="utf-8") as f:
        return json.load(f)


def find_defined_symbols(
    cell_source: str,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return sets of (defined_classes, defined_functions, assigned_names) in cell_source using AST."""
    classes: Set[str] = set()
    functions: Set[str] = set()
    assignments: Set[str] = set()
    try:
        tree = ast.parse(cell_source)
    except SyntaxError:
        return classes, functions, assignments

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assignments.add(t.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                assignments.add(node.target.id)
    return classes, functions, assignments


def check_notebook_zero_shadowing(notebook: Dict[str, Any]) -> List[str]:
    """Return a list of shadowing violations across all code cells in notebook."""
    violations: List[str] = []
    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        classes, functions, assignments = find_defined_symbols(src)

        shadowed_classes = classes.intersection(PROTECTED_CLASSES)
        if shadowed_classes:
            violations.append(
                f"Cell {idx} redefines protected class(es): {sorted(shadowed_classes)}"
            )

        shadowed_functions = functions.intersection(PROTECTED_FUNCTIONS)
        if shadowed_functions:
            violations.append(
                f"Cell {idx} redefines protected function(s): {sorted(shadowed_functions)}"
            )

        shadowed_assignments = assignments.intersection(PROTECTED_SYMBOLS)
        if shadowed_assignments:
            violations.append(
                f"Cell {idx} rebinds protected symbol(s) via assignment: {sorted(shadowed_assignments)}"
            )

    return violations


def assert_notebook_static_parity(notebook: Dict[str, Any]) -> None:
    """Enforce static notebook structure, zero shadowing, and Cell 96 non-overwrite invariants."""
    cells = notebook.get("cells", [])
    assert (
        len(cells) == 99
    ), f"Expected exactly 99 cells in W14 baseline, got {len(cells)}"

    violations = check_notebook_zero_shadowing(notebook)
    assert (
        not violations
    ), "Found prohibited symbol redefinitions in notebook:\n" + "\n".join(violations)

    # Cell 46 must not define a local Router class
    src46 = "".join(cells[46].get("source", []))
    tree46 = ast.parse(src46)
    for node in ast.walk(tree46):
        if isinstance(node, ast.ClassDef):
            assert (
                node.name != "Router"
            ), "Cell 46 defines local class Router inside supervisor node"

    # Cell 96 must not overwrite data_detective_graph
    src96 = "".join(cells[96].get("source", []))
    tree96 = ast.parse(src96)
    assigned96 = set()
    for node in ast.walk(tree96):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned96.add(target.id)
    assert (
        "data_detective_graph" not in assigned96
    ), "Cell 96 overwrites production data_detective_graph"
    assert "_restored_graph" in assigned96


def _is_module_available(mod_name: str) -> bool:
    if mod_name in sys.modules:
        return sys.modules[mod_name] is not None
    try:
        return importlib.util.find_spec(mod_name) is not None
    except (ImportError, ValueError):
        return False


@contextlib.contextmanager
def sandbox_optional_imports(force_stub_ipython: bool = False) -> Iterator[None]:
    """Provide lightweight module stubs for optional external notebook packages when missing in CI (P1-A)."""
    saved_modules: Dict[str, Any] = {}
    added_modules: List[str] = []

    def _install_stub(name: str, attrs: Dict[str, Any]) -> None:
        if name in sys.modules and name not in saved_modules:
            saved_modules[name] = sys.modules[name]
        elif name not in sys.modules:
            added_modules.append(name)
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

    try:
        if force_stub_ipython or not _is_module_available("IPython.display"):
            _install_stub("IPython", {"get_ipython": lambda: None})
            _install_stub(
                "IPython.display",
                {"Image": lambda *a, **kw: None, "display": lambda *a, **kw: None},
            )
            sys.modules["IPython"].display = sys.modules["IPython.display"]  # type: ignore[attr-defined]

        if not _is_module_available("kagglehub"):
            _install_stub("kagglehub", {"dataset_download": lambda *a, **kw: ""})

        if not _is_module_available("tavily"):
            _install_stub("tavily", {"TavilyClient": MagicMock})

        if not _is_module_available("langmem"):
            _install_stub(
                "langmem",
                {
                    "create_manage_memory_tool": lambda *a, **kw: MagicMock(),
                    "create_search_memory_tool": lambda *a, **kw: MagicMock(),
                },
            )

        if not _is_module_available("langchain_community.agent_toolkits"):
            _dummy_tk = MagicMock()
            _dummy_tk.get_tools.return_value = []
            _install_stub("langchain_community", {})
            _install_stub(
                "langchain_community.agent_toolkits",
                {"FileManagementToolkit": lambda *a, **kw: _dummy_tk},
            )

        yield
    finally:
        for name in added_modules:
            sys.modules.pop(name, None)
        for name, orig in saved_modules.items():
            sys.modules[name] = orig


def execute_notebook_namespace(force_stub_ipython: bool = False) -> Dict[str, Any]:
    """Execute canonical notebook cells in a clean sandbox namespace while preserving global WORKING_DIRECTORY."""
    nb = load_patched_notebook()
    ns: Dict[str, Any] = {
        "__file__": str(PATCHED_NOTEBOOK),
        "logging": logging,
        "use_local_llm": False,
        "PathlibPath": Path,
    }
    orig_tools_wd = idd_tools.WORKING_DIRECTORY
    orig_reg_wd = idd_registry.WORKING_DIRECTORY
    orig_core_wd = idd_core.WORKING_DIRECTORY
    try:
        with sandbox_optional_imports(force_stub_ipython=force_stub_ipython):
            for idx in [7, 10, 16, 19, 22]:
                src = "".join(nb["cells"][idx].get("source", []))
                exec(src, ns)

            for idx in [32, 46, 60]:
                tree = ast.parse("".join(nb["cells"][idx].get("source", [])))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        stmt = ast.unparse(node)
                        try:
                            exec(stmt, ns)
                        except (ImportError, ModuleNotFoundError):
                            pass
    finally:
        idd_tools.WORKING_DIRECTORY = orig_tools_wd
        idd_registry.WORKING_DIRECTORY = orig_reg_wd
        idd_core.WORKING_DIRECTORY = orig_core_wd
        temp_dir = ns.get("_TEMP_DIRECTORY")
        if temp_dir is not None and hasattr(temp_dir, "cleanup"):
            try:
                temp_dir.cleanup()
            except Exception:
                pass

    return ns


# ===========================================================================
# 1. Static Notebook Parity & Zero Shadowing
# ===========================================================================


class TestStaticNotebookParity:
    @pytest.fixture(scope="class")
    @classmethod
    def notebook(cls) -> Dict[str, Any]:
        return load_patched_notebook()

    def test_notebook_cell_count(self, notebook: Dict[str, Any]):
        cells = notebook.get("cells", [])
        assert (
            len(cells) == 99
        ), f"Expected exactly 99 cells in W14 baseline, got {len(cells)}"

    def test_cell_4_canonical_bootstrap(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][4]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-BOOTSTRAP" in src
        assert "from idd_bootstrap import bootstrap_environment" in src
        assert "bootstrap_environment(repo_root=_repo_root)" in src
        assert 'sys.modules.get("idd_bootstrap")' in src
        assert "sys.path[:] = [_repo_root_str]" in src
        assert "except Exception as _b_err" not in src

    def test_cell_7_canonical_agent_members_and_reducers(
        self, notebook: Dict[str, Any]
    ):
        cell = notebook["cells"][7]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-WORKING-DIR-SYNC" in src
        assert "# PATCH: CP4-CANONICAL-CELL7-REDUCERS" in src
        assert "# PATCH: CP4-CANONICAL-AGENT-MEMBERS" in src
        assert "from idd_state import (" in src
        assert "AgentMembers" in src
        assert "def bool_or(" not in src
        assert "def keep_first(" not in src
        assert "def dict_merge_shallow(" not in src

    def test_cell_10_canonical_mychatopenai(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][10]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CANONICAL-MYCHATOPENAI" in src
        assert "from idd_core import" in src
        assert "MyChatOpenai" in src

    def test_cell_16_canonical_models(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][16]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CANONICAL-MODELS" in src
        assert "from idd_models import" in src
        assert "BaseNoExtrasModel" in src
        assert "Plan" in src
        assert "from idd_state import" in src
        assert "AnalysisConfig" in src

    def test_cell_19_canonical_registry(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][19]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CANONICAL-REGISTRY" in src
        assert "from idd_registry import" in src
        assert "DataFrameRegistry" in src
        assert "get_global_registry" in src
        assert "get_global_df_registry" in src
        assert "global_df_registry" in src
        assert "def get_global_df_registry" not in src

    def test_cell_22_canonical_state_and_class_to_agent(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][22]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CANONICAL-STATE" in src
        assert "from idd_state import" in src
        assert "State" in src
        assert "VizWorkerState" in src
        assert "CLASS_TO_AGENT" in src
        assert "keep_first" in src
        assert "bool_or" in src

    def test_cell_32_canonical_tools_and_scoped_registry(
        self, notebook: Dict[str, Any]
    ):
        cell = notebook["cells"][32]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CANONICAL-TOOLS" in src
        assert "# PATCH: CP4-CELL32-SCOPED-REGISTRY" in src
        assert "from idd_tools import" in src
        assert "handle_tool_errors" in src
        assert "validate_dataframe_exists" in src
        assert "_resolve_artifact_path" in src
        assert "def handle_tool_errors(" not in src
        assert "def _resolve_artifact_path(" not in src
        assert "global_df_registry." not in src
        assert "get_global_df_registry()." in src

    def test_cell_46_supervisor_router_no_local_class(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][46]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-SUPERVISOR-ROUTER" in src
        assert "from idd_graph import Router" in src
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert (
                    node.name != "Router"
                ), "Cell 46 defines local class Router inside supervisor node"

    def test_cell_48_scoped_registry_registration(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][48]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CELL48-SCOPED-REGISTRY" in src
        assert "get_global_df_registry().register_dataframe(" in src
        assert "global_df_registry.register_dataframe(" not in src

    def test_cell_60_canonical_graph(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][60]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CANONICAL-GRAPH" in src
        assert "from idd_graph import (" in src
        assert "build_graph" in src
        assert "structural_only=False" in src
        assert "node_cache_policies=CachePolicy(ttl=120)" in src
        assert src.count("_SqliteSaver(_cp_conn)") == 1
        assert "checkpointer = MemorySaver()" in src

    def test_cell_94_sqlite_guard_and_isolated_demo_db(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][94]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CELL94-GUARD" in src
        assert "except (ImportError, ModuleNotFoundError)" in src
        assert "checkpoints_demo.sqlite" in src
        # Issue #164: lossless checkpoint-level migration replaces the legacy update_state()
        # replay that skipped every snapshot lacking metadata["writes"].
        assert "# PATCH: ISSUE164-CELL94-LOSSLESS-MIGRATE" in src
        assert "dst_cp.put(" in src and "dst_cp.put_writes(" in src
        assert "update_state(" not in src
        assert "_migrated_checkpoint_count = migrate_thread(" in src

    def test_cell_96_restored_graph_no_overwrite_and_closes_conn(
        self, notebook: Dict[str, Any]
    ):
        cell = notebook["cells"][96]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CELL96-RECONCILE" in src
        assert "checkpoints_demo.sqlite" in src
        assert "_restore_conn.close()" in src
        # Issue #164: restored state is read while the backing connection is still open.
        assert "# PATCH: ISSUE164-CELL96-VERIFIED-RESTORE" in src
        assert src.index("_restored_graph.get_state(") < src.index(
            "_restore_conn.close()"
        )
        assert_notebook_static_parity(notebook)

    def test_zero_shadowing_across_all_99_cells(self, notebook: Dict[str, Any]):
        assert_notebook_static_parity(notebook)


# ===========================================================================
# 2. Runtime Symbol Identity (Including Missing-IPython Resilience — P1-A)
# ===========================================================================


class TestRuntimeSymbolIdentity:
    @pytest.fixture(scope="class")
    @classmethod
    def notebook_namespace(cls) -> Dict[str, Any]:
        return execute_notebook_namespace()

    def test_notebook_namespace_resilient_to_missing_ipython(self, monkeypatch):
        """P1-A: Executing Cell 7 and canonical cells succeeds when IPython is absent in CI."""
        monkeypatch.setitem(sys.modules, "IPython", None)
        monkeypatch.setitem(sys.modules, "IPython.display", None)
        ns = execute_notebook_namespace(force_stub_ipython=True)
        assert ns["Plan"] is idd_models.Plan
        assert ns["State"] is idd_state.State
        assert ns["CLASS_TO_AGENT"] is idd_state.CLASS_TO_AGENT

    def test_model_identity(self, notebook_namespace: Dict[str, Any]):
        assert notebook_namespace["Plan"] is idd_models.Plan
        assert notebook_namespace["BaseNoExtrasModel"] is idd_models.BaseNoExtrasModel
        assert notebook_namespace["PlanStep"] is idd_models.PlanStep

    def test_state_and_class_to_agent_identity(
        self, notebook_namespace: Dict[str, Any]
    ):
        assert notebook_namespace["State"] is idd_state.State
        assert notebook_namespace["VizWorkerState"] is idd_state.VizWorkerState
        assert notebook_namespace["AnalysisConfig"] is idd_state.AnalysisConfig
        assert notebook_namespace["CLASS_TO_AGENT"] is idd_state.CLASS_TO_AGENT
        assert notebook_namespace["CLASS_TO_AGENT"] is idd_core.CLASS_TO_AGENT
        assert notebook_namespace["keep_first"] is idd_state.keep_first
        assert notebook_namespace["bool_or"] is idd_state.bool_or
        assert notebook_namespace["dict_merge_shallow"] is idd_state.dict_merge_shallow

    def test_registry_identity(self, notebook_namespace: Dict[str, Any]):
        assert notebook_namespace["DataFrameRegistry"] is idd_registry.DataFrameRegistry
        assert (
            notebook_namespace["get_global_registry"]
            is idd_registry.get_global_registry
        )
        assert (
            notebook_namespace["get_global_df_registry"]
            is idd_registry.get_global_df_registry
        )
        assert (
            notebook_namespace["global_df_registry"] is idd_registry.global_df_registry
        )
        assert idd_registry.global_df_registry is idd_registry.get_global_registry()
        assert isinstance(
            idd_registry.global_df_registry, idd_registry.DataFrameRegistry
        )
        assert (
            notebook_namespace["get_global_df_registry"]()
            is idd_registry.get_global_registry()
        )

    def test_tools_identity(self, notebook_namespace: Dict[str, Any]):
        assert notebook_namespace["handle_tool_errors"] is idd_tools.handle_tool_errors
        assert (
            notebook_namespace["validate_dataframe_exists"]
            is idd_tools.validate_dataframe_exists
        )
        assert (
            notebook_namespace["_resolve_artifact_path"]
            is idd_tools._resolve_artifact_path
        )
        assert (
            notebook_namespace["is_safe_artifact_path"]
            is idd_tools.is_safe_artifact_path
        )

    def test_graph_and_router_identity(self, notebook_namespace: Dict[str, Any]):
        assert notebook_namespace["build_graph"] is idd_graph.build_graph
        assert notebook_namespace["Router"] is idd_graph.Router

    def test_agent_members_and_mychatopenai_identity(
        self, notebook_namespace: Dict[str, Any]
    ):
        assert notebook_namespace["AgentMembers"] is idd_state.AgentMembers
        assert notebook_namespace["MyChatOpenai"] is idd_core.MyChatOpenai


# ===========================================================================
# 3. Effective Generated Cell Execution & Behavioral Parity (P1-B/C/D, P2-1/4/5)
# ===========================================================================


class TestBehavioralParity:
    @pytest.fixture
    def tools_ns(self) -> Dict[str, Any]:
        nb = load_patched_notebook()
        ns: Dict[str, Any] = {
            "__file__": str(PATCHED_NOTEBOOK),
            "logging": logging,
        }
        for idx in [16, 19, 22]:
            exec("".join(nb["cells"][idx].get("source", [])), ns)

        tree = ast.parse("".join(nb["cells"][32].get("source", [])))
        import_nodes = [
            n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        exec(ast.unparse(ast.Module(body=import_nodes, type_ignores=[])), ns)
        return ns

    def test_codex_finding_1_positional_string_non_df_id(
        self, tools_ns: Dict[str, Any]
    ):
        """Tool without df_id parameter does NOT attempt DataFrame validation even if given string."""
        decorator = tools_ns["handle_tool_errors"]

        @decorator
        def compute_column_metrics(column_name: str, scale: float = 1.0):
            return {"column": column_name, "scaled_metric": 42.0 * scale}

        res = compute_column_metrics("revenue", 2.5)
        assert isinstance(res, dict)
        assert res.get("column") == "revenue"
        assert res.get("scaled_metric") == 105.0
        assert res.get("status") != "error"

    def test_codex_finding_2_subdir_escape_pre_mkdir_rejection(
        self, tools_ns: Dict[str, Any], tmp_path: Path
    ):
        """Path traversal via subdir is rejected before creating any directory on disk."""
        resolve_path = tools_ns["_resolve_artifact_path"]
        target_root = tmp_path / "artifacts"
        target_root.mkdir()

        canary_dir = tmp_path / "should_not_exist"
        assert not canary_dir.exists()

        with pytest.raises(ValueError, match="escapes artifacts root"):
            resolve_path(
                "plot.png",
                config={
                    "configurable": {"runtime": {"artifacts_dir": str(target_root)}}
                },
                subdir=f"../{canary_dir.name}",
            )

        assert (
            not canary_dir.exists()
        ), "Subdir escape created unauthorized directory before validation!"

    def test_cell_7_and_32_default_artifact_root_parity(self, monkeypatch):
        """P2-4: Cell 7 WORKING_DIRECTORY sync ensures _resolve_artifact_path defaults to WORKING_DIRECTORY."""
        monkeypatch.delenv("IDD_ARTIFACTS_DIR", raising=False)
        nb = load_patched_notebook()
        ns: Dict[str, Any] = {
            "__file__": str(PATCHED_NOTEBOOK),
            "logging": logging,
            "use_local_llm": False,
            "PathlibPath": Path,
        }
        orig_tools_wd = idd_tools.WORKING_DIRECTORY
        orig_reg_wd = idd_registry.WORKING_DIRECTORY
        orig_core_wd = idd_core.WORKING_DIRECTORY
        try:
            with sandbox_optional_imports():
                exec("".join(nb["cells"][7].get("source", [])), ns)
                wd = ns["WORKING_DIRECTORY"].resolve()
                assert idd_tools.WORKING_DIRECTORY.resolve() == wd
                assert idd_registry.WORKING_DIRECTORY.resolve() == wd
                assert idd_core.WORKING_DIRECTORY.resolve() == wd

                resolved = idd_tools._resolve_artifact_path("report_check.md")
                assert resolved.parent.resolve() == (wd / "artifacts").resolve()
                assert idd_tools._is_subpath(resolved, wd)
        finally:
            idd_tools.WORKING_DIRECTORY = orig_tools_wd
            idd_registry.WORKING_DIRECTORY = orig_reg_wd
            idd_core.WORKING_DIRECTORY = orig_core_wd
            temp_dir = ns.get("_TEMP_DIRECTORY")
            if temp_dir is not None and hasattr(temp_dir, "cleanup"):
                temp_dir.cleanup()

    def test_actual_cell_32_decorated_tool_with_scoped_registry(self, tmp_path: Path):
        """P1-D & P2-1: Actual generated Cell 32 tool resolves ContextVar-scoped registry end-to-end."""
        nb = load_patched_notebook()
        ns: Dict[str, Any] = {
            "__file__": str(PATCHED_NOTEBOOK),
            "logging": logging,
            "use_local_llm": False,
            "PathlibPath": Path,
            "contextmanager": contextlib.contextmanager,
            "openai_api_key": "test-key",
            "tavily_api_key": "",
        }
        orig_tools_wd = idd_tools.WORKING_DIRECTORY
        orig_reg_wd = idd_registry.WORKING_DIRECTORY
        orig_core_wd = idd_core.WORKING_DIRECTORY
        try:
            with sandbox_optional_imports():
                for idx in [7, 10, 16, 19, 22, 31, 32]:
                    exec("".join(nb["cells"][idx].get("source", [])), ns)

                drop_col_tool = ns["drop_column"]
                schema_tool = ns["get_dataframe_schema"]

                scoped_reg = idd_registry.DataFrameRegistry(
                    data_dir=tmp_path / "scoped_cell32"
                )
                df = pd.DataFrame({"keep_col": [10, 20, 30], "drop_me": [1, 2, 3]})
                scoped_id = scoped_reg.register_dataframe(df, "scoped_df")

                with idd_registry.override_global_registry(scoped_reg):
                    # Invoke actual decorated Cell 32 tool function
                    drop_fn = getattr(drop_col_tool, "func", drop_col_tool)
                    msg = drop_fn(df_id=scoped_id, column_name="drop_me")
                    assert "Column dropped successfully" in msg
                    updated_df = scoped_reg.get_dataframe(scoped_id)
                    assert list(updated_df.columns) == ["keep_col"]

                    schema_fn = getattr(schema_tool, "func", schema_tool)
                    content, artifact = schema_fn(df_id=scoped_id)
                    assert "keep_col" in content
                    assert isinstance(artifact, dict)

                # Outside override, scoped_id does not exist in default registry
                outside_res = getattr(drop_col_tool, "func", drop_col_tool)(
                    df_id=scoped_id, column_name="keep_col"
                )
                assert isinstance(outside_res, dict)
                assert outside_res.get("status") == "error"
        finally:
            idd_tools.WORKING_DIRECTORY = orig_tools_wd
            idd_registry.WORKING_DIRECTORY = orig_reg_wd
            idd_core.WORKING_DIRECTORY = orig_core_wd
            temp_dir = ns.get("_TEMP_DIRECTORY")
            if temp_dir is not None and hasattr(temp_dir, "cleanup"):
                temp_dir.cleanup()

    def test_actual_cell_46_make_supervisor_node_uses_canonical_router(self):
        """P1-D: Executing actual generated Cell 46 make_supervisor_node binds canonical idd_graph.Router."""
        import inspect

        nb = load_patched_notebook()
        ns = execute_notebook_namespace()
        exec("".join(nb["cells"][46].get("source", [])), ns)

        make_supervisor_node = ns["make_supervisor_node"]
        dummy_llms = [MagicMock() for _ in range(8)]
        node_fn = make_supervisor_node(
            dummy_llms,
            [
                "initial_analysis",
                "data_cleaner",
                "analyst",
                "file_writer",
                "visualization",
                "report_orchestrator",
            ],
            "Analyze the dataset",
        )
        assert callable(node_fn)
        inner_nodes = [
            c.cell_contents
            for c in (node_fn.__closure__ or ())
            if callable(c.cell_contents)
            and getattr(c.cell_contents, "__name__", "") == "supervisor_node"
        ]
        assert len(inner_nodes) == 1
        closure_vars = inspect.getclosurevars(inner_nodes[0])
        assert closure_vars.nonlocals.get("Router") is idd_graph.Router

    def test_actual_cell_57_emergency_correspondence_node_routes_via_class_to_agent(
        self,
    ):
        """P1-B & P1-D: Actual Cell 57 emergency_correspondence_node resolves CLASS_TO_AGENT on expect_reply=True."""
        nb = load_patched_notebook()
        ns = execute_notebook_namespace()

        cell57_src = "".join(nb["cells"][57].get("source", []))
        tree57 = ast.parse(cell57_src)
        selected_nodes = [
            node
            for node in tree57.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "emergency_correspondence_node"
        ]
        assert len(selected_nodes) == 1

        exec(ast.unparse(ast.Module(body=selected_nodes, type_ignores=[])), ns)
        emergency_fn = ns["emergency_correspondence_node"]

        insights = idd_state.AnalysisInsights(
            summary="Initial stats computed",
            correlation_insights="None",
            anomaly_insights="None",
            recommended_visualizations=[],
            recommended_next_steps=["Check nulls"],
            reply_msg_to_supervisor="Should we use median imputation?",
            finished_this_task=False,
            expect_reply=True,
        )
        pending_msg = idd_state.SendAgentMessage(
            recipient="analyst",
            message="Proceed with median imputation.",
            delivery_status=False,
            agent_obj_needs_recreated_bool=False,
            is_message_critical=False,
            immediate_emergency_reroute_to_recipient=False,
            reply_msg_to_supervisor="Replied to analyst",
            finished_this_task=True,
            expect_reply=False,
        )
        state = {
            "supervisor_to_agent_msgs": [pending_msg],
            "analysis_insights": insights,
            "last_created_obj": "analysis_insights",
            "next": "EMERGENCY_MSG",
        }
        cmd = emergency_fn(state)
        assert cmd.goto == "analyst"
        assert cmd.update["messages"].content == "Proceed with median imputation."

        # Negative control: without CLASS_TO_AGENT in namespace, emergency_correspondence_node raises NameError
        saved_cta = ns.pop("CLASS_TO_AGENT")
        try:
            with pytest.raises(NameError, match="CLASS_TO_AGENT"):
                emergency_fn(state)
        finally:
            ns["CLASS_TO_AGENT"] = saved_cta

    @pytest.mark.parametrize("sqlite_available", [True, False])
    def test_actual_cells_60_94_96_execute_sqlite_present_and_absent(
        self, tmp_path: Path, monkeypatch, sqlite_available: bool
    ):
        """P1-D & P2-5: Full generated Cells 60, 94, and 96 execute cleanly in both SQLite-present and absent modes."""
        monkeypatch.chdir(tmp_path)
        nb = load_patched_notebook()
        ns = execute_notebook_namespace()
        ns["WORKING_DIRECTORY"] = tmp_path

        # Populate dummy node callables and LLMs required by Cell 60
        ns["make_supervisor_node"] = lambda llms, members, prompt: (lambda state: {})
        for llm_name in [
            "big_picture_llm",
            "router_llm",
            "reply_llm",
            "plan_llm",
            "replan_llm",
            "progress_llm",
            "todo_llm",
            "low_reasoning_llm",
        ]:
            ns[llm_name] = MagicMock()
        ns["sample_prompt_text"] = "Analyze dataset"
        for node_var in [
            "initial_analysis_node",
            "data_cleaner_node",
            "analyst_node",
            "visualization_orchestrator",
            "viz_worker",
            "viz_join",
            "viz_evaluator_node",
            "report_orchestrator",
            "section_worker",
            "report_join",
            "report_packager_node",
            "file_writer_node",
            "emergency_correspondence_node",
        ]:
            ns[node_var] = lambda state: {}
        ns["in_memory_store"] = ns["InMemoryStore"]()
        ns["run_config"] = {"configurable": {"thread_id": "demo-thread"}}

        if not sqlite_available:
            monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", None)
        else:
            # Ensure langgraph.checkpoint.sqlite is available (use real if installed, or lightweight MemorySaver adapter)
            if not _is_module_available("langgraph.checkpoint.sqlite"):
                from langgraph.checkpoint.memory import MemorySaver

                class _FakeSqliteSaver(MemorySaver):
                    def __init__(self, conn=None):
                        super().__init__()
                        self.conn = conn

                    @classmethod
                    @contextlib.contextmanager
                    def from_conn_string(cls, conn_str: str):
                        yield cls(conn_str)

                fake_mod = types.ModuleType("langgraph.checkpoint.sqlite")
                fake_mod.SqliteSaver = _FakeSqliteSaver  # type: ignore[attr-defined]
                monkeypatch.setitem(
                    sys.modules, "langgraph.checkpoint.sqlite", fake_mod
                )

        try:
            exec("".join(nb["cells"][60].get("source", [])), ns)
            original_graph = ns["data_detective_graph"]
            assert original_graph is not None
            assert not original_graph.is_structural_only

            if not sqlite_available:
                from langgraph.checkpoint.memory import MemorySaver

                assert isinstance(ns["checkpointer"], MemorySaver)

            # Execute full Cell 94
            exec("".join(nb["cells"][94].get("source", [])), ns)
            assert ns["data_detective_graph"] is original_graph

            # Execute full Cell 96
            exec("".join(nb["cells"][96].get("source", [])), ns)
            assert (
                ns["data_detective_graph"] is original_graph
            ), "Cell 96 overwrote data_detective_graph!"
            if sqlite_available:
                assert "_restored_graph" in ns
                assert ns["_restored_graph"] is not original_graph
        finally:
            cp_conn = ns.get("_cp_conn")
            if isinstance(cp_conn, sqlite3.Connection):
                cp_conn.close()

    def test_supervisor_router_schema_contract(self):
        """Router schema matches SupervisorRouteTarget and requires metadata nullable field."""
        Router = idd_graph.Router
        schema = Router.model_json_schema()
        assert set(schema.get("required", [])) >= {
            "next",
            "next_agent_prompt",
            "next_agent_metadata",
        }
        r = Router(
            next="EMERGENCY_MSG",
            next_agent_prompt="Stop now",
            next_agent_metadata=None,
            reply_msg_to_supervisor="Stop",
            finished_this_task=True,
            expect_reply=False,
        )
        assert r.next == "EMERGENCY_MSG"
        assert r.next_agent_metadata is None

        with pytest.raises(Exception):
            Router(
                next="analyst",
                next_agent_prompt="Analyze data",
                reply_msg_to_supervisor="Analyzing",
                finished_this_task=False,
                expect_reply=True,
            )

    def test_mychatopenai_filters_prompt_cache_key(self):
        """MyChatOpenai internal kwargs filter strips prompt_cache_key."""
        adapter = idd_core.MyChatOpenai(
            model="gpt-4o",
            api_key="test-key",
        )
        assert "prompt_cache_key" in adapter._INTERNAL_KWARGS
        payload = adapter._get_request_payload_mod(
            "test", prompt_cache_key="cached-key"
        )
        assert "prompt_cache_key" not in payload


# ===========================================================================
# 4. Negative Controls: Cell 4 Bootstrap, Origin, Patcher & Guard Mutations
# ===========================================================================


def _create_repo_sandbox(tmp_path: Path) -> Path:
    """Create a complete copy of canonical modules and idd_manifest.json in tmp_path."""
    sandbox = tmp_path / "repo_sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    shutil.copy(MANIFEST_FILE, sandbox / "idd_manifest.json")
    for mod_file in idd_bootstrap.CANONICAL_MODULES:
        shutil.copy(REPO_ROOT / mod_file, sandbox / mod_file)
    return sandbox


def _extract_cell_4_bootstrap_snippet() -> str:
    nb = load_patched_notebook()
    src4 = "".join(nb["cells"][4].get("source", []))
    idx = src4.index("# PATCH: CP4-BOOTSTRAP")
    return src4[idx:]


def _append_cell_source(cell: Dict[str, Any], snippet: str) -> None:
    if isinstance(cell["source"], list):
        cell["source"].append(snippet)
    else:
        cell["source"] = cell["source"] + snippet


class TestNegativeControls:
    def test_actual_cell_4_fails_closed_on_tampered_manifest(
        self, tmp_path: Path, monkeypatch
    ):
        """P1-C & P1-D: Executing actual generated Cell 4 raises RuntimeError when manifest is tampered."""
        sandbox = _create_repo_sandbox(tmp_path)
        manifest_path = sandbox / "idd_manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["files"]["idd_models.py"] = "0" * 64
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        monkeypatch.chdir(sandbox)
        for stem in idd_bootstrap.CANONICAL_MODULE_STEMS:
            monkeypatch.delitem(sys.modules, stem, raising=False)
        monkeypatch.setattr(sys, "path", [str(sandbox)] + list(sys.path))

        cell4_bootstrap = _extract_cell_4_bootstrap_snippet()
        with pytest.raises(RuntimeError, match="Digest mismatch for 'idd_models.py'"):
            exec(cell4_bootstrap, {})

    def test_actual_cell_4_fails_closed_on_tampered_bootstrap_script(
        self, tmp_path: Path, monkeypatch
    ):
        """P1-C: Executing actual generated Cell 4 fails closed BEFORE importing idd_bootstrap if idd_bootstrap.py is tampered."""
        sandbox = _create_repo_sandbox(tmp_path)
        (sandbox / "idd_bootstrap.py").write_text(
            "# tampered bootstrap\n", encoding="utf-8"
        )

        monkeypatch.chdir(sandbox)
        for stem in idd_bootstrap.CANONICAL_MODULE_STEMS:
            monkeypatch.delitem(sys.modules, stem, raising=False)

        cell4_bootstrap = _extract_cell_4_bootstrap_snippet()
        with pytest.raises(
            RuntimeError,
            match="CP4 Bootstrap Integrity Error: idd_bootstrap.py SHA-256 mismatch",
        ):
            exec(cell4_bootstrap, {})

    def test_actual_cell_4_fails_closed_on_missing_canonical_module(
        self, tmp_path: Path, monkeypatch
    ):
        """P1-D: Executing actual generated Cell 4 raises FileNotFoundError when canonical module is missing."""
        sandbox = _create_repo_sandbox(tmp_path)
        (sandbox / "idd_models.py").unlink()

        monkeypatch.chdir(sandbox)
        for stem in idd_bootstrap.CANONICAL_MODULE_STEMS:
            monkeypatch.delitem(sys.modules, stem, raising=False)
        monkeypatch.setattr(sys, "path", [str(sandbox)] + list(sys.path))

        cell4_bootstrap = _extract_cell_4_bootstrap_snippet()
        with pytest.raises(
            FileNotFoundError, match="Canonical module 'idd_models.py' missing"
        ):
            exec(cell4_bootstrap, {})

    def test_actual_cell_4_rejects_foreign_cached_module_in_sys_modules(
        self, tmp_path: Path, monkeypatch
    ):
        """P1-C & P1-D: Executing actual generated Cell 4 fails closed if sys.modules has a foreign canonical module."""
        foreign_mod = types.ModuleType("idd_models")
        foreign_mod.__file__ = str(tmp_path / "foreign_idd_models.py")
        monkeypatch.setitem(sys.modules, "idd_models", foreign_mod)
        monkeypatch.chdir(REPO_ROOT)

        cell4_bootstrap = _extract_cell_4_bootstrap_snippet()
        with pytest.raises(
            RuntimeError,
            match="Loaded module 'idd_models' origin .* does not match verified repository root",
        ):
            exec(cell4_bootstrap, {})

    def test_actual_cell_4_promotes_repo_root_ahead_of_competing_sys_path(
        self, tmp_path: Path, monkeypatch
    ):
        """P1-C & P1-D: Executing actual generated Cell 4 places repo_root at sys.path[0] ahead of competing checkouts."""
        competing_dir = tmp_path / "competing_checkout"
        competing_dir.mkdir()
        (competing_dir / "idd_models.py").write_text(
            "raise RuntimeError('Loaded competing idd_models!')\n", encoding="utf-8"
        )

        # Put competing_dir at sys.path[0] and REPO_ROOT later in sys.path
        new_path = [
            str(competing_dir),
            *[p for p in sys.path if Path(p).resolve() != REPO_ROOT.resolve()],
            str(REPO_ROOT),
        ]
        monkeypatch.setattr(sys, "path", new_path)
        monkeypatch.chdir(REPO_ROOT)

        cell4_bootstrap = _extract_cell_4_bootstrap_snippet()
        exec(cell4_bootstrap, {})
        assert Path(sys.path[0]).resolve() == REPO_ROOT.resolve()

    def test_patcher_verify_cp4_input_cell_fails_closed_on_unexpected_symbol(self):
        """P2-2: verify_cp4_input_cell raises RequiredPatchError when an unexpected class, function, or assign is added."""
        valid_src = "class MyChatOpenai:\n    pass\ndef _construct_responses_api_payload():\n    pass\n"
        _patch_notebook.verify_cp4_input_cell(
            valid_src,
            cell_idx=10,
            patch_id="TEST-CP4",
            expected_classes={"MyChatOpenai"},
            expected_functions={"_construct_responses_api_payload"},
            expected_assigns=set(),
            required_anchors=("class MyChatOpenai:",),
        )

        with pytest.raises(
            _patch_notebook.RequiredPatchError, match="input function inventory drifted"
        ):
            _patch_notebook.verify_cp4_input_cell(
                valid_src + "\ndef extra_notebook_helper():\n    return 1\n",
                cell_idx=10,
                patch_id="TEST-CP4",
                expected_classes={"MyChatOpenai"},
                expected_functions={"_construct_responses_api_payload"},
                expected_assigns=set(),
            )

        with pytest.raises(
            _patch_notebook.RequiredPatchError,
            match="missing required input structural anchor",
        ):
            _patch_notebook.verify_cp4_input_cell(
                valid_src,
                cell_idx=10,
                patch_id="TEST-CP4",
                expected_classes={"MyChatOpenai"},
                expected_functions={"_construct_responses_api_payload"},
                expected_assigns=set(),
                required_anchors=("missing_anchor_token",),
            )

    def test_guard_mutation_controls_fail_on_mutated_notebook_clones(self):
        """P1-D: Mutating a clone of the 99-cell notebook causes the guard helper itself to raise AssertionError."""
        base_nb = load_patched_notebook()

        # 1. Mutate Cell 16 to redefine a protected class (Plan)
        nb_class_redef = copy.deepcopy(base_nb)
        _append_cell_source(nb_class_redef["cells"][16], "\nclass Plan:\n    pass\n")
        with pytest.raises(
            AssertionError,
            match="Cell 16 redefines protected class\\(es\\): \\['Plan'\\]",
        ):
            assert_notebook_static_parity(nb_class_redef)

        # 2. Mutate Cell 7 to redefine a protected reducer function (keep_first)
        nb_func_redef = copy.deepcopy(base_nb)
        _append_cell_source(
            nb_func_redef["cells"][7], "\ndef keep_first(a, b):\n    return a\n"
        )
        with pytest.raises(
            AssertionError,
            match="Cell 7 redefines protected function\\(s\\): \\['keep_first'\\]",
        ):
            assert_notebook_static_parity(nb_func_redef)

        # 3. Mutate Cell 32 to rebind a protected symbol via assignment
        nb_assign_rebind = copy.deepcopy(base_nb)
        _append_cell_source(
            nb_assign_rebind["cells"][32], "\nhandle_tool_errors = lambda f: f\n"
        )
        with pytest.raises(
            AssertionError,
            match="Cell 32 rebinds protected symbol\\(s\\) via assignment: \\['handle_tool_errors'\\]",
        ):
            assert_notebook_static_parity(nb_assign_rebind)

        # 4. Mutate Cell 96 to overwrite data_detective_graph
        nb_cell96_overwrite = copy.deepcopy(base_nb)
        _append_cell_source(
            nb_cell96_overwrite["cells"][96],
            "\ndata_detective_graph = _restored_graph\n",
        )
        with pytest.raises(
            AssertionError,
            match="Cell 96 overwrites production data_detective_graph",
        ):
            assert_notebook_static_parity(nb_cell96_overwrite)

        # 5. Mutate cell count (delete a cell)
        nb_wrong_count = copy.deepcopy(base_nb)
        nb_wrong_count["cells"].pop()
        with pytest.raises(
            AssertionError, match="Expected exactly 99 cells in W14 baseline, got 98"
        ):
            assert_notebook_static_parity(nb_wrong_count)
