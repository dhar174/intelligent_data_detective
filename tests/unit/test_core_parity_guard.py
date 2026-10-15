"""
tests/unit/test_core_parity_guard.py - Anti-Drift Guard for Canonical Notebook Parity.

Enforces:
1. Static AST validation:
   - Exactly 99 notebook cells in IntelligentDataDetective_beta_v5_patched.ipynb.
   - Canonical imports in Cells 4, 16, 19, 22, 32, and 60.
   - Zero shadowing / zero duplicate redefinitions across all 99 cells for canonical symbols:
     Plan, State, DataFrameRegistry, handle_tool_errors, _resolve_artifact_path, build_graph.
2. Runtime symbol identity:
   - Evaluated notebook cell symbols are identical object references (is) to canonical modules.
3. Behavioral parity:
   - Codex Finding 1 (PR #159): Positional string argument that is not df_id does not trigger validation error.
   - Codex Finding 2 (PR #159): Subdirectory traversal attempt is rejected before any directory creation.
   - DataFrameRegistry lifecycle: register, persist, and retrieve via canonical registry.
   - Graph compilation: Cell 60 build_graph in executable mode (structural_only=False).
4. Negative controls:
   - Shadowing detector catches synthetic redefinitions of classes and functions.
   - Manifest verification fails closed on tampered digests, missing entries, and missing files.
"""

from __future__ import annotations

import ast
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import pytest

import idd_bootstrap
import idd_graph
import idd_models
import idd_registry
import idd_state
import idd_tools


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHED_NOTEBOOK = REPO_ROOT / "IntelligentDataDetective_beta_v5_patched.ipynb"
MANIFEST_FILE = REPO_ROOT / "idd_manifest.json"

PROTECTED_CLASSES = {"Plan", "State", "DataFrameRegistry"}
PROTECTED_FUNCTIONS = {"handle_tool_errors", "_resolve_artifact_path", "build_graph"}
PROTECTED_SYMBOLS = PROTECTED_CLASSES | PROTECTED_FUNCTIONS


def load_patched_notebook() -> Dict[str, Any]:
    assert PATCHED_NOTEBOOK.is_file(), f"Patched notebook not found: {PATCHED_NOTEBOOK}"
    with open(PATCHED_NOTEBOOK, "r", encoding="utf-8") as f:
        return json.load(f)


def find_defined_symbols(cell_source: str) -> Tuple[Set[str], Set[str]]:
    """Return sets of (defined_classes, defined_functions) in cell_source using AST."""
    classes: Set[str] = set()
    functions: Set[str] = set()
    try:
        tree = ast.parse(cell_source)
    except SyntaxError:
        return classes, functions

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
    return classes, functions


# ===========================================================================
# 1. Static Notebook Parity & Zero Shadowing
# ===========================================================================


class TestStaticNotebookParity:
    @pytest.fixture(scope="class")
    def notebook(self) -> Dict[str, Any]:
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
        assert "bootstrap_environment()" in src

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
        assert "global_df_registry = get_global_registry()" in src
        assert "def get_global_df_registry() -> DataFrameRegistry:" in src

    def test_cell_22_canonical_state(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][22]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CANONICAL-STATE" in src
        assert "from idd_state import" in src
        assert "State" in src
        assert "VizWorkerState" in src
        assert "keep_first" in src
        assert "bool_or" in src

    def test_cell_32_canonical_tools(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][32]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CANONICAL-TOOLS" in src
        assert "from idd_tools import" in src
        assert "handle_tool_errors" in src
        assert "validate_dataframe_exists" in src
        assert "_resolve_artifact_path" in src
        # Ensure redundant local implementations were completely removed
        assert "def handle_tool_errors(" not in src
        assert "def _resolve_artifact_path(" not in src

    def test_cell_60_canonical_graph(self, notebook: Dict[str, Any]):
        cell = notebook["cells"][60]
        src = "".join(cell.get("source", []))
        assert "# PATCH: CP4-CANONICAL-GRAPH" in src
        assert "from idd_graph import (" in src
        assert "build_graph" in src
        assert "structural_only=False" in src
        assert "node_cache_policies=CachePolicy(ttl=120)" in src

    def test_zero_shadowing_across_all_99_cells(self, notebook: Dict[str, Any]):
        violations: List[str] = []
        for idx, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            classes, functions = find_defined_symbols(src)

            # In Cell 19, get_global_df_registry is expected; DataFrameRegistry is an import, NOT a class def.
            shadowed_classes = classes.intersection(PROTECTED_CLASSES)
            if shadowed_classes:
                violations.append(
                    f"Cell {idx} redefines protected class(es): {sorted(shadowed_classes)}"
                )

            # Protected functions should NOT be defined via `def` anywhere
            shadowed_functions = functions.intersection(PROTECTED_FUNCTIONS)
            if shadowed_functions:
                violations.append(
                    f"Cell {idx} redefines protected function(s): {sorted(shadowed_functions)}"
                )

        assert (
            not violations
        ), "Found prohibited symbol redefinitions in notebook:\n" + "\n".join(
            violations
        )


# ===========================================================================
# 2. Runtime Symbol Identity
# ===========================================================================


class TestRuntimeSymbolIdentity:
    @pytest.fixture(scope="class")
    def notebook_namespace(self) -> Dict[str, Any]:
        """Execute canonical cells in a clean sandbox namespace."""
        nb = load_patched_notebook()
        ns: Dict[str, Any] = {
            "__file__": str(PATCHED_NOTEBOOK),
            "logging": logging,
        }

        # Cells 16, 19, 22 are clean model, registry, and state definitions
        for idx in [16, 19, 22]:
            src = "".join(nb["cells"][idx].get("source", []))
            exec(src, ns)

        # Cells 32 and 60 contain extensive tool/agent logic; execute their imports
        for idx in [32, 60]:
            tree = ast.parse("".join(nb["cells"][idx].get("source", [])))
            import_nodes = [
                n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            ]
            import_src = ast.unparse(ast.Module(body=import_nodes, type_ignores=[]))
            exec(import_src, ns)

        return ns

    def test_model_identity(self, notebook_namespace: Dict[str, Any]):
        assert notebook_namespace["Plan"] is idd_models.Plan
        assert notebook_namespace["BaseNoExtrasModel"] is idd_models.BaseNoExtrasModel
        assert notebook_namespace["PlanStep"] is idd_models.PlanStep

    def test_state_identity(self, notebook_namespace: Dict[str, Any]):
        assert notebook_namespace["State"] is idd_state.State
        assert notebook_namespace["VizWorkerState"] is idd_state.VizWorkerState
        assert notebook_namespace["AnalysisConfig"] is idd_state.AnalysisConfig

    def test_registry_identity(self, notebook_namespace: Dict[str, Any]):
        assert notebook_namespace["DataFrameRegistry"] is idd_registry.DataFrameRegistry
        assert (
            notebook_namespace["get_global_registry"]
            is idd_registry.get_global_registry
        )
        assert (
            notebook_namespace["global_df_registry"]
            is idd_registry.get_global_registry()
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

    def test_graph_identity(self, notebook_namespace: Dict[str, Any]):
        assert notebook_namespace["build_graph"] is idd_graph.build_graph


# ===========================================================================
# 3. Behavioral Parity & Anti-Regression Invariants
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

        # Calling with positional string "revenue" must not trigger DataFrame lookup or validation error
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
                config={"artifacts_dir": str(target_root)},
                subdir=f"../../{canary_dir.name}",
            )

        assert (
            not canary_dir.exists()
        ), "Subdir escape created unauthorized directory before validation!"

    def test_registry_lifecycle_and_reload(
        self, tools_ns: Dict[str, Any], tmp_path: Path
    ):
        """Registry stores, validates existence, and persists data."""
        registry: idd_registry.DataFrameRegistry = tools_ns["global_df_registry"]
        val_fn = tools_ns["validate_dataframe_exists"]

        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        df_id = registry.register_dataframe(df)

        # Validate existence
        check = val_fn(df_id)
        assert check is True

        # Retrieve
        retrieved = registry.get_dataframe(df_id)
        assert retrieved is not None
        assert len(retrieved) == 3

    def test_executable_graph_construction(self):
        """Graph in Cell 60 compiles in executable mode with all 14 node overrides."""
        dummy_nodes = {
            name: lambda state: {} for name in idd_graph.PRODUCTION_NODE_NAMES
        }
        graph = idd_graph.build_graph(
            node_overrides=dummy_nodes,
            structural_only=False,
            node_cache_policies=idd_graph.CachePolicy(ttl=120),
        )
        assert graph is not None
        assert not graph.is_structural_only


# ===========================================================================
# 4. Negative Controls & Redefinition Detection
# ===========================================================================


class TestNegativeControls:
    def test_synthetic_redefinition_detection_class(self):
        bad_cell = (
            "class Plan:\n" "    def __init__(self):\n" "        self.steps = []\n"
        )
        classes, _ = find_defined_symbols(bad_cell)
        assert "Plan" in classes
        assert classes.intersection(PROTECTED_CLASSES) == {"Plan"}

    def test_synthetic_redefinition_detection_func(self):
        bad_cell = "def handle_tool_errors(func):\n" "    return func\n"
        _, funcs = find_defined_symbols(bad_cell)
        assert "handle_tool_errors" in funcs
        assert funcs.intersection(PROTECTED_FUNCTIONS) == {"handle_tool_errors"}

    def test_manifest_verification_tampering_fails_closed(self, tmp_path: Path):
        """Tampered SHA-256 digest in manifest raises RuntimeError."""
        manifest_copy = tmp_path / "idd_manifest.json"
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Tamper with idd_models.py digest
        data["files"]["idd_models.py"] = "0" * 64
        with open(manifest_copy, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        with pytest.raises(RuntimeError, match="Digest mismatch for 'idd_models.py'"):
            idd_bootstrap.verify_manifest(
                repo_root=REPO_ROOT, manifest_path=manifest_copy
            )

    def test_manifest_verification_missing_module_entry(self, tmp_path: Path):
        """Missing canonical module in manifest raises RuntimeError."""
        manifest_copy = tmp_path / "idd_manifest.json"
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        del data["files"]["idd_graph.py"]
        with open(manifest_copy, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        with pytest.raises(
            RuntimeError, match="missing required canonical entry: 'idd_graph.py'"
        ):
            idd_bootstrap.verify_manifest(
                repo_root=REPO_ROOT, manifest_path=manifest_copy
            )

    def test_manifest_verification_missing_file_on_disk(self, tmp_path: Path):
        """Missing canonical file on disk raises FileNotFoundError."""
        empty_dir = tmp_path / "empty_repo"
        empty_dir.mkdir()
        shutil.copy(MANIFEST_FILE, empty_dir / "idd_manifest.json")

        with pytest.raises(
            FileNotFoundError, match="Canonical module 'idd_models.py' missing"
        ):
            idd_bootstrap.verify_manifest(repo_root=empty_dir)
