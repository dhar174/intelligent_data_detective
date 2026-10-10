#!/usr/bin/env python3
"""
probe_notebook_cells.py — Static AST scanner for IDD Jupyter notebook cells.

Performs genuine static AST inspection of notebook code cells without executing code.
Uses validate_notebook_integrity.sanitize_cell_source to safely handle IPython magics
while preserving line numbers and statement structure.

Identifies:
- Class definitions and base classes
- Function definitions, signatures, and decorators
- Import statements (import and from ... import)
- Top-level assignments and reassignments across cells
- Exact cell indices, cell IDs, and source line numbers

Static Analysis Scope & Limitations:
- Identifies syntactic definitions and top-level statement ordering across sequential cells.
- Cannot determine final runtime bindings when assignments occur inside conditional branches,
  exception handlers, or dynamic expressions.
- Fail-closed: parse failures are explicitly reported as incomplete scans with non-zero exit code.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the repository's authoritative notebook sanitizer
try:
    from validate_notebook_integrity import sanitize_cell_source
except ImportError:
    sys.path.insert(0, str(REPO_ROOT))
    from validate_notebook_integrity import sanitize_cell_source

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class SymbolDef:
    name: str
    kind: str  # "class", "function", "async_function", "import", "assignment"
    cell_idx: int
    cell_id: str
    line_number: int
    details: str = ""
    is_top_level: bool = True
    decorators: List[str] = field(default_factory=list)


class CellASTVisitor(ast.NodeVisitor):
    """AST visitor extracting definitions within a single sanitized notebook cell."""

    def __init__(self, cell_idx: int, cell_id: str):
        self.cell_idx = cell_idx
        self.cell_id = cell_id
        self.symbols: List[SymbolDef] = []
        self._scope_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef):
        bases = [ast.unparse(b) for b in node.bases]
        decs = [ast.unparse(d) for d in node.decorator_list]
        self.symbols.append(
            SymbolDef(
                name=node.name,
                kind="class",
                cell_idx=self.cell_idx,
                cell_id=self.cell_id,
                line_number=node.lineno,
                details=f"bases=({', '.join(bases)})",
                is_top_level=(self._scope_depth == 0),
                decorators=decs,
            )
        )
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef):
        args = [a.arg for a in node.args.args]
        decs = [ast.unparse(d) for d in node.decorator_list]
        self.symbols.append(
            SymbolDef(
                name=node.name,
                kind="function",
                cell_idx=self.cell_idx,
                cell_id=self.cell_id,
                line_number=node.lineno,
                details=f"args=({', '.join(args)})",
                is_top_level=(self._scope_depth == 0),
                decorators=decs,
            )
        )
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        args = [a.arg for a in node.args.args]
        decs = [ast.unparse(d) for d in node.decorator_list]
        self.symbols.append(
            SymbolDef(
                name=node.name,
                kind="async_function",
                cell_idx=self.cell_idx,
                cell_id=self.cell_id,
                line_number=node.lineno,
                details=f"args=({', '.join(args)})",
                is_top_level=(self._scope_depth == 0),
                decorators=decs,
            )
        )
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_Import(self, node: ast.Import):
        if self._scope_depth == 0:
            for alias in node.names:
                as_name = f" as {alias.asname}" if alias.asname else ""
                self.symbols.append(
                    SymbolDef(
                        name=alias.asname or alias.name,
                        kind="import",
                        cell_idx=self.cell_idx,
                        cell_id=self.cell_id,
                        line_number=node.lineno,
                        details=f"import {alias.name}{as_name}",
                        is_top_level=True,
                    )
                )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if self._scope_depth == 0:
            mod = node.module or ""
            for alias in node.names:
                as_name = f" as {alias.asname}" if alias.asname else ""
                self.symbols.append(
                    SymbolDef(
                        name=alias.asname or alias.name,
                        kind="import",
                        cell_idx=self.cell_idx,
                        cell_id=self.cell_id,
                        line_number=node.lineno,
                        details=f"from {mod} import {alias.name}{as_name}",
                        is_top_level=True,
                    )
                )

    def visit_Assign(self, node: ast.Assign):
        if self._scope_depth == 0:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    val_repr = ""
                    try:
                        val_repr = ast.unparse(node.value)
                        if len(val_repr) > 40:
                            val_repr = val_repr[:37] + "..."
                    except Exception:
                        pass
                    self.symbols.append(
                        SymbolDef(
                            name=target.id,
                            kind="assignment",
                            cell_idx=self.cell_idx,
                            cell_id=self.cell_id,
                            line_number=node.lineno,
                            details=f"= {val_repr}",
                            is_top_level=True,
                        )
                    )

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if self._scope_depth == 0 and isinstance(node.target, ast.Name):
            val_repr = ""
            if node.value:
                try:
                    val_repr = ast.unparse(node.value)
                    if len(val_repr) > 40:
                        val_repr = val_repr[:37] + "..."
                except Exception:
                    pass
            self.symbols.append(
                SymbolDef(
                    name=node.target.id,
                    kind="assignment",
                    cell_idx=self.cell_idx,
                    cell_id=self.cell_id,
                    line_number=node.lineno,
                    details=f": {ast.unparse(node.annotation)} = {val_repr}",
                    is_top_level=True,
                )
            )


def scan_notebook(nb_path: Path) -> Tuple[List[SymbolDef], Dict[str, List[SymbolDef]], List[Dict[str, Any]]]:
    """
    Parse all code cells in the notebook via AST.
    Returns:
    - flat list of all SymbolDefs
    - symbol_map mapping symbol name to list of definitions (detecting reassignments)
    - parse_errors list if any cell fails AST parsing
    """
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    all_symbols: List[SymbolDef] = []
    symbol_map: Dict[str, List[SymbolDef]] = {}
    parse_errors: List[Dict[str, Any]] = []

    for idx, cell in enumerate(nb["cells"]):
        cell_type = cell.get("cell_type", "")
        if cell_type != "code":
            continue

        cell_id = str(cell.get("id", f"cell_{idx}"))
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)

        if not source.strip():
            continue

        # Sanitize IPython cell magics/escapes
        sanitize_res = sanitize_cell_source(source)
        if sanitize_res.excluded_reason:
            continue
        if sanitize_res.unsupported_error:
            parse_errors.append({
                "cell_idx": idx,
                "cell_id": cell_id,
                "error": sanitize_res.unsupported_error,
            })
            continue

        code_to_parse = sanitize_res.code
        try:
            tree = ast.parse(code_to_parse, filename=f"Cell_{idx}_{cell_id}")
            visitor = CellASTVisitor(cell_idx=idx, cell_id=cell_id)
            visitor.visit(tree)

            for sym in visitor.symbols:
                all_symbols.append(sym)
                symbol_map.setdefault(sym.name, []).append(sym)
        except SyntaxError as exc:
            parse_errors.append({
                "cell_idx": idx,
                "cell_id": cell_id,
                "error": f"SyntaxError at line {exc.lineno}: {exc.msg}",
            })

    return all_symbols, symbol_map, parse_errors


def main():
    parser = argparse.ArgumentParser(description="Static AST scanner for IDD notebook cells.")
    parser.add_argument(
        "--notebook",
        default="IntelligentDataDetective_beta_v5_patched.ipynb",
        help="Path to the notebook to inspect",
    )
    parser.add_argument("--symbol", help="Filter for a specific symbol name")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--redefinitions", action="store_true", help="Report only re-defined symbols")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    if not nb_path.exists():
        print(f"Error: Notebook {nb_path} does not exist", file=sys.stderr)
        sys.exit(1)

    all_symbols, symbol_map, parse_errors = scan_notebook(nb_path)

    if args.json:
        out = {
            "notebook": str(nb_path),
            "status": "incomplete" if parse_errors else "complete",
            "is_complete": len(parse_errors) == 0,
            "total_symbols": len(all_symbols),
            "parse_errors": parse_errors,
            "static_scope_note": "AST static analysis inspects syntactic definitions without execution; cannot resolve conditional runtime branches or dynamic bindings.",
            "symbols": [asdict(s) for s in all_symbols],
            "redefinitions": {
                name: [asdict(s) for s in defs]
                for name, defs in symbol_map.items()
                if len(defs) > 1 and any(d.kind in ("class", "function") for d in defs)
            },
        }
        if args.symbol:
            out["filtered_symbol"] = args.symbol
            out["filtered_matches"] = [asdict(s) for s in symbol_map.get(args.symbol, [])]
        print(json.dumps(out, indent=2))
        sys.exit(1 if parse_errors else 0)

    print(f"AST Scan Report for {nb_path.name}")
    print(f"Status: {'INCOMPLETE (Parse errors encountered)' if parse_errors else 'COMPLETE'}")
    print(f"Total Symbols Extracted: {len(all_symbols)}")
    print("Scope Note: AST static analysis inspects syntactic definitions without execution; cannot determine conditional runtime bindings.")
    if parse_errors:
        print(f"Parse Errors ({len(parse_errors)}):")
        for err in parse_errors:
            print(f"  Cell {err['cell_idx']} ({err['cell_id']}): {err['error']}")
        print("-" * 60)
        print("Error: AST scan incomplete due to parse errors.", file=sys.stderr)
        sys.exit(1)
    print("-" * 60)

    if args.symbol:
        matches = symbol_map.get(args.symbol, [])
        print(f"Matches for '{args.symbol}' ({len(matches)}):")
        for m in matches:
            top_str = "top-level" if m.is_top_level else "nested"
            print(f"  Cell {m.cell_idx:02d} (id={m.cell_id}) L{m.line_number:04d}: [{m.kind}] {m.name} ({m.details}) [{top_str}]")
        return

    # Key production targets to display in summary report
    key_targets = [
        "BaseNoExtrasModel",
        "State",
        "Plan",
        "PlanStep",
        "CompletedStepsAndTasks",
        "CleaningMetadata",
        "AnalysisInsights",
        "VizSpec",
        "Section",
        "SectionOutline",
        "ReportOutline",
        "ReportResults",
        "ListOfFiles",
        "DataFrameRegistry",
        "global_df_registry",
        "validate_dataframe_exists",
        "handle_tool_errors",
        "_tool_error",
        "_tool_failure",
        "_reduce_plan_keep_sorted",
        "keep_first",
        "_resolve_artifact_path",
        "_build_query_view",
        "delete_rows",
        "build_graph",
        "data_analysis_team_builder",
        "supervisor_node",
        "initial_analysis_node",
        "data_cleaner_node",
        "analyst_node",
        "viz_worker_node",
        "viz_join_node",
        "viz_evaluator_node",
        "report_orchestrator_node",
        "report_section_worker_node",
        "report_join_node",
        "report_packager_node",
        "file_writer_node",
        "emergency_correspondence_node",
    ]

    print("Canonical & Subsystem Symbol Locators (AST Verified):")
    for t in key_targets:
        defs = symbol_map.get(t, [])
        if defs:
            for d in defs:
                redef_flag = f" [REDEFINED x{len(defs)}]" if len(defs) > 1 else ""
                print(f"  {d.name:<28} Cell {d.cell_idx:02d} (id={d.cell_id}) L{d.line_number:04d} [{d.kind}] {d.details}{redef_flag}")
        else:
            print(f"  {t:<28} NOT FOUND in code cells")

    # Redefinitions
    multi_defs = {
        name: defs for name, defs in symbol_map.items()
        if len(defs) > 1 and any(d.kind in ("class", "function") for d in defs)
    }
    if multi_defs:
        print("\nDetected Function/Class Redefinitions across Cells:")
        for name, defs in sorted(multi_defs.items()):
            cells_str = ", ".join(f"Cell {d.cell_idx} L{d.line_number}" for d in defs)
            print(f"  {name}: {len(defs)} occurrences ({cells_str})")


if __name__ == "__main__":
    main()
