#!/usr/bin/env python3
"""
validate_notebook_integrity.py — Fail-closed structural and syntax validation
gate for Jupyter notebooks in the Intelligent Data Detective (IDD) repository.

Enforces:
1. File existence and valid JSON syntax.
2. Presence and list type of the top-level 'cells' field.
3. Exact cell count match against expected count (default: 99 for W14 completion baseline).
4. Real Python AST/syntax compilation of every code cell.
5. Safe transformation of supported IPython notebook magics/shell escapes
   while preserving line counts and line numbers for actionable diagnostics.

Exits with:
  0: Notebook passes complete integrity gate.
  1: Notebook fails integrity gate (structural mismatch, syntax error, or missing file).
  2: Command-line invocation error.

Zero external dependencies (pure Python standard library).
Zero execution of notebook code.
Zero network or API key requirements.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Sequence

DEFAULT_NOTEBOOK = "IntelligentDataDetective_beta_v5_patched.ipynb"
DEFAULT_EXPECTED_CELLS = 99
SUPPORTED_CELL_TYPES = {"code", "markdown", "raw"}
PYTHON_BODY_CELL_MAGICS = {"time", "timeit", "capture", "prun"}
IPYTHON_HELP_PATTERN = re.compile(
    r"^(\?{1,2}\s*[a-zA-Z_][a-zA-Z0-9_\.]*|[a-zA-Z_][a-zA-Z0-9_\.]*\s*\?{1,2}|\?{1,2})$"
)


def _update_multiline_string_state(line: str, in_multiline: str | None) -> str | None:
    """Track whether scanning enters or exits triple-quoted strings (''' or \"\"\")."""
    idx = 0
    while idx < len(line):
        if in_multiline is not None:
            close_idx = line.find(in_multiline, idx)
            if close_idx == -1:
                return in_multiline
            num_backslashes = 0
            check_pos = close_idx - 1
            while check_pos >= 0 and line[check_pos] == "\\":
                num_backslashes += 1
                check_pos -= 1
            if num_backslashes % 2 == 0:
                in_multiline = None
                idx = close_idx + 3
            else:
                idx = close_idx + 1
        else:
            if line[idx] == "#":
                break
            if line[idx : idx + 3] in ('"""', "'''"):
                delim = line[idx : idx + 3]
                in_multiline = delim
                idx += 3
            elif line[idx] in ('"', "'"):
                quote = line[idx]
                idx += 1
                while idx < len(line):
                    if line[idx] == "\\":
                        idx += 2
                    elif line[idx] == quote:
                        idx += 1
                        break
                    else:
                        idx += 1
            else:
                idx += 1
    return in_multiline


def sanitize_cell_source(source: str) -> str:
    """
    Transform IPython/notebook-specific syntax into valid Python
    while preserving line numbers and structure for accurate compiler diagnostics.

    Cell magic rules (%%...):
    - A cell magic is only syntactically valid on the first non-blank, non-comment line of a cell.
    - If a cell begins with a non-Python cell magic (e.g., %%bash, %%sh, %%html, %%javascript,
      %%latex, %%writefile, %%svg, %%cmd), the entire cell is treated as a whole-cell construct
      and commented out, preserving line counts so non-Python code is not compiled.
    - If a cell begins with a Python-body cell magic (e.g., %%time, %%timeit, %%capture, %%prun),
      only the leading directive line is commented out, allowing the Python body to compile.
    - Mid-cell '%%' directives are invalid in IPython; they are left intact so that Python's
      AST compiler flags the invalid syntax.

    Line magic rules (%... / !... / ?...):
    - Single '%' line magics (e.g., %matplotlib inline) and '!' shell escapes are commented
      out line-by-line.
    - Dynamic help queries (?obj or obj?) are commented out line-by-line.
    """
    lines = source.splitlines(keepends=True)
    if not lines:
        return ""

    first_non_blank_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_non_blank_idx = idx
            break

    magic_token = ""
    # Check for leading whole-cell magic (%%...)
    if first_non_blank_idx is not None:
        first_line_stripped = lines[first_non_blank_idx].strip()
        if first_line_stripped.startswith("%%"):
            magic_parts = first_line_stripped[2:].split()
            magic_token = magic_parts[0] if magic_parts else ""
            # If the magic body is not executed by Python, comment out the whole cell
            if magic_token and magic_token not in PYTHON_BODY_CELL_MAGICS:
                clean_lines: list[str] = []
                for line in lines:
                    content = line.rstrip("\r\n")
                    clean_lines.append(f"# [cell-magic {magic_token}]: {content}\n")
                return "".join(clean_lines)

    in_multiline: str | None = None
    clean_lines = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        was_in_multiline = in_multiline
        in_multiline = _update_multiline_string_state(line, in_multiline)

        # Lines inside multiline string literals must never be rewritten as magics
        if was_in_multiline is not None:
            clean_lines.append(line)
            continue

        # Leading Python-body cell magic (e.g. %%time)
        if (
            idx == first_non_blank_idx
            and stripped.startswith("%%")
            and magic_token in PYTHON_BODY_CELL_MAGICS
        ):
            leading_whitespace_len = len(line) - len(line.lstrip())
            indent = line[:leading_whitespace_len]
            content = line[leading_whitespace_len:].rstrip("\r\n")
            clean_lines.append(f"{indent}# [cell-magic]: {content}\n")
        # Line magics (single %, not %%) and shell escapes (!)
        elif (
            stripped.startswith("%") and not stripped.startswith("%%")
        ) or stripped.startswith("!"):
            leading_whitespace_len = len(line) - len(line.lstrip())
            indent = line[:leading_whitespace_len]
            content = line[leading_whitespace_len:].rstrip("\r\n")
            clean_lines.append(f"{indent}# [IPython magic/shell]: {content}\n")
        # Standalone IPython dynamic object inspection (?obj, obj?, ??obj, obj??, ?)
        elif IPYTHON_HELP_PATTERN.match(stripped):
            leading_whitespace_len = len(line) - len(line.lstrip())
            indent = line[:leading_whitespace_len]
            content = line[leading_whitespace_len:].rstrip("\r\n")
            clean_lines.append(f"{indent}# [IPython help]: {content}\n")
        else:
            # Ordinary Python line (mid-cell %% or bare %% remains intact and will trigger SyntaxError)
            clean_lines.append(line)

    return "".join(clean_lines)


def validate_notebook(
    notebook_path: str | Path,
    expected_cells: int = DEFAULT_EXPECTED_CELLS,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validate notebook structure and compile all code cells.

    Returns:
        (is_valid, list_of_error_and_diagnostic_messages)
    """
    path = Path(notebook_path)
    diagnostics: list[str] = []

    if not path.is_file():
        diagnostics.append(f"Notebook file not found: {path}")
        return False, diagnostics

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        diagnostics.append(f"Failed to read notebook file {path}: {exc}")
        return False, diagnostics

    try:
        nb_json = json.loads(content)
    except json.JSONDecodeError as exc:
        diagnostics.append(
            f"Malformed notebook JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        )
        return False, diagnostics

    if not isinstance(nb_json, dict):
        diagnostics.append(
            f"Invalid notebook format in {path}: expected JSON object at root, got {type(nb_json).__name__}"
        )
        return False, diagnostics

    if "cells" not in nb_json:
        diagnostics.append(f"Missing required 'cells' key in notebook {path}")
        return False, diagnostics

    cells = nb_json["cells"]
    if not isinstance(cells, list):
        diagnostics.append(
            f"Invalid 'cells' field in {path}: expected list, got {type(cells).__name__}"
        )
        return False, diagnostics

    actual_cell_count = len(cells)
    if actual_cell_count != expected_cells:
        diagnostics.append(
            f"Cell count mismatch in {path}: expected exactly {expected_cells} cells, found {actual_cell_count}"
        )
        return False, diagnostics

    code_cells_checked = 0
    syntax_errors: list[str] = []

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            syntax_errors.append(
                f"Cell {idx} is malformed: expected dict, got {type(cell).__name__}"
            )
            continue

        cell_id = cell.get("id", f"idx_{idx}")
        cell_type = cell.get("cell_type")

        # Reject missing or unknown cell_type
        if cell_type not in SUPPORTED_CELL_TYPES:
            syntax_errors.append(
                f"Cell {idx} (id: {cell_id}) has invalid or missing 'cell_type': {cell_type!r}. "
                f"Expected one of: 'code', 'markdown', 'raw'."
            )
            continue

        # Validate source field structure
        source_field = cell.get("source")
        if source_field is None:
            syntax_errors.append(
                f"Cell {idx} (id: {cell_id}) is missing required 'source' field."
            )
            continue

        if isinstance(source_field, list):
            if not all(isinstance(line, str) for line in source_field):
                syntax_errors.append(
                    f"Cell {idx} (id: {cell_id}) 'source' list contains non-string elements."
                )
                continue
            source_text = "".join(source_field)
        elif isinstance(source_field, str):
            source_text = source_field
        else:
            syntax_errors.append(
                f"Cell {idx} (id: {cell_id}) has invalid 'source' type: {type(source_field).__name__}. "
                f"Expected str or list of str."
            )
            continue

        # Non-code cells are structurally validated above; only code cells require AST compilation
        if cell_type != "code":
            continue

        code_cells_checked += 1
        sanitized = sanitize_cell_source(source_text)

        try:
            compile(
                sanitized,
                filename=f"{path.name}:cell_{idx}",
                mode="exec",
                flags=ast.PyCF_ONLY_AST,
            )
        except SyntaxError as err:
            err_line_text = (err.text or "").strip()
            msg = (
                f"Cell {idx} (id: {cell_id}) syntax compilation failed: "
                f"SyntaxError: {err.msg} at line {err.lineno}, column {err.offset}\n"
                f"    Source: {err_line_text}"
            )
            syntax_errors.append(msg)
        except Exception as err:
            syntax_errors.append(
                f"Cell {idx} (id: {cell_id}) unexpected compilation failure: "
                f"{type(err).__name__}: {err}"
            )

    if syntax_errors:
        diagnostics.extend(syntax_errors)
        return False, diagnostics

    if verbose:
        diagnostics.append(
            f"Successfully validated {actual_cell_count} cells ({code_cells_checked} code cells compiled cleanly)."
        )

    return True, diagnostics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Jupyter notebook structure and Python code cell syntax.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        default=DEFAULT_NOTEBOOK,
        help=f"Path to notebook file (default: {DEFAULT_NOTEBOOK})",
    )
    parser.add_argument(
        "--expected-cells",
        "-n",
        type=int,
        default=DEFAULT_EXPECTED_CELLS,
        help=f"Expected total cell count (default: {DEFAULT_EXPECTED_CELLS})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display verbose validation details upon success.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress success messages (only print errors).",
    )

    args = parser.parse_args(argv)

    target_path = Path(args.notebook)
    is_valid, messages = validate_notebook(
        notebook_path=target_path,
        expected_cells=args.expected_cells,
        verbose=args.verbose,
    )

    if not is_valid:
        print(
            f"FAILED: Notebook integrity check failed for '{target_path}':",
            file=sys.stderr,
        )
        for msg in messages:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"PASSED: Notebook '{target_path}' verified ({args.expected_cells} cells, "
            f"all code cells compiled cleanly)."
        )
        if args.verbose:
            for msg in messages:
                print(f"  {msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
