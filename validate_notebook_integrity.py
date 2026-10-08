#!/usr/bin/env python3
"""
validate_notebook_integrity.py — Fail-closed structural and syntax validation
gate for Jupyter notebooks in the Intelligent Data Detective (IDD) repository.

Enforces:
1. File existence and valid JSON syntax.
2. Presence and list type of the top-level 'cells' field.
3. Exact cell count match against expected count (default: 99 for W14 completion baseline).
4. Real Python code-object compilation of every code cell with allow-top-level-await semantics.
5. Safe transformation of supported IPython notebook magics/shell escapes
   while preserving statement structure, indentation, and line numbers for actionable diagnostics.
6. Validation of Python-bearing magic payloads (%%timeit setup code and bodies, %time, %timeit, %prun).
7. Safe exclusion and explicit diagnostic reporting of recognized non-Python cell magics (%%bash, %%html, etc.).
8. Rejection of unknown or unsupported cell magics with nonzero diagnostics.

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
from typing import NamedTuple, Sequence

DEFAULT_NOTEBOOK = "IntelligentDataDetective_beta_v5_patched.ipynb"
DEFAULT_EXPECTED_CELLS = 99
SUPPORTED_CELL_TYPES = {"code", "markdown", "raw"}

SUPPORTED_PYTHON_CELL_MAGICS = {"time", "timeit", "capture", "prun", "python"}
RECOGNIZED_NON_PYTHON_CELL_MAGICS = {
    "bash",
    "sh",
    "html",
    "javascript",
    "js",
    "latex",
    "writefile",
    "svg",
    "cmd",
    "ruby",
    "perl",
}
PYTHON_LINE_MAGICS = {"time", "timeit", "prun"}

IPYTHON_HELP_PATTERN = re.compile(
    r"^(\?{1,2}\s*[a-zA-Z_][a-zA-Z0-9_\.]*|[a-zA-Z_][a-zA-Z0-9_\.]*\s*\?{1,2}|\?{1,2})$"
)
LINE_MAGIC_RE = re.compile(r"^%([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(.*))?$")
SHELL_ASSIGN_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_,\s\(\)\[\]\.]*)\s*=\s*!(.*)$")
MAGIC_ASSIGN_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_,\s\(\)\[\]\.]*)\s*=\s*(%.*)$")


class SanitizeResult(NamedTuple):
    code: str
    excluded_reason: str | None = None
    unsupported_error: str | None = None


def _parse_and_strip_timeit_options(rest: str) -> tuple[str, str | None]:
    """
    Parse and validate %timeit / %%timeit options and return (remaining_code, error_diagnostic).

    Supported IPython timeit options:
    - '-n <N>' / '-n<N>': positive integer loop count
    - '-r <R>' / '-r<R>': positive integer repeat count
    - '-p <P>' / '-p<P>': non-negative integer precision digits
    - '-t', '-c', '-o', '-q', '--quiet': boolean flags (or combinations of flags like '-qo')

    Returns:
        (remaining_code, None) if options are valid.
        ("", error_message) if an option is invalid, unrecognised, or has a bad argument.
    """
    tokens = rest.split()
    idx = 0
    while idx < len(tokens):
        t = tokens[idx]
        if t == "--quiet":
            idx += 1
        elif t.startswith("--"):
            return "", f"unrecognized option '{t}'"
        elif t in ("-n", "-r", "-p"):
            if idx + 1 >= len(tokens):
                return "", f"option {t} requires an argument"
            val = tokens[idx + 1]
            try:
                val_int = int(val)
            except ValueError:
                return "", f"invalid integer for {t}: '{val}'"
            if t in ("-n", "-r") and val_int <= 0:
                return "", f"option {t} value must be a positive integer, got {val_int}"
            if t == "-p" and val_int < 0:
                return (
                    "",
                    f"option -p value must be a non-negative integer, got {val_int}",
                )
            idx += 2
        elif t.startswith(("-n", "-r", "-p")) and len(t) > 2:
            opt = t[:2]
            val = t[2:]
            try:
                val_int = int(val)
            except ValueError:
                return "", f"invalid integer for {opt}: '{val}'"
            if opt in ("-n", "-r") and val_int <= 0:
                return (
                    "",
                    f"option {opt} value must be a positive integer, got {val_int}",
                )
            if opt == "-p" and val_int < 0:
                return (
                    "",
                    f"option -p value must be a non-negative integer, got {val_int}",
                )
            idx += 1
        elif t.startswith("-") and len(t) > 1 and t != "-":
            flag_chars = t[1:]
            if all(ch in ("t", "c", "o", "q") for ch in flag_chars):
                idx += 1
            else:
                return "", f"unrecognized option '{t}'"
        else:
            break

    pos = 0
    for tok in tokens[:idx]:
        tok_idx = rest.find(tok, pos)
        if tok_idx != -1:
            pos = tok_idx + len(tok)
    remaining_code = rest[pos:].lstrip()
    return remaining_code, None


def _strip_timeit_options(rest: str) -> str:
    """Strip known options from %timeit / %%timeit (backward compatibility wrapper)."""
    code, _ = _parse_and_strip_timeit_options(rest)
    return code


def _scan_line(
    line: str, in_multiline: str | None, curr_depth: int
) -> tuple[str | None, int]:
    """
    Scan a line updating both multiline string state and unclosed delimiter depth.
    Characters inside strings (single or multiline) and comments do not affect delimiter depth.
    When a multiline string ends, trailing content on the same line is scanned for delimiters.
    """
    idx = 0
    depth = curr_depth
    n = len(line)

    while idx < n:
        if in_multiline is not None:
            close_idx = line.find(in_multiline, idx)
            if close_idx == -1:
                return in_multiline, depth
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
            ch = line[idx]
            if ch == "#":
                break
            elif line[idx : idx + 3] in ('"""', "'''"):
                in_multiline = line[idx : idx + 3]
                idx += 3
            elif ch in ('"', "'"):
                quote = ch
                idx += 1
                while idx < n:
                    if line[idx] == "\\":
                        idx += 2
                    elif line[idx] == quote:
                        idx += 1
                        break
                    else:
                        idx += 1
            elif ch in "([{":
                depth += 1
                idx += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
                idx += 1
            else:
                idx += 1

    return in_multiline, depth


def _update_multiline_string_state(line: str, in_multiline: str | None) -> str | None:
    """Track whether scanning enters or exits triple-quoted strings (''' or \"\"\")."""
    return _scan_line(line, in_multiline, 0)[0]


def _count_open_parens(line: str, curr_depth: int) -> int:
    """Count unclosed parentheses/brackets/braces outside comments and string literals."""
    return _scan_line(line, None, curr_depth)[1]


def sanitize_cell_source(source: str) -> SanitizeResult:
    """
    Transform IPython/notebook-specific syntax into valid Python
    while preserving line numbers, statement structure, and indentation
    for accurate compiler diagnostics.

    Cell magic rules (%%...):
    - A cell magic is only syntactically valid on the first non-blank line of a cell.
    - If a cell begins with a recognized non-Python cell magic (e.g., %%bash, %%sh, %%html),
      it is safely excluded from Python compilation and reported in diagnostics.
    - If a cell begins with a supported Python-body cell magic (e.g., %%time, %%timeit, %%python),
      its body is compiled. For %%timeit, any setup code in the header is compiled.
    - If a cell begins with an unknown or unsupported %% directive, an error diagnostic is generated.
    - Mid-cell '%%' directives are left intact so that Python flags the invalid syntax.

    Line magic rules (%... / !... / ?...):
    - Standalone '!' and non-Python '%' line magics are replaced with 'pass' at the same
      indentation to preserve block structure.
    - Line magics evaluating Python expressions (%time, %timeit, %prun) compile their payload.
    - Shell assignments (var = !cmd) transform to var = [].
    - Magic assignments (var = %magic) transform to var = None or var = (expr).
    - Multi-line expressions with continued '%' (e.g. modulo in parens) are preserved intact.
    """
    lines = source.splitlines(keepends=True)
    if not lines:
        return SanitizeResult("", None, None)

    # In IPython, cell magics (%%...) are only valid on the first non-blank line of a cell
    first_non_blank_idx: int | None = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_non_blank_idx = idx
            break

    magic_token = ""
    if first_non_blank_idx is not None:
        first_line_stripped = lines[first_non_blank_idx].strip()
        if first_line_stripped.startswith("%%"):
            magic_parts = first_line_stripped[2:].split()
            magic_token = magic_parts[0] if magic_parts else ""
            if not magic_token:
                # Bare %% without a magic name is invalid in IPython
                pass
            elif magic_token in RECOGNIZED_NON_PYTHON_CELL_MAGICS:
                clean_lines = []
                for line in lines:
                    content = line.rstrip("\r\n")
                    clean_lines.append(f"# [cell-magic {magic_token}]: {content}\n")
                return SanitizeResult(
                    "".join(clean_lines),
                    f"recognized non-Python cell magic '%%{magic_token}'",
                    None,
                )
            elif magic_token in SUPPORTED_PYTHON_CELL_MAGICS:
                pass
            else:
                return SanitizeResult(
                    source,
                    None,
                    f"unsupported cell magic '%%{magic_token}'. "
                    f"Supported Python magics: {sorted(SUPPORTED_PYTHON_CELL_MAGICS)}; "
                    f"recognized non-Python magics: {sorted(RECOGNIZED_NON_PYTHON_CELL_MAGICS)}.",
                )

    in_multiline: str | None = None
    paren_depth = 0
    clean_lines = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        leading_ws_len = len(line) - len(line.lstrip())
        indent = line[:leading_ws_len]
        was_in_multiline = in_multiline
        curr_paren_depth = paren_depth

        # If this line started inside a multiline string literal, it cannot be a magic
        if was_in_multiline is not None:
            in_multiline, paren_depth = _scan_line(line, in_multiline, paren_depth)
            clean_lines.append(line)
            continue

        # Leading cell magic on first non-blank line
        if idx == first_non_blank_idx and stripped.startswith("%%"):
            if magic_token == "timeit":
                header_rest = stripped[len("%%timeit") :].strip()
                setup_code, opt_err = _parse_and_strip_timeit_options(header_rest)
                if opt_err:
                    return SanitizeResult(
                        source,
                        None,
                        f"invalid %%timeit options '{header_rest}': {opt_err}",
                    )
                setup_code = setup_code.strip()
                if setup_code:
                    clean_lines.append(f"{indent}{setup_code}\n")
                    in_multiline, paren_depth = _scan_line(
                        setup_code, in_multiline, paren_depth
                    )
                else:
                    clean_lines.append(f"{indent}pass  # [cell-magic %%timeit]\n")
            elif magic_token in SUPPORTED_PYTHON_CELL_MAGICS:
                clean_lines.append(f"{indent}pass  # [cell-magic %%{magic_token}]\n")
            else:
                # Bare %% or mid-cell %% remains intact and will trigger SyntaxError
                clean_lines.append(line)
            continue

        # Check if line is within open parentheses (e.g. multiline expressions like modulo arithmetic)
        if curr_paren_depth > 0:
            in_multiline, paren_depth = _scan_line(line, in_multiline, paren_depth)
            clean_lines.append(line)
            continue

        # Check for shell assignment: target = !cmd
        shell_m = SHELL_ASSIGN_RE.match(stripped)
        if shell_m:
            target, cmd = shell_m.groups()
            clean_lines.append(f"{indent}{target} = []  # [IPython shell]: !{cmd}\n")
            continue

        # Check for magic assignment: target = %magic
        magic_assign_m = MAGIC_ASSIGN_RE.match(stripped)
        if magic_assign_m:
            target, magic_call = magic_assign_m.groups()
            lm_m = LINE_MAGIC_RE.match(magic_call.strip())
            if lm_m:
                lm_token, lm_rest = lm_m.groups()
                lm_rest = (lm_rest or "").strip()
                if lm_token in PYTHON_LINE_MAGICS:
                    if lm_token == "timeit":
                        py_code, opt_err = _parse_and_strip_timeit_options(lm_rest)
                        if opt_err:
                            return SanitizeResult(
                                source,
                                None,
                                f"invalid %timeit options '{lm_rest}': {opt_err}",
                            )
                        py_code = py_code.strip()
                    else:
                        py_code = lm_rest
                    if py_code:
                        clean_lines.append(
                            f"{indent}{target} = ({py_code})  # [IPython %{lm_token}]\n"
                        )
                    else:
                        clean_lines.append(
                            f"{indent}{target} = None  # [IPython %{lm_token}]\n"
                        )
                else:
                    clean_lines.append(
                        f"{indent}{target} = None  # [IPython magic]: {magic_call}\n"
                    )
            else:
                clean_lines.append(f"{indent}{target} = None\n")
            continue

        # Standalone shell escape: !cmd
        if stripped.startswith("!"):
            clean_lines.append(f"{indent}pass  # [IPython shell]: {stripped}\n")
            continue

        # Standalone line magic: %magic ...
        lm_m = LINE_MAGIC_RE.match(stripped)
        if lm_m:
            lm_token, lm_rest = lm_m.groups()
            lm_rest = (lm_rest or "").strip()
            if lm_token in PYTHON_LINE_MAGICS:
                if lm_token == "timeit":
                    py_code, opt_err = _parse_and_strip_timeit_options(lm_rest)
                    if opt_err:
                        return SanitizeResult(
                            source,
                            None,
                            f"invalid %timeit options '{lm_rest}': {opt_err}",
                        )
                    py_code = py_code.strip()
                else:
                    py_code = lm_rest
                if py_code:
                    clean_lines.append(f"{indent}{py_code}  # [IPython %{lm_token}]\n")
                else:
                    clean_lines.append(f"{indent}pass  # [IPython %{lm_token}]\n")
            else:
                clean_lines.append(f"{indent}pass  # [IPython magic]: {stripped}\n")
            continue

        # Standalone help query: ?obj, obj?, etc.
        if IPYTHON_HELP_PATTERN.match(stripped):
            clean_lines.append(f"{indent}pass  # [IPython help]: {stripped}\n")
            continue

        # Ordinary Python line
        in_multiline, paren_depth = _scan_line(line, in_multiline, paren_depth)
        clean_lines.append(line)

    return SanitizeResult("".join(clean_lines), None, None)


def validate_notebook(
    notebook_path: str | Path,
    expected_cells: int = DEFAULT_EXPECTED_CELLS,
    verbose: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validate notebook structure and compile all code cells into Python code objects.

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
    code_cells_compiled = 0
    excluded_cells: list[tuple[int, str, str]] = []
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

        # Non-code cells are structurally validated above; only code cells require compilation
        if cell_type != "code":
            continue

        code_cells_checked += 1
        sanitized = sanitize_cell_source(source_text)

        if sanitized.unsupported_error:
            syntax_errors.append(
                f"Cell {idx} (id: {cell_id}) {sanitized.unsupported_error}"
            )
            continue

        if sanitized.excluded_reason:
            excluded_cells.append((idx, cell_id, sanitized.excluded_reason))
            diagnostics.append(
                f"Cell {idx} (id: {cell_id}) excluded from Python compilation: {sanitized.excluded_reason}"
            )
            continue

        code_cells_compiled += 1
        try:
            compile(
                sanitized.code,
                filename=f"{path.name}:cell_{idx}",
                mode="exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
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
        excluded_info = ""
        if excluded_cells:
            plural = "s" if len(excluded_cells) > 1 else ""
            excluded_info = f", {len(excluded_cells)} code cell{plural} excluded from Python compilation"
        summary_msg = (
            f"Successfully validated {actual_cell_count} cells "
            f"({code_cells_compiled} code cells compiled cleanly{excluded_info})."
        )
        diagnostics.append(summary_msg)

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
