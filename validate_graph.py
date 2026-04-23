#!/usr/bin/env python3
"""
validate_graph.py — fast compile-only topology sanity check for the IDD v5
patched notebook.

Goal: catch graph-compile errors, managed-channel issues, reducer omissions,
unreachable / dead-end nodes in <5-10 seconds BEFORE committing to a full
10-25 minute notebook run. This would have caught BR-7 (managed channels)
and BR-8 (missing reducer) in seconds.

READ-ONLY: this script never modifies the notebook or any source file.
Network calls are stubbed; side-effecting cells (pip installs, shell magic,
Colab mounts) are skipped.

Usage:
    python validate_graph.py
    python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
    python validate_graph.py --inject-bug missing-node        # demo failure path
    python validate_graph.py --inject-bug missing-reducer     # demo failure path
    python validate_graph.py --verbose
"""

from __future__ import annotations

import argparse
import builtins
import os
import re
import sys
import time
import traceback
import types
from pathlib import Path

DEFAULT_NOTEBOOK = "IntelligentDataDetective_beta_v5_patched.ipynb"
COMPILE_SENTINEL = "data_analysis_team_builder.compile("
COMPILED_VAR = "data_detective_graph"
BUILDER_VAR = "data_analysis_team_builder"

RUNTIME_SOFT_BUDGET_S = 10.0
RUNTIME_HARD_BUDGET_S = 30.0


# --------------------------------------------------------------------------- #
# Cell extraction / filtering
# --------------------------------------------------------------------------- #

def load_notebook(path: Path):
    import nbformat  # local import so --help works without deps
    return nbformat.read(str(path), as_version=4)


def _strip_shell_and_magics(src: str) -> str:
    """Replace lines whose first non-whitespace char is '!' or '%' (IPython
    shell / line magics) with a `pass` statement at the original indentation.
    We use `pass` (not a blank line) so that stripping the body of an
    ``if:``/``for:``/``else:`` block still yields valid Python."""
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}pass  # stripped magic/shell")
        else:
            out.append(line)
    return "\n".join(out)


def _cell_is_pure_shell(src: str) -> bool:
    """True if every non-blank, non-comment line is a magic / shell line."""
    for line in src.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not (s.startswith("!") or s.startswith("%")):
            return False
    return True


def extract_cells_through_compile(nb, verbose: bool = False):
    """Return (list_of_(idx, cleaned_source), compile_cell_idx)."""
    cells = []
    compile_idx = None
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        src = c.source or ""
        if COMPILE_SENTINEL in src:
            compile_idx = i
        # stop AFTER the compile cell
        cells.append((i, src))
        if compile_idx is not None and i == compile_idx:
            break

    if compile_idx is None:
        raise RuntimeError(
            f"Could not find compile cell containing {COMPILE_SENTINEL!r}"
        )

    filtered = []
    for i, src in cells:
        if _cell_is_pure_shell(src):
            if verbose:
                print(f"[skip] cell {i}: pure shell/magic cell")
            continue
        cleaned = _strip_shell_and_magics(src)
        filtered.append((i, cleaned))
    return filtered, compile_idx


# --------------------------------------------------------------------------- #
# Bug injection (for self-test demonstrations)
# --------------------------------------------------------------------------- #

def _inject_bug(cells, kind: str, verbose: bool = False):
    """Return a new cells list with a synthetic defect for demonstration."""
    new = []
    for i, src in cells:
        if kind == "missing-node" and COMPILE_SENTINEL in src:
            # Inject an edge to a node that was never added. LangGraph's
            # .compile() raises ValueError on missing node targets.
            bad = (
                '\n# [validate_graph.py injected bug: missing-node]\n'
                f'{BUILDER_VAR}.add_edge("supervisor", "NODE_THAT_DOES_NOT_EXIST")\n'
            )
            src = src.replace(
                f"{COMPILED_VAR} = {BUILDER_VAR}.compile(",
                bad + f"{COMPILED_VAR} = {BUILDER_VAR}.compile(",
                1,
            )
            if verbose:
                print(f"[inject] cell {i}: added edge to nonexistent node")
        elif kind == "missing-reducer" and "class State(" in src:
            # Swap a reducer-annotated list field for a plain list to simulate
            # BR-8 (missing reducer → InvalidUpdateError on concurrent writes).
            # This won't fail compile but we flag it in post-compile checks.
            src = re.sub(
                r"Annotated\[\s*List\[[^\]]+\]\s*,\s*operator\.add\s*\]",
                "list",
                src,
                count=1,
            )
            if verbose:
                print(f"[inject] cell {i}: stripped reducer from a List field")
        new.append((i, src))
    return new


# --------------------------------------------------------------------------- #
# Execution sandbox
# --------------------------------------------------------------------------- #

PRELUDE = r"""
# -------------------------------------------------------------------------
# validate_graph.py prelude — stub network / Colab so cells are import-safe.
# -------------------------------------------------------------------------
import os as _os, sys as _sys, types as _types
_os.environ.setdefault("OPENAI_API_KEY", "sk-validate-dummy")
_os.environ.setdefault("TAVILY_API_KEY", "tvly-validate-dummy")
_os.environ.setdefault("LANGSMITH_API_KEY", "ls-validate-dummy")
_os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
_os.environ.setdefault("IDD_DISABLE_NETWORK", "1")

# Stub get_ipython so notebook cells that sniff for Colab don't crash.
import builtins as _builtins
if not hasattr(_builtins, "get_ipython"):
    def get_ipython():  # noqa: D401
        return None
    _builtins.get_ipython = get_ipython

# Make google.colab imports fail cleanly so the notebook takes the non-Colab
# branch (get_ipython() returns None → 'google.colab' in str(None) is False).
class _BlockedColab:
    def find_module(self, name, path=None):
        if name == "google.colab" or name.startswith("google.colab."):
            return self
        return None
    def load_module(self, name):
        raise ImportError("google.colab blocked by validate_graph.py")
_sys.meta_path.insert(0, _BlockedColab())

# Provide dummy keys that cell 4 normally fetches from userdata/env.
oai_key = _os.environ["OPENAI_API_KEY"]
tavily_key = _os.environ["TAVILY_API_KEY"]
"""


def build_sandbox() -> dict:
    mod = types.ModuleType("__idd_validate__")
    mod.__file__ = "<validate_graph>"
    # Register in sys.modules so pydantic's model_rebuild() can locate
    # ``sys.modules[cls.__module__]`` when resolving forward references.
    sys.modules["__idd_validate__"] = mod
    g: dict = mod.__dict__
    g["__builtins__"] = builtins
    exec(compile(PRELUDE, "<prelude>", "exec"), g, g)
    return g


def exec_cells(cells, sandbox: dict, verbose: bool = False, silence: bool = True):
    """Exec each cell. Returns list of (idx, exception|None).

    When ``silence`` is True, the notebook's stdout/stderr noise is captured
    and discarded (except when a cell raises — in that case we still get
    the exception object). This keeps validator output focused and shaves
    real wall-clock time on Windows terminals.
    """
    import io
    import contextlib
    results = []
    sink = io.StringIO()
    ctx = (
        contextlib.redirect_stdout(sink)
        if silence
        else contextlib.nullcontext()
    )
    ctx2 = (
        contextlib.redirect_stderr(sink)
        if silence
        else contextlib.nullcontext()
    )
    with ctx, ctx2:
        for i, src in cells:
            if not src.strip():
                results.append((i, None))
                continue
            filename = f"<cell {i}>"
            try:
                code = compile(src, filename, "exec")
            except SyntaxError as e:
                results.append((i, e))
                return results
            try:
                exec(code, sandbox, sandbox)
                if verbose:
                    # Print via real stderr to bypass redirect
                    print(f"[ok]   cell {i}", file=sys.__stderr__)
                results.append((i, None))
            except Exception as e:  # noqa: BLE001
                results.append((i, e))
                if COMPILE_SENTINEL in src:
                    return results
                if verbose:
                    print(f"[warn] cell {i}: {type(e).__name__}: {e}", file=sys.__stderr__)
    return results


# --------------------------------------------------------------------------- #
# Post-compile inspection
# --------------------------------------------------------------------------- #

def inspect_state(sandbox: dict):
    State = sandbox.get("State")
    info = {"field_count": 0, "reducer_fields": [], "plain_fields": []}
    if State is None or not hasattr(State, "__annotations__"):
        return info

    # Because the notebook uses ``from __future__ import annotations``, the
    # raw ``State.__annotations__`` dict holds strings (PEP 563). Use
    # typing.get_type_hints(..., include_extras=True) with the sandbox as
    # globalns so Annotated[...] wrappers are preserved.
    import typing
    try:
        hints = typing.get_type_hints(State, globalns=sandbox, include_extras=True)
    except Exception:
        # Fall back to raw annotations; reducer detection will use string match.
        hints = dict(State.__annotations__)

    info["field_count"] = len(hints)
    for name, typ in hints.items():
        # Resolved Annotated types expose __metadata__; string fallback uses
        # substring match.
        md = getattr(typ, "__metadata__", None)
        if md is not None and len(md) > 0:
            info["reducer_fields"].append(name)
        elif isinstance(typ, str) and "Annotated[" in typ:
            info["reducer_fields"].append(name)
        else:
            info["plain_fields"].append(name)
    return info


def inspect_graph(compiled):
    g = compiled.get_graph()
    nodes = list(g.nodes.keys())
    edges = list(g.edges)

    # Identify START/END sentinels by id string — they appear as "__start__"
    # and "__end__" in the CompiledGraph.
    START_IDS = {"__start__", "START"}
    END_IDS = {"__end__", "END"}

    real_nodes = [n for n in nodes if n not in START_IDS and n not in END_IDS]

    incoming: dict[str, list] = {n: [] for n in nodes}
    outgoing: dict[str, list] = {n: [] for n in nodes}
    conditional_count = 0
    for e in edges:
        conditional_count += 1 if getattr(e, "conditional", False) else 0
        src = getattr(e, "source", None)
        tgt = getattr(e, "target", None)
        if src in outgoing:
            outgoing[src].append(e)
        if tgt in incoming:
            incoming[tgt].append(e)

    unreachable = [
        n for n in real_nodes
        if not incoming.get(n)
    ]
    dead_ends = [
        n for n in real_nodes
        if not outgoing.get(n)
    ]

    return {
        "nodes": real_nodes,
        "all_node_ids": nodes,
        "edge_count": len(edges),
        "conditional_edge_count": conditional_count,
        "unreachable": unreachable,
        "dead_ends": dead_ends,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _print_header(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--notebook", default=DEFAULT_NOTEBOOK)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument(
        "--inject-bug",
        choices=["missing-node", "missing-reducer"],
        default=None,
        help="Inject a synthetic defect to demonstrate the validator's failure output",
    )
    args = p.parse_args()

    nb_path = Path(args.notebook)
    if not nb_path.is_absolute():
        nb_path = Path.cwd() / nb_path
    if not nb_path.exists():
        print(f"❌ notebook not found: {nb_path}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()

    try:
        nb = load_notebook(nb_path)
    except Exception as e:
        print(f"❌ Failed to read notebook: {e}", file=sys.stderr)
        return 2

    try:
        cells, compile_idx = extract_cells_through_compile(nb, verbose=args.verbose)
    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        return 2

    if args.inject_bug:
        cells = _inject_bug(cells, args.inject_bug, verbose=args.verbose)

    sandbox = build_sandbox()
    results = exec_cells(cells, sandbox, verbose=args.verbose)

    # Did the compile cell succeed?
    compile_result = next(
        (err for (i, err) in results if i == compile_idx), "missing"
    )

    # Collect non-fatal warnings for summary
    cell_warnings = [(i, err) for (i, err) in results if err is not None and i != compile_idx]

    if compile_result == "missing":
        print("❌ Compile cell never executed (earlier cell aborted).")
        fatal = next(((i, err) for (i, err) in results if err is not None), None)
        if fatal:
            i, err = fatal
            print(f"\nFirst failure at cell {i}: {type(err).__name__}: {err}")
            # Dump the cell source for debugging
            _print_header(f"cell {i} source")
            for (ci, src) in cells:
                if ci == i:
                    print(src)
                    break
            _print_header("traceback")
            traceback.print_exception(type(err), err, err.__traceback__)
        elapsed = time.perf_counter() - t0
        print(f"\n⏱  elapsed: {elapsed:.2f}s")
        return 1

    if compile_result is not None:
        err = compile_result
        print("❌ Graph compile FAILED.")
        print(f"   {type(err).__name__}: {err}")
        _print_header(f"compile cell (#{compile_idx}) source")
        for (ci, src) in cells:
            if ci == compile_idx:
                print(src)
                break
        _print_header("traceback")
        traceback.print_exception(type(err), err, err.__traceback__)
        elapsed = time.perf_counter() - t0
        print(f"\n⏱  elapsed: {elapsed:.2f}s")
        return 1

    # ---- success path -----------------------------------------------------
    compiled = sandbox.get(COMPILED_VAR)
    if compiled is None:
        print(f"❌ {COMPILED_VAR} not found in sandbox after compile.")
        return 1

    try:
        graph_info = inspect_graph(compiled)
    except Exception as e:
        print(f"⚠  Graph introspection failed: {e}")
        graph_info = None

    state_info = inspect_state(sandbox)

    elapsed = time.perf_counter() - t0

    print("✅ Graph compiled OK")
    print(f"   notebook : {nb_path.name}")
    print(f"   compile cell index: {compile_idx}")
    print(f"   elapsed  : {elapsed:.2f}s")

    if graph_info:
        _print_header("Graph topology")
        print(f"nodes ({len(graph_info['nodes'])}):")
        for n in sorted(graph_info["nodes"]):
            print(f"  - {n}")
        print(f"edges: {graph_info['edge_count']}  (conditional: {graph_info['conditional_edge_count']})")

        if graph_info["unreachable"]:
            print("\n⚠  unreachable nodes (no incoming edge):")
            for n in graph_info["unreachable"]:
                print(f"  - {n}")
        else:
            print("\n✅ no unreachable nodes")

        if graph_info["dead_ends"]:
            print("\n⚠  dead-end nodes (no outgoing edge and not END):")
            for n in graph_info["dead_ends"]:
                print(f"  - {n}")
        else:
            print("✅ no dead-end nodes")

    if state_info["field_count"]:
        _print_header("State schema")
        print(f"fields          : {state_info['field_count']}")
        print(f"with reducers   : {len(state_info['reducer_fields'])}")
        print(f"plain (no reducer): {len(state_info['plain_fields'])}")
        if args.verbose:
            print("  reducer fields:")
            for f in state_info["reducer_fields"]:
                print(f"    - {f}")
            print("  plain fields:")
            for f in state_info["plain_fields"]:
                print(f"    - {f}")
        # Heuristic: fields named like collections (*_list, sections, messages,
        # viz_*, report_*) without reducers are candidates for InvalidUpdateError.
        SUSPECT_PATTERNS = ("messages", "sections", "tasks", "results", "written_")
        suspect = [
            f for f in state_info["plain_fields"]
            if any(p in f for p in SUSPECT_PATTERNS)
        ]
        if suspect:
            print("\n⚠  collection-like State fields WITHOUT a reducer (potential")
            print("    InvalidUpdateError on concurrent writes — cf. BR-8):")
            for f in suspect:
                print(f"    - {f}")

    if cell_warnings:
        _print_header("Non-fatal pre-compile warnings")
        for (i, err) in cell_warnings:
            print(f"  cell {i}: {type(err).__name__}: {str(err)[:160]}")

    if elapsed > RUNTIME_HARD_BUDGET_S:
        print(f"\n❌ runtime {elapsed:.2f}s exceeded hard budget {RUNTIME_HARD_BUDGET_S}s")
        return 1
    if elapsed > RUNTIME_SOFT_BUDGET_S:
        print(f"\n⚠  runtime {elapsed:.2f}s exceeded soft budget {RUNTIME_SOFT_BUDGET_S}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
