import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from _patch_notebook import RequiredPatchError, replace_required


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHER = REPO_ROOT / "_patch_notebook.py"
PATCHED_NOTEBOOK = REPO_ROOT / "IntelligentDataDetective_beta_v5_patched.ipynb"


def _run_patcher():
    return subprocess.run(
        [sys.executable, str(PATCHER)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _cell_source(notebook, function_name):
    needle = f"def {function_name}("
    matches = [
        "".join(cell.get("source") or [])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and needle in "".join(cell.get("source") or [])
    ]
    assert (
        len(matches) == 1
    ), f"expected one cell containing {needle}, got {len(matches)}"
    return matches[0]


def _function_node(source, function_name):
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture(scope="module")
def generated_notebook():
    first = _run_patcher()
    first_bytes = PATCHED_NOTEBOOK.read_bytes()
    second = _run_patcher()
    second_bytes = PATCHED_NOTEBOOK.read_bytes()
    return {
        "first": first,
        "second": second,
        "first_bytes": first_bytes,
        "second_bytes": second_bytes,
        "notebook": json.loads(second_bytes.decode("utf-8")),
    }


def test_required_replace_fails_fast_when_anchor_drifts():
    with pytest.raises(RequiredPatchError, match="required anchor count was 0"):
        replace_required(
            "current source",
            "missing production anchor",
            "replacement",
            patch_id="TEST-REQUIRED",
        )


def test_patcher_is_clean_deterministic_and_preserves_99_cells(generated_notebook):
    for result in (generated_notebook["first"], generated_notebook["second"]):
        combined_output = result.stdout + result.stderr
        assert "anchor not found" not in combined_output.lower()
        assert "⚠️" not in combined_output
        assert "W14J required report pipeline" in result.stdout
        assert "W14J corrected file_writer final/non-final branch" in result.stdout
        assert "W14J route_to_writer uses reachable W14 state" in result.stdout

    assert generated_notebook["first_bytes"] == generated_notebook["second_bytes"]
    assert len(generated_notebook["notebook"]["cells"]) == 99


def test_generated_report_packager_has_required_canonical_pipeline(
    generated_notebook,
):
    notebook = generated_notebook["notebook"]
    source = _cell_source(notebook, "report_packager_node")
    function = _function_node(source, "report_packager_node")
    function_source = ast.get_source_segment(source, function)

    required_fragments = (
        'draft = f"# {title}\\n\\n" + "\\n\\n".join(written_sections)',
        "_dedupe_long_paragraphs(draft)",
        "_normalize_report_headings(draft, title)",
        "_polish_report_scaffold_leadins(draft)",
        '_resolve_artifact_path("final_report.md"',
        '_resolve_artifact_path("final_report.html"',
        '_resolve_artifact_path("final_report.pdf"',
        "_report_pisa.CreatePDF(",
        "rr.model_copy(update={",
        '"report_generator_complete": report_complete',
    )
    for fragment in required_fragments:
        assert fragment in function_source

    assert function_source.count("W14J-REPORT-PIPELINE") == 1
    assert function_source.index('draft = f"') < function_source.index(
        "_dedupe_long_paragraphs(draft)"
    )
    assert function_source.index("_polish_report_scaffold_leadins(draft)") < (
        function_source.index("md_path.write_text")
    )


def test_generated_file_writer_final_semantics_and_branch_selection(
    generated_notebook,
):
    source = _cell_source(generated_notebook["notebook"], "file_writer_node")
    function = _function_node(source, "file_writer_node")

    is_final_assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "is_final"
            for target in node.targets
        )
    ]
    assert len(is_final_assignments) == 1
    expression = ast.Expression(is_final_assignments[0].value)
    ast.fix_missing_locations(expression)
    code = compile(expression, "<is_final>", "eval")
    assert eval(code, {"state": {"report_generator_complete": False}}) is False
    assert eval(code, {"state": {"report_generator_complete": True}}) is True

    non_final_branches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not is_final"
    ]
    final_manifest_branches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and ast.unparse(node.test).startswith("is_final and isinstance(")
    ]
    fallback_invocations = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "file_writer_agent"
        and node.func.attr == "invoke"
    ]
    assert len(non_final_branches) == 1
    assert len(final_manifest_branches) == 1
    assert len(fallback_invocations) == 1
    assert final_manifest_branches[0].lineno < fallback_invocations[0].lineno


def test_generated_route_to_writer_uses_only_reachable_state(generated_notebook):
    source = _cell_source(generated_notebook["notebook"], "route_to_writer")
    function = _function_node(source, "route_to_writer")
    function_source = ast.get_source_segment(source, function)

    for dead_field in (
        "report_sections_agent_generated",
        "report_section_agent_count",
        "report_packager_agent_generated",
    ):
        assert dead_field not in function_source

    class ReportOutline:
        def __init__(self, count):
            self.sections = [object()] * count

    class ReportResults:
        pass

    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Literal": __import__("typing").Literal,
        "ReportOutline": ReportOutline,
        "ReportResults": ReportResults,
    }
    exec(compile(module, "<route_to_writer>", "exec"), namespace)
    route_to_writer = namespace["route_to_writer"]

    complete = {
        "report_generator_complete": True,
        "report_results": ReportResults(),
        "report_outline": ReportOutline(2),
        "written_sections": ["section 1", "section 2"],
        "report_draft": "# Complete report",
        "file_writer_complete": False,
    }
    assert route_to_writer(complete) == "file_writer"
    assert route_to_writer({**complete, "report_generator_complete": False}) == (
        "supervisor"
    )
    assert route_to_writer({**complete, "written_sections": ["section 1"]}) == (
        "supervisor"
    )
    assert route_to_writer({**complete, "file_writer_complete": True}) == "END"


def test_obsolete_report_state_fields_are_not_generated(generated_notebook):
    source = "\n".join(
        "".join(cell.get("source") or [])
        for cell in generated_notebook["notebook"]["cells"]
        if cell.get("cell_type") == "code"
    )
    for obsolete_field in (
        "report_outline_agent_generated",
        "report_sections_agent_generated",
        "report_section_agent_count",
        "report_packager_agent_generated",
        "report_content_source",
        "report_generation_trace",
    ):
        assert obsolete_field not in source
