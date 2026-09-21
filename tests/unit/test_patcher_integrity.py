import ast
import json
from pathlib import Path
import subprocess
import sys

import logging
import os
import re
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

from PIL import Image
import pytest

from _patch_notebook import RequiredPatchError, replace_required, replace_required_regex
from idd_core import (
    DataVisualization,
    FileResult,
    ListOfFiles,
    ReportOutline,
    ReportResults,
    Section,
    VisualizationResults,
    _get_artifacts_base,
    _is_subpath,
    _resolve_artifact_path,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from validate_artifact_quality import check_embeds, check_pdf


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


def test_replace_required_regex_match_counts():
    text = "alpha beta gamma beta delta"

    # 0 matches: fail-fast on 0
    with pytest.raises(
        RequiredPatchError, match="required structural anchor count was 0"
    ):
        replace_required_regex(
            text,
            r"omega",
            "replacement",
            patch_id="TEST-REGEX-ZERO",
        )

    # 1 match: successful replacement
    single = replace_required_regex(
        text,
        r"alpha",
        "first",
        patch_id="TEST-REGEX-ONE",
    )
    assert single == "first beta gamma beta delta"

    # 2 matches: fail-fast on 2+
    with pytest.raises(
        RequiredPatchError, match="required structural anchor count was 2"
    ):
        replace_required_regex(
            text,
            r"beta",
            "replacement",
            patch_id="TEST-REGEX-TWO",
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
        function_source.index("md_tmp.write_text")
    )
    assert "_report_os.replace(md_tmp, md_path)" in function_source


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


def _create_dummy_png(path: Path, color=(200, 50, 50)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (120, 80), color=color)
    img.save(str(path), format="PNG")


class DummyRuntime:
    def __init__(self, base_dir: Path):
        self.run_dir = str(base_dir)
        self.artifacts_dir = str(base_dir)


class DummyNextAgentMetadata:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DummyCommand:
    def __init__(self, goto=None, update=None):
        self.goto = goto
        self.update = update or {}

    def get(self, key, default=None):
        return self.update.get(key, default)

    def __getitem__(self, key):
        return self.update[key]


def _compile_report_packager_node(notebook, base_dir: Path, mock_safe_invoke=None):
    source = _cell_source(notebook, "report_packager_node")
    function = _function_node(source, "report_packager_node")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, "<report_packager_node>", "exec")

    logger = logging.getLogger("test_report_packager")
    runtime = DummyRuntime(base_dir)

    if mock_safe_invoke is None:

        def mock_safe_invoke(agent, payload, config=None):
            return {
                "structured_response": ReportResults(
                    markdown_report_path="",
                    html_report_path="",
                    pdf_report_path="",
                    reply_msg_to_supervisor="Done draft",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                "messages": [
                    AIMessage(
                        content="Report Packager finished draft.",
                        name="report_packager",
                    )
                ],
            }

    ns = {
        "Optional": Optional,
        "List": List,
        "Dict": Dict,
        "Any": Any,
        "Union": Union,
        "State": dict,
        "sample_prompt_text": "sample prompt",
        "ReportOutline": ReportOutline,
        "Section": Section,
        "DataVisualization": DataVisualization,
        "VisualizationResults": VisualizationResults,
        "ReportResults": ReportResults,
        "NextAgentMetadata": DummyNextAgentMetadata,
        "Command": DummyCommand,
        "AIMessage": AIMessage,
        "HumanMessage": HumanMessage,
        "SendAgentMessage": AIMessage,
        "ChatPromptTemplate": ChatPromptTemplate,
        "MessagesPlaceholder": MessagesPlaceholder,
        "report_generator_prompt_template": ChatPromptTemplate.from_messages(
            [("system", "Task: {report_task}")]
        ),
        "report_generator_tools": [],
        "DEFAULT_TOOLING_GUIDELINES": "guidelines",
        "enhanced_retrieve_mem": lambda s: "memories",
        "get_global_df_registry": lambda: None,
        "_safe_report_packager_invoke": mock_safe_invoke,
        "report_packager_agent": object(),
        "_resolve_artifact_path": _resolve_artifact_path,
        "_get_artifacts_base": _get_artifacts_base,
        "_is_subpath": _is_subpath,
        "update_memory_with_kind": lambda *args, **kwargs: None,
        "in_memory_store": None,
        "get_store": lambda: None,
        "_pl_logger": logger,
        "WORKING_DIRECTORY": base_dir,
        "RUNTIME": runtime,
        "PathlibPath": Path,
        "os": os,
        "re": re,
    }
    exec(code, ns)
    return ns["report_packager_node"]


def _compile_file_writer_node(
    notebook, base_dir: Path, mock_finalizer=None, mock_fw_agent=None
):
    source = _cell_source(notebook, "file_writer_node")
    function = _function_node(source, "file_writer_node")
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    code = compile(module, "<file_writer_node>", "exec")

    logger = logging.getLogger("test_file_writer")
    runtime = DummyRuntime(base_dir)

    if mock_fw_agent is None:
        mock_fw_agent = MagicMock()
        mock_fw_agent.invoke.side_effect = AssertionError(
            "Fallback file_writer_agent was invoked!"
        )

    ns = {
        "Optional": Optional,
        "List": List,
        "Dict": Dict,
        "Any": Any,
        "Union": Union,
        "State": dict,
        "sample_prompt_text": "sample prompt",
        "get_global_df_registry": lambda: None,
        "file_writer_tools": [],
        "ReportResults": ReportResults,
        "VisualizationResults": VisualizationResults,
        "ListOfFiles": ListOfFiles,
        "FileResult": FileResult,
        "_normalize_meta": lambda x: x or {},
        "enhanced_retrieve_mem": lambda s: "memories",
        "file_writer_prompt_template": ChatPromptTemplate.from_messages(
            [("system", "file writer system")]
        ),
        "DEFAULT_TOOLING_GUIDELINES": "guidelines",
        "HumanMessage": HumanMessage,
        "AIMessage": AIMessage,
        "SendAgentMessage": AIMessage,
        "PathlibPath": Path,
        "os": os,
        "re": re,
        "_pl_logger": logger,
        "RUNTIME": runtime,
        "WORKING_DIRECTORY": base_dir,
        "create_agent": lambda *args, **kwargs: mock_finalizer,
        "file_writer_llm": object(),
        "InMemorySaver": lambda: None,
        "in_memory_store": None,
        "ToolStrategy": lambda x: x,
        "_make_unknown_tool_guard": lambda *args: None,
        "file_writer_agent": mock_fw_agent,
        "_resolve_artifact_path": _resolve_artifact_path,
        "_get_artifacts_base": _get_artifacts_base,
        "_is_subpath": _is_subpath,
    }
    exec(code, ns)
    return ns["file_writer_node"]


def test_report_renderer_runtime_with_actual_files(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_rev.png"
    fig2 = viz_dir / "chart_growth.png"
    fig3 = viz_dir / "chart_churn.png"
    _create_dummy_png(fig1, (10, 100, 200))
    _create_dummy_png(fig2, (20, 150, 50))
    _create_dummy_png(fig3, (200, 50, 10))

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    dv1 = DataVisualization(
        path="visualizations/chart_rev.png",
        visualization_id="chart_rev",
        visualization_type="bar",
        visualization_description="Revenue comparison",
        visualization_style="seaborn",
        visualization_title="Revenue By Region",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path="visualizations/chart_growth.png",
        visualization_id="chart_growth",
        visualization_type="line",
        visualization_description="Quarterly Growth",
        visualization_style="seaborn",
        visualization_title="Quarterly Growth",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    sec1 = Section(
        name="Regional Performance",
        section_num=1,
        description="Regional numbers",
        goals=["evaluate revenue"],
        data_signals=["regional_sales"],
        expected_figures=[dv1],
        content=(
            "As discussed in detail, the northern regional business sector showed "
            "unprecedented growth of 15% during the third fiscal quarter, surpassing "
            "all projected metrics and establishing strong market leadership across "
            "all product divisions. This comprehensive analysis evaluates revenue "
            "trends, customer acquisition rates, and margin expansions across sectors."
        ),
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    sec2 = Section(
        name="Growth Trends",
        section_num=2,
        description="Growth trends over quarters",
        goals=["evaluate growth"],
        data_signals=["quarterly_growth"],
        expected_figures=[dv2],
        content=(
            "Quarterly growth shows solid upward momentum across all operational sectors. "
            "Increased investment in core product capability and enhanced distribution "
            "networks contributed significantly to sustainable profitability and long-term "
            "customer retention. Forward-looking indicators suggest continued resilience "
            "in the coming quarters, supporting overall strategic objectives."
        ),
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path="visualizations/chart_churn.png",
        visualization_id="chart_churn",
        visualization_type="scatter",
        visualization_description="Churn analysis",
        visualization_style="seaborn",
        visualization_title="Customer Churn Analysis",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Executive Summary",
            section_num=0,
            description="Overview",
            goals=["summary"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=300,
            title="Quarterly Performance Analysis",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [sec1, sec2],
        "written_sections": [
            f"## Regional Performance\n\n{sec1.content}",
            f"## Growth Trends\n\n{sec2.content}",
        ],
        "visualization_results": VisualizationResults(
            visualizations=[dv3],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is True

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"

    assert md_file.is_file() and md_file.stat().st_size > 0
    assert html_file.is_file() and html_file.stat().st_size > 0
    assert pdf_file.is_file() and pdf_file.stat().st_size > 0

    embed_checks = check_embeds(md_file, html_file)
    for check in embed_checks:
        assert check.passed, f"Embed check failed: {check.name} ({check.detail})"

    pdf_check = check_pdf(pdf_file)
    assert pdf_check.passed, f"PDF check failed: {pdf_check.detail}"

    html_content = html_file.read_text(encoding="utf-8")
    md_content = md_file.read_text(encoding="utf-8")
    assert "../visualizations/chart_rev.png" in html_content
    assert "\\" not in re.findall(r'<img[^>]+src="([^">]+)"', html_content)[0]
    assert "\\" not in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md_content)[0]


def test_report_renderer_figure_deduplication(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_dup.png"
    fig2 = viz_dir / "chart_2.png"
    fig3 = viz_dir / "chart_3.png"
    _create_dummy_png(fig1, (10, 100, 200))
    _create_dummy_png(fig2, (20, 150, 50))
    _create_dummy_png(fig3, (200, 50, 10))

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    dv1 = DataVisualization(
        path="visualizations/chart_dup.png",
        visualization_id="chart_dup",
        visualization_type="bar",
        visualization_description="Duplicate Chart",
        visualization_style="seaborn",
        visualization_title="Duplicate Chart Title",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path="visualizations/chart_2.png",
        visualization_id="chart_2",
        visualization_type="line",
        visualization_description="Chart 2",
        visualization_style="seaborn",
        visualization_title="Chart 2 Title",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path="visualizations/chart_3.png",
        visualization_id="chart_3",
        visualization_type="scatter",
        visualization_description="Chart 3",
        visualization_style="seaborn",
        visualization_title="Chart 3 Title",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    sec1 = Section(
        name="Dup Section",
        section_num=1,
        description="Duplicate chart section",
        goals=["test deduplication"],
        data_signals=["signals"],
        expected_figures=[dv1, dv2, dv3],
        content="This is section content with enough text to describe the duplicate chart scenario.",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Summary",
            section_num=0,
            description="Overview",
            goals=["summary"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Deduplication Report",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [sec1],
        "written_sections": [f"## Dup Section\n\n{sec1.content}"],
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), "visualizations/chart_dup.png", str(fig2), str(fig3)],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is True

    md_content = (reports_dir / "final_report.md").read_text(encoding="utf-8")
    html_content = (reports_dir / "final_report.html").read_text(encoding="utf-8")

    md_matches = re.findall(r"chart_dup\.png", md_content)
    html_matches = re.findall(r"chart_dup\.png", html_content)
    assert (
        len(md_matches) == 1
    ), f"Expected chart_dup.png once in markdown, got {len(md_matches)}"
    assert (
        len(html_matches) == 1
    ), f"Expected chart_dup.png once in html, got {len(html_matches)}"


def test_report_renderer_preserves_and_normalizes_preexisting_markdown_images(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_existing.png"
    fig2 = viz_dir / "chart_existing_2.png"
    fig3 = viz_dir / "chart_existing_3.png"
    _create_dummy_png(fig1, (10, 100, 200))
    _create_dummy_png(fig2, (20, 150, 50))
    _create_dummy_png(fig3, (200, 50, 10))

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    dv1 = DataVisualization(
        path="visualizations/chart_existing.png",
        visualization_id="chart_existing",
        visualization_type="bar",
        visualization_description="Existing Chart",
        visualization_style="seaborn",
        visualization_title="Existing Chart Title",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path="visualizations/chart_existing_2.png",
        visualization_id="chart_existing_2",
        visualization_type="line",
        visualization_description="Existing Chart 2",
        visualization_style="seaborn",
        visualization_title="Existing Chart 2 Title",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path="visualizations/chart_existing_3.png",
        visualization_id="chart_existing_3",
        visualization_type="scatter",
        visualization_description="Existing Chart 3",
        visualization_style="seaborn",
        visualization_title="Existing Chart 3 Title",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    preexisting_md = (
        "## Analysis Section\n\n"
        "Here is the chart:\n\n"
        f"![Existing Chart]({fig1})\n\n"
        f"![Existing Chart 2]({fig2})\n\n"
        f"![Existing Chart 3]({fig3})\n\n"
        + ("Substantive body text describing the visual observations in detail. " * 10)
    )

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Summary",
            section_num=0,
            description="Overview",
            goals=["summary"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Preexisting Image Report",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        "written_sections": [preexisting_md],
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is True

    md_content = (reports_dir / "final_report.md").read_text(encoding="utf-8")
    assert "../visualizations/chart_existing.png" in md_content
    assert md_content.count("chart_existing.png") == 1


def test_report_renderer_transactional_generation_and_stale_output(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_t1.png"
    fig2 = viz_dir / "chart_t2.png"
    fig3 = viz_dir / "chart_t3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="t1",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 1",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path=str(fig2),
        visualization_id="t2",
        visualization_type="line",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 2",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path=str(fig3),
        visualization_id="t3",
        visualization_type="scatter",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 3",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    stale_md = reports_dir / "final_report.md"
    stale_html = reports_dir / "final_report.html"
    stale_pdf = reports_dir / "final_report.pdf"

    stale_md.write_text("STALE MD CONTENT", encoding="utf-8")
    stale_html.write_text("STALE HTML CONTENT", encoding="utf-8")
    stale_pdf.write_text("STALE PDF CONTENT", encoding="utf-8")

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    class MockPisaErr:
        err = 1

    monkeypatch.setattr(
        "xhtml2pdf.pisa.CreatePDF", lambda *args, **kwargs: MockPisaErr()
    )

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Summary",
            section_num=0,
            description="Overview",
            goals=["summary"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Failed Report",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        "written_sections": [
            f"## Section 1\n\n![F1]({fig1})\n\n![F2]({fig2})\n\n![F3]({fig3})\n\nSome text."
        ],
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is False

    assert stale_md.read_text(encoding="utf-8") == "STALE MD CONTENT"
    assert stale_html.read_text(encoding="utf-8") == "STALE HTML CONTENT"
    assert stale_pdf.read_text(encoding="utf-8") == "STALE PDF CONTENT"

    tmp_files = list(reports_dir.glob("_tmp_*"))
    assert len(tmp_files) == 0, f"Leftover temporary files found: {tmp_files}"


def test_file_writer_node_final_manifest_runtime(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_rev.png"
    fig2 = viz_dir / "chart_growth.png"
    fig3 = viz_dir / "chart_churn.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    md_file.write_text(
        "# Final Report\n\n"
        f"![Rev](../visualizations/{fig1.name})\n"
        f"![Growth](../visualizations/{fig2.name})\n"
        f"![Churn](../visualizations/{fig3.name})\n",
        encoding="utf-8",
    )
    html_file.write_text(
        "<h1>Final Report</h1>\n"
        f'<img src="../visualizations/{fig1.name}">\n'
        f'<img src="../visualizations/{fig2.name}">\n'
        f'<img src="../visualizations/{fig3.name}">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    mock_fw_agent = MagicMock()
    mock_fw_agent.invoke.side_effect = AssertionError(
        "Ordinary write-agent fallback was invoked!"
    )

    fr_report = FileResult(
        write_success=True,
        file_path=str(html_file),
        file_type="html",
        file_name="final_report.html",
        file_description="Final HTML report",
        is_final_report=True,
        category_tag="report",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    fr_md = FileResult(
        write_success=True,
        file_path=str(md_file),
        file_type="markdown",
        file_name="final_report.md",
        file_description="Final MD report",
        is_final_report=False,
        category_tag="report",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    fr_pdf = FileResult(
        write_success=True,
        file_path=str(pdf_file),
        file_type="pdf",
        file_name="final_report.pdf",
        file_description="Final PDF report",
        is_final_report=False,
        category_tag="report",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    fr_v1 = FileResult(
        write_success=True,
        file_path=str(fig1),
        file_type="png",
        file_name="chart_rev.png",
        file_description="Chart Rev",
        is_final_report=False,
        category_tag="visualization",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    fr_v2 = FileResult(
        write_success=True,
        file_path=str(fig2),
        file_type="png",
        file_name="chart_growth.png",
        file_description="Chart Growth",
        is_final_report=False,
        category_tag="visualization",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    fr_v3 = FileResult(
        write_success=True,
        file_path=str(fig3),
        file_type="png",
        file_name="chart_churn.png",
        file_description="Chart Churn",
        is_final_report=False,
        category_tag="visualization",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    mock_manifest_result = ListOfFiles(
        files=[fr_report, fr_md, fr_pdf, fr_v1, fr_v2, fr_v3],
        reply_msg_to_supervisor="Manifest ready",
        finished_this_task=True,
        expect_reply=False,
    )

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": mock_manifest_result,
        "messages": [AIMessage(content="Manifest ready", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
        mock_fw_agent=mock_fw_agent,
    )

    runtime = DummyRuntime(tmp_path)
    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[
                DataVisualization(
                    path=str(fig1),
                    visualization_id="v1",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig2),
                    visualization_id="v2",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig3),
                    visualization_id="v3",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
            ],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
        "written_sections": ["# Final Report\n\nContent"],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": runtime}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is True
    assert Path(res.get("final_report_path")).resolve() == html_file.resolve()
    mock_fw_agent.invoke.assert_not_called()


def test_file_writer_node_incomplete_manifest_missing_visualization(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    md_file.write_text("# Final Report", encoding="utf-8")
    html_file.write_text("<h1>Final Report</h1>", encoding="utf-8")
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    fig1 = viz_dir / "chart_rev.png"
    fig2 = viz_dir / "chart_growth.png"
    fig3 = viz_dir / "chart_missing.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    # fig3 is intentionally NOT created on disk

    mock_fw_agent = MagicMock()
    mock_fw_agent.invoke.side_effect = AssertionError(
        "Ordinary write-agent fallback was invoked!"
    )

    fr_report = FileResult(
        write_success=True,
        file_path=str(html_file),
        file_type="html",
        file_name="final_report.html",
        file_description="Final HTML report",
        is_final_report=True,
        category_tag="report",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    fr_md = FileResult(
        write_success=True,
        file_path=str(md_file),
        file_type="markdown",
        file_name="final_report.md",
        file_description="Final MD report",
        is_final_report=False,
        category_tag="report",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    fr_pdf = FileResult(
        write_success=True,
        file_path=str(pdf_file),
        file_type="pdf",
        file_name="final_report.pdf",
        file_description="Final PDF report",
        is_final_report=False,
        category_tag="report",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    fr_v1 = FileResult(
        write_success=True,
        file_path=str(fig1),
        file_type="png",
        file_name="chart_rev.png",
        file_description="Chart Rev",
        is_final_report=False,
        category_tag="visualization",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    fr_v2 = FileResult(
        write_success=True,
        file_path=str(fig2),
        file_type="png",
        file_name="chart_growth.png",
        file_description="Chart Growth",
        is_final_report=False,
        category_tag="visualization",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    fr_v3 = FileResult(
        write_success=True,
        file_path=str(fig3),
        file_type="png",
        file_name="chart_missing.png",
        file_description="Chart Missing",
        is_final_report=False,
        category_tag="visualization",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    mock_manifest_result = ListOfFiles(
        files=[fr_report, fr_md, fr_pdf, fr_v1, fr_v2, fr_v3],
        reply_msg_to_supervisor="Manifest ready",
        finished_this_task=True,
        expect_reply=False,
    )

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": mock_manifest_result,
        "messages": [AIMessage(content="Manifest ready", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
        mock_fw_agent=mock_fw_agent,
    )

    runtime = DummyRuntime(tmp_path)
    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[
                DataVisualization(
                    path=str(fig1),
                    visualization_id="v1",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig2),
                    visualization_id="v2",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig3),
                    visualization_id="v3",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
            ],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
        "written_sections": ["# Final Report\n\nContent"],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": runtime}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is False


# --- Task B: Stale Markdown Source Selection Tests ---


def test_renderer_rejects_stale_canonical_markdown_source(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_b1.png"
    fig2 = viz_dir / "chart_b2.png"
    fig3 = viz_dir / "chart_b3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    stale_md = reports_dir / "final_report.md"
    stale_md.write_text(
        "# STALE REPORT TITLE\n\nStale old report body", encoding="utf-8"
    )

    def mock_safe_invoke(agent, payload, config=None):
        return {
            "structured_response": ReportResults(
                markdown_report_path=str(stale_md),
                html_report_path="",
                pdf_report_path="",
                reply_msg_to_supervisor="Done draft",
                finished_this_task=True,
                expect_reply=False,
            ),
            "messages": [AIMessage(content="Packager done", name="report_packager")],
        }

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path, mock_safe_invoke=mock_safe_invoke
    )

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Summary",
            section_num=0,
            description="Overview",
            goals=["summary"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Fresh Report 2026",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        "written_sections": [
            "## Fresh Section\n\nAuthoritative draft text from written sections."
        ],
        "visualization_results": VisualizationResults(
            visualizations=[
                DataVisualization(
                    path=str(fig1),
                    visualization_id="b1",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig2),
                    visualization_id="b2",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig3),
                    visualization_id="b3",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
            ],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is True

    published_md = stale_md.read_text(encoding="utf-8")
    assert "STALE REPORT TITLE" not in published_md
    assert "Stale old report body" not in published_md
    assert "Fresh Report 2026" in published_md
    assert "Authoritative draft text from written sections." in published_md


def test_renderer_rejects_stale_noncanonical_markdown_source(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_nb1.png"
    fig2 = viz_dir / "chart_nb2.png"
    fig3 = viz_dir / "chart_nb3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    rogue_md = tmp_path / "rogue_report.md"
    rogue_md.write_text("# ROGUE PROBE CONTENT\n\nShould be ignored", encoding="utf-8")

    def mock_safe_invoke(agent, payload, config=None):
        return {
            "structured_response": ReportResults(
                markdown_report_path=str(rogue_md),
                html_report_path="",
                pdf_report_path="",
                reply_msg_to_supervisor="Done draft",
                finished_this_task=True,
                expect_reply=False,
            ),
            "messages": [AIMessage(content="Packager done", name="report_packager")],
        }

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path, mock_safe_invoke=mock_safe_invoke
    )

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Summary",
            section_num=0,
            description="Overview",
            goals=["summary"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Legitimate Title",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        "written_sections": [
            "## Section One\n\nLegitimate body from written sections."
        ],
        "visualization_results": VisualizationResults(
            visualizations=[
                DataVisualization(
                    path=str(fig1),
                    visualization_id="nb1",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig2),
                    visualization_id="nb2",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig3),
                    visualization_id="nb3",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
            ],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is True

    published_md = (reports_dir / "final_report.md").read_text(encoding="utf-8")
    assert "ROGUE PROBE CONTENT" not in published_md
    assert "Legitimate Title" in published_md
    assert "Legitimate body from written sections." in published_md


# --- Task C: Distinct-Artifact Completion Checks Tests ---


def test_manifest_rejects_two_figures_with_path_aliases(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    md_file.write_text("# Final Report", encoding="utf-8")
    html_file.write_text(
        f"<h1>Final Report</h1>\n"
        f'<img src="../visualizations/{fig1.name}">\n'
        f'<img src="../visualizations/{fig2.name}">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    mock_fw_agent = MagicMock()
    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_manifest = ListOfFiles(
        files=[
            FileResult(
                write_success=True,
                file_path=str(html_file),
                file_type="html",
                file_name=html_file.name,
                file_description="html",
                is_final_report=True,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(md_file),
                file_type="markdown",
                file_name=md_file.name,
                file_description="md",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(pdf_file),
                file_type="pdf",
                file_name=pdf_file.name,
                file_description="pdf",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig1),
                file_type="png",
                file_name=fig1.name,
                file_description="v1",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig2),
                file_type="png",
                file_name=fig2.name,
                file_description="v2",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=f"visualizations/{fig1.name}",
                file_type="png",
                file_name=fig1.name,
                file_description="v1 alias",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
        ],
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    mock_finalizer.invoke.return_value = {
        "structured_response": mock_manifest,
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
        mock_fw_agent=mock_fw_agent,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[
                DataVisualization(
                    path=str(fig1),
                    visualization_id="v1",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig2),
                    visualization_id="v2",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=f"visualizations/{fig1.name}",
                    visualization_id="v1_alias",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
            ],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), f"visualizations/{fig1.name}"],
        "written_sections": ["# Final Report\n\nContent"],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is False


def test_manifest_deduplicates_three_figures_with_aliases(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    md_file.write_text(
        f"# Final Report\n\n"
        f"![Fig 1](../visualizations/{fig1.name})\n"
        f"![Fig 2](../visualizations/{fig2.name})\n"
        f"![Fig 3](../visualizations/{fig3.name})\n",
        encoding="utf-8",
    )
    html_file.write_text(
        f"<h1>Final Report</h1>\n"
        f'<img src="../visualizations/{fig1.name}">\n'
        f'<img src="../visualizations/{fig2.name}">\n'
        f'<img src="../visualizations/{fig3.name}">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    mock_fw_agent = MagicMock()
    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_manifest = ListOfFiles(
        files=[
            FileResult(
                write_success=True,
                file_path=str(html_file),
                file_type="html",
                file_name=html_file.name,
                file_description="html",
                is_final_report=True,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(md_file),
                file_type="markdown",
                file_name=md_file.name,
                file_description="md",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(pdf_file),
                file_type="pdf",
                file_name=pdf_file.name,
                file_description="pdf",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig1),
                file_type="png",
                file_name=fig1.name,
                file_description="v1",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig2),
                file_type="png",
                file_name=fig2.name,
                file_description="v2",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig3),
                file_type="png",
                file_name=fig3.name,
                file_description="v3",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
        ],
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    mock_finalizer.invoke.return_value = {
        "structured_response": mock_manifest,
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
        mock_fw_agent=mock_fw_agent,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[
                DataVisualization(
                    path=str(fig1),
                    visualization_id="v1",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=f"visualizations/{fig1.name}",
                    visualization_id="v1_alias",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig2),
                    visualization_id="v2",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig3),
                    visualization_id="v3",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
            ],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), f"visualizations/{fig1.name}", str(fig2), str(fig3)],
        "written_sections": ["# Final Report\n\nContent"],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is True
    assert len(res.get("viz_paths")) == 3


def test_manifest_rejects_duplicate_report_entries(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    md_file.write_text("# Final Report", encoding="utf-8")
    html_file.write_text(
        f"<h1>Final Report</h1>\n"
        f'<img src="../visualizations/{fig1.name}">\n'
        f'<img src="../visualizations/{fig2.name}">\n'
        f'<img src="../visualizations/{fig3.name}">\n',
        encoding="utf-8",
    )

    mock_fw_agent = MagicMock()
    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_manifest = ListOfFiles(
        files=[
            FileResult(
                write_success=True,
                file_path=str(html_file),
                file_type="html",
                file_name=html_file.name,
                file_description="html",
                is_final_report=True,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(html_file),
                file_type="html",
                file_name=html_file.name,
                file_description="duplicate html",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(md_file),
                file_type="markdown",
                file_name=md_file.name,
                file_description="md",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig1),
                file_type="png",
                file_name=fig1.name,
                file_description="v1",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig2),
                file_type="png",
                file_name=fig2.name,
                file_description="v2",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig3),
                file_type="png",
                file_name=fig3.name,
                file_description="v3",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
        ],
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    mock_finalizer.invoke.return_value = {
        "structured_response": mock_manifest,
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
        mock_fw_agent=mock_fw_agent,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path="",
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[
                DataVisualization(
                    path=str(fig1),
                    visualization_id="v1",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig2),
                    visualization_id="v2",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig3),
                    visualization_id="v3",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
            ],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
        "written_sections": ["# Final Report\n\nContent"],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is False


def test_manifest_rejects_missing_expected_figure(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    fig_ghost = viz_dir / "chart_missing_expected.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    md_file.write_text("# Final Report", encoding="utf-8")
    html_file.write_text(
        f"<h1>Final Report</h1>\n"
        f'<img src="../visualizations/{fig1.name}">\n'
        f'<img src="../visualizations/{fig2.name}">\n'
        f'<img src="../visualizations/{fig3.name}">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    mock_fw_agent = MagicMock()
    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_manifest = ListOfFiles(
        files=[
            FileResult(
                write_success=True,
                file_path=str(html_file),
                file_type="html",
                file_name=html_file.name,
                file_description="html",
                is_final_report=True,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(md_file),
                file_type="markdown",
                file_name=md_file.name,
                file_description="md",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(pdf_file),
                file_type="pdf",
                file_name=pdf_file.name,
                file_description="pdf",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig1),
                file_type="png",
                file_name=fig1.name,
                file_description="v1",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig2),
                file_type="png",
                file_name=fig2.name,
                file_description="v2",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig3),
                file_type="png",
                file_name=fig3.name,
                file_description="v3",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
        ],
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    mock_finalizer.invoke.return_value = {
        "structured_response": mock_manifest,
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
        mock_fw_agent=mock_fw_agent,
    )

    sec_with_ghost = Section(
        name="Ghost Section",
        section_num=1,
        description="Section expecting ghost figure",
        goals=["ghost test"],
        data_signals=["signals"],
        expected_figures=[
            DataVisualization(
                path=str(fig_ghost),
                visualization_id="ghost",
                visualization_type="bar",
                visualization_description="missing",
                visualization_style="s",
                visualization_title="Ghost Figure",
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
            )
        ],
        content="Section body text",
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [sec_with_ghost],
        "visualization_results": VisualizationResults(
            visualizations=[
                DataVisualization(
                    path=str(fig1),
                    visualization_id="v1",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig2),
                    visualization_id="v2",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig3),
                    visualization_id="v3",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
            ],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
        "written_sections": ["# Final Report\n\nContent"],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is False


def test_manifest_requires_canonical_report_embeds(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    md_file.write_text("# Final Report", encoding="utf-8")
    html_file.write_text(
        "<h1>Final Report With Zero Images</h1><p>Body text without embeds.</p>",
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    mock_fw_agent = MagicMock()
    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_manifest = ListOfFiles(
        files=[
            FileResult(
                write_success=True,
                file_path=str(html_file),
                file_type="html",
                file_name=html_file.name,
                file_description="html",
                is_final_report=True,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(md_file),
                file_type="markdown",
                file_name=md_file.name,
                file_description="md",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(pdf_file),
                file_type="pdf",
                file_name=pdf_file.name,
                file_description="pdf",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig1),
                file_type="png",
                file_name=fig1.name,
                file_description="v1",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig2),
                file_type="png",
                file_name=fig2.name,
                file_description="v2",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig3),
                file_type="png",
                file_name=fig3.name,
                file_description="v3",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="done",
                finished_this_task=True,
                expect_reply=False,
            ),
        ],
        reply_msg_to_supervisor="done",
        finished_this_task=True,
        expect_reply=False,
    )
    mock_finalizer.invoke.return_value = {
        "structured_response": mock_manifest,
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
        mock_fw_agent=mock_fw_agent,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[
                DataVisualization(
                    path=str(fig1),
                    visualization_id="v1",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig2),
                    visualization_id="v2",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
                DataVisualization(
                    path=str(fig3),
                    visualization_id="v3",
                    visualization_type="bar",
                    visualization_description="d",
                    visualization_style="s",
                    visualization_title="t",
                    reply_msg_to_supervisor="ok",
                    finished_this_task=True,
                    expect_reply=False,
                ),
            ],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
        "written_sections": ["# Final Report\n\nContent"],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is False


# --- Task D: Markdown-to-HTML/PDF Fidelity Tests ---


def test_markdown_parser_semantic_html_and_pdf_fidelity(
    generated_notebook, tmp_path, monkeypatch
):
    import fitz

    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "fidelity_chart.png"
    fig2 = viz_dir / "fidelity_chart_2.png"
    fig3 = viz_dir / "fidelity_chart_3.png"
    _create_dummy_png(fig1, color=(40, 120, 200))
    _create_dummy_png(fig2, color=(60, 140, 180))
    _create_dummy_png(fig3, color=(80, 160, 160))

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="fid1",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Executive Metrics Chart",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path=str(fig2),
        visualization_id="fid2",
        visualization_type="line",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Executive Metrics Chart 2",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path=str(fig3),
        visualization_id="fid3",
        visualization_type="scatter",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Executive Metrics Chart 3",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    markdown_source = (
        "## Summary Metrics\n\n"
        "Here is the executive performance table:\n\n"
        "| Metric | Target | Actual |\n"
        "| :--- | :--- | :--- |\n"
        "| Conversion Rate | 4.5% | 5.2% |\n"
        "| Churn Rate | < 2.0% | 1.4% |\n\n"
        "Key highlights:\n\n"
        "- Strong **outperformance** across all metrics\n"
        "- Sustained retention above industry benchmark\n\n"
        "Code reference for validation:\n"
        "```python\n"
        "def compute_delta(act, tgt):\n"
        "    return act - tgt\n"
        "```\n\n"
        f"![Executive Metrics Chart]({fig1})\n\n"
        f"![Executive Metrics Chart 2]({fig2})\n\n"
        f"![Executive Metrics Chart 3]({fig3})\n\n"
        "For additional details, see the [Reference Documentation](https://example.com/docs).\n\n"
        + (
            "Substantive narrative describing quarterly performance indicators in comprehensive detail. "
            * 5
        )
    )

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Executive Summary",
            section_num=0,
            description="Overview",
            goals=["fidelity"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=150,
            title="Semantic Fidelity Report",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        "written_sections": [markdown_source],
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is True

    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    assert html_file.is_file() and html_file.stat().st_size > 0
    assert pdf_file.is_file() and pdf_file.stat().st_size > 0

    html_text = html_file.read_text(encoding="utf-8")
    assert "<table" in html_text and "</table>" in html_text
    assert "<th" in html_text and "<td" in html_text
    assert "Conversion Rate" in html_text
    assert "<pre><code" in html_text
    assert "def compute_delta" in html_text
    assert "<strong>outperformance</strong>" in html_text
    assert "<ul>" in html_text and "<li>" in html_text
    assert '<a href="https://example.com/docs">Reference Documentation</a>' in html_text
    assert f'src="../visualizations/{fig1.name}"' in html_text

    doc = fitz.open(str(pdf_file))
    try:
        assert len(doc) >= 1
        page_text = " ".join(page.get_text() for page in doc)
        assert "Summary Metrics" in page_text
        assert "Conversion Rate" in page_text
        assert "compute_delta" in page_text
        images_found = sum(len(page.get_images()) for page in doc)
        assert (
            images_found >= 1
        ), f"Expected embedded image in PDF, found {images_found}"
    finally:
        doc.close()


@pytest.mark.parametrize(
    "unsafe_payload",
    [
        "javascript:alert('xss')",
        "jav&#x61;script:alert('xss')",
        "jav&#97;script:alert('xss')",
        "JaVaScRiPt:alert('xss')",
        "jav\t\nascript:alert('xss')",
        "vbscript:msgbox('xss')",
        "data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==",
    ],
)
def test_markdown_parser_sanitizes_untrusted_html_and_unsafe_links(
    generated_notebook, tmp_path, monkeypatch, unsafe_payload
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_safe.png"
    fig2 = viz_dir / "chart_safe_2.png"
    fig3 = viz_dir / "chart_safe_3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="s1",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Safe Chart",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path=str(fig2),
        visualization_id="s2",
        visualization_type="line",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Safe Chart 2",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path=str(fig3),
        visualization_id="s3",
        visualization_type="scatter",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Safe Chart 3",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    untrusted_source = (
        "## Security Audit Section\n\n"
        "Here is legitimate content.\n\n"
        '<script>alert("evil")</script>\n\n'
        '<img src="../visualizations/chart_safe.png" onload="maliciousPayload()" alt="Test Image">\n\n'
        f'<a href="{unsafe_payload}">Untrusted Link</a>\n\n'
        '[Click for safe hash](#section-safe)\n\n'
        f"![Safe Chart]({fig1})\n\n"
        f"![Safe Chart 2]({fig2})\n\n"
        f"![Safe Chart 3]({fig3})\n\n"
        + (
            "Substantive body text discussing vulnerability scan and sanitization results in detail. "
            * 5
        )
    )

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Security Review",
            section_num=0,
            description="Overview",
            goals=["security"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Sanitization Report",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        "written_sections": [untrusted_source],
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is True

    html_text = (reports_dir / "final_report.html").read_text(encoding="utf-8")
    assert "<script" not in html_text
    assert "onload=" not in html_text
    assert 'alert("evil")' not in html_text
    assert unsafe_payload not in html_text
    assert 'href="#section-safe"' in html_text
    assert "../visualizations/chart_safe.png" in html_text

    from html.parser import HTMLParser

    class _LinkChecker(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.links.append(dict(attrs).get("href"))

    _chk = _LinkChecker()
    _chk.feed(html_text)
    for link in _chk.links:
        if link:
            assert not any(
                link.lower().startswith(p)
                for p in ("javascript:", "vbscript:", "data:")
            ), f"Unsafe scheme in href: {link}"


# --- Task E: Transactional Publication & Rollback Tests ---


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_publication_replacement_failure_restores_canonical_files(
    generated_notebook, tmp_path, monkeypatch, failure_index
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_t.png"
    fig2 = viz_dir / "chart_t2.png"
    fig3 = viz_dir / "chart_t3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="t1",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path=str(fig2),
        visualization_id="t2",
        visualization_type="line",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 2",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path=str(fig3),
        visualization_id="t3",
        visualization_type="scatter",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 3",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    prior_md = reports_dir / "final_report.md"
    prior_html = reports_dir / "final_report.html"
    prior_pdf = reports_dir / "final_report.pdf"

    original_md_content = "# PREVIOUS MD CONTENT"
    original_html_content = "<h1>PREVIOUS HTML CONTENT</h1>"
    original_pdf_bytes = b"%PDF-1.4 PREVIOUS PDF BYTES"

    prior_md.write_text(original_md_content, encoding="utf-8")
    prior_html.write_text(original_html_content, encoding="utf-8")
    prior_pdf.write_bytes(original_pdf_bytes)

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    replace_call_count = 0
    real_replace = os.replace

    def mock_replace(src, dst):
        nonlocal replace_call_count
        dst_name = Path(dst).name
        if dst_name in (
            "final_report.md",
            "final_report.html",
            "final_report.pdf",
        ) and not str(dst).startswith("._"):
            if replace_call_count == failure_index:
                replace_call_count += 1
                raise OSError(
                    f"Simulated atomic replace failure at step {failure_index}"
                )
            replace_call_count += 1
        return real_replace(src, dst)

    monkeypatch.setattr("os.replace", mock_replace)

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Transactional Test",
            section_num=0,
            description="Overview",
            goals=["transactional"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Failed Replacement Report",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        "written_sections": [
            f"## Section\n\nContent\n\n![Chart]({fig1})\n\n![Chart 2]({fig2})\n\n![Chart 3]({fig3})"
        ],
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is False

    # Byte-for-byte rollback verification
    assert prior_md.read_text(encoding="utf-8") == original_md_content
    assert prior_html.read_text(encoding="utf-8") == original_html_content
    assert prior_pdf.read_bytes() == original_pdf_bytes

    # Staging and backup directories cleaned up
    leftover_staging = list(reports_dir.glob("._staging_*"))
    leftover_backup = list(reports_dir.glob("._backup_*"))
    assert len(leftover_staging) == 0, f"Leftover staging dirs: {leftover_staging}"
    assert len(leftover_backup) == 0, f"Leftover backup dirs: {leftover_backup}"


def test_publication_replacement_failure_with_no_prior_files_cleans_up(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_clean.png"
    fig2 = viz_dir / "chart_clean2.png"
    fig3 = viz_dir / "chart_clean3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="c1",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path=str(fig2),
        visualization_id="c2",
        visualization_type="line",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 2",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path=str(fig3),
        visualization_id="c3",
        visualization_type="scatter",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 3",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    replace_call_count = 0
    real_replace = os.replace

    def mock_replace(src, dst):
        nonlocal replace_call_count
        dst_name = Path(dst).name
        if dst_name in (
            "final_report.md",
            "final_report.html",
            "final_report.pdf",
        ) and not str(dst).startswith("._"):
            if replace_call_count == 1:
                replace_call_count += 1
                raise OSError("Simulated atomic replace failure at step 1")
            replace_call_count += 1
        return real_replace(src, dst)

    monkeypatch.setattr("os.replace", mock_replace)

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Transactional Test",
            section_num=0,
            description="Overview",
            goals=["transactional"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Clean Directory Failed Replacement",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        "written_sections": [
            f"## Section\n\nContent\n\n![Chart]({fig1})\n\n![Chart 2]({fig2})\n\n![Chart 3]({fig3})"
        ],
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is False

    assert not (reports_dir / "final_report.md").exists()
    assert not (reports_dir / "final_report.html").exists()
    assert not (reports_dir / "final_report.pdf").exists()


def test_publication_rollback_failure_raises_hard_runtime_error_and_retains_dirs(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart_rf.png"
    _create_dummy_png(fig1)
    fig2 = viz_dir / "chart_rf2.png"
    _create_dummy_png(fig2)
    fig3 = viz_dir / "chart_rf3.png"
    _create_dummy_png(fig3)

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="rf1",
        visualization_type="bar",
        visualization_description="d1",
        visualization_style="s1",
        visualization_title="Chart 1",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path=str(fig2),
        visualization_id="rf2",
        visualization_type="line",
        visualization_description="d2",
        visualization_style="s2",
        visualization_title="Chart 2",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path=str(fig3),
        visualization_id="rf3",
        visualization_type="scatter",
        visualization_description="d3",
        visualization_style="s3",
        visualization_title="Chart 3",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    prior_md = reports_dir / "final_report.md"
    prior_html = reports_dir / "final_report.html"
    prior_pdf = reports_dir / "final_report.pdf"
    prior_md.write_text("MD", encoding="utf-8")
    prior_html.write_text("HTML", encoding="utf-8")
    prior_pdf.write_text("PDF", encoding="utf-8")

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    replace_calls = 0
    in_rollback = False
    real_replace = os.replace

    def catastrophic_replace(src, dst):
        nonlocal replace_calls, in_rollback
        dst_name = Path(dst).name
        if dst_name in (
            "final_report.md",
            "final_report.html",
            "final_report.pdf",
        ) and not str(dst).startswith("._"):
            if not in_rollback:
                if replace_calls == 1:
                    in_rollback = True
                    raise OSError("Publication replace failed on html")
                replace_calls += 1
            else:
                raise OSError("Catastrophic disk failure during rollback restore")
        return real_replace(src, dst)

    monkeypatch.setattr("os.replace", catastrophic_replace)

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Rollback Failure",
            section_num=0,
            description="Overview",
            goals=["fail"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Catastrophic Failure Report",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        "written_sections": [
            f"## Section\n\nContent\n\n![Chart 1]({fig1})\n\n![Chart 2]({fig2})\n\n![Chart 3]({fig3})"
        ],
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
    }

    with pytest.raises(RuntimeError, match="Rollback failed during report publication"):
        report_packager_node(state)

    staging_dirs = list(reports_dir.glob("._staging_*"))
    backup_dirs = list(reports_dir.glob("._backup_*"))
    assert (
        len(staging_dirs) >= 1
    ), f"Expected retained staging dir, found {staging_dirs}"
    assert len(backup_dirs) >= 1, f"Expected retained backup dir, found {backup_dirs}"


# --- Additional Regression Tests for PR #149 correctness findings ---


def test_final_manifest_rejects_outside_root_absolute_path(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    outside_dir = tmp_path.parent / "outside_allowed_root"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    outside_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3_outside = outside_dir / "chart_outside.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3_outside)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    md_file.write_text(
        f"# Final Report\n\n![C1](../visualizations/chart1.png)\n![C2](../visualizations/chart2.png)\n![Out]({fig3_outside})\n",
        encoding="utf-8",
    )
    html_file.write_text(
        f'<h1>Final Report</h1>\n<img src="../visualizations/chart1.png">\n<img src="../visualizations/chart2.png">\n<img src="{fig3_outside}">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="v1",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 1",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path=str(fig2),
        visualization_id="v2",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 2",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3_outside = DataVisualization(
        path=str(fig3_outside),
        visualization_id="v3",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart Outside",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    mock_manifest_result = ListOfFiles(
        files=[
            FileResult(
                write_success=True,
                file_path=str(html_file),
                file_type="html",
                file_name="final_report.html",
                file_description="report",
                is_final_report=True,
                category_tag="report",
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(md_file),
                file_type="markdown",
                file_name="final_report.md",
                file_description="report",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(pdf_file),
                file_type="pdf",
                file_name="final_report.pdf",
                file_description="report",
                is_final_report=False,
                category_tag="report",
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig1),
                file_type="png",
                file_name="chart1.png",
                file_description="chart",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig2),
                file_type="png",
                file_name="chart2.png",
                file_description="chart",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
            ),
            FileResult(
                write_success=True,
                file_path=str(fig3_outside),
                file_type="png",
                file_name="chart_outside.png",
                file_description="outside",
                is_final_report=False,
                category_tag="visualization",
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
            ),
        ],
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": mock_manifest_result,
        "messages": [AIMessage(content="Manifest ready", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3_outside],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3_outside)],
        "written_sections": [md_file.read_text(encoding="utf-8")],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is False
    file_res = res.get("file_results")
    if hasattr(file_res, "files"):
        reconciled_paths = [f.file_path for f in file_res.files]
    elif isinstance(file_res, list):
        reconciled_paths = [getattr(f, "file_path", str(f)) for f in file_res]
    else:
        reconciled_paths = []
    assert str(fig3_outside) not in reconciled_paths


def test_artifact_resolution_rejects_traversal_and_symlink_escape(tmp_path):
    import types
    outside_dir = tmp_path.parent / "outside_jail"
    outside_dir.mkdir(parents=True, exist_ok=True)
    secret_file = outside_dir / "secret.png"
    secret_file.write_bytes(b"secret")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg = types.SimpleNamespace(
        configurable={"runtime": types.SimpleNamespace(artifacts_dir=str(artifacts_dir))}
    )

    # 1. Path traversal escape
    traversal_path = "../../outside_jail/secret.png"
    with pytest.raises(ValueError, match="(?i)outside|refused|artifacts"):
        _resolve_artifact_path(traversal_path, config=cfg)

    # 2. Symlink escape
    symlink_file = artifacts_dir / "symlink_secret.png"
    try:
        symlink_file.symlink_to(secret_file)
        with pytest.raises(ValueError, match="(?i)outside|refused|artifacts"):
            _resolve_artifact_path(str(symlink_file), config=cfg)
    except (OSError, NotImplementedError):
        pass


def test_missing_expected_path_is_not_replaced_by_basename_decoy(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)

    # An unrelated file with the same basename in the run root:
    decoy_file = tmp_path / "missing_chart.png"
    decoy_file.write_bytes(b"decoy contents")

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    md_file.write_text(
        "# Final Report\n\n"
        "![C1](../visualizations/chart1.png)\n"
        "![C2](../visualizations/chart2.png)\n"
        "![Missing](visualizations/missing_chart.png)\n",
        encoding="utf-8",
    )
    html_file.write_text(
        "<h1>Final Report</h1>\n"
        '<img src="../visualizations/chart1.png">\n'
        '<img src="../visualizations/chart2.png">\n'
        '<img src="../visualizations/missing_chart.png">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="v1",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 1",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path=str(fig2),
        visualization_id="v2",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 2",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv_missing = DataVisualization(
        path="visualizations/missing_chart.png",
        visualization_id="v_missing",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Missing Chart",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": ListOfFiles(
            files=[],
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        ),
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv_missing],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), "visualizations/missing_chart.png"],
        "written_sections": [md_file.read_text(encoding="utf-8")],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is False
    file_res = res.get("file_results")
    if hasattr(file_res, "files"):
        file_paths = [f.file_path for f in file_res.files]
    elif isinstance(file_res, list):
        file_paths = [getattr(f, "file_path", str(f)) for f in file_res]
    else:
        file_paths = []
    assert str(decoy_file) not in file_paths


def test_report_embed_uses_document_relative_context(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    # Embeds use ../visualizations relative to reports_dir
    md_file.write_text(
        "# Final Report\n\n"
        "![C1](../visualizations/chart1.png)\n"
        "![C2](../visualizations/chart2.png)\n"
        "![C3](../visualizations/chart3.png)\n",
        encoding="utf-8",
    )
    html_file.write_text(
        "<h1>Final Report</h1>\n"
        '<img src="../visualizations/chart1.png">\n'
        '<img src="../visualizations/chart2.png">\n'
        '<img src="../visualizations/chart3.png">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="v1",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 1",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path=str(fig2),
        visualization_id="v2",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 2",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path=str(fig3),
        visualization_id="v3",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 3",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": ListOfFiles(
            files=[],
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        ),
        "messages": [AIMessage(content="Manifest ready", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
        "written_sections": [md_file.read_text(encoding="utf-8")],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is True


def test_manifest_requires_exact_canonical_report_locations(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    # Non-canonical location: draft_report.md instead of final_report.md
    wrong_md = reports_dir / "draft_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    wrong_md.write_text(
        f"# Draft Report\n\n![C1](../visualizations/{fig1.name})\n![C2](../visualizations/{fig2.name})\n![C3](../visualizations/{fig3.name})\n",
        encoding="utf-8",
    )
    html_file.write_text(
        f'<h1>Final Report</h1>\n<img src="../visualizations/{fig1.name}">\n<img src="../visualizations/{fig2.name}">\n<img src="../visualizations/{fig3.name}">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="v1",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 1",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path=str(fig2),
        visualization_id="v2",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 2",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path=str(fig3),
        visualization_id="v3",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Chart 3",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": ListOfFiles(
            files=[],
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        ),
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(wrong_md),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), str(fig2), str(fig3)],
        "written_sections": [wrong_md.read_text(encoding="utf-8")],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is False


def test_valid_report_relative_images_and_aliases_still_work(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = reports_dir / "chart_rep.png"
    fig2 = viz_dir / "chart_viz.png"
    fig3 = viz_dir / "chart_viz3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    # fig1 is report-relative: chart_rep.png
    # fig2 and fig3 are ../visualizations/...
    md_file.write_text(
        "# Final Report\n\n"
        "![Rep](chart_rep.png)\n"
        "![Viz](../visualizations/chart_viz.png)\n"
        "![Viz3](../visualizations/chart_viz3.png)\n",
        encoding="utf-8",
    )
    html_file.write_text(
        "<h1>Final Report</h1>\n"
        '<img src="chart_rep.png">\n'
        '<img src="../visualizations/chart_viz.png">\n'
        '<img src="../visualizations/chart_viz3.png">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    dv1 = DataVisualization(
        path=str(fig1),
        visualization_id="rep1",
        visualization_type="bar",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Rep Chart",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv2 = DataVisualization(
        path="visualizations/chart_viz.png",
        visualization_id="viz2",
        visualization_type="line",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Viz Chart",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )
    dv3 = DataVisualization(
        path=str(fig3),
        visualization_id="viz3",
        visualization_type="scatter",
        visualization_description="d",
        visualization_style="s",
        visualization_title="Viz Chart 3",
        reply_msg_to_supervisor="ok",
        finished_this_task=True,
        expect_reply=False,
    )

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": ListOfFiles(
            files=[],
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        ),
        "messages": [AIMessage(content="Manifest ready", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=[dv1, dv2, dv3],
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(fig1), "visualizations/chart_viz.png", str(fig3)],
        "written_sections": [md_file.read_text(encoding="utf-8")],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is True


def test_finalizer_rejects_four_expected_figures_with_three_html_embeds(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    fig4 = viz_dir / "chart4.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)
    _create_dummy_png(fig4)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    # MD has all 4 embeds
    md_file.write_text(
        "# Final Report\n\n"
        "![C1](../visualizations/chart1.png)\n"
        "![C2](../visualizations/chart2.png)\n"
        "![C3](../visualizations/chart3.png)\n"
        "![C4](../visualizations/chart4.png)\n",
        encoding="utf-8",
    )
    # HTML only has 3 embeds (omits chart4.png)
    html_file.write_text(
        "<h1>Final Report</h1>\n"
        '<img src="../visualizations/chart1.png">\n'
        '<img src="../visualizations/chart2.png">\n'
        '<img src="../visualizations/chart3.png">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    dvs = [
        DataVisualization(
            path=str(p),
            visualization_id=f"v{i}",
            visualization_type="bar",
            visualization_description="d",
            visualization_style="s",
            visualization_title=f"Chart {i}",
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        )
        for i, p in enumerate([fig1, fig2, fig3, fig4], 1)
    ]

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": ListOfFiles(
            files=[],
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        ),
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=dvs,
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(p) for p in [fig1, fig2, fig3, fig4]],
        "written_sections": [md_file.read_text(encoding="utf-8")],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    # Must reject because chart4 is missing from HTML embeds (E <= H invariant)
    assert res.get("file_writer_complete") is False


def test_finalizer_rejects_missing_markdown_figure_coverage(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"
    # MD only has 2 embeds (omits chart3)
    md_file.write_text(
        "# Final Report\n\n"
        "![C1](../visualizations/chart1.png)\n"
        "![C2](../visualizations/chart2.png)\n",
        encoding="utf-8",
    )
    # HTML has all 3
    html_file.write_text(
        "<h1>Final Report</h1>\n"
        '<img src="../visualizations/chart1.png">\n'
        '<img src="../visualizations/chart2.png">\n'
        '<img src="../visualizations/chart3.png">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    dvs = [
        DataVisualization(
            path=str(p),
            visualization_id=f"v{i}",
            visualization_type="bar",
            visualization_description="d",
            visualization_style="s",
            visualization_title=f"Chart {i}",
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        )
        for i, p in enumerate([fig1, fig2, fig3], 1)
    ]

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": ListOfFiles(
            files=[],
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        ),
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=dvs,
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(p) for p in [fig1, fig2, fig3]],
        "written_sections": [md_file.read_text(encoding="utf-8")],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    # Must reject because chart3 is missing from MD embeds (E <= M invariant)
    assert res.get("file_writer_complete") is False


def test_renderer_rejects_missing_fourth_expected_figure(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    fig4_missing = viz_dir / "chart4_missing.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)
    # Do NOT create fig4_missing on disk!

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    dvs = [
        DataVisualization(
            path=str(p),
            visualization_id=f"v{i}",
            visualization_type="bar",
            visualization_description="d",
            visualization_style="s",
            visualization_title=f"Chart {i}",
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        )
        for i, p in enumerate([fig1, fig2, fig3, fig4_missing], 1)
    ]

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Missing Figure Report",
            section_num=0,
            description="Overview",
            goals=["test"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Missing Figure Report",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        "written_sections": [
            f"## Section\n\nContent\n\n![C1]({fig1})\n\n![C2]({fig2})\n\n![C3]({fig3})\n\n![C4]({fig4_missing})"
        ],
        "visualization_results": VisualizationResults(
            visualizations=dvs,
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(p) for p in [fig1, fig2, fig3, fig4_missing]],
    }

    with pytest.raises(RuntimeError, match="unresolved expected figures"):
        report_packager_node(state)


def test_renderer_injects_and_preserves_all_four_expected_figures(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    fig4 = viz_dir / "chart4.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)
    _create_dummy_png(fig4)

    report_packager_node = _compile_report_packager_node(
        generated_notebook["notebook"], tmp_path
    )

    dvs = [
        DataVisualization(
            path=str(p),
            visualization_id=f"v{i}",
            visualization_type="bar",
            visualization_description="d",
            visualization_style="s",
            visualization_title=f"Chart {i}",
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        )
        for i, p in enumerate([fig1, fig2, fig3, fig4], 1)
    ]

    state = {
        "messages": [HumanMessage(content="Generate report", name="supervisor")],
        "final_turn_msgs_list": [
            HumanMessage(content="Generate report", name="supervisor")
        ],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
        "report_outline": ReportOutline(
            name="Four Figures Report",
            section_num=0,
            description="Overview",
            goals=["test"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=100,
            title="Four Figures Report",
            sections=[],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "sections": [],
        # Written section has NO image embeds initially:
        "written_sections": [
            "## Section 1\n\nContent for section 1 without images embedded directly.\n\n"
            "This comprehensive narrative evaluates all key metrics in depth."
        ],
        "visualization_results": VisualizationResults(
            visualizations=dvs,
            reply_msg_to_supervisor="viz ready",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(p) for p in [fig1, fig2, fig3, fig4]],
    }

    res = report_packager_node(state)
    assert res.get("report_generator_complete") is True

    md_text = (reports_dir / "final_report.md").read_text(encoding="utf-8")
    html_text = (reports_dir / "final_report.html").read_text(encoding="utf-8")

    # All 4 figures must be present in both MD and HTML
    for fig in [fig1, fig2, fig3, fig4]:
        assert fig.name in md_text
        assert fig.name in html_text


def test_duplicate_and_unrelated_embeds_do_not_fill_missing_expectation(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    fig_unrelated = viz_dir / "unrelated.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)
    _create_dummy_png(fig_unrelated)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"

    # Embed fig1 TWICE and fig_unrelated ONCE. fig3 is omitted completely!
    # Count of embeds is 3, but set coverage of expected figures {fig1, fig2, fig3} fails.
    md_file.write_text(
        "# Final Report\n\n"
        "![C1](../visualizations/chart1.png)\n"
        "![C1 Dup](../visualizations/chart1.png)\n"
        "![Unrelated](../visualizations/unrelated.png)\n",
        encoding="utf-8",
    )
    html_file.write_text(
        "<h1>Final Report</h1>\n"
        '<img src="../visualizations/chart1.png">\n'
        '<img src="../visualizations/chart1.png">\n'
        '<img src="../visualizations/unrelated.png">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    dvs = [
        DataVisualization(
            path=str(p),
            visualization_id=f"v{i}",
            visualization_type="bar",
            visualization_description="d",
            visualization_style="s",
            visualization_title=f"Chart {i}",
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        )
        for i, p in enumerate([fig1, fig2, fig3], 1)
    ]

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": ListOfFiles(
            files=[],
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        ),
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=dvs,
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(p) for p in [fig1, fig2, fig3]],
        "written_sections": [md_file.read_text(encoding="utf-8")],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    assert res.get("file_writer_complete") is False


def test_fenced_code_image_example_does_not_count_as_embed(
    generated_notebook, tmp_path, monkeypatch
):
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    reports_dir = tmp_path / "reports"
    viz_dir = tmp_path / "visualizations"
    reports_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig1 = viz_dir / "chart1.png"
    fig2 = viz_dir / "chart2.png"
    fig3 = viz_dir / "chart3.png"
    _create_dummy_png(fig1)
    _create_dummy_png(fig2)
    _create_dummy_png(fig3)

    md_file = reports_dir / "final_report.md"
    html_file = reports_dir / "final_report.html"
    pdf_file = reports_dir / "final_report.pdf"

    # In MD, fig3 is ONLY inside a fenced code block:
    md_file.write_text(
        "# Final Report\n\n"
        "![C1](../visualizations/chart1.png)\n"
        "![C2](../visualizations/chart2.png)\n\n"
        "Example syntax:\n"
        "```markdown\n"
        "![C3](../visualizations/chart3.png)\n"
        "```\n",
        encoding="utf-8",
    )
    # In HTML, all 3 are in img tags:
    html_file.write_text(
        "<h1>Final Report</h1>\n"
        '<img src="../visualizations/chart1.png">\n'
        '<img src="../visualizations/chart2.png">\n'
        '<img src="../visualizations/chart3.png">\n',
        encoding="utf-8",
    )
    pdf_file.write_text("%PDF-1.4 dummy", encoding="utf-8")

    dvs = [
        DataVisualization(
            path=str(p),
            visualization_id=f"v{i}",
            visualization_type="bar",
            visualization_description="d",
            visualization_style="s",
            visualization_title=f"Chart {i}",
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        )
        for i, p in enumerate([fig1, fig2, fig3], 1)
    ]

    mock_finalizer = MagicMock()
    mock_finalizer.with_config.return_value = mock_finalizer
    mock_finalizer.invoke.return_value = {
        "structured_response": ListOfFiles(
            files=[],
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        ),
        "messages": [AIMessage(content="done", name="file_writer")],
    }

    file_writer_node = _compile_file_writer_node(
        generated_notebook["notebook"],
        tmp_path,
        mock_finalizer=mock_finalizer,
    )

    state = {
        "report_generator_complete": True,
        "report_results": ReportResults(
            markdown_report_path=str(md_file),
            html_report_path=str(html_file),
            pdf_report_path=str(pdf_file),
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "visualization_results": VisualizationResults(
            visualizations=dvs,
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        ),
        "viz_paths": [str(p) for p in [fig1, fig2, fig3]],
        "written_sections": [md_file.read_text(encoding="utf-8")],
        "messages": [HumanMessage(content="run")],
        "_config": {"configurable": {"runtime": DummyRuntime(tmp_path)}},
    }

    res = file_writer_node(state)
    # Must reject because fenced code block image is stripped and does not count toward M!
    assert res.get("file_writer_complete") is False
