"""
Unit tests for pipeline-stage Pydantic models in idd_core.
Covers: CleaningMetadata, AnalysisInsights, VizSpec, VizFeedback,
DataVisualization, VisualizationResults, ReportResults, FileResult,
ReportOutline, and agent_list_default_generator.
"""
import pytest
from pydantic import ValidationError

from idd_core import (
    AgentMembers,
    AnalysisInsights,
    CleaningMetadata,
    DataVisualization,
    FileResult,
    ReportOutline,
    ReportResults,
    VisualizationResults,
    VizFeedback,
    VizSpec,
    agent_list_default_generator,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BASE = dict(reply_msg_to_supervisor="test", finished_this_task=True, expect_reply=False)


def _vizspec(**overrides):
    """Return a valid minimal VizSpec (all fields set to None)."""
    defaults = dict(
        title=None, viz_type=None, df_id=None, viz_instructions=None,
        viz_id=None, columns=None, x=None, y=None, hue=None, bins=None,
        agg=None, query=None, description=None, limit=None, style=None,
        **BASE,
    )
    defaults.update(overrides)
    return VizSpec(**defaults)


def _dataviz(**overrides):
    defaults = dict(
        path="/report/viz.png",
        visualization_id="viz-1",
        visualization_type="bar",
        visualization_description="A bar chart",
        visualization_style="seaborn-v0_8-whitegrid",
        visualization_title="My Chart",
        **BASE,
    )
    defaults.update(overrides)
    return DataVisualization(**defaults)


# ---------------------------------------------------------------------------
# CleaningMetadata
# ---------------------------------------------------------------------------

class TestCleaningMetadata:
    def test_empty_steps_taken_is_valid(self):
        m = CleaningMetadata(
            steps_taken=[],
            data_description_after_cleaning="No cleaning needed.",
            **BASE,
        )
        assert m.steps_taken == []

    def test_multiple_steps_and_long_description(self):
        steps = ["remove_duplicates", "fill_missing_median", "normalize_columns"]
        m = CleaningMetadata(
            steps_taken=steps,
            data_description_after_cleaning="x" * 500,
            **BASE,
        )
        assert len(m.steps_taken) == 3

    def test_extra_field_raises_validation_error(self):
        with pytest.raises(ValidationError):
            CleaningMetadata(
                steps_taken=[],
                data_description_after_cleaning="ok",
                unexpected_field="bad",
                **BASE,
            )

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            CleaningMetadata(
                steps_taken=["step1"],
                **BASE,
            )


# ---------------------------------------------------------------------------
# AnalysisInsights
# ---------------------------------------------------------------------------

class TestAnalysisInsights:
    def test_empty_recommended_visualizations_is_valid(self):
        a = AnalysisInsights(
            summary="ok",
            correlation_insights="none",
            anomaly_insights="none",
            recommended_visualizations=[],
            recommended_next_steps=[],
            **BASE,
        )
        assert a.recommended_visualizations == []

    def test_viz_type_auto_accepted(self):
        spec = _vizspec(viz_type="auto")
        a = AnalysisInsights(
            summary="s",
            correlation_insights="c",
            anomaly_insights="a",
            recommended_visualizations=[spec],
            recommended_next_steps=["step"],
            **BASE,
        )
        assert a.recommended_visualizations[0].viz_type == "auto"

    def test_unknown_viz_type_in_vizspec_raises(self):
        with pytest.raises(ValidationError):
            _vizspec(viz_type="unknown_type")

    def test_missing_summary_raises(self):
        with pytest.raises(ValidationError):
            AnalysisInsights(
                correlation_insights="c",
                anomaly_insights="a",
                recommended_visualizations=[],
                recommended_next_steps=[],
                **BASE,
            )


# ---------------------------------------------------------------------------
# VizSpec
# ---------------------------------------------------------------------------

class TestVizSpec:
    def test_all_none_is_valid(self):
        spec = _vizspec()
        assert spec.title is None
        assert spec.viz_type is None

    @pytest.mark.parametrize("vtype", ["histogram", "scatter", "bar", "line", "box", "auto"])
    def test_valid_viz_types(self, vtype):
        spec = _vizspec(viz_type=vtype)
        assert spec.viz_type == vtype

    def test_unknown_viz_type_raises(self):
        with pytest.raises(ValidationError):
            _vizspec(viz_type="pie")

    def test_with_all_fields_populated(self):
        spec = _vizspec(
            title="Sales Chart",
            viz_type="bar",
            df_id="df-001",
            viz_instructions="group by month",
            viz_id="v-100",
            columns=["month", "revenue"],
            x="month",
            y="revenue",
            hue="region",
            bins=20,
            agg="sum",
            query="revenue > 0",
            description="Monthly revenue",
            limit=100,
            style="seaborn-v0_8-whitegrid",
        )
        assert spec.viz_type == "bar"
        assert spec.x == "month"

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            _vizspec(unknown_extra="bad")


# ---------------------------------------------------------------------------
# VizFeedback
# ---------------------------------------------------------------------------

class TestVizFeedback:
    def test_grade_revise_with_redo_list(self):
        v = VizFeedback(
            grade="revise",
            feedback="Needs bigger font",
            redo_list=["viz-1", "viz-2"],
            **BASE,
        )
        assert v.grade == "revise"
        assert len(v.redo_list) == 2

    def test_grade_acceptable_with_empty_redo_list(self):
        v = VizFeedback(
            grade="acceptable",
            feedback="Looks great",
            redo_list=[],
            **BASE,
        )
        assert v.grade == "acceptable"
        assert v.redo_list == []

    def test_grade_acceptable_with_nonempty_redo_list_also_valid(self):
        """No validator enforces redo_list is empty when grade='acceptable'."""
        v = VizFeedback(
            grade="acceptable",
            feedback="ok",
            redo_list=["viz1"],
            **BASE,
        )
        assert v.redo_list == ["viz1"]  # accepted without error

    def test_unknown_grade_raises(self):
        with pytest.raises(ValidationError):
            VizFeedback(
                grade="perfect",
                feedback="",
                redo_list=[],
                **BASE,
            )


# ---------------------------------------------------------------------------
# DataVisualization and VisualizationResults
# ---------------------------------------------------------------------------

class TestDataVisualization:
    def test_single_dataviz_valid(self):
        dv = _dataviz()
        assert dv.visualization_id == "viz-1"

    def test_visualization_results_empty_list_valid(self):
        vr = VisualizationResults(visualizations=[], **BASE)
        assert vr.visualizations == []

    def test_visualization_results_with_three_items(self):
        vizzes = [_dataviz(visualization_id=f"viz-{i}") for i in range(3)]
        vr = VisualizationResults(visualizations=vizzes, **BASE)
        assert len(vr.visualizations) == 3

    def test_extra_field_on_dataviz_raises(self):
        with pytest.raises(ValidationError):
            _dataviz(extra_nonsense="bad")


# ---------------------------------------------------------------------------
# ReportOutline
# ---------------------------------------------------------------------------

class TestReportOutline:
    def _minimal_outline(self, **overrides):
        defaults = dict(
            name="Section 1",
            section_num=1,
            description="Overview section",
            goals=["Understand the data"],
            data_signals_needed={"revenue": "numeric"},
            data_signals_available=["revenue", "date"],
            expected_figures=[],
            word_target=300,
            title="Annual Report",
            sections=[],
            **BASE,
        )
        defaults.update(overrides)
        return ReportOutline(**defaults)

    def test_missing_title_raises(self):
        with pytest.raises((ValidationError, TypeError)):
            ReportOutline(
                name="s",
                section_num=1,
                description="d",
                goals=[],
                data_signals_needed={},
                data_signals_available=[],
                expected_figures=[],
                word_target=300,
                sections=[],
                **BASE,
            )

    def test_empty_sections_is_valid(self):
        ro = self._minimal_outline()
        assert ro.sections == []

    def test_round_trip_model_dump_and_validate(self):
        ro = self._minimal_outline()
        dumped = ro.model_dump()
        restored = ReportOutline.model_validate(dumped)
        assert restored.title == ro.title
        assert restored.goals == ro.goals

    def test_with_nested_section_outline(self):
        from idd_core import SectionOutline
        so = SectionOutline(
            name="Methods",
            section_num=2,
            description="Methods section",
            goals=["describe methods"],
            data_signals_needed={"col": "str"},
            data_signals_available=["col"],
            expected_figures=[],
            word_target=200,
            **BASE,
        )
        ro = self._minimal_outline(sections=[so])
        assert len(ro.sections) == 1


# ---------------------------------------------------------------------------
# ReportResults
# ---------------------------------------------------------------------------

class TestReportResults:
    def test_all_paths_required(self):
        rr = ReportResults(
            pdf_report_path="/out/report.pdf",
            html_report_path="/out/report.html",
            markdown_report_path="/out/report.md",
            **BASE,
        )
        assert rr.pdf_report_path == "/out/report.pdf"

    def test_missing_pdf_path_raises(self):
        with pytest.raises(ValidationError):
            ReportResults(
                html_report_path="/out/report.html",
                markdown_report_path="/out/report.md",
                **BASE,
            )

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            ReportResults(
                pdf_report_path="/out/report.pdf",
                html_report_path="/out/report.html",
                markdown_report_path="/out/report.md",
                extra="oops",
                **BASE,
            )


# ---------------------------------------------------------------------------
# FileResult
# ---------------------------------------------------------------------------

class TestFileResult:
    def _valid_fileresult(**kwargs):
        defaults = dict(
            write_success=True,
            file_path="/data/output.csv",
            file_type="csv",
            file_name="output.csv",
            file_description="Final output",
            is_final_report=False,
            category_tag="data",
            **BASE,
        )
        defaults.update(kwargs)
        return FileResult(**defaults)

    def test_category_tag_is_plain_str_any_string_accepted(self):
        """category_tag is declared as plain str, NOT Literal — any value accepted."""
        fr = FileResult(
            write_success=True,
            file_path="/data/output.csv",
            file_type="csv",
            file_name="output.csv",
            file_description="desc",
            is_final_report=False,
            category_tag="my_custom_tag_not_in_any_enum",
            **BASE,
        )
        assert fr.category_tag == "my_custom_tag_not_in_any_enum"

    def test_write_success_false_with_path_valid(self):
        fr = FileResult(
            write_success=False,
            file_path="/data/failed.csv",
            file_type="csv",
            file_name="failed.csv",
            file_description="write failed",
            is_final_report=False,
            category_tag="data",
            **BASE,
        )
        assert fr.write_success is False

    def test_missing_is_final_report_raises(self):
        with pytest.raises(ValidationError):
            FileResult(
                write_success=True,
                file_path="/data/report.pdf",
                file_type="pdf",
                file_name="report.pdf",
                file_description="final report",
                category_tag="report",
                **BASE,
            )


# ---------------------------------------------------------------------------
# agent_list_default_generator
# ---------------------------------------------------------------------------

class TestAgentListDefaultGenerator:
    def test_returns_list(self):
        result = agent_list_default_generator()
        assert isinstance(result, list)

    def test_all_items_are_agent_members(self):
        result = agent_list_default_generator()
        for item in result:
            assert isinstance(item, AgentMembers)

    def test_returns_exactly_seven_items(self):
        result = agent_list_default_generator()
        assert len(result) == 7

    def test_agent_types_in_result(self):
        result = agent_list_default_generator()
        types = {item.agent_type for item in result}
        expected = {
            "initial_analysis", "data_cleaner", "analyst",
            "file_writer", "visualization", "report_packager", "supervisor",
        }
        assert types == expected
