"""
Unit tests for BaseNoExtrasModel and its field contracts.

Tests:
- extra="forbid" rejects unknown fields
- All 3 required fields enforced
- Inheritance works (subclasses inherit forbid)
- JSON schema has additionalProperties: False
"""
import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def core():
    try:
        import idd_core
        return idd_core
    except ImportError:
        pytest.skip("idd_core.py not available")


class TestBaseNoExtrasModel:
    def test_extra_fields_forbidden(self, core):
        """Unknown fields must raise ValidationError."""
        with pytest.raises(ValidationError, match="extra"):
            core.BaseNoExtrasModel(
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
                unknown_field="should fail",
            )

    def test_all_three_required_fields(self, core):
        """All 3 required fields must be present."""
        # Valid construction
        obj = core.BaseNoExtrasModel(
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        )
        assert obj.finished_this_task is True
        assert obj.expect_reply is False
        assert obj.reply_msg_to_supervisor == "done"

    def test_missing_reply_msg_raises(self, core):
        with pytest.raises(ValidationError):
            core.BaseNoExtrasModel(finished_this_task=True, expect_reply=False)

    def test_missing_finished_this_task_raises(self, core):
        with pytest.raises(ValidationError):
            core.BaseNoExtrasModel(reply_msg_to_supervisor="ok", expect_reply=False)

    def test_missing_expect_reply_raises(self, core):
        with pytest.raises(ValidationError):
            core.BaseNoExtrasModel(
                reply_msg_to_supervisor="ok", finished_this_task=True
            )

    def test_json_schema_no_additional_properties(self, core):
        schema = core.BaseNoExtrasModel.model_json_schema()
        assert schema.get("additionalProperties") is False

    def test_subclass_inherits_forbid(self, core):
        """CleaningMetadata (subclass) should also forbid extra fields."""
        with pytest.raises(ValidationError):
            core.CleaningMetadata(
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
                steps_taken=["step1"],
                data_description_after_cleaning="clean",
                extra_not_allowed="oops",
            )

    def test_valid_cleaning_metadata(self, core):
        obj = core.CleaningMetadata(
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
            steps_taken=["remove_nulls", "drop_dupes"],
            data_description_after_cleaning="Dataset cleaned.",
        )
        assert obj.steps_taken == ["remove_nulls", "drop_dupes"]

    def test_analysis_config_valid(self, core):
        obj = core.AnalysisConfig(
            reply_msg_to_supervisor="ok",
            finished_this_task=False,
            expect_reply=True,
            default_visualization_style="seaborn-v0_8-whitegrid",
            report_author="Test Author",
            datetime_format_preference="%Y-%m-%d %H:%M:%S",
            large_dataframe_preview_rows=5,
        )
        assert obj.report_author == "Test Author"
        assert obj.large_dataframe_preview_rows == 5


class TestAgentMembers:
    def test_valid_agent_type(self, core):
        obj = core.InitialAnalysis()
        assert obj.agent_type == "initial_analysis"

    def test_invalid_agent_type_raises(self, core):
        with pytest.raises(ValidationError):
            core.AgentMembers(description="x", agent_type="invalid_agent")

    def test_all_concrete_subclasses_valid(self, core):
        agents = [
            core.InitialAnalysis(),
            core.DataCleaner(),
            core.Analyst(),
            core.FileWriter(),
            core.Visualization(),
            core.ReportGenerator(),
            core.SuperVisor(),
            core.ReportOrchestrator(),
            core.ReportSection(),
        ]
        assert len(agents) == 9


class TestImagePayload:
    def test_valid_png_bytes(self, core):
        import base64
        raw = b"fakeimage"
        b64 = base64.b64encode(raw).decode()
        obj = core.ImagePayload(
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
            mime="image/png",
            payload=b64,
        )
        assert obj.payload == raw

    def test_size_limit_enforced(self, core):
        import base64
        oversized = b"x" * (2 * 1024 * 1024 + 1)
        b64 = base64.b64encode(oversized).decode()
        with pytest.raises(ValidationError, match="too large"):
            core.ImagePayload(
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
                mime="image/jpeg",
                payload=b64,
            )

    def test_invalid_base64_raises(self, core):
        with pytest.raises(ValidationError, match="Base-64"):
            core.ImagePayload(
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
                mime="image/png",
                payload="not!valid!base64!!!",
            )

    def test_raw_bytes_accepted(self, core):
        """Raw bytes (not b64) should also be accepted."""
        raw = b"smallimage"
        obj = core.ImagePayload(
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
            mime="image/png",
            payload=raw,
        )
        assert obj.payload == raw
