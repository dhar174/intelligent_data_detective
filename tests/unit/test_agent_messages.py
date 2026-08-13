"""
Unit tests for agent message models and routing types in idd_core.
Covers: AgentMembers subclasses, AgentId, AgentOrSupervisor,
CLASS_TO_AGENT, SendAgentMessage, MessagesToAgentsList, NextAgentMetadata.
"""
import pytest
from typing import get_args
from pydantic import ValidationError

from idd_core import (
    AgentId,
    AgentMembers,
    AgentOrSupervisor,
    Analyst,
    CLASS_TO_AGENT,
    DataCleaner,
    FileWriter,
    InitialAnalysis,
    MessagesToAgentsList,
    NextAgentMetadata,
    ReportGenerator,
    ReportOrchestrator,
    ReportSection,
    SendAgentMessage,
    SuperVisor,
    Visualization,
)

BASE = dict(reply_msg_to_supervisor="test", finished_this_task=True, expect_reply=False)

# ---------------------------------------------------------------------------
# AgentMembers: all 9 literal agent_type values
# ---------------------------------------------------------------------------

class TestAgentMembersTypeCoverage:
    @pytest.mark.parametrize("agent_type", [
        "initial_analysis", "data_cleaner", "analyst", "file_writer",
        "visualization", "report_orchestrator", "report_section_worker",
        "report_packager", "supervisor",
    ])
    def test_all_nine_agent_type_values_accepted(self, agent_type):
        m = AgentMembers(agent_type=agent_type)
        assert m.agent_type == agent_type

    def test_unknown_agent_type_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AgentMembers(agent_type="unknown_worker")

    def test_initial_analysis_subclass_has_correct_type(self):
        m = InitialAnalysis()
        assert m.agent_type == "initial_analysis"

    def test_report_orchestrator_subclass_has_correct_type(self):
        m = ReportOrchestrator()
        assert m.agent_type == "report_orchestrator"

    def test_report_section_subclass_has_correct_type(self):
        m = ReportSection()
        assert m.agent_type == "report_section_worker"

    def test_supervisor_subclass_has_correct_type(self):
        m = SuperVisor()
        assert m.agent_type == "supervisor"

    def test_data_cleaner_subclass(self):
        m = DataCleaner()
        assert m.agent_type == "data_cleaner"

    def test_analyst_subclass(self):
        m = Analyst()
        assert m.agent_type == "analyst"

    def test_file_writer_subclass(self):
        m = FileWriter()
        assert m.agent_type == "file_writer"

    def test_visualization_subclass(self):
        m = Visualization()
        assert m.agent_type == "visualization"

    def test_report_generator_subclass_is_report_packager(self):
        m = ReportGenerator()
        assert m.agent_type == "report_packager"


# ---------------------------------------------------------------------------
# AgentId and AgentOrSupervisor distinction
# ---------------------------------------------------------------------------

class TestAgentIdAndAgentOrSupervisor:
    def test_agent_id_has_fourteen_values(self):
        values = get_args(AgentId)
        assert len(values) == 14

    def test_agent_id_contains_expected_values(self):
        values = set(get_args(AgentId))
        expected_subset = {
            "initial_analysis", "data_cleaner", "analyst",
            "viz_worker", "viz_join", "viz_evaluator", "visualization",
            "report_orchestrator", "report_section_worker", "report_join",
            "report_packager", "file_writer", "END", "FINISH",
        }
        assert values == expected_subset

    def test_supervisor_not_in_agent_id(self):
        """supervisor is NOT in AgentId — it's only in AgentOrSupervisor."""
        valid_agent_ids = set(get_args(AgentId))
        assert "supervisor" not in valid_agent_ids

    def test_send_agent_message_recipient_supervisor_valid(self):
        """supervisor IS valid in AgentOrSupervisor (Union of AgentId and Literal['supervisor'])."""
        msg = SendAgentMessage(
            recipient="supervisor",
            message="ping",
            delivery_status=True,
            agent_obj_needs_recreated_bool=False,
            is_message_critical=False,
            immediate_emergency_reroute_to_recipient=False,
            **BASE,
        )
        assert msg.recipient == "supervisor"

    def test_send_agent_message_recipient_end_valid(self):
        msg = SendAgentMessage(
            recipient="END",
            message="done",
            delivery_status=True,
            agent_obj_needs_recreated_bool=False,
            is_message_critical=False,
            immediate_emergency_reroute_to_recipient=False,
            **BASE,
        )
        assert msg.recipient == "END"

    def test_send_agent_message_unknown_recipient_raises(self):
        with pytest.raises(ValidationError):
            SendAgentMessage(
                recipient="unknown_agent",
                message="hello",
                delivery_status=True,
                agent_obj_needs_recreated_bool=False,
                is_message_critical=False,
                immediate_emergency_reroute_to_recipient=False,
                **BASE,
            )

    def test_finish_is_valid_agent_id(self):
        valid_ids = set(get_args(AgentId))
        assert "FINISH" in valid_ids


# ---------------------------------------------------------------------------
# CLASS_TO_AGENT consistency
# ---------------------------------------------------------------------------

class TestClassToAgentConsistency:
    def test_all_class_to_agent_keys_are_importable(self):
        """Every key in CLASS_TO_AGENT must be a known importable class."""
        from idd_core import (
            AnalysisInsights, CleaningMetadata, DataVisualization,
            FileResult, InitialDescription, ReportOutline, ReportResults,
            Section, SectionOutline, VizFeedback, VisualizationResults,
        )
        importable = {
            InitialDescription, CleaningMetadata, AnalysisInsights,
            VisualizationResults, DataVisualization, VizFeedback,
            SectionOutline, Section, ReportOutline, ReportResults, FileResult,
        }
        for cls in CLASS_TO_AGENT:
            assert cls in importable, f"{cls} is in CLASS_TO_AGENT but not in importable set"

    def test_all_class_to_agent_values_are_valid_agent_ids(self):
        valid_ids = set(get_args(AgentId))
        for cls, agent_id in CLASS_TO_AGENT.items():
            assert agent_id in valid_ids, f"{cls.__name__} maps to '{agent_id}' which is not a valid AgentId"

    def test_all_eleven_expected_model_classes_present(self):
        from idd_core import (
            AnalysisInsights, CleaningMetadata, DataVisualization,
            FileResult, InitialDescription, ReportOutline, ReportResults,
            Section, SectionOutline, VizFeedback, VisualizationResults,
        )
        expected = {
            InitialDescription, CleaningMetadata, AnalysisInsights,
            VisualizationResults, DataVisualization, VizFeedback,
            SectionOutline, Section, ReportOutline, ReportResults, FileResult,
        }
        assert set(CLASS_TO_AGENT.keys()) == expected


# ---------------------------------------------------------------------------
# SendAgentMessage
# ---------------------------------------------------------------------------

class TestSendAgentMessage:
    def test_emergency_path_all_bool_flags_true(self):
        msg = SendAgentMessage(
            recipient="analyst",
            message="emergency",
            delivery_status=True,
            agent_obj_needs_recreated_bool=True,
            is_message_critical=True,
            immediate_emergency_reroute_to_recipient=True,
            **BASE,
        )
        assert msg.immediate_emergency_reroute_to_recipient is True
        assert msg.is_message_critical is True

    def test_normal_path_all_bool_flags_false(self):
        msg = SendAgentMessage(
            recipient="data_cleaner",
            message="routine update",
            delivery_status=False,
            agent_obj_needs_recreated_bool=False,
            is_message_critical=False,
            immediate_emergency_reroute_to_recipient=False,
            **BASE,
        )
        assert msg.delivery_status is False

    def test_recipient_supervisor(self):
        msg = SendAgentMessage(
            recipient="supervisor",
            message="task complete",
            delivery_status=True,
            agent_obj_needs_recreated_bool=False,
            is_message_critical=False,
            immediate_emergency_reroute_to_recipient=False,
            **BASE,
        )
        assert msg.recipient == "supervisor"

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            SendAgentMessage(
                recipient="analyst",
                message="msg",
                delivery_status=True,
                agent_obj_needs_recreated_bool=False,
                is_message_critical=False,
                immediate_emergency_reroute_to_recipient=False,
                extra_field="bad",
                **BASE,
            )


# ---------------------------------------------------------------------------
# MessagesToAgentsList
# ---------------------------------------------------------------------------

class TestMessagesToAgentsList:
    def test_empty_messages_valid(self):
        m = MessagesToAgentsList(messages_to_agents=[], **BASE)
        assert m.messages_to_agents == []

    def test_multiple_messages_valid(self):
        def _msg(recipient):
            return SendAgentMessage(
                recipient=recipient,
                message="hello",
                delivery_status=True,
                agent_obj_needs_recreated_bool=False,
                is_message_critical=False,
                immediate_emergency_reroute_to_recipient=False,
                **BASE,
            )

        m = MessagesToAgentsList(
            messages_to_agents=[_msg("analyst"), _msg("data_cleaner"), _msg("supervisor")],
            **BASE,
        )
        assert len(m.messages_to_agents) == 3


# ---------------------------------------------------------------------------
# NextAgentMetadata
# ---------------------------------------------------------------------------

class TestNextAgentMetadata:
    def test_all_fields_none_is_valid(self):
        meta = NextAgentMetadata(
            df_id=None,
            file_type=None,
            file_name=None,
            section_name=None,
            viz_spec=None,
            notes=None,
            file_content=None,
            **BASE,
        )
        assert meta.df_id is None
        assert meta.viz_spec is None

    def test_viz_spec_round_trip(self):
        from idd_core import VizSpec
        spec = VizSpec(
            title="test", viz_type="bar", df_id="df-1",
            viz_instructions="group by col", viz_id="v-1",
            columns=["a", "b"], x="a", y="b", hue=None,
            bins=None, agg="sum", query=None, description=None,
            limit=None, style=None,
            **BASE,
        )
        meta = NextAgentMetadata(
            df_id="df-1",
            file_type=None,
            file_name=None,
            section_name=None,
            viz_spec=spec,
            notes="some notes",
            file_content=None,
            **BASE,
        )
        assert meta.viz_spec.viz_type == "bar"
        assert meta.notes == "some notes"
