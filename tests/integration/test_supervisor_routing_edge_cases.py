"""
Integration tests for supervisor routing edge cases in idd_core.
Covers CLASS_TO_AGENT coverage, AgentId/AgentOrSupervisor distinctions,
SendAgentMessage emergency routing, and MessagesToAgentsList validation.
"""
import pytest
from pydantic import ValidationError

pytestmark = [pytest.mark.integration]

from idd_core import (
    AgentId,
    AgentMembers,
    AgentOrSupervisor,
    CLASS_TO_AGENT,
    DataVisualization,
    FileResult,
    MessagesToAgentsList,
    NextAgentMetadata,
    Plan,
    PlanStep,
    SendAgentMessage,
    VisualizationResults,
    VizFeedback,
    VizSpec,
    AnalysisInsights,
    CleaningMetadata,
    ReportOutline,
    SectionOutline,
    Section,
    ReportResults,
    InitialDescription,
)

BASE = dict(reply_msg_to_supervisor="test", finished_this_task=True, expect_reply=False)


def make_viz_spec():
    return VizSpec(
        title=None, viz_type=None, df_id=None, viz_instructions=None, viz_id=None,
        columns=None, x=None, y=None, hue=None, bins=None, agg=None, query=None,
        description=None, limit=None, style=None,
        **BASE,
    )


def make_send_msg(recipient, text="hi"):
    return SendAgentMessage(
        recipient=recipient,
        message=text,
        delivery_status=False,
        agent_obj_needs_recreated_bool=False,
        is_message_critical=False,
        immediate_emergency_reroute_to_recipient=False,
        **BASE,
    )


# ---------------------------------------------------------------------------
# CLASS_TO_AGENT: all 11 entries
# ---------------------------------------------------------------------------

class TestClassToAgentCoverage:
    def test_initial_description_maps_to_initial_analysis(self):
        assert CLASS_TO_AGENT[InitialDescription] == "initial_analysis"

    def test_cleaning_metadata_maps_to_data_cleaner(self):
        assert CLASS_TO_AGENT[CleaningMetadata] == "data_cleaner"

    def test_analysis_insights_maps_to_analyst(self):
        assert CLASS_TO_AGENT[AnalysisInsights] == "analyst"

    def test_visualization_results_maps_to_viz_join(self):
        assert CLASS_TO_AGENT[VisualizationResults] == "viz_join"

    def test_data_visualization_maps_to_viz_worker(self):
        assert CLASS_TO_AGENT[DataVisualization] == "viz_worker"

    def test_viz_feedback_maps_to_viz_evaluator(self):
        assert CLASS_TO_AGENT[VizFeedback] == "viz_evaluator"

    def test_report_outline_maps_to_report_orchestrator(self):
        assert CLASS_TO_AGENT[ReportOutline] == "report_orchestrator"

    def test_section_outline_maps_to_report_orchestrator(self):
        assert CLASS_TO_AGENT[SectionOutline] == "report_orchestrator"

    def test_section_maps_to_report_section_worker(self):
        assert CLASS_TO_AGENT[Section] == "report_section_worker"

    def test_report_results_maps_to_report_packager(self):
        assert CLASS_TO_AGENT[ReportResults] == "report_packager"

    def test_file_result_maps_to_file_writer(self):
        assert CLASS_TO_AGENT[FileResult] == "file_writer"

    def test_class_to_agent_has_exactly_11_entries(self):
        assert len(CLASS_TO_AGENT) == 11

    def test_all_values_are_valid_agent_id_strings(self):
        valid_agent_ids = set(AgentId.__args__)
        for cls, agent_str in CLASS_TO_AGENT.items():
            assert agent_str in valid_agent_ids, f"{cls.__name__} -> {agent_str!r} not a valid AgentId"


# ---------------------------------------------------------------------------
# AgentId / AgentOrSupervisor distinctions
# ---------------------------------------------------------------------------

class TestAgentIdAndOrSupervisor:
    def test_supervisor_not_in_agent_id(self):
        valid_agent_ids = set(AgentId.__args__)
        assert "supervisor" not in valid_agent_ids

    def test_end_is_valid_agent_id(self):
        valid_agent_ids = set(AgentId.__args__)
        assert "END" in valid_agent_ids

    def test_finish_is_valid_agent_id(self):
        valid_agent_ids = set(AgentId.__args__)
        assert "FINISH" in valid_agent_ids

    def test_send_agent_message_with_supervisor_as_recipient(self):
        msg = make_send_msg("supervisor", text="emergency!")
        assert msg.recipient == "supervisor"

    def test_send_agent_message_invalid_recipient_raises(self):
        with pytest.raises(ValidationError):
            SendAgentMessage(
                recipient="nonexistent_agent",
                message="test",
                delivery_status=False,
                agent_obj_needs_recreated_bool=False,
                is_message_critical=False,
                immediate_emergency_reroute_to_recipient=False,
                **BASE,
            )

    def test_agent_or_supervisor_accepts_all_14_agent_id_values(self):
        agent_ids = AgentId.__args__
        for aid in agent_ids:
            msg = make_send_msg(aid, text="routing test")
            assert msg.recipient == aid

    def test_agent_or_supervisor_accepts_supervisor(self):
        msg = make_send_msg("supervisor")
        assert msg.recipient == "supervisor"


# ---------------------------------------------------------------------------
# SendAgentMessage routing flags
# ---------------------------------------------------------------------------

class TestSendAgentMessageRoutingFlags:
    def test_emergency_reroute_flag_true(self):
        msg = SendAgentMessage(
            recipient="supervisor",
            message="critical failure",
            delivery_status=False,
            agent_obj_needs_recreated_bool=False,
            is_message_critical=True,
            immediate_emergency_reroute_to_recipient=True,
            **BASE,
        )
        assert msg.immediate_emergency_reroute_to_recipient is True
        assert msg.is_message_critical is True

    def test_emergency_reroute_flag_false(self):
        msg = make_send_msg("analyst")
        assert msg.immediate_emergency_reroute_to_recipient is False
        assert msg.is_message_critical is False

    def test_critical_message_without_emergency_reroute(self):
        msg = SendAgentMessage(
            recipient="data_cleaner",
            message="important note",
            delivery_status=False,
            agent_obj_needs_recreated_bool=False,
            is_message_critical=True,
            immediate_emergency_reroute_to_recipient=False,
            **BASE,
        )
        assert msg.is_message_critical is True
        assert msg.immediate_emergency_reroute_to_recipient is False


# ---------------------------------------------------------------------------
# MessagesToAgentsList
# ---------------------------------------------------------------------------

class TestMessagesToAgentsList:
    def test_empty_messages_list(self):
        obj = MessagesToAgentsList(messages_to_agents=[], **BASE)
        assert obj.messages_to_agents == []

    def test_single_message(self):
        msg = make_send_msg("analyst")
        obj = MessagesToAgentsList(messages_to_agents=[msg], **BASE)
        assert len(obj.messages_to_agents) == 1
        assert obj.messages_to_agents[0].recipient == "analyst"

    def test_multiple_different_recipients(self):
        messages = [
            make_send_msg("data_cleaner"),
            make_send_msg("analyst"),
            make_send_msg("visualization"),
        ]
        obj = MessagesToAgentsList(messages_to_agents=messages, **BASE)
        recipients = [m.recipient for m in obj.messages_to_agents]
        assert set(recipients) == {"data_cleaner", "analyst", "visualization"}

    def test_duplicate_recipients_allowed(self):
        messages = [make_send_msg("analyst"), make_send_msg("analyst")]
        obj = MessagesToAgentsList(messages_to_agents=messages, **BASE)
        assert len(obj.messages_to_agents) == 2

    def test_supervisor_as_recipient(self):
        msg = make_send_msg("supervisor", text="report back")
        obj = MessagesToAgentsList(messages_to_agents=[msg], **BASE)
        assert obj.messages_to_agents[0].recipient == "supervisor"

    def test_all_non_supervisor_agent_types_as_recipients(self):
        agent_types = [
            "initial_analysis", "data_cleaner", "analyst",
            "visualization", "report_orchestrator", "file_writer",
        ]
        messages = [make_send_msg(at) for at in agent_types]
        obj = MessagesToAgentsList(messages_to_agents=messages, **BASE)
        assert len(obj.messages_to_agents) == len(agent_types)


# ---------------------------------------------------------------------------
# NextAgentMetadata routing
# ---------------------------------------------------------------------------

class TestNextAgentMetadata:
    def test_all_none_fields_valid(self):
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
        assert meta.notes is None

    def test_df_id_and_file_type_fields(self):
        meta = NextAgentMetadata(
            df_id="df_001",
            file_type="csv",
            file_name=None,
            section_name=None,
            viz_spec=None,
            notes=None,
            file_content=None,
            **BASE,
        )
        assert meta.df_id == "df_001"
        assert meta.file_type == "csv"

    def test_section_name_field(self):
        meta = NextAgentMetadata(
            df_id=None,
            file_type=None,
            file_name=None,
            section_name="Introduction",
            viz_spec=None,
            notes=None,
            file_content=None,
            **BASE,
        )
        assert meta.section_name == "Introduction"

    def test_notes_field(self):
        meta = NextAgentMetadata(
            df_id=None,
            file_type=None,
            file_name=None,
            section_name=None,
            viz_spec=None,
            notes="additional context for next agent",
            file_content=None,
            **BASE,
        )
        assert meta.notes == "additional context for next agent"

    def test_round_trip_model_dump_validate(self):
        meta = NextAgentMetadata(
            df_id="df_42",
            file_type="parquet",
            file_name="output.parquet",
            section_name="results",
            viz_spec=None,
            notes="ready for viz",
            file_content=None,
            **BASE,
        )
        dumped = meta.model_dump()
        restored = NextAgentMetadata.model_validate(dumped)
        assert restored.df_id == meta.df_id
        assert restored.file_name == meta.file_name
        assert restored.notes == meta.notes
