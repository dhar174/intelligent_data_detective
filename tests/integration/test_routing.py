"""
Integration tests for supervisor routing with canonical options and schemas.

Tests that:
- Supervisor routes correctly to known agent nodes in AGENT_MEMBERS.
- Supervisor options include terminal / FINISH tokens.
- AgentMembers and Router models define proper next destination Literals.

These tests execute offline and do NOT require an API key.
"""

from __future__ import annotations

import typing
import pytest

pytestmark = pytest.mark.integration

VALID_ROUTES = {
    "initial_analysis",
    "data_cleaner",
    "analyst",
    "visualization",
    "viz_worker",
    "viz_join",
    "viz_evaluator",
    "report_orchestrator",
    "report_section_worker",
    "report_join",
    "report_packager",
    "file_writer",
    "EMERGENCY_MSG",
    "FINISH",
    "__end__",
    "END",
}


@pytest.fixture(scope="module")
def core():
    try:
        import idd_core

        if not idd_core.HAS_LANGCHAIN:
            pytest.skip("LangChain not available")
        return idd_core
    except ImportError:
        pytest.skip("idd_core.py not available")


class TestSupervisorRouting:
    def test_route_destinations_are_known_nodes(self, core):
        """All possible supervisor routing targets must be in the known valid set."""
        members = getattr(core, "AGENT_MEMBERS", None) or getattr(core, "members", None)
        assert (
            members is not None
        ), "AGENT_MEMBERS/members constant not exposed in idd_core"
        unknown = set(members) - VALID_ROUTES - {"supervisor"}
        assert len(members) > 0, "At least one routing target must be defined"
        assert not unknown, f"Unknown routing targets found: {unknown}"

    def test_supervisor_options_include_finish(self, core):
        """Supervisor routing options must include a terminal/finish option."""
        options = getattr(core, "options", None) or getattr(core, "AGENT_OPTIONS", None)
        assert options is not None, "'options' constant not exposed in idd_core"
        finish_tokens = {"FINISH", "END", "__end__"}
        assert finish_tokens & set(
            options
        ), f"No terminal option found in supervisor options: {options}"

    def test_agent_members_enum_or_literal(self, core):
        """AgentMembers must define agent_type (without next), and Router must define next."""
        model = getattr(core, "AgentMembers", None)
        assert model is not None, "AgentMembers not exposed in idd_core"
        hints = typing.get_type_hints(model)
        assert "agent_type" in hints, "AgentMembers must have 'agent_type' field"
        assert "description" in hints, "AgentMembers must have 'description' field"
        assert (
            "next" not in hints
        ), "AgentMembers must NOT have 'next' field (routing belongs to Router)"
        agent_type_hint = hints["agent_type"]
        args = getattr(agent_type_hint, "__args__", ())
        assert (
            len(args) == 9
        ), f"AgentMembers.agent_type must have 9 options, got {len(args)}"

        # Verify canonical Router model and its destination options
        router_cls = getattr(core, "Router", None)
        assert router_cls is not None, "Router model not exposed in idd_core"
        router_hints = typing.get_type_hints(router_cls)
        assert "next" in router_hints, "Router model must have 'next' field"
        router_next_args = getattr(router_hints["next"], "__args__", ())
        assert router_next_args, "Router.next must have destination options"
        assert set(router_next_args).issubset(
            VALID_ROUTES
        ), f"Router.next has destinations outside VALID_ROUTES: {set(router_next_args) - VALID_ROUTES}"
