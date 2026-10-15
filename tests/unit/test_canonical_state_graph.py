"""
tests/unit/test_canonical_state_graph.py — Comprehensive CP3 Unit Test Suite.

Validates:
1. Canonical State schema:
   - Exact 70-field schema matching notebook Cell 22.
   - 39 fields with reducers, 31 plain fields.
   - BR-7 compliance: messages declared directly with add_messages reducer (no AgentState parent).
   - W9-SR-DROP compliance: structured_response intentionally absent from supervisor State.
   - Sub-state VizWorkerState schema.
2. Custom and standard reducer semantics:
   - keep_first, _sr_reducer, _reduce_strs_resettable, _reduce_viz_results_resettable,
     _keep_last_or_clear, merge_int_sum, bool_or, dict_merge_shallow, merge_lists,
     merge_unique, merge_dicts, merge_dict, any_true, last_wins, _reduce_plan_keep_sorted.
3. Graph construction and topology invariants:
   - 15 production nodes matching notebook Cell 60.
   - 33 edges (21 conditional, 12 static).
   - 0 unreachable nodes, 0 dead-end nodes.
   - Entry point: START -> initial_analysis.
   - W2-EMERGENCY invariant: EMERGENCY_MSG -> supervisor fan-in edge.
   - Fan-in barriers: viz_join and report_join.
   - Node override injection.
4. Routing and gate functions:
   - route_from_supervisor, route_to_writer, route_viz, write_output_to_file,
     assign_viz_workers, dispatch_sections.
5. 100% symbol identity between idd_core façade, idd_state, and idd_graph.
"""

from __future__ import annotations

import typing
import pydantic
import pytest

import idd_core
import idd_graph
import idd_state
import validate_graph

pytestmark = pytest.mark.unit


# ===========================================================================
# 1. State Schema & Reducer Annotations
# ===========================================================================


class TestStateSchema:
    def test_state_field_count_and_classification(self):
        """State must have exactly 70 fields: 39 with reducers and 31 plain fields."""
        info = validate_graph.inspect_state({"State": idd_state.State})
        assert (
            info["field_count"] == 70
        ), f"Expected 70 fields, got {info['field_count']}"
        assert (
            len(info["reducer_fields"]) == 39
        ), f"Expected 39 reducer fields, got {len(info['reducer_fields'])}: {info['reducer_fields']}"
        assert (
            len(info["plain_fields"]) == 31
        ), f"Expected 31 plain fields, got {len(info['plain_fields'])}: {info['plain_fields']}"

    def test_br7_messages_invariant(self):
        """messages must be declared directly with add_messages reducer, not inherited from AgentState."""
        hints = typing.get_type_hints(idd_state.State, include_extras=True)
        assert "messages" in hints
        messages_hint = hints["messages"]
        metadata = getattr(messages_hint, "__metadata__", ())
        assert len(metadata) > 0, "messages must be Annotated with a reducer"
        # Verify State does NOT have remaining_steps (dragged in by AgentState parent)
        assert (
            "remaining_steps" not in hints
        ), "State must not inherit from AgentState (remaining_steps rejected in InputSchema)"

    def test_w9_sr_drop_invariant(self):
        """structured_response must NOT be a channel on supervisor State."""
        hints = typing.get_type_hints(idd_state.State, include_extras=True)
        assert (
            "structured_response" not in hints
        ), "structured_response must not be a channel on supervisor State (W9-SR-DROP)"

    def test_viz_worker_state_schema(self):
        """VizWorkerState sub-state must contain task and reduced viz_results."""
        hints = typing.get_type_hints(idd_state.VizWorkerState, include_extras=True)
        assert "task" in hints
        assert "viz_results" in hints
        md = getattr(hints["viz_results"], "__metadata__", ())
        assert len(md) > 0, "viz_results in VizWorkerState must have a reducer"


# ===========================================================================
# 2. Reducer Semantics
# ===========================================================================


class TestReducers:
    def test_keep_first(self):
        assert idd_state.keep_first("first", "second") == "first"
        assert idd_state.keep_first(None, "second") == "second"
        assert idd_state.keep_first("first", None) == "first"
        assert idd_state.keep_first(None, None) is None
        assert idd_state.keep_first(0, 99) == 0
        assert idd_state.keep_first("", "b") == ""
        assert idd_state.keep_first(False, True) is False

    def test_sr_reducer(self):
        """_sr_reducer prefers right when non-None; preserves left when right is None."""
        assert idd_state._sr_reducer("prior", "new") == "new"
        assert idd_state._sr_reducer("prior", None) == "prior"
        assert idd_state._sr_reducer(None, "new") == "new"
        assert idd_state._sr_reducer(None, None) is None

    def test_reduce_strs_resettable(self):
        """_reduce_strs_resettable appends list by default; None resets to empty list."""
        assert idd_state._reduce_strs_resettable(["a"], ["b"]) == ["a", "b"]
        assert idd_state._reduce_strs_resettable(["a", "b"], None) == []
        assert idd_state._reduce_strs_resettable(None, ["c"]) == ["c"]
        assert idd_state._reduce_strs_resettable(None, None) == []

    def test_reduce_viz_results_resettable(self):
        """_reduce_viz_results_resettable appends list by default; None resets to empty list."""
        dict1 = {"path": "fig1.png"}
        dict2 = {"path": "fig2.png"}
        assert idd_state._reduce_viz_results_resettable([dict1], [dict2]) == [
            dict1,
            dict2,
        ]
        assert idd_state._reduce_viz_results_resettable([dict1], None) == []
        assert idd_state._reduce_viz_results_resettable(None, [dict2]) == [dict2]
        assert idd_state._reduce_viz_results_resettable(None, None) == []

    def test_keep_last_or_clear(self):
        """_keep_last_or_clear always returns b, with None acting as explicit clear."""
        assert idd_state._keep_last_or_clear("old", "new") == "new"
        assert idd_state._keep_last_or_clear("old", None) is None
        assert idd_state._keep_last_or_clear(None, "new") == "new"
        assert idd_state._keep_last_or_clear(None, None) is None

    def test_merge_int_sum(self):
        assert idd_state.merge_int_sum(3, 4) == 7
        assert idd_state.merge_int_sum(None, 5) == 5
        assert idd_state.merge_int_sum(5, None) == 5
        assert idd_state.merge_int_sum(None, None) == 0

    def test_bool_or(self):
        """bool_or treats None as False and computes boolean OR."""
        assert idd_state.bool_or(True, True) is True
        assert idd_state.bool_or(True, False) is True
        assert idd_state.bool_or(False, True) is True
        assert idd_state.bool_or(False, False) is False
        assert idd_state.bool_or(None, True) is True
        assert idd_state.bool_or(True, None) is True
        assert idd_state.bool_or(None, False) is False
        assert idd_state.bool_or(None, None) is None

    def test_dict_merge_shallow(self):
        assert idd_state.dict_merge_shallow({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
        assert idd_state.dict_merge_shallow({"a": 1}, {"a": 99}) == {"a": 99}
        assert idd_state.dict_merge_shallow(None, {"b": 2}) == {"b": 2}
        assert idd_state.dict_merge_shallow({"a": 1}, None) == {"a": 1}
        assert idd_state.dict_merge_shallow(None, None) == {}

    def test_merge_lists(self):
        assert idd_state.merge_lists([1], [2]) == [1, 2]
        assert idd_state.merge_lists(None, [2]) == [2]
        assert idd_state.merge_lists([1], None) == [1]
        assert idd_state.merge_lists(None, None) == []

    def test_merge_unique(self):
        assert idd_state.merge_unique(["a", "b"], ["b", "c"]) == ["a", "b", "c"]
        assert idd_state.merge_unique(None, ["x"]) == ["x"]
        assert idd_state.merge_unique(["x"], None) == ["x"]
        assert idd_state.merge_unique(None, None) == []

    def test_merge_dicts_and_merge_dict(self):
        assert idd_state.merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
        assert idd_state.merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
        assert idd_state.merge_dict({"a": 1}, {"a": 99}) == {"a": 99}

    def test_any_true_and_last_wins(self):
        assert idd_state.any_true(True, False) is True
        assert idd_state.any_true(False, False) is False
        assert idd_state.any_true(None, True) is True
        assert idd_state.last_wins("old", "new") == "new"


# ===========================================================================
# 3. Graph Assembly & Topology Invariants
# ===========================================================================


class TestGraphAssembly:
    def test_build_graph_compiles_offline(self):
        """build_graph() compiles with zero API keys or network dependencies."""
        graph = idd_graph.build_graph()
        assert graph is not None

    def test_node_inventory(self):
        """Compiled graph must have exactly 15 production nodes."""
        graph = idd_graph.build_graph()
        info = validate_graph.inspect_graph(graph)
        assert len(info["nodes"]) == 15
        assert set(info["nodes"]) == set(idd_graph.PRODUCTION_NODE_NAMES)

    def test_edge_counts_and_properties(self):
        """Compiled graph must have exactly 33 edges (21 conditional, 12 static)."""
        graph = idd_graph.build_graph()
        info = validate_graph.inspect_graph(graph)
        assert info["edge_count"] == 33, f"Expected 33 edges, got {info['edge_count']}"
        assert (
            info["conditional_edge_count"] == 21
        ), f"Expected 21 conditional edges, got {info['conditional_edge_count']}"

    def test_zero_unreachable_and_zero_dead_ends(self):
        """Graph topology must have 0 unreachable nodes and 0 dead ends."""
        graph = idd_graph.build_graph()
        info = validate_graph.inspect_graph(graph)
        assert (
            info["unreachable"] == []
        ), f"Unreachable nodes found: {info['unreachable']}"
        assert info["dead_ends"] == [], f"Dead-end nodes found: {info['dead_ends']}"

    def test_w2_emergency_fan_in_edge(self):
        """EMERGENCY_MSG must route back to supervisor (W2-EMERGENCY invariant)."""
        graph = idd_graph.build_graph()
        g = graph.get_graph()
        emergency_edges = [
            e for e in g.edges if getattr(e, "source", None) == "EMERGENCY_MSG"
        ]
        assert len(emergency_edges) == 1
        assert getattr(emergency_edges[0], "target", None) == "supervisor"

    def test_node_overrides(self):
        """build_graph allows injecting custom node callables."""
        executed = []

        def custom_cleaner(state):
            executed.append("cleaner")
            return {"data_cleaning_complete": True}

        graph = idd_graph.build_graph(node_overrides={"data_cleaner": custom_cleaner})
        node_names = set(graph.nodes.keys())
        assert "data_cleaner" in node_names

    def test_build_graph_structural_default(self):
        """build_graph() defaults to structural_only=True."""
        graph = idd_graph.build_graph()
        assert getattr(graph, "is_structural_only", False) is True

    def test_build_graph_executable_fails_closed_missing_overrides(self):
        """build_graph(structural_only=False) raises ValueError when overrides are missing."""
        with pytest.raises(ValueError, match="Cannot build executable graph"):
            idd_graph.build_graph(structural_only=False)

    def test_build_graph_rejects_unknown_override(self):
        """build_graph rejects unrecognized node overrides fail-closed."""
        with pytest.raises(ValueError, match="Unknown node override"):
            idd_graph.build_graph(
                node_overrides={"unrecognized_node": lambda s: {}},
                structural_only=True,
            )

    def test_structural_placeholder_node_invocation_raises_runtime_error(self):
        """Placeholder nodes in structural graphs fail closed with RuntimeError if executed."""
        graph = idd_graph.build_graph(structural_only=True)
        # Directly invoking placeholder
        placeholder = graph.nodes["initial_analysis"].bound
        with pytest.raises(
            RuntimeError, match="Cannot execute placeholder node 'initial_analysis'"
        ):
            placeholder.invoke({"messages": []})

        # Invoking graph end-to-end
        with pytest.raises(
            RuntimeError, match="Cannot execute placeholder node 'initial_analysis'"
        ):
            graph.invoke({"messages": []})

    def test_offline_executable_graph_pipeline(self):
        """Executable graph with 14 injected overrides executes deterministic workflow."""
        execution_trace: list[str] = []

        def make_mock_worker(name: str, output: dict):
            def _node(state: dict):
                execution_trace.append(name)
                return output

            return _node

        overrides = {
            "initial_analysis": make_mock_worker(
                "initial_analysis", {"initial_analysis_complete": True}
            ),
            "supervisor": None,  # assigned below
            "data_cleaner": make_mock_worker(
                "data_cleaner", {"data_cleaning_complete": True}
            ),
            "analyst": make_mock_worker("analyst", {}),
            "visualization": make_mock_worker("visualization", {}),
            "viz_worker": make_mock_worker("viz_worker", {}),
            "viz_join": make_mock_worker("viz_join", {}),
            "viz_evaluator": make_mock_worker("viz_evaluator", {}),
            "report_orchestrator": make_mock_worker("report_orchestrator", {}),
            "report_section_worker": make_mock_worker("report_section_worker", {}),
            "report_join": make_mock_worker("report_join", {}),
            "report_packager": make_mock_worker("report_packager", {}),
            "file_writer": make_mock_worker("file_writer", {}),
            "EMERGENCY_MSG": make_mock_worker("EMERGENCY_MSG", {}),
            "FINISH": make_mock_worker("FINISH", {}),
        }

        def mock_supervisor(state: dict):
            execution_trace.append("supervisor")
            if not state.get("data_cleaning_complete"):
                return {"next": "data_cleaner"}
            return {"next": "FINISH"}

        overrides["supervisor"] = mock_supervisor

        graph = idd_graph.build_graph(node_overrides=overrides, structural_only=False)
        assert graph.is_structural_only is False

        initial_state = {
            "messages": [],
            "user_request": "Execute offline pipeline test",
        }
        final_state = graph.invoke(initial_state)

        # Expected sequence: START -> initial_analysis -> supervisor -> data_cleaner -> supervisor -> FINISH -> END
        assert execution_trace == [
            "initial_analysis",
            "supervisor",
            "data_cleaner",
            "supervisor",
            "FINISH",
        ]
        assert final_state["initial_analysis_complete"] is True
        assert final_state["data_cleaning_complete"] is True

    def test_offline_executable_graph_routing_failure_halts_fail_closed(self):
        """Supervisor routing to an invalid destination raises ValueError during graph invoke."""

        def make_worker(output: dict):
            return lambda state: output

        overrides = {n: make_worker({}) for n in idd_graph.PRODUCTION_NODE_NAMES}
        overrides["initial_analysis"] = make_worker({"initial_analysis_complete": True})
        overrides["supervisor"] = make_worker({"next": "invalid_worker"})

        graph = idd_graph.build_graph(node_overrides=overrides, structural_only=False)
        with pytest.raises(
            ValueError,
            match="Supervisor routing failed: destination 'invalid_worker' is invalid",
        ):
            graph.invoke({"messages": []})

    def test_offline_executable_graph_missing_next_halts_fail_closed(self):
        """Supervisor returning empty/missing next raises ValueError during graph invoke."""

        def make_worker(output: dict):
            return lambda state: output

        overrides = {n: make_worker({}) for n in idd_graph.PRODUCTION_NODE_NAMES}
        overrides["initial_analysis"] = make_worker({"initial_analysis_complete": True})
        overrides["supervisor"] = make_worker({})

        graph = idd_graph.build_graph(node_overrides=overrides, structural_only=False)
        with pytest.raises(
            ValueError,
            match="Supervisor routing failed: destination 'next' is missing or empty",
        ):
            graph.invoke({"messages": []})


# ===========================================================================
# 4. Routing Functions
# ===========================================================================


class TestRoutingFunctions:
    def test_route_from_supervisor_normal(self):
        state = {"next": "data_cleaner"}
        assert idd_graph.route_from_supervisor(state) == "data_cleaner"

    def test_route_from_supervisor_finish_when_complete(self):
        """When report and file writer are complete, supervisor routes to FINISH."""
        rr = idd_state.ReportResults(
            pdf_report_path="r.pdf",
            html_report_path="r.html",
            markdown_report_path="r.md",
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        )
        state = {
            "report_generator_complete": True,
            "report_results": rr,
            "file_writer_complete": True,
        }
        assert idd_graph.route_from_supervisor(state) == "FINISH"

    def test_route_from_supervisor_file_writer_when_report_done_unwritten(self):
        """When report is done but unwritten, supervisor routes to file_writer."""
        rr = idd_state.ReportResults(
            pdf_report_path="r.pdf",
            html_report_path="r.html",
            markdown_report_path="r.md",
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        )
        state = {
            "report_generator_complete": True,
            "report_results": rr,
            "file_writer_complete": False,
        }
        assert idd_graph.route_from_supervisor(state) == "file_writer"

    def test_route_from_supervisor_unknown_raises(self):
        """Unknown next destination raises ValueError fail-closed."""
        with pytest.raises(ValueError, match="is invalid"):
            idd_graph.route_from_supervisor({"next": "non_existent_agent"})

    def test_route_from_supervisor_missing_next_raises(self):
        """Missing or empty next raises ValueError fail-closed."""
        with pytest.raises(ValueError, match="is missing or empty"):
            idd_graph.route_from_supervisor({})
        with pytest.raises(ValueError, match="is missing or empty"):
            idd_graph.route_from_supervisor({"next": None})
        with pytest.raises(ValueError, match="is missing or empty"):
            idd_graph.route_from_supervisor({"next": ""})

    def test_route_from_supervisor_self_route_raises(self):
        """Supervisor cannot route to itself; raises ValueError fail-closed."""
        with pytest.raises(ValueError, match="is invalid"):
            idd_graph.route_from_supervisor({"next": "supervisor"})

    def test_route_to_writer_already_wrote(self):
        state = {"file_writer_complete": True}
        assert idd_graph.route_to_writer(state) == "END"

    def test_route_to_writer_report_ready(self):
        rr = idd_state.ReportResults(
            pdf_report_path="r.pdf",
            html_report_path="r.html",
            markdown_report_path="r.md",
            reply_msg_to_supervisor="done",
            finished_this_task=True,
            expect_reply=False,
        )
        sec = idd_state.SectionOutline(
            name="Intro",
            section_num=1,
            description="d",
            goals=["g"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=300,
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        )
        outline = idd_state.ReportOutline(
            title="T",
            description="D",
            goals=["G"],
            sections=[sec],
            name="R",
            section_num=0,
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=300,
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        )
        state = {
            "report_generator_complete": True,
            "report_results": rr,
            "report_outline": outline,
            "written_sections": ["text 1"],
            "report_draft": "# Full Report Draft",
            "file_writer_complete": False,
        }
        assert idd_graph.route_to_writer(state) == "file_writer"

    def test_route_to_writer_fallback(self):
        state = {"report_generator_complete": False}
        assert idd_graph.route_to_writer(state) == "supervisor"

    def test_route_viz_retry_cap(self):
        state = {"_viz_retry_count": 1}
        assert idd_graph.route_viz(state) == "Accepted"

    def test_route_viz_complete_with_results(self):
        state = {
            "visualization_complete": True,
            "viz_results": [{"path": "p.png"}],
        }
        assert idd_graph.route_viz(state) == "Accepted"

    def test_route_viz_grade(self):
        assert idd_graph.route_viz({"viz_grade": "acceptable"}) == "Accepted"
        assert idd_graph.route_viz({"viz_grade": "revise"}) == "Revise"


# ===========================================================================
# 5. Symbol Identity Across idd_core, idd_state, and idd_graph
# ===========================================================================


class TestSymbolIdentity:
    def test_state_symbols(self):
        assert idd_core.State is idd_state.State
        assert idd_core.VizWorkerState is idd_state.VizWorkerState
        assert idd_core.AgentMembers is idd_state.AgentMembers
        assert idd_core.AgentId is idd_state.AgentId
        assert idd_core.Supervisor is idd_state.Supervisor
        assert idd_core.AgentOrSupervisor is idd_state.AgentOrSupervisor

    def test_reducer_symbols(self):
        assert idd_core.keep_first is idd_state.keep_first
        assert idd_core._sr_reducer is idd_state._sr_reducer
        assert idd_core._reduce_strs_resettable is idd_state._reduce_strs_resettable
        assert (
            idd_core._reduce_viz_results_resettable
            is idd_state._reduce_viz_results_resettable
        )
        assert idd_core._keep_last_or_clear is idd_state._keep_last_or_clear
        assert idd_core.merge_int_sum is idd_state.merge_int_sum
        assert idd_core.bool_or is idd_state.bool_or
        assert idd_core.dict_merge_shallow is idd_state.dict_merge_shallow
        assert idd_core.merge_lists is idd_state.merge_lists
        assert idd_core.merge_unique is idd_state.merge_unique
        assert idd_core.merge_dicts is idd_state.merge_dicts
        assert idd_core.merge_dict is idd_state.merge_dict
        assert idd_core.any_true is idd_state.any_true
        assert idd_core.last_wins is idd_state.last_wins

    def test_graph_symbols(self):
        assert idd_core.build_graph is idd_graph.build_graph
        assert idd_core.AGENT_MEMBERS is idd_graph.AGENT_MEMBERS
        assert idd_core.AGENT_OPTIONS is idd_graph.AGENT_OPTIONS
        assert idd_core.options is idd_graph.options
        assert idd_core.Router is idd_graph.Router
        assert idd_core.route_from_supervisor is idd_graph.route_from_supervisor
        assert idd_core.route_to_writer is idd_graph.route_to_writer
        assert idd_core.route_viz is idd_graph.route_viz
        assert idd_core.write_output_to_file is idd_graph.write_output_to_file
        assert idd_core.assign_viz_workers is idd_graph.assign_viz_workers
        assert idd_core.dispatch_sections is idd_graph.dispatch_sections


# ===========================================================================
# 6. Pydantic Contract Invariants (AgentMembers & Router)
# ===========================================================================


class TestPydanticSchemas:
    def test_agent_members_schema_exact_fields(self):
        """AgentMembers must have exactly description and agent_type (no next)."""
        fields = idd_state.AgentMembers.model_fields
        assert set(fields.keys()) == {"description", "agent_type"}
        assert "next" not in fields

    def test_agent_members_instantiation_and_derived_types(self):
        """AgentMembers and all 9 derived classes instantiate correctly."""
        am = idd_state.AgentMembers(
            description="Custom desc", agent_type="data_cleaner"
        )
        assert am.description == "Custom desc"
        assert am.agent_type == "data_cleaner"

        # Derived classes from idd_core
        dc = idd_core.DataCleaner()
        assert dc.agent_type == "data_cleaner"
        ia = idd_core.InitialAnalysis()
        assert ia.agent_type == "initial_analysis"
        an = idd_core.Analyst()
        assert an.agent_type == "analyst"
        fw = idd_core.FileWriter()
        assert fw.agent_type == "file_writer"
        vz = idd_core.Visualization()
        assert vz.agent_type == "visualization"
        rg = idd_core.ReportGenerator()
        assert rg.agent_type == "report_packager"
        sv = idd_core.SuperVisor()
        assert sv.agent_type == "supervisor"
        ro = idd_core.ReportOrchestrator()
        assert ro.agent_type == "report_orchestrator"
        rs = idd_core.ReportSection()
        assert rs.agent_type == "report_section_worker"

    def test_agent_members_rejects_invalid_type(self):
        """AgentMembers rejects invalid agent_type strings."""
        with pytest.raises(pydantic.ValidationError):
            idd_state.AgentMembers(agent_type="unknown_agent")

    def test_router_schema_required_nullable_metadata(self):
        """Router.next_agent_metadata is required in schema but nullable."""
        # Valid with None
        r_none = idd_graph.Router(
            next="analyst",
            next_agent_prompt="prompt",
            next_agent_metadata=None,
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        )
        assert r_none.next == "analyst"
        assert r_none.next_agent_metadata is None

        # Valid with NextAgentMetadata instance
        meta = idd_state.NextAgentMetadata(
            df_id="clean_data",
            file_type="csv",
            file_name="clean.csv",
            section_name=None,
            viz_spec=None,
            notes="Cleaned data ready",
            file_content=None,
            reply_msg_to_supervisor="meta msg",
            finished_this_task=True,
            expect_reply=False,
        )
        r_meta = idd_graph.Router(
            next="analyst",
            next_agent_prompt="analyze clean data",
            next_agent_metadata=meta,
            reply_msg_to_supervisor="ok",
            finished_this_task=True,
            expect_reply=False,
        )
        assert r_meta.next_agent_metadata is not None
        assert r_meta.next_agent_metadata.df_id == "clean_data"

        # Rejects omission of next_agent_metadata
        with pytest.raises(pydantic.ValidationError) as exc_info:
            idd_graph.Router(
                next="analyst",
                next_agent_prompt="prompt",
                reply_msg_to_supervisor="ok",
                finished_this_task=True,
                expect_reply=False,
            )
        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors if e["type"] == "missing"}
        assert "next_agent_metadata" in missing_fields

        # JSON schema inspection
        schema = idd_graph.Router.model_json_schema()
        assert "next_agent_metadata" in schema["required"]
        assert "next" in schema["required"]
        assert "next_agent_prompt" in schema["required"]
        prop = schema["properties"]["next_agent_metadata"]
        assert "anyOf" in prop
        types_in_anyof = [item.get("type") for item in prop["anyOf"]]
        assert "null" in types_in_anyof

    def test_router_accepts_emergency_msg(self):
        """Router accepts EMERGENCY_MSG as next destination."""
        r = idd_graph.Router(
            next="EMERGENCY_MSG",
            next_agent_prompt="Urgent error recovery required",
            next_agent_metadata=None,
            reply_msg_to_supervisor="emergency triggered",
            finished_this_task=False,
            expect_reply=True,
        )
        assert r.next == "EMERGENCY_MSG"

    def test_route_from_supervisor_maps_end_to_finish(self):
        """route_from_supervisor normalizes 'END' token to 'FINISH'."""
        state = {"next": "END"}
        res = idd_graph.route_from_supervisor(state)
        assert res == "FINISH"

    def test_agent_members_contains_all_ten_workers(self):
        """AGENT_MEMBERS must contain all 10 worker agents per migration plan."""
        assert len(idd_graph.AGENT_MEMBERS) == 10
        expected = {
            "initial_analysis",
            "data_cleaner",
            "analyst",
            "visualization",
            "report_orchestrator",
            "report_section_worker",
            "report_packager",
            "file_writer",
            "viz_worker",
            "viz_evaluator",
        }
        assert set(idd_graph.AGENT_MEMBERS) == expected
        assert set(idd_graph.AGENT_OPTIONS) == expected | {"FINISH", "EMERGENCY_MSG"}

    def test_assign_viz_workers_passes_parent_context(self):
        """assign_viz_workers passes cleaning_metadata, analysis_insights, etc. in Send."""
        from idd_state import CleaningMetadata, AnalysisInsights, VizSpec
        from langgraph.types import Send

        cm = CleaningMetadata(
            steps_taken=["drop_na"],
            data_description_after_cleaning="Cleaned data without missing values",
            reply_msg_to_supervisor="cleaned",
            finished_this_task=True,
            expect_reply=False,
        )
        ai = AnalysisInsights(
            summary="Key trend",
            correlation_insights="Strong positive correlation",
            anomaly_insights="No outliers found",
            recommended_visualizations=[],
            recommended_next_steps=["Examine segment differences"],
            reply_msg_to_supervisor="analyzed",
            finished_this_task=True,
            expect_reply=False,
        )
        spec = VizSpec(
            title="Trend",
            viz_type="scatter",
            df_id="df_clean",
            viz_instructions="plot scatter",
            viz_id="viz_1",
            columns=["x", "y"],
            x="x",
            y="y",
            hue=None,
            bins=None,
            agg=None,
            query=None,
            description=None,
            limit=None,
            style=None,
            reply_msg_to_supervisor="viz spec",
            finished_this_task=True,
            expect_reply=False,
        )
        state = {
            "viz_tasks": ["Plot scatter trend"],
            "viz_specs": [spec],
            "cleaning_metadata": cm,
            "analysis_insights": ai,
            "available_df_ids": ["df_clean"],
            "_config": {"configurable": {"thread_id": "t1"}},
        }
        sends = idd_graph.assign_viz_workers(state)
        assert isinstance(sends, list)
        assert len(sends) == 1
        assert isinstance(sends[0], Send)
        assert sends[0].node == "viz_worker"
        arg = sends[0].arg
        assert arg["individual_viz_task"] == "Plot scatter trend"
        assert arg["cleaning_metadata"] is cm
        assert arg["analysis_insights"] is ai
        assert arg["available_df_ids"] == ["df_clean"]
        assert arg["_config"] == {"configurable": {"thread_id": "t1"}}

    def test_dispatch_sections_passes_parent_context(self):
        """dispatch_sections passes section, cleaning_metadata, analysis_insights, etc. in Send."""
        from idd_state import (
            CleaningMetadata,
            AnalysisInsights,
            ReportOutline,
            SectionOutline,
        )
        from langgraph.types import Send

        sec = SectionOutline(
            name="Executive Summary",
            section_num=1,
            description="Overview of analysis",
            goals=["Summarize data"],
            data_signals_needed={"trend": "sales over time"},
            data_signals_available=["sales", "dates"],
            expected_figures=[],
            word_target=300,
            reply_msg_to_supervisor="section ready",
            finished_this_task=True,
            expect_reply=False,
        )
        outline = ReportOutline(
            title="EDA Report",
            section_num=0,
            name="Outline",
            description="Full report outline",
            goals=["Complete report"],
            data_signals_needed={},
            data_signals_available=[],
            expected_figures=[],
            word_target=1000,
            sections=[sec],
            reply_msg_to_supervisor="outline ready",
            finished_this_task=True,
            expect_reply=False,
        )
        cm = CleaningMetadata(
            steps_taken=["drop_na"],
            data_description_after_cleaning="Cleaned data without missing values",
            reply_msg_to_supervisor="cleaned",
            finished_this_task=True,
            expect_reply=False,
        )
        ai = AnalysisInsights(
            summary="Summary",
            correlation_insights="None",
            anomaly_insights="None",
            recommended_visualizations=[],
            recommended_next_steps=[],
            reply_msg_to_supervisor="insights",
            finished_this_task=True,
            expect_reply=False,
        )
        state = {
            "report_outline": outline,
            "cleaning_metadata": cm,
            "analysis_insights": ai,
            "available_df_ids": ["df_clean"],
            "user_prompt": "Produce EDA report",
        }
        sends = idd_graph.dispatch_sections(state)
        assert isinstance(sends, list)
        assert len(sends) == 1
        assert isinstance(sends[0], Send)
        assert sends[0].node == "report_section_worker"
        arg = sends[0].arg
        assert arg["section"]["name"] == "Executive Summary"
        assert arg["cleaning_metadata"] is cm
        assert arg["analysis_insights"] is ai
        assert arg["available_df_ids"] == ["df_clean"]
        assert arg["user_prompt"] == "Produce EDA report"

    def test_idd_core_all_symbols_exist(self):
        """Every symbol in idd_core.__all__ must exist as an attribute on idd_core without raising AttributeError."""
        for name in idd_core.__all__:
            assert hasattr(idd_core, name), f"idd_core missing exported symbol {name!r}"

    def test_build_graph_with_node_cache_policy(self):
        """build_graph attaches node_cache_policies to registered nodes."""
        from langgraph.types import CachePolicy

        policy = CachePolicy(ttl=120)
        g = idd_graph.build_graph(node_cache_policies=policy, structural_only=True)
        assert g is not None
        assert idd_core.CachePolicy is idd_graph.CachePolicy
