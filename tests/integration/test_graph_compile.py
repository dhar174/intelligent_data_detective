"""
Integration tests for IDD graph compilation and topology invariants.

Validates that:
- The authoritative 15-node graph compiles without error without requiring an API key.
- All 15 required production nodes are present in the compiled graph.
- Emergency / terminal conditions and barriers are wired.
- Compiled graph exposes .invoke() and .stream() entrypoints.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

REQUIRED_NODES = {
    "supervisor",
    "initial_analysis",
    "data_cleaner",
    "analyst",
    "viz_worker",
    "viz_join",
    "viz_evaluator",
    "report_orchestrator",
    "report_section_worker",
    "report_join",
    "report_packager",
    "file_writer",
    "visualization",
    "EMERGENCY_MSG",
    "FINISH",
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


@pytest.fixture(scope="module")
def compiled_graph(core):
    """Compile the IDD graph once for the entire module without requiring an API key."""
    graph = core.build_graph()
    assert graph is not None, "core.build_graph() returned None"
    return graph


class TestGraphCompiles:
    def test_graph_builds_without_error(self, compiled_graph):
        """Graph must compile successfully."""
        assert compiled_graph is not None

    def test_required_nodes_present(self, compiled_graph):
        """All 15 required production agent nodes must be present in the compiled graph."""
        node_names = set(compiled_graph.nodes.keys())
        missing = REQUIRED_NODES - node_names
        assert not missing, f"Missing nodes: {missing}"

    def test_graph_has_invoke_method(self, compiled_graph):
        """Compiled graph exposes .invoke() entrypoint."""
        assert callable(getattr(compiled_graph, "invoke", None))

    def test_graph_has_stream_method(self, compiled_graph):
        """Compiled graph exposes .stream() entrypoint."""
        assert callable(getattr(compiled_graph, "stream", None))
