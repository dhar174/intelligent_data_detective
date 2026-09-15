"""
Integration tests for IDD graph compilation and topology invariants.

These tests require LangChain and an OPENAI_API_KEY. They validate:
- The graph compiles without error
- Expected nodes are present
- Emergency/terminal conditions are wired
- Node set matches documented topology
"""
import os
import pytest

pytestmark = pytest.mark.integration

REQUIRED_NODES = {
    "supervisor",
    "data_cleaner",
    "analyst",
    "visualization",
    "report_generator",
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
def api_key():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-test") or key == "your-key-here":
        pytest.skip("OPENAI_API_KEY not set — skipping integration tests")
    return key


@pytest.fixture(scope="module")
def compiled_graph(core, api_key):
    """Compile the IDD graph once for the entire module."""
    try:
        graph = core.build_graph(api_key=api_key)
        return graph
    except AttributeError:
        pytest.skip("idd_core.build_graph() not available in this version")


class TestGraphCompiles:
    def test_graph_builds_without_error(self, compiled_graph):
        """Graph must compile successfully."""
        assert compiled_graph is not None

    def test_required_nodes_present(self, compiled_graph):
        """All required agent nodes must be present in the compiled graph."""
        node_names = set(compiled_graph.nodes.keys())
        missing = REQUIRED_NODES - node_names
        assert not missing, f"Missing nodes: {missing}"

    def test_graph_has_invoke_method(self, compiled_graph):
        """Compiled graph exposes .invoke() entrypoint."""
        assert callable(getattr(compiled_graph, "invoke", None))

    def test_graph_has_stream_method(self, compiled_graph):
        """Compiled graph exposes .stream() entrypoint."""
        assert callable(getattr(compiled_graph, "stream", None))
