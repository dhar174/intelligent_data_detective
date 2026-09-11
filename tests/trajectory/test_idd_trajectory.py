"""
Trajectory evaluation tests for IDD v5.

These tests require a real OpenAI API key and are NOT run in CI.
They are gated by @pytest.mark.trajectory and the OPENAI_API_KEY env var.

Run manually:
    pytest tests/trajectory/ -m trajectory -v

Uses agentevals trajectory match evaluator in "superset" mode — extra internal
nodes (viz_worker, viz_join, etc.) don't cause failures.
"""
import json
import os
import pytest

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

pytestmark = pytest.mark.trajectory


@pytest.fixture(scope="module")
def idd_dataset():
    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", "idd_test_dataset.json"
    )
    with open(fixture_path) as f:
        return json.load(f)["examples"]


@pytest.fixture(scope="module")
def dirty_csv_path():
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_dirty.csv")


@pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_trajectory_dataset_loads(idd_dataset):
    """Verify the trajectory dataset structure is correct."""
    assert len(idd_dataset) >= 1
    for example in idd_dataset:
        assert "inputs" in example
        assert "outputs" in example
        assert "csv_path" in example["inputs"]
        assert "question" in example["inputs"]
        assert "expected_agents_visited" in example["outputs"]


@pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
@pytest.mark.slow
def test_basic_analysis_trajectory(idd_dataset, dirty_csv_path, tmp_path):
    """
    Run one IDD workflow end-to-end and verify the trajectory visits
    at minimum the required agent nodes.

    Uses agentevals create_trajectory_match_evaluator with mode="superset"
    so extra IDD-internal nodes (viz_worker, viz_join, etc.) don't fail.

    NOTE: This test takes 5-25 minutes. NEVER cancel it.
    """
    try:
        from agentevals.trajectory.match import create_trajectory_match_evaluator
    except ImportError:
        pytest.skip("agentevals not installed. Run: pip install agentevals")

    example = idd_dataset[0]
    expected_nodes = example["outputs"]["expected_agents_visited"]

    # The evaluator checks that our trajectory is a SUPERSET of expected_nodes
    # (IDD may visit extra internal nodes — that's fine)
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset"
    )

    # IDD workflow would be invoked here once we have the compiled graph.
    # Placeholder for the real invocation:
    # from idd_core import ...  # import compiled graph
    # result = graph.invoke({
    #     "user_prompt": example["inputs"]["question"],
    #     "current_dataframe": dirty_csv_path,
    # }, config={"configurable": {"thread_id": "test-trajectory-01"}})
    # actual_nodes_visited = extract_nodes_from_trace(result)
    # evaluation = evaluator(
    #     outputs=actual_nodes_visited,
    #     reference_outputs=expected_nodes,
    # )
    # assert evaluation["score"] is True

    pytest.skip(
        "Full trajectory test requires compiled graph + API key. "
        "Placeholder structure is in place — implement once graph compilation is verified."
    )


@pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_trajectory_fixtures_are_valid(dirty_csv_path):
    """Verify the trajectory test CSV fixture is well-formed."""
    import pandas as pd

    df = pd.read_csv(dirty_csv_path)
    assert len(df) > 0, "Fixture CSV must not be empty"
    assert "value" in df.columns
    assert "category" in df.columns
    assert "score" in df.columns

    # Verify it has known quality issues (missing values and duplicates)
    assert df.isnull().sum().sum() > 0, "Fixture must have missing values"
    assert df.duplicated().sum() > 0, "Fixture must have duplicate rows"
