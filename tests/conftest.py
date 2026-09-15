"""
Shared fixtures for the IDD v5 testing harness.

All fixtures that need to reset global state (global_df_registry, Plan version counter)
are handled here to ensure test isolation.
"""
import threading
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# idd_core availability guard
# ---------------------------------------------------------------------------

def _idd_core():
    """Lazy import guard — tests skip cleanly if idd_core.py doesn't exist yet."""
    try:
        import idd_core
        return idd_core
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# DataFrameRegistry fixtures — fresh registry per test to prevent state leakage
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    """Fresh DataFrameRegistry with capacity=5 for each test."""
    core = _idd_core()
    if core is None:
        pytest.skip("idd_core.py not available")
    return core.DataFrameRegistry(capacity=5)


@pytest.fixture
def global_registry_reset():
    """
    Reset the global_df_registry singleton between tests.
    Tests that use validate_dataframe_exists must use this fixture.
    """
    core = _idd_core()
    if core is None:
        pytest.skip("idd_core.py not available")
    # Save original state
    original = core.global_df_registry
    # Replace with a fresh registry
    fresh = core.DataFrameRegistry(capacity=20)
    core.global_df_registry = fresh
    yield fresh
    # Restore after test
    core.global_df_registry = original


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """5-row DataFrame with numeric, text, and category columns."""
    return pd.DataFrame({
        "numeric": [1.0, 2.0, 3.0, None, 5.0],
        "text": ["a", "b", "c", "d", "e"],
        "category": ["X", "X", "Y", "Y", "Z"],
    })


@pytest.fixture
def dirty_df():
    """DataFrame with missing values and duplicates."""
    return pd.DataFrame({
        "value": [1, 2, None, 2, 5],
        "label": ["a", "b", "c", "b", None],
        "score": [0.1, float("nan"), 0.3, 0.2, 0.5],
    })


@pytest.fixture
def large_df():
    """100-row DataFrame for LRU eviction tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x": rng.integers(0, 100, 100).tolist(),
        "y": rng.random(100).tolist(),
    })


# ---------------------------------------------------------------------------
# Registered DataFrame fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def registered_df(registry, sample_df, tmp_path):
    """Register sample_df into a fresh registry; returns (df_id, registry)."""
    csv_path = tmp_path / "sample.csv"
    sample_df.to_csv(csv_path, index=False)
    df_id = registry.register_dataframe(sample_df, "test_df", raw_path=str(csv_path))
    return df_id, registry


# ---------------------------------------------------------------------------
# Temp directory
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir(tmp_path):
    """Alias for tmp_path."""
    return tmp_path


# ---------------------------------------------------------------------------
# Plan version isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def reset_plan_version_counter():
    """
    Reset Plan._next counter for tests that need deterministic plan_version.
    Note: Plan uses a ClassVar threading.Lock + itertools.count — we snapshot
    the counter position before the test and note the post-test delta.
    This fixture does NOT reset the counter (itertools.count can't be reset)
    but documents the intentional isolation issue for test authors.
    WARNING: tests should not assert on exact plan_version values, only ordering.
    """
    yield
    # Nothing to reset — plan_version is intentionally monotonic


# ---------------------------------------------------------------------------
# LangChain availability
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def has_langchain():
    """True if langchain is importable."""
    try:
        import langchain_core  # noqa: F401
        return True
    except ImportError:
        return False
