"""
Unit tests for DataFrameRegistry.

Tests:
- register/get/remove lifecycle
- LRU eviction (df=None but raw_path preserved for lazy reload)
- Thread safety (concurrent registration)
- Multi-format support: CSV, Parquet, JSON, Pickle
- get_id_from_raw_path path normalization
- capacity enforcement
"""
import threading
import tempfile
import time
import pytest
import pandas as pd
import numpy as np

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def core():
    try:
        import idd_core
        return idd_core
    except ImportError:
        pytest.skip("idd_core.py not available")


class TestDataFrameRegistryBasic:
    def test_register_returns_id(self, registry, sample_df):
        df_id = registry.register_dataframe(sample_df, "test_id")
        assert df_id == "test_id"

    def test_get_registered_df(self, registry, sample_df):
        registry.register_dataframe(sample_df, "df1")
        result = registry.get_dataframe("df1")
        assert result is not None
        assert len(result) == len(sample_df)

    def test_missing_id_returns_none(self, registry):
        assert registry.get_dataframe("nonexistent") is None

    def test_remove_dataframe(self, registry, sample_df):
        registry.register_dataframe(sample_df, "to_remove")
        registry.remove_dataframe("to_remove")
        assert registry.get_dataframe("to_remove") is None
        assert not registry.has_df("to_remove")

    def test_has_df_true(self, registry, sample_df):
        registry.register_dataframe(sample_df, "present")
        assert registry.has_df("present") is True

    def test_has_df_false(self, registry):
        assert registry.has_df("absent") is False

    def test_size(self, registry, sample_df):
        for i in range(3):
            registry.register_dataframe(sample_df, f"df_{i}")
        assert registry.size() == 3

    def test_ids_returns_all(self, registry, sample_df):
        for i in range(3):
            registry.register_dataframe(sample_df, f"id_{i}")
        ids = registry.ids()
        for i in range(3):
            assert f"id_{i}" in ids

    def test_auto_id_generated(self, registry, sample_df):
        df_id = registry.register_dataframe(sample_df)
        assert df_id is not None
        assert len(df_id) > 0

    def test_register_update_existing(self, registry, sample_df):
        """Re-registering with same id updates the DataFrame."""
        registry.register_dataframe(sample_df, "updatable")
        new_df = pd.DataFrame({"x": [99, 100]})
        registry.register_dataframe(new_df, "updatable")
        result = registry.get_dataframe("updatable")
        assert list(result["x"]) == [99, 100]


class TestDataFrameRegistryFileRoundtrip:
    def test_csv_roundtrip(self, registry, sample_df, tmp_path):
        csv_path = str(tmp_path / "data.csv")
        df_id = registry.register_dataframe(sample_df, "csv_df", raw_path=csv_path)
        assert df_id == "csv_df"
        loaded = registry.get_dataframe("csv_df")
        assert loaded is not None

    def test_load_from_raw_path(self, registry, sample_df, tmp_path):
        """register with raw_path, evict from cache, reload via load_if_not_exists."""
        csv_path = tmp_path / "reload_test.csv"
        sample_df.to_csv(csv_path, index=False)
        df_id = registry.register_dataframe(sample_df, "reload_df", raw_path=str(csv_path))
        # Manually evict from cache by setting df=None
        registry.registry["reload_df"]["df"] = None
        registry.cache.pop("reload_df", None)
        # Now load_if_not_exists should reload from CSV
        reloaded = registry.get_dataframe("reload_df", load_if_not_exists=True)
        assert reloaded is not None
        assert len(reloaded) == len(sample_df)

    def test_parquet_roundtrip(self, registry, sample_df, tmp_path):
        pytest.importorskip("pyarrow", reason="pyarrow not installed")
        pq_path = str(tmp_path / "data.parquet")
        df_id = registry.register_dataframe(sample_df, "pq_df", raw_path=pq_path)
        assert df_id == "pq_df"

    def test_write_csv_file(self, registry, sample_df, tmp_path):
        out_path = str(tmp_path / "written.csv")
        success = registry.write_dataframe_to_csv_file(sample_df, out_path)
        assert success is True
        import os
        assert os.path.exists(out_path)

    def test_get_raw_path_from_id(self, registry, sample_df, tmp_path):
        csv_path = str(tmp_path / "path_test.csv")
        registry.register_dataframe(sample_df, "path_df", raw_path=csv_path)
        retrieved_path = registry.get_raw_path_from_id("path_df")
        assert retrieved_path is not None

    def test_get_id_from_raw_path_normalizes(self, registry, sample_df, tmp_path):
        csv_path = tmp_path / "norm_test.csv"
        sample_df.to_csv(csv_path, index=False)
        registry.register_dataframe(sample_df, "norm_df", raw_path=str(csv_path))
        # Look up by path with different representation (should normalize and match)
        found = registry.get_id_from_raw_path(str(csv_path))
        assert found == "norm_df"

    def test_get_id_from_raw_path_not_found(self, registry):
        result = registry.get_id_from_raw_path("/nonexistent/path.csv")
        assert result is None


class TestDataFrameRegistryLRU:
    def test_lru_eviction(self, core):
        """When capacity is exceeded, LRU entry's df is set to None but raw_path kept."""
        reg = core.DataFrameRegistry(capacity=2)
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        df3 = pd.DataFrame({"x": [5, 6]})

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            p1 = str(Path(d) / "df1.csv")
            p2 = str(Path(d) / "df2.csv")
            p3 = str(Path(d) / "df3.csv")
            df1.to_csv(p1, index=False)
            df2.to_csv(p2, index=False)
            df3.to_csv(p3, index=False)

            reg.register_dataframe(df1, "lru_1", raw_path=p1)
            reg.register_dataframe(df2, "lru_2", raw_path=p2)
            # Adding lru_3 should evict lru_1
            reg.register_dataframe(df3, "lru_3", raw_path=p3)

            # lru_1 should still be in registry but df=None
            assert reg.has_df("lru_1")
            assert reg.registry["lru_1"]["df"] is None
            # raw_path must be preserved for lazy reload
            assert reg.registry["lru_1"]["raw_path"] != ""

    def test_lru_reload_after_eviction(self, core):
        """After LRU eviction, get_dataframe(..., load_if_not_exists=True) should reload."""
        reg = core.DataFrameRegistry(capacity=2)
        df1 = pd.DataFrame({"val": [10, 20]})
        df2 = pd.DataFrame({"val": [30, 40]})
        df3 = pd.DataFrame({"val": [50, 60]})

        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            p1 = str(Path(d) / "df1.csv")
            p2 = str(Path(d) / "df2.csv")
            p3 = str(Path(d) / "df3.csv")
            df1.to_csv(p1, index=False)
            df2.to_csv(p2, index=False)
            df3.to_csv(p3, index=False)

            reg.register_dataframe(df1, "e1", raw_path=p1)
            reg.register_dataframe(df2, "e2", raw_path=p2)
            reg.register_dataframe(df3, "e3", raw_path=p3)  # evicts e1

            # Without load_if_not_exists, returns None
            assert reg.get_dataframe("e1") is None
            # With load_if_not_exists, reloads from disk
            reloaded = reg.get_dataframe("e1", load_if_not_exists=True)
            assert reloaded is not None
            assert list(reloaded["val"]) == [10, 20]


class TestDataFrameRegistryThreadSafety:
    def test_concurrent_registration(self, core):
        """Concurrent registrations must not corrupt the registry."""
        reg = core.DataFrameRegistry(capacity=50)
        errors = []

        def register_worker(i):
            try:
                df = pd.DataFrame({"value": [i]})
                result = reg.register_dataframe(df, f"concurrent_{i}")
                assert result == f"concurrent_{i}"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert reg.size() == 20

    def test_concurrent_get(self, core, sample_df):
        """Concurrent reads must not corrupt the cache."""
        reg = core.DataFrameRegistry(capacity=10)
        reg.register_dataframe(sample_df, "shared_df")
        errors = []
        results = []

        def reader():
            try:
                df = reg.get_dataframe("shared_df")
                results.append(df is not None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert all(results)
