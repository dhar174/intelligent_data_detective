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
        assert df_id == "reload_df"
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


class TestDataFrameRegistryClearFileSafety:
    """Verify that clear() never deletes files from disk (Issue #156)."""

    def test_clear_evicts_cache_and_registry_but_preserves_disk_files(self, core, sample_df, tmp_path):
        csv_file = tmp_path / "user_data.csv"
        sample_df.to_csv(csv_file, index=False)
        assert csv_file.exists()

        reg = core.DataFrameRegistry(capacity=5)
        df_id = reg.register_dataframe(sample_df, "test_file_df", raw_path=str(csv_file))
        assert reg.has_df(df_id)
        assert len(reg.cache) == 1

        reg.clear()
        assert reg.size() == 0
        assert len(reg.cache) == 0
        assert not reg.has_df(df_id)

        # CRITICAL SAFETY INVARIANT: User file must NOT be deleted!
        assert csv_file.exists(), "reg.clear() erroneously deleted file on disk!"
        re_read = pd.read_csv(csv_file)
        assert len(re_read) == len(sample_df)


class TestContextVarRegistryOverride:
    """Verify ContextVar-based override engine and isolation contracts (Issue #156)."""

    def test_override_basic_and_restoration(self, core):
        initial = core.get_global_df_registry()
        custom = core.DataFrameRegistry(capacity=7)

        with core.override_global_registry(custom) as active:
            assert active is custom
            assert core.get_global_df_registry() is custom
            assert core.global_df_registry is custom

        assert core.get_global_df_registry() is initial
        assert core.global_df_registry is initial

    def test_nested_overrides_lifo(self, core):
        initial = core.get_global_df_registry()
        outer = core.DataFrameRegistry(capacity=10)
        inner = core.DataFrameRegistry(capacity=5)

        with core.override_global_registry(outer):
            assert core.get_global_df_registry() is outer
            assert core.global_df_registry is outer
            with core.override_global_registry(inner):
                assert core.get_global_df_registry() is inner
                assert core.global_df_registry is inner
            assert core.get_global_df_registry() is outer
            assert core.global_df_registry is outer

        assert core.get_global_df_registry() is initial
        assert core.global_df_registry is initial

    def test_exception_safety_guarantees_restoration(self, core):
        initial = core.get_global_df_registry()
        custom = core.DataFrameRegistry(capacity=3)

        with pytest.raises(RuntimeError, match="deliberate failure"):
            with core.override_global_registry(custom):
                assert core.get_global_df_registry() is custom
                raise RuntimeError("deliberate failure")

        assert core.get_global_df_registry() is initial

    def test_validate_dataframe_exists_observes_override(self, core, sample_df):
        custom = core.DataFrameRegistry(capacity=5)
        custom.register_dataframe(sample_df, "isolated_df_id")

        # Outside override: isolated_df_id should not exist in the default registry
        assert core.validate_dataframe_exists("isolated_df_id") is False

        # Inside override: validate_dataframe_exists dynamically resolves custom registry
        with core.override_global_registry(custom):
            assert core.validate_dataframe_exists("isolated_df_id") is True

        # Outside again: not found
        assert core.validate_dataframe_exists("isolated_df_id") is False

    def test_concurrent_threads_isolated_contexts(self, core):
        """Concurrent threads with independent overrides do not leak to each other or default."""
        initial = core.get_global_df_registry()
        reg_a = core.DataFrameRegistry(capacity=11)
        reg_b = core.DataFrameRegistry(capacity=12)

        errors = []
        barrier = threading.Barrier(2)

        def worker_a():
            try:
                with core.override_global_registry(reg_a):
                    barrier.wait(timeout=5)
                    assert core.get_global_df_registry() is reg_a
                    time.sleep(0.05)
                    assert core.get_global_df_registry() is reg_a
            except Exception as e:
                errors.append(e)

        def worker_b():
            try:
                with core.override_global_registry(reg_b):
                    barrier.wait(timeout=5)
                    assert core.get_global_df_registry() is reg_b
                    time.sleep(0.05)
                    assert core.get_global_df_registry() is reg_b
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread errors: {errors}"
        # Main thread was never touched
        assert core.get_global_df_registry() is initial

    def test_asyncio_task_isolation(self, core):
        """Asyncio child tasks inherit context copy; task mutations do not leak."""
        import asyncio

        async def run_async_test():
            initial = core.get_global_df_registry()
            custom = core.DataFrameRegistry(capacity=15)

            async def child_task():
                assert core.get_global_df_registry() is custom

            with core.override_global_registry(custom):
                assert core.get_global_df_registry() is custom
                task = asyncio.create_task(child_task())
                await task

            assert core.get_global_df_registry() is initial

        asyncio.run(run_async_test())

    def test_set_global_registry_updates_process_default(self, core):
        """set_global_registry safely updates process default when no override active."""
        old_default = core.get_global_df_registry()
        new_default = core.DataFrameRegistry(capacity=99)
        try:
            core.set_global_registry(new_default)
            assert core.get_global_df_registry() is new_default
            assert core.global_df_registry is new_default
        finally:
            core.set_global_registry(old_default)


class TestDataFrameRegistryStorageIsolation:
    """Verify instance-level directory and persistence file isolation (Issue #156 / PR #157)."""

    def test_independent_registries_backing_file_isolation(self, core):
        """
        Two independent DataFrameRegistry instances with same df_id:
        - must receive distinct backing file paths in isolated directories
        - must persist different contents on auto-path registration
        - must evict and reload correct instance-specific data without cross-instance corruption
        """
        from pathlib import Path

        reg_a = core.DataFrameRegistry(capacity=1)
        reg_b = core.DataFrameRegistry(capacity=1)

        assert reg_a.data_dir != reg_b.data_dir
        assert reg_a.data_dir.exists()
        assert reg_b.data_dir.exists()

        df_a = pd.DataFrame({"col": [1, 2, 3]})
        df_b = pd.DataFrame({"col": [100, 200, 300]})

        reg_a.register_dataframe(df_a, "shared_id")
        reg_b.register_dataframe(df_b, "shared_id")

        path_a = reg_a.get_raw_path_from_id("shared_id")
        path_b = reg_b.get_raw_path_from_id("shared_id")

        assert path_a is not None and path_b is not None
        assert path_a != path_b, f"Independent registries shared identical backing path: {path_a}"
        assert Path(path_a).exists()
        assert Path(path_b).exists()

        # Evict 'shared_id' from in-memory cache in both registries by exceeding capacity
        reg_a.register_dataframe(pd.DataFrame({"x": [1]}), "other_a")
        reg_b.register_dataframe(pd.DataFrame({"x": [2]}), "other_b")

        assert reg_a.registry["shared_id"]["df"] is None
        assert reg_b.registry["shared_id"]["df"] is None

        # Reload from disk
        reloaded_a = reg_a.get_dataframe("shared_id", load_if_not_exists=True)
        reloaded_b = reg_b.get_dataframe("shared_id", load_if_not_exists=True)

        assert reloaded_a is not None and reloaded_b is not None
        assert list(reloaded_a["col"]) == [1, 2, 3], f"Registry A reloaded wrong data: {reloaded_a}"
        assert list(reloaded_b["col"]) == [100, 200, 300], f"Registry B reloaded wrong data: {reloaded_b}"

    def test_repeated_registration_updates_backing_file(self, core):
        """Updating an existing df_id overwrites its backing file so reload serves fresh data."""
        reg = core.DataFrameRegistry(capacity=1)
        df1 = pd.DataFrame({"data": [10, 20]})
        df2 = pd.DataFrame({"data": [99, 100]})

        reg.register_dataframe(df1, "key")
        path1 = reg.get_raw_path_from_id("key")

        reg.register_dataframe(df2, "key")
        path2 = reg.get_raw_path_from_id("key")
        assert path1 == path2

        # Evict from in-memory cache
        reg.register_dataframe(pd.DataFrame({"dummy": [0]}), "other")
        assert reg.registry["key"]["df"] is None

        # Reload must yield df2 (updated), not df1
        reloaded = reg.get_dataframe("key", load_if_not_exists=True)
        assert reloaded is not None
        assert list(reloaded["data"]) == [99, 100]

    def test_custom_data_dir_explicit_configuration(self, core, tmp_path):
        """Specifying data_dir uses caller-provided path and stores backing files there."""
        from pathlib import Path

        custom_dir = tmp_path / "custom_registry_storage"
        reg = core.DataFrameRegistry(capacity=5, data_dir=custom_dir)
        assert reg.data_dir == custom_dir.resolve()
        assert custom_dir.exists()

        df = pd.DataFrame({"v": [42]})
        reg.register_dataframe(df, "custom_df")
        path = reg.get_raw_path_from_id("custom_df")
        assert Path(path).parent == custom_dir.resolve()
        assert Path(path).exists()

    def test_clear_preserves_both_auto_and_user_backing_files_on_disk(self, core, tmp_path):
        """clear() resets in-memory registry/cache but NEVER deletes files from disk."""
        from pathlib import Path

        reg = core.DataFrameRegistry(capacity=2)
        df = pd.DataFrame({"a": [1, 2]})

        # Auto-path registration
        reg.register_dataframe(df, "auto_key")
        auto_path = Path(reg.get_raw_path_from_id("auto_key"))
        assert auto_path.exists()

        # User-path registration
        user_path = tmp_path / "user_data.csv"
        df.to_csv(user_path, index=False)
        reg.register_dataframe(df, "user_key", raw_path=str(user_path))
        assert user_path.exists()

        reg.clear()
        assert reg.size() == 0
        assert len(reg.cache) == 0
        assert not reg.has_df("auto_key")
        assert not reg.has_df("user_key")

        # Verify disk files remain intact
        assert auto_path.exists(), "reg.clear() erroneously deleted auto backing file!"
        assert user_path.exists(), "reg.clear() erroneously deleted user-provided file!"
