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
import os
import threading
import tempfile
import time
from pathlib import Path
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


class TestExplicitSourceCopyOnWriteIntegrity:
    """Verify copy-on-write persistence for explicit-source DataFrames (Issue #156 / PR #157 P1)."""

    def test_explicit_source_is_never_overwritten(self, core, tmp_path):
        """Test A: Re-registering updated DataFrame with explicit raw_path leaves original source untouched."""
        src_file = tmp_path / "user_source.csv"
        df_v1 = pd.DataFrame({"value": [1, 2, 3]})
        df_v1.to_csv(src_file, index=False)
        original_bytes = src_file.read_bytes()

        reg = core.DataFrameRegistry(capacity=5)
        reg.register_dataframe(df_v1, "dataset_1", raw_path=str(src_file))

        # Re-register transformed DataFrame under same ID
        df_v2 = pd.DataFrame({"value": [100, 200, 300]})
        reg.register_dataframe(df_v2, "dataset_1", raw_path=str(src_file))

        # Source file must remain byte-for-byte unchanged
        assert src_file.read_bytes() == original_bytes, "Original source file was modified!"

    def test_updated_data_survives_eviction(self, core, tmp_path):
        """Test B: After LRU eviction, reloading an updated explicit-source DataFrame yields updated data."""
        src_file = tmp_path / "user_source.csv"
        df_v1 = pd.DataFrame({"value": [1, 2, 3]})
        df_v1.to_csv(src_file, index=False)
        original_bytes = src_file.read_bytes()

        reg = core.DataFrameRegistry(capacity=1)
        reg.register_dataframe(df_v1, "dataset_1", raw_path=str(src_file))

        df_v2 = pd.DataFrame({"value": [100, 200, 300]})
        reg.register_dataframe(df_v2, "dataset_1", raw_path=str(src_file))

        # Evict dataset_1 from cache by adding another DataFrame
        reg.register_dataframe(pd.DataFrame({"x": [9]}), "dataset_2")
        assert reg.registry["dataset_1"]["df"] is None

        # Reload dataset_1 from disk
        reloaded = reg.get_dataframe("dataset_1", load_if_not_exists=True)
        assert reloaded is not None
        assert list(reloaded["value"]) == [100, 200, 300], f"Reload served stale data: {reloaded}"

        # Source file remains unchanged
        assert src_file.read_bytes() == original_bytes

    def test_repeated_updates_survive_eviction(self, core, tmp_path):
        """Test C: Registering versions A, B, and C under the same ID: after eviction, C is returned."""
        src_file = tmp_path / "source_abc.csv"
        df_a = pd.DataFrame({"v": [1]})
        df_a.to_csv(src_file, index=False)

        reg = core.DataFrameRegistry(capacity=1)
        reg.register_dataframe(df_a, "target_id", raw_path=str(src_file))

        df_b = pd.DataFrame({"v": [2]})
        reg.register_dataframe(df_b, "target_id", raw_path=str(src_file))

        df_c = pd.DataFrame({"v": [3]})
        reg.register_dataframe(df_c, "target_id", raw_path=str(src_file))

        # Evict target_id
        reg.register_dataframe(pd.DataFrame({"other": [0]}), "evictor")
        assert reg.registry["target_id"]["df"] is None

        reloaded = reg.get_dataframe("target_id", load_if_not_exists=True)
        assert reloaded is not None
        assert list(reloaded["v"]) == [3], f"Expected version C (3), got {reloaded}"

    def test_failed_persistence_does_not_corrupt_registry_or_falsely_succeed(self, core, tmp_path, monkeypatch):
        """Test D: Backing-write failure reports failure and does not leave metadata pointing to stale data."""
        reg = core.DataFrameRegistry(capacity=1)
        df_v1 = pd.DataFrame({"val": [10]})
        reg.register_dataframe(df_v1, "fail_test")

        df_v2 = pd.DataFrame({"val": [20]})

        # Simulate write failure
        monkeypatch.setattr(reg, "_write_df", lambda df, path: False)

        result = reg.register_dataframe(df_v2, "fail_test")
        assert result is None, "register_dataframe should return None on failed write"

        # In-memory cached df must still be df_v1, not falsely updated
        cached = reg.get_dataframe("fail_test")
        assert list(cached["val"]) == [10]

    def test_existing_auto_path_behavior_preserved(self, core):
        """Test E: Auto-path re-registration updates backing file and reload serves fresh data."""
        reg = core.DataFrameRegistry(capacity=1)
        df1 = pd.DataFrame({"x": [1]})
        df2 = pd.DataFrame({"x": [2]})

        reg.register_dataframe(df1, "auto_id")
        reg.register_dataframe(df2, "auto_id")

        reg.register_dataframe(pd.DataFrame({"dummy": [0]}), "other")
        assert reg.registry["auto_id"]["df"] is None

        reloaded = reg.get_dataframe("auto_id", load_if_not_exists=True)
        assert reloaded is not None
        assert list(reloaded["x"]) == [2]

    def test_source_path_compatibility_accessors(self, core, tmp_path):
        """Test F: Source-path lookup behavior follows the explicitly documented contract."""
        src_file = tmp_path / "provenance.csv"
        df1 = pd.DataFrame({"a": [1]})
        df1.to_csv(src_file, index=False)

        reg = core.DataFrameRegistry(capacity=2)
        reg.register_dataframe(df1, "prov_df", raw_path=str(src_file))

        # Before update: raw_path and source_path match src_file
        assert reg.get_raw_path_from_id("prov_df") == str(src_file.resolve())
        assert reg.get_source_path_from_id("prov_df") == str(src_file.resolve())
        assert reg.get_id_from_raw_path(str(src_file)) == "prov_df"

        # After update: source_path is preserved, backing path is created in reg.data_dir
        df2 = pd.DataFrame({"a": [2]})
        reg.register_dataframe(df2, "prov_df", raw_path=str(src_file))

        assert reg.get_source_path_from_id("prov_df") == str(src_file.resolve())
        backing_path = reg.get_backing_path_from_id("prov_df")
        assert backing_path is not None
        assert Path(backing_path).parent == reg.data_dir.resolve()
        # Looking up by either source or backing path returns prov_df
        assert reg.get_id_from_raw_path(str(src_file)) == "prov_df"
        assert reg.get_id_from_raw_path(backing_path) == "prov_df"

    def test_initially_transformed_dataframe_survives_eviction(self, core, tmp_path):
        """Test A (Initial Transformed): First registration of in-memory df with explicit source persists to owned backing."""
        src_file = tmp_path / "original_source.csv"
        original_df = pd.DataFrame({"value": [1, 2, 3]})
        original_df.to_csv(src_file, index=False)
        original_bytes = src_file.read_bytes()

        transformed_df = pd.DataFrame({"value": [100, 200, 300]})

        reg = core.DataFrameRegistry(capacity=1)
        df_id = reg.register_dataframe(transformed_df, "dataset_1", raw_path=str(src_file))
        assert df_id == "dataset_1"

        # Metadata checks
        source_p = reg.get_source_path_from_id("dataset_1")
        backing_p = reg.get_backing_path_from_id("dataset_1")
        assert source_p == str(src_file.resolve())
        assert backing_p is not None
        assert source_p != backing_p
        assert Path(backing_p).parent == reg.data_dir.resolve()

        # Original source file must remain byte-for-byte unchanged
        assert src_file.read_bytes() == original_bytes

        # Backing file on disk contains the transformed data
        backed_df = pd.read_pickle(backing_p)
        assert list(backed_df["value"]) == [100, 200, 300]

        # Evict dataset_1 by exceeding capacity
        reg.register_dataframe(pd.DataFrame({"evict": [999]}), "other")
        assert reg.registry["dataset_1"]["df"] is None

        # Reload from disk: must return transformed values [100, 200, 300], NOT [1, 2, 3]
        reloaded = reg.get_dataframe("dataset_1", load_if_not_exists=True)
        assert reloaded is not None
        assert list(reloaded["value"]) == [100, 200, 300]
        assert src_file.read_bytes() == original_bytes

    def test_validation_after_initial_registration(self, core, tmp_path):
        """Test B (Initial Transformed Validation): validate_dataframe_exists after eviction restores transformed data."""
        src_file = tmp_path / "original_source.csv"
        original_df = pd.DataFrame({"value": [1, 2, 3]})
        original_df.to_csv(src_file, index=False)

        transformed_df = pd.DataFrame({"value": [100, 200, 300]})

        reg = core.DataFrameRegistry(capacity=1)
        with core.override_global_registry(reg):
            reg.register_dataframe(transformed_df, "dataset_1", raw_path=str(src_file))

            # Evict from in-memory cache
            reg.register_dataframe(pd.DataFrame({"x": [0]}), "evictor")
            assert reg.registry["dataset_1"]["df"] is None

            # Validate exists: must succeed and restore transformed data
            assert core.validate_dataframe_exists("dataset_1") is True
            working = reg.get_dataframe("dataset_1")
            assert list(working["value"]) == [100, 200, 300]

    def test_initial_backing_write_failure_leaves_no_partial_state(self, core, tmp_path, monkeypatch):
        """Test C (Initial Persistence Failure): Failure to persist initial transformed df leaves registry clean."""
        src_file = tmp_path / "original_source.csv"
        original_df = pd.DataFrame({"value": [1, 2, 3]})
        original_df.to_csv(src_file, index=False)
        original_bytes = src_file.read_bytes()

        reg = core.DataFrameRegistry(capacity=1)
        monkeypatch.setattr(reg, "_write_df", lambda df, p: False)

        transformed_df = pd.DataFrame({"value": [100, 200, 300]})
        result = reg.register_dataframe(transformed_df, "dataset_fail", raw_path=str(src_file))

        assert result is None
        assert "dataset_fail" not in reg.registry
        assert "dataset_fail" not in reg.cache
        assert "dataset_fail" not in reg.df_id_to_raw_path
        assert src_file.read_bytes() == original_bytes

    def test_existing_source_loading_without_df_preserved(self, core, tmp_path):
        """Test D (Source Loading): df=None with existing raw_path loads the file and records source_path."""
        src_file = tmp_path / "existing.csv"
        source_df = pd.DataFrame({"value": [1, 2, 3]})
        source_df.to_csv(src_file, index=False)

        reg = core.DataFrameRegistry(capacity=2)
        df_id = reg.register_dataframe(df=None, df_id="existing_id", raw_path=str(src_file))
        assert df_id == "existing_id"

        loaded = reg.get_dataframe("existing_id")
        assert loaded is not None
        assert list(loaded["value"]) == [1, 2, 3]
        assert reg.get_source_path_from_id("existing_id") == str(src_file.resolve())
        assert reg.get_raw_path_from_id("existing_id") == str(src_file.resolve())

    def test_subsequent_transformations_after_initial_registration(self, core, tmp_path):
        """Test E (Subsequent Updates): Updates after initial registration continue to use owned backing."""
        src_file = tmp_path / "source.csv"
        original_df = pd.DataFrame({"value": [1, 2, 3]})
        original_df.to_csv(src_file, index=False)

        reg = core.DataFrameRegistry(capacity=1)
        v1 = pd.DataFrame({"value": [10, 20, 30]})
        reg.register_dataframe(v1, "seq_df", raw_path=str(src_file))

        v2 = pd.DataFrame({"value": [100, 200, 300]})
        reg.register_dataframe(v2, "seq_df", raw_path=str(src_file))

        # Evict
        reg.register_dataframe(pd.DataFrame({"x": [0]}), "evictor")
        assert reg.registry["seq_df"]["df"] is None

        reloaded = reg.get_dataframe("seq_df", load_if_not_exists=True)
        assert list(reloaded["value"]) == [100, 200, 300]
        assert pd.read_csv(src_file)["value"].tolist() == [1, 2, 3]


class TestDataFrameIdFilesystemContainment:
    """Verify DataFrame ID decoupling and storage directory containment (Issue #156 / PR #157 P1)."""

    @pytest.mark.parametrize("malicious_id", [
        "normal_id",
        "some-uuid-like-value",
        "../outside",
        "../../escape",
        "folder/child",
        r"..\outside",
        r"C:\temp\outside",
        "/absolute/path",
        "",
    ])
    def test_various_ids_never_escape_owned_storage(self, core, malicious_id):
        """Verify candidate IDs (traversal, absolute, nested, empty) never escape data_dir."""
        reg = core.DataFrameRegistry(capacity=5)
        df = pd.DataFrame({"col": [1, 2, 3]})

        df_id = reg.register_dataframe(df, malicious_id)
        assert df_id == malicious_id

        backing_path = Path(reg.get_backing_path_from_id(malicious_id))
        data_dir_resolved = reg.data_dir.resolve()

        # Mandatory containment verification
        assert backing_path.is_relative_to(data_dir_resolved), (
            f"Backing path '{backing_path}' escaped data_dir '{data_dir_resolved}' for id '{malicious_id}'"
        )
        assert backing_path.exists()

    def test_parent_traversal_does_not_write_in_parent(self, core):
        """Test A: Parent traversal ../outside does not create files outside owned directory."""
        reg = core.DataFrameRegistry(capacity=5)
        df = pd.DataFrame({"col": [1]})
        parent_dir = reg.data_dir.parent.resolve()
        parent_files_before = set(parent_dir.iterdir())

        reg.register_dataframe(df, "../outside")

        parent_files_after = set(parent_dir.iterdir())
        unexpected_files = parent_files_after - parent_files_before - {reg.data_dir}
        assert not unexpected_files, f"Unexpected files created in parent dir: {unexpected_files}"

    def test_multiple_traversal_levels_contained(self, core):
        """Test B: ../../escape does not escape owned directory."""
        reg = core.DataFrameRegistry(capacity=5)
        df = pd.DataFrame({"col": [1]})
        reg.register_dataframe(df, "../../escape")
        backing_path = Path(reg.get_backing_path_from_id("../../escape"))
        assert backing_path.is_relative_to(reg.data_dir.resolve())
        assert backing_path.exists()

    def test_windows_path_syntax_contained(self, core):
        """Test C: Windows path syntax (backslashes, colons) is safely contained."""
        reg = core.DataFrameRegistry(capacity=5)
        df = pd.DataFrame({"col": [1]})
        reg.register_dataframe(df, r"..\win_outside")
        reg.register_dataframe(df, r"C:\temp\win_abs")
        for test_id in [r"..\win_outside", r"C:\temp\win_abs"]:
            path = Path(reg.get_backing_path_from_id(test_id))
            assert path.is_relative_to(reg.data_dir.resolve())
            assert path.exists()

    def test_collision_resistance_for_different_ids(self, core):
        """Test E: Distinct logical IDs never map to the same backing filename."""
        reg = core.DataFrameRegistry(capacity=10)
        df = pd.DataFrame({"a": [1]})
        test_ids = [
            "outside",
            "../outside",
            "../../outside",
            r"..\outside",
            "folder/outside",
            "outside_1",
            "outside_2",
            "",
            "normal_id",
        ]
        backing_paths = set()
        for tid in test_ids:
            reg.register_dataframe(df, tid)
            p = reg.get_backing_path_from_id(tid)
            assert p not in backing_paths, f"Backing path collision detected for id '{tid}': {p}"
            backing_paths.add(p)

    def test_eviction_and_reload_with_sanitized_id(self, core):
        """Test F: Safe filename mapping remains consistent across registration, eviction, and reload."""
        reg = core.DataFrameRegistry(capacity=1)
        df = pd.DataFrame({"data": [42, 43]})
        reg.register_dataframe(df, "../special_id")

        # Evict from cache
        reg.register_dataframe(pd.DataFrame({"x": [0]}), "evictor")
        assert reg.registry["../special_id"]["df"] is None

        # Reload
        reloaded = reg.get_dataframe("../special_id", load_if_not_exists=True)
        assert reloaded is not None
        assert list(reloaded["data"]) == [42, 43]

    def test_explicit_source_path_distinct_from_auto_filename_policy(self, core, tmp_path):
        """Test H: Explicit user-supplied source paths remain distinct from auto-generated policy."""
        reg = core.DataFrameRegistry(capacity=5)
        user_file = tmp_path / "explicit_user.csv"
        df = pd.DataFrame({"v": [99]})
        df.to_csv(user_file, index=False)

        reg.register_dataframe(df, "norm_key", raw_path=str(user_file))
        assert reg.get_source_path_from_id("norm_key") == str(user_file.resolve())
        assert not Path(reg.get_source_path_from_id("norm_key")).is_relative_to(reg.data_dir.resolve())

    def test_distinct_unicode_surrogate_ids_do_not_collide(self, core):
        """Test I: Distinct Unicode surrogate strings do not collapse to identical backing filenames."""
        reg = core.DataFrameRegistry(capacity=5)
        id_a = "\ud800"
        id_b = "\udc00"

        df_a = pd.DataFrame({"val": [1, 2]})
        df_b = pd.DataFrame({"val": [100, 200]})

        reg.register_dataframe(df_a, id_a)
        reg.register_dataframe(df_b, id_b)

        path_a = reg.get_backing_path_from_id(id_a)
        path_b = reg.get_backing_path_from_id(id_b)

        assert path_a is not None and path_b is not None
        assert path_a != path_b, f"Surrogate IDs mapped to identical backing path: {path_a}"
        assert Path(path_a).exists()
        assert Path(path_b).exists()

        # Evict from cache
        reg.cache.clear()
        reg.registry[id_a]["df"] = None
        reg.registry[id_b]["df"] = None

        # Reload both independently
        reloaded_a = reg.get_dataframe(id_a, load_if_not_exists=True)
        reloaded_b = reg.get_dataframe(id_b, load_if_not_exists=True)

        assert reloaded_a is not None and reloaded_b is not None
        assert list(reloaded_a["val"]) == [1, 2]
        assert list(reloaded_b["val"]) == [100, 200]

    def test_legitimate_unicode_ids_supported(self, core):
        """Test J: Legitimate Unicode IDs (Japanese, accented, emoji) remain supported and distinct."""
        reg = core.DataFrameRegistry(capacity=5)
        test_ids = ["café", "データ", "emoji🚀"]
        backing_paths = set()

        for tid in test_ids:
            df = pd.DataFrame({"name": [tid]})
            reg.register_dataframe(df, tid)
            bp = reg.get_backing_path_from_id(tid)
            assert bp not in backing_paths
            backing_paths.add(bp)
            assert Path(bp).exists()

        reg.cache.clear()
        for tid in test_ids:
            reg.registry[tid]["df"] = None
            reloaded = reg.get_dataframe(tid, load_if_not_exists=True)
            assert reloaded is not None
            assert list(reloaded["name"]) == [tid]


class TestLosslessDataFramePersistence:
    """Verify lossless DataFrame snapshot persistence across cache eviction (PR #157 P1)."""

    def test_column_name_types_preserved_losslessly(self, core):
        """Test A: Mixed column label types (int 0 vs str '0') preserved across cache eviction."""
        reg = core.DataFrameRegistry(capacity=1)
        original_df = pd.DataFrame(
            [[10, 20, "text"]],
            columns=[0, "0", "col_str"],
        )
        assert type(original_df.columns[0]) is int
        assert type(original_df.columns[1]) is str

        df_id = reg.register_dataframe(original_df, "mixed_cols")
        assert df_id == "mixed_cols"

        # Evict from in-memory cache
        reg.register_dataframe(pd.DataFrame({"evict": [1]}), "evictor")
        assert reg.registry["mixed_cols"]["df"] is None

        # Reload from backing snapshot
        reloaded = reg.get_dataframe("mixed_cols", load_if_not_exists=True)
        assert reloaded is not None
        pd.testing.assert_frame_equal(original_df, reloaded, check_dtype=True, check_names=True)
        assert type(reloaded.columns[0]) is int
        assert type(reloaded.columns[1]) is str

    def test_timezone_aware_datetimes_preserved_losslessly(self, core):
        """Test B: Timezone-aware datetime columns preserved across cache eviction without coercion."""
        reg = core.DataFrameRegistry(capacity=1)
        original_df = pd.DataFrame({
            "utc_time": pd.date_range("2026-01-01", periods=3, tz="UTC"),
            "ny_time": pd.date_range("2026-06-01", periods=3, tz="America/New_York"),
        })

        df_id = reg.register_dataframe(original_df, "tz_dates")
        assert df_id == "tz_dates"

        # Evict
        reg.register_dataframe(pd.DataFrame({"evict": [1]}), "evictor")
        assert reg.registry["tz_dates"]["df"] is None

        # Reload
        reloaded = reg.get_dataframe("tz_dates", load_if_not_exists=True)
        assert reloaded is not None
        pd.testing.assert_frame_equal(original_df, reloaded, check_dtype=True)
        assert reloaded["utc_time"].dtype == original_df["utc_time"].dtype
        assert str(reloaded["utc_time"].dt.tz) == "UTC"
        assert reloaded["ny_time"].dtype == original_df["ny_time"].dtype
        assert "America/New_York" in str(reloaded["ny_time"].dt.tz)

    def test_nontrivial_index_structures_preserved_losslessly(self, core):
        """Test C: Custom index names and MultiIndexes preserved across cache eviction."""
        reg = core.DataFrameRegistry(capacity=1)

        # 1. Custom named index
        custom_idx_df = pd.DataFrame(
            {"val": [100, 200]},
            index=pd.Index(["row_a", "row_b"], name="entity_id"),
        )
        reg.register_dataframe(custom_idx_df, "named_idx")
        reg.register_dataframe(pd.DataFrame({"evict": [1]}), "evictor1")
        assert reg.registry["named_idx"]["df"] is None

        reloaded_idx = reg.get_dataframe("named_idx", load_if_not_exists=True)
        assert reloaded_idx is not None
        pd.testing.assert_frame_equal(custom_idx_df, reloaded_idx, check_index_type=True, check_names=True)

        # 2. MultiIndex
        tuples = [("region_1", 2024), ("region_1", 2025), ("region_2", 2024)]
        multi_idx = pd.MultiIndex.from_tuples(tuples, names=["region", "year"])
        multi_df = pd.DataFrame({"metric": [1.5, 2.5, 3.5]}, index=multi_idx)

        reg.register_dataframe(multi_df, "multi_idx")
        reg.register_dataframe(pd.DataFrame({"evict": [2]}), "evictor2")
        assert reg.registry["multi_idx"]["df"] is None

        reloaded_multi = reg.get_dataframe("multi_idx", load_if_not_exists=True)
        assert reloaded_multi is not None
        pd.testing.assert_frame_equal(multi_df, reloaded_multi, check_index_type=True, check_names=True)

    def test_nullable_and_categorical_dtypes_preserved_losslessly(self, core):
        """Test D: Nullable extension dtypes (Int64, boolean) and category dtypes preserved."""
        reg = core.DataFrameRegistry(capacity=1)
        original_df = pd.DataFrame({
            "nullable_int": pd.Series([10, None, 30], dtype="Int64"),
            "nullable_bool": pd.Series([True, False, None], dtype="boolean"),
            "categorical_col": pd.Series(["alpha", "beta", "alpha"], dtype="category"),
        })

        reg.register_dataframe(original_df, "ext_dtypes")
        reg.register_dataframe(pd.DataFrame({"evict": [1]}), "evictor")
        assert reg.registry["ext_dtypes"]["df"] is None

        reloaded = reg.get_dataframe("ext_dtypes", load_if_not_exists=True)
        assert reloaded is not None
        pd.testing.assert_frame_equal(original_df, reloaded, check_dtype=True, check_categorical=True)

    def test_initial_explicit_source_with_transformed_schema(self, core, tmp_path):
        """Test E: Initial registration with explicit CSV source but transformed schema preserves both."""
        src_file = tmp_path / "raw_user_input.csv"
        src_file.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        original_bytes = src_file.read_bytes()

        reg = core.DataFrameRegistry(capacity=1)
        # Working DataFrame has rich schema with tz-aware timestamp and int column
        transformed_df = pd.DataFrame(
            [[10, pd.Timestamp("2026-03-15 12:00:00+0000", tz="UTC")]],
            columns=[0, "tz_col"],
        )

        reg.register_dataframe(transformed_df, "trans_ds", raw_path=str(src_file))

        # External source file must remain byte-for-byte untouched
        assert src_file.read_bytes() == original_bytes

        # Evict
        reg.register_dataframe(pd.DataFrame({"evict": [1]}), "evictor")
        assert reg.registry["trans_ds"]["df"] is None

        # Reload must come from lossless backing snapshot, NOT lossy/stale raw_path
        reloaded = reg.get_dataframe("trans_ds", load_if_not_exists=True)
        assert reloaded is not None
        pd.testing.assert_frame_equal(transformed_df, reloaded, check_dtype=True, check_names=True)
        assert src_file.read_bytes() == original_bytes

    def test_reregistration_lossless_schema_update(self, core):
        """Test F: Updating registered DataFrame updates backing snapshot with new rich schema."""
        reg = core.DataFrameRegistry(capacity=1)
        v1 = pd.DataFrame({"val": [1, 2]})
        reg.register_dataframe(v1, "upd_key")

        # Update with rich schema
        v2 = pd.DataFrame(
            [[999, "hello"]],
            columns=[0, "str_0"],
            index=pd.Index(["row_1"], name="custom_row"),
        )
        reg.register_dataframe(v2, "upd_key")

        # Evict
        reg.register_dataframe(pd.DataFrame({"evict": [1]}), "evictor")
        assert reg.registry["upd_key"]["df"] is None

        # Reload must yield v2 losslessly
        reloaded = reg.get_dataframe("upd_key", load_if_not_exists=True)
        assert reloaded is not None
        pd.testing.assert_frame_equal(v2, reloaded, check_dtype=True, check_names=True)

    def test_legacy_validation_maintains_lossless_schema_on_evicted_frame(self, core):
        """Test G: validate_dataframe_exists restores evicted frame with full fidelity."""
        reg = core.DataFrameRegistry(capacity=1)
        with core.override_global_registry(reg):
            df_rich = pd.DataFrame(
                [[42, pd.Timestamp("2026-01-01T00:00:00Z")]],
                columns=[0, "utc_col"],
            )
            reg.register_dataframe(df_rich, "val_ds")

            # Evict
            reg.register_dataframe(pd.DataFrame({"evict": [1]}), "evictor")
            assert reg.registry["val_ds"]["df"] is None

            # Validate exists
            assert core.validate_dataframe_exists("val_ds") is True

            # Working frame restored
            working = reg.get_dataframe("val_ds")
            assert working is not None
            pd.testing.assert_frame_equal(df_rich, working, check_dtype=True, check_names=True)

    def test_atomic_write_failure_simulation_on_pkl_backing_path(self, core, monkeypatch):
        """Test H: Write failure during snapshot creation leaves temporary files cleaned up."""
        reg = core.DataFrameRegistry(capacity=5)
        df = pd.DataFrame({"col": [1, 2, 3]})

        # Register v1 successfully
        reg.register_dataframe(df, "atomic_df")
        backing_path = Path(reg.get_backing_path_from_id("atomic_df"))
        assert backing_path.exists()
        original_mtime = backing_path.stat().st_mtime_ns

        # Simulate failure during atomic os.replace on update
        def mock_replace(src, dst):
            raise OSError("Simulated disk replacement failure")

        monkeypatch.setattr(os, "replace", mock_replace)

        v2 = pd.DataFrame({"col": [99, 100]})
        result = reg.register_dataframe(v2, "atomic_df")
        assert result is None

        # Existing snapshot remains unharmed and uncorrupted
        assert backing_path.exists()
        assert backing_path.stat().st_mtime_ns == original_mtime
        backed_df = pd.read_pickle(backing_path)
        pd.testing.assert_frame_equal(df, backed_df)

        # No lingering temporary files in data_dir
        tmp_files = [p for p in reg.data_dir.iterdir() if p.name.startswith(".tmp_")]
        assert not tmp_files, f"Lingering tmp files found: {tmp_files}"

    def test_format_compatibility_for_external_source_paths(self, core, tmp_path):
        """Test I: External non-pickle sources (.csv, .json, optional .parquet) continue to load seamlessly."""
        reg = core.DataFrameRegistry(capacity=1)

        # CSV source
        csv_file = tmp_path / "ext.csv"
        pd.DataFrame({"x": [1]}).to_csv(csv_file, index=False)
        reg.register_dataframe(df=None, df_id="from_csv", raw_path=str(csv_file))

        # JSON source
        json_file = tmp_path / "ext.json"
        pd.DataFrame({"z": [3]}).to_json(json_file, orient="records")
        reg.register_dataframe(df=None, df_id="from_json", raw_path=str(json_file))

        assert reg.get_dataframe("from_csv", load_if_not_exists=True)["x"].iloc[0] == 1
        assert reg.get_dataframe("from_json", load_if_not_exists=True)["z"].iloc[0] == 3

        # Parquet source (optional dependency pyarrow/fastparquet)
        try:
            import pyarrow  # noqa: F401
            pq_file = tmp_path / "ext.parquet"
            pd.DataFrame({"y": [2]}).to_parquet(pq_file, index=False)
            reg.register_dataframe(df=None, df_id="from_pq", raw_path=str(pq_file))
            assert reg.get_dataframe("from_pq", load_if_not_exists=True)["y"].iloc[0] == 2
        except ImportError:
            pass

    def test_backing_snapshot_extension_is_pkl_and_strictly_contained(self, core):
        """Test J: Generated backing paths always end with .pkl and are contained in data_dir."""
        reg = core.DataFrameRegistry(capacity=5)
        df = pd.DataFrame({"a": [1]})
        reg.register_dataframe(df, "check_ext")

        backing_path = Path(reg.get_backing_path_from_id("check_ext"))
        assert backing_path.suffix == ".pkl"
        assert backing_path.is_relative_to(reg.data_dir.resolve())
        assert backing_path.exists()
