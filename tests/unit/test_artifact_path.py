"""
Unit tests for _resolve_artifact_path and _get_artifacts_base.

Validates:
- Normal file paths resolve within base dir
- Path traversal attacks raise ValueError
- Absolute paths outside base raise ValueError
- Subdirectory support creates nested dirs
- empty/invalid file_name raises ValueError
"""
import os
import types
import pytest
from pathlib import Path as PathlibPath

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def core():
    try:
        import idd_core
        return idd_core
    except ImportError:
        pytest.skip("idd_core.py not available")


@pytest.fixture()
def tmp_base(tmp_path):
    """A fresh temp directory to use as the artifacts base."""
    base = tmp_path / "artifacts"
    base.mkdir()
    return base


def make_config(artifacts_dir=None):
    """Build a minimal RunnableConfig-like object."""
    configurable = {}
    if artifacts_dir is not None:
        runtime = types.SimpleNamespace(artifacts_dir=str(artifacts_dir))
        configurable["runtime"] = runtime
    return types.SimpleNamespace(configurable=configurable)


class TestResolveArtifactPath:
    def test_simple_filename_within_base(self, core, tmp_base):
        """A plain filename resolves to base/filename."""
        cfg = make_config(tmp_base)
        p = core._resolve_artifact_path("output.csv", config=cfg)
        assert p == tmp_base / "output.csv"
        assert p.parent.exists()

    def test_path_traversal_double_dot_raises(self, core, tmp_base):
        """../../etc/passwd must raise ValueError."""
        cfg = make_config(tmp_base)
        with pytest.raises(ValueError, match="(?i)outside|refused|artifacts"):
            core._resolve_artifact_path("../../etc/passwd", config=cfg)

    def test_path_traversal_sibling_dir_raises(self, core, tmp_base):
        """../sibling/file must raise ValueError."""
        cfg = make_config(tmp_base)
        with pytest.raises(ValueError, match="(?i)outside|refused|artifacts"):
            core._resolve_artifact_path("../sibling/file.txt", config=cfg)

    def test_absolute_path_outside_base_raises(self, core, tmp_base):
        """An absolute path outside the artifacts base must raise ValueError."""
        cfg = make_config(tmp_base)
        outside = str(tmp_base.parent / "outside.txt")
        with pytest.raises(ValueError, match="(?i)outside|refused|artifacts"):
            core._resolve_artifact_path(outside, config=cfg)

    def test_subdir_creates_nested_path(self, core, tmp_base):
        """Providing subdir= creates the nested directory and resolves within it."""
        cfg = make_config(tmp_base)
        p = core._resolve_artifact_path("plot.png", config=cfg, subdir="charts")
        assert p == (tmp_base / "charts" / "plot.png").resolve()
        assert p.parent.exists()

    def test_nested_filename_within_base_ok(self, core, tmp_base):
        """A nested relative path that stays inside base is accepted."""
        cfg = make_config(tmp_base)
        p = core._resolve_artifact_path("subdir/report.html", config=cfg)
        assert tmp_base in p.parents
        assert p.parent.exists()

    def test_empty_filename_raises(self, core, tmp_base):
        """Empty string must raise ValueError."""
        cfg = make_config(tmp_base)
        with pytest.raises(ValueError):
            core._resolve_artifact_path("", config=cfg)

    def test_none_filename_raises(self, core, tmp_base):
        """None must raise ValueError."""
        cfg = make_config(tmp_base)
        with pytest.raises((ValueError, TypeError)):
            core._resolve_artifact_path(None, config=cfg)

    def test_create_parents_true_creates_dirs(self, core, tmp_base):
        """create_parents=True creates parent dirs automatically."""
        cfg = make_config(tmp_base)
        p = core._resolve_artifact_path("deep/nested/file.txt", config=cfg, create_parents=True)
        assert p.parent.exists()

    def test_create_parents_false_does_not_create_dirs(self, core, tmp_base):
        """create_parents=False leaves parent dirs uncreated."""
        cfg = make_config(tmp_base)
        p = core._resolve_artifact_path(
            "noncreated/dir/file.txt", config=cfg, create_parents=False
        )
        assert not p.parent.exists()

    def test_config_none_falls_back_to_working_directory(self, core):
        """With config=None, falls back to WORKING_DIRECTORY/artifacts."""
        p = core._resolve_artifact_path("fallback.csv", config=None)
        assert "artifacts" in str(p)
        assert p.parent.exists()

    def test_path_traversal_within_subdir_raises(self, core, tmp_base):
        """Path traversal via subdir escape must raise."""
        cfg = make_config(tmp_base)
        with pytest.raises(ValueError, match="(?i)outside|refused|artifacts"):
            core._resolve_artifact_path("../../escape.txt", config=cfg, subdir="charts")
