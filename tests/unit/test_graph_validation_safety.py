"""
tests/unit/test_graph_validation_safety.py — Safety regression tests verifying
that validate_graph.py strictly suppresses package-installation subprocess calls.

Enforces:
1. Extraction of Cell 4 (bootstrap installation cell) directly from the authoritative
   notebook IntelligentDataDetective_beta_v5_patched.ipynb.
2. Interception of subprocess.check_call to prove zero installer invocations occur
   when executed through the guarded validator under all caller environment flag
   variations:
   - absent (not set)
   - "" (empty string)
   - "0" (explicit disable)
   - "false" (explicit false)
   - "1" (explicit true)
3. Testing with both use_local_llm=True and use_local_llm=False.
4. Clean restoration of caller environment state on exit and on error.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import validate_graph


NOTEBOOK_PATH = Path("IntelligentDataDetective_beta_v5_patched.ipynb")


def _extract_cell_4_source() -> str:
    """Extract Cell 4 (dependency bootstrap cell) directly from the notebook."""
    assert NOTEBOOK_PATH.exists(), f"Target notebook not found: {NOTEBOOK_PATH}"
    nb_data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = nb_data["cells"]
    cell_4 = cells[4]
    source = cell_4.get("source", [])
    if isinstance(source, list):
        src_text = "".join(source)
    else:
        src_text = str(source)
    assert (
        "_skip_notebook_installs" in src_text
    ), "Cell 4 does not contain install skip logic"
    assert (
        "subprocess.check_call" in src_text
    ), "Cell 4 does not contain pip install calls"
    return src_text


@pytest.mark.parametrize(
    "env_val",
    [None, "", "0", "false", "1"],
)
@pytest.mark.parametrize(
    "use_local_llm",
    [True, False],
)
def test_installer_suppression_invariant(
    env_val: str | None,
    use_local_llm: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Verify that validate_graph's execution sandbox and harness strictly prevent
    pip / subprocess installer calls regardless of the caller's environment setting
    and local LLM flag.
    """
    # 1. Establish the caller environment
    if env_val is None:
        monkeypatch.delenv("IDD_SKIP_NOTEBOOK_INSTALLS", raising=False)
    else:
        monkeypatch.setenv("IDD_SKIP_NOTEBOOK_INSTALLS", env_val)

    prior_env = os.environ.get("IDD_SKIP_NOTEBOOK_INSTALLS")

    # 2. Mock subprocess.check_call in the host environment
    mock_subprocess_check_call = MagicMock()
    monkeypatch.setattr(subprocess, "check_call", mock_subprocess_check_call)

    # 3. Execute Cell 4 through the guarded validator harness
    with validate_graph._guard_installation_suppression():
        assert os.environ.get("IDD_SKIP_NOTEBOOK_INSTALLS") == "1"

        sandbox = validate_graph.build_sandbox()
        # Mock subprocess in sandbox if present
        if "subprocess" in sandbox:
            sandbox["subprocess"].check_call = mock_subprocess_check_call
        sandbox["use_local_llm"] = use_local_llm

        cell_4_source = _extract_cell_4_source()
        cleaned_source = validate_graph._strip_shell_and_magics(cell_4_source)
        results = validate_graph.exec_cells(
            [(4, cleaned_source)], sandbox, verbose=False, silence=True
        )

    # 4. Assert zero installer invocations occurred
    assert len(results) == 1
    assert results[0][1] is None, f"Cell 4 raised unexpected error: {results[0][1]}"
    assert mock_subprocess_check_call.call_count == 0, (
        f"Installer was invoked {mock_subprocess_check_call.call_count} times! "
        f"Calls: {mock_subprocess_check_call.call_args_list}"
    )

    # 5. Assert caller environment was cleanly restored
    assert os.environ.get("IDD_SKIP_NOTEBOOK_INSTALLS") == prior_env


def test_guard_restores_env_on_exception(monkeypatch: pytest.MonkeyPatch):
    """Verify that _guard_installation_suppression restores env even when an error occurs."""
    monkeypatch.setenv("IDD_SKIP_NOTEBOOK_INSTALLS", "original_value")

    with pytest.raises(RuntimeError, match="deliberate failure"):
        with validate_graph._guard_installation_suppression():
            assert os.environ["IDD_SKIP_NOTEBOOK_INSTALLS"] == "1"
            raise RuntimeError("deliberate failure")

    assert os.environ.get("IDD_SKIP_NOTEBOOK_INSTALLS") == "original_value"


def test_guard_cleans_up_when_previously_unset(monkeypatch: pytest.MonkeyPatch):
    """Verify that _guard_installation_suppression removes the env var if previously unset."""
    monkeypatch.delenv("IDD_SKIP_NOTEBOOK_INSTALLS", raising=False)

    with pytest.raises(RuntimeError, match="deliberate failure"):
        with validate_graph._guard_installation_suppression():
            assert os.environ["IDD_SKIP_NOTEBOOK_INSTALLS"] == "1"
            raise RuntimeError("deliberate failure")

    assert "IDD_SKIP_NOTEBOOK_INSTALLS" not in os.environ
