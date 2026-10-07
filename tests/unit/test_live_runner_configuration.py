import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import run_notebook_live as runner


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("override", [None, "IntelligentDataDetective_beta_v5.ipynb"])
def test_runner_selection_in_fresh_process(override):
    env = os.environ.copy()
    env.pop("IDD_NOTEBOOK", None)
    if override is not None:
        env["IDD_NOTEBOOK"] = override
    result = subprocess.run(
        [sys.executable, "-c", "import run_notebook_live as r; print(r.NOTEBOOK_PATH)"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
    )
    expected = override or "IntelligentDataDetective_beta_v5_patched.ipynb"
    assert result.stdout.strip() == str(REPO_ROOT / expected)


def test_runner_missing_selection_fails_before_kernel_in_fresh_process(tmp_path):
    missing = tmp_path / "missing.ipynb"
    env = {**os.environ, "IDD_NOTEBOOK": str(missing)}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_notebook_live.py")],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert f"Selected notebook: {missing}" in result.stdout
    assert f"Notebook not found: {missing}" in result.stdout
    assert "Probing scientific stack" not in result.stdout


@pytest.mark.parametrize("damage", [
    "no_output", "invalid_json", "empty_modules", "missing_module", "invalid_ok",
    "not_object", "failed_import", "wrong_flag", "wrong_normalization", "execution_failure",
])
def test_scientific_probe_rejects_incomplete_or_invalid_results(monkeypatch, damage, capsys):
    monkeypatch.setenv("IDD_SKIP_NOTEBOOK_INSTALLS", "1")
    payload = {
        "modules": {name: {"ok": True, "version": "test"} for name in runner.SCIENTIFIC_MODULES},
        "install_flag": "1",
        "installs_skipped": True,
    }
    if damage == "empty_modules":
        payload["modules"] = {}
    elif damage == "missing_module":
        payload["modules"].pop("scipy.stats")
    elif damage == "invalid_ok":
        payload["modules"]["scipy.stats"]["ok"] = "true"
    elif damage == "not_object":
        payload = []
    elif damage == "failed_import":
        payload["modules"]["scipy.stats"] = {"ok": False, "error": "ModuleNotFoundError: scipy"}
    elif damage == "wrong_flag":
        payload["install_flag"] = "0"
    elif damage == "wrong_normalization":
        payload["installs_skipped"] = False

    class Client:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook

        def execute(self):
            if damage == "execution_failure":
                raise RuntimeError("kernel unavailable")
            text = "SCIENTIFIC_STACK_PREFLIGHT=" + ("{" if damage == "invalid_json" else json.dumps(payload))
            self.notebook.cells[0]["outputs"] = [] if damage == "no_output" else [{"text": text}]

    monkeypatch.setitem(sys.modules, "nbformat", SimpleNamespace(v4=SimpleNamespace(
        new_notebook=lambda: SimpleNamespace(cells=[]),
        new_code_cell=lambda source: {"source": source, "outputs": []},
    )))
    monkeypatch.setitem(sys.modules, "nbclient", SimpleNamespace(NotebookClient=Client))
    assert not runner.probe_kernel_scientific_stack("python3")
    assert "ERR" in capsys.readouterr().out
