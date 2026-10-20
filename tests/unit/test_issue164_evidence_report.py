"""
Issue #164 evidence-report integrity checks (Codex review P2-4).

The versioned evidence report `docs/evidence/issue-164-runtime-matrix.md` embeds a
machine-readable ```json issue164-evidence``` block. These tests mechanically verify it
against the *immutable* artifacts at the recorded tested implementation SHA (via
`git show <sha>:<path>`), so the report cannot drift from what was actually tested:

1. Every recorded artifact digest equals SHA-256 of the committed (LF) git blob.
2. The recorded real-kernel test inventory equals the test functions defined in the
   smoke files at the tested SHA (derived from the AST, not prose).
3. Every recorded negative-control mutation id appears in the smoke file at the tested SHA.
4. Every cited CI run records a checkout that is the tested SHA (or a PR merge commit
   whose head parent is the tested SHA), and the report states its hash convention.

Requires full git history for the tested SHA (CI checks out with `fetch-depth: 0`).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = REPO_ROOT / "docs" / "evidence" / "issue-164-runtime-matrix.md"
BLOCK_RE = re.compile(r"```json issue164-evidence\s*\n(.*?)\n```", re.DOTALL)


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout


@pytest.fixture(scope="module")
def evidence() -> Dict[str, Any]:
    text = EVIDENCE_PATH.read_text(encoding="utf-8")
    match = BLOCK_RE.search(text)
    assert match, "Evidence report is missing its ```json issue164-evidence``` block"
    return json.loads(match.group(1))


@pytest.fixture(scope="module")
def tested_sha(evidence: Dict[str, Any]) -> str:
    sha = evidence["tested_implementation_sha"]
    assert re.fullmatch(r"[0-9a-f]{40}", sha), sha
    try:
        kind = _git("cat-file", "-t", sha).decode().strip()
    except subprocess.CalledProcessError as exc:  # pragma: no cover - environment issue
        pytest.fail(
            f"Tested SHA {sha} is not available locally (shallow clone?). "
            f"Fetch full history: {exc.stderr.decode(errors='replace')}"
        )
    assert kind == "commit", kind
    return sha


def _test_functions(source: str) -> List[str]:
    tree = ast.parse(source)
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    names.append(f"{node.name}::{item.name}")
    return sorted(names)


def test_hash_convention_is_declared(evidence: Dict[str, Any]) -> None:
    assert evidence["digest_convention"] == (
        "sha256 of committed git blob (LF line endings); "
        "Windows CRLF working copies must be normalized CRLF->LF before hashing"
    )


def test_recorded_digests_match_committed_blobs(
    evidence: Dict[str, Any], tested_sha: str
) -> None:
    digests = evidence["artifact_sha256"]
    assert set(digests) >= {
        "IntelligentDataDetective_beta_v5_patched.ipynb",
        "idd_manifest.json",
        "idd_bootstrap.py",
        "_patch_notebook.py",
        "tests/smoke/test_real_kernel_notebook.py",
    }
    for rel_path, recorded in digests.items():
        blob = _git("show", f"{tested_sha}:{rel_path}")
        assert hashlib.sha256(blob).hexdigest() == recorded, rel_path


def test_recorded_kernel_test_inventory_matches_executed_tests(
    evidence: Dict[str, Any], tested_sha: str
) -> None:
    inventory = evidence["kernel_test_inventory"]
    for rel_path, recorded in inventory.items():
        source = _git("show", f"{tested_sha}:{rel_path}").decode("utf-8")
        assert sorted(recorded) == _test_functions(source), rel_path


def test_recorded_negative_controls_exist_in_tested_smoke_file(
    evidence: Dict[str, Any], tested_sha: str
) -> None:
    source = _git(
        "show", f"{tested_sha}:tests/smoke/test_real_kernel_notebook.py"
    ).decode("utf-8")
    defined = set(_test_functions(source))
    controls = evidence["negative_controls"]
    assert controls, "No negative controls recorded"
    for control in controls:
        assert control["test"] in defined, control["test"]
        assert f'"{control["mutation_id"]}"' in source, control["mutation_id"]


def test_cited_runs_checkout_the_tested_sha(
    evidence: Dict[str, Any], tested_sha: str
) -> None:
    runs = evidence["ci_runs"]
    assert {r["platform"] for r in runs} >= {"ubuntu-latest", "windows-latest"}
    for run in runs:
        assert re.fullmatch(r"\d+", str(run["run_id"])), run
        assert run["head_sha"] == tested_sha, run
        assert run["conclusion"] == "success", run
        checkout = run["checkout_sha"]
        if checkout != tested_sha:
            # PR merge checkout: the refs/pull/N/merge object is ephemeral and is not
            # fetched by later checkouts, so verify the parents recorded from the run log
            # line `checkout <sha> parents <base> <head>`.
            parents = run["checkout_parents"]
            assert all(re.fullmatch(r"[0-9a-f]{40}", p) for p in parents), run
            assert parents[-1] == tested_sha, run
