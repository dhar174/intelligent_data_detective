"""Strict execution uses real Jupyter with synthetic notebooks, never model calls."""

import asyncio
import copy
import json
import os
from pathlib import Path
import platform
import sys

import nbformat
import pytest

from tests.smoke.test_real_kernel_notebook import _scoped_environ

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("IDD_RUN_KERNEL_SMOKE") != "1",
        reason="provision a real kernel and set IDD_RUN_KERNEL_SMOKE=1 and IDD_KERNEL_NAME",
    ),
]

NO_OUTBOUND_SOURCE = """
import socket as _test_socket, ipaddress as _test_ip
_test_connect = _test_socket.socket.connect
def _test_local_connect(sock, address):
    if isinstance(address, tuple):
        host = address[0]
        try:
            local = _test_ip.ip_address(host).is_loopback
        except ValueError:
            local = host == 'localhost'
        if not local:
            raise AssertionError('OUTBOUND_NETWORK_FORBIDDEN')
    return _test_connect(sock, address)
_test_socket.socket.connect = _test_local_connect
"""


def proof_module():
    import idd_proof

    return idd_proof


def execute(
    tmp_path,
    sources,
    *,
    timeout=30,
    expected_python=None,
    observation_cells=None,
    setup=None,
):
    proof = proof_module()
    workspace, evidence_dir = proof.create_attempt(tmp_path / "attempt")
    # The same versioned observer is importable from the isolated kernel cwd.
    import shutil

    shutil.copy2(Path(proof.__file__), workspace / "idd_proof.py")
    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(s, metadata={"tags": ["raises-exception"]})
            for s in sources
        ]
    )
    # Test-only guard: allow Jupyter's local sockets, refuse outbound traffic.
    nb.cells[0].source = NO_OUTBOUND_SOURCE + nb.cells[0].source
    evidence = proof.Evidence(evidence_dir, "synthetic-no-key")
    if setup:
        setup(workspace, evidence)
    env = proof.child_environment(workspace, "sample_dirty", "none", os.environ)
    # Match the established harness's no-key environment; block external sockets
    # in the synthetic first cell without blocking Jupyter loopback sockets.
    original = copy.deepcopy([c.source for c in nb.cells])
    with _scoped_environ():
        try:
            asyncio.run(
                proof.execute_strict(
                    nb,
                    workspace=workspace,
                    evidence=evidence,
                    kernel=os.environ["IDD_KERNEL_NAME"],
                    expected_python=Path(expected_python or sys.executable).absolute(),
                    expected_version=platform.python_version(),
                    env=env,
                    timeout=timeout,
                    observation_cells=observation_cells or {},
                )
            )
        except Exception as exc:
            error = exc
        else:
            error = None
    assert original == [c.source for c in nb.cells]
    assert (evidence_dir / "executed.ipynb").is_file()
    return (
        error,
        json.loads((evidence_dir / "evidence.json").read_text()),
        workspace,
        evidence_dir,
    )


@pytest.mark.parametrize(
    "failure",
    [
        "raise RuntimeError('bootstrap-canary')",
        "import time; time.sleep(30)",
        "import os; os._exit(42)",
    ],
)
def test_strict_error_stops_before_sentinel_and_preserves_partial(tmp_path, failure):
    error, record, workspace, evidence_dir = execute(
        tmp_path, [failure, "open('MODEL_SENTINEL', 'w').write('forbidden')"], timeout=1
    )
    assert error is not None
    assert record["execution"]["status"] == "FAILED"
    assert record["first_failure"]["cell_index"] == 0
    assert not (workspace / "MODEL_SENTINEL").exists()
    saved = nbformat.read(evidence_dir / "executed.ipynb", as_version=4)
    assert saved.cells[1].execution_count is None


def test_child_interpreter_mismatch_stops_before_first_cell(tmp_path):
    error, record, workspace, _ = execute(
        tmp_path,
        ["open('MODEL_SENTINEL', 'w').write('forbidden')"],
        expected_python=tmp_path / "not-python",
    )
    assert error is not None
    assert not (workspace / "MODEL_SENTINEL").exists()
    assert record["observations"]["identity"]["status"] == "MISMATCH"


def test_synthetic_success_is_never_cp5_pass(tmp_path):
    error, record, _, _ = execute(tmp_path, ["print('synthetic only')"])
    assert error is None
    assert record["execution"]["status"] == "COMPLETED"
    assert record["execution_kind"] == "synthetic-no-key"
    assert record["cp5_pass"] is False
    assert record["artifact_inspection"] == "PENDING_INDEPENDENT_REVIEW"
    assert record["observations"]["identity"]["sys_executable"] == str(
        Path(sys.executable).absolute()
    )


def test_secret_output_is_restricted_not_published_provenance(tmp_path):
    error, record, _, evidence_dir = execute(
        tmp_path,
        [
            "print('sk-SYNTHETIC-secret-canary'); raise ValueError('sk-SYNTHETIC-secret-canary')"
        ],
    )
    assert error is not None
    assert "secret-canary" not in json.dumps(record)
    assert "secret-canary" in (evidence_dir / "console.log").read_text()


def test_missing_graph_attestation_prevents_later_cell(tmp_path):
    error, record, workspace, _ = execute(
        tmp_path,
        ["pass", "open('MODEL_SENTINEL', 'w').write('forbidden')"],
        observation_cells={0: "graph"},
    )
    assert error is not None
    assert record["observations"]["graph"]["status"] != "VERIFIED"
    assert not (workspace / "MODEL_SENTINEL").exists()


def test_graph_replacement_is_detected_by_read_only_observer(tmp_path):
    root = Path(__file__).resolve().parents[2]
    setup = f"""
import sys, sqlite3
from pathlib import Path
sys.path.insert(0, {str(root)!r})
from idd_graph import build_graph, Router
from langgraph.checkpoint.sqlite import SqliteSaver
_cp_conn = sqlite3.connect('checkpoints.sqlite', check_same_thread=False)
_cp_conn.execute('CREATE TABLE sentinel (value INTEGER)')
_cp_conn.execute('INSERT INTO sentinel VALUES (42)')
_cp_conn.commit()
data_detective_graph = build_graph(structural_only=True, checkpointer=SqliteSaver(_cp_conn))
run_config = {{'configurable': {{'thread_id': 'synthetic-observer-state'}}}}
data_detective_graph.update_state(run_config, {{'user_prompt': 'synthetic only'}}, as_node='initial_analysis')
before_state = data_detective_graph.get_state(run_config)
before_config = __import__('copy').deepcopy(run_config)
before_changes = _cp_conn.total_changes
before_nodes = tuple(data_detective_graph.builder.nodes)
"""
    verify = """
assert before_changes == _cp_conn.total_changes
assert before_nodes == tuple(data_detective_graph.builder.nodes)
assert before_config == run_config
assert before_state == data_detective_graph.get_state(run_config)
assert _cp_conn.execute('SELECT value FROM sentinel').fetchall() == [(42,)]
data_detective_graph = object()
"""
    error, record, workspace, _ = execute(
        tmp_path,
        [setup, verify, "open('MODEL_SENTINEL', 'w').write('forbidden')"],
        observation_cells={0: "graph", 1: "late"},
    )
    assert error is not None
    assert record["observations"]["graph"]["status"] == "VERIFIED"
    assert record["observations"]["late"]["status"] == "MISMATCH"
    assert not (workspace / "MODEL_SENTINEL").exists()


def test_fresh_log_failure_never_starts_kernel_or_sentinel(tmp_path):
    def setup(workspace, evidence):
        (workspace / "notebook_run_log.txt").mkdir()

    error, record, workspace, _ = execute(
        tmp_path, ["open('MODEL_SENTINEL', 'w').write('forbidden')"], setup=setup
    )
    assert error is not None
    assert record["observations"]["identity"]["status"] == "NOT_REACHED"
    assert not (workspace / "MODEL_SENTINEL").exists()


def test_runner_observer_exception_retains_original_failure(tmp_path, monkeypatch):
    proof = proof_module()

    async def broken(*args, **kwargs):
        raise RuntimeError("synthetic secret-canary")

    monkeypatch.setattr(proof, "observe", broken)
    error, record, workspace, _ = execute(
        tmp_path, ["open('MODEL_SENTINEL', 'w').write('forbidden')"]
    )
    assert isinstance(error, RuntimeError)
    assert record["first_failure"]["type"] == "RuntimeError"
    assert "secret-canary" not in json.dumps(record)
    assert not (workspace / "MODEL_SENTINEL").exists()


def test_snapshot_mutation_after_cell_is_not_verified(tmp_path):
    def setup(workspace, evidence):
        evidence.data["sources"] = {
            "idd_proof.py": proof_module().hashes(
                (workspace / "idd_proof.py").read_bytes()
            )
        }

    error, record, _, _ = execute(
        tmp_path,
        ["from pathlib import Path; Path('idd_proof.py').write_text('changed')"],
        setup=setup,
    )
    assert error is not None
    assert record["provenance"] != "VERIFIED"


def test_real_bootstrap_failure_stops_sentinel_with_original_cell_sources(tmp_path):
    from tests.unit.test_core_parity_guard import load_patched_notebook

    notebook = load_patched_notebook()
    original = ["".join(notebook["cells"][index]["source"]) for index in (1, 4)]
    error, record, workspace, directory = execute(
        tmp_path, ["pass", *original, "open('MODEL_SENTINEL', 'w').write('forbidden')"]
    )
    assert (
        error is not None
    )  # the fresh synthetic workspace deliberately has no manifest
    assert not (workspace / "MODEL_SENTINEL").exists()
    saved = nbformat.read(directory / "executed.ipynb", as_version=4)
    assert [cell.source for cell in saved.cells[1:3]] == original
    assert record["first_failure"]["cell_index"] == 2


def test_secondary_save_failure_does_not_replace_cell_error(tmp_path, monkeypatch):
    proof = proof_module()
    save = proof.Evidence.save

    def failed_save(self):
        if self.data["first_failure"]:
            raise OSError("synthetic evidence failure")
        return save(self)

    monkeypatch.setattr(proof.Evidence, "save", failed_save)
    error, _, workspace, directory = execute(
        tmp_path,
        [
            "raise ValueError('original cell failure')",
            "open('MODEL_SENTINEL', 'w').write('forbidden')",
        ],
    )
    from nbclient.exceptions import CellExecutionError

    assert isinstance(error, CellExecutionError)
    assert "original cell failure" in str(error)
    assert (directory / "executed.ipynb").is_file()
    assert not (workspace / "MODEL_SENTINEL").exists()


def test_preparation_cli_verifies_source_and_kernel_without_loading_credentials(
    tmp_path,
):
    import shutil
    import subprocess

    proof = proof_module()
    source = tmp_path / "source"
    source.mkdir()
    original = Path(__file__).resolve().parents[2]
    for name in proof.SOURCE_FILES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / name, target)

    def git(*args):
        return (
            subprocess.check_output(
                ["git", "-C", str(source), *args], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

    git("init")
    git("add", ".")
    git(
        "-c",
        "user.name=Synthetic test",
        "-c",
        "user.email=synthetic@example.invalid",
        "commit",
        "-m",
        "synthetic source snapshot",
    )
    sha = git("rev-parse", "HEAD")
    # These must neither be read nor copied into the runtime snapshot.
    (source / ".env").write_text(
        "OPENAI_API_KEY=sk-SYNTHETIC-secret-canary\nLANGSMITH_TRACING=true\n"
    )
    env = proof.child_environment(tmp_path, "sample_dirty", "none", os.environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)  # parent CLI must protect the source too
    process = subprocess.run(
        [
            sys.executable,
            str(source / "run_notebook_live.py"),
            "--proof",
            "--prepare-only",
            "--attempt-dir",
            str(tmp_path / "cli-attempt"),
            "--source-sha",
            sha,
            "--kernel-name",
            os.environ["IDD_KERNEL_NAME"],
            "--expected-python",
            str(Path(sys.executable).absolute()),
            "--expected-python-version",
            platform.python_version(),
            "--scenario",
            "sample_dirty",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    record = json.loads(
        (tmp_path / "cli-attempt" / "evidence" / "evidence.json").read_text()
    )
    assert record["execution"]["status"] == "NOT_EXECUTED_PREPARATION_ONLY"
    assert record["source_sha"] == sha
    assert len(record["sources"]) == len(proof.SOURCE_FILES)
    assert "secret-canary" not in json.dumps(record)
    assert not (tmp_path / "cli-attempt" / "workspace" / ".env").exists()
    assert record["observations"]["graph"]["status"] == "NOT_REACHED"
    assert not (source / "__pycache__").exists()


def test_console_close_failure_does_not_hide_cell_error(tmp_path, monkeypatch):
    original_open = Path.open

    class BadClose:
        def __init__(self, stream):
            self.stream = stream

        def write(self, text):
            return self.stream.write(text)

        def flush(self):
            return self.stream.flush()

        def close(self):
            self.stream.close()
            raise OSError("synthetic close failure")

    def open_path(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return (
            BadClose(stream) if path.name == "console.log" and mode == "x" else stream
        )

    monkeypatch.setattr(Path, "open", open_path)
    error, record, _, _ = execute(
        tmp_path, ["raise ValueError('original cell failure')"]
    )
    from nbclient.exceptions import CellExecutionError

    assert isinstance(error, CellExecutionError)
    assert record["first_failure"]["type"] == "CellExecutionError"
    assert record["execution"]["status"] == "FAILED"


def test_final_notebook_save_failure_cannot_leave_completed_status(
    tmp_path, monkeypatch
):
    write = nbformat.write
    calls = []

    def failed_final_save(*args, **kwargs):
        calls.append(True)
        if len(calls) > 1:
            raise OSError("synthetic final save failure")
        return write(*args, **kwargs)

    monkeypatch.setattr(nbformat, "write", failed_final_save)
    error, record, _, _ = execute(tmp_path, ["print('synthetic only')"])
    assert isinstance(error, OSError)
    assert record["execution"]["status"] == "FAILED"
    assert record["evidence_completeness"] == "SAVE_FAILED"


def test_required_stage_not_reached_cannot_be_verified(tmp_path):
    error, record, _, _ = execute(
        tmp_path, ["print('synthetic only')"], observation_cells={99: "late"}
    )
    assert error is not None
    assert record["provenance"] != "VERIFIED"
    assert record["observations"]["late"]["status"] == "NOT_REACHED"


def test_canonical_observer_matches_actual_module_import_order(tmp_path):
    from tests.smoke.test_real_kernel_notebook import execute_cells_in_real_kernel

    root = Path(__file__).resolve().parents[2]
    probe = f"""
import idd_proof
from types import SimpleNamespace
from pathlib import Path
value = idd_proof.observe_namespace(globals(), 'canonical')
evidence = SimpleNamespace(data={{'sources': {{}}}})
idd_proof.verify_observation('canonical', value, Path({str(root)!r}), evidence, None, None)
assert set(value['modules']) == set(idd_proof.CANONICAL)
assert all(info['path'] and info['hashes'] for info in value['modules'].values())
print('__IDD_KERNEL_PROBE_JSON__:' + __import__('json').dumps({{'status': 'verified'}}))
"""
    _, probes, _ = execute_cells_in_real_kernel(
        [
            ("deny_outbound", NO_OUTBOUND_SOURCE),
            1,
            4,
            7,
            10,
            16,
            19,
            22,
            28,
            31,
            32,
            ("observer_import_order", probe),
        ],
        work_dir=tmp_path / "work",
    )
    assert probes["observer_import_order"]["status"] == "verified"
