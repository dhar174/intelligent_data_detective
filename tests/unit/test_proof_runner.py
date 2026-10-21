"""CP5-A regressions: proof setup must fail before any analytical execution."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_notebook_live as runner


def proof_module():
    import idd_proof

    return idd_proof


def test_proof_interface_is_opt_in():
    assert hasattr(runner, "run_proof"), "strict proof execution is missing"


@pytest.mark.parametrize(
    "stale",
    [
        "checkpoints.sqlite",
        "checkpoints.sqlite-wal",
        "checkpoints.sqlite-shm",
        "_idd_resume.flag",
        "current_run_thread_id.txt",
        "IDD_results/IDD_run_old",
    ],
)
def test_existing_attempt_is_never_cleaned_or_reused(tmp_path, stale):
    proof = proof_module()
    attempt = tmp_path / "attempt"
    old = attempt / "workspace" / stale
    old.parent.mkdir(parents=True)
    old.write_text("preserve me")
    with pytest.raises(proof.ProofError, match="ATTEMPT_EXISTS"):
        proof.create_attempt(attempt)
    assert old.read_text() == "preserve me"


def test_attempt_symlink_is_not_followed(tmp_path):
    proof = proof_module()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(proof.ProofError):
        proof.create_attempt(link / "attempt")
    assert not list(outside.iterdir())


def test_child_environment_does_not_read_credentials_or_inherit_tracing(tmp_path):
    proof = proof_module()

    class NoSecrets(dict):
        def get(self, key, default=None):
            assert key not in {"OPENAI_API_KEY", "LANGSMITH_API_KEY"}
            return super().get(key, default)

    env = proof.child_environment(
        tmp_path,
        "retail_orders",
        "none",
        NoSecrets(
            PATH="/bin", LANGSMITH_TRACING="true", OPENAI_API_KEY="secret-canary"
        ),
    )
    assert "OPENAI_API_KEY" not in env
    assert env["LANGSMITH_TRACING"] == "false"
    assert env["LANGCHAIN_TRACING_V2"] == "false"
    assert env["IDD_SKIP_NOTEBOOK_INSTALLS"] == "1"


@pytest.mark.parametrize(
    "damage", ["missing", "relative", "mismatch", "env", "wrapper", "provisioner"]
)
def test_kernel_is_explicit_and_verified(tmp_path, damage):
    proof = proof_module()
    python = Path(__import__("sys").executable).resolve()
    spec = SimpleNamespace(
        argv=[str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        env={},
        resource_dir=str(tmp_path),
    )
    if damage == "relative":
        spec.argv[0] = "python"
    if damage == "mismatch":
        spec.argv[0] = str(tmp_path / "other-python")
    if damage == "env":
        spec.env = {"OPENAI_API_KEY": "secret-canary"}
    if damage == "wrapper":
        spec.argv[1:1] = ["-c", "pass"]
    if damage == "provisioner":
        spec.metadata = {"kernel_provisioner": {"provisioner_name": "remote-plugin"}}

    class Manager:
        def get_kernel_spec(self, name):
            if damage == "missing":
                raise KeyError(name)
            return spec

    with pytest.raises(proof.ProofError, match="KERNEL"):
        proof.verify_kernel("explicit", python, Manager())


def test_interpreter_verification_does_not_collapse_distinct_venvs(
    tmp_path, monkeypatch
):
    proof = proof_module()
    first, other = tmp_path / "venv-one" / "python", tmp_path / "venv-two" / "python"
    # Both environments can resolve to the same underlying binary on POSIX.
    monkeypatch.setattr(Path, "resolve", lambda self: Path("/same/python"))
    spec = SimpleNamespace(
        argv=[str(other), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        env={},
        resource_dir=str(tmp_path),
    )
    manager = SimpleNamespace(get_kernel_spec=lambda name: spec)
    with pytest.raises(proof.ProofError, match="KERNEL"):
        proof.verify_kernel("explicit", first, manager)


def test_evidence_initialization_failure_prevents_run(tmp_path, monkeypatch):
    proof = proof_module()

    def deny(*args, **kwargs):
        raise PermissionError("synthetic secret-canary")

    monkeypatch.setattr(proof, "atomic_json", deny)
    with pytest.raises(PermissionError):
        proof.Evidence(tmp_path, "synthetic-no-key")
    assert not (tmp_path / "notebook_run_log.txt").exists()


def test_first_failure_survives_secondary_evidence_failure(tmp_path, monkeypatch):
    proof = proof_module()
    evidence = proof.Evidence(tmp_path, "synthetic-no-key")
    original = RuntimeError("synthetic secret-canary")
    evidence.fail(original, "cell", 4)
    evidence.fail(OSError("later"), "save", None)
    data = json.loads((tmp_path / "evidence.json").read_text())
    assert data["first_failure"]["type"] == "RuntimeError"
    assert data["first_failure"]["cell_index"] == 4
    assert "secret-canary" not in json.dumps(data)
    assert data["cp5_pass"] is False


@pytest.mark.parametrize(
    "damage",
    ["multiple_runs", "wrong_run", "multiple_reports", "foreign_log", "outside_link"],
)
def test_association_refuses_ambiguity(associated_attempt, tmp_path, damage):
    proof, workspace, directory, observations, run, execution = associated_attempt
    run_id = observations["runtime"]["run_id"]
    reports = run / "artifacts" / run_id / "reports"
    (reports / "final_report.html").write_text("<p>test</p>")
    log = workspace / "notebook_run_log.txt"
    if damage == "multiple_runs":
        (run.parent / "IDD_run_other").mkdir()
    elif damage == "wrong_run":
        observations["runtime"]["run_id"] = "wrong"
    elif damage == "multiple_reports":
        (run / "other" / "reports").mkdir(parents=True)
    elif damage == "foreign_log":
        log.write_text("12:34:56 [INFO] run_default_id-20261019-1234-abcdefab FINAL\n")
    elif damage == "outside_link":
        try:
            (run / "outside").symlink_to(tmp_path / "outside")
        except OSError:
            pytest.skip("symlink creation unavailable")
    expected = {
        "multiple_runs": "ASSOCIATION_RUN_AMBIGUOUS",
        "wrong_run": "ASSOCIATION_RUN_AMBIGUOUS",
        "multiple_reports": "ASSOCIATION_REPORT_AMBIGUOUS",
        "foreign_log": "ASSOCIATION_FOREIGN_LOG",
        "outside_link": "ASSOCIATION_SYMLINK",
    }
    with pytest.raises(proof.ProofError, match=expected[damage]):
        proof.associate_outputs(workspace, observations, directory, execution=execution)


@pytest.mark.parametrize(
    "stale",
    [
        "checkpoints.sqlite-wal",
        "checkpoints.sqlite-shm",
        "checkpoints.sqlite",
        "current_run_thread_id.txt",
        "_idd_resume.flag",
        "IDD_results/IDD_run_old/final_report.html",
    ],
)
def test_pristine_check_refuses_state_created_during_setup(tmp_path, stale):
    proof = proof_module()
    path = tmp_path / stale
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("do not delete")
    with pytest.raises(proof.ProofError, match="STALE"):
        proof.assert_pristine(tmp_path)
    assert path.read_text() == "do not delete"


def test_actual_install_flag_is_part_of_kernel_identity(tmp_path):
    proof = proof_module()
    observed = proof.observe_namespace({}, "identity")
    assert "install_flag" in observed


def test_both_validator_failures_are_recorded_without_touching_mtime(tmp_path):
    proof = proof_module()
    workspace, evidence_dir = proof.create_attempt(tmp_path / "attempt")
    run = workspace / "IDD_results" / "IDD_run_test"
    run.mkdir(parents=True)
    (workspace / "notebook_run_log.txt").write_text(
        "12:34:56 FINAL viz=True report=True\n"
    )
    for name, exit_code in (
        ("validate_run.py", 1),
        ("validate_artifact_quality.py", 2),
    ):
        (workspace / name).write_text(
            f"import sys; print('synthetic failure'); sys.exit({exit_code})"
        )
    evidence = proof.Evidence(evidence_dir, "synthetic-no-key")
    evidence.data["scenario"] = "sample_dirty"
    (evidence_dir / "console.log").write_text("synthetic")
    before = run.stat().st_mtime_ns
    assert proof.run_validators(workspace, run, evidence) is False
    assert [v["exit_code"] for v in evidence.data["validators"]] == [1, 2]
    assert run.stat().st_mtime_ns == before
    assert all("--latest" not in v["command"] for v in evidence.data["validators"])


def test_frozen_source_inventory_rejects_disk_mutation(tmp_path):
    proof = proof_module()
    (tmp_path / "idd_tools.py").write_text("original\n")
    frozen = {"idd_tools.py": proof.hashes((tmp_path / "idd_tools.py").read_bytes())}
    (tmp_path / "idd_tools.py").write_text("changed\n")
    with pytest.raises(proof.ProofError, match="SOURCE"):
        proof.verify_snapshot(tmp_path, frozen)


def timing_observation(utc, **fields):
    return {
        "status": "VERIFIED",
        "utc": utc,
        "requested_utc": utc,
        "received_utc": utc,
        **fields,
    }


@pytest.fixture
def associated_attempt(tmp_path):
    proof = proof_module()
    workspace, directory = proof.create_attempt(tmp_path / "attempt")
    proof.Evidence(directory, "synthetic-no-key")
    run_id = "run_default_id-20261020-1234-01234567"
    run = workspace / "IDD_results" / ("IDD_run_" + run_id)
    (run / "artifacts" / run_id / "reports").mkdir(parents=True)
    (workspace / "notebook_run_log.txt").write_text(
        "12:34:56 [INFO] FINAL viz=True report=True\n"
    )
    (workspace / "current_run_thread_id.txt").write_text("thread-test")
    (directory / "console.log").write_text(f"Starting stream (thread_id={run_id})\n")
    observations = {
        "identity": timing_observation(
            "2026-10-20T12:34:51+00:00", utc_offset_seconds=0
        ),
        "runtime": timing_observation(
            "2026-10-20T12:34:55+00:00",
            run_id=run_id,
            thread_id="thread-test",
            displayed_stream_identifier=run_id,
        ),
        "stream": timing_observation("2026-10-20T12:34:57+00:00"),
        "late": timing_observation("2026-10-20T12:35:25+00:00"),
    }
    execution = {
        "status": "COMPLETED",
        "started_utc": "2026-10-20T12:34:50+00:00",
        "ended_utc": "2026-10-20T12:35:30+00:00",
        "last_cell_index": 98,
    }
    return proof, workspace, directory, observations, run, execution


def test_log_without_ids_requires_positive_console_and_thread_binding(
    associated_attempt,
):
    proof, workspace, directory, observations, run, execution = associated_attempt
    assert (
        proof.associate_outputs(workspace, observations, directory, execution=execution)
        == run
    )
    (directory / "console.log").write_text("unrelated old log")
    with pytest.raises(proof.ProofError, match="ASSOCIATION"):
        proof.associate_outputs(workspace, observations, directory, execution=execution)


def test_mixed_timestamp_log_is_rejected(associated_attempt):
    proof, workspace, directory, observations, _, execution = associated_attempt
    with (workspace / "notebook_run_log.txt").open("a") as log:
        log.write("01:02:03 [INFO] FINAL viz=True report=True\n")
    with pytest.raises(proof.ProofError, match="ASSOCIATION"):
        proof.associate_outputs(workspace, observations, directory, execution=execution)


def test_final_log_after_stream_is_associated_until_execution_ends(
    associated_attempt,
):
    proof, workspace, directory, observations, run, execution = associated_attempt
    # Cell 81 persists artifacts after the Cell 75 observation; Cell 83 logs FINAL.
    (workspace / "notebook_run_log.txt").write_text(
        "12:35:20 [INFO] FINAL viz=True report=True\n"
    )
    assert (
        proof.associate_outputs(workspace, observations, directory, execution=execution)
        == run
    )


@pytest.mark.parametrize("timestamp", ["12:34:44", "12:35:36"])
def test_logs_outside_actual_execution_are_rejected(associated_attempt, timestamp):
    proof, workspace, directory, observations, _, execution = associated_attempt
    with (workspace / "notebook_run_log.txt").open("a") as log:
        log.write(timestamp + " [INFO] other attempt\n")
    with pytest.raises(proof.ProofError, match="ASSOCIATION_MIXED_LOG_INTERVAL"):
        proof.associate_outputs(workspace, observations, directory, execution=execution)


@pytest.mark.parametrize("location", ["pipeline", "console"])
def test_foreign_run_ids_are_rejected_inside_valid_interval(
    associated_attempt, location
):
    proof, workspace, directory, observations, _, execution = associated_attempt
    path = (
        workspace / "notebook_run_log.txt"
        if location == "pipeline"
        else directory / "console.log"
    )
    with path.open("a") as log:
        log.write("12:35:20 [INFO] run_default_id-20261020-1234-abcdefab\n")
    with pytest.raises(proof.ProofError, match="ASSOCIATION_FOREIGN_LOG"):
        proof.associate_outputs(workspace, observations, directory, execution=execution)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_record",
        "missing_start",
        "missing_end",
        "invalid_end",
        "naive_end",
        "failed",
        "running",
        "end_before_start",
        "end_before_late_observation",
        "different_attempt",
        "ambiguous_day",
        "missing_clock_bracket",
        "unverified_observation",
        "reversed_clock_bracket",
        "child_clock_skew",
        "unordered_observations",
        "missing_stream",
    ],
)
def test_unreliable_execution_timing_fails_closed(associated_attempt, damage):
    proof, workspace, directory, observations, _, execution = associated_attempt
    if damage == "missing_record":
        execution = None
    elif damage in ("missing_start", "missing_end"):
        execution.pop("started_utc" if damage == "missing_start" else "ended_utc")
    elif damage == "invalid_end":
        execution["ended_utc"] = "not-a-timestamp"
    elif damage == "naive_end":
        execution["ended_utc"] = "2026-10-20T12:35:30"
    elif damage in ("failed", "running"):
        execution["status"] = damage.upper()
    elif damage == "end_before_start":
        execution["ended_utc"] = "2026-10-20T12:34:49+00:00"
    elif damage == "end_before_late_observation":
        execution["ended_utc"] = "2026-10-20T12:35:24+00:00"
    elif damage == "different_attempt":
        execution.update(
            started_utc="2026-10-19T12:34:50+00:00",
            ended_utc="2026-10-19T12:35:30+00:00",
        )
    elif damage == "ambiguous_day":
        execution["started_utc"] = "2026-10-19T12:34:50+00:00"
    elif damage == "missing_clock_bracket":
        observations["identity"].pop("requested_utc")
    elif damage == "unverified_observation":
        observations["stream"]["status"] = "UNAVAILABLE"
    elif damage == "reversed_clock_bracket":
        observations["stream"]["received_utc"] = "2026-10-20T12:34:56+00:00"
    elif damage == "child_clock_skew":
        observations["stream"]["utc"] = "2026-10-20T12:35:02.000001+00:00"
    elif damage == "unordered_observations":
        observations["stream"] = timing_observation("2026-10-20T12:34:54+00:00")
    elif damage == "missing_stream":
        observations.pop("stream")
    with pytest.raises(proof.ProofError, match="ASSOCIATION_EXECUTION_TIMING_INVALID"):
        proof.associate_outputs(workspace, observations, directory, execution=execution)


@pytest.mark.parametrize("offset", [0, -14400, 19800])
@pytest.mark.parametrize("foreign_time", [None, "23:59:44", "00:00:36"])
def test_association_across_local_midnight(associated_attempt, offset, foreign_time):
    proof, workspace, directory, observations, run, execution = associated_attempt
    zone = timezone(timedelta(seconds=offset))
    local_start = datetime(2026, 10, 20, 23, 59, 50, tzinfo=zone)
    shift = local_start - datetime.fromisoformat(execution["started_utc"])
    for record in (execution, *observations.values()):
        for key, value in record.items():
            if key == "utc" or key.endswith("_utc"):
                record[key] = (datetime.fromisoformat(value) + shift).isoformat()
    observations["identity"]["utc_offset_seconds"] = offset
    log = "23:59:55 [INFO] STAGE analyst START\n00:00:20 [INFO] FINAL report=True\n"
    if foreign_time:
        log += foreign_time + " [INFO] other attempt\n"
    (workspace / "notebook_run_log.txt").write_text(log)
    if foreign_time:
        with pytest.raises(proof.ProofError, match="ASSOCIATION_MIXED_LOG_INTERVAL"):
            proof.associate_outputs(
                workspace, observations, directory, execution=execution
            )
    else:
        assert (
            proof.associate_outputs(
                workspace, observations, directory, execution=execution
            )
            == run
        )


@pytest.mark.parametrize("timestamp", ["12:34:45", "12:35:35"])
def test_second_precision_and_clock_tolerance_are_bounded(
    associated_attempt, timestamp
):
    proof, workspace, directory, observations, run, execution = associated_attempt
    execution["started_utc"] = "2026-10-20T12:34:50.900000+00:00"
    execution["ended_utc"] = "2026-10-20T12:35:30.100000+00:00"
    (workspace / "notebook_run_log.txt").write_text(timestamp + " [INFO] FINAL\n")
    assert (
        proof.associate_outputs(workspace, observations, directory, execution=execution)
        == run
    )


@pytest.mark.parametrize("damage", ["missing_thread", "wrong_thread", "wrong_display"])
def test_delayed_finalization_preserves_identifier_bindings(associated_attempt, damage):
    proof, workspace, directory, observations, _, execution = associated_attempt
    path = workspace / "current_run_thread_id.txt"
    if damage == "missing_thread":
        path.unlink()
    elif damage == "wrong_thread":
        path.write_text("thread-other")
    else:
        observations["runtime"]["displayed_stream_identifier"] = "other"
    with pytest.raises(proof.ProofError, match="ASSOCIATION"):
        proof.associate_outputs(workspace, observations, directory, execution=execution)


def test_actual_model_and_embedding_settings_are_allowlisted():
    proof = proof_module()
    model = SimpleNamespace(
        model_name="gpt-5-nano",
        reasoning={"effort": "low", "summary": "auto"},
        model_kwargs={"text": {"verbosity": "low"}},
        max_retries=2,
        request_timeout=120,
        use_responses_api=True,
        output_version="responses/v1",
        api_key="secret-canary",
        client_configuration={"secret": "secret-canary"},
    )
    ns = {role: model for role in proof.MODEL_ROLES}
    ns.update(
        sample_prompt_text="secret-canary",
        RUNTIME=SimpleNamespace(_config={"recursion_limit": 400}),
        MEM_EMBEDDINGS=SimpleNamespace(
            model="text-embedding-3-small", api_key="secret-canary"
        ),
        DIM=1536,
    )
    record = proof.observe_namespace(ns, "request")
    assert record["embedding"]["model"] == "text-embedding-3-small"
    assert record["models"]["analyst_llm"]["settings"]["reasoning_effort"] == "low"
    assert "secret-canary" not in json.dumps(record)


def test_unrepresentable_configuration_is_not_given_a_lossy_hash():
    proof = proof_module()
    with pytest.raises(proof.ProofError, match="CONFIGURATION"):
        proof.config_hash({"opaque": object()})


def test_symlink_output_parent_cannot_bind_external_artifacts(
    associated_attempt, tmp_path
):
    import shutil

    proof, workspace, directory, observations, _, execution = associated_attempt
    external = tmp_path / "external-results"
    shutil.move(workspace / "IDD_results", external)
    try:
        (workspace / "IDD_results").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(proof.ProofError, match="ASSOCIATION"):
        proof.associate_outputs(workspace, observations, directory, execution=execution)


def test_validator_timeout_still_records_both_attempted_results(
    associated_attempt, monkeypatch
):
    import subprocess

    proof, workspace, directory, _, run, _ = associated_attempt
    evidence = proof.Evidence(directory, "synthetic-no-key")
    evidence.data["scenario"] = "sample_dirty"
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(
                command, 120, output=b"partial validator output"
            )
        return SimpleNamespace(returncode=1, stdout=b"failed quality", stderr=b"")

    monkeypatch.setattr(proof.subprocess, "run", invoke)
    assert proof.run_validators(workspace, run, evidence) is False
    assert len(calls) == 2
    assert evidence.data["validators"][0]["status"] == "TIMEOUT"
    assert evidence.data["validators"][0]["exit_code"] is None
