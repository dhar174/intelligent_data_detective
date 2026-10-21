"""CP5-A strict execution and evidence. No credential discovery or provider calls.

The observer reads a kernel namespace through Jupyter user_expressions. It never
executes graph nodes, restores state, updates configuration, or adds notebook cells.
Raw notebooks/logs are restricted evidence; evidence.json is an allowlisted index.
"""

from __future__ import annotations

import ast
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import uuid

NOTEBOOK = "IntelligentDataDetective_beta_v5_patched.ipynb"
CANONICAL = (
    "idd_bootstrap",
    "idd_models",
    "idd_registry",
    "idd_state",
    "idd_tools",
    "idd_graph",
    "idd_core",
)
SOURCE_FILES = (
    NOTEBOOK,
    "_patch_notebook.py",
    "idd_manifest.json",
    "run_notebook_live.py",
    "idd_proof.py",
    "validate_run.py",
    "validate_artifact_quality.py",
    "memory_config.yaml",
    "tests/trajectory/fixtures/sample_dirty.csv",
) + tuple(name + ".py" for name in CANONICAL)
DEPENDENCIES = (
    "nbclient",
    "nbformat",
    "jupyter-client",
    "ipykernel",
    "langgraph",
    "langgraph-checkpoint",
    "langgraph-checkpoint-sqlite",
    "langchain",
    "langchain-core",
    "langchain-openai",
    "langmem",
    "pydantic",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "openai",
    "xhtml2pdf",
    "pypdf",
    "pymupdf",
    "Pillow",
    "Markdown",
    "bleach",
)
OBSERVATION_CELLS = {
    4: "bootstrap",
    32: "canonical",
    48: "input",
    51: "request",
    60: "graph",
    72: "runtime",
    75: "stream",
    94: "migration",
    96: "late",
}
CLOCK_TOLERANCE = timedelta(seconds=5)
MODEL_ROLES = (
    "big_picture_llm",
    "router_llm",
    "reply_llm",
    "plan_llm",
    "replan_llm",
    "todo_llm",
    "progress_llm",
    "mid_substep_llm",
    "small_detail_llm",
    "low_reasoning_llm",
    "initial_analyst_llm",
    "data_cleaner_llm",
    "analyst_llm",
    "visualization_orchestrator_llm",
    "viz_evaluator_llm",
    "viz_worker_llm",
    "report_orchestrator_llm",
    "report_section_worker_llm",
    "report_packager_llm",
    "file_writer_llm",
    "memsearch_query_llm",
    "quick_summary_llm",
    "summary_llm",
    "complex_summary_llm",
    "critical_complex_summary_llm",
)


class ProofError(RuntimeError):
    """Controlled failure code, safe to include in the public evidence index."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _utc_timestamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("Timing evidence must include a UTC offset")
    return stamp.astimezone(timezone.utc)


def _observation_times(value):
    requested = _utc_timestamp(value["requested_utc"])
    observed = _utc_timestamp(value["utc"])
    received = _utc_timestamp(value["received_utc"])
    if requested > received or not (
        requested - CLOCK_TOLERANCE <= observed <= received + CLOCK_TOLERANCE
    ):
        raise ValueError("Child clock is not correlated with the runner")
    return requested, observed, received


def _execution_log_interval(observations, execution):
    """Require successful runner timing and bounded, ordered child observations."""
    try:
        if not isinstance(execution, dict) or execution["status"] != "COMPLETED":
            raise ValueError("Execution did not complete successfully")
        started = _utc_timestamp(execution["started_utc"])
        ended = _utc_timestamp(execution["ended_utc"])
        lower = (started - CLOCK_TOLERANCE).replace(microsecond=0)
        upper = ended + CLOCK_TOLERANCE
        # HH:MM:SS log prefixes cannot disambiguate a full day's clock values.
        if ended < started or upper - lower >= timedelta(days=1):
            raise ValueError("Execution interval is reversed or date-ambiguous")
        for stage in ("identity", "runtime", "stream"):
            if stage not in observations:
                raise ValueError("Required timing observation is missing")
        previous_received, previous_observed = started, None
        for stage in ("identity", *OBSERVATION_CELLS.values()):
            if stage not in observations:
                continue
            value = observations[stage]
            if value["status"] != "VERIFIED":
                raise ValueError("Timing observation was not verified")
            requested, observed, received = _observation_times(value)
            if not previous_received <= requested <= received <= ended or (
                previous_observed is not None and observed < previous_observed
            ):
                raise ValueError("Execution and observation timing disagree")
            previous_received, previous_observed = received, observed
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ProofError("ASSOCIATION_EXECUTION_TIMING_INVALID") from exc
    # The prefix truncates fractional seconds: include the intersecting first second.
    return lower, upper


def hashes(data):
    return {
        "raw_sha256": hashlib.sha256(data).hexdigest(),
        "lf_sha256": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest(),
    }


def text_hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def config_hash(value):
    # Do not serialize arbitrary object reprs (which may contain client secrets).
    try:
        return text_hash(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ProofError("CONFIGURATION_NOT_SERIALIZABLE") from exc


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


class Evidence:
    def __init__(self, directory, execution_kind):
        self.directory = Path(directory)
        self.data = {
            "schema_version": 1,
            "attempt_id": str(uuid.uuid4()),
            "created_utc": utc_now(),
            "execution_kind": execution_kind,
            "attempt_status": "PREPARING",
            "cp5_pass": False,
            "execution": {"status": "NOT_REACHED"},
            "provenance": "NOT_REACHED",
            "association": "NOT_REACHED",
            "automatic_validation": "NOT_REACHED",
            "artifact_inspection": "PENDING_INDEPENDENT_REVIEW",
            "observations": {
                name: {"status": "NOT_REACHED"}
                for name in ("identity", *OBSERVATION_CELLS.values())
            },
            "first_failure": None,
            "secondary_failures": [],
            "files": {},
            "output_directory": "NOT_REACHED",
            "kernel_cleanup": "NOT_REACHED",
            "evidence_completeness": "IN_PROGRESS",
            "raw_evidence_policy": "RESTRICTED_UNTIL_MANUALLY_SANITIZED",
        }
        self.save()

    def save(self):
        self.data["updated_utc"] = utc_now()
        atomic_json(self.directory / "evidence.json", self.data)

    def fail(self, error, stage, cell_index=None):
        self.data["attempt_status"] = "FAILED"
        failure = {
            "type": type(error).__name__,
            "stage": stage,
            "cell_index": cell_index,
            "utc": utc_now(),
        }
        if isinstance(error, ProofError):
            failure["code"] = str(error)
        if self.data["first_failure"] is None:
            self.data["first_failure"] = failure
        else:
            self.data["secondary_failures"].append(failure)
        self.save()


def create_attempt(directory):
    directory = Path(directory).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ProofError("ATTEMPT_SYMLINK")
    if directory.exists():
        raise ProofError("ATTEMPT_EXISTS")
    directory.mkdir(mode=0o700)  # parent must exist; never clean or reuse anything
    workspace, evidence = directory / "workspace", directory / "evidence"
    workspace.mkdir(mode=0o700)
    evidence.mkdir(mode=0o700)
    return workspace, evidence


def assert_pristine(workspace):
    """Fail rather than delete checkpoint/output state, including orphan sidecars."""
    try:
        names = [p.name for p in workspace.iterdir()]
        forbidden = {
            "_idd_resume.flag",
            "current_run_thread_id.txt",
            "artifacts",
            "WORKING_DIRECTORY",
            "IDD_results",
            "notebook_run_log.txt",
        }
        if any(name in forbidden or name.startswith("checkpoints") for name in names):
            raise ProofError("STALE_ATTEMPT_STATE")
    except OSError as exc:
        raise ProofError("STALE_STATE_INACCESSIBLE") from exc


def child_environment(workspace, scenario, credential_source, environ):
    # Only explicit system launch settings; never iterate or dump the environment.
    names = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "LANG",
        "LC_ALL",
    )
    env = {name: environ.get(name) for name in names if environ.get(name) is not None}
    for name, folder in (
        ("IPYTHONDIR", "ipython"),
        ("JUPYTER_CONFIG_DIR", "jupyter-config"),
        ("JUPYTER_DATA_DIR", "jupyter-data"),
        ("JUPYTER_RUNTIME_DIR", "jupyter-runtime"),
        ("XDG_CACHE_HOME", "cache"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("TMPDIR", "tmp"),
        ("TEMP", "tmp"),
        ("TMP", "tmp"),
    ):
        path = workspace / folder
        path.mkdir(exist_ok=True)
        env[name] = str(path)
    env.update(
        {
            "IDD_SKIP_NOTEBOOK_INSTALLS": "1",
            "IDD_ALLOW_RECOVERY": "0",
            "IDD_SAMPLE_DATASET": scenario,
            "GDRIVE_BASE": str(workspace / "IDD_results"),
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "ANONYMIZED_TELEMETRY": "False",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if credential_source == "process":
        key = environ.get("OPENAI_API_KEY")
        if not key:
            raise ProofError("PROCESS_CREDENTIAL_MISSING")
        env["OPENAI_API_KEY"] = key
    elif credential_source != "none":
        raise ProofError("CREDENTIAL_SOURCE_INVALID")
    return env


def verify_kernel(name, expected_python, manager=None):
    from jupyter_client.kernelspec import KernelSpecManager

    if not name or not Path(expected_python).is_absolute():
        raise ProofError("KERNEL_REQUIRED")
    try:
        spec = (manager or KernelSpecManager()).get_kernel_spec(name)
    except Exception as exc:
        raise ProofError("KERNEL_UNAVAILABLE") from exc
    argv = spec.argv
    if (
        not argv
        or not Path(argv[0]).is_absolute()
        or os.path.normcase(os.path.abspath(argv[0]))
        != os.path.normcase(os.path.abspath(expected_python))
    ):
        raise ProofError("KERNEL_INTERPRETER_MISMATCH")
    allowed = (
        ["-m", "ipykernel_launcher", "-f", "{connection_file}"],
        ["-Xfrozen_modules=off", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    )
    provisioner = getattr(spec, "metadata", {}).get("kernel_provisioner", {})
    if (
        argv[1:] not in allowed
        or spec.env
        or provisioner.get("provisioner_name", "local-provisioner")
        != "local-provisioner"
        or provisioner.get("config")
    ):
        raise ProofError("KERNEL_UNAPPROVED_LAUNCH_CONFIGURATION")
    return {
        "name": name,
        "argv": argv,
        "resource_dir": spec.resource_dir,
        "environment_overrides": [],
    }


def verify_source(source, sha, workspace):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ProofError("SOURCE_SHA_REQUIRED")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], stderr=subprocess.DEVNULL
        )

    if git("rev-parse", "HEAD").decode().strip() != sha:
        raise ProofError("SOURCE_HEAD_MISMATCH")
    inventory = {}
    for name in SOURCE_FILES:
        blob = git("show", f"{sha}:{name}")
        local = source / name
        if (
            local.is_symlink()
            or hashes(local.read_bytes())["lf_sha256"] != hashes(blob)["lf_sha256"]
        ):
            raise ProofError("SOURCE_BYTES_MISMATCH")
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(blob)
        inventory[name] = hashes(blob)
    manifest = json.loads((workspace / "idd_manifest.json").read_text())
    if any(
        inventory[name]["lf_sha256"] != digest
        for name, digest in manifest["files"].items()
    ):
        raise ProofError("MANIFEST_MISMATCH")
    return inventory


def verify_snapshot(workspace, inventory):
    for name, frozen in inventory.items():
        path = workspace / name
        if (
            path.is_symlink()
            or not path.is_file()
            or hashes(path.read_bytes()) != frozen
        ):
            raise ProofError("SOURCE_SNAPSHOT_MUTATED")


def owned_path(path, workspace):
    return path.resolve().is_relative_to(workspace.resolve()) and not any(
        p.is_symlink() for p in (path, *path.parents) if p != workspace.parent
    )


def _connection_closed(conn):
    if conn is None:
        return "UNAVAILABLE"
    try:
        conn.in_transaction
    except Exception:
        return True
    return False


def observe_namespace(ns, stage):
    """Read-only observer. Called in the child; all IDs are kernel-lifetime only."""
    result = {
        "status": "OBSERVED",
        "utc": utc_now(),
        "observer_path": str(Path(__file__).resolve()),
        "observer_sha256": hashes(Path(__file__).read_bytes())["raw_sha256"],
    }
    if stage == "identity":
        versions = {}
        for name in DEPENDENCIES:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "UNAVAILABLE"
        result.update(
            sys_executable=os.path.abspath(sys.executable),
            resolved_executable=str(Path(sys.executable).resolve()),
            sys_prefix=sys.prefix,
            base_prefix=sys.base_prefix,
            python_version=platform.python_version(),
            cwd=str(Path.cwd().resolve()),
            dependencies=versions,
            install_flag=os.environ.get("IDD_SKIP_NOTEBOOK_INSTALLS"),
            tracing_disabled=os.environ.get("LANGSMITH_TRACING") == "false"
            and os.environ.get("LANGCHAIN_TRACING_V2") == "false",
            utc_offset_seconds=datetime.now().astimezone().utcoffset().total_seconds(),
        )
    elif stage in ("bootstrap", "canonical"):
        stems = ("idd_bootstrap",) if stage == "bootstrap" else CANONICAL
        result["modules"] = {}
        for name in stems:
            module = sys.modules.get(name)
            path = Path(module.__file__).resolve() if module is not None else None
            result["modules"][name] = {
                "path": str(path) if path else None,
                "hashes": hashes(path.read_bytes()) if path else None,
            }
        if stage == "canonical":
            checks = (
                ("Plan", "idd_models"),
                ("State", "idd_state"),
                ("DataFrameRegistry", "idd_registry"),
                ("handle_tool_errors", "idd_tools"),
            )
            result["identities"] = {
                name: ns.get(name) is getattr(sys.modules.get(module), name, None)
                and ns.get(name) is not None
                for name, module in checks
            }
            result["state_fields"] = len(ns["State"].__annotations__)
    elif stage == "input":
        raw = Path(ns["raw_path_str"]).resolve()
        # Paths and IDs are validated before publishing; no rows or prompt text.
        result.update(
            input_path=str(raw),
            input_hashes=hashes(raw.read_bytes()),
            dataframe_id=ns["df_id"],
            dataframe_name=ns["df_name"],
            rows=len(ns["df"]),
            columns=len(ns["df"].columns),
            template_prompt_sha256=text_hash(ns["sample_prompt_text"]),
        )
    elif stage == "request":
        models = {}
        for role in MODEL_ROLES:
            model = ns[role]
            model_id = getattr(model, "model_name", None)
            if model_id not in ("gpt-5-mini", "gpt-5-nano"):
                raise ProofError("MODEL_CONFIGURATION_UNAVAILABLE")
            reasoning = getattr(model, "reasoning", None) or {}
            text = (getattr(model, "model_kwargs", None) or {}).get("text", {})
            settings = {
                "reasoning_effort": reasoning.get("effort"),
                "reasoning_summary": reasoning.get("summary"),
                "verbosity": text.get("verbosity"),
                "max_retries": getattr(model, "max_retries", None),
                "timeout": getattr(model, "request_timeout", None),
                "output_version": getattr(model, "output_version", None),
                "use_responses_api": getattr(model, "use_responses_api", None),
            }
            if any(
                v is not None and not isinstance(v, (str, bool, int, float))
                for v in settings.values()
            ):
                raise ProofError("MODEL_SETTINGS_UNAVAILABLE")
            models[role] = {
                "model": model_id,
                "settings": settings,
                "settings_sha256": config_hash(settings),
            }
        result.update(
            final_prompt_sha256=text_hash(ns["sample_prompt_text"]),
            config_sha256=config_hash(ns["RUNTIME"]._config),
            models=models,
        )
        embedding = ns["MEM_EMBEDDINGS"]
        if (
            getattr(embedding, "model", None) != "text-embedding-3-small"
            or ns["DIM"] != 1536
        ):
            raise ProofError("EMBEDDING_CONFIGURATION_UNAVAILABLE")
        result["embedding"] = {"model": "text-embedding-3-small", "dimensions": 1536}
    elif stage in ("graph", "stream", "migration", "late"):
        graph = ns["data_detective_graph"]
        result["graph_id"] = id(graph)
        if not hasattr(graph, "get_graph"):
            return result
        drawable = graph.get_graph()
        cp = graph.checkpointer
        conn = getattr(cp, "conn", None)
        storage = None
        if conn is not None:
            storage = [
                str(Path(row[2]).resolve())
                for row in conn.execute("PRAGMA database_list").fetchall()
                if row[1] == "main" and row[2]
            ]
        result.update(
            graph_id=id(graph),
            checkpointer_id=id(cp),
            checkpointer_class=type(cp).__module__ + "." + type(cp).__name__,
            checkpoint_storage=storage,
            nodes=sorted(
                n for n in drawable.nodes if n not in ("__start__", "__end__")
            ),
            edges=len(drawable.edges),
            conditional_edges=sum(bool(e.conditional) for e in drawable.edges),
        )
        if stage in ("migration", "late"):
            result.update(
                demo_path=str(
                    Path(ns["WORKING_DIRECTORY"]).resolve() / "checkpoints_demo.sqlite"
                ),
                migrated_checkpoints=ns.get(
                    "_migrated_checkpoint_count", "UNAVAILABLE"
                ),
                restored_history_count=(
                    len(ns["_restored_history_ids"])
                    if "_restored_history_ids" in ns
                    else "UNAVAILABLE"
                ),
                restore_connection_closed=_connection_closed(ns.get("_restore_conn")),
            )
        if stage == "graph":
            result["router_identity"] = (
                ns.get("Router")
                is getattr(sys.modules.get("idd_graph"), "Router", None)
                and ns.get("Router") is not None
            )
    elif stage == "runtime":
        cfg = ns["run_config"]
        result.update(
            run_id=ns["RUNTIME"].run_id,
            thread_id=cfg["configurable"]["thread_id"],
            displayed_stream_identifier=ns["run_id"],
            final_prompt_sha256=text_hash(ns["initial_state"]["user_prompt"]),
            config_sha256=config_hash(cfg),
        )
    else:
        raise ProofError("OBSERVER_STAGE_UNKNOWN")
    return result


async def observe(client, stage, timeout=30):
    expression = f"__import__('json').dumps(__import__('idd_proof').observe_namespace(globals(), {stage!r}))"
    msg_id = client.kc.execute(
        "",
        silent=True,
        store_history=False,
        user_expressions={"idd": expression},
        stop_on_error=True,
    )

    async def reply():
        while True:
            message = await client.kc.get_shell_msg()
            if message.get("parent_header", {}).get("msg_id") == msg_id:
                return message

    response = await asyncio.wait_for(reply(), timeout)
    value = response["content"].get("user_expressions", {}).get("idd", {})
    if value.get("status") != "ok":
        raise ProofError("REQUIRED_OBSERVATION_UNAVAILABLE")

    async def idle():
        while True:
            msg = await client.kc.get_iopub_msg()
            if (
                msg.get("parent_header", {}).get("msg_id") == msg_id
                and msg["msg_type"] == "status"
                and msg["content"]["execution_state"] == "idle"
            ):
                return

    await asyncio.wait_for(idle(), timeout)
    return json.loads(ast.literal_eval(value["data"]["text/plain"]))


def verify_observation(
    stage, value, workspace, evidence, expected_python, expected_version
):
    verify_snapshot(workspace, evidence.data.get("sources", {}))
    if (
        value.get("observer_path") != str((workspace / "idd_proof.py").resolve())
        or value.get("observer_sha256")
        != hashes((workspace / "idd_proof.py").read_bytes())["raw_sha256"]
    ):
        raise ProofError("OBSERVER_ORIGIN_MISMATCH")
    if stage == "identity":
        if (
            os.path.normcase(value.get("sys_executable", ""))
            != os.path.normcase(os.path.abspath(expected_python))
            or value.get("python_version") != expected_version
            or value.get("cwd") != str(workspace.resolve())
            or value.get("install_flag") != "1"
            or not value.get("tracing_disabled")
        ):
            raise ProofError("KERNEL_IDENTITY_MISMATCH")
    elif stage in ("bootstrap", "canonical"):
        for name, info in value["modules"].items():
            path = workspace / (name + ".py")
            if info["path"] != str(path.resolve()) or info["hashes"] != hashes(
                path.read_bytes()
            ):
                raise ProofError("CANONICAL_ORIGIN_MISMATCH")
        if stage == "canonical" and (
            not all(value["identities"].values()) or value["state_fields"] != 70
        ):
            raise ProofError("CANONICAL_IDENTITY_MISMATCH")
    elif stage == "input":
        path = Path(value["input_path"])
        scenario = evidence.data["scenario"]
        if (
            path
            != workspace / "tests" / "trajectory" / "fixtures" / (scenario + ".csv")
            or value["dataframe_name"] != scenario
            or value["dataframe_id"] != scenario
            or value["input_hashes"] != hashes(path.read_bytes())
        ):
            raise ProofError("INPUT_IDENTITY_MISMATCH")
    elif stage in ("graph", "stream", "migration", "late"):
        old = evidence.data["observations"]["graph"]
        if stage != "graph" and (
            value["graph_id"] != old.get("graph_id")
            or value["checkpointer_id"] != old.get("checkpointer_id")
        ):
            raise ProofError("GRAPH_REPLACED")
        if (
            len(value["nodes"]) != 15
            or value["edges"] != 33
            or value["conditional_edges"] != 21
        ):
            raise ProofError("GRAPH_TOPOLOGY_MISMATCH")
        if stage == "graph" and not value.get("router_identity"):
            raise ProofError("ROUTER_IDENTITY_MISMATCH")
        storage = value["checkpoint_storage"]
        if storage is not None and storage != [
            str((workspace / "checkpoints.sqlite").resolve())
        ]:
            raise ProofError("CHECKPOINT_LOCATION_MISMATCH")
        if stage == "late" and storage is not None:
            demo = Path(value["demo_path"])
            if (
                not demo.is_relative_to(workspace)
                or demo == Path(storage[0])
                or not demo.is_file()
                or value["restore_connection_closed"] is not True
            ):
                raise ProofError("CHECKPOINT_DEMO_MISMATCH")
            if (
                not isinstance(value["migrated_checkpoints"], int)
                or value["migrated_checkpoints"] < 1
                or value["restored_history_count"] != value["migrated_checkpoints"]
            ):
                raise ProofError("CHECKPOINT_HISTORY_UNAVAILABLE")
    elif stage == "runtime":
        if (
            not re.fullmatch(r"run_default_id-\d{8}-\d{4}-[0-9a-f]{8}", value["run_id"])
            or not re.fullmatch(r"thread-[0-9a-f-]{36}", value["thread_id"])
            or value["displayed_stream_identifier"] != value["run_id"]
            or value["final_prompt_sha256"]
            != evidence.data["observations"]["request"].get("final_prompt_sha256")
        ):
            raise ProofError("RUNTIME_IDENTITY_MISMATCH")


async def execute_strict(
    nb,
    *,
    workspace,
    evidence,
    kernel,
    expected_python,
    expected_version,
    env,
    timeout=3600,
    observation_cells=None,
):
    import nbformat
    from nbclient import NotebookClient

    stages = OBSERVATION_CELLS if observation_cells is None else observation_cells
    source_cells = [(cell.cell_type, cell.source) for cell in nb.cells]
    index = None
    current_stage = "initialization"
    primary_error = None
    console = None
    client = None
    stream_failure = False

    def save_partial():
        path = evidence.directory / "executed.ipynb"
        temporary = evidence.directory / "executed.tmp"
        with temporary.open("w", encoding="utf-8") as output:
            nbformat.write(nb, output)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        evidence.data["files"]["executed_notebook"] = {
            "path": str(path),
            **hashes(path.read_bytes()),
        }

    class Client(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            nonlocal stream_failure
            content = msg.get("content", {})
            if msg["msg_type"] == "stream":
                text = content.get("text", "")
                console.write(text)
                console.flush()
                if (
                    "Traceback (most recent call last)" in text
                    or "Streaming error:" in text
                ):
                    stream_failure = True
            elif msg["msg_type"] == "error":
                console.write("\n".join(content.get("traceback", [])) + "\n")
                console.flush()
            return super().process_message(msg, cell, cell_index)

    async def attest(client, stage):
        requested_utc = utc_now()
        evidence.data["observations"][stage] = {
            "status": "UNAVAILABLE",
            "requested_utc": requested_utc,
        }
        evidence.save()
        value = await observe(client, stage)
        value.update(requested_utc=requested_utc, received_utc=utc_now())
        try:
            try:
                _observation_times(value)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ProofError("OBSERVATION_CLOCK_MISMATCH") from exc
            verify_observation(
                stage, value, workspace, evidence, expected_python, expected_version
            )
        except Exception:
            # Do not publish an unvalidated response (paths/IDs could contain secrets).
            evidence.data["observations"][stage]["status"] = "MISMATCH"
            raise
        value["status"] = "VERIFIED"
        evidence.data["observations"][stage] = value
        evidence.save()

    try:
        assert_pristine(workspace)
        if "force_raise_errors" not in NotebookClient.class_traits():
            raise ProofError("NBCLIENT_STRICT_CONTROLS_UNAVAILABLE")
        evidence.data["nbclient"] = {
            "version": importlib.metadata.version("nbclient"),
            "allow_errors": False,
            "force_raise_errors": True,
            "allow_error_names": [],
        }
        # Exclusive creation is required; a stale/inaccessible log is never truncated.
        with (workspace / "notebook_run_log.txt").open("x", encoding="utf-8"):
            pass
        console = (evidence.directory / "console.log").open("x", encoding="utf-8")
        evidence.data["execution"] = {
            "status": "RUNNING",
            "started_utc": utc_now(),
            "last_cell_index": None,
        }
        evidence.save()
        client = Client(
            nb,
            kernel_name=kernel,
            timeout=timeout,
            startup_timeout=60,
            allow_errors=False,
            force_raise_errors=True,
            allow_error_names=[],
            interrupt_on_timeout=False,
            skip_cells_with_tag="proof-never-skip-" + uuid.uuid4().hex,
            resources={"metadata": {"path": str(workspace)}},
        )
        client.reset_execution_trackers()
        client.km = client.create_kernel_manager()
        client.km.connection_file = str(workspace / "jupyter-runtime" / "kernel.json")
        Path(client.km.connection_file).parent.mkdir(exist_ok=True)
        async with client.async_setup_kernel(cwd=str(workspace), env=env):
            try:
                current_stage = "identity"
                await attest(client, "identity")
                for index, cell in enumerate(nb.cells):
                    current_stage = "cell"
                    evidence.data["execution"]["last_cell_index"] = index
                    evidence.save()
                    await client.async_execute_cell(
                        cell, index, execution_count=client.code_cells_executed + 1
                    )
                    if stream_failure:
                        raise ProofError("STREAM_FAILURE")
                    if index in stages:
                        current_stage = stages[index]
                        await attest(client, current_stage)
                    save_partial()
                if source_cells != [(cell.cell_type, cell.source) for cell in nb.cells]:
                    raise ProofError("NOTEBOOK_SOURCE_MUTATED")
                if any(
                    evidence.data["observations"][stage]["status"] != "VERIFIED"
                    for stage in stages.values()
                ):
                    raise ProofError("REQUIRED_OBSERVATION_NOT_REACHED")
                verify_snapshot(workspace, evidence.data.get("sources", {}))
                evidence.data["provenance"] = "VERIFIED"
                evidence.data["execution"]["status"] = "COMPLETED"
            except BaseException:
                # Copy owned temporary artifacts before kernel shutdown removes them.
                try:
                    temp = workspace / "tmp"
                    if temp.exists() and not any(
                        p.is_symlink() for p in temp.rglob("*")
                    ):
                        shutil.copytree(
                            temp,
                            evidence.directory / "partial-work",
                            dirs_exist_ok=False,
                        )
                except Exception:
                    evidence.data["partial_artifacts"] = "SAVE_FAILED"
                raise
    except BaseException as exc:
        primary_error = exc
        evidence.data["execution"]["status"] = "FAILED"
        try:
            evidence.fail(exc, current_stage, index)
        except Exception:
            pass  # retain the original exception; last atomic record is recoverable
        raise
    finally:
        final_error = None

        def record_save_error(error):
            evidence.data["evidence_completeness"] = "SAVE_FAILED"
            evidence.data["execution"]["status"] = "FAILED"
            evidence.data["provenance"] = "INCOMPLETE"
            try:
                evidence.fail(error, "finalization", index)
            except Exception:
                pass

        if console is not None:
            try:
                console.close()
            except Exception as exc:
                final_error = exc
                record_save_error(exc)
        try:
            evidence.data["kernel_cleanup"] = (
                "CONTEXT_CLOSED"
                if client is not None and client.kc is None
                else "NOT_STARTED_OR_INCOMPLETE"
            )
            save_partial()
            candidates = list((workspace / "IDD_results").glob("IDD_run_*"))
            run_id = evidence.data["observations"].get("runtime", {}).get("run_id")
            if (
                len(candidates) == 1
                and candidates[0].name == "IDD_run_" + str(run_id)
                and owned_path(candidates[0], workspace)
            ):
                evidence.data["output_directory"] = str(candidates[0])
            elif candidates:
                evidence.data["output_directory"] = "UNAVAILABLE_AMBIGUOUS"
            evidence.data["execution"]["ended_utc"] = utc_now()
            for name, path in (
                ("console_log", evidence.directory / "console.log"),
                ("pipeline_log", workspace / "notebook_run_log.txt"),
            ):
                if path.is_file():
                    evidence.data["files"][name] = {
                        "path": str(path),
                        **hashes(path.read_bytes()),
                    }
            if final_error is None:
                evidence.data["evidence_completeness"] = (
                    "CAPTURED" if primary_error is None else "PARTIAL_CAPTURED"
                )
            evidence.save()
        except Exception as exc:
            record_save_error(exc)
            if final_error is None:
                final_error = exc
        if primary_error is None and final_error is not None:
            raise final_error


def associate_outputs(workspace, observations, evidence_directory=None, *, execution):
    runtime = observations["runtime"]
    run_id = runtime["run_id"]
    candidates = list((workspace / "IDD_results").glob("IDD_run_*"))
    if (
        len(candidates) != 1
        or candidates[0].name != "IDD_run_" + run_id
        or not candidates[0].is_dir()
    ):
        raise ProofError("ASSOCIATION_RUN_AMBIGUOUS")
    run = candidates[0]
    if not owned_path(run, workspace) or any(p.is_symlink() for p in run.rglob("*")):
        raise ProofError("ASSOCIATION_SYMLINK")
    reports = [p for p in run.rglob("reports") if p.is_dir()]
    if reports != [run / "artifacts" / run_id / "reports"]:
        raise ProofError("ASSOCIATION_REPORT_AMBIGUOUS")
    pipeline = workspace / "notebook_run_log.txt"
    log = pipeline.read_text(encoding="utf-8")
    directory = evidence_directory or workspace.parent / "evidence"
    try:
        console = (directory / "console.log").read_text(encoding="utf-8")
        thread = (
            (workspace / "current_run_thread_id.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
        displayed = re.findall(r"Starting stream \(thread_id=([^\s)]+)\)", console)
        if (
            displayed != [run_id]
            or thread != runtime["thread_id"]
            or runtime["displayed_stream_identifier"] != run_id
        ):
            raise ProofError("ASSOCIATION_IDENTIFIERS_MISMATCH")
        zone = timezone(
            timedelta(seconds=observations["identity"]["utc_offset_seconds"])
        )
        start, end = _execution_log_interval(observations, execution)
        start, end = start.astimezone(zone), end.astimezone(zone)
        times = re.findall(r"^(\d{2}):(\d{2}):(\d{2})\b", log, re.M)
        if not times:
            raise ProofError("ASSOCIATION_LOG_TIMESTAMPS_MISSING")
        for hh, mm, ss in times:
            clock = start.replace(
                hour=int(hh), minute=int(mm), second=int(ss), microsecond=0
            )
            if not any(start <= clock + timedelta(days=d) <= end for d in (-1, 0, 1)):
                raise ProofError("ASSOCIATION_MIXED_LOG_INTERVAL")
    except (KeyError, OSError, TypeError, ValueError, OverflowError) as exc:
        raise ProofError("ASSOCIATION_BINDING_UNAVAILABLE") from exc
    seen = set(re.findall(r"run_default_id-\d{8}-\d{4}-[0-9a-f]{8}", log))
    seen.update(re.findall(r"run_default_id-\d{8}-\d{4}-[0-9a-f]{8}", console))
    if not log or seen - {run_id}:
        raise ProofError("ASSOCIATION_FOREIGN_LOG")
    for report in run.rglob("*"):
        if report.suffix.lower() not in (".html", ".md") or not report.is_file():
            continue
        text = report.read_text(encoding="utf-8")
        refs = re.findall(
            r"<img\b[^>]*\bsrc=[\"\']([^\"\']+)", text, re.I
        ) + re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        for ref in refs:
            if ref.startswith("data:image/"):
                continue
            image = (report.parent / ref.replace("\\", "/")).resolve()
            if not image.is_relative_to(run.resolve()) or "://" in ref:
                raise ProofError("ASSOCIATION_EXTERNAL_FIGURE")
    return run


def run_validators(workspace, run_dir, evidence, *, expected_python):
    """Run both validators with the attested kernel's interpreter path."""
    log = workspace / "notebook_run_log.txt"
    before = run_dir.stat().st_mtime_ns
    evidence.data["run_directory_mtime_ns"] = before
    commands = [
        [
            str(expected_python),
            str(workspace / "validate_run.py"),
            str(run_dir),
            "--log-path",
            str(log),
            "--window",
            "180",
            "--json",
        ],
        [
            str(expected_python),
            str(workspace / "validate_artifact_quality.py"),
            str(run_dir),
        ],
    ]
    results = []
    for name, command in zip(("validate_run", "validate_artifact_quality"), commands):
        started = utc_now()
        try:
            process = subprocess.run(
                command,
                cwd=workspace,
                capture_output=True,
                timeout=120,
                env=child_environment(
                    workspace, evidence.data["scenario"], "none", os.environ
                ),
                check=False,
            )
            exit_code = process.returncode
            status = "PASSED" if exit_code == 0 else "FAILED"
            raw_output = (process.stdout or b"") + (process.stderr or b"")
        except subprocess.TimeoutExpired as exc:
            exit_code, status = None, "TIMEOUT"
            raw_output = (exc.output or b"") + (exc.stderr or b"")
        except OSError:
            exit_code, status, raw_output = (
                None,
                "START_FAILED",
                b"Validator could not start.",
            )
        path = evidence.directory / (name + ".log")
        with path.open("xb") as output:
            output.write(raw_output)
        results.append(
            {
                "command": command,
                "exit_code": exit_code,
                "status": status,
                "started_utc": started,
                "ended_utc": utc_now(),
                "output": {"path": str(path), **hashes(path.read_bytes())},
            }
        )
        evidence.data["validators"] = results
        evidence.save()
    if run_dir.stat().st_mtime_ns != before:
        raise ProofError("VALIDATION_TIMESTAMP_CHANGED")
    evidence.data["run_directory_mtime_ns"] = before
    evidence.data["automatic_validation"] = (
        "PASSED" if all(p["exit_code"] == 0 for p in results) else "FAILED"
    )
    console = (evidence.directory / "console.log").read_text(encoding="utf-8")
    hits = re.findall(
        r"Traceback|RECOVERY|W2-BA-finalhop|W4-|failed_native|path_normalized_missing|GraphRecursionError",
        log.read_text(encoding="utf-8") + console,
    )
    evidence.data["complete_log_scan"] = {
        "matched_markers": sorted(set(hits)),
        "status": "FAILED" if hits else "CLEAR",
    }
    evidence.save()
    return evidence.data["automatic_validation"] == "PASSED" and not hits


def run_proof(args, source):
    """CLI boundary: explicit provenance, isolated snapshot, no legacy key loaders."""
    evidence = None
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        if args.resume or args.preflight or args.check_langsmith:
            raise ProofError("PROOF_INCOMPATIBLE_OPTIONS")
        if not all(
            (
                args.attempt_dir,
                args.source_sha,
                args.kernel_name,
                args.expected_python,
                args.expected_python_version,
                args.scenario,
            )
        ):
            raise ProofError("PROOF_EXPLICIT_SELECTION_REQUIRED")
        if os.environ.get("IDD_SKIP_NOTEBOOK_INSTALLS", "1").strip().lower() not in (
            "1",
            "true",
            "yes",
            "on",
        ):
            raise ProofError("NOTEBOOK_INSTALLS_MUST_BE_DISABLED")
        if not args.prepare_only and args.credential_source != "process":
            raise ProofError("PROOF_EXECUTION_REQUIRES_EXPLICIT_PROCESS_CREDENTIAL")
        if args.prepare_only and args.credential_source != "none":
            raise ProofError("PREPARATION_MUST_BE_NO_KEY")
        if Path(args.attempt_dir).resolve().is_relative_to(source.resolve()):
            raise ProofError("ATTEMPT_MUST_BE_OUTSIDE_SOURCE")
        workspace, directory = create_attempt(Path(args.attempt_dir))
        evidence = Evidence(
            directory,
            (
                "no-key-preparation"
                if args.prepare_only
                else "provider-execution-requires-owner-authorization"
            ),
        )
        evidence.data.update(
            source_sha=args.source_sha,
            scenario=args.scenario,
            credential_provenance=args.credential_source,
            external_services={
                "langsmith": "disabled",
                "tavily": "credential-not-forwarded",
                "openai_endpoint": "provider-default",
            },
        )
        evidence.data["sources"] = verify_source(source, args.source_sha, workspace)
        evidence.data["kernelspec"] = verify_kernel(
            args.kernel_name, Path(args.expected_python)
        )
        evidence.save()
        env = child_environment(
            workspace, args.scenario, args.credential_source, os.environ
        )
        import nbformat

        if args.prepare_only:
            # Zero production cells, including no imports of model/embedding code.
            notebook = nbformat.v4.new_notebook()
            asyncio.run(
                execute_strict(
                    notebook,
                    workspace=workspace,
                    evidence=evidence,
                    kernel=args.kernel_name,
                    expected_python=Path(args.expected_python),
                    expected_version=args.expected_python_version,
                    env=env,
                    observation_cells={},
                )
            )
            evidence.data["execution"]["status"] = "NOT_EXECUTED_PREPARATION_ONLY"
            evidence.data["attempt_status"] = "PREPARED_NO_PRODUCTION_EXECUTION"
            evidence.data["provenance"] = "PREPARATION_ONLY"
            evidence.save()
            return 0
        notebook = nbformat.read(workspace / NOTEBOOK, as_version=4)
        if len(notebook.cells) != 99:
            raise ProofError("NOTEBOOK_CELL_COUNT_MISMATCH")
        asyncio.run(
            execute_strict(
                notebook,
                workspace=workspace,
                evidence=evidence,
                kernel=args.kernel_name,
                expected_python=Path(args.expected_python),
                expected_version=args.expected_python_version,
                env=env,
            )
        )
        run_dir = associate_outputs(
            workspace,
            evidence.data["observations"],
            directory,
            execution=evidence.data["execution"],
        )
        evidence.data.update(
            association="VERIFIED",
            output_directory=str(run_dir),
            artifacts=[
                {
                    "path": str(p.relative_to(run_dir)),
                    "bytes": p.stat().st_size,
                    **hashes(p.read_bytes()),
                }
                for p in sorted(run_dir.rglob("*"))
                if p.is_file()
            ],
        )
        evidence.save()
        passed = run_validators(
            workspace, run_dir, evidence, expected_python=args.expected_python
        )
        if passed:
            evidence.data["attempt_status"] = "AUTOMATIC_CHECKS_COMPLETE_PENDING_REVIEW"
            evidence.save()
        else:
            evidence.fail(ProofError("AUTOMATIC_VALIDATION_FAILED"), "validators")
        return 0 if passed else 1  # never a CP5 PASS; human inspection remains pending
    except Exception as exc:
        if evidence is not None:
            try:
                evidence.fail(exc, "runner")
            except Exception:
                pass
        print(
            "ERR Strict proof attempt failed ("
            + (str(exc) if isinstance(exc, ProofError) else type(exc).__name__)
            + "); inspect restricted attempt evidence.",
            file=sys.stderr,
        )
        return 1
