"""
run_notebook_live.py -- Headless execution of the IDD v5 patched notebook

Usage:
    python run_notebook_live.py           # fresh run (deletes checkpoints.sqlite)
    python run_notebook_live.py --resume  # resume from last checkpoint
    python run_notebook_live.py --preflight  # scientific imports only; no full runtime proof

Defaults to IntelligentDataDetective_beta_v5_patched.ipynb.
Set IDD_NOTEBOOK deliberately to select another notebook.
--preflight checks six scientific imports and the install flag in the selected
Jupyter kernel, not all notebook imports, API keys, graph execution, or reporting.

Requirements:
    pip install nbclient nbformat jupyter_client ipykernel

NEVER cancel this script -- the notebook takes 6-25 minutes to complete.
"""

import argparse
import os
import sys
import glob
import re
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

# Ensure stdout/stderr use UTF-8 so Unicode in notebook output doesn't crash the runner
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent
NOTEBOOK_NAME = os.environ.get("IDD_NOTEBOOK", "IntelligentDataDetective_beta_v5_patched.ipynb")
NOTEBOOK_PATH = REPO_ROOT / NOTEBOOK_NAME
OUTPUT_DIR = REPO_ROOT / "IDD_results"
TIMEOUT = 3600  # 60 minutes — analyst/data_cleaner each cap at ~15-20 min with recovery
LANGSMITH_ENV_NAMES = (
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_WORKSPACE_ID",
    "LANGCHAIN_CALLBACKS_BACKGROUND",
)
SCIENTIFIC_MODULES = (
    "numpy",
    "pandas",
    "scipy.stats",
    "sklearn.preprocessing",
    "matplotlib.pyplot",
    "matplotlib.figure",
)


def _valid_openai_api_key(value: str) -> bool:
    """Reject common env-var corruption before it reaches Authorization headers."""
    if not value:
        return False
    if any(ch.isspace() for ch in value):
        return False
    if value.startswith(("At ", "Traceback", "Error", "Exception")):
        return False
    return value.startswith("sk-") and len(value) >= 30


def load_api_key():
    """Load a valid OPENAI_API_KEY from process, dotenv, or Windows env scopes."""
    dotenv_values = _load_dotenv_values({"OPENAI_API_KEY"})
    candidates: list[tuple[str, str]] = []
    candidates.append(("process", os.environ.get("OPENAI_API_KEY", "").strip()))
    if "OPENAI_API_KEY" in dotenv_values:
        candidates.append((".env", dotenv_values["OPENAI_API_KEY"].strip()))
    if sys.platform == "win32":
        for scope in ("User", "Machine"):
            candidates.append((scope, _read_windows_env_var("OPENAI_API_KEY", scope)))

    saw_invalid = False
    for source, key in candidates:
        key = (key or "").strip()
        if not key:
            continue
        if not _valid_openai_api_key(key):
            saw_invalid = True
            print(
                f"WARN Ignoring invalid OPENAI_API_KEY from {source} environment scope"
            )
            continue
        os.environ["OPENAI_API_KEY"] = key
        print(f"OK  OPENAI_API_KEY loaded from {source} environment scope")
        return key

    if saw_invalid:
        os.environ.pop("OPENAI_API_KEY", None)
        print("ERR OPENAI_API_KEY was found but no valid key-like value was available")
    else:
        print("ERR OPENAI_API_KEY is not set -- notebook will likely fail LLM calls")
    return ""


def _read_windows_env_var(name: str, scope: str) -> str:
    try:
        return subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f'[System.Environment]::GetEnvironmentVariable("{name}","{scope}")',
            ],
            text=True,
        ).strip()
    except Exception as exc:
        print(f"Warning: Could not read {name} from {scope} env: {exc}")
        return ""


def _load_dotenv_values(names: set[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for env_path in (REPO_ROOT / ".env", REPO_ROOT / ".env.local"):
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in names or key in values:
                continue
            values[key] = value.strip().strip('"').strip("'")
    return values


def load_langsmith_env():
    """Load LangSmith/LangChain tracing vars without printing secret values."""
    names = set(LANGSMITH_ENV_NAMES)
    dotenv_values = _load_dotenv_values(names)
    loaded: dict[str, str] = {}

    for name in sorted(names):
        value = os.environ.get(name, "").strip()
        source = "process"
        if not value and name in dotenv_values:
            value = dotenv_values[name].strip()
            source = ".env"
        if not value and sys.platform == "win32":
            for scope in ("User", "Machine"):
                value = _read_windows_env_var(name, scope).strip()
                if value:
                    source = scope
                    break
        if value:
            os.environ[name] = value
            loaded[name] = source

    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_CALLBACKS_BACKGROUND", "false")
        loaded.setdefault("LANGSMITH_TRACING", "default:true")
        loaded.setdefault("LANGCHAIN_TRACING_V2", "default:true")
        loaded.setdefault("LANGCHAIN_CALLBACKS_BACKGROUND", "default:false")

    if loaded:
        visible = ", ".join(
            f"{key}<{source}>" for key, source in sorted(loaded.items())
        )
        print(f"OK  LangSmith tracing environment visible: {visible}")
    else:
        print("WARN LangSmith tracing environment not visible to runner")
    return loaded


def select_kernel_name() -> str:
    """Determine available kernel: prefer py312-codex over stale python3 spec."""
    try:
        _ks = subprocess.check_output(
            ["jupyter", "kernelspec", "list", "--json"], text=True
        )
        import json as _json

        _kernels = _json.loads(_ks).get("kernelspecs", {})
        if "py312-codex" in _kernels:
            return "py312-codex"
        if "python3" in _kernels:
            return "python3"
        if _kernels:
            return list(_kernels.keys())[0]
    except Exception:
        pass
    return "py312-codex"


def check_langsmith_cli() -> bool:
    """Check the LangSmith CLI without printing any credential values."""
    exe = shutil.which("langsmith")
    if not exe:
        print("WARN langsmith CLI executable is not on PATH")
        return False
    try:
        result = subprocess.run(
            ["langsmith", "--version"],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        output = (result.stdout or result.stderr or "").strip()
        if output:
            print(f"OK  langsmith CLI: {output}")
        return result.returncode == 0
    except Exception as exc:
        print(f"WARN langsmith CLI check failed: {exc}")
        return False


def probe_langsmith_kernel_env() -> bool:
    """Verify the child Jupyter kernel sees tracing env vars; prints booleans only."""
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        print(f"WARN Cannot probe kernel env; missing notebook dependency: {exc}")
        return False

    kernel_name = select_kernel_name()
    print(f"OK  Probing LangSmith env in kernel: {kernel_name}")
    names_literal = repr(list(LANGSMITH_ENV_NAMES))
    code = (
        "import os, json\n"
        f"names = {names_literal}\n"
        "presence = {name: bool(os.environ.get(name)) for name in names}\n"
        "print('LANGSMITH_KERNEL_ENV=' + json.dumps(presence, sort_keys=True))\n"
    )
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(code))
    try:
        client = NotebookClient(
            nb,
            timeout=120,
            kernel_name=kernel_name,
            allow_errors=False,
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute()
    except Exception as exc:
        print(f"WARN LangSmith kernel env probe failed: {exc}")
        return False

    probe_line = ""
    for out in nb.cells[0].get("outputs", []):
        text = "".join(out.get("text", ""))
        for line in text.splitlines():
            if line.startswith("LANGSMITH_KERNEL_ENV="):
                probe_line = line
    if not probe_line:
        print("WARN LangSmith kernel env probe produced no result")
        return False
    print(probe_line)
    import json as _json

    presence = _json.loads(probe_line.split("=", 1)[1])
    required = (
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_CALLBACKS_BACKGROUND",
    )
    missing = [name for name in required if not presence.get(name)]
    if missing:
        print(f"WARN LangSmith kernel env missing required keys: {', '.join(missing)}")
        return False
    print("OK  LangSmith env is visible inside the child Jupyter kernel")
    return True


def configure_skip_notebook_installs() -> str:
    """Ensure IDD_SKIP_NOTEBOOK_INSTALLS=1 by default; preserve user overrides."""
    if "IDD_SKIP_NOTEBOOK_INSTALLS" not in os.environ:
        os.environ["IDD_SKIP_NOTEBOOK_INSTALLS"] = "1"
        print("OK  Notebook dependency bootstrap: disabled by live-run harness")
    else:
        val = os.environ["IDD_SKIP_NOTEBOOK_INSTALLS"]
        print(f"INFO Notebook dependency bootstrap: preserved from environment (IDD_SKIP_NOTEBOOK_INSTALLS={val})")
    return os.environ["IDD_SKIP_NOTEBOOK_INSTALLS"]


def probe_kernel_scientific_stack(kernel_name: str | None = None) -> bool:
    """Check scientific imports and install-flag inheritance, not the full runtime."""
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:
        print(f"WARN Cannot probe kernel scientific stack; missing notebook dependency: {exc}")
        return False

    if not kernel_name:
        kernel_name = select_kernel_name()

    print(f"OK  Probing scientific stack in kernel: {kernel_name}")
    code = (
        "import importlib, json, os\n"
        f"modules = {SCIENTIFIC_MODULES!r}\n"
        "results = {}\n"
        "for name in modules:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        package = importlib.import_module(name.split('.', 1)[0])\n"
        "        results[name] = {'ok': True, 'version': getattr(package, '__version__', 'unknown')}\n"
        "    except Exception as exc:\n"
        "        results[name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}\n"
        "flag = os.environ.get('IDD_SKIP_NOTEBOOK_INSTALLS')\n"
        "payload = {'modules': results, 'install_flag': flag,\n"
        "           'installs_skipped': (flag or '').strip().lower() in {'1', 'true', 'yes', 'on'}}\n"
        "print('SCIENTIFIC_STACK_PREFLIGHT=' + json.dumps(payload, sort_keys=True))\n"
    )
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(code))
    try:
        client = NotebookClient(
            nb,
            timeout=60,
            kernel_name=kernel_name,
            allow_errors=False,
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute()
    except Exception as exc:
        print(f"ERR Scientific stack kernel probe execution failed: {exc}")
        return False

    probe_line = ""
    for out in nb.cells[0].get("outputs", []):
        text = "".join(out.get("text", ""))
        for line in text.splitlines():
            if line.startswith("SCIENTIFIC_STACK_PREFLIGHT="):
                probe_line = line
    if not probe_line:
        print("ERR Scientific stack kernel probe produced no result")
        return False

    import json as _json
    try:
        payload = _json.loads(probe_line.split("=", 1)[1])
        if not isinstance(payload, dict):
            raise ValueError("expected a result object")
        data = payload.get("modules")
        if not isinstance(data, dict) or set(data) != set(SCIENTIFIC_MODULES):
            raise ValueError("missing or unexpected scientific import results")
        if any(not isinstance(info, dict) or type(info.get("ok")) is not bool for info in data.values()):
            raise ValueError("invalid scientific import status")
        expected_flag = os.environ.get("IDD_SKIP_NOTEBOOK_INSTALLS")
        expected_skip = (expected_flag or "").strip().lower() in {"1", "true", "yes", "on"}
        if (
            "install_flag" not in payload
            or payload["install_flag"] != expected_flag
            or type(payload.get("installs_skipped")) is not bool
            or payload["installs_skipped"] != expected_skip
        ):
            raise ValueError("IDD_SKIP_NOTEBOOK_INSTALLS inheritance or normalization mismatch")
    except (ValueError, TypeError) as exc:
        print(f"ERR Invalid scientific stack probe result: {exc}")
        return False
    failed = [f"{name} ({info.get('error', 'unknown error')})" for name, info in sorted(data.items()) if not info.get("ok")]
    if failed:
        print(f"ERR Kernel scientific stack preflight failed: {', '.join(failed)}")
        return False

    versions_str = ", ".join(f"{name} {info.get('version', '?')}" for name, info in sorted(data.items()))
    print(f"OK  Kernel scientific stack preflight passed ({kernel_name}): {versions_str}")
    print(f"OK  Kernel install flag inherited; installs_skipped={expected_skip}")
    return True


def check_nbclient():
    try:
        import nbclient  # noqa
        import nbformat  # noqa

        print("OK  nbclient and nbformat are available")
        return True
    except ImportError as e:
        print(f"ERR Missing dependency: {e}")
        print(
            "    Install with: pip install nbclient nbformat jupyter_client ipykernel"
        )
        return False


def execute_notebook(resume: bool = False):
    import asyncio
    import nbformat
    from nbclient import NotebookClient

    # Suppress ZMQ/tornado warning on Windows about ProactorEventLoop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if not NOTEBOOK_PATH.exists():
        print(f"ERR Notebook not found: {NOTEBOOK_PATH}")
        print(
            "    Set IDD_NOTEBOOK to an existing notebook filename if overriding the default."
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    executed_nb_path = OUTPUT_DIR / f"executed_{timestamp}.ipynb"

    print("\n" + "=" * 60)
    print(f"Executing: {NOTEBOOK_PATH.name}")
    print(f"Timeout:   {TIMEOUT}s ({TIMEOUT // 60} min)")
    print(f"Output:    {executed_nb_path}")
    print("=" * 60 + "\n")

    with open(NOTEBOOK_PATH, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    kernel_name = select_kernel_name()
    print(f"OK  Using kernel: {kernel_name}")

    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name=kernel_name,
        allow_errors=True,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )

    cell_errors = []
    start = datetime.now()

    # Start log-tail thread: streams notebook_run_log.txt to stdout in real time
    import threading as _threading

    _log_file = REPO_ROOT / "notebook_run_log.txt"
    # Truncate / create; tolerate Windows file-share lock from the launching shell.
    try:
        _log_file.write_text("", encoding="utf-8")
    except PermissionError as _trunc_exc:
        print(
            f"W  could not truncate {_log_file.name} ({_trunc_exc}); continuing with append"
        )
        try:
            with open(_log_file, "a", encoding="utf-8") as _f:
                _f.write(
                    f"\n--- run resumed at {datetime.now().isoformat(timespec='seconds')} ---\n"
                )
        except Exception as _app_exc:
            print(f"W  could not append marker to {_log_file.name}: {_app_exc}")
    _stop_tail = _threading.Event()

    def _tail_log_to_stdout(log_path: Path, stop_evt: _threading.Event) -> None:
        import time

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(
                    0, 2
                )  # seek to end (file is empty at start, robust for later writes)
                while not stop_evt.is_set():
                    line = fh.readline()
                    if line:
                        print(f"[PIPELINE] {line}", end="", flush=True)
                    else:
                        time.sleep(0.5)
        except Exception as exc:
            print(f"[PIPELINE tail error] {exc}", flush=True)

    _tail_thread = _threading.Thread(
        target=_tail_log_to_stdout, args=(_log_file, _stop_tail), daemon=True
    )
    _tail_thread.start()
    print(f"OK  Log tail started → {_log_file}")

    # Clean stale checkpoints so graph starts fresh (grows unboundedly across runs).
    # Skip deletion when --resume so the kernel can restore from the prior checkpoint.
    ckpt_file = REPO_ROOT / "checkpoints.sqlite"
    if resume:
        print(f"OK  --resume mode: preserving {ckpt_file.name} for checkpoint restore")
    elif ckpt_file.exists():
        for _attempt in range(5):
            try:
                # Also delete WAL/SHM sidecar files first
                for _sidecar in ckpt_file.parent.glob("checkpoints.sqlite-*"):
                    try:
                        _sidecar.unlink()
                    except OSError:
                        pass
                ckpt_file.unlink()
                print(f"OK  Deleted stale {ckpt_file.name} for clean run")
                break
            except PermissionError:
                import time as _time

                print(
                    f"WARN {ckpt_file.name} locked (attempt {_attempt+1}/5), retrying in 2s..."
                )
                _time.sleep(2)
        else:
            print(
                f"WARN Could not delete {ckpt_file.name} — another process may hold it open. "
                f"Kill lingering Python kernels and retry."
            )

    try:
        client.execute()
        elapsed = (datetime.now() - start).total_seconds()
        print(f"\nOK  Notebook completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    except Exception as exc:
        elapsed = (datetime.now() - start).total_seconds()
        exc_str = str(exc).encode("utf-8", errors="replace").decode("utf-8")
        print(
            f"\nWARN Notebook raised an exception after {elapsed:.0f}s: {exc_str[:500]}"
        )
        # Collect cell errors from outputs
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    cell_errors.append(
                        {
                            "cell_index": i,
                            "cell_number": i + 1,
                            "ename": out.get("ename", ""),
                            "evalue": out.get("evalue", ""),
                        }
                    )
        if cell_errors:
            print("\nERR Cell errors detected:")
            for err in cell_errors:
                print(
                    f"    Cell {err['cell_number']} (idx {err['cell_index']}): "
                    f"{err['ename']}: {err['evalue']}"
                )

    # NotebookClient is configured with allow_errors=True so long-running runs
    # still save the executed notebook. Surface those cell errors explicitly.
    if not cell_errors:
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    cell_errors.append(
                        {
                            "cell_index": i,
                            "cell_number": i + 1,
                            "ename": out.get("ename", ""),
                            "evalue": out.get("evalue", ""),
                        }
                    )
        if cell_errors:
            print("\nERR Cell errors detected in saved notebook:")
            for err in cell_errors:
                print(
                    f"    Cell {err['cell_number']} (idx {err['cell_index']}): "
                    f"{err['ename']}: {err['evalue']}"
                )

    # Save executed notebook regardless of success
    with open(executed_nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"OK  Executed notebook saved: {executed_nb_path}")

    # Stop log tail thread and drain remaining lines
    _stop_tail.set()
    _tail_thread.join(timeout=3)
    log_size = _log_file.stat().st_size if _log_file.exists() else 0
    print(f"OK  Pipeline log: {_log_file} ({log_size} bytes)")

    return nb, cell_errors, executed_nb_path


def extract_output_paths_from_notebook(nb):
    """Scan cell outputs for printed artifact paths."""
    path_pattern = re.compile(
        r'(?:saved to|report saved|HTML report|Markdown report|PDF report|artifact|path)[:\s]+([^\s\n"\']+\.(?:html|pdf|md|png))',
        re.IGNORECASE,
    )
    found = set()
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            text = ""
            if out.get("output_type") in ("stream", "display_data", "execute_result"):
                text = "".join(
                    out.get("text", out.get("data", {}).get("text/plain", ""))
                )
            for match in path_pattern.findall(text):
                match = match.strip().rstrip(".,;")
                if Path(match).suffix.lower() in (".html", ".pdf", ".md", ".png"):
                    found.add(match)
    return sorted(found)


def scan_artifacts():
    """Scan known output locations for artifacts."""
    patterns = [
        # Primary: IDD_run dirs created by persist_to_drive
        str(OUTPUT_DIR / "IDD_run_*" / "**" / "*.html"),
        str(OUTPUT_DIR / "IDD_run_*" / "**" / "*.pdf"),
        str(OUTPUT_DIR / "IDD_run_*" / "**" / "*.md"),
        str(OUTPUT_DIR / "IDD_run_*" / "**" / "*.png"),
        # Flat in IDD_results
        str(OUTPUT_DIR / "**" / "*.html"),
        str(OUTPUT_DIR / "**" / "*.pdf"),
        str(OUTPUT_DIR / "**" / "*.md"),
        str(OUTPUT_DIR / "**" / "*.png"),
        # Repo-root artifacts dir
        str(REPO_ROOT / "artifacts" / "**" / "*.html"),
        str(REPO_ROOT / "artifacts" / "**" / "*.pdf"),
        str(REPO_ROOT / "artifacts" / "**" / "*.md"),
        str(REPO_ROOT / "artifacts" / "**" / "*.png"),
        str(REPO_ROOT / "WORKING_DIRECTORY" / "artifacts" / "**" / "*.html"),
        str(REPO_ROOT / "WORKING_DIRECTORY" / "artifacts" / "**" / "*.pdf"),
        str(REPO_ROOT / "WORKING_DIRECTORY" / "artifacts" / "**" / "*.md"),
        str(REPO_ROOT / "WORKING_DIRECTORY" / "artifacts" / "**" / "*.png"),
    ]
    found = {}
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            ext = Path(p).suffix.lower()
            found.setdefault(ext, []).append(p)
    return found


def print_artifact_summary(artifacts_by_ext, notebook_paths):
    print("\n" + "=" * 60)
    print("ARTIFACT SUMMARY")
    print("=" * 60)

    ext_labels = {".html": "HTML", ".pdf": "PDF", ".md": "Markdown", ".png": "PNG"}
    all_ok = True

    for ext, label in ext_labels.items():
        paths = artifacts_by_ext.get(ext, [])
        if paths:
            print(f"OK  {label} ({len(paths)} file(s)):")
            for p in paths:
                print(f"    {p}")
        else:
            print(f"MISS {label}: none found")
            all_ok = False

    if notebook_paths:
        print("\nPaths extracted from notebook cell outputs:")
        for p in notebook_paths:
            print(f"    {p}")

    print(
        "\n"
        + (
            "OK  All artifact types present."
            if all_ok
            else "WARN Some artifact types missing — check notebook execution."
        )
    )
    print("=" * 60)
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Run IDD v5 patched notebook headlessly (override with IDD_NOTEBOOK)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from last SQLite checkpoint. Reads thread_id from current_run_thread_id.txt, "
            "writes _idd_resume.flag so the notebook kernel picks it up, and skips "
            "checkpoints.sqlite deletion."
        ),
    )
    parser.add_argument(
        "--check-langsmith",
        action="store_true",
        help=(
            "Load LangSmith env vars, verify CLI availability, and probe the child "
            "Jupyter kernel without running the full notebook. Secret values are never printed."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check six scientific imports and install-flag inheritance in Jupyter; not a full runtime/environment proof.",
    )
    args = parser.parse_args()

    print(f"OK  Selected notebook: {NOTEBOOK_PATH}")
    if not NOTEBOOK_PATH.is_file():
        print(f"ERR Notebook not found: {NOTEBOOK_PATH}")
        print("    Set IDD_NOTEBOOK to an existing notebook filename if overriding the default.")
        sys.exit(1)

    if args.preflight:
        configure_skip_notebook_installs()
        kernel_name = select_kernel_name()
        sci_ok = probe_kernel_scientific_stack(kernel_name)
        sys.exit(0 if sci_ok else 1)

    if args.check_langsmith:
        loaded = load_langsmith_env()
        cli_ok = check_langsmith_cli()
        kernel_ok = probe_langsmith_kernel_env()
        if not loaded:
            print("ERR LangSmith env was not loaded")
        sys.exit(0 if loaded and cli_ok and kernel_ok else 1)

    resume_flag_path = REPO_ROOT / "_idd_resume.flag"
    tid_path = REPO_ROOT / "current_run_thread_id.txt"

    if args.resume:
        if not tid_path.exists():
            print(
                "ERR --resume: current_run_thread_id.txt not found — no prior run to resume"
            )
            sys.exit(1)
        saved_tid = tid_path.read_text(encoding="utf-8").strip()
        if not saved_tid:
            print("ERR --resume: current_run_thread_id.txt is empty")
            sys.exit(1)
        ckpt = REPO_ROOT / "checkpoints.sqlite"
        if not ckpt.exists():
            print(
                "ERR --resume: checkpoints.sqlite not found — cannot resume without checkpoint"
            )
            sys.exit(1)
        # Write resume flag; notebook kernel reads this file to activate resume mode
        resume_flag_path.write_text(saved_tid, encoding="utf-8")
        print(f"OK  --resume: will resume thread_id={saved_tid}")
        print(f"OK  --resume: _idd_resume.flag written, checkpoints.sqlite preserved")
    else:
        # Fresh run: remove resume flag so notebook doesn't accidentally resume
        if resume_flag_path.exists():
            resume_flag_path.unlink()
        print("OK  Fresh run (resume flag cleared)")

    skip_notebook_installs = configure_skip_notebook_installs()
    load_api_key()
    load_langsmith_env()

    if not check_nbclient():
        if resume_flag_path.exists():
            resume_flag_path.unlink()
        sys.exit(1)

    kernel_name = select_kernel_name()
    if (
        skip_notebook_installs.strip().lower() in {"1", "true", "yes", "on"}
        and not probe_kernel_scientific_stack(kernel_name)
    ):
        print("ERR Scientific stack preflight failed; aborting live notebook execution.")
        if resume_flag_path.exists():
            resume_flag_path.unlink()
        sys.exit(1)

    nb, cell_errors, executed_nb_path = execute_notebook(resume=args.resume)
    notebook_paths = extract_output_paths_from_notebook(nb)
    artifacts = scan_artifacts()
    success = print_artifact_summary(artifacts, notebook_paths)

    # Clean up resume flag after run completes (success or failure)
    if resume_flag_path.exists():
        resume_flag_path.unlink()

    if cell_errors:
        print(f"\nERR {len(cell_errors)} cell error(s) occurred during execution:")
        for err in cell_errors:
            print(f"    Cell {err['cell_number']}: {err['ename']}: {err['evalue']}")
        sys.exit(2)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
