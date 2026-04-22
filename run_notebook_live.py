"""
run_notebook_live.py -- Headless execution of IntelligentDataDetective_beta_v5_patched.ipynb

Usage:
    python run_notebook_live.py

Requirements:
    pip install nbclient nbformat jupyter_client ipykernel

NEVER cancel this script -- the notebook takes 6-25 minutes to complete.
"""
import os
import sys
import glob
import re
import subprocess
from datetime import datetime
from pathlib import Path

# Ensure stdout/stderr use UTF-8 so Unicode in notebook output doesn't crash the runner
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent
PATCHED_NB = REPO_ROOT / "IntelligentDataDetective_beta_v5_patched.ipynb"
OUTPUT_DIR = REPO_ROOT / "IDD_results"
TIMEOUT = 3600  # 60 minutes — analyst/data_cleaner each cap at ~15-20 min with recovery


def load_api_key():
    """Load OPENAI_API_KEY from process env or Windows User env scope."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        key = subprocess.check_output(
            ["powershell", "-command",
             '[System.Environment]::GetEnvironmentVariable("OPENAI_API_KEY","User")'],
            text=True,
        ).strip()
    except Exception as e:
        print(f"Warning: Could not read OPENAI_API_KEY from User env: {e}")
        key = ""
    if key:
        os.environ["OPENAI_API_KEY"] = key
        print("OK  OPENAI_API_KEY loaded from User environment scope")
    else:
        print("ERR OPENAI_API_KEY is not set -- notebook will likely fail LLM calls")
    return key


def check_nbclient():
    try:
        import nbclient  # noqa
        import nbformat  # noqa
        print("OK  nbclient and nbformat are available")
        return True
    except ImportError as e:
        print(f"ERR Missing dependency: {e}")
        print("    Install with: pip install nbclient nbformat jupyter_client ipykernel")
        return False


def execute_notebook():
    import asyncio
    import nbformat
    from nbclient import NotebookClient

    # Suppress ZMQ/tornado warning on Windows about ProactorEventLoop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if not PATCHED_NB.exists():
        print(f"ERR Patched notebook not found: {PATCHED_NB}")
        print("    Run: python _patch_notebook.py  first")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    executed_nb_path = OUTPUT_DIR / f"executed_{timestamp}.ipynb"

    print("\n" + "=" * 60)
    print(f"Executing: {PATCHED_NB.name}")
    print(f"Timeout:   {TIMEOUT}s ({TIMEOUT // 60} min)")
    print(f"Output:    {executed_nb_path}")
    print("=" * 60 + "\n")

    with open(PATCHED_NB, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Determine available kernel: prefer py312-codex (Python 3.12) over stale python3 spec
    import subprocess as _sp
    try:
        _ks = _sp.check_output(["jupyter", "kernelspec", "list", "--json"], text=True)
        import json as _json
        _kernels = _json.loads(_ks).get("kernelspecs", {})
        if "py312-codex" in _kernels:
            kernel_name = "py312-codex"
        elif "python3" in _kernels:
            kernel_name = "python3"
        else:
            kernel_name = list(_kernels.keys())[0]
    except Exception:
        kernel_name = "py312-codex"
    print(f"OK  Using kernel: {kernel_name}")

    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name=kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )

    cell_errors = []
    start = datetime.now()

    # Start log-tail thread: streams notebook_run_log.txt to stdout in real time
    import threading as _threading

    _log_file = REPO_ROOT / "notebook_run_log.txt"
    _log_file.write_text("", encoding="utf-8")  # truncate / create
    _stop_tail = _threading.Event()

    def _tail_log_to_stdout(log_path: Path, stop_evt: _threading.Event) -> None:
        import time
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(0, 2)  # seek to end (file is empty at start, robust for later writes)
                while not stop_evt.is_set():
                    line = fh.readline()
                    if line:
                        print(f"[PIPELINE] {line}", end="", flush=True)
                    else:
                        time.sleep(0.5)
        except Exception as exc:
            print(f"[PIPELINE tail error] {exc}", flush=True)

    _tail_thread = _threading.Thread(target=_tail_log_to_stdout, args=(_log_file, _stop_tail), daemon=True)
    _tail_thread.start()
    print(f"OK  Log tail started → {_log_file}")

    # Clean stale checkpoints so graph starts fresh (grows unboundedly across runs)
    ckpt_file = REPO_ROOT / "checkpoints.sqlite"
    if ckpt_file.exists():
        ckpt_file.unlink()
        print(f"OK  Deleted stale {ckpt_file.name} for clean run")

    try:
        client.execute()
        elapsed = (datetime.now() - start).total_seconds()
        print(f"\nOK  Notebook completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    except Exception as exc:
        elapsed = (datetime.now() - start).total_seconds()
        exc_str = str(exc).encode("utf-8", errors="replace").decode("utf-8")
        print(f"\nWARN Notebook raised an exception after {elapsed:.0f}s: {exc_str[:500]}")
        # Collect cell errors from outputs
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    cell_errors.append({
                        "cell_index": i,
                        "cell_number": i + 1,
                        "ename": out.get("ename", ""),
                        "evalue": out.get("evalue", ""),
                    })
        if cell_errors:
            print("\nERR Cell errors detected:")
            for err in cell_errors:
                print(f"    Cell {err['cell_number']} (idx {err['cell_index']}): "
                      f"{err['ename']}: {err['evalue']}")

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
                text = "".join(out.get("text", out.get("data", {}).get("text/plain", "")))
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

    print("\n" + ("OK  All artifact types present." if all_ok else
                  "WARN Some artifact types missing — check notebook execution."))
    print("=" * 60)
    return all_ok


def main():
    load_api_key()

    if not check_nbclient():
        sys.exit(1)

    nb, cell_errors, executed_nb_path = execute_notebook()
    notebook_paths = extract_output_paths_from_notebook(nb)
    artifacts = scan_artifacts()
    success = print_artifact_summary(artifacts, notebook_paths)

    if cell_errors:
        print(f"\nERR {len(cell_errors)} cell error(s) occurred during execution:")
        for err in cell_errors:
            print(f"    Cell {err['cell_number']}: {err['ename']}: {err['evalue']}")
        sys.exit(2)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
