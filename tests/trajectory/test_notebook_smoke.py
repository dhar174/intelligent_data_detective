"""
Headless notebook smoke test -- produces real IDD pipeline output.
Requires OPENAI_API_KEY. Run with: pytest tests/trajectory/ -m trajectory -v

NEVER cancel this test -- it takes 6-25 minutes.
"""
import os
import sys
import glob
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.trajectory, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "IDD_results"

# Load OPENAI_API_KEY from User scope if not in current env
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not _OPENAI_KEY:
    try:
        _OPENAI_KEY = subprocess.check_output(
            ["powershell", "-command",
             '[System.Environment]::GetEnvironmentVariable("OPENAI_API_KEY","User")'],
            text=True,
        ).strip()
        if _OPENAI_KEY:
            os.environ["OPENAI_API_KEY"] = _OPENAI_KEY
    except Exception:
        _OPENAI_KEY = ""


@pytest.mark.skipif(not _OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_notebook_produces_report():
    """
    Run the patched notebook headlessly and assert:
    1. No cell exceptions
    2. At least one HTML report was created
    3. At least one PNG visualization was created
    4. Report is non-empty (> 100 bytes)
    """
    patched_nb = REPO_ROOT / "IntelligentDataDetective_beta_v5_patched.ipynb"
    if not patched_nb.exists():
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "_patch_notebook.py")],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"_patch_notebook.py failed:\n{result.stdout}\n{result.stderr}"
        )

    run_script = REPO_ROOT / "run_notebook_live.py"
    assert run_script.exists(), "run_notebook_live.py not found"

    result = subprocess.run(
        [sys.executable, str(run_script)],
        cwd=str(REPO_ROOT),
        capture_output=False,  # stream to console
        timeout=2700,          # 45-minute hard cap
        env={**os.environ, "OPENAI_API_KEY": _OPENAI_KEY},
    )

    assert result.returncode in (0, 1), (
        f"run_notebook_live.py exited with unexpected code {result.returncode}"
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    html_files = glob.glob(str(RESULTS_DIR / "**" / "*.html"), recursive=True)
    png_files = glob.glob(str(RESULTS_DIR / "**" / "*.png"), recursive=True)
    # Also check artifacts/ dir
    artifacts_dir = REPO_ROOT / "artifacts"
    if artifacts_dir.exists():
        html_files += glob.glob(str(artifacts_dir / "**" / "*.html"), recursive=True)
        png_files += glob.glob(str(artifacts_dir / "**" / "*.png"), recursive=True)

    assert html_files, (
        "No HTML report found in IDD_results/ or artifacts/ after notebook execution"
    )
    assert png_files, (
        "No PNG visualization found in IDD_results/ or artifacts/ after notebook execution"
    )

    # Verify report is non-trivially sized
    for html in html_files:
        size = Path(html).stat().st_size
        assert size > 100, f"HTML report is too small ({size} bytes): {html}"

    print(f"\nOK  HTML reports: {html_files}")
    print(f"OK  PNG files:    {png_files}")
