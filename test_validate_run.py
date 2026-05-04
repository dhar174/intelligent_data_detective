"""Tests for validate_run.py — the 12-criteria production-bar scorer."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import validate_run as vr  # noqa: E402


RUN88 = REPO_ROOT / "IDD_results" / "IDD_run_run_default_id-20260423-1026-3ecefbee"
LOG = REPO_ROOT / "notebook_run_log.txt"


# --------------------------------------------------------------------------
# Run 88 baseline — must fail the production bar.
# --------------------------------------------------------------------------

@pytest.mark.skipif(not RUN88.is_dir(), reason="Run 88 fixture missing")
def test_score_run88_is_below_threshold():
    crits = vr.evaluate(RUN88, LOG, window_min=30)
    score = sum(1 for c in crits if c.pass_)
    assert len(crits) == 12, "must score all 12 criteria"
    assert 4 <= score <= 9, (
        f"Run 88 expected to fall in the 4..9 range (the documented "
        f"baseline-to-beat); got {score}/12"
    )
    # Specific known-bad criteria for Run 88.
    by_id = {c.id: c for c in crits}
    assert by_id[7].pass_ is False, "C7 (text content) should fail on Run 88"
    assert by_id[10].pass_ is False, "C10 (PDF >=30KB) should fail on Run 88"
    assert by_id[11].pass_ is False, "C11 (no stub files) should fail on Run 88"


# --------------------------------------------------------------------------
# C7 — HTML text stripping.
# --------------------------------------------------------------------------

def test_html_text_strip_basic():
    html = (
        "<html><head><title>x</title>"
        "<style>body{color:red}</style>"
        "<script>var a=1;</script></head>"
        "<body><h1>Hello</h1>   <p>World&nbsp;of   IDD</p>"
        "<img src='data:image/png;base64,AAAAAAAA'/></body></html>"
    )
    text = vr.strip_html_text(html)
    assert "Hello" in text
    assert "World" in text and "IDD" in text
    # script + style content removed
    assert "var a=1" not in text
    assert "color:red" not in text
    # No tag fragments left behind
    assert "<" not in text and ">" not in text


def test_html_text_strip_collapses_whitespace():
    html = "<p>one</p>\n\n\n<p>two</p>\t\t<p>three</p>"
    text = vr.strip_html_text(html)
    assert text == "one two three"


# --------------------------------------------------------------------------
# C9 — PNG distinctness.
# --------------------------------------------------------------------------

def _make_png(path: Path, payload: bytes) -> None:
    # Not a real PNG — validate_run.py only hashes raw bytes.
    path.write_bytes(payload)


def test_png_distinctness_collapse_same_image_diff_filename(tmp_path):
    figs_dir = tmp_path / "figures"
    figs_dir.mkdir()
    same = b"PIXELS-A" * 32
    other = b"PIXELS-B" * 32
    third = b"PIXELS-C" * 32
    # Three filenames that should collapse to the SAME slug
    # ("sample__hist__value") because they only differ in the trailing 8-hex.
    _make_png(figs_dir / "sample__hist__value__aaaaaaaa.png", same)
    _make_png(figs_dir / "sample__hist__value__bbbbbbbb.png", same)  # dup hash
    _make_png(figs_dir / "sample__hist__value__cccccccc.png", same)  # dup hash
    # Even adding two distinct content files but same slug shouldn't pass:
    crits = vr.evaluate(tmp_path, LOG, window_min=10**6)
    c9 = next(c for c in crits if c.id == 9)
    assert c9.pass_ is False, (
        "Three identical-content PNGs sharing one slug must NOT pass C9")

    # Now add three genuinely distinct images with distinct slugs:
    _make_png(figs_dir / "sample__scatter__xy__11111111.png", other)
    _make_png(figs_dir / "sample__bar__cat__22222222.png", third)
    crits = vr.evaluate(tmp_path, LOG, window_min=10**6)
    c9 = next(c for c in crits if c.id == 9)
    assert c9.pass_ is True, f"Expected C9 pass after 3 distinct viz; got: {c9.detail}"


def test_slug_strips_trailing_hex():
    assert vr.slug_of("sample__hist__value__f7a42858.png") == "sample__hist__value"
    assert vr.slug_of("foo_deadbeef.png") == "foo"
    assert vr.slug_of("no_hash_here.png") == "no_hash_here"


# --------------------------------------------------------------------------
# C11 — marker / stub file detection.
# --------------------------------------------------------------------------

def test_marker_file_detection(tmp_path):
    reports = tmp_path / "artifacts" / "run_x" / "reports"
    reports.mkdir(parents=True)
    # Bad files (every one of these must trigger C11 fail).
    for n in [
        "EDA_final_submission_marker.txt",
        "EDA_Report_final_ack.txt",
        "EDA_Report_final_commit.txt",
        "EDA_Report_final_ready.txt",
        "EDA_Report_final_stub.txt",
        "EDA_Report_final_trigger.txt",
        "EDA_Report_final_note.txt",
        "EDA_Report_final_completion.txt",
        "EDA_Report_final_review.txt",
        "EDA_Report.html_note.txt",  # generic .txt in reports/ — also bad
    ]:
        (reports / n).write_text("x")
    # Allowed files
    (reports / "manifest.txt").write_text("ok")
    (reports / "index.txt").write_text("ok")

    crits = vr.evaluate(tmp_path, LOG, window_min=10**6)
    c11 = next(c for c in crits if c.id == 11)
    assert c11.pass_ is False
    assert "10" in c11.detail or "marker" in c11.detail.lower()

    # Remove all bad files; allowlist remains — C11 must now pass.
    for p in list(reports.glob("*.txt")):
        if p.name not in vr.TXT_ALLOWLIST:
            p.unlink()
    crits = vr.evaluate(tmp_path, LOG, window_min=10**6)
    c11 = next(c for c in crits if c.id == 11)
    assert c11.pass_ is True, c11.detail


# --------------------------------------------------------------------------
# Sanity — find_latest_run picks newest mtime.
# --------------------------------------------------------------------------

def test_find_latest_run_picks_newest(tmp_path):
    base = tmp_path / "IDD_results"
    base.mkdir()
    a = base / "IDD_run_old"; a.mkdir()
    b = base / "IDD_run_new"; b.mkdir()
    c = base / "not_a_run"; c.mkdir()
    os.utime(a, (1_000_000, 1_000_000))
    os.utime(b, (2_000_000, 2_000_000))
    os.utime(c, (3_000_000, 3_000_000))  # not matching prefix
    assert vr.find_latest_run(base) == b


# --------------------------------------------------------------------------
# Output shape.
# --------------------------------------------------------------------------

def test_render_json_shape(tmp_path):
    # Minimal empty run dir — every criterion will fail, but output must be
    # well-formed JSON with the required keys.
    crits = vr.evaluate(tmp_path, LOG, window_min=10**6)
    import json as _json
    out = _json.loads(vr.render_json(tmp_path, crits))
    assert set(out.keys()) >= {"run", "score", "total", "criteria"}
    assert out["total"] == 12
    for c in out["criteria"]:
        assert set(c.keys()) == {"id", "name", "pass", "detail"}
