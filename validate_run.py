"""validate_run.py — Score an IDD_run_* directory against the 12-criteria
acceptance bar (replaces the legacy 8/8 bar).

Usage:
    python validate_run.py IDD_results/IDD_run_<id>
    python validate_run.py --latest
    python validate_run.py --json
    python validate_run.py --latest --log-path notebook_run_log.txt --window 30

Pure-stdlib, Python 3.12+. No third-party deps.

Exit code: 0 only if score == 12/12, else 1 (suitable for CI gating).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = REPO_ROOT / "notebook_run_log.txt"
DEFAULT_RESULTS_DIR = REPO_ROOT / "IDD_results"
DEFAULT_WINDOW_MIN = 30

# Allowlist of harmless .txt names inside reports/ — anything else fails C11.
TXT_ALLOWLIST = {"manifest.txt", "index.txt"}

# Fail-on-match suffix patterns for stub/marker txt files (C11).
MARKER_TXT_RE = re.compile(
    r"_(ack|commit|ready|stub|trigger|note|note2|marker|completion|review|"
    r"submission|submission_init|submission_note|submission_summary|submit|"
    r"summary|overview|placeholder)\.txt$",
    re.IGNORECASE,
)

# Recovery / fallback markers (C3, C5).
RECOVERY_RE = re.compile(r"\bRECOVERY\b|\bW2-BA-finalhop\b|\bW4-")
FALLBACK_RE = re.compile(r"recovery synthesized|W2-BR-FALLBACK|synthesized via",
                         re.IGNORECASE)
TRACEBACK_RE = re.compile(r"Traceback")
FINAL_RE = re.compile(r"\bFINAL\b")
FINAL_FLAGS_RE = re.compile(r"viz=True\s+report=True")

# C12 heuristics.
CORRELATION_RE = re.compile(r"r\s*=\s*-?\d+\.\d+|correlation[^<]{0,40}?-?\d+\.\d+",
                            re.IGNORECASE)
ANOMALY_RE = re.compile(r"anomal|outlier|missing", re.IGNORECASE)

# Time at start of a log line: "HH:MM:SS ".
LOG_TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\b")


@dataclass
class Criterion:
    id: int
    name: str
    pass_: bool
    detail: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pass"] = d.pop("pass_")
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_run(results_dir: Path) -> Path | None:
    if not results_dir.is_dir():
        return None
    candidates = [p for p in results_dir.iterdir()
                  if p.is_dir() and p.name.startswith("IDD_run_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def strip_html_text(html: str) -> str:
    """Remove tags + script/style content, return whitespace-trimmed text."""
    # Drop script & style blocks entirely (their text doesn't count as content).
    html = re.sub(r"<script\b[^>]*>.*?</script>", " ", html,
                  flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<style\b[^>]*>.*?</style>", " ", html,
                  flags=re.IGNORECASE | re.DOTALL)
    # Strip remaining tags.
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def select_final_html(reports_dir: Path) -> Path | None:
    if not reports_dir.is_dir():
        return None
    htmls = list(reports_dir.glob("*.html"))
    if not htmls:
        return None
    priority = (
        re.compile(r"final_submission\.html$", re.IGNORECASE),
        re.compile(r"final_for_supervisor\.html$", re.IGNORECASE),
        re.compile(r"final_for_submission\.html$", re.IGNORECASE),
        re.compile(r"final_report\.html$", re.IGNORECASE),
        re.compile(r"final\.html$", re.IGNORECASE),
        re.compile(r"final.*\.html$", re.IGNORECASE),
    )
    for pat in priority:
        for p in sorted(htmls):
            if pat.search(p.name):
                return p
    # Fallback: largest html.
    return max(htmls, key=lambda p: p.stat().st_size)


def find_reports_dir(run_dir: Path) -> Path | None:
    """Locate reports/ anywhere under the run dir."""
    for p in run_dir.rglob("reports"):
        if p.is_dir():
            return p
    return None


def find_pdfs(run_dir: Path) -> list[Path]:
    return [p for p in run_dir.rglob("*.pdf") if p.is_file()]


def find_figures(run_dir: Path) -> list[Path]:
    figs: list[Path] = []
    for d in run_dir.rglob("figures"):
        if d.is_dir():
            figs.extend(p for p in d.glob("*.png") if p.is_file())
    if not figs:  # fallback — all PNGs anywhere in the run dir
        figs = [p for p in run_dir.rglob("*.png") if p.is_file()]
    return figs


def slug_of(filename: str) -> str:
    """Strip trailing 8-hex hash (and extension) so cosmetic-hash variants of
    the same image collapse to a single slug."""
    stem = Path(filename).stem
    # Trailing __HEX (8) or _HEX (8).
    return re.sub(r"_{1,2}[0-9a-fA-F]{8}$", "", stem).lower()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_log_lines_in_window(log_path: Path, center_mtime: float,
                             window_min: int) -> list[tuple[int, str]]:
    """Return [(lineno, line)] whose HH:MM:SS prefix falls within
    [center - window, center + window], using center_mtime's date."""
    if not log_path.is_file():
        return []
    center = _dt.datetime.fromtimestamp(center_mtime)
    delta = _dt.timedelta(minutes=window_min)
    lo, hi = center - delta, center + delta
    out: list[tuple[int, str]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            m = LOG_TS_RE.match(line)
            if not m:
                continue
            hh, mm, ss = map(int, m.groups())
            try:
                ts = center.replace(hour=hh, minute=mm, second=ss,
                                    microsecond=0)
            except ValueError:
                continue
            # Allow a date roll: pick the date that puts ts closest to center.
            for cand in (ts, ts - _dt.timedelta(days=1),
                         ts + _dt.timedelta(days=1)):
                if lo <= cand <= hi:
                    out.append((i, line.rstrip("\n")))
                    break
    return out


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def evaluate(run_dir: Path, log_path: Path,
             window_min: int = DEFAULT_WINDOW_MIN) -> list[Criterion]:
    run_mtime = run_dir.stat().st_mtime
    log_lines = load_log_lines_in_window(log_path, run_mtime, window_min)

    reports_dir = find_reports_dir(run_dir)
    final_html_path = select_final_html(reports_dir) if reports_dir else None
    final_html_text = ""
    final_html_raw = ""
    if final_html_path:
        try:
            final_html_raw = final_html_path.read_text(
                encoding="utf-8", errors="replace")
            final_html_text = strip_html_text(final_html_raw)
        except OSError:
            pass

    md_sibling = None
    if final_html_path:
        cand = final_html_path.with_suffix(".md")
        if cand.is_file():
            md_sibling = cand

    pdfs = find_pdfs(run_dir)
    figures = find_figures(run_dir)

    crits: list[Criterion] = []

    # 1. FINAL marker present (windowed).
    final_lines = [(n, l) for (n, l) in log_lines if FINAL_RE.search(l)]
    crits.append(Criterion(
        1, "FINAL marker present",
        bool(final_lines),
        f"log line {final_lines[-1][0]}" if final_lines
        else f"no FINAL within ±{window_min}min of run mtime",
    ))

    # 2. viz=True report=True on the FINAL line.
    if final_lines:
        last_final = final_lines[-1][1]
        ok = bool(FINAL_FLAGS_RE.search(last_final))
        crits.append(Criterion(
            2, "viz=True report=True",
            ok,
            "both flags" if ok else f"flags missing in: {last_final[:120]}",
        ))
    else:
        crits.append(Criterion(2, "viz=True report=True", False,
                               "no FINAL line to inspect"))

    # 3. 0 recoveries / W2-BA-finalhop / W4 negatives (windowed).
    rec_hits = [(n, l) for (n, l) in log_lines if RECOVERY_RE.search(l)]
    crits.append(Criterion(
        3, "0 recoveries / W2-BA-finalhop / W4 negatives",
        not rec_hits,
        f"{len(rec_hits)} hits"
        + (f" (first @ line {rec_hits[0][0]})" if rec_hits else ""),
    ))

    # 4. 0 Tracebacks (windowed log + any *.log inside run dir).
    log_tb = [(n, l) for (n, l) in log_lines if TRACEBACK_RE.search(l)]
    extra_tb_files: list[str] = []
    for log_in_run in run_dir.rglob("*.log"):
        try:
            txt = log_in_run.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if TRACEBACK_RE.search(txt):
            extra_tb_files.append(str(log_in_run.relative_to(run_dir)))
    total_tb = len(log_tb) + len(extra_tb_files)
    detail4 = "0 hits" if total_tb == 0 else (
        f"{len(log_tb)} log hits, {len(extra_tb_files)} *.log files: "
        f"{extra_tb_files[:3]}"
    )
    crits.append(Criterion(4, "0 Tracebacks", total_tb == 0, detail4))

    # 5. All structured outputs native Pydantic (no fallback markers).
    fb_hits = [(n, l) for (n, l) in log_lines if FALLBACK_RE.search(l)]
    crits.append(Criterion(
        5, "All structured outputs native Pydantic",
        not fb_hits,
        "no fallback markers" if not fb_hits
        else f"{len(fb_hits)} fallback marker(s) (first @ line {fb_hits[0][0]})",
    ))

    # 6. PDF in run subdir, non-zero size.
    nonzero_pdfs = [p for p in pdfs if p.stat().st_size > 0]
    crits.append(Criterion(
        6, "PDF in run subdir",
        bool(nonzero_pdfs),
        (f"{len(nonzero_pdfs)} pdf(s); first="
         f"{nonzero_pdfs[0].name} ({nonzero_pdfs[0].stat().st_size}B)")
        if nonzero_pdfs else "no non-empty PDF found",
    ))

    # 7. Report HTML text-only content >= 3000 chars.
    text_len = len(final_html_text)
    crits.append(Criterion(
        7, "Report HTML text-only content >= 3000 chars",
        text_len >= 3000,
        f"text_len={text_len} ({final_html_path.name if final_html_path else 'no html'})",
    ))

    # 8. >= 4 distinct sections.
    sections = 0
    src = "html h2/h3"
    if final_html_raw:
        sections = len(re.findall(r"<h[23]\b", final_html_raw, re.IGNORECASE))
    if sections < 4 and md_sibling:
        try:
            md = md_sibling.read_text(encoding="utf-8", errors="replace")
            md_sections = len(re.findall(r"^##\s+", md, re.MULTILINE))
            if md_sections > sections:
                sections, src = md_sections, "md ## headings"
        except OSError:
            pass
    crits.append(Criterion(
        8, "Report contains >= 4 distinct sections",
        sections >= 4,
        f"{sections} sections via {src}",
    ))

    # 9. >= 3 distinct visualizations (sha256 + slug).
    hashes: set[str] = set()
    slugs: set[str] = set()
    for p in figures:
        try:
            hashes.add(sha256_file(p))
        except OSError:
            continue
        slugs.add(slug_of(p.name))
    ok9 = len(hashes) >= 3 and len(slugs) >= 3
    crits.append(Criterion(
        9, "Viz count >= 3 distinct visualizations",
        ok9,
        f"{len(figures)} png file(s), {len(hashes)} unique hash(es), "
        f"{len(slugs)} unique slug(s)",
    ))

    # 10. PDF size >= 30 KB.
    big_pdfs = [p for p in pdfs if p.stat().st_size >= 30 * 1024]
    crits.append(Criterion(
        10, "PDF size >= 30KB",
        bool(big_pdfs),
        (f"{big_pdfs[0].name}={big_pdfs[0].stat().st_size}B" if big_pdfs
         else (f"largest pdf={max((p.stat().st_size for p in pdfs), default=0)}B"
               " (< 30KB)")),
    ))

    # 11. No stub / marker files in reports/.
    bad_txt: list[str] = []
    if reports_dir:
        for p in reports_dir.rglob("*.txt"):
            name = p.name
            if name in TXT_ALLOWLIST:
                continue
            if MARKER_TXT_RE.search(name):
                bad_txt.append(name)
            else:
                # Any other .txt in reports/ is also a fail per spec.
                bad_txt.append(name)
    crits.append(Criterion(
        11, "No stub/marker files",
        not bad_txt,
        ("clean" if not bad_txt
         else f"{len(bad_txt)} marker/txt file(s); first 3: {bad_txt[:3]}"),
    ))

    # 12. Report references actual analyst findings.
    has_corr = bool(CORRELATION_RE.search(final_html_text)) if final_html_text else False
    has_anom = bool(ANOMALY_RE.search(final_html_text)) if final_html_text else False
    ok12 = has_corr and has_anom
    detail12_parts = []
    if not has_corr:
        detail12_parts.append("no correlation values found")
    if not has_anom:
        detail12_parts.append("no anomaly/outlier/missing keyword")
    if ok12:
        detail12_parts.append("correlation + anomaly keyword present")
    crits.append(Criterion(
        12, "Report references actual analyst findings", ok12,
        "; ".join(detail12_parts),
    ))

    return crits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render_human(run_dir: Path, crits: list[Criterion]) -> str:
    score = sum(1 for c in crits if c.pass_)
    total = len(crits)
    lines = ["=== validate_run.py ===",
             f"Run: {run_dir.name}",
             f"Path: {run_dir.as_posix()}",
             "",
             f"{'Criterion':<60}{'Result':<10}Detail"]
    for c in crits:
        head = f"{c.id:<2}. {c.name}"
        lines.append(f"{head:<60}{('PASS' if c.pass_ else 'FAIL'):<10}{c.detail}")
    verdict = "PASS — production bar reached" if score == total \
        else f"FAIL — production bar = {total}/{total}"
    lines += ["", f"SCORE: {score} / {total}        ({verdict})"]
    return "\n".join(lines)


def render_json(run_dir: Path, crits: list[Criterion]) -> str:
    score = sum(1 for c in crits if c.pass_)
    return json.dumps({
        "run": run_dir.name,
        "path": run_dir.as_posix(),
        "score": score,
        "total": len(crits),
        "criteria": [c.to_dict() for c in crits],
    }, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", help="Path to an IDD_run_* directory.")
    p.add_argument("--latest", action="store_true",
                   help="Auto-pick newest IDD_run_* under IDD_results/.")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="Emit machine-readable JSON instead of a table.")
    p.add_argument("--log-path", default=str(DEFAULT_LOG),
                   help="Path to notebook_run_log.txt (default: repo root).")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                   help="Where to look for IDD_run_* dirs when --latest.")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW_MIN,
                   help="Log time-window in minutes around run dir mtime.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.latest:
        run_dir = find_latest_run(Path(args.results_dir))
        if run_dir is None:
            print(f"ERROR: no IDD_run_* dir found in {args.results_dir}",
                  file=sys.stderr)
            return 2
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        print("ERROR: provide a run_dir or --latest", file=sys.stderr)
        return 2

    if not run_dir.is_dir():
        print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
        return 2

    crits = evaluate(run_dir.resolve(), Path(args.log_path),
                     window_min=args.window)
    if args.as_json:
        print(render_json(run_dir, crits))
    else:
        print(render_human(run_dir, crits))

    score = sum(1 for c in crits if c.pass_)
    return 0 if score == len(crits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
