"""Validate IDD final artifacts for user-facing quality.

This complements validate_run.py. The production scorer proves that the
pipeline completed and emitted artifacts; this script checks whether those
artifacts are parseable and usable by a reader.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "IDD_results"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


class ReportHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []
        self.headings: list[str] = []
        self.text_parts: list[str] = []
        self._tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tag = tag.lower()
        if self._tag == "img":
            attr_map = {k.lower(): v for k, v in attrs}
            src = attr_map.get("src")
            if src:
                self.images.append(src)

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        self.text_parts.append(text)
        if self._tag in {"h1", "h2", "h3", "h4"}:
            self.headings.append(text)

    def handle_endtag(self, tag: str) -> None:
        if self._tag == tag.lower():
            self._tag = None


def latest_run(results_dir: Path) -> Path:
    runs = [p for p in results_dir.glob("IDD_run_*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No IDD_run_* directories under {results_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def find_reports_dir(run_dir: Path) -> Path:
    reports = [p for p in run_dir.rglob("reports") if p.is_dir()]
    if not reports:
        raise FileNotFoundError(f"No reports directory under {run_dir}")
    return max(reports, key=lambda p: sum(1 for _ in p.glob("*")))


def final_report_paths(reports_dir: Path) -> tuple[Path, Path, Path]:
    htmls = list(reports_dir.glob("*.html"))
    if not htmls:
        raise FileNotFoundError(f"No HTML report under {reports_dir}")
    canonical = reports_dir / "final_report.html"
    html_path = canonical if canonical.is_file() else max(htmls, key=lambda p: p.stat().st_size)
    md_path = html_path.with_suffix(".md")
    pdf_path = html_path.with_suffix(".pdf")
    return md_path, html_path, pdf_path


def resolve_report_path(reports_dir: Path, ref: str) -> Path:
    normalized = ref.replace("\\", "/")
    return (reports_dir / normalized).resolve()


def is_embedded_image(ref: str) -> bool:
    return ref.lower().startswith("data:image/")


def check_pdf(pdf_path: Path) -> Check:
    if not pdf_path.is_file():
        return Check("parseable PDF", False, f"missing {pdf_path}")
    failures: list[str] = []
    page_count = 0
    text_chars = 0
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        text_chars = sum(len(page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # noqa: BLE001 - validation reports exact parser error
        failures.append(f"pypdf: {type(exc).__name__}: {exc}")
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        page_count = max(page_count, doc.page_count)
        text_chars = max(text_chars, sum(len(page.get_text("text")) for page in doc))
    except Exception as exc:  # noqa: BLE001 - validation reports exact parser error
        failures.append(f"PyMuPDF: {type(exc).__name__}: {exc}")
    passed = not failures and page_count > 0 and text_chars > 500
    detail = f"pages={page_count}, text_chars={text_chars}"
    if failures:
        detail += "; " + " | ".join(failures)
    return Check("parseable PDF", passed, detail)


def check_embeds(md_path: Path, html_path: Path) -> list[Check]:
    checks: list[Check] = []
    reports_dir = html_path.parent
    html = html_path.read_text(encoding="utf-8", errors="replace")
    md = (
        md_path.read_text(encoding="utf-8", errors="replace")
        if md_path.is_file()
        else ""
    )

    parser = ReportHtmlParser()
    parser.feed(html)
    md_images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)

    html_missing = [
        src
        for src in parser.images
        if not is_embedded_image(src)
        and not resolve_report_path(reports_dir, src).is_file()
    ]
    md_missing = [
        src for src in md_images if not resolve_report_path(reports_dir, src).is_file()
    ]

    checks.append(
        Check(
            "HTML embeds visualizations",
            len(parser.images) >= 3 and not html_missing,
            f"img_tags={len(parser.images)}, missing={html_missing[:3]}",
        )
    )
    checks.append(
        Check(
            "Markdown embeds visualizations",
            len(md_images) >= 3 and not md_missing,
            f"image_tags={len(md_images)}, missing={md_missing[:3]}",
        )
    )
    return checks


def check_root_html_embeds(run_dir: Path) -> Check:
    root_html = run_dir / "final_report.html"
    if not root_html.is_file():
        return Check("root HTML embeds visualizations", False, "missing final_report.html")
    parser = ReportHtmlParser()
    parser.feed(root_html.read_text(encoding="utf-8", errors="replace"))
    missing = [
        src
        for src in parser.images
        if not is_embedded_image(src)
        and not (root_html.parent / src.replace("\\", "/")).resolve().is_file()
    ]
    return Check(
        "root HTML embeds visualizations",
        len(parser.images) >= 3 and not missing,
        f"img_tags={len(parser.images)}, missing={missing[:3]}",
    )


def check_repetition(html_path: Path) -> Check:
    html = html_path.read_text(encoding="utf-8", errors="replace")
    parts = [
        re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", part)).strip().lower()
        for part in re.split(r"\n\s*\n|</p>|</h2>|</h3>", html, flags=re.I)
    ]
    paragraphs = [p for p in parts if len(p) > 120]
    counts = Counter(paragraphs)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    passed = len(paragraphs) >= 5 and duplicates <= max(2, len(paragraphs) // 5)
    return Check(
        "low duplicate paragraph rate",
        passed,
        f"paragraphs={len(paragraphs)}, unique={len(counts)}, duplicates={duplicates}",
    )


def check_heading_quality(html_path: Path) -> Check:
    parser = ReportHtmlParser()
    parser.feed(html_path.read_text(encoding="utf-8", errors="replace"))
    normalized = [re.sub(r"\s+", " ", h).strip().casefold() for h in parser.headings]
    adjacent_duplicates = [
        parser.headings[idx]
        for idx in range(1, len(normalized))
        if normalized[idx] and normalized[idx] == normalized[idx - 1]
    ]
    repeated_titles = [
        heading
        for heading, count in Counter(normalized).items()
        if heading and count > 1
    ]
    passed = not adjacent_duplicates and len(repeated_titles) <= 1
    return Check(
        "clean report headings",
        passed,
        (
            f"headings={len(parser.headings)}, "
            f"adjacent_duplicates={adjacent_duplicates[:3]}, "
            f"repeated={repeated_titles[:3]}"
        ),
    )


def check_stakeholder_readability(html_path: Path) -> Check:
    html = html_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
    lower = text.casefold()
    scaffold_patterns = [
        r"\bthis section should\b",
        r"\bthis section addresses\b",
        r"\bpurpose:\s*provide\b",
        r"\brecommended next steps for this section\b",
        r"\bthe following data signals will be needed\b",
        r"\bwrite as markdown\b",
        r"\breturn only\b",
        r"\bcall the .*structured response\b",
    ]
    scaffold_hits = [
        pattern for pattern in scaffold_patterns if re.search(pattern, lower)
    ]
    stakeholder_terms = {
        "finding": len(re.findall(r"\bfinding|findings|showed|shows|indicates\b", lower)),
        "implication": len(re.findall(r"\bimplication|meaning|suggests|therefore|risk|opportunity\b", lower)),
        "recommendation": len(re.findall(r"\brecommend|recommendation|next action|next step|should\b", lower)),
    }
    technical_dump_hits = len(
        re.findall(
            r"(?:\{['\"][^{}]{1,80}['\"]\s*:|\b<class\b|pydantic|traceback|structured_response|AIMessage\()",
            text,
        )
    )
    passed = (
        len(text) >= 3000
        and not scaffold_hits
        and all(count >= 1 for count in stakeholder_terms.values())
        and technical_dump_hits == 0
    )
    return Check(
        "stakeholder-readable narrative",
        passed,
        (
            f"text_chars={len(text)}, scaffold_hits={scaffold_hits[:3]}, "
            f"terms={stakeholder_terms}, technical_dump_hits={technical_dump_hits}"
        ),
    )


def check_no_stray_text_artifacts(run_dir: Path) -> Check:
    allowed_names = {
        "final_report.md",
    }
    text_files = [
        p
        for p in run_dir.rglob("*.txt")
        if p.name not in allowed_names
    ]
    suspicious_patterns = re.compile(
        r"(final|ready|submit|submission|stop|respond|metadata|summary|outline|status)",
        re.I,
    )
    suspicious = [p for p in text_files if suspicious_patterns.search(p.name)]
    passed = not text_files and not suspicious
    return Check(
        "no stray text artifacts",
        passed,
        (
            f"txt_files={len(text_files)}, "
            f"suspicious={[str(p.relative_to(run_dir)) for p in suspicious[:5]]}"
        ),
    )


def check_chart_set(run_dir: Path) -> Check:
    names = sorted({p.stem.lower() for p in run_dir.rglob("*.png")})
    descriptive_names = [
        re.sub(r"_{1,2}[0-9a-f]{8,32}$", "", name)
        for name in names
        if not re.fullmatch(r"[0-9a-f]{8,32}", name)
    ]
    descriptive_names = sorted(set(descriptive_names))
    weak_tokens = ("_id_", "of_id", "between_id", " id ", "by_name", "mean_id")
    id_dominated = bool(descriptive_names) and all(
        any(token in name for token in weak_tokens) for name in descriptive_names
    )
    passed = len(names) >= 3 and not id_dominated
    return Check(
        "non-trivial chart selection",
        passed,
        f"unique_pngs={len(names)}, id_dominated={id_dominated}, names={descriptive_names[:5]}",
    )


def evaluate(run_dir: Path) -> list[Check]:
    reports_dir = find_reports_dir(run_dir)
    md_path, html_path, pdf_path = final_report_paths(reports_dir)
    checks = [check_pdf(pdf_path)]
    checks.extend(check_embeds(md_path, html_path))
    checks.append(check_root_html_embeds(run_dir))
    checks.append(check_repetition(html_path))
    checks.append(check_heading_quality(html_path))
    checks.append(check_stakeholder_readability(html_path))
    checks.append(check_no_stray_text_artifacts(run_dir))
    checks.append(check_chart_set(run_dir))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    run_dir = (
        latest_run(args.results_dir)
        if args.latest or args.run_dir is None
        else args.run_dir
    )
    checks = evaluate(run_dir)
    print("=== validate_artifact_quality.py ===")
    print(f"Run: {run_dir.name}")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{check.name:<34} {status:<5} {check.detail}")
    score = sum(check.passed for check in checks)
    print(f"SCORE: {score} / {len(checks)}")
    return 0 if score == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
