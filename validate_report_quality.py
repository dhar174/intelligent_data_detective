"""Validate that the final IDD report is agent-generated stakeholder prose.

This complements validate_run.py and validate_artifact_quality.py. Those gates
prove completion and artifact usability; this one fails scaffold/blueprint
reports and verifies the no-bypass report-agent markers emitted by W11.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "IDD_results"
DEFAULT_LOG = REPO_ROOT / "notebook_run_log.txt"

SCAFFOLD_PHRASES = (
    "This section addresses:",
    "The cleaned dataset context is:",
    "The cleaning record indicates:",
    "Visual evidence assigned to this section",
    "Recommended next steps for this section are:",
    "Summarize the dataset, major cleaning actions",
    "The persisted analysis summary states:",
)

STAKEHOLDER_TERMS = (
    "key finding",
    "key findings",
    "what this means",
    "recommend",
    "next step",
    "business",
    "stakeholder",
    "implication",
    "decision",
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


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


def final_report_text(run_dir: Path) -> tuple[Path, str]:
    reports_dir = find_reports_dir(run_dir)
    markdown_reports = list(reports_dir.glob("*.md"))
    if markdown_reports:
        path = max(markdown_reports, key=lambda p: p.stat().st_size)
        return path, path.read_text(encoding="utf-8", errors="replace")
    html_reports = list(reports_dir.glob("*.html"))
    if not html_reports:
        raise FileNotFoundError(f"No Markdown or HTML report under {reports_dir}")
    path = max(html_reports, key=lambda p: p.stat().st_size)
    parser = TextExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return path, "\n".join(parser.parts)


def check_no_scaffold_phrases(text: str) -> Check:
    hits = [phrase for phrase in SCAFFOLD_PHRASES if phrase.lower() in text.lower()]
    return Check(
        "no scaffold/blueprint phrases",
        not hits,
        f"hits={hits[:8]}",
    )


def check_executive_summary_is_not_instruction(text: str) -> Check:
    match = re.search(r"(?is)##\s*Executive Summary\s*(.*?)(?:\n##\s+|\Z)", text)
    if not match:
        return Check("executive summary narrative", False, "missing Executive Summary")
    section = re.sub(r"\s+", " ", match.group(1)).strip()
    instruction_like = section.lower().startswith(
        (
            "summarize ",
            "describe ",
            "explain ",
            "write ",
            "this section addresses",
        )
    )
    return Check(
        "executive summary narrative",
        bool(section) and not instruction_like,
        f"chars={len(section)}, instruction_like={instruction_like}",
    )


def check_not_dict_dump(text: str) -> Check:
    dictish = len(re.findall(r"\b\w+=", text)) + len(
        re.findall(r"['\"][A-Za-z_]+['\"]\s*:", text)
    )
    words = max(1, len(re.findall(r"\w+", text)))
    ratio = dictish / words
    return Check(
        "not dominated by object/dict dumps",
        ratio < 0.025,
        f"dictish_tokens={dictish}, words={words}, ratio={ratio:.4f}",
    )


def check_stakeholder_language(text: str) -> Check:
    lowered = text.lower()
    hits = sorted({term for term in STAKEHOLDER_TERMS if term in lowered})
    return Check(
        "stakeholder-facing findings/actions",
        len(hits) >= 3,
        f"markers={hits}",
    )


def check_agent_authenticity_log(log_path: Path) -> list[Check]:
    if not log_path.is_file():
        return [
            Check("report section agents invoked", False, f"missing log {log_path}"),
            Check("report packager agent invoked", False, f"missing log {log_path}"),
            Check("final route no-bypass proof", False, f"missing log {log_path}"),
        ]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    section_starts = len(re.findall(r"STATE report_section_agent\.invoke\.start", text))
    section_ends = len(re.findall(r"STATE report_section_agent\.invoke\.end", text))
    packager_ok = "STATE report_packager_agent.invoke.end" in text
    no_bypass_route = bool(
        re.search(
            r"STATE route_to_writer .*report_ready=True .*agent_ready=True",
            text,
        )
    )
    return [
        Check(
            "report section agents invoked",
            section_starts >= 4 and section_ends >= 4,
            f"starts={section_starts}, ends={section_ends}",
        ),
        Check(
            "report packager agent invoked",
            packager_ok,
            f"packager_end={packager_ok}",
        ),
        Check(
            "final route no-bypass proof",
            no_bypass_route,
            f"agent_ready_route={no_bypass_route}",
        ),
    ]


def evaluate(run_dir: Path, log_path: Path) -> list[Check]:
    report_path, text = final_report_text(run_dir)
    checks = [
        Check(
            "report loaded",
            bool(text.strip()),
            f"path={report_path}, chars={len(text)}",
        ),
        check_no_scaffold_phrases(text),
        check_executive_summary_is_not_instruction(text),
        check_not_dict_dump(text),
        check_stakeholder_language(text),
    ]
    checks.extend(check_agent_authenticity_log(log_path))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()

    run_dir = (
        latest_run(args.results_dir)
        if args.latest or args.run_dir is None
        else args.run_dir
    )
    checks = evaluate(run_dir, args.log_path)
    print("=== validate_report_quality.py ===")
    print(f"Run: {run_dir.name}")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{check.name:<38} {status:<5} {check.detail}")
    score = sum(check.passed for check in checks)
    print(f"SCORE: {score} / {len(checks)}")
    return 0 if score == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
