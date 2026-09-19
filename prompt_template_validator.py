#!/usr/bin/env python3
"""
Prompt Template Formatting Validation Tool

Validates ChatPromptTemplate instances defined in the Intelligent Data Detective
notebook for:

- mistaken double-braced placeholders such as ``{{user_prompt}}``
- unmatched / malformed Python-format braces
- placeholder / ``.partial(...)`` consistency
- MessagesPlaceholder naming conventions

Template extraction and string analysis use Python's ``ast`` module rather than
regular expressions over raw source. This makes multiline definitions,
triple-quoted strings, escaped quotes, f-strings, and chained calls far less
fragile.

The default validation target is the committed runnable patched notebook.
Override it with a positional path, ``$IDD_NOTEBOOK``, or ``$NOTEBOOK_PATH``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from string import Formatter


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class IssueType(str, Enum):
    PARSE_ERROR = "parse_error"
    DOUBLE_BRACE_PLACEHOLDER = "double_brace_placeholder"
    AMBIGUOUS_DOUBLE_BRACES = "ambiguous_double_braces"
    UNMATCHED_BRACES = "unmatched_braces"
    UNDECLARED_PLACEHOLDER = "undeclared_placeholder"
    DYNAMIC_PARTIAL_ARGUMENTS = "dynamic_partial_arguments"
    NON_STANDARD_MESSAGES_PLACEHOLDER = "non_standard_messages_placeholder"


@dataclass(slots=True)
class ValidationIssue:
    type: IssueType
    template: str
    message: str
    severity: Severity
    cell_index: int | None = None
    line: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["type"] = self.type.value
        data["severity"] = self.severity.value
        return data


@dataclass(slots=True)
class ExtractedTemplate:
    """A ChatPromptTemplate assignment plus the AST needed to analyze it."""

    name: str
    cell_index: int
    start_line: int
    source: str
    value_node: ast.expr


@dataclass(slots=True)
class ValidationReport:
    templates: list[ExtractedTemplate]
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    @property
    def info(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.INFO]

    def to_dict(self) -> dict:
        return {
            "total_templates": len(self.templates),
            "template_names": [template.name for template in self.templates],
            "total_issues": len(self.issues),
            "summary": {
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
                "info_count": len(self.info),
            },
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "info": [issue.to_dict() for issue in self.info],
        }


@dataclass(frozen=True, slots=True)
class PartialInfo:
    names: frozenset[str]
    has_dynamic_kwargs: bool = False


# ---------------------------------------------------------------------------
# AST / formatting helpers
# ---------------------------------------------------------------------------


def _strip_ipython_magics(source: str) -> str:
    """Return Python-parseable source while preserving original line numbers.

    A cell magic such as ``%%bash`` means the rest of the cell is not Python,
    so the entire cell is blanked. Ordinary ``%`` line magics and ``!`` shell
    escapes are blanked line-by-line.
    """

    lines = source.splitlines()
    first_nonempty = next(
        (line.lstrip() for line in lines if line.strip()),
        "",
    )
    if first_nonempty.startswith("%%"):
        return "\n".join("" for _ in lines)

    cleaned = [
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in lines
    ]
    return "\n".join(cleaned)


def _root_call_name(node: ast.expr) -> str | None:
    """Return the root object name for a chained call expression."""

    current: ast.AST = node
    while True:
        if isinstance(current, ast.Call):
            current = current.func
        elif isinstance(current, ast.Attribute):
            current = current.value
        else:
            break
    return current.id if isinstance(current, ast.Name) else None


def _find_from_messages_call(node: ast.expr) -> ast.Call | None:
    """Find ``.from_messages(...)`` inside a chained expression."""

    current: ast.AST = node
    while isinstance(current, ast.Call):
        func = current.func
        if isinstance(func, ast.Attribute) and func.attr == "from_messages":
            return current
        if isinstance(func, ast.Attribute):
            current = func.value
        else:
            return None
    return None


def _partial_info(value_node: ast.expr) -> PartialInfo:
    """Collect keyword names from every chained ``.partial(...)`` call.

    ``.partial(x=foo(a=1)).partial(y=2)`` correctly yields ``{"x", "y"}``.
    ``.partial(**mapping)`` is marked dynamic because its names cannot be
    established statically.
    """

    names: set[str] = set()
    has_dynamic_kwargs = False
    current: ast.AST = value_node

    while isinstance(current, ast.Call):
        func = current.func
        if isinstance(func, ast.Attribute) and func.attr == "partial":
            for keyword in current.keywords:
                if keyword.arg is None:
                    has_dynamic_kwargs = True
                else:
                    names.add(keyword.arg)

        if isinstance(func, ast.Attribute):
            current = func.value
        else:
            break

    return PartialInfo(frozenset(names), has_dynamic_kwargs)


def _messages_placeholder_var_name(call: ast.Call) -> str | None:
    """Return the statically-known MessagesPlaceholder variable name."""

    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value

    for keyword in call.keywords:
        if (
            keyword.arg == "variable_name"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            return keyword.value.value

    return None


class _StringLiteralCollector(ast.NodeVisitor):
    """Collect string values as they exist at runtime.

    For f-strings, ``ast`` has already applied Python's brace escaping to the
    literal segments. For example, ``f"{{user_prompt}}"`` contributes the
    runtime segment ``"{user_prompt}"``. That makes the collected values safe
    to inspect as inputs to ChatPromptTemplate without confusing Python's own
    f-string escaping with LangChain template escaping.
    """

    def __init__(self) -> None:
        self.strings: list[ast.Constant] = []

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                self.strings.append(value)
            elif isinstance(value, ast.FormattedValue):
                # Its runtime value is not statically knowable, but nested
                # expressions may contain their own string constants that are
                # not part of the prompt literal, so do not recurse into them.
                continue

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.strings.append(node)


def _collect_prompt_strings(from_messages_call: ast.Call) -> list[ast.Constant]:
    collector = _StringLiteralCollector()
    collector.visit(from_messages_call)
    return collector.strings


def _preview(text: str, limit: int = 120) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


def _field_root(field_name: str) -> str:
    """Return the top-level variable name from a format field."""

    root = field_name.split("!", 1)[0].split(":", 1)[0]
    root = root.split(".", 1)[0].split("[", 1)[0]
    return root.strip()


def _format_fields(text: str) -> tuple[set[str], str | None]:
    """Parse Python-format fields, respecting escaped ``{{`` / ``}}`` braces."""

    fields: set[str] = set()
    try:
        for _, field_name, _, _ in Formatter().parse(text):
            if field_name is None:
                continue
            root = _field_root(field_name)
            if root and not root.isdigit():
                fields.add(root)
    except ValueError as exc:
        return fields, str(exc)

    return fields, None


# Match only things that structurally look like an escaped field name.
# Arbitrary escaped JSON such as {{"name": "{user_prompt}"}} does not match.
_DOUBLE_FIELD_RE = re.compile(
    r"\{\{\s*"
    r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\[\]{}]+\])*"
    r"(?:![rsa])?(?::[^{}]+)?)"
    r"\s*\}\}"
)


def _double_brace_fields(text: str) -> list[tuple[str, str]]:
    """Return ``(raw_field, root_name)`` for ``{{field}}``-shaped escapes."""

    found: list[tuple[str, str]] = []
    for match in _DOUBLE_FIELD_RE.finditer(text):
        raw = match.group(1).strip()
        root = _field_root(raw)
        if root:
            found.append((raw, root))
    return found


def _assignment_name_and_value(
    node: ast.Assign | ast.AnnAssign,
) -> tuple[str, ast.expr] | None:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
        return node.targets[0].id, node.value

    if isinstance(node.target, ast.Name) and node.value is not None:
        return node.target.id, node.value

    return None


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class PromptTemplateValidator:
    """Validate ChatPromptTemplate definitions inside a Jupyter notebook."""

    # Variables commonly supplied by the runtime rather than through .partial().
    KNOWN_RUNTIME_VARS = frozenset(
        {
            "messages",
            "user_prompt",
            "available_df_ids",
            "tool_descriptions",
            "output_format",
            "dataset_description",
            "data_sample",
            "memories",
            "cleaned_dataset_description",
            "cleaning_metadata",
            "analysis_insights",
            "visualization_results",
            "analysis_config",
            "tooling_guidelines",
            "file_name",
            "file_type",
            "content",
            "visualization_task",
            "report_task",
            "agents",
            "completed_agents",
            "remaining_agents",
            "plan_summary",
            "plan_steps",
            "past_steps",
            "completed_tasks",
            "completed_steps",
            "latest_progress",
            "to_do_list",
            "leftover_to_do_list",
            "reply_msg_to_supervisor",
            "finished_this_task",
            "expect_reply",
            "last_agent_id",
            "last_message",
            "members",
            "viz_revise_count",
        }
    )

    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = Path(notebook_path)

    def load_notebook(self) -> dict:
        return json.loads(self.notebook_path.read_text(encoding="utf-8"))

    def extract_prompt_templates(
        self,
        notebook: dict,
    ) -> tuple[list[ExtractedTemplate], list[ValidationIssue]]:
        """Extract direct ChatPromptTemplate.from_messages assignments."""

        templates: list[ExtractedTemplate] = []
        parse_issues: list[ValidationIssue] = []

        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            clean_source = _strip_ipython_magics(source)

            if not clean_source.strip():
                continue

            try:
                tree = ast.parse(clean_source)
            except SyntaxError as exc:
                parse_issues.append(
                    ValidationIssue(
                        type=IssueType.PARSE_ERROR,
                        template=f"<cell {cell_index}>",
                        cell_index=cell_index,
                        line=exc.lineno,
                        message=(
                            f"Could not parse cell as Python ({exc.msg}); "
                            "prompt-template extraction skipped for this cell."
                        ),
                        severity=Severity.WARNING,
                    )
                )
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue

                assignment = _assignment_name_and_value(node)
                if assignment is None:
                    continue

                name, value_node = assignment
                if _find_from_messages_call(value_node) is None:
                    continue
                if _root_call_name(value_node) != "ChatPromptTemplate":
                    continue

                templates.append(
                    ExtractedTemplate(
                        name=name,
                        cell_index=cell_index,
                        start_line=node.lineno,
                        source=ast.get_source_segment(clean_source, node) or "",
                        value_node=value_node,
                    )
                )

        return templates, parse_issues

    def _prompt_strings(
        self,
        template: ExtractedTemplate,
    ) -> tuple[ast.Call, list[ast.Constant]]:
        from_messages_call = _find_from_messages_call(template.value_node)
        if from_messages_call is None:
            # Extraction guarantees this, but fail explicitly if the invariant
            # is ever broken by a future refactor.
            raise ValueError(
                f"{template.name} no longer contains ChatPromptTemplate.from_messages()"
            )
        return from_messages_call, _collect_prompt_strings(from_messages_call)

    def _check_brace_issues(
        self,
        template: ExtractedTemplate,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        _, strings = self._prompt_strings(template)
        partial = _partial_info(template.value_node)

        all_single_fields: set[str] = set()
        parsed: list[tuple[ast.Constant, set[str], str | None]] = []

        for const in strings:
            fields, parse_error = _format_fields(const.value)
            all_single_fields.update(fields)
            parsed.append((const, fields, parse_error))

        declared_or_expected = (
            set(partial.names) | set(self.KNOWN_RUNTIME_VARS) | all_single_fields
        )

        for const, _, parse_error in parsed:
            text = const.value

            if parse_error is not None:
                issues.append(
                    ValidationIssue(
                        type=IssueType.UNMATCHED_BRACES,
                        template=template.name,
                        cell_index=template.cell_index,
                        line=getattr(const, "lineno", None),
                        message=f"Malformed template braces: {parse_error}",
                        severity=Severity.ERROR,
                        detail=_preview(text),
                    )
                )
                # Formatter could not parse the string reliably, so do not
                # make secondary claims about its escaped fields.
                continue

            for raw_field, root in _double_brace_fields(text):
                if root in declared_or_expected:
                    issues.append(
                        ValidationIssue(
                            type=IssueType.DOUBLE_BRACE_PLACEHOLDER,
                            template=template.name,
                            cell_index=template.cell_index,
                            line=getattr(const, "lineno", None),
                            message=(
                                f"Found double-braced placeholder '{{{{{raw_field}}}}}'. "
                                "ChatPromptTemplate treats doubled braces as a literal "
                                f"'{{{raw_field}}}' instead of substituting the field. "
                                f"Use '{{{raw_field}}}' unless literal braces are intended."
                            ),
                            severity=Severity.ERROR,
                            detail=_preview(text),
                        )
                    )
                else:
                    issues.append(
                        ValidationIssue(
                            type=IssueType.AMBIGUOUS_DOUBLE_BRACES,
                            template=template.name,
                            cell_index=template.cell_index,
                            line=getattr(const, "lineno", None),
                            message=(
                                f"Found escaped field-shaped text '{{{{{raw_field}}}}}', "
                                "but the field is not otherwise declared or known at "
                                "runtime. This may be an intentional literal brace escape "
                                "or a misspelled placeholder."
                            ),
                            severity=Severity.WARNING,
                            detail=_preview(text),
                        )
                    )

        return issues

    def _check_undeclared_placeholders(
        self,
        template: ExtractedTemplate,
    ) -> list[ValidationIssue]:
        _, strings = self._prompt_strings(template)
        placeholders: set[str] = set()

        for const in strings:
            fields, parse_error = _format_fields(const.value)
            if parse_error is None:
                placeholders.update(fields)

        partial = _partial_info(template.value_node)
        undeclared = placeholders - set(partial.names) - set(self.KNOWN_RUNTIME_VARS)

        issues: list[ValidationIssue] = []

        if partial.has_dynamic_kwargs:
            issues.append(
                ValidationIssue(
                    type=IssueType.DYNAMIC_PARTIAL_ARGUMENTS,
                    template=template.name,
                    cell_index=template.cell_index,
                    message=(
                        "Template uses .partial(**mapping); static validation cannot "
                        "determine every partial variable name."
                    ),
                    severity=Severity.INFO,
                )
            )
            # Unknown **kwargs may provide otherwise-undeclared fields, so avoid
            # false-positive undeclared warnings.
            return issues

        if undeclared:
            issues.append(
                ValidationIssue(
                    type=IssueType.UNDECLARED_PLACEHOLDER,
                    template=template.name,
                    cell_index=template.cell_index,
                    message=(
                        "Placeholders with no .partial(...) default and not in the "
                        "known runtime-variable list: "
                        + ", ".join(sorted(undeclared))
                    ),
                    severity=Severity.WARNING,
                    detail=", ".join(sorted(undeclared)),
                )
            )

        return issues

    def _check_messages_placeholder(
        self,
        template: ExtractedTemplate,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        from_messages_call, _ = self._prompt_strings(template)

        for node in ast.walk(from_messages_call):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "MessagesPlaceholder"
            ):
                continue

            variable_name = _messages_placeholder_var_name(node)
            if variable_name is not None and variable_name != "messages":
                issues.append(
                    ValidationIssue(
                        type=IssueType.NON_STANDARD_MESSAGES_PLACEHOLDER,
                        template=template.name,
                        cell_index=template.cell_index,
                        line=node.lineno,
                        message=(
                            "Non-standard MessagesPlaceholder variable name "
                            f"'{variable_name}' (project convention is 'messages')."
                        ),
                        severity=Severity.INFO,
                    )
                )

        return issues

    def validate_template(
        self,
        template: ExtractedTemplate,
    ) -> list[ValidationIssue]:
        return [
            *self._check_brace_issues(template),
            *self._check_undeclared_placeholders(template),
            *self._check_messages_placeholder(template),
        ]

    def validate_all_templates(self) -> ValidationReport:
        notebook = self.load_notebook()
        templates, parse_issues = self.extract_prompt_templates(notebook)

        issues = list(parse_issues)
        for template in templates:
            issues.extend(self.validate_template(template))

        return ValidationReport(templates=templates, issues=issues)

    @staticmethod
    def print_report(report: ValidationReport) -> None:
        print("=" * 80)
        print("PROMPT TEMPLATE VALIDATION REPORT")
        print("=" * 80)
        print(f"Total templates found: {len(report.templates)}")
        if report.templates:
            print(
                "Template names: "
                + ", ".join(template.name for template in report.templates)
            )
        print()
        print(f"Issues found: {len(report.issues)}")
        print(f"  - Errors:   {len(report.errors)}")
        print(f"  - Warnings: {len(report.warnings)}")
        print(f"  - Info:     {len(report.info)}")
        print()

        for label, bucket in (
            ("ERRORS", report.errors),
            ("WARNINGS", report.warnings),
            ("INFO", report.info),
        ):
            if not bucket:
                continue

            print(f"{label}:")
            print("-" * 40)
            for issue in bucket:
                print(f"  Template: {issue.template}")
                if issue.cell_index is not None:
                    print(f"  Cell:     {issue.cell_index}")
                if issue.line is not None:
                    print(f"  Line:     {issue.line}")
                print(f"  Type:     {issue.type.value}")
                print(f"  Message:  {issue.message}")
                if issue.detail:
                    print(f"  Detail:   {issue.detail}")
                print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_notebook_path() -> Path:
    explicit = os.environ.get("IDD_NOTEBOOK")
    legacy = os.environ.get("NOTEBOOK_PATH")

    if explicit:
        return Path(explicit)
    if legacy:
        return Path(legacy)

    return Path(__file__).parent / "IntelligentDataDetective_beta_v5_patched.ipynb"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ChatPromptTemplate definitions in an Intelligent Data "
            "Detective Jupyter notebook."
        )
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        type=Path,
        help=(
            "Notebook path. Defaults to $IDD_NOTEBOOK, then $NOTEBOOK_PATH, "
            "then IntelligentDataDetective_beta_v5_patched.ipynb next to this script."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the validation report as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable report. Ignored when --json is used.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return exit code 1 when warnings are present, even if there are no errors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    notebook_path = args.notebook or _default_notebook_path()

    try:
        report = PromptTemplateValidator(notebook_path).validate_all_templates()
    except FileNotFoundError:
        message = f"Notebook not found: {notebook_path}"
        if args.as_json:
            print(json.dumps({"fatal_error": message}, indent=2))
        else:
            print(message, file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        message = f"Invalid notebook JSON in {notebook_path}: {exc}"
        if args.as_json:
            print(json.dumps({"fatal_error": message}, indent=2))
        else:
            print(message, file=sys.stderr)
        return 2
    except OSError as exc:
        message = f"Could not read notebook {notebook_path}: {exc}"
        if args.as_json:
            print(json.dumps({"fatal_error": message}, indent=2))
        else:
            print(message, file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    elif not args.quiet:
        PromptTemplateValidator.print_report(report)

    if report.errors:
        return 1
    if args.fail_on_warning and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
