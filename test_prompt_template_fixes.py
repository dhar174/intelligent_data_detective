import json
from pathlib import Path

from prompt_template_validator import (
    IssueType,
    PromptTemplateValidator,
    Severity,
)
from _patch_notebook import _fix_runtime_prompt_braces


def _report(tmp_path: Path, source: str):
    notebook = {"cells": [{"cell_type": "code", "source": source.splitlines(True)}]}
    path = tmp_path / "prompt.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return PromptTemplateValidator(path).validate_all_templates()


def test_validator_detects_escaped_runtime_placeholder(tmp_path):
    report = _report(
        tmp_path,
        'prompt = ChatPromptTemplate.from_messages([("system", "{{user_prompt}}")])',
    )
    assert any(
        issue.type is IssueType.DOUBLE_BRACE_PLACEHOLDER
        and issue.severity is Severity.ERROR
        for issue in report.errors
    )


def test_validator_accepts_multiline_partial_template(tmp_path):
    report = _report(
        tmp_path,
        """prompt = (
    ChatPromptTemplate.from_messages([
        ("system", '''Objective: {user_prompt}
Schema: {output_schema_name}'''),
    ])
    .partial(output_schema_name="Plan")
)""",
    )
    assert not report.errors


def test_validator_keeps_literal_and_f_string_braces_out_of_runtime_check(
    tmp_path,
):
    report = _report(
        tmp_path,
        """literal = f"{{not_a_prompt_field}}"
prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the literal '{{not_a_prompt_field}}' and {user_prompt}")
])""",
    )
    assert not any(
        issue.type is IssueType.DOUBLE_BRACE_PLACEHOLDER for issue in report.errors
    )


def test_validator_passes_regenerated_production_notebook():
    notebook = Path(__file__).parent / "IntelligentDataDetective_beta_v5_patched.ipynb"
    report = PromptTemplateValidator(notebook).validate_all_templates()
    assert report.templates
    assert not report.errors


def test_patcher_only_rewrites_runtime_prompt_fields():
    cells = [
        {
            "cell_type": "code",
            "source": (
                'prompt = ChatPromptTemplate.from_messages([("system", '
                '"{{user_prompt}} {{literal}}")])\n'
                'value = f"{{literal}}"'
            ),
        }
    ]
    assert _fix_runtime_prompt_braces(cells) == 1
    source = "".join(cells[0]["source"])
    assert "{user_prompt}" in source
    assert "{{literal}}" in source
    assert 'f"{{literal}}"' in source
