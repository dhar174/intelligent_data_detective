from __future__ import annotations

from types import SimpleNamespace

from _patch_notebook import patch_reasoning_summary_concat


def _build_old_block(indent: str) -> str:
    return (
        f'{indent}summary_text = ""\n'
        f"{indent}if summary:\n"
        f"{indent}    if isinstance(summary, list):\n"
        "\n"
        f"{indent}        for s in summary:\n"
        f'{indent}            stext = s.get("text") if isinstance(s, dict) else getnestedattr(s, "text", getattr(s, "text", ""))\n'
        f"{indent}            summary_text += str(stext)\n"
        f"{indent}    if isinstance(summary, dict):\n"
        f'{indent}        summary_text += str(summary.get("text", ""))\n'
        f"{indent}    if isinstance(summary, str):\n"
        f"{indent}        summary_text += str(summary)\n"
    )


def _build_function_source(body: str) -> str:
    return (
        "def getnestedattr(obj, key, default=None):\n"
        "    return getattr(obj, key, default)\n\n"
        "def extract(summary):\n"
        f"{body}"
        "    return locals().get('summary_text', '')\n"
    )


def _execute_extract(source: str, summary):
    ns = {}
    exec(source, ns)
    return ns["extract"](summary)


def test_patch_reasoning_summary_concat_transforms_indentation_variants():
    src_12 = _build_old_block(" " * 12)
    src_16 = _build_old_block(" " * 16)
    transformed = patch_reasoning_summary_concat(src_12 + "\n" + src_16)
    assert "summary_text +=" not in transformed
    assert transformed.count("summary_parts = []") == 2
    assert transformed.count('summary_text = "".join(summary_parts)') == 2


def test_patch_reasoning_summary_concat_is_idempotent():
    source = _build_old_block(" " * 16)
    once = patch_reasoning_summary_concat(source)
    twice = patch_reasoning_summary_concat(once)
    assert once == twice


def test_patch_reasoning_summary_concat_preserves_summary_output_semantics():
    original_source = _build_function_source(_build_old_block("    "))
    patched_source = patch_reasoning_summary_concat(original_source)

    cases = [
        None,
        "plain-summary",
        {"text": "dict-summary"},
        [{"text": "a"}, {"text": "b"}],
        [SimpleNamespace(text="x"), {"text": "y"}],
        [1, {"text": "z"}],
    ]

    for summary in cases:
        assert _execute_extract(original_source, summary) == _execute_extract(
            patched_source, summary
        )
