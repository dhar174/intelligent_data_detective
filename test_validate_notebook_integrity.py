"""
test_validate_notebook_integrity.py — Regression test suite for validate_notebook_integrity.py.

Covers the full matrix required by Issue #152 and PR reviews:
1. Valid 99-cell notebook -> PASS (exit 0)
2. 0-cell notebook -> FAIL (exit 1)
3. 98-cell notebook -> FAIL (exit 1)
4. 100-cell notebook -> FAIL (exit 1)
5. Malformed JSON -> FAIL (exit 1)
6. Missing 'cells' key -> FAIL (exit 1)
7. 'cells' with wrong type (dict/int/str) -> FAIL (exit 1)
8. 99-cell notebook with invalid Python in a code cell -> FAIL (exit 1)
9. Source notebook (98 cells) vs patched notebook (99 cells) distinction
10. Useful diagnostic identifies the failing cell index, id, line, and syntax error
11. Supported notebook line magics (%matplotlib inline, !pip show, ?help) pass
12. Non-Python cell magics (%%bash, %%html) excluded from compilation and reported in diagnostics -> PASS
13. Leading Python-body cell magics (%%time) pass, while invalid bodies FAIL
14. Mid-cell %% magic placement rejected as invalid syntax -> FAIL
15. Missing or unknown cell_type values rejected fail-closed -> FAIL
16. Malformed 'source' field (missing, non-string elements, wrong type) -> FAIL
17. Syntax errors alongside magics fail with actionable diagnostic
18. Multiline string literals ending in ? or containing ?/%/! preserved -> PASS
19. Nonexistent file fails with exit 1
20. Real committed IntelligentDataDetective_beta_v5_patched.ipynb passes
21. (F1) Real code-object compilation rejects module-scope syntax errors
    (return, break, continue, duplicate args, nonlocal, yield) -> FAIL
22. (F1) Top-level await is permitted under ast.PyCF_ALLOW_TOP_LEVEL_AWAIT -> PASS
23. (F1) Static compilation only, zero runtime execution side-effects -> PASS
24. (F3) %%timeit header setup code compiled (passes when valid, fails when broken)
25. (F3) %time, %timeit, %prun line magics compile Python payloads (passes when valid, fails when broken)
26. (F3) Unknown or unsupported %%magic rejected with nonzero diagnostic -> FAIL
27. (F4) Standalone !shell and %magic within suites preserve indentation; broken python fails
28. (F4) Shell assignments (var = !cmd) and magic assignments (var = %magic) preserved; invalid assignments fail
29. (F4) Multiline modulo expressions in parens preserved intact; invalid expressions fail
30. (F5) %timeit and %%timeit valid options (short, attached, flags) compile correctly -> PASS
31. (F5) %timeit and %%timeit invalid options (missing args, non-int, unknown flags) fail with diagnostic -> FAIL
32. (F6) Multiline strings with closing parens on boundary lines correctly restore paren depth for line magics -> PASS
33. (F6) Multiline strings with unclosed parens or syntax errors fail compilation -> FAIL
"""

from __future__ import annotations

import json
from pathlib import Path

from validate_notebook_integrity import (
    DEFAULT_EXPECTED_CELLS,
    main,
    sanitize_cell_source,
    validate_notebook,
)


def _create_synthetic_notebook(
    cell_count: int,
    code_cells: list[tuple[int, str]] | None = None,
    cell_ids: dict[int, str] | None = None,
) -> dict:
    """Generate a synthetic notebook dictionary with specified cell count and code cells."""
    code_map = dict(code_cells or [])
    id_map = cell_ids or {}
    cells = []
    for idx in range(cell_count):
        cell_id = id_map.get(idx, f"cell_{idx}")
        if idx in code_map:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {"id": cell_id},
                    "id": cell_id,
                    "outputs": [],
                    "source": [code_map[idx]],
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {"id": cell_id},
                    "id": cell_id,
                    "source": [f"# Section {idx}\n", "Some markdown text."],
                }
            )
    return {
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
    }


def test_valid_99_cell_notebook(tmp_path: Path):
    """Test 1: Valid 99-cell synthetic notebook with valid Python code passes."""
    nb_data = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[
            (0, "import os\nimport sys\n"),
            (10, "def compute_sum(a, b):\n    return a + b\n"),
            (
                50,
                "class DataContainer:\n    def __init__(self, val):\n        self.val = val\n",
            ),
            (98, "print('Final step reached')\n"),
        ],
    )
    nb_file = tmp_path / "valid_99.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True
    assert diagnostics == []

    exit_code = main([str(nb_file), "-q"])
    assert exit_code == 0


def test_zero_cell_notebook(tmp_path: Path):
    """Test 2: 0-cell notebook fails fail-closed."""
    nb_data = _create_synthetic_notebook(cell_count=0)
    nb_file = tmp_path / "zero_cell.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is False
    assert any("expected exactly 99 cells, found 0" in msg for msg in diagnostics)

    exit_code = main([str(nb_file), "-q"])
    assert exit_code == 1


def test_98_cell_notebook(tmp_path: Path):
    """Test 3: 98-cell notebook (e.g. unpatched baseline count) fails for 99-cell expectation."""
    nb_data = _create_synthetic_notebook(cell_count=98)
    nb_file = tmp_path / "nb_98.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is False
    assert any("expected exactly 99 cells, found 98" in msg for msg in diagnostics)

    exit_code = main([str(nb_file), "-q"])
    assert exit_code == 1


def test_100_cell_notebook(tmp_path: Path):
    """Test 4: 100-cell notebook fails for 99-cell expectation."""
    nb_data = _create_synthetic_notebook(cell_count=100)
    nb_file = tmp_path / "nb_100.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is False
    assert any("expected exactly 99 cells, found 100" in msg for msg in diagnostics)

    exit_code = main([str(nb_file), "-q"])
    assert exit_code == 1


def test_malformed_json_notebook(tmp_path: Path):
    """Test 5: Malformed JSON file fails with informative JSON parsing error."""
    nb_file = tmp_path / "corrupt.ipynb"
    nb_file.write_text("{\n  'unclosed': True,\n", encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is False
    assert any("Malformed notebook JSON" in msg for msg in diagnostics)

    exit_code = main([str(nb_file), "-q"])
    assert exit_code == 1


def test_missing_cells_key(tmp_path: Path):
    """Test 6: Valid JSON missing top-level 'cells' key fails."""
    nb_data = {"metadata": {"name": "test"}, "nbformat": 4}
    nb_file = tmp_path / "no_cells.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is False
    assert any("Missing required 'cells' key" in msg for msg in diagnostics)

    exit_code = main([str(nb_file), "-q"])
    assert exit_code == 1


def test_cells_wrong_type(tmp_path: Path):
    """Test 7: 'cells' field with non-list type (dict, string, int) fails."""
    for bad_value in [{"dict_instead": True}, "a string", 123]:
        nb_data = {"cells": bad_value, "nbformat": 4}
        nb_file = tmp_path / f"bad_cells_{type(bad_value).__name__}.ipynb"
        nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

        is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
        assert is_valid is False
        assert any("Invalid 'cells' field" in msg for msg in diagnostics)

        exit_code = main([str(nb_file), "-q"])
        assert exit_code == 1


def test_invalid_python_in_code_cell(tmp_path: Path):
    """Test 8: 99-cell notebook containing Python syntax error in a code cell fails."""
    nb_data = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[
            (42, "def syntax_broken(\n    if x == 1:\n        return True\n"),
        ],
        cell_ids={42: "bad_cell_42"},
    )
    nb_file = tmp_path / "bad_syntax.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is False
    assert any(
        "Cell 42 (id: bad_cell_42)" in msg and "SyntaxError" in msg
        for msg in diagnostics
    )

    exit_code = main([str(nb_file), "-q"])
    assert exit_code == 1


def test_source_vs_patched_notebook_validation(tmp_path: Path):
    """Test 9: Source notebook valid structure vs patched notebook invalid structure."""
    source_nb = _create_synthetic_notebook(cell_count=98)
    patched_corrupt = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[(12, "for i in range(10)\n    pass\n")],  # missing colon
    )

    src_path = tmp_path / "source.ipynb"
    patched_path = tmp_path / "patched.ipynb"
    src_path.write_text(json.dumps(source_nb), encoding="utf-8")
    patched_path.write_text(json.dumps(patched_corrupt), encoding="utf-8")

    # Source notebook passes its own 98-cell structural contract
    is_valid_src, diagnostics_src = validate_notebook(src_path, expected_cells=98)
    assert is_valid_src is True
    assert diagnostics_src == []

    # But validating the corrupt patched notebook against 99 expected cells FAILS
    is_valid, diagnostics = validate_notebook(patched_path, expected_cells=99)
    assert is_valid is False
    assert any("Cell 12" in msg and "SyntaxError" in msg for msg in diagnostics)


def test_actionable_diagnostics_content(tmp_path: Path):
    """Test 10: Diagnostics report exact cell index, cell id, line number, and offending source."""
    broken_code = "x = 10\ny = 20\ndef broken_fn(:\n    return x + y\n"
    nb_data = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[(17, broken_code)],
        cell_ids={17: "cell_target_17"},
    )
    nb_file = tmp_path / "diagnostic_test.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is False
    assert len(diagnostics) == 1
    diag = diagnostics[0]

    assert "Cell 17" in diag
    assert "id: cell_target_17" in diag
    assert "SyntaxError" in diag
    assert "line 3" in diag
    assert "broken_fn(:" in diag


def test_supported_notebook_magics_accepted(tmp_path: Path):
    """Test 11: Valid Python code containing IPython line magics, shell commands, and help syntax compiles."""
    code_with_magics = (
        "%matplotlib inline\n"
        "import matplotlib.pyplot as plt\n"
        "!pip show langchain_experimental\n"
        "total = sum(range(100))\n"
        "?plt.plot\n"
        "plt.title('Sample')\n"
    )
    nb_data = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[(5, code_with_magics)],
    )
    nb_file = tmp_path / "magics.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    assert DEFAULT_EXPECTED_CELLS == 99
    sanitized = sanitize_cell_source(code_with_magics)
    assert "# [IPython magic]: %matplotlib inline" in sanitized.code
    assert "# [IPython shell]: !pip show langchain_experimental" in sanitized.code
    assert "# [IPython help]: ?plt.plot" in sanitized.code

    is_valid, diagnostics = validate_notebook(
        nb_file, expected_cells=DEFAULT_EXPECTED_CELLS
    )
    assert is_valid is True
    assert diagnostics == []


def test_non_python_cell_magic_whole_cell_handled(tmp_path: Path):
    """Test 12: Non-Python cell magics (%%bash, %%html) are excluded from compilation and reported."""
    bash_cell = "%%bash\necho 'Hello from bash'\nexit 0\n"
    html_cell = "%%html\n<div class='custom'>\n  <p>Header</p>\n</div>\n"
    nb_data = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[(3, bash_cell), (7, html_cell)],
    )
    nb_file = tmp_path / "cell_magics.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    sanitized_bash = sanitize_cell_source(bash_cell)
    assert all(
        line.startswith("# [cell-magic bash]:")
        for line in sanitized_bash.code.splitlines()
    )

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True
    assert len(diagnostics) == 2
    assert any("%%bash" in d for d in diagnostics)
    assert any("%%html" in d for d in diagnostics)

    # In verbose mode, reports exclusion counts
    is_valid_v, diag_v = validate_notebook(nb_file, expected_cells=99, verbose=True)
    assert is_valid_v is True
    assert any("2 code cells excluded from Python compilation" in d for d in diag_v)


def test_python_body_cell_magic_handled(tmp_path: Path):
    """Test 13: Leading Python-body cell magics (%%time) pass, while invalid Python bodies fail."""
    valid_time_cell = "%%time\nx = sum(range(1000))\nprint(x)\n"
    invalid_time_cell = "%%time\ndef broken(:\n    pass\n"

    # Valid case passes
    nb_data_valid = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[(10, valid_time_cell)],
    )
    nb_file_valid = tmp_path / "time_valid.ipynb"
    nb_file_valid.write_text(json.dumps(nb_data_valid), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file_valid, expected_cells=99)
    assert is_valid is True
    assert diagnostics == []

    # Invalid Python body under %%time fails AST compilation
    nb_data_invalid = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[(10, invalid_time_cell)],
        cell_ids={10: "bad_time_cell"},
    )
    nb_file_invalid = tmp_path / "time_invalid.ipynb"
    nb_file_invalid.write_text(json.dumps(nb_data_invalid), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file_invalid, expected_cells=99)
    assert is_valid is False
    assert any(
        "Cell 10 (id: bad_time_cell)" in msg and "SyntaxError" in msg
        for msg in diagnostics
    )


def test_invalid_mid_cell_magic_rejected(tmp_path: Path):
    """Test 14: Mid-cell %% magic placement is rejected by AST compiler as invalid syntax."""
    mid_cell_magic = "x = 1\n%%bash\necho hello\n"
    nb_data = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[(4, mid_cell_magic)],
        cell_ids={4: "mid_magic_cell"},
    )
    nb_file = tmp_path / "mid_cell_magic.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is False
    assert any(
        "Cell 4 (id: mid_magic_cell)" in msg and "SyntaxError" in msg
        for msg in diagnostics
    )


def test_missing_or_unknown_cell_type_rejected(tmp_path: Path):
    """Test 15: Missing or unknown cell_type values are rejected fail-closed."""
    # Missing cell_type
    nb_data_missing = _create_synthetic_notebook(cell_count=99)
    del nb_data_missing["cells"][20]["cell_type"]
    nb_file_missing = tmp_path / "missing_type.ipynb"
    nb_file_missing.write_text(json.dumps(nb_data_missing), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file_missing, expected_cells=99)
    assert is_valid is False
    assert any("invalid or missing 'cell_type'" in msg for msg in diagnostics)

    # Unknown cell_type
    nb_data_unknown = _create_synthetic_notebook(cell_count=99)
    nb_data_unknown["cells"][20]["cell_type"] = "cod"
    nb_file_unknown = tmp_path / "unknown_type.ipynb"
    nb_file_unknown.write_text(json.dumps(nb_data_unknown), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file_unknown, expected_cells=99)
    assert is_valid is False
    assert any("invalid or missing 'cell_type'" in msg for msg in diagnostics)


def test_malformed_source_field_rejected(tmp_path: Path):
    """Test 16: Malformed 'source' field (missing, non-string list element, wrong type) is rejected."""
    # Missing source
    nb_data_no_source = _create_synthetic_notebook(cell_count=99)
    del nb_data_no_source["cells"][5]["source"]
    nb_file_no_src = tmp_path / "no_source.ipynb"
    nb_file_no_src.write_text(json.dumps(nb_data_no_source), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file_no_src, expected_cells=99)
    assert is_valid is False
    assert any("missing required 'source' field" in msg for msg in diagnostics)

    # Non-string element in source list
    nb_data_bad_elem = _create_synthetic_notebook(cell_count=99)
    nb_data_bad_elem["cells"][5]["source"] = ["print('hi')\n", 12345]
    nb_file_bad_elem = tmp_path / "bad_elem.ipynb"
    nb_file_bad_elem.write_text(json.dumps(nb_data_bad_elem), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file_bad_elem, expected_cells=99)
    assert is_valid is False
    assert any("contains non-string elements" in msg for msg in diagnostics)


def test_syntax_error_with_magics_rejected(tmp_path: Path):
    """Test 17: Real syntax errors are NOT swallowed even when preceded or followed by magics."""
    code_with_bad_python_and_magics = (
        "%matplotlib inline\n"
        "!pip show langchain\n"
        "def broken_function(\n"
        "    return 42\n"
    )
    nb_data = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[(8, code_with_bad_python_and_magics)],
        cell_ids={8: "magic_and_broken"},
    )
    nb_file = tmp_path / "bad_magic.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is False
    assert any(
        "Cell 8 (id: magic_and_broken)" in msg and "SyntaxError" in msg
        for msg in diagnostics
    )


def test_multiline_string_with_question_marks_accepted(tmp_path: Path):
    """Test 18: Python multiline string literals ending in ? or containing ?/%/! are preserved without corruption."""
    code_with_strings = (
        'message = """Why?\nBecause."""\n'
        'sql_query = """\n'
        "SELECT * FROM users\n"
        "WHERE active = ?\n"
        '"""\n'
        'prompt = """\n'
        "%not_a_magic\n"
        "!not_a_shell\n"
        "Why?\n"
        '"""\n'
        'regular_str = "Is this fine?"\n'
        "x = 100\n"
    )
    nb_data = _create_synthetic_notebook(
        cell_count=99,
        code_cells=[(15, code_with_strings)],
    )
    nb_file = tmp_path / "string_literals.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True
    assert diagnostics == []


def test_nonexistent_file():
    """Test 19: Nonexistent file fails with clean error and exit 1."""
    is_valid, diagnostics = validate_notebook("non_existent_file_12345.ipynb")
    assert is_valid is False
    assert any("Notebook file not found" in msg for msg in diagnostics)

    exit_code = main(["non_existent_file_12345.ipynb", "-q"])
    assert exit_code == 1


def test_current_committed_patched_notebook():
    """Test 20: The repository's current committed patched notebook passes 99-cell integrity."""
    committed_path = Path("IntelligentDataDetective_beta_v5_patched.ipynb")
    assert committed_path.exists(), "Committed patched notebook must exist"

    is_valid, diagnostics = validate_notebook(
        committed_path, expected_cells=99, verbose=True
    )
    assert is_valid is True, f"Committed notebook failed validation: {diagnostics}"

    exit_code = main([str(committed_path), "-q"])
    assert exit_code == 0


def test_f1_module_level_syntax_errors_rejected(tmp_path: Path):
    """
    Test 21 (F1): Real code-object compilation rejects Python syntax errors that pass AST-only checks:
    - return outside function
    - break outside loop
    - continue outside loop
    - duplicate parameter names in function definition
    - nonlocal outside enclosing function
    - yield outside function
    """
    cases = [
        ("return 42\n", "return"),
        ("break\n", "break"),
        ("continue\n", "continue"),
        ("def duplicate_args(a, a):\n    pass\n", "duplicate argument"),
        ("nonlocal x\n", "nonlocal"),
        ("yield 1\n", "yield"),
    ]
    for idx, (invalid_code, err_keyword) in enumerate(cases):
        nb_data = _create_synthetic_notebook(99, code_cells=[(0, invalid_code)])
        nb_file = tmp_path / f"invalid_f1_{idx}.ipynb"
        nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

        is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
        assert is_valid is False, f"Failed to reject: {invalid_code}"
        assert any(
            "syntax compilation failed" in d and err_keyword in d.lower()
            for d in diagnostics
        ), f"Diagnostic missing {err_keyword} in {diagnostics}"


def test_f1_top_level_await_allowed(tmp_path: Path):
    """Test 22 (F1): Top-level await is permitted under ast.PyCF_ALLOW_TOP_LEVEL_AWAIT."""
    code = (
        "import asyncio\n"
        "async def get_val():\n"
        "    return 42\n"
        "result = await get_val()\n"
    )
    nb_data = _create_synthetic_notebook(99, code_cells=[(0, code)])
    nb_file = tmp_path / "top_level_await.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True
    assert diagnostics == []


def test_f1_no_runtime_execution(tmp_path: Path):
    """Test 23 (F1): Proves compilation pass performs static compilation only and never executes code."""
    marker_file = tmp_path / "side_effect_marker.txt"
    code = (
        f"import pathlib\n"
        f"pathlib.Path({str(marker_file)!r}).write_text('executed')\n"
        f"raise RuntimeError('Runtime execution must not occur!')\n"
    )
    nb_data = _create_synthetic_notebook(99, code_cells=[(0, code)])
    nb_file = tmp_path / "no_exec.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True
    assert not marker_file.exists(), "Side effect file was created during validation!"


def test_f3_timeit_setup_code_valid_and_broken(tmp_path: Path):
    """Test 24 (F3): %%timeit compiles setup code in header; passes when valid, fails when broken."""
    # Positive case: valid setup code and flags
    valid_timeit = "%%timeit -n 100 -r 5 x = 1\ny = x + 1\n"
    nb_data = _create_synthetic_notebook(99, code_cells=[(0, valid_timeit)])
    nb_file = tmp_path / "valid_timeit.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")
    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True

    # Negative case: syntactically broken setup code
    broken_timeit = "%%timeit x = (\npass\n"
    nb_data_bad = _create_synthetic_notebook(99, code_cells=[(0, broken_timeit)])
    nb_file_bad = tmp_path / "broken_timeit.ipynb"
    nb_file_bad.write_text(json.dumps(nb_data_bad), encoding="utf-8")
    is_valid_bad, diagnostics_bad = validate_notebook(nb_file_bad, expected_cells=99)
    assert is_valid_bad is False
    assert any("syntax compilation failed" in d for d in diagnostics_bad)


def test_f3_python_bearing_line_magics(tmp_path: Path):
    """Test 25 (F3): %time, %timeit, %prun compile Python payloads; pass when valid, fail when broken."""
    # Positive
    valid_magics = (
        "%time sum(range(100))\n"
        "%timeit -n 50 -r 3 sum(range(10))\n"
        "res = %time sum(range(10))\n"
    )
    nb_data = _create_synthetic_notebook(99, code_cells=[(0, valid_magics)])
    nb_file = tmp_path / "valid_line_magics.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")
    is_valid, _ = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True

    # Negative: broken expression in %time
    broken_line_magic = "%time def broken(:\n    pass\n"
    nb_data_bad = _create_synthetic_notebook(99, code_cells=[(0, broken_line_magic)])
    nb_file_bad = tmp_path / "broken_line_magic.ipynb"
    nb_file_bad.write_text(json.dumps(nb_data_bad), encoding="utf-8")
    is_valid_bad, diagnostics_bad = validate_notebook(nb_file_bad, expected_cells=99)
    assert is_valid_bad is False
    assert any("syntax compilation failed" in d for d in diagnostics_bad)


def test_f3_unsupported_cell_magic_rejected(tmp_path: Path):
    """Test 26 (F3): Unknown / unsupported cell magic rejected with nonzero diagnostic."""
    code = "%%unsupported_custom_magic\nx = 1\n"
    nb_data = _create_synthetic_notebook(99, code_cells=[(0, code)])
    nb_file = tmp_path / "unsupported_magic.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")
    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is False
    assert any(
        "unsupported cell magic" in d and "unsupported_custom_magic" in d
        for d in diagnostics
    )
    exit_code = main([str(nb_file), "-q"])
    assert exit_code == 1


def test_f4_control_flow_indentation_preserved(tmp_path: Path):
    """Test 27 (F4): Standalone !shell and %magic within suites preserve indentation; broken python fails."""
    # Positive
    code = "if True:\n" "    !echo ok\n" "    %pwd\n" "    x = 1\n"
    nb_data = _create_synthetic_notebook(99, code_cells=[(0, code)])
    nb_file = tmp_path / "control_flow_indent.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")
    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True

    # Negative companion: bad indentation following magic
    bad_indent = "if True:\n" "    !echo ok\n" "   x = 1\n"
    nb_data_bad = _create_synthetic_notebook(99, code_cells=[(0, bad_indent)])
    nb_file_bad = tmp_path / "bad_indent.ipynb"
    nb_file_bad.write_text(json.dumps(nb_data_bad), encoding="utf-8")
    is_valid_bad, diagnostics_bad = validate_notebook(nb_file_bad, expected_cells=99)
    assert is_valid_bad is False
    assert any("indent" in d.lower() for d in diagnostics_bad)

    # Negative companion: missing colon
    missing_colon = "if True\n" "    !echo ok\n"
    nb_data_colon = _create_synthetic_notebook(99, code_cells=[(0, missing_colon)])
    nb_file_colon = tmp_path / "missing_colon.ipynb"
    nb_file_colon.write_text(json.dumps(nb_data_colon), encoding="utf-8")
    is_valid_colon, diagnostics_colon = validate_notebook(
        nb_file_colon, expected_cells=99
    )
    assert is_valid_colon is False
    assert any("syntax compilation failed" in d for d in diagnostics_colon)


def test_f4_assignments_transformed(tmp_path: Path):
    """Test 28 (F4): Shell assignments (var = !cmd) and magic assignments (var = %magic) preserved."""
    code = (
        "files = !echo file1 file2\n"
        "current_dir = %pwd\n"
        "assert isinstance(files, list)\n"
    )
    nb_data = _create_synthetic_notebook(99, code_cells=[(0, code)])
    nb_file = tmp_path / "assignments.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")
    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True

    # Negative companion: invalid assignment target
    bad_target = "123 = !echo ok\n"
    nb_data_bad = _create_synthetic_notebook(99, code_cells=[(0, bad_target)])
    nb_file_bad = tmp_path / "bad_target.ipynb"
    nb_file_bad.write_text(json.dumps(nb_data_bad), encoding="utf-8")
    is_valid_bad, diagnostics_bad = validate_notebook(nb_file_bad, expected_cells=99)
    assert is_valid_bad is False
    assert any("syntax compilation failed" in d for d in diagnostics_bad)


def test_f4_multiline_modulo_not_mangled(tmp_path: Path):
    """Test 29 (F4): Multiline modulo arithmetic in parens preserved intact without magic rewriting."""
    code = "x = 10\n" "y = 3\n" "val = (\n" "    x\n" "    % y\n" ")\n"
    nb_data = _create_synthetic_notebook(99, code_cells=[(0, code)])
    nb_file = tmp_path / "modulo_parens.ipynb"
    nb_file.write_text(json.dumps(nb_data), encoding="utf-8")
    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True

    # Negative companion: malformed expression inside parens
    bad_modulo = "val = (\n" "    x\n" "    % (\n" ")\n"
    nb_data_bad = _create_synthetic_notebook(99, code_cells=[(0, bad_modulo)])
    nb_file_bad = tmp_path / "bad_modulo.ipynb"
    nb_file_bad.write_text(json.dumps(nb_data_bad), encoding="utf-8")
    is_valid_bad, diagnostics_bad = validate_notebook(nb_file_bad, expected_cells=99)
    assert is_valid_bad is False
    assert any("syntax compilation failed" in d for d in diagnostics_bad)


def test_f5_timeit_options_valid(tmp_path: Path):
    """Test 30 (F5): %timeit and %%timeit valid options (short, attached, flags) compile correctly."""
    valid_snippets = [
        "%%timeit -n 100 -r 5\nx = 1\n",
        "%%timeit -n100 -r5\nx = 1\n",
        "%%timeit -n 50 -r 3 -p 4 -t\nx = 1\n",
        "%%timeit -q -o\nx = 1\n",
        "%%timeit --quiet\nx = 1\n",
        "%%timeit -n 10 -r 2 a = 10\nb = a + 1\n",
        "%timeit -n 100 -r 5 sum([1, 2, 3])\n",
        "%timeit -n10 -r3 len('hello')\n",
        "res = %timeit -o -n 10 [i for i in range(10)]\n",
        "%timeit -t -c -q -o 42\n",
        "%%timeit -p 0\nx = 1\n",
    ]
    for i, snippet in enumerate(valid_snippets):
        nb_data = _create_synthetic_notebook(99, code_cells=[(0, snippet)])
        nb_file = tmp_path / f"valid_timeit_{i}.ipynb"
        nb_file.write_text(json.dumps(nb_data), encoding="utf-8")
        is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
        assert is_valid is True, f"Snippet {i} failed validation: {diagnostics}"


def test_f5_timeit_options_invalid(tmp_path: Path):
    """Test 31 (F5): %timeit and %%timeit invalid options (missing args, non-int, unknown flags) fail with diagnostic."""
    invalid_cases = [
        ("%%timeit -n banana\nx = 1\n", "banana"),
        ("%%timeit -n\nx = 1\n", "option -n requires an argument"),
        ("%%timeit -r nope\nx = 1\n", "nope"),
        ("%%timeit -r\nx = 1\n", "option -r requires an argument"),
        ("%%timeit --not-a-real-option\nx = 1\n", "unrecognized option"),
        ("%%timeit -nbanana\nx = 1\n", "banana"),
        ("%%timeit -rnope\nx = 1\n", "nope"),
        ("%%timeit -p -1\nx = 1\n", "non-negative integer"),
        ("%%timeit -p nope\nx = 1\n", "nope"),
        ("%%timeit -p\nx = 1\n", "option -p requires an argument"),
        ("%%timeit -n 0\nx = 1\n", "positive integer"),
        ("%%timeit -r 0\nx = 1\n", "positive integer"),
        ("%%timeit -n0\nx = 1\n", "positive integer"),
        ("%%timeit -r0\nx = 1\n", "positive integer"),
        ("%%timeit -x\nx = 1\n", "unrecognized option"),
        ("%timeit -r nope x + 1\n", "nope"),
        ("%timeit -n\n", "option -n requires an argument"),
        ("%timeit --bogus x = 1\n", "unrecognized option"),
        ("res = %timeit --unknown_flag x\n", "unrecognized option"),
    ]
    for i, (snippet, expected_diag) in enumerate(invalid_cases):
        nb_data = _create_synthetic_notebook(99, code_cells=[(0, snippet)])
        nb_file = tmp_path / f"invalid_timeit_{i}.ipynb"
        nb_file.write_text(json.dumps(nb_data), encoding="utf-8")
        is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
        assert (
            is_valid is False
        ), f"Snippet {i} should have failed validation: {snippet}"
        assert any(
            expected_diag in d for d in diagnostics
        ), f"Snippet {i} diagnostics missing expected text '{expected_diag}': {diagnostics}"
        exit_code = main([str(nb_file), "-q"])
        assert exit_code == 1, f"Snippet {i} exit code was not 1"


def test_f6_multiline_string_paren_tracking_valid(tmp_path: Path):
    """Test 32 (F6): Multiline strings with closing parens on boundary lines correctly restore paren depth for line magics."""
    valid_snippets = [
        # Quoted from review: valid python followed by line magic
        'x = ("""alpha\nbeta""")\n%pwd\n',
        # Nested delimiters with multiline string
        'data = [("""first\nsecond"""), 42]\n%matplotlib inline\n',
        # Misleading delimiters inside multiline string
        'text = """((( [[[\n))) ]]]\n%not_a_real_magic"""\n%pwd\n',
        # Multiline string followed by ordinary Python
        'message = ("""first\nsecond""")\nresult = len(message)\n',
        # Single-quote triple delimiters
        "val = ('''start\nmiddle\nend''')\n%pwd\n",
        # Multiple triple-quoted strings with parens
        's = ("""one\ntwo""") + ("""three\nfour""")\n%pwd\n',
    ]
    for i, snippet in enumerate(valid_snippets):
        nb_data = _create_synthetic_notebook(99, code_cells=[(0, snippet)])
        nb_file = tmp_path / f"valid_f6_{i}.ipynb"
        nb_file.write_text(json.dumps(nb_data), encoding="utf-8")
        is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
        assert is_valid is True, f"Snippet {i} failed validation: {diagnostics}"


def test_f6_multiline_string_paren_tracking_invalid(tmp_path: Path):
    """Test 33 (F6): Multiline strings with unclosed parens or syntax errors fail compilation."""
    invalid_cases = [
        # Unclosed paren across multiline string
        ('x = ("""alpha\nbeta"""\n%pwd\n', "syntax compilation failed"),
        # Valid multiline string followed by malformed Python
        (
            'x = ("""alpha\nbeta""")\ndef broken(:\n    pass\n',
            "syntax compilation failed",
        ),
        # Unclosed bracket enclosing multiline string
        ('data = [("""first\nsecond""), 42\n', "syntax compilation failed"),
    ]
    for i, (snippet, expected_diag) in enumerate(invalid_cases):
        nb_data = _create_synthetic_notebook(99, code_cells=[(0, snippet)])
        nb_file = tmp_path / f"invalid_f6_{i}.ipynb"
        nb_file.write_text(json.dumps(nb_data), encoding="utf-8")
        is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
        assert (
            is_valid is False
        ), f"Snippet {i} should have failed validation: {snippet}"
        assert any(
            expected_diag in d for d in diagnostics
        ), f"Snippet {i} diagnostics missing expected text '{expected_diag}': {diagnostics}"
        exit_code = main([str(nb_file), "-q"])
        assert exit_code == 1, f"Snippet {i} exit code was not 1"
