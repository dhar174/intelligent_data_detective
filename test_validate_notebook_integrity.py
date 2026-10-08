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
12. Non-Python cell magics (%%bash, %%html) handled as whole-cell constructs -> PASS
13. Leading Python-body cell magics (%%time) pass, while invalid bodies FAIL
14. Mid-cell %% magic placement rejected as invalid syntax -> FAIL
15. Missing or unknown cell_type values rejected fail-closed -> FAIL
16. Malformed 'source' field (missing, non-string elements, wrong type) -> FAIL
17. Syntax errors alongside magics fail with actionable diagnostic
18. Multiline string literals ending in ? or containing ?/%/! preserved -> PASS
19. Nonexistent file fails with exit 1
20. Real committed IntelligentDataDetective_beta_v5_patched.ipynb passes
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
    assert "# [IPython magic/shell]: %matplotlib inline" in sanitized
    assert "# [IPython magic/shell]: !pip show langchain_experimental" in sanitized
    assert "# [IPython help]: ?plt.plot" in sanitized

    is_valid, diagnostics = validate_notebook(
        nb_file, expected_cells=DEFAULT_EXPECTED_CELLS
    )
    assert is_valid is True
    assert diagnostics == []


def test_non_python_cell_magic_whole_cell_handled(tmp_path: Path):
    """Test 12: Non-Python cell magics (%%bash, %%html) are handled as whole-cell constructs and pass."""
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
        line.startswith("# [cell-magic bash]:") for line in sanitized_bash.splitlines()
    )

    is_valid, diagnostics = validate_notebook(nb_file, expected_cells=99)
    assert is_valid is True
    assert diagnostics == []


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
