"""
test_validate_notebook_integrity.py — Regression test suite for validate_notebook_integrity.py.

Covers the full matrix required by Issue #152:
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
11. Supported notebook magics (%matplotlib inline, !pip show) pass
12. Syntax errors alongside magics fail with actionable diagnostic
13. Nonexistent file fails with exit 1
14. Real committed IntelligentDataDetective_beta_v5_patched.ipynb passes
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

    # When targeting the patched notebook with standard 99 expected cells, it must FAIL
    is_valid, diagnostics = validate_notebook(patched_path, expected_cells=99)
    assert is_valid is False
    assert any("Cell 12" in msg and "SyntaxError" in msg for msg in diagnostics)

    # Validating source for 99 also fails
    is_valid_src, _ = validate_notebook(src_path, expected_cells=99)
    assert is_valid_src is False


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
    """Test 11: Valid Python code containing IPython line magics, shell commands, and cell magics compiles."""
    code_with_magics = (
        "%matplotlib inline\n"
        "import matplotlib.pyplot as plt\n"
        "!pip show langchain_experimental\n"
        "%%time\n"
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


def test_syntax_error_with_magics_rejected(tmp_path: Path):
    """Test 12: Real syntax errors are NOT swallowed even when preceded or followed by magics."""
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


def test_nonexistent_file():
    """Test 13: Nonexistent file fails with clean error and exit 1."""
    is_valid, diagnostics = validate_notebook("non_existent_file_12345.ipynb")
    assert is_valid is False
    assert any("Notebook file not found" in msg for msg in diagnostics)

    exit_code = main(["non_existent_file_12345.ipynb", "-q"])
    assert exit_code == 1


def test_current_committed_patched_notebook():
    """Test 14: The repository's current committed patched notebook passes 99-cell integrity."""
    committed_path = Path("IntelligentDataDetective_beta_v5_patched.ipynb")
    assert committed_path.exists(), "Committed patched notebook must exist"

    is_valid, diagnostics = validate_notebook(
        committed_path, expected_cells=99, verbose=True
    )
    assert is_valid is True, f"Committed notebook failed validation: {diagnostics}"

    exit_code = main([str(committed_path), "-q"])
    assert exit_code == 0
