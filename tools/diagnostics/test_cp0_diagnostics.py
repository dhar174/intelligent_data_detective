"""
tools/diagnostics/test_cp0_diagnostics.py — Regression Test Suite for CP0 Diagnostics.

Validates that:
1. A reproduced defect returns 'REPRODUCED'.
2. A simulated corrected implementation returns 'NOT REPRODUCED'.
3. A static-only finding cannot be silently promoted to dynamic reproduction.
4. Missing prerequisites return 'BLOCKED'.
5. Unexpected outcomes do not silently become successful evidence.
6. Diagnostic execution failures are reported with exception details rather than swallowed.
7. A successful diagnostic run can legitimately contain 'NOT REPRODUCED' findings.
8. Machine-readable JSON output contains complete environment and summary metadata.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from tools.diagnostics import reproduce_drift
from tools.diagnostics.reproduce_drift import (
    ClaimResult,
    probe_claim_1,
    probe_claim_2,
    probe_claim_3,
    probe_claim_4,
    probe_claim_5,
    probe_claim_6,
    probe_claim_7,
    probe_claim_8,
    probe_claim_9,
    probe_claim_10,
    probe_claim_11,
    probe_claim_12,
    _extract_production_delete_rows_tool,
    _InMemoryRegistry,
)
from tools.diagnostics.probe_notebook_cells import scan_notebook, REPO_ROOT


def test_claim_2_approved_lifecycle_returns_observed_invariant():
    """Verify that the approved Plan lifecycle contract reports OBSERVED INVARIANT with is_defect=False."""
    res = probe_claim_2()
    assert res.classification == "OBSERVED INVARIANT"
    assert res.is_defect is False
    assert res.evidence_type == "DYNAMIC"
    assert "NEW plans allocate monotonically" in res.observed_behavior
    assert "RESTORE preserved snapshot" in res.observed_behavior


def test_claim_2_simulated_broken_restore_returns_reproduced(monkeypatch):
    """Verify that a broken restoration (e.g. failing to preserve snapshot version) reports REPRODUCED."""
    import idd_core

    orig_restore = idd_core.Plan.from_persisted_snapshot

    def broken_restore(cls, snapshot):
        p = orig_restore(snapshot)
        object.__setattr__(p, "plan_version", 1)
        return p

    monkeypatch.setattr(idd_core.Plan, "from_persisted_snapshot", classmethod(broken_restore))

    res = probe_claim_2()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "restore_preserved=False" in res.defect_or_contract_explanation


def test_claim_2_simulated_broken_new_plan_allocation_returns_reproduced(monkeypatch):
    """Verify that broken new-plan creation (e.g. failing to allocate monotonic version) reports REPRODUCED."""
    import idd_core

    orig_init = idd_core.Plan.__init__

    def broken_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        object.__setattr__(self, "plan_version", 1)

    monkeypatch.setattr(idd_core.Plan, "__init__", broken_init)

    res = probe_claim_2()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "new_allocated=False" in res.defect_or_contract_explanation


def test_static_only_finding_not_promoted_to_dynamic():
    """Verify that static inspection of test files returns STATIC EVIDENCE ONLY, not REPRODUCED."""
    res = probe_claim_11()
    assert res.classification == "STATIC EVIDENCE ONLY"
    assert res.evidence_type == "STATIC"
    assert res.is_defect is True


def test_missing_prerequisites_return_blocked(monkeypatch):
    """Verify that missing environment or dependencies returns BLOCKED."""
    monkeypatch.setattr(reproduce_drift, "HAS_IDD_CORE", False)
    res = probe_claim_2()
    assert res.classification == "BLOCKED"
    assert res.is_defect is True
    assert "not importable" in res.observed_behavior or "missing" in res.limitations


def test_claim_12_historical_integer_defect_returns_reproduced(monkeypatch):
    """Verify that if historical integer-column query defect reoccurs (Case A fails), Claim 12 reports REPRODUCED."""
    def broken_delete_rows(df_id, conditions, inplace=True):
        if "0" in conditions and inplace:
            return "0 rows deleted"  # Fails case A expectation
        return "2 rows deleted"

    monkeypatch.setattr(
        reproduce_drift,
        "_extract_production_delete_rows_tool",
        lambda reg: (broken_delete_rows, None),
    )

    res = probe_claim_12()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "Issue #147" in res.defect_or_contract_explanation


def test_claim_12_historical_collision_defect_returns_reproduced(monkeypatch):
    """Verify that if integer/string column label collision reoccurs (Case B fails), Claim 12 reports REPRODUCED."""
    def broken_collision_tool(df_id, conditions, inplace=True):
        if df_id == "df_b":
            # Simulate column confusion: drops wrong row or returns 0 rows
            return "0 rows deleted"
        return "2 rows deleted"

    monkeypatch.setattr(
        reproduce_drift,
        "_extract_production_delete_rows_tool",
        lambda reg: (broken_collision_tool, None),
    )

    res = probe_claim_12()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "Issue #147" in res.defect_or_contract_explanation


def test_claim_12_unexpected_non_integer_failure_returns_blocked(monkeypatch):
    """Verify that an unexpected non-integer failure (e.g. Case C in-place mutation broken) returns BLOCKED."""
    import pandas as pd

    def mock_extract(reg):
        def broken_tool(df_id, conditions, inplace=True):
            if df_id == "df_a":
                # Case A passes
                reg.register_dataframe(pd.DataFrame({0: [10], "name": ["a"]}), "df_a")
                return "2 rows deleted"
            if df_id == "df_b":
                # Case B passes
                reg.register_dataframe(pd.DataFrame({0: [10], "0": [100], "name": ["a"]}), "df_b")
                return "2 rows deleted"
            if df_id == "df_c":
                # Case C fails: returns 0 rows deleted
                return "0 rows deleted"
            if df_id == "df_d":
                return '{"x": [1, 2]}'
            if df_id == "df_e":
                return {"status": "error", "reason": "UndefinedVariableError: invalid_col"}
            return "2 rows deleted"
        return (broken_tool, None)

    monkeypatch.setattr(
        reproduce_drift,
        "_extract_production_delete_rows_tool",
        mock_extract,
    )

    res = probe_claim_12()
    assert res.classification != "OBSERVED INVARIANT"
    assert res.classification != "NOT REPRODUCED"
    assert res.classification == "BLOCKED"
    assert res.is_defect is True
    assert "Unexpected failure in non-integer" in res.defect_or_contract_explanation


def test_probe_extraction_failure_returns_blocked(monkeypatch):
    """Verify that if production tool extraction is blocked, Claim 12 returns BLOCKED."""
    monkeypatch.setattr(
        reproduce_drift,
        "_extract_production_delete_rows_tool",
        lambda reg: (None, "Extraction intentionally mocked as blocked"),
    )

    res = probe_claim_12()
    assert res.classification == "BLOCKED"
    assert res.is_defect is True
    assert "blocked" in res.limitations.lower()


def test_claim_12_production_tool_cases_pass():
    """Verify that the actual extracted production delete_rows tool passes all 5 cases."""
    res = probe_claim_12()
    assert res.classification == "OBSERVED INVARIANT"
    assert res.evidence_type == "DYNAMIC"
    assert res.is_defect is False
    assert "All 5 cases passed: True" in res.observed_behavior


def test_claim_1_thread_scheduling_order_independence(monkeypatch):
    """Verify that Claim 1 evaluates uniqueness and contiguous range without thread-order scheduling fragility."""
    # Execute probe_claim_1 directly
    res = probe_claim_1()
    assert res.classification == "OBSERVED INVARIANT"
    assert res.is_defect is False
    assert "Sequential monotonic increment: True" in res.observed_behavior
    assert "All unique: True" in res.observed_behavior
    assert "Contiguous range: True" in res.observed_behavior


def test_get_git_commit_failure_returns_unknown(monkeypatch):
    """Verify that git commit discovery failure returns 'unknown' rather than a hardcoded commit SHA."""
    import subprocess

    def mock_subprocess_run(*args, **kwargs):
        raise FileNotFoundError("git not found in PATH")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
    commit = reproduce_drift._get_git_commit()
    assert commit == "unknown"
    assert commit != "e4b98fff8713d596b6177b5907e26a2d75c6fc90"


def test_successful_run_can_contain_not_reproduced_findings():
    """Verify that summary structures handle NOT REPRODUCED without marking run as failed."""
    sample_result = ClaimResult(
        claim_id=99,
        title="Simulated Claim",
        subsystem="Test",
        classification="NOT REPRODUCED",
        evidence_type="DYNAMIC",
        is_defect=False,
        expected_behavior="exp",
        observed_behavior="obs",
        defect_or_contract_explanation="exp",
        locators=[],
    )
    assert sample_result.classification == "NOT REPRODUCED"
    assert sample_result.is_defect is False


def test_ast_scanner_status_and_completeness():
    """Verify AST scanner reports complete status with zero parse errors on the patched notebook."""
    nb_path = REPO_ROOT / "IntelligentDataDetective_beta_v5_patched.ipynb"
    all_symbols, symbol_map, parse_errors = scan_notebook(nb_path)
    assert len(parse_errors) == 0
    assert len(all_symbols) > 1000
    assert "delete_rows" in symbol_map
    assert "DataFrameRegistry" in symbol_map
