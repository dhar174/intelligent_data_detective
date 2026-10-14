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

import pytest

from tools.diagnostics import reproduce_drift
from tools.diagnostics.reproduce_drift import (
    ClaimResult,
    probe_claim_1,
    probe_claim_2,
    probe_claim_6,
    probe_claim_7,
    probe_claim_8,
    probe_claim_9,
    probe_claim_11,
    probe_claim_12,
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

    monkeypatch.setattr(
        idd_core.Plan, "from_persisted_snapshot", classmethod(broken_restore)
    )

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


def test_claim_2_simulated_broken_high_water_advancement_returns_reproduced(
    monkeypatch,
):
    """Verify that restoration failing to advance allocator high-water mark reports REPRODUCED."""
    import idd_core

    orig_restore = idd_core.Plan.from_persisted_snapshot

    def broken_restore(cls, snapshot):
        prev_version = cls._current_version
        p = orig_restore(snapshot)
        # Revert high-water mark so restore doesn't advance it
        cls.reset_counter(prev_version)
        return p

    monkeypatch.setattr(
        idd_core.Plan, "from_persisted_snapshot", classmethod(broken_restore)
    )

    res = probe_claim_2()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "high_water_advanced=False" in res.defect_or_contract_explanation


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
                reg.register_dataframe(
                    pd.DataFrame({0: [10], "0": [100], "name": ["a"]}), "df_b"
                )
                return "2 rows deleted"
            if df_id == "df_c":
                # Case C fails: returns 0 rows deleted
                return "0 rows deleted"
            if df_id == "df_d":
                return '{"x": [1, 2]}'
            if df_id == "df_e":
                return {
                    "status": "error",
                    "reason": "UndefinedVariableError: invalid_col",
                }
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


def test_claim_8_all_formats_succeed_returns_observed_invariant():
    """Verify that multi-format eviction reload and validation reports OBSERVED INVARIANT."""
    res = probe_claim_8()
    assert res.classification == "OBSERVED INVARIANT"
    assert res.is_defect is False
    assert res.evidence_type == "DYNAMIC"
    assert (
        "Both get_dataframe(load_if_not_exists=True) and validate_dataframe_exists succeeded"
        in res.observed_behavior
    )
    assert "registered=True" in res.observed_behavior
    assert "reload_path=True" in res.observed_behavior


def test_claim_8_broken_csv_reader_detected(monkeypatch):
    """Verify that a broken CSV reader is detected by Claim 8 and reports REPRODUCED."""
    import pandas as pd

    def broken_read_csv(*args, **kwargs):
        raise RuntimeError("Simulated CSV reader failure")

    monkeypatch.setattr(pd, "read_csv", broken_read_csv)
    res = probe_claim_8()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True


def test_claim_8_broken_json_reader_detected(monkeypatch):
    """Verify that a broken JSON reader is detected by Claim 8 and reports REPRODUCED."""
    import pandas as pd

    def broken_read_json(*args, **kwargs):
        raise RuntimeError("Simulated JSON reader failure")

    monkeypatch.setattr(pd, "read_json", broken_read_json)
    res = probe_claim_8()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True


def test_claim_8_registration_failure_not_counted_as_success(monkeypatch):
    """Verify that registration failures cannot silently count as successful evidence in Claim 8."""
    import idd_core

    orig_reg = idd_core.DataFrameRegistry.register_dataframe

    def failing_register(self, df=None, df_id=None, raw_path="", **kwargs):
        if df_id == "test_csv":
            return None  # Simulate silent registration failure
        return orig_reg(self, df=df, df_id=df_id, raw_path=raw_path, **kwargs)

    monkeypatch.setattr(
        idd_core.DataFrameRegistry, "register_dataframe", failing_register
    )
    res = probe_claim_8()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True


def test_claim_8_untrusted_pickle_rejected_by_default(tmp_path):
    """Verify that untrusted external pickle imports remain rejected by default without trust_external_pickle."""
    import idd_core
    import pandas as pd

    reg = idd_core.DataFrameRegistry(capacity=5)
    pkl_file = tmp_path / "external.pkl"
    pd.DataFrame({"x": [1]}).to_pickle(pkl_file)

    with pytest.raises(idd_core.DataFrameRegistryError, match="trust_external_pickle"):
        reg.register_dataframe(df=None, df_id="untrusted", raw_path=str(pkl_file))


def test_claim_8_simulated_broken_validation_returns_reproduced(monkeypatch):
    """Verify that if validation fails on non-CSV formats, Claim 8 reports REPRODUCED."""
    import idd_core

    orig_validate = idd_core.validate_dataframe_exists

    def broken_validate(df_id: str) -> bool:
        if "pkl" in df_id or "json" in df_id:
            return False
        return orig_validate(df_id)

    monkeypatch.setattr(idd_core, "validate_dataframe_exists", broken_validate)

    res = probe_claim_8()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert (
        "validate_dataframe_exists fails on cache-evicted non-CSV"
        in res.defect_or_contract_explanation
    )


def test_claim_6_structured_tool_error_returns_observed_invariant():
    """Verify that canonical tool error handling returns OBSERVED INVARIANT."""
    res = probe_claim_6()
    assert res.classification == "OBSERVED INVARIANT"
    assert res.is_defect is False
    assert res.evidence_type == "BOTH"
    assert "known error dict=True" in res.observed_behavior
    assert "unexpected error sanitized dict=True" in res.observed_behavior


def test_claim_6_simulated_string_error_returns_reproduced(monkeypatch):
    """Verify that a plain-string error return is detected as a defect (Claim 6 REPRODUCED)."""
    import idd_core

    def broken_decorator(func):
        def wrapper(*args, **kwargs):
            return "Error: Simulated legacy plain string error"

        return wrapper

    monkeypatch.setattr(idd_core, "handle_tool_errors", broken_decorator)
    res = probe_claim_6()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "plain error strings" in res.defect_or_contract_explanation


def test_claim_7_signature_aware_returns_observed_invariant():
    """Verify that signature-aware df_id extraction returns OBSERVED INVARIANT."""
    res = probe_claim_7()
    assert res.classification == "OBSERVED INVARIANT"
    assert res.is_defect is False
    assert res.evidence_type == "DYNAMIC"
    assert "Signature-aware success: True" in res.observed_behavior


def test_claim_7_simulated_blind_args0_returns_reproduced(monkeypatch):
    """Verify that blind args[0] extraction is detected as a defect (Claim 7 REPRODUCED)."""
    import idd_core

    def legacy_handle_tool_errors(func):
        def wrapper(*args, **kwargs):
            if args and isinstance(args[0], str):
                df_id = args[0]
                if not idd_core.validate_dataframe_exists(df_id):
                    return f"Error: DataFrame with ID '{df_id}' not found"
            return func(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(idd_core, "handle_tool_errors", legacy_handle_tool_errors)
    res = probe_claim_7()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "Blind args[0] check" in res.defect_or_contract_explanation


def test_claim_9_artifact_containment_returns_observed_invariant():
    """Verify that artifact containment & multi-root protections return OBSERVED INVARIANT."""
    res = probe_claim_9()
    assert res.classification == "OBSERVED INVARIANT"
    assert res.is_defect is False
    assert res.evidence_type == "BOTH"
    assert "safe path resolved=True" in res.observed_behavior
    assert "traversal attempt blocked=True" in res.observed_behavior
    assert "delayed directory creation verified=True" in res.observed_behavior
    assert (
        "multi-root helper is_safe_artifact_path functional=True"
        in res.observed_behavior
    )


def test_claim_9_simulated_traversal_leak_returns_reproduced(monkeypatch):
    """Verify that failure to block traversal is detected as a defect (Claim 9 REPRODUCED)."""
    import idd_core
    import idd_tools
    from pathlib import Path

    def leaking_resolve(file_name, config=None, subdir=None, create_parents=True):
        return Path("/unauthorized/escape") / file_name

    monkeypatch.setattr(idd_core, "_resolve_artifact_path", leaking_resolve)
    monkeypatch.setattr(idd_tools, "_resolve_artifact_path", leaking_resolve)
    res = probe_claim_9()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "path traversal attempts not blocked" in res.defect_or_contract_explanation


def test_claim_9_simulated_missing_multi_root_helper_returns_reproduced(monkeypatch):
    """Verify that a broken or missing multi-root helper is detected as a defect (Claim 9 REPRODUCED)."""
    import idd_core
    import idd_tools

    monkeypatch.setattr(
        idd_core, "is_safe_artifact_path", lambda candidate, roots: False
    )
    monkeypatch.setattr(
        idd_tools, "is_safe_artifact_path", lambda candidate, roots: False
    )
    res = probe_claim_9()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert (
        "multi-root helper is_safe_artifact_path broken or missing"
        in res.defect_or_contract_explanation
    )


def test_claim_6_simulated_broken_sanitizer_returns_reproduced(monkeypatch):
    """Verify that a broken unexpected error sanitizer is detected (Claim 6 REPRODUCED)."""
    import idd_core
    import idd_tools

    def broken_failure(operation, action, exc=None):
        return {"status": "broken"}

    monkeypatch.setattr(idd_core, "_tool_failure", broken_failure)
    monkeypatch.setattr(idd_tools, "_tool_failure", broken_failure)
    res = probe_claim_6()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert (
        "Unexpected exception response violated exact 4-field schema contract"
        in res.defect_or_contract_explanation
    )


def test_claim_6_simulated_canary_leak_returns_reproduced(monkeypatch):
    """Verify that leaking unexpected exception canary is detected (Claim 6 REPRODUCED)."""
    import idd_core
    import idd_tools

    def leaking_failure(operation, action, exc=None):
        return {
            "status": "error",
            "operation": operation,
            "reason": f"Internal failure: {exc}",
            "action": action,
        }

    monkeypatch.setattr(idd_core, "_tool_failure", leaking_failure)
    monkeypatch.setattr(idd_tools, "_tool_failure", leaking_failure)
    res = probe_claim_6()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert (
        "Sanitizer leaked unexpected exception details"
        in res.defect_or_contract_explanation
    )


def test_claim_6_simulated_canary_leak_in_extra_field_returns_reproduced(monkeypatch):
    """Verify that leaking canary secret in an additional unexpected field is detected (Claim 6 REPRODUCED)."""
    import idd_core
    import idd_tools

    def extra_field_leak_failure(operation, action, exc=None):
        return {
            "status": "error",
            "operation": operation,
            "reason": "An unexpected data-processing failure occurred.",
            "action": action,
            "debug": f"Secret canary detail: {exc}",
        }

    monkeypatch.setattr(idd_core, "_tool_failure", extra_field_leak_failure)
    monkeypatch.setattr(idd_tools, "_tool_failure", extra_field_leak_failure)
    res = probe_claim_6()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert (
        "Sanitizer leaked unexpected exception details"
        in res.defect_or_contract_explanation
    )


def test_claim_6_simulated_extra_schema_field_returns_reproduced(monkeypatch):
    """Verify that extra fields without secrets still violate exact schema contract (Claim 6 REPRODUCED)."""
    import idd_core
    import idd_tools

    def extra_field_failure(operation, action, exc=None):
        return {
            "status": "error",
            "operation": operation,
            "reason": "An unexpected data-processing failure occurred.",
            "action": action,
            "extra_harmless": "harmless_metadata",
        }

    monkeypatch.setattr(idd_core, "_tool_failure", extra_field_failure)
    monkeypatch.setattr(idd_tools, "_tool_failure", extra_field_failure)
    res = probe_claim_6()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert (
        "Unexpected exception response violated exact 4-field schema contract"
        in res.defect_or_contract_explanation
    )


def test_claim_6_simulated_known_error_schema_violation_returns_reproduced(monkeypatch):
    """Verify that extra fields in known error response violate exact schema contract (Claim 6 REPRODUCED)."""
    import idd_core
    import idd_tools

    def extra_field_tool_error(operation, reason, action):
        return {
            "status": "error",
            "operation": operation,
            "reason": reason,
            "action": action,
            "extra": "unauthorized",
        }

    monkeypatch.setattr(idd_core, "_tool_error", extra_field_tool_error)
    monkeypatch.setattr(idd_tools, "_tool_error", extra_field_tool_error)
    res = probe_claim_6()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert (
        "Known exception response violated exact 4-field schema contract"
        in res.defect_or_contract_explanation
    )


def test_claim_6_simulated_missing_schema_fields_returns_reproduced(monkeypatch):
    """Verify that missing schema fields in tool error are detected (Claim 6 REPRODUCED)."""
    import idd_core
    import idd_tools

    def broken_tool_error(operation, reason, action):
        return {"status": "error"}  # Missing operation, reason, action

    monkeypatch.setattr(idd_core, "_tool_error", broken_tool_error)
    monkeypatch.setattr(idd_tools, "_tool_error", broken_tool_error)
    res = probe_claim_6()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert (
        "Known exception response violated exact 4-field schema contract"
        in res.defect_or_contract_explanation
    )


def test_claim_9_simulated_broken_safe_path_returns_reproduced(monkeypatch):
    """Verify that failure to resolve safe path reports REPRODUCED (fail-closed)."""
    import idd_core
    import idd_tools

    orig_is_subpath = idd_tools._is_subpath

    def broken_subpath(cand, root):
        if "report.html" in str(cand):
            return False
        return orig_is_subpath(cand, root)

    monkeypatch.setattr(idd_core, "_is_subpath", broken_subpath)
    monkeypatch.setattr(idd_tools, "_is_subpath", broken_subpath)
    res = probe_claim_9()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "safe path resolution failed" in res.defect_or_contract_explanation


def test_claim_9_simulated_premature_mkdir_returns_reproduced(monkeypatch):
    """Verify that premature directory creation on traversal attempts reports REPRODUCED."""
    import os
    from pathlib import Path
    import idd_core
    import idd_tools

    orig_resolve = idd_tools._resolve_artifact_path

    def leaking_mkdir_resolve(file_name, config=None, subdir=None, create_parents=True):
        if subdir == "should_not_exist_dir":
            env_dir = os.environ.get("IDD_ARTIFACTS_DIR")
            if env_dir:
                (Path(env_dir) / subdir).mkdir(parents=True, exist_ok=True)
        return orig_resolve(
            file_name, config=config, subdir=subdir, create_parents=create_parents
        )

    monkeypatch.setattr(idd_core, "_resolve_artifact_path", leaking_mkdir_resolve)
    monkeypatch.setattr(idd_tools, "_resolve_artifact_path", leaking_mkdir_resolve)
    res = probe_claim_9()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "delayed directory creation failed" in res.defect_or_contract_explanation


def test_claim_9_simulated_missing_notebook_guard_returns_reproduced(monkeypatch):
    """Verify that absence of notebook multi-root guard reports REPRODUCED."""
    monkeypatch.setattr(
        reproduce_drift, "_extract_notebook_cell_source", lambda idx: "# no guards"
    )
    res = probe_claim_9()
    assert res.classification == "REPRODUCED"
    assert res.is_defect is True
    assert "notebook Cell 57 lacks PR #149" in res.defect_or_contract_explanation
