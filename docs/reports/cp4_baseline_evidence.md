# CP4 Baseline Evidence Document

**Generated:** 2026-10-14  
**Base Commit SHA:** `9a6ad31fcde1e5a7c99f4e21f1e382a9da77495d` (PR #161 merge commit on `origin/main`)  
**Verified CI Run on `main`:** https://github.com/dhar174/intelligent_data_detective/actions/runs/38649652574 (100% green)  
**Parent Roadmap Issue:** [#140](https://github.com/dhar174/intelligent_data_detective/issues/140)  
**Control-Plane Audit Issue:** [#144](https://github.com/dhar174/intelligent_data_detective/issues/144)  
**Implementation Tracking Issue:** [#162](https://github.com/dhar174/intelligent_data_detective/issues/162)  
**Branch:** `antigravity/cp4-canonical-notebook-integration`

---

## 1. File SHA-256 Digests

| File | SHA-256 Digest | Role / Status |
|---|---|---|
| `idd_models.py` | `86c0dc17c377e6d0fa6ce8fd016c01b959f953eb450e30c50c8ca4ba023043fe` | Canonical Pydantic models & validation |
| `idd_registry.py` | `091bcba8c55da857cc797c19653bac06ddce8cc8743a2928571903807d2e5725` | Canonical thread-safe LRU DataFrameRegistry |
| `idd_state.py` | `d489c21d9e075f9a8aafc5b2eabc3489efdf3e38c392be148b1c28b0a6450360` | Canonical 70-field State, VizWorkerState, reducers |
| `idd_tools.py` | `b6fcfdae88f9f28cf87f0fdf7c3ffefbb3b4b9c7f22000fa5401ac8e5c897eb2` | Canonical @handle_tool_errors, artifact resolution |
| `idd_graph.py` | `9f613bd604c2c8bbbb559e1f7ec18b01ac62533738d65c54ce36cd06c57753b7` | Canonical 15-node, 33-edge StateGraph builder |
| `idd_core.py` | `7a89052fff8f2743ef240d78fc9efe157f60d6b8e3269e78863ad99da80a02f3` | Canonical compatibility re-export façade |
| `idd_bootstrap.py` | `b30c61af52e4fb75d70c50730f0686bff010b996b5e14659d9bd6592cfe328d2` | Runtime bootstrap & manifest validator |
| `idd_manifest.json` | `c508a6b480f80c5b89d5b5453c1a501c444267830f7b0f89cdde860d04e0c565` | Cryptographic SHA-256 provenance manifest |
| `_patch_notebook.py` | `dbcc9ce9f26e39530a5361cf0450e728767cb2d0dda24cca9a0d133a3b113514` | Authoritative notebook patch generator |
| `IntelligentDataDetective_beta_v5_patched.ipynb` | `cedad893f0aa9e9760722f1ffd0af1e53461f81e2c290b464d26da44160d88bc` | Runnable 99-cell production notebook |
| `IntelligentDataDetective_beta_v5.ipynb` | `6cbbea283b0b0da2e5bf5d7cbcaead4dae224144301b04db790555cadfb10d43` | Original source notebook input |
| `tests/unit/test_core_parity_guard.py` | `3f8d86b193ae66097f6c6a2021ad6e2885caa514c8f8c165cd15511474b51c91` | Automated 4-layer anti-drift parity guard |

---

## 2. Notebook Structure & Topology Invariants

- **Total Notebook Cell Count:** exactly 99 cells (matching W14 completion baseline).
- **State Schema:** exactly 70 fields:
  - 39 reducer fields: `messages`, `last_agent_id`, `_report_dispatched`, `current_turn_agent_id`, `last_agent_message`, `last_agent_expects_reply`, `last_agent_reply_msg`, `last_agent_finished_this_task`, `last_created_obj`, `final_turn_msgs_list`, `supervisor_to_agent_msgs`, `emergency_reroute`, `current_plan`, `viz_tasks`, `viz_results`, `_supervisor_turn_count`, `sections`, `written_sections`, `report_draft`, `completed_plan_steps`, `progress_reports`, `completed_tasks`, `viz_paths`, `report_paths`, `run_id`, `visualization_results`, `report_results`, `file_results`, `initial_analysis_complete`, `data_cleaning_complete`, `analyst_complete`, `file_writer_complete`, `visualization_complete`, `report_generator_complete`, `_count_`, `artifacts_path`, `reports_path`, `logs_path`, `final_report_path`.
  - 31 plain fields: `next`, `_viz_retry_count`, `__active_response_tool__`, `user_prompt`, `current_dataframe`, `current_dataframe_id`, `available_df_ids`, `individual_viz_task`, `viz_revise_count`, `viz_eval_result`, `viz_grade`, `viz_feedback`, `viz_specs`, `viz_spec`, `section`, `report_outline`, `to_do_list`, `latest_progress`, `initial_analysis_agent`, `data_cleaner_agent`, `analyst_agent`, `analysis_config`, `_config`, `file_content`, `next_agent_prompt`, `next_agent_metadata`, `initial_description`, `cleaning_metadata`, `analysis_insights`, `cleaned_dataset_description`, `_id_`.
- **Graph Topology:**
  - 15 production nodes: `supervisor`, `initial_analysis`, `data_cleaner`, `analyst`, `visualization`, `viz_worker`, `viz_join`, `viz_evaluator`, `report_orchestrator`, `report_section_worker`, `report_join`, `report_packager`, `file_writer`, `EMERGENCY_MSG`, `FINISH`.
  - 33 total edges (11 static edges, 21 conditional edges, 1 START entry edge).
  - 0 unreachable nodes.
  - 0 dead-end nodes.

---

## 3. Test Skip Inventory

From `pytest test_validate_run.py tests/unit tests/integration -q -rs`:
- Total tests collected: 682
- Total passed: 663
- Total skipped: 19
  - `test_validate_run.py:29: Run 88 fixture missing` (historical telemetry fixture).
  - `tests/unit/test_registry.py:109, 1461: pyarrow not installed` (optional environment dependency).
  - `tests/unit/test_registry.py:1165, 1173, 1185, 1195, 1270, 1285: POSIX permission/ownership tests` (skipped on Windows host platform).
  - `tests/unit/test_patcher_integrity.py`: 10 skipped inline compilation tests superseded by canonical CP4 modules.
- Note: 100% of graph compilation and routing integration tests pass completely offline.
- Note: 16/16 error handling framework tests pass cleanly without `--deselect`.
- Note: 22/22 anti-drift tests pass in `test_core_parity_guard.py`.

---

## 4. CP4 Parity & Integration Results

Post-CP4 integration:
- Cells 4, 16, 19, 22, 32, and 60 in `IntelligentDataDetective_beta_v5_patched.ipynb` import directly from canonical modules (`idd_bootstrap`, `idd_models`, `idd_registry`, `idd_state`, `idd_tools`, `idd_graph`).
- Duplicate redefinitions of `Plan`, `State`, `DataFrameRegistry`, `handle_tool_errors`, `_resolve_artifact_path`, and `build_graph` have been completely eliminated from the notebook.
- Automated anti-drift guard `tests/unit/test_core_parity_guard.py` enforces static AST invariants and runtime object identity across all 99 cells.
- Zero defects reproduced, 9 observed invariants verified in `reproduce_drift.py --json`.
