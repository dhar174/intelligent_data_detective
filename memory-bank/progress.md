<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Progress

## Milestones
- [x] Repository profile captured (inventory run 2026-04-20)
- [x] Hybrid agent stack scaffolded (26 files, `repo-agent-bootstrap`)
- [x] False-positive agent deleted (`frontend-experience-specialist`)
- [x] IDD-specific agents created/customised
- [x] Memory-bank and AGENTS.md enriched with real IDD content
- [x] W14 completion proof reached (`IDD_run_run_default_id-20260504-1338-b3079aea`)
- [x] Final validators added and pushed (`validate_run.py`, `validate_artifact_quality.py`)
- [x] Validator run and W14 warnings resolved
- [x] Stale Phase 6 GitHub issues closed/rescoped after W14 proof (#112-#119 closed, #121 post-W14 hardening)
- [x] No-key CI workflow added for validator/unit/integration checks
- [ ] Maintenance drift reviewed against future repo changes

## What works
- Core test suite: **22/22 pass** (`test_intelligent_data_detective.py`)
- Error handling tests: **16/16 pass** (`test_error_handling_framework.py` passes completely without `--deselect`)
- Memory test suites: all pass
- Notebook execution: functional end-to-end (requires `OPENAI_API_KEY`; 6–25 min)
- Historical W14 completion proof: `validate_run.py` 12/12 and `validate_artifact_quality.py` 9/9 on `retail_orders`; not a fresh PR #153 proof.
- Historical W14 artifacts: canonical root `final_report.html`, `final_report.md`, `final_report.pdf`, resolving HTML images, parseable PDF, no marker `.txt` artifacts.
- Agent stack: all guidance files present, managed sections safe for future maintenance runs

## What is incomplete
- #115/#119 remain open production-completion blockers: the reported September 29 run failed 5/12 with zero sections and `report=None`. PR #153 is bootstrap/registry prerequisite work, not report routing or production certification.
- Whole-runtime consolidation of the textual Python export belongs to #140; registry ownership/deduplication and broader persistence policy remain in #141. Only selected snippet parity is tested here.
- Antigravity memory-rule proposals are inactive evidence for #152, not empirically loaded rules; [preserved memory proposal](../docs/evidence/proposals/issue-152-memory.txt), [workflow proposal](../docs/evidence/proposals/issue-152-memory-workflow.txt).
- CI intentionally excludes the full API-key notebook trajectory proof; `.github/workflows/copilot-setup-steps.yml` currently gates only no-key validator/unit/integration checks, root regression checks, notebook smoke validation, and targeted lint/format steps on `pull_request`, `push` to `main`, and `workflow_dispatch`.
- `automation.instructions.md` is a scaffolded stub; no additional automated hooks exist yet.
- Deferred prompt/report polish remains blocked unless explicitly scoped against W14 gates.
- The historical test failure in `test_error_handling_framework.py` is fully resolved by signature-aware binding in `idd_tools.py` (16/16 pass without `--deselect`).

## Validation status
PR #153 revised-worktree evidence: [commands, dependency environment, skips, known isolated failure, and actual-kernel smoke](../docs/evidence/pr-153-validation.json). Required local suites passed (451 no-key + 94 root/prompt + 15 required error-handling); actual-kernel smoke passed 3/3. The exact pre-existing signature exception still fails separately. These are local results, not hosted CI or production certification.

```bash
python -m pytest test_validate_run.py -q
python -m pytest test_validate_run.py tests/unit tests/integration -q
python -m flake8 validate_run.py validate_artifact_quality.py test_validate_run.py --max-line-length=120 --extend-ignore=E203,W503
python validate_run.py --latest --log-path notebook_run_log.txt --window 180
python validate_artifact_quality.py --latest
python3 -m pytest test_intelligent_data_detective.py -v          # 22/22
python3 -m pytest test_error_handling_framework.py -v            # 15/16
python3 -m pytest -v                                             # all suites
flake8 test_intelligent_data_detective.py --max-line-length=88 --extend-ignore=E203,E501
```

## Last meaningful update
Post-W14 repo hygiene: closed stale Phase 6 issues, rescoped #121 as defensive hardening, replaced the CI stub with no-key validation, and added a lightweight W14 patched-notebook marker regression test.
<!-- repo-agent-bootstrap:managed:end -->

<!-- session-curated:start -->
## Pipeline run history (current session)

| Run | Outcome | Notes |
|---|---|---|
| Failed Live (reported 2026-09-29) | **5/12 (`validate_run.py`), not accepted proof** | `run_default_id-20260929-0210-367d732a` (877s). Zero sections, `report=None`, two distinct visualizations. [Stable log](../docs/evidence/runs/2026-09-29-run_default_id-20260929-0210-367d732a/notebook_run_log.txt) and [reported environment/validator/provenance](../docs/evidence/runs/2026-09-29-run_default_id-20260929-0210-367d732a/provenance.json). Execution SHA/notebook hash unavailable; run predates the final PR candidate. Report-dispatch repair deferred to #115/#119. |
| 73 | Baseline; `viz=True / report=True` | Every agent hit `recursion_limit=160` and went down the recovery path. Artifacts produced but not clean. |
| 74 | **Regression — BR-7** | Managed-channel collision (`remaining_steps`) after switching `State` to inherit from `langgraph.prebuilt` AgentState. Fix: declare `messages` directly with `add_messages`. |
| 75 | **Regression — BR-8** | `InvalidUpdateError` on `structured_response` at cell 81 (concurrent writes from main node + recovery shims). Also: 12× final-hop `with_structured_output` recovery hits in `viz_evaluator`. Fix: introduce `_sr_reducer`. |
| 76 | **In flight** | BR-8 fixed. `visualization` completed cleanly. `viz_evaluator` silent ≥16 min — under investigation. |
| 81–82 | Lucky-pass | FINAL+viz+report ✅, but PYTHONHASHSEED happened to roll an order where AgentState/State sr-channel merge succeeded. |
| 83–84 | Regression | `_resolve_schema` set-iteration rolled adverse order → cell 48 `ValueError` on sr channel mismatch. |
| 85 | W7-SR-ALIGN landed | Cell 48 ✅; pipeline still failed at viz_worker on `InvalidUpdateError` (sr LastValue, multiple writers). |
| 86 | + W8-VW-NOSR | viz_worker wrapper no longer writes sr; ❌ still failed — W2-BA-finalhop recovery shim also writes sr. |
| 87 | **+ W9-SR-DROP — 7/8 GREEN** | sr field removed from supervisor State entirely. FINAL viz=True report=True. 0 recoveries / 0 finalhop / 0 tracebacks / 0 W4. ❌ PDF only. ~21 min wall. |

### Status of structured_response collision: ✅ SOLVED (W9-SR-DROP, Run 87)
Root cause was twofold: (a) `langchain.agents.factory._resolve_schemas` iterates `set({State, AgentState[ResponseT]})` nondeterministically (caused intermittent cell-48 ValueError pre-W7); (b) once W7 forced LastValue on supervisor sr channel, multiple writers in same superstep collided. W9 eliminates the supervisor channel entirely; each create_agent subgraph keeps its own sr internally and wrapper code reads it from `result["structured_response"]` Python dict (not a graph channel). Verified by static grep — no supervisor-state consumer of `state["structured_response"]`.

### Current focus (post-Run 87)
**W10-PDF-DIAG** — `STAGE file_writer DONE` marker never logged; W6-FW-PDF-FORCE block present but unreached. Likely `result["structured_response"]` KeyError or `ListOfFiles` assertion failure in file_writer_node post-invoke wrapper after W9 caused agent-subgraph→supervisor merge to drop sr. Two paths: harden extraction (W10-FW-RESULT-GUARD) or add a separate post-graph PDF writer node (W10-PDF-DIRECT).
<!-- session-curated:end -->

<!-- session-curated:2026-04-23-phase6:start -->
## Pipeline run history — Phase 6 update (2026-04-23 later)

| Run | Outcome | Notes |
|---|---|---|
| 87 | + W9-SR-DROP — **7/8 GREEN** | viz=True report=True, 0 recoveries / 0 finalhop / 0 tracebacks. ❌ PDF only. ~21 min. |
| 88 | + W10-PDF-POST — **8/8 GREEN** structurally, **HOLLOW** semantically | Historical hollow-report regression: report = 356-char placeholder, 0 sections, 1 viz embedded 5×, 25+ stub marker files in reports dir. Analyst output was rich and correct — content lost between analyst and report. Triggered Phase 6 pivot. |

### Acceptance bar — REPLACED (2026-04-23)
The "8/8 CLEAN" structural success criterion is **OBSOLETE** as of Run 88. The new active bar is the **Phase 6 12-criteria content-quality gate**:

1. Report HTML text-only content **≥ 3000 chars**
2. Report has **≥ 4 distinct sections**
3. **≥ 3 distinct visualizations** — different titles AND different image bytes
4. **PDF size ≥ 30 KB**
5. **0 stub / marker files** in reports dir (no `*_ack`, `*_commit`, `*_ready`, `*_stub`, `*_trigger`, …)
6. Report references actual analyst findings (correlation **r values** and **anomaly notes** appear verbatim)
7. (+ remaining criteria 7–12 from Phase 6 plan — TBD enumerated against G)

A run that hits 8/8 structural gates but fails any of the 12 content gates is **not** a passing run.

### What works (status update)
- Core test suite: **22/22 pass** (unchanged)
- Error handling: **15/16 pass** (unchanged known edge case)
- Memory test suites: all pass (unchanged)
- Pipeline structurally clean end-to-end (Run 87 / 88)
- W7-SR-ALIGN, W8-VW-NOSR, W9-SR-DROP, W10-PDF-POST all confirmed working

### What is incomplete (Phase 6)
- **RC1** — viz fan-out: only 1 of N analyst-recommended viz_specs survives to disk.
- **RC2** — `report_section_worker`: writes 0 sections to `written_sections`.
- **RC3** — `file_writer`: emits a 356-char placeholder for body content; spams 25+ stub-marker tools to compensate.
- **RC4** — supervisor FINAL gate accepts hollow completion (no content-validation preconditions).
- **RC5 (NEW)** — structured Pydantic outputs (`AnalysisInsights`, `ReportResults`, …) may not actually be persisted to supervisor `State` after W9-SR-DROP — wrapper code reads them from `agent.invoke()` dict, but downstream nodes may be reading stale / empty State fields. Needs forensic confirmation in Phase A.

---

## Canonical Core Roadmap Progress (Issue #140)

### Checkpoint 0 (CP0) — Completed & Merged
- [PR #155](https://github.com/dhar174/intelligent_data_detective/pull/155) merged at commit `98da83a9d90a334b5a7ea98e0acb97dd78fa0548`.
- Established comprehensive parity inventory (`docs/architecture/idd-core-parity-inventory.md`) and migration specification (`docs/plans/idd-canonical-core-migration.md`).
- Established 12-probe automated diagnostic suite (`tools/diagnostics/reproduce_drift.py` and `test_cp0_diagnostics.py`).

### Checkpoint 1 (CP1) — Implemented, Corrective Pass & Security Hardening Completed
- Branch: `antigravity/issue-156-canonical-models-registry`.
- PR: [PR #157](https://github.com/dhar174/intelligent_data_detective/pull/157).
- Canonical planning models (`idd_models.py`) with explicit Plan lifecycle semantics (new vs. restore) and RC-2 deduplication.
- Canonical thread-safe `DataFrameRegistry` (`idd_registry.py`) with `ContextVar` override engine and user-file safety.
- Atomic bridge in `idd_core.py` with `__getattr__` dynamic delegation and migrated callers in fixtures/diagnostics.
- **CP1 Corrective Pass (2026-10-12)**:
  - Resolved P1 finding from independent review `pullrequestreview-5497205553` regarding representation fidelity across cache eviction.
  - Transitioned internal backing snapshots from `.csv` to `.pkl` (`idd_registry.py`), losslessly preserving mixed integer column names, timezones, MultiIndexes, and nullable dtypes.
  - Preserved strict trusted storage boundary (internal `.pkl` in `self.data_dir`; external `raw_path` formats untouched).
  - Added 10 comprehensive regression tests (Tests A–J) in `TestLosslessDataFramePersistence` (`tests/unit/test_registry.py`).
  - GitHub Actions CI `No-key validation` passed cleanly (1m53s).
- **CP1 Security Pass (2026-10-13)**:
  - Resolved security finding from independent review `pullrequestreview-5497824478` on private storage boundaries for pickle snapshots.
  - Replaced default storage directory creation with `tempfile.mkdtemp(prefix="idd_registry_", dir=str(WORKING_DIRECTORY))` and explicit mode `0700` (`drwx------`) on POSIX.
  - Hardened file permissions on POSIX: staging `.tmp` files and final `.pkl` snapshots created with mode `0600` (`-rw-------`) prior to atomic `os.replace`.
  - Added validation for configured `data_dir`: rejects volume roots (`/`, `C:\`), non-directory paths, world-writable directories (`stat.S_IWOTH`), and unowned directories (`st.st_uid != os.getuid()`) on POSIX.
  - Enforced strict deserialization trust boundary: pickle deserialization confined to verified storage in `self.data_dir.resolve()` via `is_relative_to()`.
  - Documented Windows NTFS DACL security model in `DataFrameRegistry.__doc__`.
  - Added `TestPrivateStorageSecurity` with 9 targeted unit tests in `tests/unit/test_registry.py` (79 total tests passing in `test_registry.py`).
  - Full test suites: 537 passed, 9 skipped in Linux CI (533 passed, 13 skipped on Windows).
  - GitHub Actions CI `No-key validation` ran green (2m1s, run ID `38334366737`).
  - Pushed at commit `396df69`.
- Zero notebook churn across 99 cells and full baseline quality gate compliance.
- Status: **CP1 MERGED TO MAIN** (PR #157 merged at commit `196f813e0d74d0ee71782634c8028a00989de5c2`, hosted CI passed 546/546). Issue #156 closed.

### Checkpoint 2 (CP2) — Canonical Tools & Error Handling Hardening (Issue #158)
- Scope Delivered:
  - Canonical tools and error handling module: `idd_tools.py`.
  - Standardized 4-field error dictionary contract: `{'status': 'error', 'operation': str, 'reason': str, 'action': str}` matching notebook Cell 32.
  - Exception sanitization via `_tool_failure()`: logs diagnostic tracebacks locally via `logger.exception()` while presenting user-safe guidance for unexpected exceptions; distinct actionable messages for known domain errors (`FileNotFoundError`, `KeyError`, `ValueError`, `EmptyDataError`, `ParserError`, `DtypeWarning`).
  - Signature-aware argument binding in `@handle_tool_errors`: inspects `inspect.signature(func)` and uses `.bind(*args, **kwargs)` with `.apply_defaults()`. Detects argument count mismatches, binds `df_id` regardless of positional index, keyword argument, or parameter object attribute (`.df_id`), and skips DataFrame validation completely for tools without a `df_id` parameter (resolving Claim 7 and historical failure in `test_error_handling_framework.py`).
  - Canonical `validate_dataframe_exists()` delegating to authoritative `get_global_registry().get_dataframe(df_id, load_if_not_exists=True)`, supporting all source formats across cache evictions while preserving context isolation.
  - Artifact path resolution (`_resolve_artifact_path`, `_get_artifacts_base`, `_is_subpath`, `is_safe_artifact_path`) preserving PR #149 security invariants: path traversal rejection, delayed parent directory creation, and multi-root read validation for downstream reporting nodes.
  - Compatibility façade in `idd_core.py` updated with 100% symbol re-exports and identity.
- Claim Resolutions:
  - Claim 6 (Structured tool error schema divergence): **OBSERVED INVARIANT**.
  - Claim 7 (Signature-aware df_id extraction): **OBSERVED INVARIANT**.
  - Claim 9 (Artifact path containment & multi-root protections): **OBSERVED INVARIANT**.
- Verification Baseline:
  - `test_error_handling_framework.py`: **16/16 passed** without `--deselect`!
  - `tests/unit/test_canonical_tools.py`: **24/24 passed** (new unit test suite).
  - `tests/unit/test_handle_tool_errors.py`: **24/24 passed** (migrated from string to dict assertions).
  - `tools/diagnostics/reproduce_drift.py` & `tools/diagnostics/test_cp0_diagnostics.py`: **28/28 passed**, 12/12 evaluated cleanly with 0 blocked/failed.
  - Full test suites: **574 unit/integration passed**, **110 core/memory passed**, **17 prompt passed**.
  - Zero notebook churn: 0 notebook lines modified, 99 cells intact, 15 nodes / 33 edges state graph verified.
- Status: **CP2 COMPLETE & VERIFIED** (Issue #158).

