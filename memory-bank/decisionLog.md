<!-- repo-agent-bootstrap:file-kind=memory-bank -->
# Decision Log

Architectural decisions and root-cause analyses for the IDD v5 pipeline. Append-only; cite sentinel IDs and run numbers.

---

## 2026-04-23 — ADR: Drop `structured_response` from supervisor State (W9-SR-DROP)

### Status
✅ Accepted and verified (Run 87).

### Context
Across Runs 75–86 the supervisor graph repeatedly failed with one of two errors that traced back to the same field, `State.structured_response`:

1. **Pre-W7 era (Runs 83–84):** `ValueError` at notebook cell 48 (the `create_agent(...)` calls) — only on some runs, others (Runs 81–82) passed unchanged.
2. **Post-W7 era (Runs 85–86):** `InvalidUpdateError: At key 'structured_response': Can receive only one value per step` during the `viz_worker` fan-out superstep.

### Root cause
`langchain.agents.factory._resolve_schemas` constructs the merged StateSchema/InputSchema/OutputSchema for `create_agent` by iterating `set({State, AgentState[ResponseT]})`. Python `set` iteration order is governed by PYTHONHASHSEED and therefore **nondeterministic across processes**. For any field name appearing on both schemas — `structured_response`, `messages`, `remaining_steps` — the "winning" channel definition depends on which class iterates last.

For `structured_response` specifically:
- Supervisor `State` declared `Annotated[Optional[BaseNoExtrasModel], _sr_reducer]` → langgraph `BinaryOperatorAggregate` channel.
- `AgentState[ResponseT]` declares `Annotated[ResponseT, OmitFromSchema(input=True, output=False)]` → langgraph `LastValue` channel.

`langgraph.channels.last_value._add_schema` (state.py:14–19) only tolerates a type mismatch between two schemas if the **second** channel is `LastValue`. With nondeterministic iteration, ~half of process starts produce StateSchema=LastValue + InputSchema=BinaryOperatorAggregate → `ValueError` at create_agent time. The other half pass (Runs 81–82 were lucky hash-seed rolls).

W7-SR-ALIGN fixed the cell-48 nondeterminism by aligning the user annotation to AgentState's exactly (both LastValue + same `OmitFromSchema` metadata → set merges to a single channel regardless of order). But this collapsed the supervisor sr channel to `LastValue`, so concurrent writes from (a) viz_worker wrapper, (b) W2-BA-finalhop recovery shim, (c) agent-subgraph propagation all collided in the same superstep → `InvalidUpdateError`.

W8-VW-NOSR removed writer (a) but writers (b) and (c) still triggered the error. Renaming-per-writer (Path A) was rejected as too invasive.

### Decision
**Path B — remove `structured_response` from the supervisor `State` TypedDict entirely.** Each `create_agent` subgraph still owns its own `structured_response` via internal `AgentState[ResponseT]`. Wrapper nodes read it from the Python-dict return of `agent.invoke()` (e.g., `result["structured_response"]`) — this is dict access, **not** a langgraph channel read. Static grep across the patched notebook confirmed zero supervisor-state reads of `state["structured_response"]` before adopting this decision.

### Consequences
- ✅ **Run 87 verified:** FINAL marker reached with viz=True report=True, 0 `InvalidUpdateError`, 0 recoveries, 0 W2-BA-finalhop hits, 0 W4 negatives, 0 tracebacks. Native structured Pydantic outputs throughout. ~21 min wall.
- ✅ Without a supervisor channel for sr, parallel writers silently drop instead of colliding at LastValue's "one write per step" contract — exactly the desired behaviour.
- ⚠️ Side effect: `file_writer_node`'s wrapper code expects `result["structured_response"]` to be a `ListOfFiles` Pydantic instance after agent.invoke. Post-W9, the agent-subgraph→supervisor-state merge can return a dict missing the key (or with a stale value), causing silent failure in the file_writer post-invoke path. **This is the W10-PDF-DIAG remaining gap** — `STAGE file_writer DONE` marker never logged, W6-FW-PDF-FORCE block unreached.

### Sentinels
- `W7-SR-ALIGN` — `_patch_notebook.py` ~L7668 (cell 22 State annotation aligned to AgentState). Required precondition.
- `W8-VW-NOSR` — `_patch_notebook.py` ~L7720 (cell 57 viz_worker wrapper drops sr from return dict). Necessary but insufficient.
- `W9-SR-DROP` — cell 22 `structured_response` field removed from `State` TypedDict. The breakthrough.

### Pattern (recorded in `systemPatterns.md`)
For `create_agent` compatibility with custom supervisor `State`, do not add custom-reducer fields whose names collide with `AgentState` fields (`structured_response`, `messages`, `remaining_steps`). Either omit the field from `State` (preferred for sr) or mirror AgentState's annotation exactly (only safe under single-writer guarantees).

### References
- `langchain.agents.factory._resolve_schemas` — set-iteration nondeterminism (root cause of cell-48 ValueError).
- `langgraph.channels.last_value._add_schema` (state.py:14–19) — type-mismatch tolerance rule (only when second channel is LastValue).
- Session files: `run85-86-summary.md`, `run87-summary.md`, `run87-postmortem.md`.

### Follow-up
- **W10-PDF-DIAG** (next session): harden `file_writer_node` extraction to defensively handle missing `structured_response` key (W10-FW-RESULT-GUARD), or add a separate post-graph PDF-writer node that reads HTML from disk (W10-PDF-DIRECT).

---

## 2026-04-23 (later) — ADR: Phase 6 pivot — content quality replaces structural cleanliness

### Status
✅ Accepted. Active acceptance bar.

### Context
Run 87 hit 7/8 structural gates and Run 88 hit 8/8 — but Run 88's deliverable is a **Potemkin pipeline**: 356-char placeholder report, 0 sections, 1 viz embedded 5× with cosmetic hash variation, 1983-byte title-only PDF, and 25+ stub-marker files (`*_ack`, `*_commit`, `*_ready`, `*_stub`, `*_trigger`, …) spammed by `file_writer` to compensate for empty `written_sections`. Analyst output was rich and correct; content was lost downstream.

### Decision
**Retire the "8/8 CLEAN" structural success criterion. Adopt a 12-criteria content-quality acceptance bar.** The first six criteria (the rest TBD as Phase G is enumerated):
1. Report HTML text-only ≥ 3000 chars
2. Report has ≥ 4 distinct sections
3. ≥ 3 distinct visualizations (different titles AND different image bytes)
4. PDF size ≥ 30 KB
5. 0 stub/marker files in reports dir
6. Report references actual analyst findings (correlation r values, anomaly notes appear verbatim)

A run that passes 8/8 structural gates but fails any content gate is **not** a passing run.

### Consequences
- All Phase 6 work (issues #112–#118 under epic #119) is judged against this bar.
- Existing W7–W10 patches stay in place; they are necessary but not sufficient.
- The previously-tracked "PDF artifact emission" subgoal collapses into criterion 4 (PDF ≥ 30 KB), which requires real HTML body content first.

---

## 2026-04-23 (later) — ADR: Wave 5 mode — direct-to-notebook edits; patcher → changelog

### Status
✅ Accepted.

### Context
Through Wave 4, all notebook fixes flowed through `_patch_notebook.py` as sentinel-bracketed blocks (~80 sentinels accumulated). With W7/W8/W9/W10 confirmed working and Phase 6 requiring rapid iterative content fixes, the patcher indirection adds latency without buying safety.

### Decision
**Wave 5: edit `IntelligentDataDetective_beta_v5.ipynb` directly.** `_patch_notebook.py` is preserved as a historical changelog only. Pre-cutover backups live at `_wave5_backup_20260423-091557/` (SOURCE notebook + patcher + patched notebook all preserved).

### Consequences
- Faster iteration on Phase 6 fixes.
- The patcher must NOT be re-introduced for new Wave 5 work; if a regression requires re-applying a Wave 4 sentinel, do it as a direct notebook edit and reference the original sentinel ID in the commit / decision log.
- Rollback for any Phase 6 cell change is the `_wave5_backup_20260423-091557/` snapshot.

---

## 2026-04-23 (later) — ADR: `ChatOpenAI` direct over `MyChatOpenai` (legacy)

### Status
✅ Accepted (per user, 2026-04-23). ⚠️ Conflicts with managed agent doc — see Consequences.

### Context
`MyChatOpenai` (Cell 5) was a custom wrapper around `ChatOpenAI` that historically handled o-series model quirks and the OpenAI Responses API. Per user, those quirks no longer require the wrapper layer.

### Decision
**Use `ChatOpenAI` directly going forward.** Existing `MyChatOpenai` call sites can be migrated opportunistically; new code must not introduce new `MyChatOpenai` references.

### Consequences
- ⚠️ This conflicts with `.github/agents/notebook-specialist.agent.md`, which (in its `<!-- repo-agent-bootstrap:managed -->` block) still mandates `MyChatOpenai`. **The managed block is not edited** (per maintenance contract). The divergence is documented here and in `memory-bank/activeContext.md`. When the managed block is next regenerated by `repo-agent-bootstrap`, the new `ChatOpenAI` rule should be re-asserted.
- `memory-bank/systemPatterns.md` managed block also still says "Always use MyChatOpenai" — same constraint applies.

---

## 2026-04-23 (later) — ADR: Zero stub/marker files in reports directory

### Status
✅ Accepted. Hard gate (criterion 5 of Phase 6 12-criteria).

### Context
Run 88's `file_writer` LLM compensated for an empty body by calling marker tools 25+ times, producing a clutter of files matching `*_ack`, `*_commit`, `*_ready`, `*_stub`, `*_trigger`. These have no downstream consumer.

### Decision
**Reports dir must contain 0 stub/marker files.** Implemented in Phase E by:
1. Removing stub-marker tools from `file_writer`'s tool list.
2. Capping total tool calls per `file_writer` invocation.
3. Rewriting the `file_writer` system prompt to forbid placeholder/marker emissions.

### Consequences
- Tools that other agents legitimately call as no-op markers (if any) must be re-scoped or renamed; `file_writer` must not have access.
- Validation: glob the reports dir post-run; any match against the stub patterns above fails the run.

---

## 2026-04-23 (later) — ADR: GitHub epic + 7 sub-issues for Phase 6

### Status
✅ Accepted.

### Context
Phase 6 spans seven distinct workstreams (forensic, instrumentation, viz pipeline, report pipeline, file_writer, supervisor gate, validation) with sequential dependencies (A → B → {C,D,E} → F → G).

### Decision
Tracked under **epic `dhar174/intelligent_data_detective#119`** with seven sub-issues:

| Phase | Issue | Scope |
|---|---|---|
| A | #112 | Forensic deep-dive (RC1–RC5, no code changes) |
| B | #113 | Instrument-only telemetry run |
| C | #114 | Fix viz pipeline |
| D | #115 | Fix report pipeline |
| E | #116 | Fix `file_writer` |
| F | #117 | Tighten supervisor FINAL gate |
| G | #118 | Validate against 12-criteria |

Labels: `epic`, `phase-6`, `forensic`, `instrumentation`, `viz-pipeline`, `report-pipeline`, `file-writer`, `supervisor`, `validation`.

### Consequences
- Cross-cutting fixes get filed against the closest sub-issue; epic #119 is the rollup.
- Phase A's RC5 (structured-output State persistence post-W9) is the newest concern and may re-open W9-SR-DROP design space if confirmed.

---

## 2026-04-23 (later) — Note: Wave 5 cutover backup location

`_wave5_backup_20260423-091557/` is the canonical pre-cutover snapshot (SOURCE `IntelligentDataDetective_beta_v5.ipynb` + `_patch_notebook.py` + the most-recent patched notebook). Use this as the rollback target for any Phase 6 notebook regression. Do not delete or compress without superseding-snapshot agreement.
## W14 completion baseline and notebook workflow

### Context
Earlier sessions explored direct notebook edits and several Phase 6 recovery strategies. The final completion path re-established `_patch_notebook.py` as the durable source for notebook behavior changes and committed `IntelligentDataDetective_beta_v5_patched.ipynb` as the runnable W14 notebook.

### Decision
For current IDD v5 work:
- Edit `_patch_notebook.py` for notebook behavior changes.
- Regenerate `IntelligentDataDetective_beta_v5_patched.ipynb`.
- Treat `IntelligentDataDetective_beta_v5_patched.ipynb` as the committed runnable artifact for manual and automated proof runs.
- Preserve the W14 baseline `IDD_run_run_default_id-20260504-1338-b3079aea` unless a future task explicitly changes the completion bar.

### Evidence
- `validate_run.py --latest --log-path notebook_run_log.txt --window 180` scored 12/12.
- `validate_artifact_quality.py --latest` scored 9/9.
- Native structured-output markers were present for initial analysis, data cleaning, analysis, visualization, report orchestration, report sections, report packaging, and file writing.
- Visualization fan-in reached `sent_count=3 received_count=3`.
- Recovery/final-hop/native-failure/path-normalization/traceback markers were zero.
- Final artifacts included canonical root `final_report.html`, `final_report.md`, and `final_report.pdf`; root HTML image paths resolved and the PDF was parseable.

### Consequences
- Historical notes about Run 88 hollow/Potemkin artifacts remain useful as regression context, but they are no longer active blockers.
- Future prompt/report polish must keep both validators green and preserve the no-marker, no-recovery, 3/3 visualization fan-in contract.
- Root docs and agent instructions should reference the patcher-generated W14 notebook workflow rather than stale 27-cell/direct-edit guidance.

---

## 2026-10-12 — ADR: Lossless `.pkl` internal backing snapshots in DataFrameRegistry (PR #157 CP1 Corrective Pass)

### Status
✅ Accepted and verified (CI green 1m53s, 10 regression tests in `TestLosslessDataFramePersistence`).

### Context
During independent code review of PR #157 (`pullrequestreview-5497205553`), a Priority 1 (P1) defect was identified: internal backing snapshots generated by `DataFrameRegistry._safe_backing_filename(df_id)` used `.csv`.
While CSV is human-readable, CSV serialization in pandas is lossy:
1. Integer column names (e.g. `0`) are stringified upon reload to `'0'` (or `'0.1'` on collisions).
2. Timezone-aware datetimes lose their timezone information or coerce to naive strings.
3. MultiIndex columns and rows are flattened to simple ranges.
4. Categorical, boolean nullable (`pd.NA`), and Int64 dtypes degrade to object or float64.

### Decision
1. Change internal backing snapshots in `DataFrameRegistry._safe_backing_filename()` to use `.pkl` extension (`{safe_prefix}_{digest[:16]}.pkl` or `df_{digest[:24]}.pkl`).
2. Enforce trusted storage boundary:
   - Internal `.pkl` snapshots are strictly confined to `self.data_dir` and verified via `_get_owned_backing_path()` path containment checks.
   - External user files (`raw_path`) retain their original caller-specified format (`.csv`, `.parquet`, `.json`) byte-for-byte untouched; they are never rewritten to `.pkl`.

### Consequences
- DataFrames evicted from LRU cache are reloaded with 100% representation fidelity (`pd.testing.assert_frame_equal(check_dtype=True, check_names=True)` passes).
- Downstream tools (such as query views or column indexers) no longer experience type drift on evicted DataFrames.
- Security boundary is maintained: unpickling is restricted strictly to registry-owned files within `self.data_dir`.

---

## 2026-10-13 — ADR: Private storage boundary and filesystem security hardening in DataFrameRegistry (PR #157 CP1 Security Pass)

### Status
✅ Accepted and verified (CI green 2m1s run ID 38334366737, 9 unit tests in `TestPrivateStorageSecurity`, 79 tests in `test_registry.py`).

### Context
During independent code review of PR #157 (`pullrequestreview-5497824478`), a security finding was raised regarding the filesystem boundary for internal `.pkl` snapshots. Using Python pickle serialization requires strict filesystem access controls to guarantee that:
1. Temporary snapshots cannot be inspected or tampered with by other local users on shared multi-user systems.
2. The registry never adopts untrusted directories (e.g. volume roots, world-writable directories, or directories owned by other users) as snapshot storage.
3. Pickle deserialization is strictly confined to registry-owned storage and cannot be redirected to arbitrary paths.

### Decision
1. **Isolated Default Directory**:
   - Replaced default storage directory creation with `tempfile.mkdtemp(prefix="idd_registry_", dir=str(WORKING_DIRECTORY))`.
   - On POSIX, explicitly set directory permissions to `0700` (`drwx------`) to ensure exclusive ownership by the calling process UID.
2. **Snapshot File Permission Hardening**:
   - Staging files (`.tmp_*`) and final `.pkl` snapshots are created with restrictive permissions (`0600`, `-rw-------`) on POSIX before atomic `os.replace`.
3. **Configured `data_dir` Validation**:
   - Rejects filesystem volume roots (e.g., `/` on POSIX, `C:\` on Windows).
   - Rejects non-directory files.
   - On POSIX: rejects world-writable directories (`stat.S_IWOTH`) and directories not owned by the current process UID (`st.st_uid != os.getuid()`).
   - If group-writable, validates directory GID matches process GID.
4. **Trusted Deserialization Boundary**:
   - In `get_dataframe(load_if_not_exists=True)`, reloading `.pkl` / `.pickle` files asserts `path.resolve().is_relative_to(self.data_dir.resolve())` and rejects paths matching the directory root itself.
5. **Windows NTFS DACL Model**:
   - Documented in `DataFrameRegistry.__doc__` that Windows security relies on NTFS DACL inheritance within `%TEMP%` / `%USERPROFILE%`, which defaults to access restricted to the user principal.

### Consequences
- Unpickling is strictly confined to verified, private, registry-owned storage.
- Multi-user isolation is guaranteed on POSIX via 0700/0600 permissions, and on Windows via NTFS DACLs.
- Existing user files (`raw_path`) and published artifacts are never affected.
- 0 lines of notebook churn; 100% backward compatible with canonical API.

---

## 2026-10-13 — ADR: Canonical Tools, Error Handling Hardening, and Signature-Aware Binding (CP2, Issue #158)

### Status
✅ Accepted and verified (`test_error_handling_framework.py` 16/16 pass without `--deselect`, 24 unit tests in `test_canonical_tools.py`, 24 migrated tests in `test_handle_tool_errors.py`, 28/28 diagnostic tests in `test_cp0_diagnostics.py`).

### Context
During the canonical core migration (Roadmap Issue #140, Checkpoint 2 Issue #158), parity audits between `idd_core.py` and the authoritative notebook identified three critical defects/drifts:
1. **Claim 6**: `idd_core.py` returned plain string error messages (`Error: ...`) upon exceptions, whereas the production notebook (Cell 32) returned standardized 4-field dictionaries with sanitized exceptions.
2. **Claim 7**: `@handle_tool_errors` in `idd_core.py` naively checked `args[0]` to extract `df_id`. If a decorated tool did not take `df_id` as its first parameter (or lacked a `df_id` parameter entirely, such as a general text/numeric tool), passing an arbitrary argument (e.g. `"hello"`) triggered an erroneous DataFrame registry lookup. This was the exact root cause of the historical edge-case failure in `test_error_handling_framework.py` (`test_integration_with_different_function_signatures`) requiring `--deselect`.
3. **Claim 9**: Artifact path resolution in `idd_core.py` lacked strict containment checks, delayed parent directory creation, and PR #149's multi-root read validation helper (`is_safe_artifact_path`).

### Decision
1. **Established Canonical Module `idd_tools.py`**:
   - Implemented standardized 4-field error dictionary: `{"status": "error", "operation": str, "reason": str, "action": str}` via `_tool_error()`.
   - Implemented `_tool_failure()`: logs diagnostic tracebacks via `logger.exception()` while presenting sanitized, user-safe guidance ("An unexpected data-processing failure occurred.") for unexpected exceptions. Distinct actionable guidance preserved for known domain exceptions (`FileNotFoundError`, `KeyError`, `ValueError`, `EmptyDataError`, `ParserError`, `DtypeWarning`).
2. **Signature-Aware Argument Binding**:
   - In `@handle_tool_errors`, use `inspect.signature(func).bind(*args, **kwargs)` with `.apply_defaults()`.
   - Reject parameter count/type mismatches immediately with structured errors.
   - Extract `df_id` strictly if present in bound arguments, explicit keyword arguments, or parameter object attributes (`val.df_id`).
   - If a function lacks a `df_id` parameter, completely skip DataFrame existence validation.
3. **Canonical DataFrame Validation**:
   - `validate_dataframe_exists(df_id)` delegates to `get_global_registry().get_dataframe(df_id, load_if_not_exists=True)`, ensuring support for all registered formats (.csv, .parquet, .pkl, .json) across cache evictions.
4. **Secure Path Resolution & PR #149 Protections**:
   - `_resolve_artifact_path()` enforces strict single-root containment within the active artifacts directory (`config.configurable.runtime.artifacts_dir` -> `RUNTIME.artifacts_dir` -> `os.environ["IDD_ARTIFACTS_DIR"]` -> `WORKING_DIRECTORY / "artifacts"`). Rejects path traversal (`../`), absolute escapes, symlink escapes, and root targeting with `ValueError`.
   - Delayed parent directory creation: `mkdir(parents=True, exist_ok=True)` executed only after path containment validation succeeds.
   - Implemented `is_safe_artifact_path()` for multi-root read verification across `_allowed_roots` for downstream report/packager nodes without weakening single-root write containment.
5. **Transitional Façade**:
   - Updated `idd_core.py` to re-export all canonical symbols with 100% symbol identity.

### Consequences
- `test_error_handling_framework.py` passes 16/16 without `--deselect`!
- Claims 6, 7, and 9 report `OBSERVED INVARIANT`.
- All tool exceptions are sanitized against leaking internal system paths or tracebacks.
- Single-root write containment protects against arbitrary filesystem traversal, while multi-root read validation safely supports document publishing.
- 0 notebook lines modified across 99 cells; state graph topology verified across 15 nodes and 33 edges.

---

## 2026-10-14 — ADR: Canonical State, Reducers, 15-Node LangGraph Assembly, and Review Resolutions (CP3, Issue #160, PR #161)

### Status
✅ Accepted and verified (Remote Hosted CI run `#38647374243` 100% green; 650 unit/integration tests pass, 54 canonical state graph tests pass, 49 diagnostic tests pass, all 8 review comments resolved on PR #161).

### Context
Under Roadmap Issue #140 (Checkpoint 3, Issue #160), the goal was extracting the state schema, reducers, agent models, and 15-node StateGraph assembly from the 99-cell notebook into canonical root modules (`idd_state.py`, `idd_graph.py`), while re-exporting through `idd_core.py`.
Subsequent code reviews from Codex and Copilot on PR #161 identified several architectural and resilience findings:
1. Codex #1 & #2: `assign_viz_workers` and `dispatch_sections` constructed `Send` objects with minimal task/section parameters, starving downstream workers of essential parent context (`cleaning_metadata`, `analysis_insights`, `available_df_ids`, `_config`). Also, section items required dictionary normalization.
2. Copilot #1: When `HAS_LANGCHAIN = False`, `idd_core.py` lacked fallback stubs for 7 LangChain message symbols and `MyChatOpenai`, causing `AttributeError` on `from idd_core import *`.
3. Copilot #2 & #3: Suggested rewiring `viz_evaluator` directly to `report_orchestrator`/`analyst` and deleting the `report_orchestrator -> report_join` edge.
4. Copilot #4: `AGENT_MEMBERS` in `idd_graph.py` listed only 6 agents instead of all 10 worker agents.
5. Copilot #5: `Router.next` did not type `EMERGENCY_MSG`, and `"END"` was unhandled in `route_from_supervisor()`.
6. Copilot #6: Diagnostic Claim 11 calculated `cond_edge_count` but omitted it from the condition check in `reproduce_drift.py`.

### Decision
1. **Canonical State Schema & Reducer Architecture (`idd_state.py`)**:
   - Implemented authoritative 70-field `State(TypedDict, total=False)` matching notebook Cell 22 (39 fields with reducers, 31 plain fields).
   - Enforced **BR-7 Invariant**: `State.messages` declared directly as `Annotated[list[AnyMessage], add_messages]`.
   - Enforced **W9-SR-DROP Invariant**: `structured_response` omitted from supervisor `State`.
   - Preserved `AgentId` at exact 14 values matching Cell 7.
2. **Canonical Dual-Mode Graph Assembly (`idd_graph.py`)**:
   - `build_graph()` constructs the authoritative 15-node, 33-edge (21 conditional) StateGraph matching `validate_graph.py`.
   - Dual compilation modes: `structural_only=True` (placeholder nodes raise `RuntimeError` if executed); `structural_only=False` (validates all 14 node overrides, failing closed with `ValueError` on missing/unknown nodes).
   - Defined `SupervisorRouteTarget` (typed with 'END', 'FINISH', 'EMERGENCY_MSG') for `Router.next`; normalized `nxt == 'END'` to `'FINISH'` in `route_from_supervisor()`.
   - Expanded `AGENT_MEMBERS` to 10 worker agents; retained `AGENT_OPTIONS` with 6 direct delegation targets.
3. **Fan-Out State Context Propagation & Normalization**:
   - `assign_viz_workers` injects `cleaning_metadata`, `analysis_insights`, `available_df_ids`, `_config`, `messages`, and `final_turn_msgs_list` alongside `task` and `spec`.
   - `dispatch_sections` normalizes sections to dicts and injects `cleaning_metadata`, `analysis_insights`, `viz_results`, `available_df_ids`, `_config`, and `user_request`.
4. **Topology Invariant Defense**:
   - Firmly rejected rewiring suggestions: defended `viz_evaluator -> supervisor` under Cell 60 patch `W2-BR6` and defended `report_orchestrator -> report_join` as a required barrier synchronization edge.
5. **Façade Resilience (`idd_core.py`)**:
   - Added complete fallback stubs for `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`, `BaseMessage`, `AnyMessage`, `add_messages`, and `MyChatOpenai` when `HAS_LANGCHAIN = False`.
6. **Diagnostic Dynamic Verification**:
   - Enforced `cond_edge_count != 21` check in `probe_claim_11()` and added regression simulation test in `test_cp0_diagnostics.py` (49/49 pass).

### Consequences
- PR #161 has all 8 review comments replied to and resolved, with GitHub Actions CI run `#38647374243` 100% green.
- Claim 11 verified as `OBSERVED INVARIANT`.
- 15-node, 33-edge graph topology preserved with 0 churn on 99-cell notebook.
- Clean separation of concerns between `AgentId` (agent membership) and `SupervisorRouteTarget` (routing targets).
- Subworkers in fan-out executions receive full context, eliminating state-starvation crashes and premature aborts.

---

## 2026-10-15 — ADR: Canonical Notebook Integration, Anti-Drift Guard, and Control-Plane Alignment (CP4, Issue #162, PR #163)

### Status
✅ Accepted and verified (`test_core_parity_guard.py` 22/22 pass, `test_patcher_integrity.py` 108 pass / 10 skipped, `validate_notebook_integrity.py` passes with 99 cells, `validate_graph.py` passes with 15 nodes / 33 edges / 70 fields).

### Context
Following the completion and review of Checkpoints 1–3, the canonical Python modules (`idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, `idd_graph.py`, `idd_core.py`) were verified in offline unit and integration tests, but the production runnable notebook `IntelligentDataDetective_beta_v5_patched.ipynb` still carried internal inline definitions of models, registry, state reducers, tools, and graph assembly.
To eliminate this split-brain (Roadmap Issue #140, Companion Issues #144 and #145, Implementation Issue #162), Checkpoint 4 (CP4) requires:
1. Rewiring the notebook via `_patch_notebook.py` to import and execute the canonical modules directly.
2. Eliminating duplicate inline redefinitions.
3. Establishing an automated anti-drift regression guard (`test_core_parity_guard.py`) asserting static AST invariants, runtime symbol identity (`is`), behavioral parity, and negative control failures.
4. Aligning control-plane instructions, agent personas, and documentation across the repository.

### Decision
1. **Canonical Module Execution in Notebook (`_patch_notebook.py`)**:
   - Updated patch sentinels:
     - **Cell 4**: `# PATCH: CP4-BOOTSTRAP` (`idd_bootstrap.bootstrap_environment()`).
     - **Cell 16**: `# PATCH: CP4-CANONICAL-MODELS` (`idd_models.Plan`, `BaseNoExtrasModel`, `idd_state.AnalysisConfig`).
     - **Cell 19**: `# PATCH: CP4-CANONICAL-REGISTRY` (`idd_registry.DataFrameRegistry`, `global_df_registry = get_global_registry()`, `get_global_df_registry`).
     - **Cell 22**: `# PATCH: CP4-CANONICAL-STATE` (`idd_state.State`, `VizWorkerState`, all 15 reducers).
     - **Cell 32**: `# PATCH: CP4-CANONICAL-TOOLS` (`idd_tools.handle_tool_errors`, `validate_dataframe_exists`, `_resolve_artifact_path`).
     - **Cell 60**: `# PATCH: CP4-CANONICAL-GRAPH` (`idd_graph.build_graph(..., structural_only=False, node_cache_policies=CachePolicy(ttl=120))`).
   - Regenerated `IntelligentDataDetective_beta_v5_patched.ipynb` maintaining strictly 99 cells.
2. **Canonical Module Capabilities**:
   - `idd_graph.py`: Added support for `node_cache_policies: Optional[Union[Any, Dict[str, Any]]] = None`, preserving `CachePolicy(ttl=120)` on `builder.add_node()`. Re-exported in `idd_core.py`.
   - `idd_graph.py`: Supported duck-typing in `route_to_writer` for mock/test objects while enforcing schema validation on real instances.
   - `idd_state.py`: Exported `default_an_config = AnalysisConfig(...)` and re-exported in `idd_core.py`.
3. **Integrity Manifest & Bootstrap**:
   - Created `idd_manifest.json` recording SHA-256 digests of all 6 canonical modules. Added `!idd_manifest.json` exception in `.gitignore`.
   - Created `idd_bootstrap.py` for runtime environment configuration and provenance/digest verification.
4. **Anti-Drift Guard Suite (`tests/unit/test_core_parity_guard.py`)**:
   - Implemented 22 comprehensive tests covering static AST sweeps across all 99 cells, runtime object identity assertions (`is`), behavioral validation (Codex Findings 1 & 2), and negative controls (tampered digest, missing module, redefinition detection).
5. **Patcher Test Modernization (`tests/unit/test_patcher_integrity.py`)**:
   - Updated patcher integrity tests to accommodate canonical imports and skip redundant inline AST compilation when notebook uses canonical modules.
6. **Control-Plane Documentation Alignment**:
   - Updated `AGENTS.md`, `GEMINI.md`, `.agents/rules/code-architecture.md`, `.agents/rules/validation-and-gates.md`, `.agents/agents/*`, `.github/copilot-instructions.md`, `.github/instructions/backend.instructions.md`, `.github/instructions/notebook.instructions.md`.
   - Removed stale 15/16 error test exceptions; all 16 tests in `test_error_handling_framework.py` are mandatory and strictly blocking.

### Consequences
- The architectural split-brain is fully resolved: the notebook executes canonical modules directly.
- The 99-cell notebook structure is strictly preserved.
- Future drifts are prevented by `test_core_parity_guard.py` running in CI.
- All 16 error handling tests and all anti-drift tests must pass in CI without exclusions.
- Ready for Stage 4 full test gate execution and Stage 5 knowledge closeout.

