<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Active Context

## Current objective
Merge and certify Draft PR #175 ("fix(report): initialize Cell 57 figure metadata before instruction"), which successfully resolves the Cell 57 `UnboundLocalError` discovered during IDD CP5 Two-Run Live Proof Campaign Attempt 1. Complete Antigravity architectural review has passed with APPROVED verdict. Awaiting owner merge to main and subsequent owner re-authorization for restarting the CP5 live proof campaign.

## Why it matters now
The pre-existing instruction-formatting ordering defect in Cell 57 (`section_worker`), where `default_instruction` referenced `expected_viz_str` in an f-string before line 2723 assigned it, caused an `UnboundLocalError` at Cell 75 during CP5 Attempt 1 (`retail_orders`). Patch CP5-SW-FIGURE-ORDER in `_patch_notebook.py` cleanly relocates the figure-metadata initialization ahead of instruction formatting within the active W13J-SW transformation. 225/225 tests across 6 suites pass locally, 8/8 in `test_section_worker.py` pass both locally and in a real Jupyter kernel, strictly 99 cells are preserved, and regeneration idempotency is 100% verified.

## Current status
- Draft PR #175 Architectural Review:
  - Branch: `codex/cp5-cell57-expected-viz`
  - Head SHA: `f9bda27d252c54ce49d7814625b1325502ab76df`
  - Base SHA: `5d75b94ac2eba61737e05b588232c72f4f4a389a` on `main`
  - Review Verdict: **APPROVED FOR MERGE**
  - Defect Fixed: `UnboundLocalError: local variable 'expected_viz_str' referenced before assignment` in `section_worker` (Cell 57). Lines 2716-2719 now initialize/populate `expected_viz_str` ahead of `default_instruction` formatting at line 2721.
  - Test Evidence: 225/225 tests pass locally across 6 suites (8/8 `test_section_worker.py`, 38/38 parity guard, 16/16 error handling, 76/76 runner, 54/54 diagnostics, 33/33 notebook integrity). 1,003 broad tests pass (0 failures), 47 real-kernel tests pass.
  - Invariants: strictly 99 cells intact; only Cell 57 source differs across entire notebook; graph topology (15 nodes/33 edges/70 State fields) preserved; regeneration byte-identical (SHA-256 `7844ac35e5edd5567fa68d47315b55e87e838692c6064dc43cecfc491e8986ec`).
- CP5 Live Proof Campaign Status:
  - Attempt 1 failure evidence safely preserved at `C:\Users\darf3\Documents\cp5_attempts\attempt_live_retail_orders_20261021_run1` (untouched).
  - Campaign remains halted with $0.00 spend until owner merges PR #175 and provides explicit re-authorization.

## Relevant files
- `_patch_notebook.py` (Cell 57 active W13J-SW sentinel block updated)
- `IntelligentDataDetective_beta_v5_patched.ipynb` (Cell 57 regenerated)
- `tests/unit/test_section_worker.py` (8 new regression tests)
- `idd_proof.py`
- `tools/proof_runner.py`
- `idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, `idd_graph.py`, `idd_core.py`, `idd_bootstrap.py`
- `validate_run.py`
- `validate_artifact_quality.py`
- `tests/unit/test_core_parity_guard.py`
- `tests/unit/test_proof_runner.py`

## Next recommended steps
1. Owner merges Draft PR #175 (`f9bda27d252c54ce49d7814625b1325502ab76df`) into `main`.
2. Sync `main` in the primary local environment and verify local notebook compilation.
3. Owner provides explicit re-authorization for the CP5 live proof campaign retry.
4. Execute Attempt 1 retry on `retail_orders` scenario via `idd_proof.py` and validate against the 12/12 + 9/9 twin production gates.
<!-- repo-agent-bootstrap:managed:end -->

<!-- session-curated:start -->
## 2026-09-29 — Historical Failed Live Run (not completion proof)

### Live Run Summary
- **Run ID**: `run_default_id-20260929-0210-367d732a` (877s / 14.6 min wall time)
- **Evidence**: [named log](../docs/evidence/runs/2026-09-29-run_default_id-20260929-0210-367d732a/notebook_run_log.txt) and [provenance/validator summary](../docs/evidence/runs/2026-09-29-run_default_id-20260929-0210-367d732a/provenance.json). Run/date/environment association is reported context, not encoded in the log. Exact execution source SHA/notebook hash and original validator output were not captured; this predates the reviewed PR head.
- **Runtime Environment**: Clean Python 3.12.2 isolated stack (`numpy 1.26.4`, `pandas 2.2.1`, `scipy 1.12.0`, `matplotlib 3.8.3`). Zero package mutation via `# PATCH: P2-SKIP-INSTALLS` (`IDD_SKIP_NOTEBOOK_INSTALLS=1`).
- **Initialization Fix**: Resolved `UnboundLocalError` on `path` in `DataFrameRegistry.register_dataframe` via `# PATCH: P3-DF-REG-PATH`.
- **`validate_run.py` Score**: **5 / 12**
  - **PASS**: Criterion 1 (FINAL marker present), Criterion 3 (0 recoveries / 0 W2-BA-finalhop / 0 W4 negatives), Criterion 4 (0 Tracebacks), Criterion 5 (Native Pydantic structured outputs), Criterion 11 (0 stub marker files — clean!).
  - **FAIL**: Criterion 2 (`report=None`), Criteria 6–8, 10, 12 (report stage incomplete), Criterion 9 (2 unique viz slugs vs 3 required).

### Critical Forensic Discovery: Root Cause of RC2 (#115)
- `dispatch_sections` in Cell 57 constructs `Send("report_section_worker", {"section": s.model_dump(...)})` without forwarding `cleaning_metadata` (and companion parent state: `analysis_insights`, `viz_results`, `available_df_ids`).
- `section_worker` guards on `cm = state.get("cleaning_metadata")`. When `cm is None`, it immediately returns `Command(goto="data_cleaner")`.
- Every section worker aborted to `data_cleaner` without writing sections, resulting in `written_sections_count=0` at `report_join`.
- `data_cleaner` skipped ("Data cleaning already complete"), and supervisor looped until the 15-turn limit terminated the run with `report=None`.
- **Deferred action (#115/#119)**: Fix report dispatch in a separate task. PR #153 does not certify end-to-end reporting or relax the 12/12 + 9/9 production gate.

### PR #153 bounded contracts and handoff
- `run_notebook_live.py` defaults to the patched notebook; deliberate `IDD_NOTEBOOK` overrides are preserved.
- `--preflight` is a scientific-only diagnostic: six imports plus install-flag inheritance in the selected Jupyter kernel. It does not verify all mandatory notebook imports, API credentials, graph execution, or reporting.
- `intelligentdatadetective_beta_v5.py` is a **textual notebook export**, not a supported importable/runnable module. It retains notebook syntax and a late future import. Tests compile selected registry/install/supervisor snippets only; whole-runtime consolidation is deferred to #140.
- Touched registry cases are tested against both snippets; existing-ID omitted-path behavior follows the notebook, and new-ID retained paths match normalized I/O paths. Broader ownership/deduplication and existing-ID persistence policy remain in #141/#140.
- The root log is restored from pre-PR commit `f4a749e`; it is historical evidence, not a final-candidate run.
- Proposed memory rules were removed from `.agents/rules/` and preserved as inactive [memory](../docs/evidence/proposals/issue-152-memory.txt) and [workflow](../docs/evidence/proposals/issue-152-memory-workflow.txt) proposals for #152. Posting the handoff to #152 was denied (HTTP 403); runtime discovery is not claimed.
- Real-kernel smoke command (no paid calls): `IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight python -m pytest tests/smoke/test_scientific_kernel.py -q -s`. Provision with `python -m pip install nbclient==0.11.0 nbformat==5.11.1 jupyter-client==8.10.0 ipykernel==7.4.0` and `python -m ipykernel install --user --name idd-preflight`. CI runs this separately from the optional-Jupyter-free unit suite.

## Session context — pipeline debugging (mid-session checkpoint)

### Current objective
Achieve a **production-quality clean run** of the full IDD v5 pipeline — zero recovery hops, all artifacts emitted, suitable for portfolio inclusion.

### High-water mark
- **Run 76 in flight.** `visualization` node passed cleanly. `viz_evaluator` has been silent ≥16 min — actively investigating whether it is looping or stuck on an LLM call.
- Earlier baselines: Run 73 finished `viz=True/report=True` but every agent hit `recursion_limit=160` (recovery path); Run 74 regressed on BR-7 (managed channels); Run 75 regressed on BR-8 (`InvalidUpdateError` on `structured_response` at cell 81).

### Open work
- **Wave 4 patches ready to apply** (already prepared as `_patch_notebook.py` sentinels):
  - `W2-EMERGENCY` — give `EMERGENCY_MSG` an outgoing edge (currently a dead-end).
  - `W2-BF2` — analyst fix.
  - `W2-BF6` — agent factory fix.
  - `W2-REC6` — fast-fail on unknown tool name.
- **Run 77** will use `python run_notebook_live.py --resume` (saves 12–15 min by skipping checkpointed nodes) **only if State schema is unchanged**; any new field/reducer invalidates `checkpoints.sqlite`.

### Active design decisions (do not regress)
- `recursion_limit = 160` (inner agents) / `400` (outer graph) — explicitly raised by user.
- `State.messages` is declared **directly** as `Annotated[list[AnyMessage], add_messages]`, **not** inherited from `langchain.agents.AgentState` / `langgraph.prebuilt.chat_agent_executor.AgentState` (BR-7 — those bases drag in the `remaining_steps` managed channel that LangGraph rejects in InputSchema).
- `structured_response: Annotated[Optional[BaseNoExtrasModel], _sr_reducer]` — the `_sr_reducer` is `lambda left, right: left if right is None and left is not None else right` (prefer-non-None last write). Required because main node + recovery shims both write concurrently and the default LastValue reducer raised `InvalidUpdateError` (BR-8).
<!-- session-curated:end -->

<!-- session-curated:2026-04-23:start -->
## 2026-04-23 — W9-SR-DROP unblocks supervisor graph; focus shifts to W10-PDF-DIAG

### Current focus
**W10-PDF-DIAG** — PDF artifact emission is the sole remaining cleanliness blocker after W9-SR-DROP. The supervisor graph now reaches FINAL with `viz=True report=True` cleanly (Run 87, ~21 min wall, 7/8 criteria GREEN).

### What changed (cell 22 of patched notebook)
- ✅ **W7-SR-ALIGN** — User `State.structured_response` annotation aligned exactly to langchain `AgentState`'s shape: `NotRequired[Annotated[Optional[Any], OmitFromSchema(input=True, output=False)]]`. Eliminated `_resolve_schema` set-iteration nondeterminism that intermittently failed cell 48 (`create_agent` calls) by colliding `_sr_reducer` BinaryOperatorAggregate with AgentState's LastValue channel for the same field.
- 🟡 **W8-VW-NOSR** (cell 57) — Removed `"structured_response": sr` from viz_worker wrapper return dict. Necessary but insufficient on its own (recovery shims also wrote sr).
- ✅ **W9-SR-DROP** (cell 22, breakthrough) — Completely removed `structured_response` from supervisor State. Each `create_agent` subgraph still owns its own sr via internal `AgentState[ResponseT]`; wrapper code reads it from `result["structured_response"]` (Python-dict from agent.invoke, NOT a langgraph supervisor channel). Static grep confirmed zero supervisor-state reads of `state["structured_response"]`. Without a supervisor channel, parallel writers (viz fan-out, recovery shims, agent subgraph propagation) silently drop instead of colliding at LastValue's "one write per step" contract.

### Run 87 result (post-W9)
7/8 GREEN. FINAL `viz=True report=True`. 0 recoveries, 0 W2-BA-finalhop, 0 tracebacks, 0 W4 negatives. All 11 graph stages traversed (initial_analysis → data_cleaner → analyst → visualization + viz_worker fan-out + viz_join → viz_evaluator → report_orchestrator + section_worker + report_join → report_packager → file_writer → FINAL). Native structured Pydantic outputs throughout (`CleaningMetadata`, `AnalysisInsights`, `ReportResults`, etc.). Wall ~21 min.

### Remaining gap — W10-PDF-DIAG
PDF artifact missing from `IDD_results/IDD_run_run_default_id-20260423-0951-6c29a6f7/`. HTML (244 KB) + MD written by file_writer LLM agent tools; PDF the lone gap. `W6-FW-PDF-FORCE` block IS present in cell 57 `file_writer_node` but **never executed at runtime** (zero `[W6-FW-PDF-FORCE]` log lines). `STAGE file_writer DONE` marker never logged — silent failure between `file_writer_agent.invoke()` return and the W6 try-block. Suspect `result["structured_response"]` KeyError or `assert isinstance(file_results, ListOfFiles)` failing because W9-SR-DROP causes the agent-subgraph→supervisor-state merge to drop sr.

### Recommended W10 patches
1. **W10-FW-RESULT-GUARD** — defensively handle missing key: `file_results = result.get("structured_response") or result.get("output") or result`; rebuild from tool calls if not `ListOfFiles`.
2. **W10b-FW-LOG** — add `print('STAGE file_writer post-invoke ok')` immediately after invoke to localize the silence.
3. **W10-PDF-DIRECT (alt)** — separate post-graph PDF-writer node that reads HTML from disk and renders via xhtml2pdf, independent of file_writer wrapper.

### Outstanding (non-blocking)
- `IDD_results/executed_20260423_055108.ipynb` is 0 bytes (runner `nbformat.write` flush bug → W11-NB-FLUSH).
- `outputs/` clutter (~50 redundant tool-marker files) → optional W12-OUTPUT-DEDUPE.
- `_diag_cmp.py`, `_diag2.py` left in repo root from earlier diag work.

### Hard constraints honored
- Did NOT modify `IntelligentDataDetective_beta_v5.ipynb` or `_patch_notebook.py` this session segment.
<!-- session-curated:2026-04-23:end -->

<!-- session-curated:2026-04-23-phase6:start -->
## 2026-04-23 (later) — Phase 6: structural CLEAN ✅, deliverable HOLLOW ❌ — pivot to content quality

### Current focus
**Phase 6 — content-quality / Potemkin-pipeline forensics.** The Wave 4/5 structural-cleanliness goal is *done* (Run 87 7/8, Run 88 8/8). The new — and now sole — goal is **producing a deliverable that a human would actually read**. The "8/8 CLEAN" success criterion is **OBSOLETE** and has been replaced by the new 12-criteria content bar (see decisionLog).

### Run 88 result — structurally CLEAN, semantically HOLLOW (historical regression)
8/8 GREEN structural pass. But the artifacts were hollow:
- **Report text:** literally 356 chars, verbatim placeholder: `"Final report placeholder created. HTML, Markdown, and PDF artifacts prepared per instruction."`
- **Sections in report:** 0 (`Final report assembled from 0 sections`).
- **Visualizations:** 1 PNG (analyst recommended 3); embedded 5× in HTML with cosmetic hash variations.
- **PDF:** 1983 bytes — title page only.
- **Stub files in reports dir:** 25+ marker files (`*_ack`, `*_commit`, `*_ready`, `*_stub`, `*_trigger`, …) — `file_writer` LLM compensated for empty `written_sections` by spamming marker tools to "look busy".
- **Analyst output was rich and correct** (full `AnalysisInsights` with 3 distinct `VizSpec`s, correlation_insights, anomaly_insights). **The content was lost between analyst and report.**

### W7–W10 patches confirmed working (DO NOT regress)
- **W7-SR-ALIGN** (cell 22): user `State.structured_response: NotRequired[Annotated[Optional[Any], OmitFromSchema(input=True, output=False)]]`.
- **W8-VW-NOSR** (cell 57): viz_worker wrapper return no longer carries `structured_response`.
- **W9-SR-DROP** (cell 22): `structured_response` removed entirely from supervisor `State`.
- **W10-PDF-POST** (cell 83): post-graph deterministic xhtml2pdf converter; produces a valid PDF independent of `file_writer`. (Note: with current hollow HTML, the PDF is also hollow → still ≪ 30 KB target.)

### Historical Wave 5 mode — superseded by W14
This section records a temporary direct-edit experiment. It is superseded by the W14 workflow: edit `_patch_notebook.py`, regenerate `IntelligentDataDetective_beta_v5_patched.ipynb`, and validate the patched notebook.

### MyChatOpenai is LEGACY (per user, 2026-04-23)
Going forward, **use `ChatOpenAI` directly**, not `MyChatOpenai`. ⚠️ This **conflicts with `.github/agents/notebook-specialist.agent.md`**, which (in its managed block) still mandates `MyChatOpenai`. We are **not** editing the agent doc (managed), but the divergence is documented here and in `decisionLog.md`. New cells should use `ChatOpenAI`; existing `MyChatOpenai` call sites can be migrated opportunistically.

### Phase 6 plan (locked)
Source: `C:\Users\darf3\.copilot\session-state\6b170d0a-bbd7-4870-ba34-e2c0cb0d4241\plan.md`.

| Phase | Goal | GH Issue |
|---|---|---|
| **A** | Forensic deep-dive (no code changes) — RC1 viz fan-out drop, RC2 `report_section_worker` writes 0 sections, RC3 file_writer placeholder + stub spam, RC4 supervisor accepts hollow completion, **RC5** structured Pydantic outputs may not actually be persisted to State after W9-SR-DROP | #112 |
| B | Instrument-only telemetry run | #113 |
| C | Fix viz pipeline | #114 |
| D | Fix report pipeline (`Section.body` min-length validator, `written_sections` reducer audit) | #115 |
| E | Fix `file_writer` (rewrite prompt, restrict tool list to remove stub-marker tools, cap tool calls) | #116 |
| F | Tighten supervisor FINAL gate with content-validation preconditions | #117 |
| G | Validation against 12-criteria; expect 5–8 runs to converge | #118 |

**GitHub epic:** `dhar174/intelligent_data_detective#119`. Labels: `epic`, `phase-6`, `forensic`, `instrumentation`, `viz-pipeline`, `report-pipeline`, `file-writer`, `supervisor`, `validation`.

### Forensic agents in flight
- `forensic-pipeline` (explore) — completed; **results stranded** in cleared agent context. Re-harvest required.
- `forensic-sr-persistence` (explore) — completed; **results stranded**. Re-harvest required.
Both produced full forensic reports but the agent contexts cleared before harvest. Phase A continuation must either re-launch them or fall back to direct inspection.

### Hard constraints (Phase 6)
- Do **not** revert W7/W8/W9/W10 patches.
- Do **not** add `structured_response` back to supervisor `State`.
- Do **not** edit managed `.github/agents/*` blocks (e.g., notebook-specialist's MyChatOpenai rule).
- Wave 5 source-of-truth = the notebook itself; do not re-route fixes through `_patch_notebook.py`.

### Relevant files (Phase 6)
- `IntelligentDataDetective_beta_v5.ipynb` (Wave 5 direct-edit target; cells 22, 57, 83 are hot)
- `_wave5_backup_20260423-091557/` (cutover backup)
- `IDD_results/IDD_run_*-20260423-*/` (Run 87, Run 88 artifacts)
- Phase 6 plan: `C:\Users\darf3\.copilot\session-state\6b170d0a-bbd7-4870-ba34-e2c0cb0d4241\plan.md`

### Next recommended steps
1. **Phase A**: re-run forensic exploration of viz fan-out (RC1), report_section_worker writes (RC2), file_writer prompt + tool list (RC3), supervisor FINAL gate (RC4), and structured-output State persistence post-W9 (RC5).
2. **Phase B**: instrument-only telemetry run; capture per-stage counts of `written_sections`, `viz_specs`, `viz_results`.
3. Defer C–G until Phase A confirms each RC and Phase B gives a measured baseline.
<!-- session-curated:2026-04-23-phase6:end -->

<!-- session-curated:2026-04-29-phase6-patches:start -->
## 2026-04-29 — Phase 6 patch chain applied; awaiting keyed full run

### Current focus
The source notebook `IntelligentDataDetective_beta_v5.ipynb` now contains the Phase 6 patch chain for the Run89 content failures. Static/unit validation is green, but the production result is not yet remeasured because this shell has no `OPENAI_API_KEY` and no `.env` / `.env.local`.

### What changed
- `viz_worker` no longer returns the result of `dict.update()`; it builds and returns a real update dict.
- `save_viz_for_state()` now returns only this worker's new visualization dict snapshots and paths, matching `operator.add` reducers for `viz_results` and `viz_paths`.
- `section_worker` now handles dict section payloads, defines expected visualization context before prompt construction, uses `msg.content_blocks`, rejects short/invalid content, and returns explicit non-completion state instead of silent `{}`.
- `Section.content` requires substantive prose (`min_length=100`).
- `report_packager_node` blocks packaging when section count or total section characters are insufficient.
- The direct `report_orchestrator -> report_join` edge is removed/commented so section worker branches contribute `written_sections` first.
- `route_to_writer()` now gates final routing on content readiness: section count/chars, distinct visualization IDs, and prior file-write state.
- `file_writer_node` fixes metadata content extraction, corrects final-output logic, and blocks final writes for insufficient report content.

### Validation completed
- Notebook JSON parses; cell count remains 98.
- All 41 code cells compile when shell magics are stripped.
- `python validate_graph.py --notebook IntelligentDataDetective_beta_v5.ipynb` compiles: 15 nodes, no unreachable/dead-end nodes.
- `python -m pytest test_validate_run.py -q` passes 8/8.
- `python -m pytest test_intelligent_data_detective.py -q` passes 22/22.

### Blocked validation
Fresh full notebook execution and the post-patch `validate_run.py` score are blocked until API configuration is available. Last measured score remains the old Run89 baseline: 7/12.

### Next recommended steps
1. Provide `OPENAI_API_KEY` in the shell or a local dotenv file, then run a fresh notebook execution without resuming stale checkpoints.
2. Score the newest run with `python validate_run.py --latest --log-path notebook_run_log.txt --window 45`.
3. If C9 fails, inspect `STATE viz_join` and distinct `viz_results` IDs/paths.
4. If C7/C8/C12 fail, inspect `STATE report_section_worker.*` and `STATE report_packager.pre_draft`.
5. If C10 fails after content improves, inspect deterministic PDF generation.
<!-- session-curated:2026-04-29-phase6-patches:end -->

<!-- session-curated:2026-04-30-phase6-complete:start -->
## 2026-04-30 — Phase 6 production bar reached (12/12)

### Current status
The source notebook `IntelligentDataDetective_beta_v5.ipynb` now completes the Phase 6 production validation bar. Latest measured run:

```bash
python validate_run.py --latest --log-path notebook_run_log.txt --window 45
# SCORE: 12 / 12 (PASS — production bar reached)
```

Latest passing run directory:
- `IDD_results\IDD_run_run_default_id-20260430-0637-f03d7315`

### Evidence from latest run
- FINAL marker present in `notebook_run_log.txt` with `viz=True report=True`.
- 3 distinct visualization PNGs, each with unique slug and image hash.
- 5 report sections in the final HTML.
- Final report HTML text length: 17,346 chars.
- Final PDF size: 121,220 bytes.
- No stub/marker files in the reports directory.
- Report references analyst findings, including correlation and anomaly/missing-value language.
- No recoveries, W2/W4 negative markers, tracebacks, or structured-output fallback markers in the validation window.

### Most recent patch
The final-state inspection cell now emits a deterministic completion line after verifying:
- `viz_results` contains at least 3 visualizations.
- `report_results` is a `ReportResults` instance.
- Markdown, HTML, and PDF report paths exist and are non-empty.

It prints and appends this line to both the repository-level `notebook_run_log.txt` and the temp working-directory log so `validate_run.py` can inspect the same run evidence.

### Next recommended steps
1. Treat the current notebook as the Phase 6 completion baseline.
2. Preserve `validate_run.py` criteria; do not weaken the 12/12 production bar.
3. If future notebook patches touch graph routing, reducers, report packaging, or finalization, rerun the full keyed notebook execution and `validate_run.py --latest --log-path notebook_run_log.txt --window 45`.
<!-- session-curated:2026-04-30-phase6-complete:end -->

<!-- session-curated:2026-04-30-artifact-quality:start -->
## 2026-04-30 — Artifact quality bar added and reached

### Current status
The previous 12/12 production run proved pipeline completion but exposed artifact-quality gaps: the PDF was pseudo-PDF bytes, HTML/Markdown did not embed images, repeated figure-reference prose appeared across sections, and deterministic visualizations were ID-dominated.

The notebook now passes both gates:

```bash
python validate_run.py --latest --log-path notebook_run_log.txt --window 45
# SCORE: 12 / 12

python validate_artifact_quality.py --latest
# Historical score before later validator expansion; current W14 bar is SCORE: 9 / 9
```

Latest stronger baseline:
- `IDD_results\IDD_run_run_default_id-20260430-1920-eaf0025a`

### Evidence from latest run
- PDF is standards-compliant enough for both pypdf and PyMuPDF; pypdf reads 5 pages and 7,637 extracted text chars.
- HTML contains 3 `<img>` tags and all image paths resolve from the report directory.
- Markdown contains 3 image embeds and all image paths resolve from the report directory.
- Visualization set is no longer ID-dominated: distribution of `value`, relationship between `value` and `score`, and average `score` by `category`.
- Duplicate long-paragraph audit passes: 18 paragraphs, 15 unique, 3 duplicates.

### Files added/changed
- `validate_artifact_quality.py` — new artifact-quality validator for parseable PDF, embedded image paths, repetition, and non-trivial chart selection.
- `IntelligentDataDetective_beta_v5.ipynb` — report packager now writes a real ReportLab PDF, embeds images in HTML/Markdown, resolves report-relative image paths, de-duplicates repeated long blocks, and deterministic analyst fallback avoids identifier-dominated charts.

### Next recommended steps
1. Preserve both gates for future notebook changes: `validate_run.py` 12/12 and `validate_artifact_quality.py` 5/5.
2. If report generation changes, inspect artifacts directly in addition to checking validator scores.
<!-- session-curated:2026-04-30-artifact-quality:end -->

<!-- session-curated:2026-05-01-report-agent-loops:start -->
## 2026-05-01 — Agent-owned report path still loops on tool/file termination

### Current focus
The notebook can produce valid-looking artifacts through guarded/recovery paths, but the no-shortcuts goal remains open. Latest evidence from `IDD_results\IDD_run_run_default_id-20260501-0240-070cc4bd\artifacts\run_default_id-20260501-0240-070cc4bd` shows report agents creating many small `.txt` marker/status files while trying to finish.

### What the artifact review found
- Tiny `.txt` files such as `status.txt`, `final_note.txt`, `final_ready_note.txt`, `stop_file.txt`, `final_complete.txt`, `logs\respond_trigger.txt`, and `tools\respond_request.txt` were written by report agents, not by expected final packaging.
- Contents are termination boilerplate such as "End of response", "Final answer posted", and "Submitting final report artifacts".
- Notebook output shows repeated `write_file` tool calls during `report_section_agent.invoke` and `report_packager_agent.invoke`.
- The same execution path hit `GraphRecursionError` recovery for report section agents and then reached `report_packager_agent.invoke`, confirming the files are symptoms of nested agent tool/termination loops.
- Some logged paths are malformed (`artifactsun_default_id...`, `logseport_paths.txt`, `logsespond_trigger.txt`), so path normalization for agent-provided filenames also needs inspection.

### Working hypothesis
Report agents have access to broad file tools while also being required to return structured `Section` / `ReportResults` outputs. When the model fails to terminate through the structured response path, it writes status/marker `.txt` files as a proxy for completion and continues looping until recursion recovery.

### LangSmith status
LangSmith is not active in the current runner environment. The Python package is installed (`langsmith==0.6.2`), but the CLI is not on PATH, no LangSmith/LangChain tracing env vars are visible in Process/User/Machine scope, `.env`/`.env.local` are absent, and `run_notebook_live.py` prints `WARN LangSmith tracing environment not visible to runner`.

### Next recommended steps
1. Restrict or remove arbitrary `write_file` access from `report_section_agent`; section workers should return `Section` objects only.
2. Restrict `report_packager_agent` so final artifact rendering is handled by deterministic renderer code after agent-owned content is accepted, not by marker file writes.
3. Add validation that fails on stray marker/status `.txt` files and malformed concatenated artifact paths.
4. Install/expose LangSmith CLI and make `run_notebook_live.py` load tracing variables from the same source as `OPENAI_API_KEY`, without printing secrets.

### Latest W13P failure/fix
The W13P run `IDD_run_run_default_id-20260501-0317-3bc357d6` reached `report_packager_agent.invoke.start` after joining 5 sections (`written_sections_count=5`, `total_chars=14490`) but failed with `NameError: name 'GraphRecursionError' is not defined` in `report_packager_node`. `_patch_notebook.py` now emits a local import alias (`from langgraph.errors import GraphRecursionError as _W13PGraphRecursionError`) before the W13P try/except, and the regenerated patched notebook validates with 99 cells, 0 robust syntax errors, W13P import present, and `test_validate_run.py` 8/8.

### GitHub tracking
- `#124` — Report agents create marker `.txt` artifacts during tool-loop termination.
- `#125` — LangSmith tracing env and CLI are not visible to notebook runner.
<!-- session-curated:2026-05-01-report-agent-loops:end -->

<!-- session-curated:2026-05-04-w13r-trace-fixes:start -->
## 2026-05-04 — W13Q trace findings and W13R patch status

### Current focus
The completion goal remains a fully traced notebook run where each node/agent submits its expected structured output class and produces stakeholder-ready artifacts without marker/status spam. LangSmith tracing is now active in the runner path; the next milestone is proving the patched notebook reaches clean completion without recursion recovery.

### W13Q traced findings
- W13Q successfully scoped report roles enough that outline-stage `write_file`, PDF, HTML, and Markdown rendering tool calls dropped to zero in the trace.
- The run still did not complete: visualization evaluation saw `viz_results_count=0` immediately after `viz_join` logged three unique visualization ids, then repeatedly called `list_visualizations`.
- `list_visualizations` returned zero charts because its state gatherer did not handle `DataVisualization` objects and because df_id queries such as `sample_dirty` were treated as restrictive text filters.
- Report orchestration still looped because the factory prompt used the raw template (`prompt.messages[0].prompt.template`), leaving literals such as `{report_task}` in the live system prompt.
- Outline/section report roles also still had `read_file`; traces showed PNG reads, which polluted context with binary/base64-like content.

### W13R patch status
`_patch_notebook.py` now regenerates `IntelligentDataDetective_beta_v5_patched.ipynb` with:
- rendered role-specific report generator prompts and explicit expected schema/tool contracts;
- no `read_file` for outline/section report roles;
- visualization listing that accepts structured `DataVisualization` objects and treats current-dataset queries as non-restrictive;
- evaluator invoke state that includes `visualization_results`, `viz_paths`, `artifacts_path`, available df ids, and current dataframe identifiers.

Static validation passes: patched notebook has 99 cells, robust syntax validation reports 0 errors, W13R markers are present, the old report prompt anchor is absent, and `python -m pytest test_validate_run.py -q` passes 8/8.

### Next recommended steps
1. Run `IntelligentDataDetective_beta_v5_patched.ipynb` through `run_notebook_live.py` with LangSmith enabled.
2. Confirm `viz_evaluator` sees all generated visualizations and avoids the `list_visualizations` loop.
3. Confirm report orchestrator/section/packager agents return `ReportOutline`, `Section`, and `ReportResults` without recursion recovery.
4. Audit every output file after the run, especially `final_report.html`, Markdown/PDF reports, JSON metadata, PNGs, and any small `.txt`/status files.
<!-- session-curated:2026-05-04-w13r-trace-fixes:end -->

<!-- session-curated:2026-05-04-prompt-parity:start -->
## 2026-05-04 — Secondary prompt-parity requirement

### Current focus
Primary priority remains clean notebook completion: native structured outputs, correct state visibility, no recovery paths, correct supervisor routing, and successful final artifacts. Prompt-template parity is now tracked as a secondary priority after those blockers.

### Requirement
- Static/factory prompts introduced for stability may stay, but they should be revised to resemble the original prompt templates in structure, tone, format, and wording.
- Dynamic context previously inserted into templates must still be directly provided to each agent through rendered messages or invoke state.
- Required context includes dataframe ids, cleaned dataset description, data sample, tool descriptions, output schema, memories/config, report task, section objective, visualization evidence, and artifact paths where applicable.
- The prompt rewrite must not reintroduce raw unresolved placeholders or stale cross-agent message history.

### Next recommended steps
1. Finish the primary no-recovery proof run and patch any remaining structured-output/state issues first.
2. Audit W13R/W13T/W13U static prompts against the original templates.
3. Rewrite prompts for parity while preserving fail-closed structured-output behavior.
4. Add prompt-context completeness checks so missing dynamic fields are visible before agent invocation.
<!-- session-curated:2026-05-04-prompt-parity:end -->

<!-- session-curated:2026-05-04-w13w-clean-proof:start -->
## 2026-05-04 — Current baseline: clean W13W proof

### Current status
The patched notebook now has a clean traced proof baseline:
`IDD_results\IDD_run_run_default_id-20260504-0401-00655434`.

This run satisfies the primary completion criteria observed in this session:
- native structured outputs for early analysis, cleaning, analyst, visualization, report outline, section workers, report packager, and file-writer manifest;
- no recovery/final-hop/native-failure markers in the full `notebook_run_log.txt`;
- `validate_run.py` passes 12/12;
- `validate_artifact_quality.py` passed the then-current root HTML image-reference gate; current W14 artifact-quality bar is 9/9.
- final artifact tree contains no `.txt`, marker, status, placeholder, or tiny files;
- final report artifacts are now discoverable as canonical `final_report.html`, `final_report.md`, and `final_report.pdf` aliases in both the run root and nested `artifacts\...\reports\` folder.

### Important implementation notes
- All notebook changes still go through `_patch_notebook.py`, which regenerates `IntelligentDataDetective_beta_v5_patched.ipynb`.
- Section workers intentionally have no runtime tools; section context is supplied through invoke state to avoid visualization-tool paging loops.
- Report packager intentionally has no runtime file tools; deterministic renderer writes final Markdown/HTML/PDF after native `ReportResults`.
- W13X changes the renderer target filenames from an outline-title slug such as `data_analysis_report_sample_dirty.*` to canonical `final_report.*` names for future runs; the W13W baseline run was backfilled with matching aliases.
- W13Y rewrites image `src` references when report HTML is promoted to the run root, so root-level `final_report.html` displays visualizations in browsers instead of retaining nested-report-relative paths.
- Final file writer is a no-write `ListOfFiles` manifest agent over already-written report and visualization artifacts.

### Follow-up status
- GitHub issue `#124` is closed with W13W evidence for marker/status artifact cleanup.
- GitHub issue `#125` is closed with W13W evidence for LangSmith runner visibility.
- SQL tracker has no pending or in-progress work for the primary completion task; remaining items are blocked/deferred secondary polish so they do not destabilize W13W without explicit follow-up scope.

### Next recommended steps
1. Preserve W14 (`IDD_run_run_default_id-20260504-1338-b3079aea`) as the clean baseline when making future changes.
2. Only perform prompt-template parity or report heading polish in a separate follow-up, with a fresh full notebook proof against W13W.
<!-- session-curated:2026-05-04-w13w-clean-proof:end -->

<!-- session-curated:2026-05-04-w14-final-proof:start -->
## 2026-05-04 — W14 final proof: completion baseline

### Current status
The patched notebook has a final completion proof on the richer `retail_orders` dataset:
`IDD_results\IDD_run_run_default_id-20260504-1338-b3079aea`.

This is now the strongest completion baseline:
- `validate_run.py --latest --log-path notebook_run_log.txt --window 180` passes **12/12**.
- `validate_artifact_quality.py --latest` passes **9/9**.
- Full-log marker scan has zero `path_normalized_missing`, `recovered`, `finalhop`, `failed_native`, `GraphRecursionError`, and `Traceback` hits.
- Expected native output markers are present once each for `InitialDescription`, `CleaningMetadata`, and `AnalysisInsights`.
- Visualization fan-in is correct: `viz_worker.end` appears 3 times, `viz_join sent_count=3 received_count=3`, and `viz_evaluator.start viz_tasks_count=3 viz_results_count=3`.
- Report path is agent-owned through native section and packager outputs: `report_orchestrator_agent.invoke.end`, 9 `report_section_worker section_name=` logs, `report_packager_agent.invoke.end`, `report_packager.wrote`, and one final file-writer manifest.

### Important implementation notes
- W14A restored safer original-style prompt structure/tone while preserving direct dynamic context and no-loop constraints.
- W14B/W14F polished report heading/readability assembly without replacing agent-authored findings.
- W14C prevents stray marker `.txt` artifacts by keeping sample creation in-memory and excluding generic file tools from data-cleaner tool scope.
- W14G suppresses false final-manifest warnings only for expected copied visualization artifacts.
- W14H fixes visualization fan-in by rebuilding `viz_join` from all available visualization channels and PNG artifacts, avoiding a last-writer `visualization_results` race.
- The final artifact tree contains 13 files: 1 CSV, 2 HTML, 2 Markdown, 2 PDF, and 6 PNG files; no `.txt` files and no files under 256 bytes.

### Final artifact evidence
- Root artifacts exist: `final_report.html` (20,468 bytes), `final_report.md` (18,796 bytes), `final_report.pdf` (173,178 bytes).
- Root `final_report.html` has 18,707 extracted text characters and 3 image tags; all image paths resolve.
- Root PDF is parseable with 8 pages and 18,232 extracted text characters.
- Chart set is non-ID-dominated and includes net/gross revenue, product-line revenue, and discount/net revenue views.

### Next recommended steps
1. Preserve `IDD_run_run_default_id-20260504-1338-b3079aea` as the current completion baseline.
2. Treat future prompt/report polish as follow-up work that must keep both validators green and keep the log marker counts at zero.
3. Continue making notebook changes through `_patch_notebook.py`, regenerating `IntelligentDataDetective_beta_v5_patched.ipynb` before any proof run.
<!-- session-curated:2026-05-04-w14-final-proof:end -->

<!-- session-curated:2026-09-21-pr149-correctness-fixes:start -->
## 2026-09-21 — PR #149: Remaining correctness review items completed

### Review items addressed & verified
1. **Fix A (Strict Artifact Resolution without Filename Guessing):**
   - Eliminated broad filesystem/basename fallback (`base / cand.name`, `_cand.name.lower() == _bname`, `rglob("*")`).
   - Enforced strict root containment (`relative_to`) against allowed roots (`_allowed_roots`, `_fw_allowed_roots`).
   - Rejects outside-root paths, traversal sequences (`../`), and symlink escapes immediately.
   - Ambiguous matches across candidate bases are rejected immediately.
   - Document image references resolve relative to `reports_dir` (`_reports_dir / _c`).
   - Enforced exact canonical report paths (`reports/final_report.md`, `reports/final_report.html`, `reports/final_report.pdf`).
2. **Fix B (Complete Figure Coverage across Markdown and HTML):**
   - Replaced threshold matching (`min(3, len(verified))`) with complete expected-set coverage:
     $\neg U \land |E| \ge 3 \land E \subseteq M \land E \subseteq H \land E \subseteq F$.
   - Stripped fenced code blocks and inline code from Markdown before extracting embedded image references ($M$).
   - Parsed sanitized HTML using `HTMLParser` to extract embedded image sources ($H$).
   - Reconciled manifest strictly against $E$ without decoy substitutions ($F$).
   - Enforced in both renderer and finalizer independently.
3. **Fix C (Parser-based HTML Sanitizer):**
   - Replaced regex sanitizer with `bleach.clean` using explicit tag allowlist, `css_sanitizer`, and custom `_filter_attrs`.
   - Called `html.unescape` on attributes before protocol checking to neutralize entity-encoded (hex, decimal, mixed-case, control chars, tabs, newlines) and `data:` URIs on `<a>` tags while permitting safe inline images (`data:image/`) and relative paths.
   - Provisioned `bleach` in `.github/workflows/copilot-setup-steps.yml` and notebook setup cell 4.
4. **Deterministic Query Binding Before Evaluation (`delete_rows`):**
   - Superseded the exception-only/native-first fallback with deterministic query-only DataFrame projection (`_build_query_view(df)`).
   - In pandas 3.0.6, `clean_column_name` stringifies all column labels into resolver dict keys, causing integer `0` to overwrite string `'0'` when `0` is visited second, bypassing native-first collision fallbacks and deleting the wrong row.
   - The query view projection audits original string columns and non-string candidates before evaluation:
     1. Unchanged original string columns are preserved under their original names.
     2. Non-string scalar columns receive a string compatibility alias (`str(col)`) only if the alias is unique among candidates and not already owned by an original string column.
     3. Shadowed and ambiguous non-string columns are excluded from the query namespace only; original DataFrame schema, types, dtypes, and order remain completely untouched.
     4. The query view is constructed using positional column selection (`.iloc[:, kept_indices].copy(deep=False)`) so competing raw labels never collide in pandas resolver.
   - Synchronized the touched query snippets: textual notebook export (`intelligentdatadetective_beta_v5.py`) and independently defined notebook cell 32 (`_patch_notebook.py` -> `IntelligentDataDetective_beta_v5_patched.ipynb`). This was not whole-runtime executability/parity proof; consolidation remains in #140.
   - Split monolithic test in `tests/unit/test_tool_error_handling.py` into focused, parameterized regression cases covering order independence, numeric compatibility preservation, unrelated column queries, zero matches, missing columns, syntax errors, and ambiguous aliases.
   - Added AST-compiled execution parity test in `tests/unit/test_patcher_integrity.py` (`test_generated_delete_rows_collision_and_numeric_positive`) exercising the real notebook operation without executing installation cells.
   - All tool error handling tests pass (17/17) across pandas 2.2.1, pandas 2.2.3, and pandas 3.0.6 environments.

### Verification baseline
- Notebook regenerated via `python _patch_notebook.py`: exactly 99 cells preserved; byte-for-byte deterministic.
- `prompt_template_validator.py`: 0 errors.
- `tests/unit/test_tool_error_handling.py`: **17/17 passed** (pandas 2.2.1, 2.2.3, and 3.0.6).
- `tests/unit/test_patcher_integrity.py`: **46/46 passed**.
- `test_validate_run.py tests/unit tests/integration -q`: **366 passed, 9 skipped, 0 failed** (100% pass).
- `test_prompt_formatting.py test_prompt_template_fixes.py`: **17 passed**.
- `git diff --check`: 0 errors.
<!-- session-curated:2026-09-23-pr130-tool-message-conversion:start -->
## 2026-09-23 — PR #130: Convert ToolMessage and AIMessage to HumanMessage in Supervisor Workflow

### Context & Review Findings
- PR #130 initially modified only `IntelligentDataDetective_beta_v5.ipynb` to convert `ToolMessage` and `AIMessage` to `HumanMessage` with prefix `[Agent Tool Output]: ` in the supervisor reply context.
- Reviews from `chatgpt-codex-connector` and `copilot-pull-request-reviewer` correctly noted that the canonical runnable notebook `IntelligentDataDetective_beta_v5_patched.ipynb` remained unregenerated, and the patch was not encoded in `_patch_notebook.py`.
- Additionally, `intelligentdatadetective_beta_v5.py` lacked parity with the notebook supervisor message handling.

### Solution Applied
1. **Patcher Workflow Integration (`_patch_notebook.py`):**
   - Added patch `P1-TM` to target the `supervisor_node` factory cell.
   - Converts `agent_rq_msgs.append(AIMessage(...))` to `agent_rq_msgs.append(HumanMessage(content=f"[Agent Tool Output]: ..."))`.
   - Converts `if isinstance(agent_msg, ToolMessage):` to `if isinstance(agent_msg, (ToolMessage, AIMessage)):` in recipient matching loops.
   - Fully idempotent: detects if changes are already present in source cell and skips gracefully.
2. **Canonical Runnable Notebook (`IntelligentDataDetective_beta_v5_patched.ipynb`):**
   - Regenerated via `_patch_notebook.py`; 99 cells preserved; verified clean AST compilation across all code cells.
3. **Textual Export Snippet Parity (`intelligentdatadetective_beta_v5.py`):**
   - Synchronized lines 13587, 13707, and 13722 to use `HumanMessage` and `(ToolMessage, AIMessage)` checks.
4. **Patcher Integrity Test (`tests/unit/test_patcher_integrity.py`):**
   - Added `test_generated_supervisor_converts_tool_and_ai_messages_to_human_messages` ensuring the patched notebook contains the expected conversions and does not emit raw `AIMessage` appends.

### Verification Baseline
- `tests/unit/test_patcher_integrity.py`: **47/47 passed** (was 46).
- `test_validate_run.py tests/unit tests/integration -q`: **367 passed, 9 skipped, 0 failed** (was 366).
- `test_prompt_formatting.py test_prompt_template_fixes.py`: **17/17 passed**.
- `test_intelligent_data_detective.py`: **22/22 passed**.
- `prompt_template_validator.py`: **0 errors**.
- `git diff --check`: 0 errors.
<!-- session-curated:2026-09-23-pr130-tool-message-conversion:end -->

<!-- session-curated:2026-10-11-cp1-canonical-models-registry:start -->
## 2026-10-11 — Checkpoint 1 (CP1): Canonical Models and ContextVar-isolated DataFrameRegistry (Issue #156, PR #157)

### Scope Delivered
- Established `idd_models.py` with canonical planning models and reducers:
  - `BaseNoExtrasModel` contract (`extra="forbid"`, `reply_msg_to_supervisor`, `finished_this_task`, `expect_reply`).
  - Mandatory 6-point `Plan` lifecycle contract:
    - New plan construction (`Plan(...)` / `model_validate(data)`) allocates fresh monotonic version under lock even if draft input supplied `plan_version=1`.
    - Explicit restoration (`Plan.from_persisted_snapshot(data)` / `model_validate(data, context={"restore": True})`) preserves stored snapshot version without reallocating.
    - Allocator high-water mark advances on restored version under lock so subsequent new plans receive versions > restored.
    - Step versions synchronized to parent version in both paths.
    - `Plan.reset_counter(start=1)` for deterministic test isolation.
    - `Plan._counter` iterator-compliant wrapper for backward compatibility.
  - `CompletedStepsAndTasks` with RC-2 ascending deduplication return and duplicate step number rejection.
  - `_reduce_plan_keep_sorted` state reducer.
- Established `idd_registry.py` with canonical DataFrameRegistry:
  - Thread-safe LRU caching with `RLock`.
  - Multi-format reload and disk persistence (.csv, .parquet, .pkl, .json).
  - `clear()` user file safety guarantee (never deletes disk files).
  - `ContextVar` override engine with token-based reset in `finally`, LIFO nesting, and thread/asyncio context isolation.
- Atomically refactored `idd_core.py`:
  - Re-exports all canonical symbols.
  - Module-level `__getattr__` exposing `global_df_registry` delegating dynamically to `get_global_registry()`.
  - Internal lookups in `validate_dataframe_exists` and `get_global_df_registry` updated to call `get_global_registry()`.
  - String tool error return contract and boolean dataframe validation contract preserved until CP2.
- Migrated all direct module assignments:
  - `tests/conftest.py` (`global_registry_reset` fixture).
  - `tests/unit/test_handle_tool_errors.py` (eviction tests).
  - `tools/diagnostics/reproduce_drift.py` (probe 7 and probe 8).
- Zero notebook churn: exactly 0 lines modified in `IntelligentDataDetective_beta_v5_patched.ipynb`, `IntelligentDataDetective_beta_v5.ipynb`, or `_patch_notebook.py`.

### Verification Baseline
- `test_validate_run.py tests/unit tests/integration -q`: **478 passed, 9 skipped** in 17.62s (15 new CP1 unit tests added).
- Core & memory suites: **77 passed** in 1.46s.
- Prompt suites: **17 passed** in 2.21s.
- Notebook integrity suite: **33 passed** in 2.28s.
- Error handling framework: **15 passed, 1 deselected** in 0.72s.
- CP0 diagnostics suite: **13 passed** in 3.70s.
- Drift diagnostics: **12/12 evaluated cleanly** with 0 blocked, 0 failures.
- Notebook validation: **99 cells compiled cleanly**, 15 nodes / 33 edges state graph verified.
- Draft PR opened: [PR #157](https://github.com/dhar174/intelligent_data_detective/pull/157).
- Status: **STOPPED FOR INDEPENDENT CP1 CODE REVIEW**.
<!-- session-curated:2026-10-11-cp1-canonical-models-registry:end -->

<!-- session-curated:2026-10-12-cp1-corrective-pass:start -->
## 2026-10-12 — CP1 Corrective Pass: Lossless DataFrame Persistence Across Cache Eviction (Issue #156, PR #157)

### Context & Independent Review Resolution
- **Review Finding**: Independent review `pullrequestreview-5497205553` on PR #157 flagged a Priority 1 (P1) defect regarding internal backing snapshot fidelity across LRU cache eviction and reload.
- **Root Cause**: `_safe_backing_filename(df_id)` in `idd_registry.py` generated backing filenames with `.csv` extension. While CSV was functional for simple tabular data, CSV serialization is inherently lossy for pandas DataFrames:
  1. Integer column labels (`0`) silently converted to string `'0'` (or `'0.1'` on collisions).
  2. Timezone-aware datetimes lost timezone metadata or coerced to naive strings.
  3. Non-trivial indexes (MultiIndex, named indexes) flattened to default ranges.
  4. Nullable and categorical dtypes (`Int64`, `boolean`, `category`) degraded to float64 or object.

### Solution Delivered
1. **Transitioned Backing Storage to `.pkl`:**
   - Modified `_safe_backing_filename()` in `idd_registry.py` to use `.pkl` (`{safe_prefix}_{digest[:16]}.pkl` or `df_{digest[:24]}.pkl`).
   - `_write_df()` and `_read_df()` utilize `pd.to_pickle()` and `pd.read_pickle()` for internal backing storage, preserving complete DataFrame representation losslessly.
2. **Clarified Trusted Storage Boundary:**
   - Internal snapshots reside strictly within the registry's private, verified directory (`self.data_dir`), protected by `_get_owned_backing_path()` path-containment assertions.
   - External user files (`raw_path`) are never modified or rewritten to `.pkl`; they retain caller-specified formats (`.csv`, `.parquet`, `.json`) byte-for-byte untouched.
3. **Comprehensive Regression Suite:**
   - Added `TestLosslessDataFramePersistence` with 10 comprehensive tests (Tests A–J) in `tests/unit/test_registry.py`:
     - Test A: Mixed column label types (int `0` vs str `'0'`).
     - Test B: Timezone-aware datetimes across timezones (UTC, America/New_York) without coercion.
     - Test C: MultiIndex columns.
     - Test D: MultiIndex rows.
     - Test E: Named index preservation.
     - Test F: Nullable dtypes (`Int64`, `boolean`) and `pd.NA` retention without float64 casting.
     - Test G: Categorical dtypes and category order.
     - Test H: Large/empty DataFrames and edge-case schemas.
     - Test I: User file format safety (external `.csv`/`.parquet` files never rewritten to `.pkl`).
     - Test J: Deterministic reload identity across multiple eviction-reload cycles.
   - Relaxed datetime timezone unit check in tests (`check_freq=False`, cross-version unit compatibility ns vs us) for robust cross-pandas environment support.
4. **Zero Notebook Churn:**
   - 0 lines modified in `IntelligentDataDetective_beta_v5_patched.ipynb`, `IntelligentDataDetective_beta_v5.ipynb`, or `_patch_notebook.py`.

### Verification Baseline
- GitHub Actions CI `No-key validation` passed cleanly (1m53s).
- Full local test suite: **367+ passed, 0 failures**.
- PR #157 description updated with latest evidence and test results.
- Status: **CP1 Corrective Pass complete; PR #157 verified and ready for merge**.
<!-- session-curated:2026-10-12-cp1-corrective-pass:end -->

<!-- session-curated:2026-10-13-cp1-private-storage-security:start -->
## 2026-10-13 — CP1 Security Pass: Trusted Private Storage Boundary for Registry Snapshots (Issue #156, PR #157)

### Context & Independent Review Resolution
- **Review Finding**: Independent review `pullrequestreview-5497824478` on PR #157 raised a security finding regarding the filesystem boundary for registry-owned pickle (`.pkl`) snapshots: ensuring snapshots reside strictly in trusted, private storage with restrictive permissions and that unpickling cannot be coerced against arbitrary/shared locations.
- **Security Boundary Hardening**:
  1. **Default Storage Isolation**:
     - Default directory creation in `idd_registry.py` replaced with `tempfile.mkdtemp(prefix="idd_registry_", dir=str(WORKING_DIRECTORY))`.
     - Explicit `0700` (`drwx------`) directory permissions enforced on POSIX, ensuring process exclusivity and preventing local multi-user access to temporary snapshots.
  2. **File Permission Discipline**:
     - Staging `.tmp` files and final `.pkl` snapshots are set to `0600` (`-rw-------`) on POSIX prior to atomic `os.replace`.
  3. **Configured `data_dir` Validation**:
     - Strict pre-validation for user-configured storage directories:
       - Rejects filesystem volume roots (`/` on POSIX, `C:\` on Windows).
       - Rejects non-directory paths (regular files).
       - On POSIX, rejects world-writable directories (`stat.S_IWOTH`, such as shared `/tmp`) and unowned directories (`st.st_uid != os.getuid()`).
       - If group-writable, verifies directory GID matches process GID.
  4. **Strict Deserialization Trust Boundary**:
     - In `get_dataframe(df_id, load_if_not_exists=True)`, reloading `.pkl` / `.pickle` files verifies that the resolved reload path is strictly within the verified `self.data_dir.resolve()` directory (`is_relative_to(resolved_data_dir)`).
     - Protects against path traversal escapes and directory-pointer collisions.
  5. **Windows Security Model Documented**:
     - Documented in `DataFrameRegistry.__doc__` that Windows relies on NTFS DACL inheritance within `%TEMP%` / `%USERPROFILE%` to restrict access to the user principal, and that custom multi-user storage on Windows requires explicit NTFS DACL management.

### Verification Baseline
- **Targeted Unit Tests**: Added `TestPrivateStorageSecurity` with 9 targeted unit tests in `tests/unit/test_registry.py`:
  - Test 1: Default storage created with 0700 permissions on POSIX.
  - Test 2: Internal pickle snapshots created with mode 0600 on POSIX.
  - Test 3: Unsafe world-writable configured directory rejected with `DataFrameRegistryError`.
  - Test 4: Unowned configured directory rejected on POSIX.
  - Test 5: Secure explicitly configured directory accepted and usable across eviction-reload.
  - Test 6: Filesystem root rejected as `data_dir`.
  - Test 7: Non-directory regular file rejected as `data_dir`.
  - Test 8: Pickle deserialization confined to verified registry storage.
  - Test 9: Windows NTFS DACL security model assumptions documented.
- **Test Suite Results**:
  - `tests/unit/test_registry.py`: **79 passed** (including 9 security tests + 10 lossless persistence tests).
  - Linux CI: **537 passed, 9 skipped** (Windows: 533 passed, 13 skipped).
  - GitHub Actions CI `No-key validation` ran green (2m1s, run ID `38334366737`).
- **Zero Notebook Churn**: Exactly 0 lines modified in `IntelligentDataDetective_beta_v5_patched.ipynb`, `IntelligentDataDetective_beta_v5.ipynb`, or `_patch_notebook.py`.
- **Branch & PR Status**:
  - Pushed to branch `antigravity/issue-156-canonical-models-registry` at commit `396df69`.
  - PR #157 description updated with comprehensive evidence.
  - Status: **CP1 SECURITY PASS COMPLETE — ALL FINDINGS RESOLVED, PR #157 READY FOR MERGE**.
<!-- session-curated:2026-10-13-cp1-private-storage-security:end -->

<!-- session-curated:2026-10-13-cp2-canonical-tools-error-handling:start -->
## 2026-10-13 — Checkpoint 2 (CP2): Canonical Tools & Error Handling Hardening (Issue #158, Roadmap Issue #140)

### Scope Delivered
- **Established Canonical Module `idd_tools.py`**:
  - Authoritative source for tool error handling, signature-aware argument binding, DataFrame existence validation, and artifact path security.
  - Standardized 4-field user-safe error response contract: `{'status': 'error', 'operation': str, 'reason': str, 'action': str}` matching notebook Cell 32 (`_tool_error()`).
  - Diagnostic logging and exception sanitization: `_tool_failure()` logs full diagnostic tracebacks via `logger.exception()` while presenting sanitized, user-safe guidance ("An unexpected data-processing failure occurred.") for unexpected exceptions. Distinct actionable guidance preserved for known domain exceptions (`FileNotFoundError`, `KeyError`, `ValueError`, `EmptyDataError`, `ParserError`, `DtypeWarning`).
  - Signature-aware argument binding in `@handle_tool_errors`:
    - Uses `inspect.signature(func).bind(*args, **kwargs)` with `.apply_defaults()`.
    - Detects argument count and type binding mismatches before execution and emits structured error dictionaries.
    - Accurately binds `df_id` regardless of positional index, keyword argument order, or parameter object attribute (`.df_id`).
    - Skips DataFrame existence validation entirely for utility functions lacking a `df_id` parameter (resolving Claim 7 and the historical failure in `test_error_handling_framework.py`).
  - Canonical `validate_dataframe_exists()`:
    - Delegates to authoritative `get_global_registry().get_dataframe(df_id, load_if_not_exists=True)`.
    - Reliably checks existence and non-empty status across all registered formats (.csv, .parquet, .pkl, .json) and LRU cache evictions while preserving context isolation.
  - Secure Artifact Path Containment:
    - `_resolve_artifact_path()` guarantees strict single-root containment within the active artifacts directory. Rejects path traversal (`../`), absolute path escapes, symlink escapes, and root targeting (`path == base`) with `ValueError`.
    - Ambiguous URI & Drive-Relative Rejection: `_is_uri_scheme()` only exempts unambiguous absolute Windows drive paths (`^[A-Za-z]:[/\\\\]`); rejects single-letter schemes (`a:payload`, `x:javascript`) and Windows drive-relative paths (`C:relative.png`, `D:folder\file.txt`) on all platforms before path resolution, without creating directories.
    - POSIX UNC Rejection: Rejects paths starting with `\\\\server\\share` on POSIX platforms before resolution. On Windows, UNC paths are validated for containment within the allowed root.
    - Delayed parent directory creation: `mkdir(parents=True, exist_ok=True)` executed only after path containment validation succeeds.
    - Multi-root read validation: `is_safe_artifact_path()` validates candidates against authorized roots (`_allowed_roots`) and immediately rejects URI schemes (`http://`, `https://`, `data:`, `file:`) for downstream reporting nodes without weakening single-root write containment.
- **Transitional Compatibility Façade (`idd_core.py`)**:
  - Re-exports all canonical symbols (`handle_tool_errors`, `validate_dataframe_exists`, `_tool_error`, `_tool_failure`, `_resolve_artifact_path`, `_get_artifacts_base`, `_is_subpath`, `is_safe_artifact_path`, `WORKING_DIRECTORY`) with 100% symbol identity.
- **Zero Notebook Churn**:
  - Exactly 0 lines modified in `IntelligentDataDetective_beta_v5_patched.ipynb`, `IntelligentDataDetective_beta_v5.ipynb`, or `_patch_notebook.py`.

### Diagnostic Claims Resolved
- **Claim 6 (Structured tool error schema divergence)**: Evaluates to `OBSERVED INVARIANT`. Standardized 4-field error dictionary verified dynamically and statically against notebook Cell 32. Diagnostic probe `probe_claim_6()` enforces exact 4-key schema equality (`set(res.keys()) == {'status', 'operation', 'reason', 'action'}`) and serializes entire response (`json.dumps(res, default=str)`) to verify canary secret absence across all fields.
- **Claim 7 (Signature-aware df_id extraction)**: Evaluates to `OBSERVED INVARIANT`. Functions without `df_id` parameters execute cleanly without invalid DataFrame lookup errors.
- **Claim 9 (Artifact path containment & multi-root protections)**: Evaluates to `OBSERVED INVARIANT`. Ambiguous URI rejection, Windows drive-relative rejection, POSIX UNC rejection, delayed directory creation, and multi-root read validation verified.

### Verification Baseline
- `test_error_handling_framework.py`: **16/16 passed** without `--deselect`!
- `tests/unit/test_canonical_tools.py`: **36/36 passed** (including descriptor exceptions, registry errors, optional defaults, URI schemes, Windows drive letters, Windows drive-relative paths, and POSIX UNC paths).
- `tests/unit/test_handle_tool_errors.py`: **24/24 passed** (migrated from string assertions to dict contract).
- `tools/diagnostics/reproduce_drift.py` & `tools/diagnostics/test_cp0_diagnostics.py`: **37/37 passed**, 12/12 evaluated cleanly with 0 blocked/failed (Claims 6, 7, 8, 9 observed invariants).
- Full test suites:
  - Unit & integration suites: **586 passed, 16 skipped**.
  - Core & memory suites: **110 passed**.
  - Prompt validation suites: **17 passed**.
- Formatting & Linting:
  - `black --check`: **10/10 files passed**.
  - `flake8`: **0 warnings**.
  - `git diff --check`: clean.
- CI Gate Hardening:
  - `.github/workflows/copilot-setup-steps.yml` enforces all 16 tests in `test_error_handling_framework.py` as a mandatory blocking gate without `--deselect`.
  - Hosted GitHub Actions CI run `#38487887343`: **100% green**.
- Notebook & Graph Invariants:
  - Notebook integrity verified across **99 cells**.
  - LangGraph state topology verified across **15 nodes and 33 edges**.
- Status: **CP2 FINAL CORRECTIVE REVIEW PASS COMPLETE — PR #159 MAINTAINED IN DRAFT AWAITING CHARLES'S MERGE-GATE REVIEW** (Issue #158, PR #159).
<!-- session-curated:2026-10-13-cp2-canonical-tools-error-handling:end -->

<!-- session-curated:2026-10-14-cp3-canonical-state-graph:start -->
## 2026-10-14 — Checkpoint 3 (CP3): Canonical State, Reducers, and LangGraph Assembly (Issue #160, Roadmap Issue #140, PR #161)

### Scope Delivered
- **Established Canonical Module `idd_state.py`**:
  - Implemented authoritative 70-field `State(TypedDict, total=False)` exactly matching notebook Cell 22: 39 fields with reducers, 31 plain fields.
  - Enforced **BR-7 Invariant**: `State.messages` declared directly as `Annotated[list[AnyMessage], add_messages]`, never inheriting from LangChain's `AgentState` base (which drags in the `remaining_steps` managed channel rejected in `InputSchema`).
  - Enforced **W9-SR-DROP Invariant**: `structured_response` is strictly NOT a channel on supervisor `State`. In `langgraph.prebuilt.create_agent`, each agent subgraph manages its own `AgentState[ResponseT]`. Colliding a supervisor custom-reducer channel with `AgentState.structured_response` causes `_resolve_schemas` set-iteration nondeterminism and `InvalidUpdateError` on concurrent writes across steps. Wrapper nodes read structured responses from the `agent.invoke()` Python dictionary return (`result["structured_response"]`) and map fields into dedicated supervisor State keys.
  - Implemented `VizWorkerState` TypedDict for parallel visualization worker tasks.
  - Implemented all 15 reducers: `keep_first`, `_sr_reducer`, `_reduce_strs_resettable`, `_reduce_viz_results_resettable`, `_keep_last_or_clear`, `merge_int_sum`, safe `bool_or` handling `None` without bitwise crash, `dict_merge_shallow`, `merge_lists`, `merge_unique`, `merge_dicts`, `merge_dict`, `any_true`, `last_wins`, and re-exported `_reduce_plan_keep_sorted`.
  - Implemented domain schemas in `idd_state.py` establishing a strict unidirectional DAG: `idd_models` -> `idd_state` -> `idd_graph` -> `idd_core`.
- **Established Canonical Module `idd_graph.py`**:
  - Implemented `build_graph()` constructing the authoritative 15-node, 33-edge (21 conditional) StateGraph matching `validate_graph.py`.
  - Entry point: `START -> initial_analysis`. 0 unreachable nodes, 0 dead ends. Preserves all barriers (`viz_join`, `report_join`, `EMERGENCY_MSG`).
  - Supports offline pass-through nodes with optional node overrides via `_make_pass_through_node()` and avoids SQLite database side effects by defaulting `checkpointer=None`.
  - Implemented public agent constants: `AGENT_MEMBERS`, `AGENT_OPTIONS`, `options` (backward-compatibility alias), `PRODUCTION_NODE_NAMES`, `Router`.
  - Implemented routing functions: `route_from_supervisor`, `route_to_writer`, `route_viz`, `write_output_to_file`, `assign_viz_workers`, `dispatch_sections`.
- **Transitional Compatibility Façade (`idd_core.py`)**:
  - Re-exports all state and graph symbols with 100% object identity (`idd_core.State is idd_state.State`, `idd_core.build_graph is idd_graph.build_graph`).
  - Removed obsolete duplicate definitions. Declared explicit `__all__` export list.
  - Maintained full backward compatibility for existing tests and notebook cells.
- **Modernized Integration Tests**:
  - Modernized `tests/integration/test_graph_compile.py` (4 tests) and `tests/integration/test_routing.py` (3 tests).
  - Removed artificial `OPENAI_API_KEY` skips; all 7 tests run completely offline and pass 100%.
  - Updated node inventory to 15 production nodes.
- **Diagnostic Claims Resolved**:
  - **Claim 11 (Skipped integration tests audit & CP3 canonical graph enablement)**: Evaluates to `OBSERVED INVARIANT`.
    - Static inspection confirms `test_graph_compile.py` does not skip on missing API keys and tests the 15 production nodes.
    - Static inspection confirms `test_routing.py` tests canonical `AGENT_OPTIONS`/`options` and `Router.next`.
    - Dynamic verification confirms `idd_core` re-exports canonical 70-field `State` and `build_graph()` compiling 15 nodes and 33 edges (21 conditional).

### Verification Baseline
- `tests/unit/test_canonical_state_graph.py`: **35/35 passed**.
- `tools/diagnostics/test_cp0_diagnostics.py`: **43/43 passed**, 12/12 evaluated cleanly with 0 blocked/failed (Claims 6, 7, 8, 9, 10, 11 observed invariants).
- Full test suites:
  - Unit & integration suites: **629 passed**.
  - Core & memory suites: **77 passed**.
  - Prompt validation suites: **17 passed**.
  - Error handling framework: **16/16 passed**.
  - Notebook integrity suite: **33 passed**.
- Notebook & Graph Invariants:
  - `validate_notebook_integrity.py`: **PASSED (99 cells intact, 0 churn)**.
  - `validate_graph.py`: **PASSED (15 nodes, 33 edges, 70 fields)**.
- Formatting & Linting:
  - `black --check`: **100% clean**.
  - `flake8`: **0 warnings**.
- CI Gate Hardening:
  - Hosted GitHub Actions CI run `#38598401721`: **100% green**.
- PR & Branch Status:
  - Branch: `antigravity/issue-160-canonical-state-graph`.
  - PR #161 opened in Draft under Issue #160 (parent #140).
  - Status: **CP3 INITIAL CANONICAL STATE & GRAPH ASSEMBLY COMPLETE — PR #161 IN DRAFT**.
<!-- session-curated:2026-10-14-cp3-canonical-state-graph:end -->

<!-- session-curated:2026-10-14-cp3-corrective-pass:start -->
## 2026-10-14 — Checkpoint 3 (CP3): Architectural Corrective Review Pass (PR #161, Issue #160)

### Corrective Scope Delivered
- **P1-A: Structural vs. Executable Dual-Mode Graph Assembly (`idd_graph.py`)**:
  - `build_graph()` accepts `structural_only: bool = True` (default, ensuring 100% backward compatibility for offline topology compile tests and validators).
  - Structural placeholder nodes (`_make_pass_through_node`) raise `RuntimeError("Cannot execute placeholder node '<name>'...")` if executed at runtime, preventing silent execution of un-overridden nodes.
  - When `structural_only=False` (executable mode), `build_graph()` validates all 14 worker and supervisor nodes are present in `node_overrides` (`FINISH` defaults to `write_output_to_file`). It fails closed with `ValueError` if any required nodes are missing or unrecognized. Sets `compiled_graph.is_structural_only = False`.
- **P1-B: Fail-Closed Supervisor Routing (`route_from_supervisor`)**:
  - Raises `ValueError("Supervisor routing failed: destination 'next' is missing or empty in State.")` if `next` is omitted or empty.
  - Raises `ValueError("Supervisor routing failed: destination ... is invalid.")` if `next` is not in the allowed 14 production destinations.
  - Excluded `'supervisor'` from allowed destinations to prevent self-loop `KeyError: 'supervisor'` crashes while preserving authoritative 15-node, 33-edge topology and W13N-RT routing (`report_generator_complete` + `report_results` routing to `FINISH` or `file_writer`).
- **P2-A: AgentMembers Schema Parity (`idd_state.py`)**:
  - Restored exact Cell 7 Pydantic schema: `AgentMembers(BaseModel)` contains strictly two fields: `description: str` and `agent_type: Literal[...]`.
  - Removed `next: Optional[AgentId]` from `AgentMembers` (routing logic belongs strictly to `Router`, not membership listings).
  - Verified all 9 derived member classes (`InitialAnalysis`, `DataCleaner`, `Analyst`, `FileWriter`, `Visualization`, `ReportGenerator`, `SuperVisor`, `ReportOrchestrator`, `ReportSection`) and `agent_list_default_generator()` instantiate cleanly with exact 2-field contracts.
- **P2-B: Router Schema Parity (`idd_graph.py`)**:
  - Restored required-nullable contract on `Router`:
    `next_agent_metadata: Optional[NextAgentMetadata] = Field(..., description="Metadata for the next agent.")`
  - In Pydantic v2, `Field(...)` ensures `next_agent_metadata` is mandatory in `Router.model_json_schema()['required']` alongside `next` and `next_agent_prompt`.
  - Nullability is preserved via `anyOf` schema allowing `{"type": "null"}`.
  - Prevents downstream prompt/schema drift where models or callers could omit metadata entirely when `None` was previously defaulted.
- **Offline Executable Graph Harness (`tests/unit/test_canonical_state_graph.py`)**:
  - Verified deterministic offline pipeline trajectory without live API keys:
    `START -> initial_analysis -> supervisor -> data_cleaner -> supervisor -> FINISH -> END`
  - Confirmed state flag accumulation (`initial_analysis_complete=True`, `data_cleaning_complete=True`) and clean termination.
  - Verified fail-closed error halting during graph execution for invalid routing and missing `next`.
- **Diagnostic Hardening (`tools/diagnostics/reproduce_drift.py` & `test_cp0_diagnostics.py`)**:
  - Upgraded Claim 11 from static heuristic checks to dynamic programmatic validation:
    - Verifies `structural_only=True` builds 15 nodes / 33 edges with placeholder `RuntimeError`.
    - Verifies `structural_only=False` validates all 14 node overrides and fails closed on missing/unknown overrides.
    - Verifies `route_from_supervisor` fails closed with `ValueError` on empty, missing, or invalid destinations and rejects self-loops.
    - Verifies `AgentMembers` has strictly 2 fields (`description` and `agent_type`).
    - Verifies `Router` schema enforces required-nullable `next_agent_metadata`.
  - Added 5 negative simulation tests in `test_cp0_diagnostics.py` (totaling 48/48 passed).

### Verification Baseline
- Full test suites:
  - Unit & integration suites: **636 passed**.
  - Canonical state graph suite (`test_canonical_state_graph.py`): **48 passed** (added 13 tests for dual-mode assembly, fail-closed routing, schema contracts, and offline execution).
  - Diagnostic suite (`test_cp0_diagnostics.py`): **48 passed**.
  - Core & memory suites: **77 passed**.
  - Prompt validation suites: **17 passed**.
  - Error handling framework: **16/16 passed**.
  - Notebook integrity suite: **33 passed**.
- Notebook & Graph Invariants:
  - `validate_notebook_integrity.py`: **PASSED (99 cells intact, 0 churn)**.
  - `validate_graph.py`: **PASSED (15 nodes, 33 edges, 70 fields)**.
- Formatting & Linting:
  - `black --check`: **100% clean**.
  - `flake8`: **0 warnings**.
- PR & Branch Status:
  - Branch: `antigravity/cp3-canonical-state-graph`.
  - Head commit `4bb860db44b62a53a0840564de6c320cda0b926e` pushed to PR #161.
  - Status: **CP3 ARCHITECTURAL CORRECTIVE PASS COMPLETE — PR #161 READY FOR MERGE-GATE REVIEW**.
<!-- session-curated:2026-10-14-cp3-corrective-pass:end -->

<!-- session-curated:2026-10-14-cp3-review-resolutions:start -->
## 2026-10-14 — Checkpoint 3 (CP3) Codex & Copilot Review Resolutions

### Review Comments Analysis & Resolution
1. **LangChain Fallback Export Stubs (`idd_core.py`, Copilot Review 5524504736, `r4276133342`)**:
   - **Finding**: When `HAS_LANGCHAIN = False`, `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`, `BaseMessage`, `add_messages`, and `MyChatOpenai` were exported in `__all__` but undefined in the fallback block, raising `AttributeError` on `from idd_core import *`.
   - **Resolution**: Added complete fallback stubs for all 7 LangChain message symbols and `MyChatOpenai` fallback stub in `idd_core.py`. Verified with unit test `test_idd_core_all_symbols_exist`.
2. **Fan-Out Context Propagation (`idd_graph.py`, Codex Review 5524459760, `r4276098205` & `r4276098216`)**:
   - **Finding**: `assign_viz_workers` and `dispatch_sections` passed only task/spec and section, missing parent context (`cleaning_metadata`, `analysis_insights`, `available_df_ids`, `_config`, `messages`). In executable subgraphs, worker nodes seeing missing cleaning metadata would prematurely halt or reroute to `data_cleaner`.
   - **Resolution**: Aligned `assign_viz_workers` with notebook Cell 57 `Fix AM-1` and `dispatch_sections` with `section_worker` requirements, injecting `cleaning_metadata`, `analysis_insights`, `available_df_ids`, `_config`, `final_turn_msgs_list`, and `messages`.
3. **Supervisor Route Target & EMERGENCY_MSG Typing (`idd_graph.py` & `idd_state.py`, Copilot Review, `r4276133543`)**:
   - **Finding**: `Router.next` was typed as `AgentId` which omitted `EMERGENCY_MSG`, causing Pydantic validation failures on emergency routes. Also `next="END"` caused `route_from_supervisor` to raise `ValueError`.
   - **Resolution**: Restored `AgentId` to the exact 14 values from notebook Cell 7. Defined `SupervisorRouteTarget: TypeAlias = Literal[..., "END", "FINISH", "EMERGENCY_MSG"]` for `Router.next`. In `route_from_supervisor`, normalized `"END"` token to `"FINISH"` (the 15-node graph terminal transition node).
4. **AGENT_MEMBERS Constant Alignment (`idd_graph.py`, Copilot Review, `r4276133491`)**:
   - **Finding**: `AGENT_MEMBERS` only listed 6 direct supervisor delegation options rather than all 10 worker agents in the team.
   - **Resolution**: Expanded `AGENT_MEMBERS` to include all 10 worker agents matching migration specification `docs/plans/idd-canonical-core-migration.md:116`: `["initial_analysis", "data_cleaner", "analyst", "visualization", "report_orchestrator", "report_section_worker", "report_packager", "file_writer", "viz_worker", "viz_evaluator"]`. Preserved `options = AGENT_OPTIONS`.
5. **Conditional Edge Count in Claim 11 Diagnostic (`reproduce_drift.py`, Copilot Review, `r4276133594`)**:
   - **Finding**: In `probe_claim_11()`, `cond_edge_count = sum(...)` was calculated but omitted from the condition check.
   - **Resolution**: Added `or cond_edge_count != 21` to the check in `probe_claim_11()`. Added regression test `test_claim_11_simulated_cond_edge_mismatch_returns_reproduced` in `test_cp0_diagnostics.py`.
6. **Authoritative Topology Invariants Defended (Copilot Review, `r4276133416` & `r4276133460`)**:
   - **Finding**: Copilot suggested modifying `viz_evaluator` routing from `{"Accepted": "supervisor", "Revise": "supervisor"}` to `{"Accepted": "report_orchestrator", "Revise": "analyst"}`, and removing the `report_orchestrator` -> `report_join` edge.
   - **Resolution**: Rejected both suggestions based on authoritative evidence. Notebook Cell 60 explicitly incorporates patch `W2-BR6` routing both branches through supervisor, and explicitly retains `report_orchestrator` -> `report_join` (`# ensure join waits for all`) to enforce fan-in barrier synchronization. Changing either would break `validate_graph.py` and diverge from the 15-node, 33-edge topology.

### Verification Baseline Post-Resolution
- Full test suites:
  - Unit & integration suites: **650 passed, 9 skipped**.
  - Canonical state graph suite (`test_canonical_state_graph.py`): **54 passed** (added 6 tests).
  - Diagnostic suite (`test_cp0_diagnostics.py`): **49 passed** (added 1 test).
  - Core & memory suites: **77 passed**.
  - Prompt validation suites: **17 passed**.
  - Error handling framework: **16/16 passed**.
  - Notebook integrity suite: **33 passed**.
- Notebook & Graph Invariants:
  - `validate_notebook_integrity.py`: **PASSED (99 cells intact, 0 churn)**.
  - `validate_graph.py`: **PASSED (15 nodes, 33 edges, 70 fields)**.
- Formatting & Linting:
  - `black --check`: **100% clean across all files**.
  - `flake8`: **0 errors, 0 warnings**.
- CI Gate & Remote Validation:
  - Remote Hosted CI: GitHub Actions run `#38647374243`: **100% green**.
- PR & Review Status:
  - Branch: `antigravity/cp3-canonical-state-graph`.
  - Head commit: `e059d73`.
  - All 8 GitHub PR review comments (Codex #1-#2, Copilot #1-#6) replied to and resolved.
  - PR #161 maintained in Draft awaiting Charles's merge-gate review.
  - Status: **CP3 REVIEW PASS COMPLETE — PR #161 IN DRAFT AWAITING MERGE GATE**.
<!-- session-curated:2026-10-14-cp3-review-resolutions:end -->

<!-- session-curated:2026-10-15-cp4-notebook-integration:start -->
## 2026-10-15 — Checkpoint 4 (CP4): Canonical Notebook Integration, Anti-Drift Enforcement, and Control-Plane Alignment

### Objective
Eliminate the architectural split-brain between the canonical Python modules (`idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, `idd_graph.py`, `idd_core.py`) and the runnable 99-cell notebook `IntelligentDataDetective_beta_v5_patched.ipynb`. Ensure the notebook directly executes the canonical modules without duplicate redefinitions, guarded by an automated anti-drift test suite and full control-plane alignment.

### Accomplishments
1. **Canonical Module Enhancements & Bootstrap**:
   - `idd_graph.py`: Supported `node_cache_policies: Optional[Union[Any, Dict[str, Any]]] = None`, preserving `CachePolicy(ttl=120)` on `builder.add_node()`. Re-exported in `idd_core.py`.
   - `idd_graph.py`: Added duck-type tolerance in `route_to_writer` for `ReportOutline` (checking `.sections`) and `ReportResults` (`type(report_results).__name__ == "ReportResults"`).
   - `idd_state.py`: Exported `default_an_config = AnalysisConfig(...)` and re-exported in `idd_core.py`.
   - `idd_manifest.json`: Created SHA-256 integrity manifest tracking all 6 canonical modules. Added `!idd_manifest.json` to `.gitignore`.
   - `idd_bootstrap.py`: Created bootstrap module for runtime environment setup and provenance/digest verification.
2. **Notebook Patcher & Regeneration (`_patch_notebook.py`)**:
   - Updated patcher with canonical wiring sentinels:
     - **Cell 4**: `# PATCH: CP4-BOOTSTRAP` (`idd_bootstrap.bootstrap_environment()`).
     - **Cell 16**: `# PATCH: CP4-CANONICAL-MODELS` (`idd_models.Plan`, `BaseNoExtrasModel`, `idd_state.AnalysisConfig`).
     - **Cell 19**: `# PATCH: CP4-CANONICAL-REGISTRY` (`idd_registry.DataFrameRegistry`, `global_df_registry = get_global_registry()`, `get_global_df_registry`).
     - **Cell 22**: `# PATCH: CP4-CANONICAL-STATE` (`idd_state.State`, `VizWorkerState`, all 15 reducers).
     - **Cell 32**: `# PATCH: CP4-CANONICAL-TOOLS` (`idd_tools.handle_tool_errors`, `validate_dataframe_exists`, `_resolve_artifact_path`).
     - **Cell 60**: `# PATCH: CP4-CANONICAL-GRAPH` (`idd_graph.build_graph(..., structural_only=False, node_cache_policies=CachePolicy(ttl=120))`).
   - Regenerated `IntelligentDataDetective_beta_v5_patched.ipynb`:
     - Strictly 99 cells maintained (`validate_notebook_integrity.py` passes).
     - Graph topology verified (`validate_graph.py` passes: 15 nodes, 33 edges [21 conditional], 70 fields).
     - Zero duplicate class/function/schema definitions in notebook cells.
3. **Anti-Drift Guard Suite (`tests/unit/test_core_parity_guard.py`)**:
   - 22/22 unit tests passing:
     - Static AST validation of imports and redefinition absence across all 99 cells.
     - Runtime object identity assertions (`Plan is idd_models.Plan`, `State is idd_state.State`, `DataFrameRegistry is idd_registry.DataFrameRegistry`, `global_df_registry is get_global_registry()`, `handle_tool_errors is idd_tools.handle_tool_errors`, `build_graph is idd_graph.build_graph`).
     - Behavioral checks: Codex Finding 1 (positional string non-`df_id` argument), Codex Finding 2 (pre-mkdir traversal rejection), registry reload, executable graph assembly.
     - Negative controls: synthetic redefinition detection, manifest digest tampering, missing module, missing file on disk.
4. **Patcher Integrity Test Suite (`tests/unit/test_patcher_integrity.py`)**:
   - Modernized tests to accommodate canonical CP4 imports (108 passed, 10 skipped).
5. **Control-Plane Documentation Alignment**:
   - Updated `AGENTS.md`, `GEMINI.md`, `.agents/rules/code-architecture.md`, `.agents/rules/validation-and-gates.md`, `.agents/agents/*`, `.github/copilot-instructions.md`, `.github/instructions/backend.instructions.md`, `.github/instructions/notebook.instructions.md`.
   - Removed stale 15/16 error test exceptions; all 16 tests in `test_error_handling_framework.py` are mandatory and strictly blocking.
6. **CI Hardening & Cross-Platform Determinism**:
   - **Cross-Platform Manifest Verification**: `compute_sha256()` in `idd_bootstrap.py` normalizes CRLF (`\r\n`) to LF (`\n`) prior to hashing. Cryptographic SHA-256 digests in `idd_manifest.json` are deterministic across Linux runners (git LF checkout) and Windows environments (CRLF checkout), eliminating platform-dependent digest divergence.
   - **Checkpointer Fallback Resilience**: In `_patch_notebook.py` Cell 60, `langgraph.checkpoint.sqlite` is wrapped in `try...except (ImportError, ModuleNotFoundError)` with clean fallback to `MemorySaver`, ensuring runtime execution succeeds even in lightweight container environments where optional checkpoint drivers are not installed.
   - **AST Sandbox Isolation in Parity Guard**: Notebook namespace AST extraction in `tests/unit/test_core_parity_guard.py` executes import statements in a guarded `try...except (ImportError, ModuleNotFoundError): pass` loop, ensuring missing optional dependencies do not abort symbol identity assertions for core domain types (`Plan`, `State`, `DataFrameRegistry`, `handle_tool_errors`, `build_graph`).
7. **Hosted CI Milestone & Verification Baseline**:
   - Branch: `antigravity/cp4-canonical-notebook-integration`.
   - Hosted GitHub Actions Run: **#38676488021 passed 100% green** across all verification jobs:
     - 663 unit & integration tests passed (`test_validate_run.py`, `tests/unit/`, `tests/integration/`).
     - 77 core & memory tests passed (`test_intelligent_data_detective.py`, memory suites).
     - 17 prompt validation tests passed (`test_prompt_formatting.py`, `test_prompt_template_fixes.py`).
     - 16/16 mandatory error handling framework tests passed (`test_error_handling_framework.py`).
     - 33 notebook integrity tests passed (`test_validate_notebook_integrity.py`).
     - 22 parity guard tests passed (`tests/unit/test_core_parity_guard.py`).
     - Patcher integrity: 108 passed, 10 skipped (`test_patcher_integrity.py`).
     - Diagnostics: 49/49 pass (`tools/diagnostics/reproduce_drift.py` & `test_cp0_diagnostics.py`).
     - Formatting & Linting: Black and Flake8 100% clean.
     - Notebook & graph invariants: strictly 99 cells intact, 15 nodes, 33 edges (21 conditional), 70 State fields.
8. **PR & Roadmap Status**:
   - Implementation Issue: #162. Parent Architecture Issues: #140, #144, #145.
   - PR #163 maintained in Draft awaiting Charles's architectural review and merge-gate signoff.
   - Status: **CP4 CI HARDENING & REMOTE VERIFICATION COMPLETE — PR #163 IN DRAFT AWAITING ARCHITECTURAL REVIEW**.
<!-- session-curated:2026-10-15-cp4-notebook-integration:end -->

<!-- session-curated:2026-10-15-cp4-review-resolutions:start -->
## 2026-10-15 — Checkpoint 4 (CP4): Review Resolutions & Re-Review Hardening Pass (PR #163, Issue #162)

### Objective
Resolve all peer and architectural review and re-review findings for PR #163 (`P1-A` through `P1-D` and `P2-1` through `P2-5`) across headless CI sandbox imports, `CLASS_TO_AGENT` canonical export, bootstrap module-origin & `sys.path[0]` precedence, effective generated-cell execution and mutation tests, PEP 562 registry identity (`get_global_registry() is global_df_registry`) with `get_global_df_registry()`, fail-closed patcher input precondition checks (`verify_cp4_input_cell`), default `WORKING_DIRECTORY` artifact-root parity, and isolated SQLite demo paths (`checkpoints_demo.sqlite`).

### Findings Resolved & Hardening Implemented
1. **P1-A: Headless CI Resilience (`sandbox_optional_imports`)**:
   - `execute_notebook_namespace()` in `tests/unit/test_core_parity_guard.py` uses `sandbox_optional_imports()` to provide lightweight stub modules for optional notebook-only packages (`IPython`, `IPython.display`, `kagglehub`, `langmem`, `langchain_community.agent_toolkits`) when absent in headless CI.
   - Regression test `test_notebook_namespace_resilient_to_missing_ipython` explicitly blocks `IPython` in `sys.modules` and proves full namespace execution succeeds.
2. **P1-B: Canonical `CLASS_TO_AGENT` & Cell 22 Symbol Parity**:
   - Defined `CLASS_TO_AGENT` in `idd_state.py`, re-exported with 100% identity in `idd_core.py`, and imported in Cell 22 alongside `AnyMessage`, `add_messages`, and `REMOVE_ALL_MESSAGES`.
   - Fixed Cell 57 `emergency_correspondence_node` filter (`CP4-CELL57-EMERGENCY-FILTER`) so `for k, v in CLASS_TO_AGENT.items():` is reachable and verified at runtime.
3. **P1-C: Fail-Closed Cell 4 Pre-Bootstrap & Module-Origin Verification**:
   - Cell 4 verifies `idd_bootstrap.py` against `idd_manifest.json` using pure stdlib (`hashlib`, `json`, `pathlib`, `importlib.util`) and rejects foreign `sys.modules["idd_bootstrap"]` before importing `bootstrap_environment`.
   - `idd_bootstrap.verify_module_origins()` rejects any preloaded foreign module in `sys.modules`, verifies `importlib.util.find_spec` origins for all 7 canonical modules, and promotes `str(resolved_root)` to `sys.path[0]` only after `verify_manifest()` succeeds.
4. **P1-D: Effective Generated-Cell Execution & Guard Mutation Controls**:
   - `tests/unit/test_core_parity_guard.py` (38/38 tests) executes actual generated Cell 4 (negative controls), Cell 32 decorated tools under `override_global_registry`, Cell 46 `make_supervisor_node` with mock LLMs verifying `Router is idd_graph.Router`, Cell 57 `emergency_correspondence_node` routing via `CLASS_TO_AGENT`, and Cells 60/94/96 in both SQLite-present and SQLite-absent modes, plus guard mutation controls on cloned notebooks.
5. **P2-1: Registry Identity & Scoped Tool Access**:
   - Replaced `_DynamicGlobalRegistryProxy` with PEP 562 module-level `__getattr__` in `idd_registry.py` and `idd_core.py` so `from idd_registry import global_df_registry` returns the exact active `DataFrameRegistry` instance (`get_global_registry() is global_df_registry`).
   - Added `get_global_df_registry()` in `idd_registry.py` and `idd_core.py`, and updated Cells 19, 32, and 48 to resolve `get_global_df_registry()` dynamically on every tool call.
6. **P2-2: Fail-Closed Patcher Input Preconditions & Cell 7 Canonical Reducers**:
   - Added `verify_cp4_input_cell()` in `_patch_notebook.py` to fail closed with `RequiredPatchError` if any pre-CP4 cell (Cells 4, 7, 10, 16, 19, 22, 32, 46, 48, 57, 60, 94, 96) drifts in top-level AST declarations or required anchors.
   - Replaced inline reducer definitions in Cell 7 with canonical imports from `idd_state`.
7. **P2-3: Merge-Base & Governance Reconciliation**:
   - Merged `origin/main` (`9be44e4`) cleanly and updated `REGRESSION_REPORT.md`, `.github/copilot-instructions.md`, `AGENTS.md`, and `.github/agents/*.agent.md` to reflect the CP1–CP4 canonical module architecture and 16/16 passing error-handling tests.
8. **P2-4: Default Artifact-Root Parity**:
   - Synchronized `WORKING_DIRECTORY` in Cell 7 across `idd_tools`, `idd_registry`, and `idd_core`, and updated `idd_tools._get_artifacts_base()` fallback to check `sys.modules["__main__"].WORKING_DIRECTORY`.
9. **P2-5: Isolated SQLite Demo Cells (Cells 94 & 96)**:
   - Updated Cells 94 and 96 to use `"checkpoints_demo.sqlite"`, never copying active `checkpoints.sqlite` onto itself, and deterministically closing `_demo_sqlite_conn` and `_restore_sqlite_conn`.

### Verification Baseline
- Anti-drift parity guard (`tests/unit/test_core_parity_guard.py`): **38/38 passed**.
- Patcher integrity suite (`tests/unit/test_patcher_integrity.py`): **118/118 passed (0 skipped)**.
- Mandatory error handling framework (`test_error_handling_framework.py`): **16/16 passed**.
- Primary unit and integration suites (`test_validate_run.py`, `tests/unit/`, `tests/integration/`): **689 passed**.
- Root regression & notebook integrity suites: **110/110 passed**.
- Prompt regression suite: **17/17 passed**.
- Notebook & Graph Invariants:
  - `validate_notebook_integrity.py`: **PASSED (strictly 99 cells intact, 0 warnings)**.
  - `validate_graph.py`: **PASSED (15 nodes, 33 edges [21 conditional], 70 State fields)**.
  - `idd_bootstrap.py`: verified across all 7 canonical modules.
- Code Quality: `black --check` and `flake8` **100% clean**.
- Status: **CP4 RE-REVIEW HARDENING PASS COMPLETE — PR #163 READY FOR FINAL APPROVAL & MERGE GATE**.
<!-- session-curated:2026-10-15-cp4-review-resolutions:end -->

<!-- session-curated:2026-10-21-cp5-readiness-gate:start -->
## 2026-10-21 — Checkpoint 5 (CP5): Final No-Key Readiness Gate (Issue #165, PR #171 on PR #170 & PR #169)

### Milestone Objective
Establish the definitive, verified no-key production readiness gate for the Intelligent Data Detective pipeline prior to owner authorization for live execution spend ($0.00 spend incurred).

### Lineage & Merged PRs
- **Head SHA**: `281803b6681af98d0a20a7e0bbfc34c45a3407fb` (branch `main`).
- **PR #169**: Issue #164 real-kernel verification, checkpoint migration integrity, and evidence verifier (`tools/verify_issue_164_evidence.py`).
- **PR #170**: CP5-A fail-closed proof runner (`tools/proof_runner.py`), child-kernel provenance, and preparation CLI (`idd_proof.py`).
- **PR #171**: CP5 diagnostic harness and test suite polish (repaired Claim 12 production tool cases in `test_cp0_diagnostics.py`).

### Verification Baseline (Windows 11, Python 3.12.2)
1. **Notebook & Graph Invariants**:
   - `validate_notebook_integrity.py`: strictly 99 cells intact, 0 warnings.
   - `validate_graph.py`: 15 nodes, 33 edges (21 conditional), 70 State fields.
2. **Core Parity & Runner Unit Tests**:
   - `tests/unit/test_core_parity_guard.py`: **38/38 passed**.
   - `tests/unit/test_proof_runner.py`: **38/38 passed**.
   - `tools/verify_issue_164_evidence.py`: **5/5 passed**.
3. **Mandatory Error Handling & Diagnostics**:
   - `test_error_handling_framework.py`: **16/16 passed** (strictly blocking).
   - `tools/diagnostics/test_cp0_diagnostics.py`: **54/54 passed** (including repaired Claim 12 production tool cases).
4. **Broad Unit & Integration Suites**:
   - Unit, integration, validate_run suites: **732 passed, 9 skipped** (1 legacy Run 88 fixture, 8 NTFS permission/pyarrow isolation).
   - Core regression, memory lifecycle, prompt template suites: **127/127 passed**.
5. **Real-Kernel Smokes (No External API Keys)**:
   - `tests/smoke/test_scientific_kernel.py`: **3/3 passed**.
   - `tests/smoke/test_real_kernel_proof_runner.py`: **18/18 passed**.
   - `tests/smoke/test_real_kernel_notebook.py`: **21/21 passed**.

### Proof Runner Preparation CLI (`idd_proof.py`) & Fail-Closed Controls
- **Out-of-Tree Preparation**:
  - Executed in fresh attempt directories outside the source repo for both `retail_orders` and `sample_dirty` scenarios.
  - Exit code `0`, status `PREPARED_NO_PRODUCTION_EXECUTION`.
  - Zero production graph cells executed, `cp5_pass=False`, zero credentials loaded.
- **Negative Preparation Control**:
  - Verified that `KERNEL_IDENTITY_MISMATCH` triggers controlled exit `1` and preserves partial attempt evidence for forensics.

### Status & Recommendation
- **READY FOR OWNER AUTHORIZATION**. No live spend authorized ($0.00 spend).
<!-- session-curated:2026-10-21-cp5-readiness-gate:end -->

<!-- session-curated:2026-10-22-pr175-review-approval:start -->
## 2026-10-22 — Draft PR #175: Cell 57 Figure Metadata Order Remediation (APPROVED)

### Review Summary & Verdict
- **PR**: Draft PR #175 ("fix(report): initialize Cell 57 figure metadata before instruction")
- **Review Head SHA**: `f9bda27d252c54ce49d7814625b1325502ab76df` on branch `codex/cp5-cell57-expected-viz`
- **Base SHA**: `5d75b94ac2eba61737e05b588232c72f4f4a389a` on `main`
- **Verdict**: **APPROVED FOR MERGE**. Formal independent architectural review completed by Antigravity.

### Root Cause & Defect Analysis
During IDD CP5 Two-Run Live Proof Campaign Attempt 1 (`retail_orders`), execution halted at Cell 75 with `STREAM_FAILURE`.
Forensic investigation revealed an `UnboundLocalError` in notebook Cell 57 (`section_worker`):
- `default_instruction` f-string construction referenced `expected_viz_str` at line 2716 prior to the figure-metadata initialization block at lines 2723–2726.
- When section execution commenced, the local variable `expected_viz_str` was unbound, terminating worker execution before invoking `report_section_agent`.
- Campaign stopping rules took effect immediately: Attempt 1 evidence was safely preserved at `C:\Users\darf3\Documents\cp5_attempts\attempt_live_retail_orders_20261021_run1` and Attempt 2 (`sample_dirty`) was refused.

### Resolution Mechanism: Patch CP5-SW-FIGURE-ORDER
- In `_patch_notebook.py` (lines 11053–11064), the active section-scoped W13J-SW transformation cleanly swaps figure metadata preparation ahead of `default_instruction` formatting.
- In regenerated Cell 57, lines 2716–2719 now initialize/populate `expected_viz_str` before line 2721 constructs `default_instruction`.
- Preserves all figure titles, descriptions, types, paths, IDs, instruction composition, and compact-prompt behavior.
- Zero change to the other 98 cells; only Cell 57 source differs across the entire notebook.
- Regeneration is 100% byte-for-byte deterministic: SHA-256 `7844ac35e5edd5567fa68d47315b55e87e838692c6064dc43cecfc491e8986ec`.

### Test & Validation Evidence
- **Focused Regressions (`tests/unit/test_section_worker.py`)**: 8/8 passed.
  - Reproduces pre-fix state: 6 tests failed with `UnboundLocalError` on baseline code, 2 pass-through controls passed.
  - Post-fix state: all 8 tests pass cleanly.
  - Executed inside a real fresh Jupyter kernel: 8/8 passed (`GENERATED_CELL57_NO_KEY_KERNEL_PASS`).
- **Local Test Suites (225/225 passed across 6 suites)**:
  - `tests/unit/test_section_worker.py`: 8/8
  - `tests/unit/test_core_parity_guard.py`: 38/38
  - `test_error_handling_framework.py`: 16/16
  - `tests/unit/test_proof_runner.py`: 76/76
  - `tools/diagnostics/test_cp0_diagnostics.py`: 54/54
  - `test_validate_notebook_integrity.py`: 33/33
- **Broad Suites**: 1,003 unit/integration tests passed (0 failures, 3 expected skips).
- **Real-Kernel Smokes**: 47/47 passed (3 scientific + 23 proof runner + 21 notebook).
- **Invariants**: 99 cells intact, 15 nodes / 33 edges / 70 State fields preserved, canonical manifest SHA-256 digests 100% verified.

### Next Steps & Gating
1. Ready for owner merge into `main`.
2. Live spend remains strictly gated at $0.00 until owner re-authorizes the CP5 proof campaign on the merged revision.
<!-- session-curated:2026-10-22-pr175-review-approval:end -->

