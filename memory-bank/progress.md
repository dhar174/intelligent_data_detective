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
- [ ] Validator run and any warnings resolved
- [ ] Maintenance drift reviewed against future repo changes

## What works
- Core test suite: **22/22 pass** (`test_intelligent_data_detective.py`)
- Error handling tests: **15/16 pass** (1 known edge-case failure — acceptable)
- Memory test suites: all pass
- Notebook execution: functional end-to-end (requires `OPENAI_API_KEY`; 6–25 min)
- Agent stack: all guidance files present, managed sections safe for future maintenance runs

## What is incomplete
- No CI pipeline — tests must be run manually.
- `automation.instructions.md` is a scaffolded stub; no automated hooks exist yet.
- The one known test failure in `test_error_handling_framework.py` is an edge case in function signature handling and has no functional impact on the main system.

## Validation status
```bash
python3 -m pytest test_intelligent_data_detective.py -v          # 22/22
python3 -m pytest test_error_handling_framework.py -v            # 15/16
python3 -m pytest -v                                             # all suites
flake8 test_intelligent_data_detective.py --max-line-length=88 --extend-ignore=E203,E501
```

## Last meaningful update
[2026-04-20] Scaffolded and fully customised agent stack for IDD repo.
<!-- repo-agent-bootstrap:managed:end -->

<!-- session-curated:start -->
## Pipeline run history (current session)

| Run | Outcome | Notes |
|---|---|---|
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
| 88 | + W10-PDF-POST — **8/8 GREEN** structurally, **HOLLOW** semantically | All structural gates pass; PDF emitted (1983 B). But report = 356-char placeholder, 0 sections, 1 viz embedded 5×, 25+ stub marker files in reports dir. **Potemkin pipeline.** Analyst output was rich and correct — content lost between analyst and report. Triggered Phase 6 pivot. |

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

### Phase 6 milestones
- [x] Run 87 structural pass (7/8) — W9-SR-DROP verified
- [x] Run 88 structural pass (8/8) — W10-PDF-POST verified
- [x] Wave 5 cutover (direct-to-notebook) + backups at `_wave5_backup_20260423-091557/`
- [x] GitHub epic + 7 sub-issues filed (#119 epic; #112–#118 phases A–G)
- [x] Forensic agents launched (`forensic-pipeline`, `forensic-sr-persistence`) — results stranded; re-harvest pending
- [ ] Phase A — forensic confirmation of RC1–RC5 (re-launch or direct inspection)
- [ ] Phase B — telemetry-only instrumented run
- [ ] Phase C — viz pipeline fix
- [ ] Phase D — report pipeline fix (`Section.body` min-length validator, `written_sections` reducer audit)
- [ ] Phase E — `file_writer` prompt rewrite + tool-list restriction + tool-call cap
- [ ] Phase F — supervisor FINAL gate content-validation preconditions
- [ ] Phase G — validation against 12-criteria (expect 5–8 convergence runs)

### Known artifacts to inspect
- `IDD_results/IDD_run_*-20260423-*/report.html` — verify 356-char placeholder
- `IDD_results/IDD_run_*-20260423-*/*.pdf` — verify ≈1983 bytes
- `IDD_results/IDD_run_*-20260423-*/*_{ack,commit,ready,stub,trigger}*` — count stub files
<!-- session-curated:2026-04-23-phase6:end -->

