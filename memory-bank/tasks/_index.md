<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Tasks Index

## In Progress
- [BOOTSTRAP-001] Maintain repository agent stack - Keep managed guidance aligned with the codebase

## Pending
- [BOOTSTRAP-002] Evaluate third-party agent assets - Vendor only pinned, licensed sources

## Completed
- none yet
<!-- repo-agent-bootstrap:managed:end -->

<!-- session-curated:start -->
## Open — pipeline debugging (current session)

- [WAVE-4] Apply prepared `_patch_notebook.py` sentinels: **W2-EMERGENCY** (EMERGENCY_MSG outgoing edge), **W2-BF2** (analyst fix), **W2-BF6** (agent factory fix), **W2-REC6** (fast-fail on unknown tool name).
- [RUN-77] Launch full pipeline run after Wave 4. Use `python run_notebook_live.py --resume` **only if State schema is unchanged**; otherwise delete `checkpoints.sqlite` and run from scratch.
- [WAVE-5] Fold ✅ WORKING `_patch_notebook.py` sentinels into the source notebook permanently and retire the corresponding patcher entries. Triage table: `patcher-audit.md` (session state).
- [VIZ-EVAL] Investigate `viz_evaluator` silence in Run 76 (≥16 min, no logs). 12× final-hop `with_structured_output` recovery hits in Run 75 suggest a structured-output retry loop. Verify `VizFeedback` schema satisfies `BaseNoExtrasModel` required fields.
- [CELL-48] Resolve `structured_response` channel-type collision warning emitted at cell 48 of the notebook.
<!-- session-curated:end -->

