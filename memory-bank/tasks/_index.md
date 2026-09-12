<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Tasks Index

## In Progress
- [MAINT-001] Keep W14 completion context current - Update docs/memory when notebook workflow or proof baseline changes

## Pending
- [FOLLOWUP-001] Future prompt/report polish - Only with a fresh proof that preserves W14 gates
- [FOLLOWUP-002] Data-cleaner defensive hardening - Resolve #121 through `_patch_notebook.py` without treating it as a current completion blocker

## Completed
- [BOOTSTRAP-001] Repository agent stack scaffolded and customized
- [W14-001] IDD v5 completion baseline proved and pushed (`IDD_run_run_default_id-20260504-1338-b3079aea`)
- [W14-002] Final validators established (`validate_run.py` 12/12, `validate_artifact_quality.py` 9/9)
- [W14-003] Stale Phase 6 GitHub issues #112-#119 closed after W14 proof
- [CI-001] No-key CI workflow added for validator/unit/integration tests and targeted lint
<!-- repo-agent-bootstrap:managed:end -->

<!-- session-curated:start -->
## Open — pipeline debugging (current session)

- [WAVE-4] Apply prepared `_patch_notebook.py` sentinels: **W2-EMERGENCY** (EMERGENCY_MSG outgoing edge), **W2-BF2** (analyst fix), **W2-BF6** (agent factory fix), **W2-REC6** (fast-fail on unknown tool name).
- [RUN-77] Launch full pipeline run after Wave 4. Use `python run_notebook_live.py --resume` **only if State schema is unchanged**; otherwise delete `checkpoints.sqlite` and run from scratch.
- [WAVE-5] Fold ✅ WORKING `_patch_notebook.py` sentinels into the source notebook permanently and retire the corresponding patcher entries. Triage table: `patcher-audit.md` (session state).
- [VIZ-EVAL] Investigate `viz_evaluator` silence in Run 76 (≥16 min, no logs). 12× final-hop `with_structured_output` recovery hits in Run 75 suggest a structured-output retry loop. Verify `VizFeedback` schema satisfies `BaseNoExtrasModel` required fields.
- [CELL-48] Resolve `structured_response` channel-type collision warning emitted at cell 48 of the notebook.
<!-- session-curated:end -->

