<!-- repo-agent-bootstrap:file-kind=memory-bank -->
# Decision Log

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
