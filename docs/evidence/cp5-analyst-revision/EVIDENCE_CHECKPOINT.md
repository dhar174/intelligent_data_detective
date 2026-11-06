# CP5 analyst revision repair — evidence checkpoint

Status: **local documentation, evidence extraction, workflow gate, and verification complete** · Evidence state: **partial** (hosted CI and external review pending; live acceptance remains stopped)

## Slice card

- **Goal:** produce a sanitized forensic report for the failed analyst revision turn and add the Windows CI gate for the revision completion regressions.
- **Parent task:** offline repair based on `88bac1c4c0b5687ceaee9cc01e9514270380214f`, within Issue #165. The parent task keeps merge, issue closure, and paid campaign execution stopped.
- **Files owned here:** `docs/cp5-analyst-revision-repair.md`, this directory's evidence JSON and checkpoint, and `.github/workflows/real-kernel-windows.yml`.
- **Boundary:** do not edit another agent's `red.txt`, tests, notebook, or canonical schemas. Do not publish prompts, model reasoning, credentials, or raw logs.
- **Verification:** read back the report, JSON, checkpoint, and workflow; inspect the exact diff. Root owns the regression test results.
- **Stop:** no merge, Issue #165 closure, model review invocation, live retry, or paid campaign continuation.

## Baseline and worktree

- Worktree: `codex/cp5-analyst-revision`, pinned at `88bac1c4c0b5687ceaee9cc01e9514270380214f` when this slice began.
- Project baseline refs read: `memory-bank/activeContext.md` and `memory-bank/progress.md`.
- Parent reports the pre-change baseline as 111 passing tests with two dependency warnings. A separate mandatory error-handling run passed 16 tests in 2.16 seconds; the later grouped 198-test command also includes that suite, so the separate count must not be added to the grouped total.
- Root's candidate adds `_apply_cp5_analyst_revision_patch` inside the active native-completion transformation. Root owns production code and test results; no canonical schemas are in scope.
- Independent review identified a current-source `llm_input_messages` compatibility concern in LangChain 1.4.6; it is separate from the live recursion failure and outside this repair. No shared pre-model hook is being changed.

## Completed evidence reads

- The archived attempt bundle associates Attempt 1 with source SHA `88bac1c4c0b5687ceaee9cc01e9514270380214f`, tree `6e6bff4f9e06d052466831dfeadf7d948cb8f731`, and a 99-cell notebook whose LF hash matches the bundle.
- The read-only SQLite checkpoint store passes `PRAGMA integrity_check`, has 13 outer-thread checkpoints for `thread-3a766093-d6bc-451e-804e-1266ea655012`, and covers metadata steps -1 through 11. Its accessible message writes do not contain the inner revision tool-call identifiers.
- Cell 75 output contains 14 `list_visualizations` and 34 `get_visualization` name occurrences overall. From the first revision marker to the end, it contains 13 and 26 respectively, plus one `get_column_names` occurrence. No raw stream text was copied.
- All three original evidence-file raw and LF hashes recorded in `evidence.json` match the archived files.
- The sanitized `forensic-evidence.json` is written. It records 27 deduplicated outer-thread message tool calls with IDs, `ai` roles, safe allowlisted argument values, canonical argument fingerprints, and checkpoint observations. It separately labels 39 revision retrieval-header occurrences and one column-name header; the failed inner calls have no captured IDs, arguments, or message roles.
- The hash-only `retrieval-trace.json` records 40 revision header events. It identifies the earliest repeated rendered `get_visualization` result at header 1043/display step 41 as matching header 1015/display step 33, with a different result in between. The fingerprints establish repeated rendered output only, not equal arguments, invocation identity, or the cause of nontermination.
- The controlled `effective-contract.json` readback confirms that genuine feedback is visible before and after; the repair removes the positive `respond` instruction, adds one runtime-context user message, and includes prior findings while keeping `AnalysisInsights` bound. The factory-level `any` choice is normalized by installed `ChatOpenAI.bind_tools` to `required` before canonical SDK serialization. This probe made no provider request and does not recover the missing failed request payload.
- A safe scan found 7 `warning`/`warn` token occurrences in the archived console log and none in the pipeline log; no warning text was copied or classified.
- Open PR #129 has no direct runtime target overlap with analyst revision completion. Its patcher pass scans all code cells for `summary_text` concatenation and clears outputs on rewritten cells, so a later PR #129 integration should regenerate and recheck the notebook.
- The Windows workflow gate now includes native completion, visualization evaluator completion, the new analyst revision completion test, and the CP5 revision-route integration test with API keys blank and recovery disabled.
- The owned report, JSON, checkpoint, and workflow were read back. The workflow parses as YAML and the new gate names all four existing/new suites; the tracked workflow diff contains only this gate. Root reports notebook integrity and graph-structure validation passed, with the graph soft-budget warning recorded.
- Root's grouped no-key command passed 198 tests with 3 dependency warnings in 16.58 seconds; this includes the mandatory error-handling suite and is counted once. Separate reported results: patcher integrity 118 passed with one `fitz` `FutureWarning` in 24.58 seconds; native completion 107 passed with 2 dependency warnings in 30.62 seconds; new analyst revision completion 14 passed with 2 dependency warnings in 19.68 seconds; new revision-route case 1 passed with 4 dependency warnings in 10.13 seconds (included in a broader run, not additive); full local scientific/notebook/proof-kernel suites 55 passed with no warnings or skips in 482.24 seconds. One earlier added-kernel test passed in 18.02 seconds and is not a full kernel gate.
- The exact analyst test command was confirmed by its owner: `python -m pytest tests/unit/test_analyst_revision_completion.py -q --maxfail=1`. The full local kernel gate used `IDD_RUN_KERNEL_SMOKE=1`, `IDD_KERNEL_NAME=idd-preflight`, and blank provider keys; it is not hosted CI.
- Root reports the frozen local candidate notebook hashes as raw `88356e218fb6645fc7ec8f17598e1218c12c1898ed10993c50f83b51cd1b0804` and LF `7614c5f342b9033d97f4ea4c392a559d7fb312b85fafb3063b4bed2a50a109cc`; source cells 25, 44, and 57 changed. Seven canonical modules and `idd_manifest.json` match the pinned base after LF normalization. These are local snapshot hashes, not a commit or hosted provenance record.
- A supplemental replay on unchanged pinned base `88bac1c` was captured after implementation and produced two PNGs, 160 graph steps, and 40 cycles across both patterns. Its harness initially missed `plt`, `ALLOWED_SPEC_KEYS`, `hashlib`, `base64`, and `glob`; an initial 160-model-call assertion was corrected to 40 cycles. These were harness setup/assertion issues, not production defects. Earlier `red2` predates implementation and remains untouched.
- Final broad no-key suite: `python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_intelligent_data_detective.py test_error_handling_framework.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q` passed 1164, skipped 9, with 6 warnings in 143.45 seconds. Four warnings were dependency warnings (`langchain_experimental`, `trustcallSend`, `langchain_community`, `fitz`); two were Matplotlib `set_ticklabels` warnings from real-plot integration. Provider key variables were blank; recovery was disabled.
- Combined focused gate: `python -m pytest -o addopts= tests/unit/test_analyst_revision_completion.py tests/unit/test_native_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py -q` passed 168 with 4 warnings in 81.93 seconds. This overlaps the separate test results above and is not additive.
- Narrow skip audit: `python -m pytest -o addopts= test_validate_run.py tests/unit/test_registry.py -q -rs` passed 94, skipped 9, in 11.44 seconds. Skips: one missing Run88 fixture, two missing `pyarrow`, and six POSIX permissions/ownership cases not applicable on Windows.
- Hosted CI and external review remain pending; the local broad suite and local real-kernel suite are not hosted results. Root owns integration/commit and any later hosted handoff.
- The approved A–H acceptance concepts and concrete focused test mapping are recorded in [cp5-analyst-revision-repair.md](../../cp5-analyst-revision-repair.md); the SQLite integration test overlaps B, F, and H, so counts are not one-to-one.

## Next steps

1. Hand the read-back report/checkpoint and scoped file diff to root for integration and commit review.
2. Keep hosted CI, external review, push/PR, campaign retry, and Issue #165 closure pending their separate authorized steps.
3. Preserve the evidence limits: no raw failed provider payload or inner-call IDs/arguments/roles; do not turn the failed live attempt or controlled probe into CP5 acceptance evidence.

## Drift check

The slice remains within the offline forensic/documentation/CI scope. No new owner, schema, persistence surface, or provider execution has been introduced. Decision: **local slice complete; external verification pending**. No CP5 completion or acceptance claim is made.
