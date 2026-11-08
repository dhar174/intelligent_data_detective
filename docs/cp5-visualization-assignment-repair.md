# CP5 visualization assignment and Python progress repair

Base: `b5a7e067abd409920ad638b2afeed85e2d749156` (merged PR #194).

## Evidence and causal limits

Fresh retail attempt `99a933ef-9313-4584-8bf6-66ce8013a063`, thread `thread-2212d7e8-78e4-4869-b0cd-cddf61807068`, stopped with native GraphRecursionError at recursion 160 in Cell 75 after 501.3 seconds, exit 1, cleanup CONTEXT_CLOSED. Its seven sibling pending visualization-result writes identify the absent result as `discount_rate_hist`. All eight actual VizSpecs have df_id retail_orders. A readable 1600x1200 discount_rate_hist.png and three tool-generated histogram variants survive in partial-work (11 PNGs total), although this worker never published native completion. Visual inspection finds overlapping annotations/legend, so file existence is not acceptance or proof of correctness. Three scatter workers completed, including gross_vs_units; the observed optional LOWESS dependency error does not establish the cause of the failed histogram. No evaluator, report, validators, or sample_dirty execution occurred. Original evidence/checkpoints remain untouched.

The failed inner worker messages are not persisted in the outer SQLite checkpoint. Public error evidence includes missing DataFrame/df Python globals. Do not claim exact failed tool arguments or infer a complete live causal sequence from the seven successful histories. The offline reproduction below establishes concrete message-contract and bounded-progress defects, not an exact replay of unavailable failed history.

The real generated worker reduces VizSpec to viz_instructions and visualization_id before rendering its request. Structured data-source and encoding fields are lost to that request. The Python wrapper has a fresh namespace on every call; an optional explicit df_id binds the canonical registry frame as df. Registry IDs do not create Python variable aliases. The previous model-facing description does not clearly state this per-call requirement. The actual visualization catalog binds the decorated StructuredTool with code/df_id arguments, not the internal query-only PythonAstREPLTool: there is no query/code schema mismatch.

## Narrow correction

The active patcher transformation changes generated Cells 44 and 57 only. The worker appends the complete canonical JSON VizSpec to its rendered runtime context. Its execution instructions and visualization-local Python tool description state fresh namespaces, explicit df_id on each data call, same-call imports/transforms/saving, and actual returned saved paths. Tool names, arguments, defaults, schema and other agents' Python descriptions remain unchanged.

A visualization-local native wrap_model_call middleware derives progress from checkpointed messages after the newest named runtime_context. It counts repeated known Python tool-error results, ignoring optional memory calls. Three consecutive redundant errors stop before another model call: one identical error therefore fails after its fourth tool occurrence, and alternating already-seen errors also reach a bound. A new error provides diagnostic progress; a successful non-memory result clears the streak and seen errors. There is no shared counter or new State field. The guard does not fabricate results or modify grades, artifacts, history, tool-call pairing or native completion. It does not claim to detect every possible nontermination pattern. Native VisualizationResults and strict owned-PNG checks remain required.

The first offline fixture revision omitted required nullable VizSpec fields and failed fixture validation; that was corrected before production edits. Corrected baseline: two failures (missing complete context and native GraphRecursionError instead of bounded error), two existing correction/reset cases passing. First green run exposed a guard reset defect (seen errors retained after successful calculation); fixed before broad validation. These development failures are retained separately.

New generated-factory/node regressions cover real registered data to actual readable PNG to native VisualizationResults and real join, explicit tool description, fourth identical error stop without completion, alternating errors, new diagnostics, corrected calls, successful-calculation reset, and new assignment on the same native thread. Existing strict failure, source ownership, revision, evaluator, cleaner-ID correction, checkpoint and graph suites remain applicable. Real-Jupyter smoke also executes the new native histogram and bounded-loop cases.

## Preserved boundaries and independent observations

No shared-hook, local Qwen, runtime-type, public API, canonical module, schema, State, reducers, graph, model IDs/settings/retries/timeouts, recursion limit, validators or production threshold changes. Strict IDD_ALLOW_RECOVERY=0 and 12/12 plus 9/9 remain required. Historical evidence is read-only. PR #129 remains deferred and issue #165 open. No paid process runs during the repair. Saved response usage subtotal is a partial USD 0.91929209, excluding embeddings and unsaved/in-flight responses; it is not a reconciled bill. Existing campaign limits are unchanged.

Separately observed, not repaired here: the Python wrapper replaces a frozen RuntimeCtx with RunnableConfig inside its local REPL namespace. A causal connection to this failed histogram was not established. Optional statsmodels absence affected a scatter attempt that later completed. Neither is silently included in this repair. All future CP5 acceptance still requires direct inspection of actual data, report, statistics and every real PNG; a green scripted regression cannot qualify a live report.

## Verification

Final commands, exit codes, test-count union, notebook hashes, hosted exact-HEAD checks and independent review are appended after verification. Publication and merge require exact-HEAD CI and a fresh actual Antigravity approval under the owner's standing authorization.

Current local receipts before publication: focused native/provenance/analyst/evaluator/report/revision integration selection **305 passed, six warnings**; patcher/parity/canonical state selection **211 passed, four warnings**; both notebook/graph validators, Black, Flake8 and git diff --check exit 0. Regenerated twice, exact bytes identical. Raw notebook SHA-256 `0dce59217243007312f76c36665629fd22a76a4f67ae121397176a5d6ff164ff`; LF-normalized `e4f4fa924b555b51026ae5c01f7931757a6b463a6105fdd2a7451c4d90c5e2b9`. Seven canonical modules, manifest and production validators compare unchanged to base. Broad supported no-key and all three real-Jupyter selections are still running at initial Draft publication; they are not claimed green here. Final exact-HEAD local/hosted/reviewer receipts will be posted in the PR conversation, without reusing an older-HEAD verdict.

Commands (all offline, recovery 0, no provider sockets):

```text
python -m pytest -o addopts= tests/unit/test_native_completion.py tests/unit/test_figure_provenance.py tests/unit/test_analyst_revision_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py -q
python -m pytest -o addopts= tests/unit/test_patcher_integrity.py tests/unit/test_core_parity_guard.py tests/unit/test_canonical_state_graph.py -q
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_error_handling_framework.py test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q
IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight python -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -v
python -m black --check tests/smoke/test_real_kernel_notebook.py tests/unit/test_native_completion.py
python -m flake8 tests/smoke/test_real_kernel_notebook.py --max-line-length=120 --extend-ignore=E203,W503,E501
python _patch_notebook.py  # twice; compare exact bytes and hashes
```
