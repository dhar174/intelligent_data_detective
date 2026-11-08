# CP5 visualization worker scope and retrieval progress repair

Base: `4e5e9160852573a5013177947ffcdc27ef1ff7fe` (merged PR #196).

## Verified failure and causal limits

Fresh retail attempt `c4909864-3a26-4445-8357-b34c601f8507`, thread `thread-6e20b4bd-2206-4e49-9592-ff2938be230a`, exited 1 after 579.5 seconds with Cell 75 STREAM_FAILURE / native GraphRecursionError at limit160. Cleanup CONTEXT_CLOSED. No evaluator/report/validators or sample_dirty run. Six of eight native worker results persisted as pending outer writes; missing native IDs were viz_units_vs_gross and viz_price_vs_gross. Their PNGs exist, but file existence does not establish native completion or acceptance. Failed inner messages are not persisted; do not claim an exact replay or per-worker call sequence.

Allowlisted public tool-message headers in the whole live log show 45 get_visualization, four list_visualizations and 21 read_file events. These are logged events across workers, not fresh invocation counts or proof that each result is identical. Labels reach step161 alongside the per-invocation160 recursion budget; displayed steps retain inner checkpoint offsets. Seven VisualizationResults header events are not seven successful worker returns: only six worker writes are persisted. The private model content and image payloads are excluded from the sanitized packet.

Actual cleaned CSV: 180 rows, zero duplicates, zero missing values; exact equality with raw median-fill then deduplicate. Discount median0.1445 differs from deduplicate-first0.146 because operation order differs. Native analyst uses correct180 rows and matching statistics. Cleaner narrative incorrectly says176; retained separately, not silently repaired. No final report was produced or qualified. Numeric-conversion 1D warnings, the independently observed REPL RuntimeCtx replacement, and optional LOWESS dependency remain outside this repair.

PR #196 correctly delivered the full spec and fresh-Python binding description, with a guard for known Python errors. This live failure follows a different retrieval pattern, with no such Python error prefixes in the sampled public error-class scan. That narrower guard did not solve the remaining scope conflict or bound retrieval loops. Do not reuse its offline success as a live pass.

## Effective request before and after

Before: the actual generated worker renders a legacy open-ended persistence/perfection template. Its Human runtime_context includes one spec, while an AIMessage named supervisor also tells each worker: Please generate all requested visualizations for the cleaned dataset. The original full analysis/report request is another active user turn. The template insists on continuing through uncertainty and restarting if every rubric criterion is not met. These competing scopes are verified in a real generated-factory model spy. They are a concrete instruction defect; absence of failed inner history limits claims about which exact sentence drove a specific live call.

After: retain the existing native factory system and ToolStrategy(VisualizationResults), but construct a compact named Human runtime_context with separate original_user_request, pipeline_assignment, complete visualization_task/spec, data description, analysis insights, registered IDs, owned run directory, memories and emergency assignments. A named Human visualization_assignment explicitly assigns only this worker's figure. It requires real data, actual saved path/ID, verification and correction of concrete issues, then native completion; the canonical join/evaluator assesses the full set and may request genuine revisions. Incomplete required output remains finished_this_task=False and fails existing strict publication checks. The legacy template remains available but no longer supplies this active worker request. Emergency rendering uses the same scoped boundary.

The visualization-local model middleware retains the known-Python-error guard and additionally counts unchanged retrieval results for list/get_visualizations, read_file/list_available_files and schema/column lookup. Results are paired with actual assistant tool calls by call ID/name. Fingerprints hash complete content/artifact (including changed preview bytes without publishing them). Three consecutive redundant results cause an actionable ValueError before another model call: fourth identical lookup, or fifth alternating A/B lookup. New result/preview/page information resets the redundant streak; successful non-memory non-retrieval work clears retrieval history. Optional memory calls and failed work do not manufacture progress. Tracking is derived from messages since the newest runtime_context, without counters in shared state. No result, grade, approval, PNG or path is synthesized.

This is a targeted no-progress bound, not a proof that every possible loop terminates early. New/changing outputs and successful non-retrieval calls can continue legitimately; native recursion160 remains unchanged as the final bound.

## Offline red/green and boundaries

Before production edits: seven new generated-factory/node cases yielded four failures and three passes. Scope spy sees the pipeline-wide AI supervisor assignment; repeated nonempty inventory/detail and alternating retrievals reach native GraphRecursionError160 instead of a controlled error. The scope assertion was strengthened to test the conflicting active assignment directly, rather than merely JSON format; it failed on the unchanged baseline. After repair all seven pass. Existing real registry to actual histogram PNG to native completion/join and strict failures remain covered. New cases also exercise real preview blocks, different paging results, successful work and new assignments on the same native thread. Real-Jupyter smoke executes the scoped native completion, nonempty retrieval stop and legitimate preview/paging progression.

Generated changes are restricted to Cells44/57. No seven-module/manifest, State/schema/reducer, graph, shared-hook, local-Qwen-hook, runtime type, public tool API, model/reasoning/timeout/retry, artifact ownership, native completion validator or production threshold change. 99 cells,70 State fields,39 reducers,15 nodes,33 edges. Strict recovery0,12/12 and9/9 preserved. Historical evidence untouched; no paid process during this repair. #129 deferred and #165 open. Carried saved-response subtotal USD0.98579967 is partial (22 saved responses vs130 HTTP success occurrences in this attempt; embeddings and unsaved responses unresolved), not a reconciled bill; limits unchanged.

Local full focused/parity/no-key/kernel, regeneration and validators are running at report creation. Exact-HEAD results, warnings/skips, distinct test-count union, hashes, hosted Linux/Windows and actual fresh independent Antigravity review will be recorded in PR conversation before any merge or retry. No pending gate is claimed green.


The first full focused selection yielded345passes and one integration harness failure: RuntimeTaskChatModel split the old plaintext marker at a physical newline, so it read the remainder of JSON-escaped context as an ID. Production had returned the correct complete spec. The controlled adapter now decodes runtime JSON and validates assigned_spec as the real VizSpec, retaining all exact-ID, reducer, provenance, SQLite reopening, two-thread isolation and genuine revision/acceptance assertions. This is a format adaptation, not an assertion waiver. That failed selection is archived separately; all final gates must pass after the change. The compact runtime now carries assigned_spec as a real JSON object in addition to the explicit task text.

Own pre-publication review found that a new retrieval result should also clear the earlier known-Python-error streak. Corrected and covered with a real new-information retrieval between failing Python calls. The owned offline validation batch was deliberately stopped before publication to avoid certifying mixed source; no final pass is claimed from that interrupted batch. Final gates rerun on unchanged candidate source.


Final candidate pre-publication receipts: **15 passed** (14 assignment/progress/native cases plus the real SQLite revision cycle), four warnings; fresh patcher/parity/state **211 passed**, four warnings; notebook and graph validators, Black, Flake8 and git diff --check pass. Full focused/broad and all three real-Jupyter suites remain pending at initial Draft publication; final exact-HEAD receipts will be posted before merge. These selections overlap and must not be summed as distinct cases.

Two final regenerations are byte-identical. Raw notebook SHA-256 `1be2e83d96c579f452eb43f7f9bc5f5fd98990ced9bf445478b1f5f2cff0c6b3`; LF-normalized `7c34119ed084f041ce51a9875c5a02c263aaac8f4eb8cbf759bf3eb750b4e93b`. Canonical seven modules, manifest and both production validators compare unchanged to base.

Commands (offline, recovery0; kernel env IDD_RUN_KERNEL_SMOKE=1 / IDD_KERNEL_NAME=idd-preflight):

```text
python -m pytest -o addopts= tests/unit/test_native_completion.py tests/integration/test_cp5_revision_route.py -k '<assignment/progress cases or sqlite_generated>' -q
python -m pytest -o addopts= tests/unit/test_patcher_integrity.py tests/unit/test_core_parity_guard.py tests/unit/test_canonical_state_graph.py -q
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
python -m pytest -o addopts= tests/unit/test_datetime_conversion.py tests/unit/test_native_completion.py tests/unit/test_figure_provenance.py tests/unit/test_analyst_revision_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_viz_join.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py -q
python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_error_handling_framework.py test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q
python -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -v
python -m black --check tests/smoke/test_real_kernel_notebook.py tests/unit/test_native_completion.py
python -m flake8 tests/smoke/test_real_kernel_notebook.py --max-line-length=120 --extend-ignore=E203,W503,E501
python _patch_notebook.py  # twice, compare exact bytes/hashes
```

The exact long -k expression and all argv/exit codes are retained in local command receipts. Final hosted/reviewer/merge receipts belong to the exact candidate SHA; no previous-HEAD approval substitutes for them. No further live retry is qualified by this report alone.
