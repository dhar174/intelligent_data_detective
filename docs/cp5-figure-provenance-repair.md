# CP5 figure-provenance repair

Base: `ffd44169e49ddb11158bef99104334e2fb792a36`, freshly fetched main.
Implementation branch: `codex/cp5-figure-provenance`, isolated worktree.
Implementation and regression checks are offline. One later owner-authorized live attempt of the initial candidate is recorded below; it failed and the campaign stopped. Historical evidence, strict validators, issue state and merge authority remain protected.

## Verified failure and baseline

Original retail attempt `56ff5d74-a3f7-4a42-b422-48df420d35e1` at the base revision reached a genuine acceptable visualization verdict, then failed in `report_packager_node` because `retail_orders_net_revenue_vs_gross_revenue.png` did not exist. The revised worker had reported that intended filename after failed custom plotting attempts even though a fallback scatter tool created a differently named real PNG. The actual filename was confined to ToolMessage.artifact, which native Responses serialization omits from model input.

The worker checked completion flags but published the nonexistent path and wrote it to memory. Join checked structured metadata strings without validating the file; its path-only duplicate shortcut bypassed the later filesystem checks. Report planning propagated the existing bad record into sections. The packager correctly rejected five references to the one nonexistent filename. This repair leaves the packager unchanged.

The initial baseline probe loads the frozen base notebook from Git into the existing offline loaders without changing the on-disk candidate. It runs the 31 new cases: **28 fail and 3 pass**, exit 1. Those failures establish missing visible paths, false worker completion, invalid structured inventory, conflicting identities, unsafe references, unfinished duplicates, and the failed-custom/successful-fallback boundary. The three passing cases are existing negative controls. These initial fixtures invented run_dir and did not qualify the real RuntimeCtx contract; the replacement coverage below supersedes that assumption. Receipt: `figure-provenance-baseline-final.txt` under the local CP5 attempts directory.

Initial probes also exposed test-loader omissions: `typing.io` shadowed stdlib `io`, and the Python REPL fixture lacked real `ast`, `textwrap`, and `builtins` imports. These were corrected in the offline loader to match production imports. Early results containing those fixture failures are retained but are not used as the definitive red evidence.

## Correction

The active `_patch_notebook.py` transformation changes only generated Cells **32 and 57**:

- Successful histogram, scatter, box, and violin results now include compact JSON model-visible metadata with the actual image_path and relevant plotting fields. The native `(content, artifact)` contract, names and argument schemas stay intact. Image bytes/base64 are excluded. Failure returns remain failures.
- Histogram and violin saves now use the existing canonical `_resolve_artifact_path`, as scatter and box already do. Their legacy save location could otherwise violate the current-run checks. Rendering semantics and model settings are unchanged.
- The worker receives an explicit actual-saved-path contract and the concrete owned current-run directory, and keeps its assigned visualization ID. Before any visualization-memory write or state publication, every output asset must resolve through canonical path safety, remain inside RUNTIME.artifacts_dir, exist as a nonempty file, and verify as a readable PNG.
- Relative references retain the existing explicit artifacts_path policy. URI, traversal, stale/outside-run and symlink escapes are rejected without creating directories. No disk discovery, guessed substitution, renaming, or ID rewriting occurs.
- Join validates structured records, envelopes, and path-only references before deduplication. Unfinished records cannot hide behind duplicates. Equivalent ID/path bindings still deduplicate; contradictory ID/path bindings raise a controlled error.

No canonical module, manifest, State field, graph node/edge, schema, model setting, SDK retry, revision limit, shared hook, evaluator, safe invocation wrapper, production validator or packager is changed. Static comparisons confirm those functions/modules remain identical to base. The inactive legacy save_viz_for_state helper is not used as the repair boundary.

## Initial candidate verification (d744905; runtime qualification superseded)

Final completed local selections (counts overlap unless stated otherwise):

```text
python -m pytest -o addopts= tests/unit/test_figure_provenance.py tests/unit/test_viz_join.py tests/unit/test_native_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_cp5_report_grounding.py tests/integration/test_viz_join_reporting.py tests/integration/test_cp5_revision_route.py -q
# exit 0: 251 passed, 6 warnings, 178.01 seconds (focused run; subsequently included in the final broad rerun)

python -m pytest -o addopts= tests/unit/test_patcher_integrity.py tests/unit/test_core_parity_guard.py tests/unit/test_canonical_state_graph.py test_error_handling_framework.py tests/unit/test_proof_runner.py tests/unit/test_live_runner_configuration.py -q
# exit 0: 316 passed, 4 warnings, 60.83 seconds; includes all 16 mandatory errors

python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration -q
# exit 0: 1083 passed, 9 skipped, 8 warnings, 245.88 seconds (final broad rerun after explicit owned-directory delivery)

python -m pytest -o addopts= test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q
# exit 0: 127 passed, 22.72 seconds

python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
# exit 0: 99 cells, 42 code cells compiled

python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
# exit 0: 15 nodes, 33 edges, 70 fields, 39 reducers; 12.88 seconds exceeded its nonblocking 10-second soft budget
```

The broad selection contains the 251 focused and unit portions of the 316 integrity cases. Root/prompt 127 and mandatory-error 16 cases are separate from the broad selection: **1,226 distinct passing non-kernel cases**, plus 9 skips. Repeated invocations and focused subsets are not added again.

Coverage includes actual decorated plotting tools and native Responses payloads, real native factory completion after a custom plotting failure, preserved assigned IDs despite different filenames, rejection before memory publication, corrupt/empty/missing assets, owned absolute/relative paths, URI/traversal rejection, conflicting identities, unfinished duplicates, and existing generated-function SQLite revision/reporting continuity. Existing downstream missing-report-figure checks remain strict.

The real-Jupyter smoke suite adds a generated-worker provenance case. Intermediate selections exposed legacy diagnostic assertions, structural fixtures claiming a nonexistent chart, a test-only indentation error introduced while updating an assertion, and a verification race when the graph compiler created a checkout-local SQLite file during an isolation probe. The embedded probe indentation was corrected and compiled, structural fixtures now create a genuinely owned PNG, and the final kernel run was sequential with no concurrent validator activity. Isolation assertions and production rejection checks remain intact.

```text
IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight python -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -v
# final exit 0: 56 passed in 562.75 seconds
```

The final kernel selection covers scientific imports, both graph/checkpointer paths, shared/separate-main plotting, generated native provenance and analyst revision, migration/security negative controls, and all proof-runner kernel checks. Earlier full/targeted kernel failures are retained in local receipts; none remains unresolved. Across the disjoint broad, root/prompt, mandatory-error and kernel selections, **1,282 distinct cases passed**, with 9 broad-selection skips. Focused runs and reruns are not added again.

Warnings observed are dependency deprecations and Matplotlib tick-label warnings. The 9 broad skips are reported as skips, not passes. No claim of live qualification follows from these checks.

Two full patcher regenerations produced identical notebook bytes, including after formatting the new transformation with an AST-equivalence check:

- Windows raw SHA-256: `673e29cdc7cde559b82109bbd64cb4bb486085c55133f88e42e601357a68638a`.
- LF-normalized SHA-256: `5344e0f42b00f01aa3788d0acd3a117e2a086127d73161a3694508463b8abd05`.
- Cells changed: `[32, 57]`. Seven canonical modules, manifest and both production validators match base after LF normalization.

Final readback reconfirmed the hashes above and two further byte-identical regenerations. A final validator wrapper initially failed from an external temporary working directory (a relative script path, Windows output decoding, then the graph validator lacking its checkout-relative sample prompt). These wrapper attempts are not counted as passing gates. Re-running the documented commands from the isolated checkout with UTF-8 output passed: notebook integrity exit 0 and graph compilation exit 0, 15 nodes/33 edges/70 fields/39 reducers in 10.26 seconds (nonblocking soft-budget warning). The final kernel suite was already finished, so no isolation probe ran concurrently. No validator behavior was modified.

The existing Windows no-key workflow explicitly includes the new provenance module. Hosted Linux/Windows CI and independent review remain pending publication of a committed candidate; no hosted result is claimed for this local change.

## Owner-authorized live failure and offline runtime-contract correction

After initial publication as Draft PR #189 at d744905f901eb6b0a0c62d28e2f66085f49e37e1, Linux workflow 42447367119 and Windows workflows 42447368768/42447320461 passed. Antigravity initially approved that exact HEAD by static inspection (zero independent tests). Those receipts are historical; they did not qualify the production runtime contract.

The owner explicitly approved another live run. Fresh retail_orders attempt fca1b092-94e7-415a-a096-28cf055c3a3b on thread thread-870f7150-3916-4bb0-9961-3c94ea7b6cdd ended with runner exit 1 after 370.9 seconds. Cleaning and analysis completed natively. Visualization stopped in the new guard with `ValueError: Visualization asset has no current run`; Cell 75 recorded STREAM_FAILURE. Both validators were NOT_REACHED, CP5 is not accepted, and sample_dirty was not started. All 99 executed cell sources match that frozen candidate. Six PNGs were preserved. Kernel cleanup is CONTEXT_CLOSED; runner 43044 and kernel 40380 were verified absent.

The introduced defect was a runtime contract mismatch: generated Cell 51 RuntimeCtx is frozen and exposes the per-run root as `artifacts_dir`, not `run_dir`. The worker prompt and guard read the nonexistent field. Offline fixtures invented `SimpleNamespace(run_dir=...)`, hiding the defect. Antigravity independently confirmed the mismatch and withdrew its initial approval. No source or evidence from the failed attempt was rewritten.

The same focused repair now uses `_cp5_current_run_root()` to read authoritative `RuntimeCtx.artifacts_dir` for both the worker prompt and validation. Missing roots fail before any model request. Relative references use an explicit state artifacts_path when supplied, otherwise that same runtime root. The redundant legacy path-only run_dir/file checks are replaced by the already stronger shared validator; canonical containment, URI/traversal rejection, nonempty readable-PNG checks, completion flags and ID/path conflict checks remain active.

Offline loaders instantiate the actual generated frozen Cell 51 RuntimeCtx class. Plot success tests use production directory topology `base/artifacts/run-fixture/visualizations` with no IDD_ARTIFACTS_DIR override. Structured and path-only inventories cover relative paths with and without state artifacts_path, stale runs, and unavailable runtime roots. Worker negatives assert the specific missing/corrupt/outside failure rather than passing on any runtime error. Real-kernel probes exercise the real runtime class through native plotting/worker completion and path-only join.

The corrected real-runtime subset against d744905 before the production edit produced **6 failures and 1 pass**, exit 1 (runtime-contract-red-final.txt). An earlier fixture draft attempted to mutate the frozen runtime, and a first green draft treated a native Command as a dict; both test-only mistakes were corrected and their failed receipts retained. The seven additional cases make 38 provenance cases. The focused selection passes **258 tests, 6 warnings, 109.77 seconds**, exit 0.

```text
python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_error_handling_framework.py test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q
# exit 0:1,233 passed, 9 skipped, 8 warnings, 181.50 seconds; all 16 mandatory errors included
```

The latest notebook/graph validators pass: 99 cells, 42 compiled code cells, 70 fields, 39 reducers, 15 nodes, 33 edges; graph compile 8.61 seconds. Two regenerations are byte-identical. Only generated Cells 32/57 differ from the verified base; Cell 51 RuntimeCtx, canonical modules, manifest, model/hook/evaluator/packager/validator behavior and graph remain unchanged.

Latest notebook raw Windows SHA256: `2f73c94442ea51f68bdfcd9b80b6d8fd3e90a576b8992979b61e4b6a371988a8`; LF-normalized SHA256: `9598cd072f926cecf888d5ea2ce8e43d77786c7a3a17c909f6c7136b97e9d14a`.

```text
IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight python -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -v
# exit 0: 56 passed in 405.52 seconds (new real-runtime candidate)
```

The disjoint current combined no-key and kernel selections provide **1,289 distinct passing cases, 9 skips**. The 258 focused passes, earlier candidate receipts and reruns are not added again. Hosted Linux/Windows and Antigravity receipts must be checked against the corrected PR HEAD; older successful runs do not qualify it.

Recovered checkpoint-usage estimate for the stopped live attempt: $0.04001269; carried campaign estimate: $0.47324310. These are partial numeric saved-usage estimates, not reconciled provider bills. Embeddings, unsaved worker responses, retries and in-flight charges remain unresolved. No further paid IDD requests or replacement live run followed the failure. The corrected candidate requires new exact-HEAD hosted CI and independent review; those receipts belong in the PR conversation. The first candidate's approval cannot be reused for the corrected HEAD.

## Separate finding and stop conditions

The existing heatmap renderer fails in an isolated valid-data probe with undefined `cols`. Its AST remains unchanged. It is documented in [cp5-heatmap-rendering-followup.md](cp5-heatmap-rendering-followup.md), not silently repaired or hidden with a fixture global. Only the four verified working plotting tools are claimed by the success-path regressions.

One owner-authorized first-candidate live attempt failed as described above. The campaign is stopped; the corrected candidate has only offline verification and has not been retried live. Further live execution requires a fresh owner decision after the new candidate receipts. #129 remains deferred and #165 remains open. Live figure semantics, the eventual final report, and the two production acceptance thresholds (12/12 and 9/9) remain unqualified by this offline repair.
