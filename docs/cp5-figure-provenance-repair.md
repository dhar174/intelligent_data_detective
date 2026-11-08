# CP5 figure-provenance repair

Base: `ffd44169e49ddb11158bef99104334e2fb792a36`, freshly fetched main.
Implementation branch: `codex/cp5-figure-provenance`, isolated worktree.
No live run, paid provider call, credential loading, historical-evidence modification, merge, or issue closure is part of this work.

## Verified failure and baseline

Retail attempt `56ff5d74-a3f7-4a42-b422-48df420d35e1` reached a genuine acceptable visualization verdict, then failed in `report_packager_node` because `retail_orders_net_revenue_vs_gross_revenue.png` did not exist. The revised worker had reported that intended filename after failed custom plotting attempts even though a fallback scatter tool created a differently named real PNG. The actual filename was confined to ToolMessage.artifact, which native Responses serialization omits from model input.

The worker checked completion flags but published the nonexistent path and wrote it to memory. Join checked structured metadata strings without validating the file; its path-only duplicate shortcut bypassed the later filesystem checks. Report planning propagated the existing bad record into sections. The packager correctly rejected five references to the one nonexistent filename. This repair leaves the packager unchanged.

The final baseline probe loads the frozen base notebook from Git into the existing offline loaders without changing the on-disk candidate. It runs the 31 new cases: **28 fail and 3 pass**, exit 1. Those failures establish missing visible paths, false worker completion, invalid structured inventory, conflicting identities, unsafe references, unfinished duplicates, and the failed-custom/successful-fallback boundary. The three passing cases are existing negative controls. Receipt: `figure-provenance-baseline-final.txt` under the local CP5 attempts directory.

Initial probes also exposed test-loader omissions: `typing.io` shadowed stdlib `io`, and the Python REPL fixture lacked real `ast`, `textwrap`, and `builtins` imports. These were corrected in the offline loader to match production imports. Early results containing those fixture failures are retained but are not used as the definitive red evidence.

## Correction

The active `_patch_notebook.py` transformation changes only generated Cells **32 and 57**:

- Successful histogram, scatter, box, and violin results now include compact JSON model-visible metadata with the actual image_path and relevant plotting fields. The native `(content, artifact)` contract, names and argument schemas stay intact. Image bytes/base64 are excluded. Failure returns remain failures.
- Histogram and violin saves now use the existing canonical `_resolve_artifact_path`, as scatter and box already do. Their legacy save location could otherwise violate the current-run checks. Rendering semantics and model settings are unchanged.
- The worker receives an explicit actual-saved-path contract and the concrete owned current-run directory, and keeps its assigned visualization ID. Before any visualization-memory write or state publication, every output asset must resolve through canonical path safety, remain inside RUNTIME.run_dir, exist as a nonempty file, and verify as a readable PNG.
- Relative references retain the existing explicit artifacts_path policy. URI, traversal, stale/outside-run and symlink escapes are rejected without creating directories. No disk discovery, guessed substitution, renaming, or ID rewriting occurs.
- Join validates structured records, envelopes, and path-only references before deduplication. Unfinished records cannot hide behind duplicates. Equivalent ID/path bindings still deduplicate; contradictory ID/path bindings raise a controlled error.

No canonical module, manifest, State field, graph node/edge, schema, model setting, SDK retry, revision limit, shared hook, evaluator, safe invocation wrapper, production validator or packager is changed. Static comparisons confirm those functions/modules remain identical to base. The inactive legacy save_viz_for_state helper is not used as the repair boundary.

## Offline verification

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

## Separate finding and stop conditions

The existing heatmap renderer fails in an isolated valid-data probe with undefined `cols`. Its AST remains unchanged. It is documented in [cp5-heatmap-rendering-followup.md](cp5-heatmap-rendering-followup.md), not silently repaired or hidden with a fixture global. Only the four verified working plotting tools are claimed by the success-path regressions.

No live retry was performed. The campaign remains stopped pending owner approval. #129 remains deferred and #165 remains open. Live figure semantics, the eventual final report, and the two production acceptance thresholds (12/12 and 9/9) remain unqualified by this offline repair.
