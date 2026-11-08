# CP5 cleaner parallel mutation repair

Current base: `aa75dc0f4423bfcc679fb24d240697207d3416e3` (merged PR #195). Original failure/reproduction base: `6906ef4e5652294698c8201f6de0dd20193bc64a` (merged PR #193).

A fresh retail attempt `85902ab5-bbe6-422a-8e00-bd19bec6b8fb`, thread `thread-90a37812-0c65-4581-a22d-37ed11cbdae9`, reached native analysis and visualization. It was manually stopped at the data-integrity gate after successful cleaning messages disagreed with later numeric missingness. This was not a native completion exception or a completed proof. The owned process tree stopped after 414.2 seconds, exit 1. Native evidence remains PREPARING / cleanup NOT_REACHED; a separate process check found no remaining attempt-owned Python. No production validators or paid sample_dirty run occurred. Historical evidence/checkpoints/PNGs remain untouched.

Verified public call IDs in one assistant batch:

- `call_lMF5mTV8nkdk1gWArFZQxR1J`: fill_missing_median(discount_rate).
- `call_LOK9D6YtmNBj93rtxbP6TrJa`: fill_missing_median(satisfaction_score).
- `call_pI3Ueok655IvOLdZb7hflXL9`: convert_data_types(order_date).
- `call_TuhAa2gqfh2rlIkkj1uNXu9k`: check_missing_values.

The first three reported success. Later analyst statistics had 180 rows but 169 present satisfaction scores. Exact live read/write timing is not logged. A deterministic offline probe of the real generated decorated imputation tool and isolated DataFrameRegistry synchronized two reads before replacement and reproduced two successes with a remaining null. Native factory/node tests reproduce both competing fills and fill/conversion loss. This establishes the defect and causal mechanism without inventing a precise live scheduling trace.

Data tools copy a registry frame, modify it, then register a replacement. Registry locks cover individual access/registration, not the whole read-copy-write tool operation. Native ToolNode runs a batch concurrently; one copied replacement can erase a sibling tool's changes. Locking only the final registry write cannot solve that race.

The active cleaner transformation adds a factory-local RLock through native wrap_tool_call around complete non-memory tool handlers, including reads/exports. manage_memory/search_memory bypass it and retain native handling and bounded ID correction. Preserve tool names, flat arguments, result/call-ID pairing, native CleaningMetadata completion, model parallel-tool settings and strict exception propagation. The context manager releases on failure. This is not transaction rollback or a deterministic ordering guarantee for independent parallel calls; separate factory instances/external writers are outside its scope. The production graph uses one cleaner factory before analysis/visualization.

Only generated Cell44 / create_data_cleaner_agent differs from current base. Shared hook, other hosted factories, local Qwen path, canonical seven modules / manifest, State 70 fields / 39 reducers, graph 15 nodes / 33 edges, 99 cells, schemas, model settings/timeouts/retries, recursion 160, worker/join and both production validators are unchanged. Inherited Cell32 datetime compatibility is byte-identical to merged PR #195. Production thresholds remain 12/12 and 9/9.

## Baseline and dependency evidence

Before production edits on original base: two native regressions failed, 161 deselected, 3 warnings, exit 1. Final stronger date-value fixtures also fail against the actual original baseline notebook loaded from git. After PR #195 merged, the same final fixtures were rerun against its real generated notebook without serialization: both fail specifically at lost-update assertions (remaining satisfaction null or lost date dtype). The current serialization candidate passes both, retains paired successful ToolMessages, and terminates natively after exactly two model calls. A third case verifies unexpected native exception propagation without completion flags, then real imputation/native completion with the same factory on a fresh thread. All three are also executed in a real Jupyter kernel alongside eight independent datetime compatibility cases.

Original candidate HEAD `479e0cf0830f9272fa1a58441b7d571edd70d33f` passed 1351 distinct local cases but failed all required hosted runs: Linux 42471956262 (one conversion failure / 1157 passed / 3 skipped), Windows 42471958628 and 42471926045 (each one kernel failure / 24 passed). Antigravity approved code at that old HEAD; its approval and failed checks do not certify the refreshed HEAD. Hosted Pandas 3.1.1 failures matched a pre-existing obsolete datetime keyword incompatibility. The assertions were retained. Separate PR #195 removed only that keyword, passed its own exact-HEAD Linux/Windows runs 42473941827 / 42473944326 / 42473910651 and independent approval at `a73c5c36842bdc23af5e2061068da24dc72c3a78`, then merged to `aa75dc0f4423bfcc679fb24d240697207d3416e3` with verified equal tree `7cd10a5ca3c48dace8fab9f41b1539d462ae3ca1`.

This branch was rebased onto that merged base; the smoke conflict was resolved by preserving all eight date cases and all three cleaner cases. Original reviewed commit is retained on local backup branch codex/cp5-cleaner-serialization-pre-datetime, and all old receipts remain separate. New full local gates, new exact-HEAD hosted Linux/Windows and a new independent review are required before merge/retry. No failed run or old approval is reused as a green gate.

Saved usage from the stopped attempt is a partial $0.07632387; carried saved-response subtotal $0.75888100 excludes embeddings and unsaved/in-flight responses and is not a reconciled bill. No new live process started during either repair. Budgets unchanged; #129 deferred, #165 open. Cleaner free-text 176 versus actual 180 rows and existing numeric-conversion issues remain separately documented. Serialization does not certify all model narrative or cleaning choices: actual data/report/PNG inspection remains required before qualifying either CP5 scenario.

## Refreshed local validation

Local Python 3.12.2 / Pandas 2.2.1. This section applies to the refreshed source based on merged PR #195; hosted current-Pandas verification and independent review of the new final HEAD remain required.


| Selection | Exit | Result |
|---|---:|---|
| `serialization` | 0 | 3 passed, 161 deselected, 2 warnings in 8.93s |
| `focused` | 0 | 333 passed, 6 warnings in 166.45s (0:02:46) |
| `patcher-final` | 0 | 211 passed, 4 warnings in 30.48s |
| `notebook` | 0 | Passed |
| `graph` | 0 | Passed |
| `kernel` | 0 | 56 passed in 426.36s (0:07:06) |
| `broad` | 0 | 1303 passed, 9 skipped, 8 warnings in 228.94s (0:03:48) |
| `format` | 0 | Passed |
| `lint` | 0 | Passed |
| `regen1` | 0 | Passed |
| `regen2` | 0 | Passed |

Exact refreshed command argument receipts:

```text
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe -m pytest -o addopts= tests/unit/test_native_completion.py -k cleaner_parallel_mutations or cleaner_tool_serialization -q
exit=0; seconds=12.55
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe -m pytest -o addopts= tests/unit/test_datetime_conversion.py tests/unit/test_native_completion.py tests/unit/test_figure_provenance.py tests/unit/test_analyst_revision_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_viz_join.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py -q --junitxml=C:\Users\darf3\Documents\cp5_attempts\completion_heatmap_20261108\cleaner-serialization-refresh\focused.xml
exit=0; seconds=170.53
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe -m pytest -o addopts= tests/unit/test_patcher_integrity.py tests/unit/test_core_parity_guard.py tests/unit/test_canonical_state_graph.py -q
exit=0; seconds=35.24
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
exit=0; seconds=0.59
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
exit=0; seconds=8.92
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -v --junitxml=C:\Users\darf3\Documents\cp5_attempts\completion_heatmap_20261108\cleaner-serialization-refresh\kernel.xml
exit=0; seconds=429.56
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_error_handling_framework.py test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q --junitxml=C:\Users\darf3\Documents\cp5_attempts\completion_heatmap_20261108\cleaner-serialization-refresh\broad.xml
exit=0; seconds=233.34
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe -m black --check tests/smoke/test_real_kernel_notebook.py tests/unit/test_native_completion.py
exit=0; seconds=0.5
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe -m flake8 tests/smoke/test_real_kernel_notebook.py --max-line-length=120 --extend-ignore=E203,W503,E501
exit=0; seconds=0.56
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe _patch_notebook.py
exit=0; seconds=3.53
C:\Users\darf3\AppData\Local\Programs\Python\Python312\python.exe _patch_notebook.py
exit=0; seconds=3.5
```

1,359 distinct passing local cases by JUnit classname/name union; repeated/overlapping selections are counted once. Broad: 1,303 passed, nine skipped, eight warnings, including all 16 mandatory error-handling tests (14 behavior and two documentation cases). Skips: missing Run88 fixture, two optional pyarrow cases, six POSIX-only cases on Windows. Full real Jupyter: 56 passed with no skips or failures. Remaining dependency, plotting-tick and PDF API warnings are retained.

Regenerated twice, with byte-identical output. Raw notebook SHA-256 `1bcaa89ba617298a365449ece4fd52c4582c1b6057f1ec11062d9c73f594296d`; LF-normalized `0f7e010644ff35db90a3cf2259b582e75742e75db5707039c21aea53ec78003b`. Exact comparison to current base confirms only Cell 44 / cleaner factory differs; Cell 32 is inherited unchanged. Canonical seven modules, manifest and both production validators compared unchanged. Validators confirm 99 cells, 70 fields, 39 reducers, 15 nodes and 33 edges.
