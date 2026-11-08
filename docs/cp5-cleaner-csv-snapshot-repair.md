# CP5 cleaner CSV snapshot freshness

Base: `7c9bf14af6d299d192c62c02165a889c481a092c` (merged PR #203).

The cleaner could export a CSV, change the registered DataFrame afterward, and
publish native `CleaningMetadata(finished_this_task=True)` without exporting the
changed data. This repair supplies observed snapshot currentness and rejects
completion when a known CSV snapshot no longer matches its registered data.
The model must perform a real corrective export and complete natively. No file
or completion object is manufactured by the guard.

## Verified failure and reproduction

Retail attempt `38b4310c-2eea-45e1-83d1-30a3044ed42c`, SQLite thread
`thread-aaefe4f8-264e-409a-ab8c-ec6104d32295`, used the exact base above.
The public paired tool sequence included successful CSV export
`call_0UhxosXns6lqk0yGLRIgvS12`, subsequent satisfaction imputation
`call_PUktEydiSzKRFD5Z6PCJbjsD`, and native `CleaningMetadata` completion
`call_wpoYwyKiELlsgoy0hZimU1yt`. There was no subsequent export.

The native description claimed 180 rows and zero missing values. The actual
export had 180 rows and **11 missing satisfaction scores**; its SHA-256 was
`f96ccb0f57e3c682280442dd194fd30881af7851a5c579be947f86acd3e2135a`.
The supervising process was stopped externally after this integrity check.
Its original evidence remains unchanged (`PREPARING`, cleanup not reached);
the separate supervisor receipt records `STOPPED`, exit 1, after 270.1 seconds.
Neither production validator ran, and `sample_dirty` was not executed.

An offline controlled model reproduced the boundary through the actual generated
factory and node, native `create_agent`/`ToolStrategy(CleaningMetadata)`, real
registry and decorated export/imputation tools. The sequence exported one missing
value, filled it in the registry, then completed natively. The old node published
`data_cleaning_complete=True`, while the CSV still had that missing value.
The assertion rejecting that result failed as expected (exit 1). This establishes
a reproducible completion-boundary omission; it does not explain private model
decisions in other failed attempts.

Three initial regression cases were red before production edits. The first test
draft also had an incorrect full-path expectation for the existing basename-only
tool content; that expectation was corrected and all three cases remained red for
the intended missing guard/context/receipt behaviors. Existing test function
bodies were verified unchanged by AST comparison.
After completing coverage, the final 19 cases were also run against the exact
Git-generated baseline notebook (LF SHA-256
`5ddcf90d4ef03f4cf0a7af328c70a9754ec3f09bca6dec481541d500e727a934`)
through a child-only source-loader override: **19 failed, zero setup errors**.
This final baseline comparison did not change the candidate notebook or replace
production functions with test templates.

## Narrow correction

The active `apply_cp5_cleaner_grounding_patch` transformation in
`_patch_notebook.py` owns the correction. Generated changes affect only:

- **Cell 44, `create_data_cleaner_agent`:** after an actual successful CSV export,
  the existing serialized data-tool middleware records a fingerprint of the
  current registered DataFrame. It copies the returned `ToolMessage`, preserving
  content, call ID, status and all original artifact fields, and adds the internal
  JSON-compatible `idd_cleaner_csv_snapshot_v1` receipt. Failed exports do not
  receive a receipt. Snapshot capture occurs under the existing data-tool lock.
- The cleaner-local model middleware reads paired receipts only since the latest
  runtime assignment. It compares the latest receipt per DataFrame against the
  current registry contents and supplies current/outdated status in the existing
  copied runtime-context message. A stale snapshot requires a genuine re-export;
  documented `overwrite=True` replaces the same filename.
- Immediately after a valid native `CleaningMetadata` response, the middleware
  checks currentness again. A finished response with an outdated snapshot raises
  an actionable integrity error before publishing completion. Fingerprint
  failures also fail closed and retain their original exception as the cause.
- **Cell 57, `_safe_data_cleaner_invoke`:** CSV-integrity errors bypass legacy
  recovery even if `IDD_ALLOW_RECOVERY=1`. Normal strict execution remains at `0`.

The fingerprint combines an in-memory indexed CSV representation, the missing
value mask and dtype descriptions. It writes no file. It is an assignment-local
registry snapshot receipt, not a cross-platform artifact digest. Read-only calls
and actual no-op mutations preserve currentness; real changes require export.
Both orders of a parallel export/mutation batch are tested through real tools and
the existing lock. A later export of another DataFrame cannot refresh the stale
one. Reused native threads retain history without importing an earlier
assignment's snapshot obligation.

`CleaningMetadata`, registered tool schemas, model settings, retries, timeouts,
recursion limits, memory correction and no-progress bounds are unchanged. The
shared request hook, canonical modules/manifest, runtime types, reducers, graph,
artifact ownership and both production validators are unchanged. The notebook
retains **99 cells, 70 State fields, 39 reducers, 15 nodes and 33 edges**.

## Limits and separately confirmed issues

The guard checks **known successful CSV exports**. It does not create or certify
a missing export, verify arbitrary natural-language claims, or prove that a file
was not subsequently edited, removed or corrupted. Full proof still requires
actual cleaned-data, PNG and report inspection alongside the unmodified 12/12
and 9/9 production gates. A current snapshot is never a substitute for native
completion or genuine evaluator acceptance. No failed historical attempt is
qualified by this repair.

Fingerprinting examines the registered data before requests and native completion
after a CSV export; its cost scales with the DataFrame size. Receipt hashes are
for currentness within the execution environment, not portable proof of data
identity across environments. No checkpoint resume is authorized.

Two pre-existing export-tool defects were independently reproduced and remain
outside this change: explicitly supplying `na_rep` passes that keyword twice to
Pandas, and default same-name versioning references undefined `stem`. The real
tool returns failure in both cases. Corrective regression exports use documented
`overwrite=True`; no export-tool API or implementation is changed here.

The first local parity invocation also failed in the unchanged publication
rollback test with a Windows `PermissionError` (210 passed, one failed). Its log,
JUnit and 11 synthetic fixture files were preserved separately. The unchanged
three isolated rollback cases and full 211-test parity selection then passed.
The original cause is unconfirmed; no report code, assertion or timeout was
changed. Initial expanded CSV coverage had six fixture failures (CSV intentionally
omitted the original index; two fixtures used a nonexistent registry method).
Those fixtures were corrected to compare exported row values and use the actual
registry API; all 18 cases then passed. Original failure receipts are retained.
A nineteenth case verifies the stale-snapshot guard under actual hosted-history
pressure while retaining checkpoint history. Its first fixture invocation used
a summary limit inconsistent with the existing spy's asserted 32-token contract;
using that unchanged test contract corrected the fixture and the case passed.
No production request or summary budget changed.

## Verification receipts

Two regenerations were byte-identical. Candidate notebook SHA-256:

- Raw bytes: `d1e1c4a09b33d925aeb7cc4ffe3032e41ba2d6cfbf3f920aecbdb88c40e38c20`.
- LF-normalized: `077d9b78a8ad7bc4d774ce9770f0872209d848cf1af82ab6e4db16ee662efc49`.

The pre-publication focused selection passed 414 cases with six warnings;
the additional history-pressure case passed separately. Patcher/state/parity
passed 211 cases after the disclosed unchanged-test rerun. Notebook and graph
validators, format checks and lint passed. These results do not imply hosted CI
or live qualification.

Final candidate commands, distinct counts, regeneration hashes and exact-HEAD
hosted/review receipts are recorded with the PR. The focused snapshot selection
contains 19 cases; its real-Jupyter smoke exercises those same cases in an attested,
credential-free `idd-preflight` kernel with network connections denied. Nested
kernel cases and repeated invocations must not be added again to distinct totals.

Local receipts are retained in the isolated CP5 operator bundle under
`completion_heatmap_20261108/cleaner-export-freshness-repair`; raw historical
logs, private reasoning, credentials and image payloads are not published.
Linux and native Windows hosted CI and actual Antigravity exact-HEAD approval
remain merge gates. PR #129 remains deferred, issue #165 remains open, and live
execution remains stopped until those gates pass.
