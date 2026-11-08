# CP5 numeric Series validation

Base: `2c6384addef0a64982df32d4bd7607d68f7003f1` (merged PR #200).

## Verified failure

Fresh retail attempt `b00f5557-c15d-4909-83ef-f1f1b3c1fb14`, thread `thread-662eeec1-2de1-4ca4-8048-ba12215820a3`, stopped after 106.4 seconds with Cell 75 STREAM_FAILURE and cleanup CONTEXT_CLOSED. SQLite's public exception is `Cleaner returned unfinished CleaningMetadata; completion was not published.` The genuine native CleaningMetadata has finished_this_task=False and explicitly identifies unresolved numeric datatype conversions reporting `Column must be 1D`. Strict completion correctly prevented analysis/visualization/reporting; no production validators or paid sample_dirty run occurred. The repair must not upgrade that output or resume the failed checkpoint.

The actual canonical user request matches the intended scenario request (SHA-256 `7ac9af4cc77491b42e25e2f61670f93a12c65e678a77cf38a36ee41c9a86b711`), confirming PR #200's boundary works. Actual retained export has 180 rows, zero duplicates/missing values, and SHA-256 `63ecc0422c67c0f512cf318950c23f82f055b3b32d6aa70a7606c0263c4971bc`. Existing numeric columns are already numeric, so the converter can falsely reject even valid typed input. The native description's 176-row claim remains a separate prose error; the export is an inspected real file, not a synthesized successful cleaning result. Its supplemental inspection copy is never injected or counted as a native final artifact.

Only two provider responses were saved in outer checkpoints versus nine successful response log occurrences, plus ten embedding occurrences. Attempt saved-response subtotal USD 0.00073415 brings the carried partial subtotal to USD 1.24926750. This excludes unsaved/embedding billing and is not a reconciled bill. No paid process runs during this repair; existing limits are unchanged.

## Root cause and minimal correction

The real decorated `convert_data_types` calls `pd.to_numeric(s, errors=errors)` on a Series, then requires the returned values to be instances of `pd.Int64Dtype` or `pd.Float64Dtype`. Those are dtype descriptors, not the returned column container. Pandas documents [Series output for Series input](https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html). A second identical condition blocks the nullable-integer path as well.

The active patcher replaces only these two conditions with the existing `is_1d_vector(s2)` check. It retains the original pre-conversion dimensionality check and both post-conversion dimensionality checks. All parsing, locale cleanup, nullable conversion, downcast/cast code, per-column error handling, dry-run logic, registration/raw provenance and message/artifact formats remain unchanged. It does not modify errors or completion flags to manufacture success. Other dtype-width/downcast/boolean/category semantics and installed-Pandas compatibility for other policies are unchanged and are not newly promised by this correction.

The generated diff is confined to Cell 32. AST comparison proves that, after normalizing precisely the two condition replacements, the entire Cell 32 tree is identical to base. All other 98 cells, cleaner factory/node/strict validator, shared/Qwen hooks, model configuration, schemas, State, graph, retries/timeouts/recursion/revision limits, canonical seven modules/manifest, proof runner and production validators remain unchanged. Invariants: 99 cells, 70 State fields, 39 reducers, 15 nodes, 33 edges; strict 12/12 plus 9/9 still required.

## Offline reproduction and controls

Before production edits, the real generated decorated tool and cleaner/factory tests produce **8 failed, 3 passed**, exit 1. Valid integer strings, float values/strings, nullable integers, locale conversion, coercion and dry-run fail with the false one-dimensionality error. The native cleaner reads the actual model-visible conversion JSON, returns genuine unfinished CleaningMetadata and the unchanged strict node rejects it, reproducing the live boundary. The JSON is also checked against the ToolMessage artifact; the script does not rely on model-invisible metadata alone. The strengthened visible-message baseline was repeated with the same 8/3 result; repeated runs are not distinct tests.

After correction: **11 passed**, two dependency warnings. Tests verify actual values/dtypes, nondefault index and unrelated-column preservation, native parsing/coercion behavior, dry-run object identity, and continued rejection of real two-dimensional duplicate-label columns. Real cleaner/factory/tool/registry/export tests consume model-visible numeric evidence and reach native CleaningMetadata with valid registered data and a real matching CSV; a forced unfinished response still raises despite valid data and a real matching CSV. No production code constructs or rewrites a completion. These are controlled mechanism tests, not a replay of unavailable private inner histories or live qualification.

The existing real-Jupyter provenance case calls all eleven new cases, while the inherited proof-kernel test also retains all native-completion cases. The same kernel and 120-second per-cell bounds remain; no smoke case or timeout is removed or weakened. Existing datetime compatibility, strict failure, revision/reporting and isolated SQLite regressions remain required.

## Initial publication receipts

Target 11 passed; patcher/parity/state 211 passed. Full focused/broad no-key/all16 mandatory/kernel/validator/determinism and exact-HEAD Linux/Windows checks are running and are not yet claimed green. Fresh installed Antigravity exact-HEAD review is required before merge or live execution. Final commands/exits/counts/warnings/skips/CI/hash receipts will be posted in the PR conversation with overlap counted once.

Raw notebook SHA-256 `96ea375b9107354a9b9e51ace891d6f0231f286a26f283b21efa2de64cf5ffdb`; LF-normalized `763042b3a65c015a43c2a5d9b1696f4268be623deceb00e91c88ea263d382707`.

Historical evidence remains untouched. The cleaner row-count prose, REPL RuntimeCtx observation and prior report-reference failure are separate. #129 stays deferred and #165 open. Any fresh qualification requires new preparation and fresh sequential retail/sample attempts on the same final merged source/environment, with actual production gates and data/figure/report inspection before sample_dirty proceeds.
