# CP5 Python REPL namespace and execution isolation

Base: `b5032607a70ba6af1237a0172a7f607cb958b6d7` (merged heatmap PR #190).

## Observed failure and bounded reproduction

Fresh retail attempt `e04b091c-379a-4a5c-a4c9-5c029be58953`, thread
`thread-0f9fe329-6b0d-44bf-a8d3-dde15f58f225`, failed strictly in Cell 75.
Its terminal error rejected a claimed PNG path outside the current run. The
preserved temporary directory contained seven genuine PNGs; one claimed path
dropped a character from the directory name. No file was substituted, no
checkpoint was resumed, and neither production validator was reached.

That path claim is distinct from the REPL defect repaired here. Repeated Python
name errors prompted a separate offline investigation through the actual
generated, decorated `python_repl_tool`, with the real registry and no providers.

The shared `PythonAstREPLTool` received a fresh globals dictionary and a separate
empty locals dictionary. Functions, lambdas and generator expressions could not
resolve variables assigned earlier in the same submitted code. Rebinding `df`
also persisted the old globals binding instead of the submitted result.

A controlled two-thread probe paused the first real invocation after namespace
setup, then executed a second call for a different registry DataFrame. The first
call printed the second frame's value (100 instead of 10), and persisted that
frame under the first ID. This is reproducible data contamination, independent
of the terminal path claim and the lossy console stream formatter.

Before production edits, the nine-case regression selection returned exit 1:
**5 failed, 4 passed**. The failures were the three lexical-scope cases, DataFrame
rebinding, and the forced overlapping calls. The four passing controls covered
fresh-call variables, authored-only registration, ordinary code errors, and
outside-sandbox writes.

## Narrow correction

The active patcher adds an execution lock around the existing REPL's output
capper, namespace setup, sandbox, execution and persistence. Calls receive a
fresh dictionary shared by globals and locals, so normal Python lexical scope
works within one call while transient bindings remain absent from later calls.
The auto-registration pass excludes unchanged injected bindings; it continues
to register authored or rebound DataFrames and persists `df` under its supplied
registry ID. `functools.wraps` preserves the registered tool's signature.

Only generated Cell 32 changes. No canonical module, manifest, State field,
graph edge, tool schema, model setting, shared hook, validator or figure-path
guard changes. Existing REPL error-return and registry persistence semantics
are retained; this does not introduce transaction rollback for user code.
Serialization covers this REPL tool, not every other notebook plotting tool.

## Verification receipts

Sanitized command receipts and JUnit files are outside the repository under
`C:/Users/darf3/Documents/cp5_attempts/completion_heatmap_20261108/repl-repair`.
Raw historical logs, private model reasoning and image payloads are excluded.

- Red: `python -m pytest -o addopts= tests/unit/test_native_completion.py -k test_repl_ -q`
  — exit 1, 5 failed / 4 passed / 128 deselected, two dependency warnings.
- Green: same selection — exit 0, 9 passed / 128 deselected, same warnings.
- Focused native/provenance/revision/evaluator/join/report selection: exit 0,
  **298 passed**, six warnings.
- Supported broad no-key selection: exit 0, **1268 passed, 9 skipped**, eight
  warnings. This includes all **16 mandatory error-handling cases**, patcher,
  parity, canonical State/graph, proof-runner and root regressions. Skips: one
  unavailable historical Run 88 fixture, two optional-pyarrow cases, and six
  POSIX-only permissions/ownership cases on native Windows.
- Notebook and graph validators, Black checks for both edited test modules,
  and smoke-module Flake8: each exit 0.
- Two patcher invocations: exit 0, byte-identical notebook. Raw Windows SHA-256:
  `464aa1d160f1bf255c2ac2eb685a269a419795b0d1fe061dd52027ae527409a7`.
  LF-normalized SHA-256:
  `5bdcdaadeb5c889f0ebc73a249c50b3c6ff7af94500dd295b2f2c38087a53410`.
  Invariants: **99 cells, 70 State fields, 15 nodes, 33 edges, 39 reducers**.
  An AST comparison changes only `python_repl_tool` and adds its private
  serialization wrapper; existing Cell 32 plot functions are unchanged.
- Scientific, notebook and proof-runner real-Jupyter selection with
  `IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight`: first invocation exit 1,
  **55 passed / 1 failed**. The added probe completed its assertions but its
  success marker followed verbose tool output without a line break, so the
  harness could not parse the receipt. A newline before the marker repaired the
  test framing; targeted real-kernel rerun exit 0, **1 passed / 24 deselected**.
  The concurrent test additionally verifies restoration of the original stdout.
  All **56 distinct** kernel cases have successful receipts. Combined with the
  broad selection, this is **1324 distinct passing cases**, nine skips; focused
  and repeated invocations are not counted again.

Commands (full argv, elapsed time and exit codes also retained in JSON receipts):

```text
python -m pytest -o addopts= tests/unit/test_native_completion.py tests/unit/test_figure_provenance.py tests/unit/test_analyst_revision_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_viz_join.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py -q
python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_error_handling_framework.py test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
python _patch_notebook.py
python -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -v
python -m pytest -o addopts= tests/smoke/test_real_kernel_notebook.py -k generated_figure_provenance -v
```

The existing real-Jupyter provenance smoke also exercises same-call nested
scope and overlapping real REPL calls. Broader selections overlap focused tests;
their passing counts must not be added together as distinct cases.

## Campaign boundaries

The owner authorized further fresh attempts and merge after Antigravity approval
and CI. A source repair requires new exact-HEAD review and Linux/Windows gates
before merge/retry. Both final scenarios must qualify the same source revision.
`sample_dirty` remains gated on retail's genuine 12/12, 9/9 and artifact inspection.
PR #129 stays deferred; issue #165 stays open. Historical failed-run evidence is
preserved. Recovery, synthetic completion and threshold weakening remain forbidden.

The failed attempt's saved-response estimate is $0.10416983; cumulative partial
saved usage is $0.57741293. This excludes unsaved responses and embeddings and is
not a reconciled provider bill. Existing monitored spend and wall-clock limits
remain in force; offline regression execution makes no IDD provider calls.
