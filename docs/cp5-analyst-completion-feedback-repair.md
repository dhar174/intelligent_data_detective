# CP5 analyst completion and evaluator-feedback repair

## Scope and baseline

- Verified remote `main` and implementation base: `e20ee85497e767f5df348fc55e0d820f039b30c6`.
- Isolated branch: `codex/cp5-analyst-completion-feedback`.
- Failed retail attempt: `373e4cd9-d6f3-4164-8910-48261850a82e`.
- Historical thread: `thread-6e8566c6-1b8c-45d2-b7c1-b35ce6fc74ab`.
- This repair uses offline scripted models only. No historical checkpoint is resumed,
  no credentials are loaded, and no paid run is authorized by these results.
- #129 remains deferred, #165 remains open, and the campaign remains stopped.

## Verified failure and causal limits

The saved native `AnalysisInsights` response said revised specifications had been
prepared. It contained the two targeted specifications and the unaffected third
specification, but its top-level `finished_this_task` was false. The node raised
`analyst integrity: AnalysisInsights did not finish the assigned task`. The strict
runner exited with code 1. Production acceptance validators were not reached.

The evaluator's detailed findings were in `VizFeedback.reply_msg_to_supervisor`.
Its shorter `feedback` field referred to per-figure notes above. The analyst context
builder omitted that detailed reply. A network-denied probe of the actual generated
factory/node confirmed that the reply was absent from effective model messages.

The active factory prompt requested a complete revision plan but did not explicitly
define top-level completion independently of downstream rendering and acceptance.
The returned public response is consistent with confusion at that boundary. The
model's internal reason for choosing false is not proven. Scripted regressions prove
instruction delivery and native contracts, not future model compliance.

The saved response also embedded `rendering_output_path` and a proposed filename in
two `style` strings. This is recorded separately as invalid field usage. This patch
reinforces existing worker destination ownership in the prompt; it neither accepts
nor rewrites that historical response, and does not introduce a style parser or
change worker behavior.

## Correction

The active `_patch_notebook.py` analyst transformation changes only generated
Cells 44 and 57.

1. The factory prompt defines revision completion as preparing a complete,
   evidence-backed plan. Rendering, saving PNGs, and evaluation remain downstream.
   It distinguishes the top-level analyst flag from nested specification flags and
   requires an honest false response with a specific blocker when necessary.
2. The runtime context adds `evaluator_review` with the genuine evaluator's
   `reply_msg_to_supervisor`, `feedback`, and `redo_list`. Existing context fields
   remain available. Native and serialized feedback objects are supported through
   the existing accessor and preview-sanitization helpers.
3. The prompt directs the analyst to translate renderer-facing feedback into
   `viz_instructions` and keep `style` free of paths, save code, and JSON fragments.

No exception handling, retry, native result validation, completion flags, evaluator
grades, tool registration, or stored history is rewritten. No new recovery path is
introduced. `ToolStrategy(AnalysisInsights)` remains the completion mechanism.

## Offline regression evidence

Seven new unit cases cover detailed review delivery in native and serialized form,
with and without summary pressure, on an inner thread reused after initial analysis;
planning completion before any new PNG exists; and rejection of unfinished output
both with and without a detailed plan. The tests inspect effective model messages
and Responses API input serialization. Network access is denied and provider key
variables are removed by the existing fixture.

The detailed unfinished-response case includes a sanitized malformed style string
with an embedded `rendering_output_path`; it must still fail without memory writes,
new visualization tasks, or PNGs. This tests rejection of the unfinished response,
not semantic validation of arbitrary styles on a finished response.

The existing real-kernel analyst smoke test now requires the detailed evaluator reply
and the explicit completion contract at the actual model boundary.

Before production edits, the corrected baseline run produced **5 expected failures,
2 passes, 18 deselections, and 2 dependency warnings**, exit 1. Four failures exposed
the missing review field and one exposed the missing completion instructions.
The two strict-failure cases already passed on the baseline.

An earlier test-authoring run also failed two assertions because its fixture already
contained `viz_tasks`. Those assertions were corrected to compare the pre-call task
list with the post-failure list. They are not counted as production defects.

After the repair, the analyst suite produced **25 passes and 2 dependency warnings**,
exit 0. The real-kernel analyst probe produced **1 pass and 26 deselections**, exit 0.

The focused suite produced **554 passes, 23 warnings, zero skips**, exit 0, in
258.56 seconds. Its breakdown is:

| Suite | Passed |
| --- | ---: |
| Native completion | 246 |
| Visualization evaluator | 39 |
| Report grounding | 30 |
| Revision integration, including real generated functions and isolated SQLite | 12 |
| Patcher integrity, including exact-byte regeneration comparison | 118 |
| Core parity | 38 |
| Canonical State and graph | 55 |
| Mandatory error handling | 16 |

The final analyst suite, including the malformed-style negative fixture, passed
25 tests with 2 warnings in 31.29 seconds. JUnit testcase identifiers establish
**580 distinct passing tests** across the final analyst, focused, and kernel runs.
Repeated analyst invocations and the red baseline are not added to this count.
There are no failures or skips in those final selections; the kernel command
intentionally deselected 26 unrelated kernel cases.

Warnings were dependency deprecations (`langchain-experimental`, LangGraph `Send`,
`fitz`, and `langchain-community`) and existing Matplotlib tick-label warnings.
Formatting checks on both changed test files and `git diff --check` passed.
An intermediate formatting check detected the added negative fixture's layout;
Black corrected it and the repeated formatting check passed.

## Regeneration and invariant evidence

- 99 cells; changed generated cells are exactly 44 and 57.
- Notebook validator compiled all 42 code cells, exit 0.
- Graph validator compiled 15 nodes and 33 edges with 70 State fields, exit 0.
- AST comparison found only `create_analyst_agent` and `_cp5_analyst_context` changed
  among functions/classes in those two cells. The node and safe wrapper are unchanged.
- Model-definition Cell 41 and shared-hook Cell 43 are unchanged.
- Seven canonical modules, manifest, and both production validators match the base.
- Consecutive regeneration raw SHA-256:
  `673c2051e63ca2f2ca8bc03dd46c2dcab6796a3e4acfa32f2814e4bc95ea81dc`.
- LF-normalized SHA-256:
  `b62aeac42ad84288ba14b50b85f0b09e87624d2a2589ed3352d2ee064f4ff260`.

The final notebook hash still matches after the patcher suite regenerated it.
The suite compares the actual bytes from two consecutive generations for equality.

The first implementation-time regeneration failed with a `replace_required`
argument-count error. The added instruction strings were corrected to concatenate
inside the existing replacement argument, then regeneration succeeded twice. The
failed invocation is not a successful regeneration receipt.

## Commands and receipts

Commands run from the isolated checkout with Python 3.12.2 at
`C:/Users/darf3/AppData/Local/Programs/Python/Python312/python.exe`.
Receipts are outside the repository in
`C:/Users/darf3/.codex/worktrees/cp5-analyst-completion-feedback/`.

```text
python -m pytest -o addopts= tests/unit/test_analyst_revision_completion.py -k "detailed_review or planning_completion or unfinished_cannot" -q --tb=short --junitxml=../red-corrected.xml
python _patch_notebook.py
python -m pytest -o addopts= tests/unit/test_analyst_revision_completion.py -q --tb=short --junitxml=../analyst-green.xml
python -m pytest -o addopts= tests/unit/test_analyst_revision_completion.py -q --tb=short --junitxml=../analyst-final.xml
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
python -m pytest -o addopts= tests/unit/test_native_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py tests/unit/test_patcher_integrity.py tests/unit/test_core_parity_guard.py tests/unit/test_canonical_state_graph.py test_error_handling_framework.py -q --tb=short --junitxml=../focused.xml
IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight python -m pytest -o addopts= tests/smoke/test_real_kernel_notebook.py -k generated_analyst_revision -q --tb=short --junitxml=../kernel.xml
```

Environment assignments above are notation for the PowerShell environment variables
used by the smoke invocation. JUnit arguments were passed as the absolute receipt
paths equivalent to the relative paths shown.

## Handoff limitations

This is an uncommitted local repair for owner review. No PR publication, hosted CI,
independent Antigravity review, merge, or paid qualification is claimed. Offline
completion tests do not establish the strict live 12/12 and 9/9 acceptance thresholds.
The full broad no-key selection and full kernel suites were not run for this narrow
repair; the explicit selected gates and single relevant kernel probe are listed above.
The next paid run requires owner approval.
