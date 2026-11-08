# CP5 cleaner assignment and observed progress

Base: `e875263d104629ab6f18cfdc832adcee9a00f79a` (merged REPL PR #191).

## Verified live observations

Two fresh retail attempts used this identical reviewed source. They did not
qualify CP5, and `sample_dirty` was not executed.

- Attempt `4f050376-2e8b-497e-b133-9ab4bb7f4850`, thread
  `thread-44e2c84e-56ba-4811-85fd-3ac1ee7f2273`: native
  `GraphRecursionError` at the unchanged 160-step limit during cleaning. The
  stream recorded 54 business-tool results: 22 schema lookups, eight missing-value
  checks, two duplicate tools, four exports, 17 memory operations and one row
  deletion. Cell 75 stopped strictly; the kernel closed normally. Validators were
  not reached. Logged step labels are formatter labels, not fresh invocation counts.
- A subsequent offline real-factory probe received the actual supervisor
  assignment and exposed `CleaningMetadata`. A scripted 39-call sequence followed
  by native completion succeeded; this established an operable completion route,
  not an explanation for the live model's choices or a live cleaning success.
- Attempt `d0f09a0e-9450-4e1f-816b-4b198730ad09`, thread
  `thread-ec7ac1c5-9e1b-4b2c-aa0b-f956d646d1dd`: the cleaner returned native
  `CleaningMetadata(finished_this_task=True)`, claiming imputation, deduplication,
  category normalization and outlier clipping. It had only retrieved schema and
  exported files. Both exported CSVs were identical to the current raw input:
  **184 rows, four exact duplicates, eight missing discount values and eleven
  missing satisfaction values**. The supervising agent rejected this semantic
  gate and stopped its owned process tree before downstream proof acceptance.
  The original native evidence remains unchanged (`PREPARING`/cleanup not reached
  after the external stop); the separate stop and integrity receipts describe
  the actual outcome. The runner and Python processes under that attempt's cwd
  were verified absent. Neither validator was reached.

The second failure demonstrates that native structured completion alone is not
semantic proof. No unchanged CSV was accepted as cleaned, and no later reporting
or figure output from this stopped attempt qualifies the campaign.

## Effective boundary and correction

The old node overwrote the original user request with a fixed recipe, including
unconditional category normalization and outlier clipping, then rendered a long
persistence template with a competing best-effort completion deadline. The node
also published completion flags for an unfinished native result, including its
cached-result path. These are verified code behaviors; attributing every live
model decision to them would be an inference.

The active patcher changes only generated **Cells 44 and 57**:

- Deliver the original request, current supervisor/emergency assignment,
  description/sample, configuration, retrieved memories and tooling guidelines
  as current named human context. Keep the original request separate from the
  cleaner assignment. Do not forward a previous agent's message as the new task.
- Replace the unconditional recipe and best-effort success instruction with an
  explicit `CleaningMetadata` contract: perform supported, necessary actions,
  reuse existing observations, report actual actions and caveats, and mark
  unresolved work unfinished. Exporting or writing memory is not cleaning.
- Before the existing hosted request-budget middleware, supply current registry
  row/missing/duplicate counts and the last eight paired tool results. Limit the
  result text, exclude tool artifacts, and annotate reported errors. Replace a
  copy of the newest runtime-context message; do not alter checkpoint history or
  caller-owned messages. The existing message budget still applies.
- Derive an assignment-local no-progress bound from paired message history:
  three consecutive redundant retrieval results fail on the fourth identical
  observation; new retrieval information or an observed successful cleaning
  action resets the streak. Three consecutive completed memory-only rounds fail
  closed. Memory or exports cannot reset the retrieval streak. New assignments
  on the same thread get independent bounds.
- Reject unfinished or unusable native completion before publishing readiness,
  memory updates or a cached-success skip. Retain the existing native response
  schema and tool-choice behavior. Do not construct replacement metadata.

The cleaner's existing bounded memory-ID correction is unchanged. The shared
hook, canonical modules/manifest, graph, State, tool schemas, model settings,
recursion limit, artifact guards and production validators are unchanged. No
automatic data rewrite or synthetic completion is introduced. Existing emergency
message delivery behavior is retained; unrelated message lifecycle issues are
outside this repair.

## Limits and acceptance

The progress context supplies evidence to the model; it is not a formal verifier
of arbitrary natural-language claims. A dishonest `finished_this_task=True`
response can still require rejection by artifact/data inspection. CP5 acceptance
continues to require the actual 12/12 and 9/9 receipts plus inspection of cleaning,
analysis, figures and reports. The failed semantic gate above is preserved as
evidence of that distinction. No automatic validator threshold is weakened.

Six new cases failed before production edits (exit 1). After the patch those six
passed. An expanded cleaner/shared-history selection passed **47 cases**. The
first wider focused invocation had **305 passes / one failure**: the existing
history test expected unchanged effective runtime-context text. Its assertion
now checks the copied progress context, unchanged caller context, exact retained
message order/pairs and unchanged checkpoint history. The shared reducer and
budget assertions remain intact. Additional coverage verifies real exported
dirty data remains visible, real mutation updates the profile, new information
resets the retrieval streak, and a new assignment resets the guard.

The canonical structured tool-error envelope must not count as a successful
mutation. An additional red case reproduced that classification bug in the
candidate; it is fixed and covered. Two preliminary verification batches were
intentionally interrupted while correcting this case and restoring the existing
final CSV export requirement. Interrupted selections are unverified and are not
counted as passing. Their separate receipts are retained.

Final verification (all commands exit 0):

- Cleaner/shared-history selection: **48 passed**, two dependency warnings.
- Focused native/provenance/revision/evaluator/join/report selection:
  **308 passed**, six warnings.
- Supported broad no-key selection: **1278 passed, 9 skipped**, eight warnings.
  This includes all **16 mandatory error-handling cases**, patcher/parity,
  canonical State/graph, proof-runner, core, memory, prompt and validator cases.
  Skips: one absent Run 88 historical fixture, two optional-pyarrow cases, and
  six POSIX-only permissions/ownership cases on native Windows.
- Scientific, notebook and proof-runner real-Jupyter suites: **56 passed** with
  `IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight`. The existing provenance
  smoke additionally executes the real cleaner mutations/profile and unfinished
  completion rejection. No IDD provider was contacted by offline checks.
- Notebook/graph validators, edited-test Black checks, smoke Flake8 and two
  byte-identical patcher regenerations passed. Only Cells 44/57 differ from base.
  **99 cells, 70 State fields, 15 nodes, 33 edges, 39 reducers** remain intact.
- Notebook raw Windows SHA-256:
  `76dc2bb2cb674acd52b02477101e9861b47bd3e56f104b335a5163b4d9b8f546`.
  LF-normalized SHA-256:
  `05bd9d52f82a60cf1baa360d463330d63af0fddfbc201da2c67304c6bb7931ce`.
- Broad and kernel selections provide **1334 distinct passing cases**, nine
  skips. Focused and repeated invocations are subsets and are not counted again.

```text
python -m pytest -o addopts= tests/unit/test_native_completion.py -k "cleaner or hosted_history" -q
python -m pytest -o addopts= tests/unit/test_native_completion.py tests/unit/test_figure_provenance.py tests/unit/test_analyst_revision_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_viz_join.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py -q
python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_error_handling_framework.py test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
python _patch_notebook.py
python -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -v
```

Command/exit/JUnit and sanitized call-ID/argument/fingerprint receipts live outside the repository
under `C:/Users/darf3/Documents/cp5_attempts/completion_heatmap_20261108/cleaner-repair`.
No raw model reasoning, credentials, image payloads or unsanitized logs are
included here. The saved-response campaign estimate is **$0.60859842**, with
unsaved responses, embeddings and final billing explicitly unreconciled.

Owner authority allows merge after independent Antigravity exact-HEAD approval
and verified Linux/Windows CI, then fresh strict attempts. PR #129 remains
deferred, issue #165 remains open, and both final scenarios must qualify the
same final source. No checkpoint resume or recovery is authorized.
