# CP5 visualization memory ID contract

Base: `23e46c600746d1e74fd48fab8859c70eb174400c` (merged cleaner PR #192).

## Verified failure

Fresh retail attempt `8e5d5189-51ca-40b0-8cf9-90b1dd3ae8a7`, thread
`thread-a5f216e9-1d3c-48d9-9216-bf331d39cf70`, completed native cleaning and
analysis but failed strictly in visualization. Cell 75 reported `STREAM_FAILURE`;
the actual exception was LangMem's missing-ID `ValueError`. The verified stack
ends in native ToolNode execution, `_viz_manage_memory_safe`, and LangMem
`manage_memory`. The runner exited 1 after 336.8 seconds and closed its kernel.
Neither production validator was reached; no later scenario was started.

The original failed worker's tool arguments were not persisted in the outer
checkpoint. The exception proves that update/delete reached LangMem without an
ID after the adapter; it does not prove whether the model originally supplied
one of those actions or an invalid alias that the adapter remapped.

The legacy visualization adapter rebuilt the tool from an untyped Python
function. It lost the native enum/UUID argument constraints, silently stripped
IDs on create, and remapped arbitrary invalid actions to update. It then called
LangMem's `.func` directly. Update/delete without IDs still raised, and neither
the model nor native validation received a corrective tool result.

Offline red cases used the real generated factory, worker, native inner agent,
real LangMem tool/store and valid current-runtime PNG fixtures. The baseline had
**8 failures / 4 passes**. Missing update/delete IDs reproduced the halt; other
failures established the lost schema, silent create-ID stripping, invalid-action
normalization, and absence of bounded correction. Valid create/update/delete
and unexpected-store-error controls already passed.

## Narrow correction

The active patcher changes only generated **Cells 42 and 44**:

- Register the original LangMem tool directly for
  `("memories", "visualization")`. Keep `manage_memory`, its flat
  `content`/`action`/`id` parameters and default create action. Restore the native
  action enum and UUID validation; invalid aliases are no longer silently
  remapped. This intentionally restores constraints that the old adapter lost.
- Remove the adapter's silent ID stripping and its fallback that could remove
  the memory tool. A missing registration anchor now fails regeneration closed.
- Apply the cleaner's already-tested role-local ID policy to visualization.
  The patcher derives the same native guard body with visualization-specific
  names and error marker; the cleaner factory source is unchanged.
- Clone the bound memory tool description to explain omitting ID for create
  and obtaining an existing ID from `search_memory` for update/delete. Preserve
  the original tool and its native schema object.
- Reject the two known ID combinations before store access. On the first
  malformed round, return a native error `ToolMessage` paired to the original
  call ID. A later ID-contract error in the same assignment raises an actionable
  exception. Parallel malformed calls count as one correction round; the newest
  named runtime context scopes the bound, including reused threads.
- Delegate valid calls and other malformed inputs to native LangMem/LangChain
  validation. Propagate unexpected store/tool exceptions. Completion still
  requires native `VisualizationResults` and the existing real-asset checks.

No replacement metadata, figure approval, recovery completion or file discovery
is introduced. Visualization tools remain available. Canonical modules/manifest,
State, graph, `VisualizationResults`/`DataVisualization` schemas, model settings,
SDK retries, recursion160, shared model hook, worker/join and both production
validators are unchanged. No new public parameter, module, State field or graph
node is added. Restoring the native memory argument constraints is an explicit
part of this repair, not a claim that the legacy schema was identical.

## Offline verification

Final local commands all exited 0:

- Visualization memory regressions: **14 passed**, two dependency warnings.
  Cases cover both missing-ID actions, supplied ID on create, native schema and
  description, corrected calls, repeated violation, parallel correction, new
  assignment on the same thread, all valid actions, native malformed UUID/action
  validation and unexpected store failure. Store contents and call IDs are checked.
- Focused native/provenance/revision/evaluator/join/reporting: **322 passed**, six
  warnings. Broad supported no-key: **1292 passed, 9 skipped**, eight warnings,
  including all **16 mandatory error-handling tests**.
- Full scientific/notebook/proof-runner real-Jupyter selection: **56 passed**
  with `IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight`. The existing real
  provenance smoke additionally executes corrected native visualization memory
  followed by valid worker completion.
- Notebook and graph validators, Black and Flake8 passed. A final patcher/parity/
  canonical-state selection passed **211 cases** after retaining the old code-cell
  filter in the stricter registration transformation. Generated notebook bytes
  remained unchanged by that filter.
- Two regenerations were byte-identical. Raw Windows notebook SHA-256:
  `101071aba25671f487846238df162f98e029f6f3d332b2db54b84fca3130ec14`.
  LF-normalized SHA-256:
  `1ff15c3668b75710a2cb31e0f3c8777f9fd9ae766d2494cc49e8f45fedd505d0`.
  **99 cells, 70 State fields, 15 nodes, 33 edges, 39 reducers** remain intact.
- Broad and kernel selections provide **1348 distinct passing cases**, nine
  skips. Focused/repeated selections are not added again. Skips are the absent
  historical Run88 fixture, two optional-pyarrow cases and six POSIX-only cases
  on native Windows.

The first green invocation had 11 passes and one test expectation failure: the
native error formatter omits the Pydantic `literal` type label. The corrected
assertion checks its actual allowed-action error, error status and unchanged
store. No production behavior was loosened to satisfy it.

```text
python -m pytest -o addopts= tests/unit/test_native_completion.py -k viz_memory -q
python -m pytest -o addopts= tests/unit/test_native_completion.py tests/unit/test_figure_provenance.py tests/unit/test_analyst_revision_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_viz_join.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py -q
python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_error_handling_framework.py test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q
python -m pytest -o addopts= tests/unit/test_patcher_integrity.py tests/unit/test_core_parity_guard.py tests/unit/test_canonical_state_graph.py -q
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
python _patch_notebook.py
python -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -v
```

Full argv/exit/JUnit/hash receipts are outside the repository under
`C:/Users/darf3/Documents/cp5_attempts/completion_heatmap_20261108/viz-memory-repair`.
No raw model reasoning, credentials, image payloads or unsanitized logs are
published. All offline model boundaries are controlled; no IDD provider was
contacted by regression execution.

## Separate observations and remaining gates

The failed attempt's actual cleaned CSV had 180 rows, zero duplicates and no
missing values, with deduplicated gross/net totals preserved. Its free-text
cleaning description incorrectly said 176 rows, and numeric dtype conversions
reported `Column must be 1D` while retaining already-numeric dtypes. These are
separate observations, not silently repaired here. No report or CP5 success was
accepted; later runs still require actual cleaning/statistical/figure/report
inspection in addition to 12/12 and 9/9.

Required exact-HEAD hosted Linux/Windows CI and independent Antigravity approval
remain gates before merge or another fresh live attempt. Native Windows local
receipts do not pretend to be hosted CI. PR #129 remains deferred; issue #165
stays open. Budgets and spending limits are unchanged. Historical evidence is
untouched, and checkpoint resume is prohibited.
