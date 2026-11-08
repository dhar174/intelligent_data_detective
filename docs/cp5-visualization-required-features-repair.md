# CP5 visualization required-feature and completion-status contract

Base: `d35f341ed9e61de03d328375b273ad12a03f81f9` (merged PR #197).

## Actual failure and limits

Fresh retail attempt `fe1e1f17-40f0-4194-bb0b-b62d7e0ac446`, thread `thread-15c968e3-03bf-48ad-ac3a-b95c009e4ab7`, exited1 after253.6s with Cell75 STREAM_FAILURE; cleanup CONTEXT_CLOSED. This was not recursion or the retrieval guard. SQLite's error value matches the unchanged strict validator message: Visualization worker did not complete its required figures. Four sibling results persisted; no evaluator/report/validators or paid sample_dirty run occurred. Historical evidence remains unchanged and failed.

Public native ToolMessage output reports parent finished_this_task=True (scatter created/saved), with a DataVisualization child finished_this_task=False for Units vs Gross Revenue by Product Line. The corresponding assigned upstream VizSpec also has finished_this_task=False, as do the other pending plans. Failed inner histories are unavailable; this does not prove copying of the flag or an exact model decision trace. Stream display normalization damages Windows escape sequences in printed representations, so parsed display paths are not treated as canonical paths or used for any substitution.

The actual readable scatter PNG survives in partial-work. Direct visual inspection confirms that its required overall regression line is absent, and its hue colors have no usable legend. A file therefore exists but the explicit assignment is incomplete. The false child and strict rejection must be preserved; they are not repaired by flipping a flag. The built-in create_scatter_plot exposes no regression-line option; Python is available for that required custom feature. No scatter failure/statsmodels error was observed in this attempt's known-literal scan. Do not claim a tool algorithm or optional dependency defect from this evidence.

Actual cleaned CSV has180rows,0duplicates,0missing and matches deduplicate-then-median-fill (SHA63ecc0422c67c0f512cf318950c23f82f055b3b32d6aa70a7606c0263c4971bc). Cleaner176row prose, numeric-conversion 1D warnings and the independently noted REPL RuntimeCtx replacement remain separate. No final report was produced. This attempt added partial saved-response USD0.04285944, carried subtotalUSD1.02865911 (16saved responses vs18HTTP success occurrences; embeddings/unsaved billing unresolved). Limits unchanged; no new paid process during repair.

## Narrow correction

Only the active _cp5_visualization_messages helper changes, through _patch_notebook.py and regeneration of Cell57. It explicitly requires Python when a built-in cannot deliver a required feature (for example an overall regression line or custom annotations), with explicit df_id, complete code in one call and the owned output path. A PNG missing required features remains unfinished. Native completion flags describe this rendering task: each child is True only after its required figure is genuinely complete/verified; parent True only when all required children are complete; otherwise parent False with blocker. Upstream pending plan status is identified as separate from current figure status. No model output flag is rewritten.

Worker analytical context now contains the unchanged statistical summary, correlation insights, anomaly insights and recommended next steps, without another task's completion envelope or other visualization recommendations. The original AnalysisInsights object and full canonical assigned VizSpec remain intact in graph state; the complete assigned spec is still delivered. No statistics are recalculated or changed. This separates analysis/plan status from render status without changing any public schema.

The actual viz_worker, strict output checks, wrappers, factories, ToolStrategy(VisualizationResults), artifact validation, join/evaluator authority and retrieval guard are AST-unchanged. All other98 notebook cells compare byte-identical to base. Seven canonical modules/manifest/production validators, State70fields/39reducers, graph15nodes/33edges, 99cells, model IDs/reasoning/timeouts/retries, recursion160, shared/Qwen hooks, tool names/arguments/calculations and12/12+9/9 thresholds remain unchanged. Recovery0; no autoapproval, resultsynthesis, pathnormalization or checkpoint resume.

## Offline red/green

Before production edits: **3failed,2passed** across five new cases. Real generated factory/node plus a controlled model follows the task contract: baseline chooses the built-in scatter, generates a real PNG without the required fit, returns the same parentTrue/childFalse inconsistency, and strict publication rejects it. The analytical-context separation assertion also fails. Both forced unfinished-child cases (parentFalse and parentTrue, real readable PNG present) already fail closed as expected.

After correction: **5passed**. The controlled model chooses the real Python tool for required features; actual registered data is plotted by group with an overall fitted line and legend. Matplotlib artists and labels, readable actual PNG, native VisualizationResults, real join, assigned ID and unchanged source DataFrame are verified. Both upstream plan status values are tested; neither is treated as authoritative render status. Forced native unfinished children still fail unchanged with no successful flags or figure publication. Statistical context preserves exact findings and caller state. These are controlled mechanism regressions, not an exact live replay or live qualification.

Real-Jupyter smoke also exercises required-feature rendering/native completion, refusal to upgrade an unfinished child, and preserved statistical context. Existing initial/revision/evaluator/native strict failure and SQLite continuity regressions remain required.

## Verification at initial publication

Target5 passed (two dependency warnings); patcher/parity/state211 passed (four warnings). Notebook/graph validators, Black, Flake8 and diff check passed. Two regenerations are byte-identical. Raw notebook SHA256 `6a0152f4746424eb3b81270a109140fddb691a819e9747e646e62f065fb55a8e`; LF-normalized `7bc81281e1d08d8b27f699c8791e38cccda7c249037fea3177439223c7bb443b`.

Full focused/broad no-key and scientific/notebook/proof-runner Jupyter suites are still running and are not claimed green here. Final exact-HEAD command/exit/count receipts, hosted Linux/Windows and actual fresh Antigravity review will be posted in PR conversation before merge or retry. Overlapping test selections must be counted once. #129 deferred, #165 open; campaign stopped pending those gates, existing spend limits unchanged.


Commands (offline, recovery0; real-kernel env IDD_RUN_KERNEL_SMOKE=1 / IDD_KERNEL_NAME=idd-preflight):

```text
python -m pytest -o addopts= tests/unit/test_native_completion.py -k 'viz_required_custom_features or viz_child_unfinished_is_never or viz_statistical_context' -q --tb=short
python -m pytest -o addopts= tests/unit/test_patcher_integrity.py tests/unit/test_core_parity_guard.py tests/unit/test_canonical_state_graph.py -q
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
python -m pytest -o addopts= tests/unit/test_datetime_conversion.py tests/unit/test_native_completion.py tests/unit/test_figure_provenance.py tests/unit/test_analyst_revision_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_viz_join.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py -q
python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_error_handling_framework.py test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q
python -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -v
python -m black --check tests/smoke/test_real_kernel_notebook.py tests/unit/test_native_completion.py
python -m flake8 tests/smoke/test_real_kernel_notebook.py --max-line-length=120 --extend-ignore=E203,W503,E501
python _patch_notebook.py  # twice, exact-byte/hash comparison
```
