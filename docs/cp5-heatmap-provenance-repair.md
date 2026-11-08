# CP5 heatmap save and provenance repair

## Source and authorization

Verified main/base: `ff8e84e22a0bc02033082cb5833d05b27c637e5a`, PR #189 merge. GitHub reports merged at 2026-11-08T05:51:00Z; accepted HEAD `536f5fbead7f2c038be881fa3987a4d524518369` is an ancestor. Implementation branch: `codex/cp5-heatmap-provenance`, isolated worktree. Changes remain uncommitted for owner review.

Only offline implementation and tests are authorized. No production notebook execution, credentials, providers, paid spend, checkpoint resume, live retry, commit/push/PR publication, hosted CI dispatch, Antigravity review request, merge, or issue closure is part of this delivery. #129 stays deferred, #165 open, campaign stopped. Historical execution directories are untouched.

## Verified defect and minimal correction

The actual generated heatmap on a registered numeric retail fixture returned `Failed to generate correlation heatmap: name 'cols' is not defined`. Its save block referenced an unbound outer cols variable, wrote to the old figures directory, used a histogram marker, and hid the saved path from native model input.

The existing active figure-provenance transformation now includes this fifth plot tool. Twelve added patcher lines bind filename columns to final `corr.columns`, sanitize them with the existing scatter/box convention, use the correlation_heatmap marker, and reuse existing current-run canonical saving/model-visible metadata transformations. No new production helper is introduced.

Only generated Cell 32 changes. The function signature/defaults/decorators and all source preceding its save block are unchanged, including correlation, selection/coercion, sampling, clustering/order, masking and rendering. Other Cell 32 definitions and all other cells (including frozen RuntimeCtx, native worker/join, shared hook and model configuration) match base. Seven canonical modules, manifest and both production validators match base after LF normalization.

Successful content contains the authoritative image_path, plot_type, dataframe_id and actual columns through the existing JSON helper. The native `(content, artifact)` contract retains bytes/base64 in artifacts; native Responses model input excludes those payloads. No filename guessing, ID rewriting, synthetic completion, schema expansion or strict-check relaxation occurs.

## Red/green coverage

Twenty-six additional cases extend the existing provenance module (64 total): one common plotting-tool wire case; 20 combinations of selection/order, Seaborn/Matplotlib fallback, and bytes/base64 mode; four genuine failures; one native heatmap worker-to-join case. The latter uses existing VizSpec type auto with explicit heatmap instructions, the actual frozen RuntimeCtx and genuine native VisualizationResults, preserving the assigned ID and reducers.

Before production edits: `python -m pytest -o addopts= tests/unit/test_figure_provenance.py -k heatmap -q` returned exit 1: **23 failed, 3 passed, 38 deselected, 2 warnings, 24.78 seconds**. The three passing existing-behavior controls cover missing DataFrames, insufficient numeric columns, and empty filtered data. No global cols or artifact environment override was supplied.

The first green attempt returned exit 1:25 passed / 1 failed (27.51 seconds). Its remaining failure was a test assertion counting the join's additional memory write as a worker write; the assertion was moved to the worker boundary. No production change was needed. The corrected focused run includes all 26 passing heatmap cases:

```text
python -m pytest -o addopts= tests/unit/test_figure_provenance.py tests/unit/test_native_completion.py tests/unit/test_viz_join.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_cp5_report_grounding.py tests/integration/test_viz_join_reporting.py tests/integration/test_cp5_revision_route.py -q
# exit 0:284 passed, 6 warnings, 133.42 seconds
```

Tests verify plotted matrices against pandas, actual included/order-correct columns, unique heatmap filenames, current-run viz_dir ownership, readable saved/encoded PNGs, real serialized model content, native two-call completion and assigned-ID preservation through the real join. Failure cases retain missing/insufficient/empty/save-error responses and publish no saved path. Existing stale/corrupt/conflicting asset and incomplete native-completion checks remain included.

```text
python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_error_handling_framework.py test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q
# exit 0: 1,259 passed, 9 skipped, 8 warnings in 202.95 seconds; all 16 mandatory errors included
```

Warnings in the focused/broad runs are dependency deprecations and existing Matplotlib tick-label warnings. The 9 skips are reported as skips, never passes.

```text
IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight python -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -v
# exit 0: 56 passed in 475.62 seconds
```

The disjoint combined no-key and kernel selections provide **1,315 distinct passing cases and 9 skips**. Focused passes and reruns are not added again. The real-kernel provenance case now exercises native heatmap worker-to-join completion and Matplotlib fallback with byte output. Strict proof failure, source isolation, SQLite lifecycle and migration negative controls all pass.

No overlapping focused/rerun/platform totals are added as distinct cases. Formatting check for both changed test modules passes. Receipts are retained outside source under `C:/Users/darf3/Documents/cp5_attempts/heatmap_repair_ff8e84e_20261108`.

## Determinism and invariants

Two patcher regenerations produced identical bytes. Raw Windows notebook SHA256: `3ea9522fced4ab336c2e6c669701763f78d1b329151cb59f4a49d1462d870998`; LF-normalized SHA256: `6f645b97f6243bf526e489cbccd489d3f431e90e6dc0da30363b3fbc9b4ad03e`. Exactly 99 cells remain, with 42 code cells compiling. Graph validator: 15 nodes, 33 edges, 70 State fields, 39 reducers, 8.57 seconds; exit 0. Notebook validator: exit 0. Both production acceptance thresholds remain 12/12 and 9/9; they were not executed as live acceptance gates.

Reproducibility and validation commands (each exit 0):

```text
python _patch_notebook.py  # run twice; exact bytes equal
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
python -m black --check tests/unit/test_figure_provenance.py tests/smoke/test_real_kernel_notebook.py
```

## Separate unchanged findings and limitations

A pre-existing Pydantic serializer warning was observed during the planning probe: the heatmap's default figsize is a tuple while its existing annotation is List[float]. This is unchanged; regression render cases supply an explicit valid list. A separate post-repair offline diagnostic invoked the actual tool with only df_id (all defaults), verified its current-run PNG and visible path, and confirmed that the serializer warning remains nonfatal. This diagnostic is not added to the pytest case count. No unrelated parameter, statistical or renderer correction is included.

This is local offline qualification only. No new Linux/Windows hosted results, independent review, live figure acceptance or CP5 pass is claimed. Publication, exact-HEAD hosted CI/review and any further live campaign authorization remain separate owner steps. Historical figure-provenance reports/failure evidence are preserved.
