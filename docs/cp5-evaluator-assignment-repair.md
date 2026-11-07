# CP5 evaluator assignment repair

Base: `d128494b1c4266fa485758a8cb00da0ff4bfeb83`, freshly fetched from main.
Branch: `codex/cp5-evaluator-assignment`, isolated implementation worktree.
No live execution, provider requests, credential loading, merge, or issue closure is part of this repair.

## Evidence and scope

Retail attempt `24c1ef75-e1a8-439c-9a43-2b21ea992d6a` completed initial and revision analysis through native AnalysisInsights. The subsequent evaluator returned `grade="revise", finished_this_task=False`; the existing strict check correctly raised `viz_evaluator integrity: Unfinished native VizFeedback`. Both production validators were NOT_REACHED. This repair does not reinterpret that failed run as successful.

The checkpoint still held the analyst's "Revise the visualization plan" assignment in `next_agent_prompt`. The generated evaluator node used that value verbatim as its final HumanMessage. Its factory instructed evaluation, while its final assignment requested revision work. An offline generated-function model spy confirms the conflicting final assignment reaches the model. A deterministic instruction-following model reproduces the unfinished-feedback failure when given that task. This establishes the executable instruction conflict; it does not prove why the particular live model chose its flag.

Before changing production code, eight new cases ran against the baseline: four failed and four strict-failure cases passed. The four failures cover normal and pressured message delivery, the instruction-following failure reproduction, and a reused evaluator thread. The retained red receipt is `evaluator-assignment-red.txt` in the local CP5 evidence directory.

## Correction

The active native evaluator transformation in `_patch_notebook.py` creates a final named `HumanMessage(name="evaluation_assignment")` after emergency-assignment handling. That message explicitly assigns evaluation of the current figures and makes the newest review assignment authoritative. The prior assignment and evaluator feedback remain labeled background. Current figure metadata, specifications, analytical context, and the original user request continue to arrive through the existing rendered runtime context.

A completed review can natively return either acceptable or revise with `finished_this_task=True`. A revise verdict carries concrete feedback and a redo list; the supervisor, analyst and visualization workers retain their existing responsibilities. A genuinely unfinished review still returns false and raises the same strict integrity exception. No response flag, evaluator grade, schema output, or figure is synthesized or coerced.

Only generated Cell 57 changes. The shared history hook, factory, tool bindings, retrieval guard, recursion limit, revision budget, model settings, retries, schemas, reducers, and graph topology are unchanged. The existing shared hook already protects the newest HumanMessage under pressure; the new pressure regression verifies delivery after actual summarization without altering checkpoint-style caller messages.

## Offline verification

Final disjoint pytest selections total **515 passed**, with no skips or failures. The separately repeated 26 evaluator tests are included in that total, not added again.

```text
python -m pytest -o addopts= tests/unit/test_viz_evaluator_completion.py tests/unit/test_native_completion.py tests/unit/test_analyst_revision_completion.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py tests/unit/test_proof_runner.py tests/unit/test_live_runner_configuration.py -q
# exit 0: 288 passed, 4 warnings, 95.68 seconds

python -m pytest -o addopts= tests/unit/test_patcher_integrity.py tests/unit/test_core_parity_guard.py tests/unit/test_canonical_state_graph.py test_error_handling_framework.py -q
# exit 0: 227 passed, 4 warnings, 33.07 seconds; includes all 16 mandatory error-handling tests

python -m pytest -o addopts= tests/unit/test_viz_evaluator_completion.py -q
# exit 0: 26 passed, 2 warnings, 22.92 seconds

python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb -v
# exit 0: 99 cells, 42 code cells compiled

python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
# exit 0: 15 nodes, 33 edges, 70 State fields, 39 reduced fields

python -m black tests/unit/test_viz_evaluator_completion.py
git diff --check
# exit 0
```

The existing actual-generated SQLite integration passes with real native analyst completion, visualization workers, evaluator, plots, checkpoint reopening and isolated threads. The new reused-thread regression exercises completed revise followed by completed acceptable through the same native evaluator/checkpointer. Unfinished revise and acceptable outputs both fail even with recovery enabled.

An intermediate combined run reported 171 passes and two new fixture failures: LangGraph assigned IDs to fresh unidentified messages. The fixture now uses explicit IDs, matching retained checkpoint history. Its final assertions verify both IDs and contents remain unchanged. No production history behavior was broadened to address that fixture issue.

A fresh `idd-preflight` Jupyter kernel also ran a network-denied, credential-scrubbed probe using the generated factory and node. Three native invocations on one thread verified completed revise, completed acceptable, and strict rejection of unfinished feedback. The probe exited 0. The Windows host emitted the existing Proactor/selector warning. Other observed warnings concern dependency deprecations and Matplotlib tick labels; none remains a failing check.

Two patcher runs produced byte-identical notebook output:

- Raw Windows SHA-256: `d22acbec698516cf9dc6df5a3aab78d3f5c922b3a2c30a53d2731b1ad48c4050`.
- LF-normalized SHA-256: `ee318e3d36fad4f4cbdf0e3344c2d562a06389a0e088bf9e615f7222bb7f5b1c`.
- Only Cell 57 source differs from base; all seven canonical modules and the manifest match base after LF normalization.

The existing Windows no-key workflow already selects the modified evaluator test module. No hosted CI was dispatched or claimed green for this uncommitted candidate. Local receipts are stored outside the source checkout under the CP5 attempts directory, preserving historical run evidence.

## Remaining acceptance

This is an assignment-boundary repair. The two targeted live plot requirements and the eventual report remain unqualified; offline scripted responses cannot establish live figure acceptance. No further live attempt was started. PR #129 remains deferred and open, issue #165 remains open, and owner approval is required before another live run.
