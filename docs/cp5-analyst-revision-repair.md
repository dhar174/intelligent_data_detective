# CP5 analyst revision repair — forensic report

**Status:** offline repair, CI wiring, and local verification are complete. Hosted CI and external review remain pending. The archived live attempt failed and is not CP5 acceptance evidence. No campaign retry, merge, or Issue #165 closure is authorized by this work.

## Attempt provenance and outcome

The report is tied to CP5 Attempt 1 on `retail_orders`, source commit `88bac1c4c0b5687ceaee9cc01e9514270380214f`, tree `6e6bff4f9e06d052466831dfeadf7d948cb8f731`, and the pinned 99-cell notebook with LF SHA-256 `1016d3ab2579d8974d515c37d2a6d464061f7ad7a594e8a7e26cc323dcefd64d`.

The run lasted 658.2 seconds, from 2026-11-05 16:30:54 UTC to 16:41:53 UTC, and made 111 HTTP requests. The first failure was `STREAM_FAILURE` at notebook cell 75. The failure was in the `analyst` node with `GraphRecursionError`; the run used strict fail-closed mode (`IDD_ALLOW_RECOVERY=0`) and did not invoke a final-hop recovery shim. The attempt cost recorded in the bundle is $0.14, with reconciled cumulative CP5 spend at $0.92. Attempt 1 failed, `cp5_pass` is false, and Attempt 2 is not authorized.

The archived evidence records both scatter and box plot outputs before the analyst revision failed. Their presence does not establish artifact-quality acceptance; the source bundle marks independent artifact inspection as pending.

## Analyst revision trace

The captured cell 75 stream contains 14 `list_visualizations` and 34 `get_visualization` name-header occurrences over the whole run. From the first revision marker through the end of the stream, it contains 13 `list_visualizations`, 26 `get_visualization`, and one `get_column_names` occurrence. The failed loop is associated with a `step 185` marker and an error reporting a recursion limit of 160.

These are header counts from the executed notebook stream, not invocation IDs. The failed analyst subgraph calls are absent from the accessible SQLite message writes, so this archive does not provide their call IDs, arguments, argument fingerprints, or per-message roles. The sanitized evidence JSON keeps this distinction explicit. It includes IDs, `ai` roles, safe allowlisted argument values, argument fingerprints, and checkpoint observations only for 27 deduplicated outer-thread message tool calls.

### Why the marker is 185 while the budget is 160

The archived error reports a recursion budget of 160. The installed LangGraph version recorded for this run is 1.2.16. In its installed `langgraph/pregel/_loop.py`, the Pregel loop initializes `step` from checkpoint metadata plus one and sets `stop` from that starting step plus the configured recursion limit (lines 1880–1881 and 2138–2139). A reported step can therefore include a checkpoint-relative offset; the marker does not establish a configured budget of 185.

The accessible SQLite database contains 13 outer-thread checkpoints, with metadata steps -1 through 11. It does not expose the analyst subgraph's starting checkpoint step, so the precise offset behind the cell 75 step-185 marker cannot be reconstructed. The candidate patcher diff contains no recursion-limit change; the inner budget remains 160 and is not raised to 185.

## Evidence integrity and limits

The archived checkpoint database was opened with SQLite URI `mode=ro` and `PRAGMA query_only=ON`; `PRAGMA integrity_check` returned `ok`. The executed notebook and pinned source each have 99 cells, and the source notebook LF hash matches the evidence bundle. The raw and LF hashes recorded for the archived console log, executed notebook, and pipeline log all match their original evidence manifest values.

The original bundle marks the capture `PARTIAL_CAPTURED`; provenance, run association, and automatic validation were not reached, and independent artifact inspection remains pending. A safe scan found seven `warning`/`warn` token occurrences in the archived console log and none in the pipeline log. Their text is not reproduced or classified here.

The machine-readable, sanitized extraction is [forensic-evidence.json](evidence/cp5-analyst-revision/forensic-evidence.json). It contains selected provenance, checkpoint metadata, call IDs, allowlisted safe outer-call arguments, SHA-256 fingerprints, and aggregate inner-revision header counts. It contains no raw logs, prompts, model reasoning, credentials, or unfiltered tool arguments.

## Separate current-source compatibility note

The installed LangChain version is 1.4.6. Its `langchain/agents/factory.py` model node builds the model request from `request.messages` (lines 1502, 1504, and 1506). The pinned notebook has `llm_input_messages` hook output in source cells 40 and 43. An internal read-only review flagged that field as a current-source compatibility concern because the installed model node does not consume it. This observation is separate from the archived analyst recursion failure and is not live root-cause proof. The controlled model-bound spy remains the authority for the hook behavior and scoped pin; no shared pre-model hook change is part of this repair.

## Controlled before/after contract probe

The local generated-node/factory probe and canonical Responses serializer comparison are recorded in [effective-contract.json](evidence/cp5-analyst-revision/effective-contract.json). This is a controlled serialization experiment with no provider client or credentials; it is not the missing raw request from the failed live call.

| Controlled observation | Before repair | After repair |
| --- | --- | --- |
| Native response contract | `AnalysisInsights` bound; `respond` unavailable | `AnalysisInsights` bound; `respond` unavailable |
| Positive instruction to use `respond` | Present | Removed |
| Genuine supervisor feedback visible | Yes | Yes |
| Prior findings visible | No | Yes |
| Runtime-context messages | 0 | 1 |
| Serialized role list | `system`, `user`, `assistant`, `assistant` | Same roles plus a final `user` runtime-context message |
| Scripted native response | `AnalysisInsights`, one model call | `AnalysisInsights`, one model call |
| LangChain factory tool choice | `any` | `any` |
| Canonical SDK serializer tool choice after installed adapter normalization | `required` | `required` |

The role arrays contain no `developer` role; they identify message roles only, not message contents. The tool-choice row is deliberately split by layer: the factory spy observes LangChain `any`, while the installed `ChatOpenAI.bind_tools` implementation inherited by `MyChatOpenai` normalizes it to `required` before SDK serialization. The canonical serializer probe asserted `required`; this report does not claim a wire-level `any` choice or an actual provider request.

The archived supervisor `next_agent_prompt` already contains genuine feedback and a redo instruction, and that instruction survives as an `AIMessage`. Feedback was therefore not wholly absent. The controlled repair review instead found a contract mismatch: a focused human task requested unavailable `respond`; `_safe_analyst_invoke` stripped the rendered system context; and prior findings plus explicit figure metadata were not forwarded (the fallback list exposed two filesystem-known figures). The serialized role listing alone does not establish the original request's message contents. The original failed provider payload is unavailable, so the evidence cannot establish a single causal model-side explanation.

The initial analyst leg did complete despite the conflicting instruction. The repair carries prior findings, figure metadata, and runtime context into the actual revision request, keeps the declared native `AnalysisInsights` completion contract, and refuses to synthesize success from `respond` or a stale response. This is consistency with the native contract and controlled context probe, not live efficacy evidence.

## Repair boundary and CI gate

The candidate repair adds `_apply_cp5_analyst_revision_patch` to the active native-completion transform. No canonical model, state, or tool schema changes are in scope. Native-completion review found that a same-thread missing completion clears the stale structured response and enters the native recurse path; it did not establish a separate stale-response reuse defect. The repair stays focused on analyst revision completion.

The associated bounded guard tracks repeated visualization retrieval results within the analyst history, fails closed when the same known result is requested repeatedly without progress, and allows distinct or preview retrievals. A successful relevant dataframe tool call resets the repetition streak. Native malformed, unfinished, or exceptional outputs continue to fail closed; supervisor handoff retains the genuine feedback and requires a fresh declared native response. These behaviors are covered by the test map below; they do not alter canonical schemas or the 160-step recursion budget.

Root reports the frozen local candidate notebook still has raw SHA-256 `88356e218fb6645fc7ec8f17598e1218c12c1898ed10993c50f83b51cd1b0804` and LF SHA-256 `7614c5f342b9033d97f4ea4c392a559d7fb312b85fafb3063b4bed2a50a109cc`, with source changes limited to cells 25, 44, and 57. All seven canonical modules and `idd_manifest.json` were rechecked and match the pinned base after LF normalization. These are frozen local candidate hashes, not a commit or hosted provenance record.

The Windows real-kernel workflow has a no-key gate for `test_native_completion.py`, `test_viz_evaluator_completion.py`, the new `test_analyst_revision_completion.py`, and `test_cp5_revision_route.py`, with recovery disabled in the step. The A–H labels below are acceptance concepts, not disjoint test groups; the SQLite route test covers B, F, and H together.

| Acceptance concept | Focused coverage |
| --- | --- |
| **A — Initial completion** | `tests/unit/test_analyst_revision_completion.py::test_analyst_initial_completion_uses_native_schema_without_respond`; `tests/integration/test_cp5_revision_route.py::test_sqlite_generated_revision_route_keeps_real_plots_and_isolated_threads` |
| **B — Genuine revise route uses feedback, produces revised specs, and ends natively** | `tests/unit/test_analyst_revision_completion.py::test_analyst_revision_receives_real_feedback_png_inventory_and_supervisor_instruction`; `tests/integration/test_cp5_revision_route.py::test_sqlite_generated_revision_route_keeps_real_plots_and_isolated_threads` |
| **C — No effective nonexistent `respond` while `AnalysisInsights` is bound** | `tests/unit/test_analyst_revision_completion.py::test_unknown_respond_tool_never_creates_a_fabricated_completion`; `tests/unit/test_analyst_revision_completion.py::test_analyst_revision_without_native_response_cannot_reuse_prior_completion` |
| **D — Stale inventory is guarded while legitimate progress remains allowed** | `tests/unit/test_analyst_revision_completion.py::test_four_identical_visualization_lookups_fail_before_another_model_call`; `tests/unit/test_analyst_revision_completion.py::test_alternating_known_visualization_results_fail_after_three_redundant_results`; `tests/unit/test_analyst_revision_completion.py::test_successful_dataframe_tool_resets_repeated_visualization_lookup_streak`; `tests/unit/test_analyst_revision_completion.py::test_distinct_and_preview_visualization_retrievals_are_legitimate_progress` |
| **E — Unfinished or insufficient evidence fails without fabrication** | `tests/unit/test_analyst_revision_completion.py::test_analyst_native_unfinished_output_fails_with_integrity_error`; `tests/unit/test_analyst_revision_completion.py::test_unknown_respond_tool_never_creates_a_fabricated_completion`; `tests/unit/test_native_completion.py::test_native_worker_rejects_invalid_required_completion` |
| **F — A later acceptable revision after new figures proceeds** | `tests/integration/test_cp5_revision_route.py::test_sqlite_generated_revision_route_keeps_real_plots_and_isolated_threads`; `tests/unit/test_analyst_revision_completion.py::test_analyst_revision_receives_real_feedback_png_inventory_and_supervisor_instruction` |
| **G — Malformed payloads, invalid tools, native exceptions, and recursion fail closed** | `tests/unit/test_analyst_revision_completion.py::test_analyst_malformed_native_payload_preserves_structured_output_error`; `tests/unit/test_analyst_revision_completion.py::test_unknown_respond_tool_never_creates_a_fabricated_completion`; `tests/unit/test_analyst_revision_completion.py::test_pinned_base_nonempty_visualization_retrievals_reach_recursion_limit`; `tests/unit/test_native_completion.py::test_strict_native_error_preserves_exception_without_recovery`; `tests/unit/test_native_completion.py::test_real_agent_nonterminating_tool_fails_strictly` |
| **H — Canonical SQLite continuity, IDs/reducers/provenance, and thread isolation** | `tests/integration/test_cp5_revision_route.py::test_sqlite_generated_revision_route_keeps_real_plots_and_isolated_threads`; `tests/unit/test_analyst_revision_completion.py::test_analyst_retains_history_and_emergency_without_mutating_caller_state` |

The archived hash-only retrieval trace is [retrieval-trace.json](evidence/cp5-analyst-revision/retrieval-trace.json). The first revision inventory at header line 989 and final inventory at line 1616 each returned two visualizations. The earliest observable repeated rendered result was a `get_visualization` result at header 1043/display step 41 whose body hash matched header 1015/display step 33; a different rendered result appeared between them at header 1030. The extraction covers 40 revision header events through the final inventory before the streaming-error marker. Header lines/display steps are not invocation IDs, and equal result hashes do not prove equal tool arguments or establish that repetition alone caused nontermination. Inner call IDs, arguments, and message roles remain unavailable.

The following local runs are separate results and their counts should not be added together:

| Run | Result | Evidence boundary |
| --- | --- | --- |
| Grouped no-key gate | 198 passed, 3 dependency warnings, 16.58s | `python -m pytest -o addopts= tests/unit/test_core_parity_guard.py tests/unit/test_canonical_state_graph.py test_error_handling_framework.py tests/unit/test_proof_runner.py tests/unit/test_live_runner_configuration.py -q`. Includes the mandatory error-handling suite; counted once. |
| Patcher integrity | 118 passed, 1 `fitz` `FutureWarning`, 24.58s | `python -m pytest -o addopts= tests/unit/test_patcher_integrity.py -q`. |
| Native completion | 107 passed, 2 dependency warnings, 30.62s | Existing native-completion suite. |
| Analyst revision completion | 14 passed, 2 dependency warnings, 19.68s | `python -m pytest tests/unit/test_analyst_revision_completion.py -q --maxfail=1` (exact command confirmed by the test agent). |
| CP5 revision-route new case | 1 passed, 4 warnings, 10.13s | This case is included in the broader run; do not add it again. |
| Combined focused five-module gate | 168 passed, 4 warnings, 81.93s | `python -m pytest -o addopts= tests/unit/test_analyst_revision_completion.py tests/unit/test_native_completion.py tests/unit/test_viz_evaluator_completion.py tests/unit/test_cp5_report_grounding.py tests/integration/test_cp5_revision_route.py -q`. Overlaps the individual suite results above. |
| Broad no-key local suite | 1164 passed, 9 skipped, 6 warnings, 143.45s | `python -m pytest -o addopts= test_validate_run.py tests/unit tests/integration test_intelligent_data_detective.py test_error_handling_framework.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py test_validate_notebook_integrity.py test_prompt_formatting.py test_prompt_template_fixes.py -q`; provider key variables were blank and recovery was disabled. Four warnings were dependency warnings (`langchain_experimental`, `trustcallSend`, `langchain_community`, `fitz`) and two were Matplotlib `set_ticklabels` warnings from real-plot integration. |
| Narrow skip audit | 94 passed, 9 skipped, 11.44s | `python -m pytest -o addopts= test_validate_run.py tests/unit/test_registry.py -q -rs`. The skips were one missing Run88 fixture, two missing `pyarrow`, and six POSIX permissions/ownership cases not applicable on Windows. |
| Full local real-kernel suites | 55 passed, no warnings or skips, 482.24s | `python -m pytest -o addopts= tests/smoke/test_scientific_kernel.py tests/smoke/test_real_kernel_notebook.py tests/smoke/test_proof_runner_kernel.py -q`; `IDD_RUN_KERNEL_SMOKE=1`, `IDD_KERNEL_NAME=idd-preflight`, provider key variables blank. This is not hosted CI. |
| Added-kernel test | 1 passed, 18.02s | Earlier single-test result; not a full kernel gate. |

Notebook integrity also passed with 99 cells and 42 compiled code cells. Graph validation passed with 15 nodes, 33 edges, 21 conditional edges, 70 State fields, and 39 reducers; it recorded a soft-budget warning at 11.85 seconds against a 10-second threshold.

A supplemental replay against the unchanged pinned base `88bac1c` was captured after the repair implementation. It observed two PNGs, 160 graph steps, and 40 cycles across both patterns; it is not a pre-implementation production run or CP5 acceptance proof. The replay harness initially lacked `plt`, `ALLOWED_SPEC_KEYS`, `hashlib`, `base64`, and `glob` in its namespace. Its initial assertion expected 160 model calls and was corrected to 40 cycles. Those were harness setup/assertion errors, not production defects. The earlier `red.txt` artifact predates the implementation and remains untouched.

The pre-change baseline supplied for this task was 111 passing tests with two dependency warnings. The local no-key broad suite and local real-kernel gate are complete. Hosted CI and external review are separate gates recorded against the exact Draft PR HEAD; the counts above are local results.

## PR #129 overlap check

PR #129 was open at head `7fe2c97fc376d2075a72b8d2986a802332cac8de` when checked. It changes `_patch_notebook.py` and both notebook files among its six changed paths. Its patcher adds a general source pass for `summary_text` concatenation across code cells and clears outputs/execution counts for rewritten cells. It does not target the analyst revision completion path, so no direct runtime semantic conflict was found. The shared patcher and generated notebook create a file-level integration risk: if PR #129 lands before this repair, regenerate the patched notebook and rerun the relevant integrity and completion gates.

## Stop condition

This is an offline repair record. It does not authorize a new provider run, a retry of Attempt 2, merge, or Issue #165 closure. The CP5 live campaign remains stopped for owner review. Hosted results and exact HEAD are recorded in the associated Draft PR; this file records the completed local-validation snapshot.
