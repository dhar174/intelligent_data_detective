# CP5: Scope visualization worker inspection to its assignment

Base: `b2e361f36845ee782e9e1743a298ea82f74fb841` (PR #211 merged).

## Evidence

Strict retail attempt `db7db2c5-3183-4025-bdb8-e38a31df467f` cleaned the expected 180-row fixture and dispatched three genuine visualization assignments. The isolated workers still shared the run's artifact directory. Public inventory results showed both custom PNG filenames (bar and box) together: the seven logged inventory responses reported 3, 3, 7, 7, 8, 9 and 10 files. These are stream events, not counts per worker or LangGraph steps. The outer checkpoint maps the successful child write to `netrev_region_channel` and the strict unchanged-retrieval error to the third send, `satisfaction_by_productline`. The error was observed before the join; neither the evaluator nor the production validators ran. The supervising process stopped the attempt (exit 1, 556.6s). The interrupted runner did not record completed kernel cleanup. No worker state was repaired in place.

The tool outputs demonstrate that assignment isolation at the checkpoint did not isolate filesystem inventory. They do not establish the model's intent, the arguments of individual live calls, or that inventory exposure is the only failure. The scatter rendered multiple candidates, and interim chart quality still requires evaluation.

## Change

Only the generated visualization factory and worker assignment (Cells 44 and 57) change. For a dispatched `VizSpec`, the factory limits `list_visualizations` and `get_visualization` to verified current-run PNGs from paired plotting receipts after the newest assignment and its unique reserved custom output path. It passes a temporary inventory to the existing native tools and disables their shared directory scan for this role. The caller cannot substitute another root or get a sibling preview by direct path. The assigned ID resolves to the saved custom PNG when present, otherwise the latest built-in render. The retrieval result is provisional and supplies no completion or approval.

Native `VisualizationResults`, the existing PNG and join validation, the evaluator and analyst's inventory tools, strict error propagation, checkpointed state, and both production validators remain in place. No node, edge, State field, or canonical module changes.

## Regression

The network-denied generated factory/worker test renders a PNG with the real Python or built-in plot tool beside a sibling PNG. The final baseline selection failed twice on sibling exposure (exit 1; 230 deselected, two warnings). After the correction, both variants passed: only the actual assignment PNG is listed, a sibling lookup receives a paired error, and the assigned-ID lookup serializes the exact PNG bytes to the model. The worker then submits native metadata and the real join preserves the assigned ID. Earlier test-fixture failures (incomplete `VizSpec`, scripted-response scheduling and inspection of the wrong message) are separate from the verified baseline.

The focused native/provenance/evaluator/join/SQLite selection passed 368 tests (six warnings). The patcher/parity/canonical selection passed 211 tests (four warnings). Formatting and lint passed.

The notebook regenerates byte-for-byte with 99 cells, and only Cells 44 and 57 differ from the base. Graph and notebook integrity validators passed. PR receipts record exact-HEAD local, hosted, and independent review gates. Live qualification requires fresh strict proof after the reviewed merge; this repair does not close #165 or include #129.
