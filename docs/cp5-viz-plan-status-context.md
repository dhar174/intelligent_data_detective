# CP5 visualization plan status in rendering context

Base: `485ff31384ab061895c9538534b8e01fd7753fc4` (merged PR #206).

Retail attempt `41bd9e58-39e4-4944-a19c-afc5c476877f` completed cleaning
and analysis. A visualization worker produced the assigned 1000 × 800
scatterplot with discount coloring, unit-price sizing, an OLS line, and
the Pearson correlation annotation (`r = 0.815`, independently verified
against the actual cleaned CSV). Its native reply described the figure as
complete, but `VisualizationResults` set the parent to finished and the
`DataVisualization` child to unfinished. The strict worker guard correctly
rejected the response. The original attempt and PNG remain preserved;
no evaluator, reporting or final validators were reached.

The generated worker rendered the upstream `VizSpec.finished_this_task`
value in three copies of the assignment on each effective model request.
That field describes the plan, while the same field name in the native
response describes the rendered figure. The existing instructions explained
the distinction, but the irrelevant Boolean still appeared next to the plot
requirements. A scripted model spy using the real factory, generated worker,
real plot and join confirmed all three copies before and after rendering.
Both baseline cases failed the projection assertion, once for each Boolean
value. This proves the input exposure, not what caused a private model choice.

The active transformation omits only `finished_this_task` from the two
serialized spec projections in Cell 57. All other spec values, including
figure identity, data binding, columns, axes, grouping and rendering
instructions, remain intact. The canonical `VizSpec` and checkpoint state
are unchanged. The explicit native completion contract and the existing
strict rejection of empty or unfinished results remain unchanged; a
finished child is never synthesized or inferred from the presence of a PNG.

The regression verifies both upstream values, all three rendered copies
on each request, a real PNG, stable canonical state and native completion.
Existing tests continue to reject an explicitly unfinished child even when
a readable PNG exists. The real-kernel probe exercises the pending-plan case.
Exact-HEAD local, hosted and Antigravity receipts are recorded with the PR.

No public schema, shared hook, routing, model settings, artifact ownership,
or production validator changes. This correction does not guarantee that
a live model will always complete. Both scenarios still require real artifact
inspection and the unchanged **12/12 + 9/9** production checks. Keep #129
deferred and #165 open. No live execution during the repair.
