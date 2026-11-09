# CP5 native visualization completion correction

Base: `ccdc60a1678fc4ca3c1cf3736770abc915665a3a` (merged PR #208).

Retail attempt `3a06532a-9a64-4a01-8447-f3f676442355` verified the original
scenario request, completed cleaning with a matching 180-row, 13-column CSV,
and returned native analysis and four chart assignments. Three workers wrote
completed figure records. The scatter worker generated an actual 1500×1050
PNG with product-line colors, unit sizing, a fitted trend and highlighted
outliers, then submitted a native `VisualizationResults` with
`finished_this_task=True` and a child with `finished_this_task=False`.
The native worker correctly raised
`ValueError("Visualization worker did not complete its required figures")`.
The runner exited 1 at Cell 75 after 259.6 seconds, closed the kernel, and
did not reach evaluation or the production validators. The PNG was inspected
only as failed-run evidence. In the earlier attempt
`2aa0c868-674b-4f32-9115-88f4fb6a5601`, the same check stopped a claimed
success with an empty figure list, despite a saved product-line PNG.

Cell 44 now rejects these two contradictory success shapes at the model
response boundary. The first contradiction in the current assignment returns
a paired error on the original native tool-call ID, retaining the original
assistant tool call. The invalid parsed response is not published as
`structured_response`. LangChain can then ask the model for a new result.
The model must verify the assigned PNG and submit its actual metadata or
report unfinished work. A second contradiction in the same assignment
raises an actionable error. The bound comes from checkpointed message history
after the latest runtime context, so an older assignment does not consume it.

The existing worker validation, assignment ID, path and PNG provenance checks
remain authoritative. An honestly unfinished envelope still fails immediately.
The middleware neither creates figure metadata nor changes any completion
flag or evaluator grade. Registered tools, `ToolStrategy(VisualizationResults)`,
the schema, checkpointer, recursion limit, shared hook, graph, and both
production validators remain unchanged.

On the base, five focused cases produced four expected failures (the first
contradictory response stopped the worker) and one pass (honest unfinished
output remained rejected). After the fix, all five passed with two dependency
warnings. The generated-factory tests use a real plotting tool and two
assignments on the same inner thread. They verify paired errors, unmodified
rejected arguments, genuine new native completions, real PNG bytes reaching
the join, and rejection after the bounded round. Black and Flake8 passed.

Only `create_visualization_agent` in generated Cell 44 changes. Two
regenerations produced identical notebook bytes:

- Raw SHA-256: `5abf45bb8c27c262425ce3d1cc25fe42381ee369afd1d41cb734590930e1ac0b`
- LF-normalized SHA-256: `224732165520023b91a111f3da2a8bc3b684291ef5f1b1f86293221cdb819601`

The failed attempt's saved-response subtotal was $0.04236671, bringing the
campaign subtotal to $1.96731322. This is partial checkpoint usage and does
not account for all provider charges. `sample_dirty` did not start. The PR
records exact candidate validation and independent review before any merge.
