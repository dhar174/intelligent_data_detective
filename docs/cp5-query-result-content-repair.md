# CP5 DataFrame query results in model requests

Base: `dc3dfb68b0e10e1a30de220c554789be88c93a08` (merged PR #207).

In retail attempt `b9bb4685-2fa2-4f2d-a4cb-574e6892bfa0`, native cleaning
and analysis completed and three workers returned figures. The evaluator gave
a genuine `revise` verdict for the grouped bar chart. The saved PNG had a stray
zero label, even though the twelve bar values and region order agreed with the
exported CSV. The outer graph routed to the analyst. After that evaluator
message, the stopped console contained twelve `query_dataframe` result headers
with the public response `Query successful.`, and no subsequent native
`AnalysisInsights` result. The operator stopped the attempt after 932.7 seconds
(exit 1); no native exception or validator verdict was recorded. This is an
operator stop, not a native recursion failure. The historical attempt was
preserved and `sample_dirty` did not run.

A network-denied reproduction used the generated analyst factory and node,
the real decorated query tool, and a scripted chat model. A filtered sum
computed 120.0 and registered the result DataFrame. The paired Responses API
`function_call_output` still contained only `Query successful.`: the
numeric answer was in the local tool artifact. This verifies a delivery defect;
it does not establish why the live model made each call.

The active patcher changes only the successful `query_dataframe` return in
Cell 32. The tool message includes the result DataFrame ID, a JSON preview of
actual computed values, item counts, and a truncation flag. The preview stops
at 2,048 serialized characters and shows at most ten selected rows. It never
presents a partial row as a complete one. The full registered DataFrame and
native artifact, accepted arguments, calculations, and failure behavior stay
the same. The analyst still needs to return genuine native `AnalysisInsights`;
the graph, guards, validators, and acceptance thresholds are unchanged.

On the base, the three new regression cases failed because the model-visible
result was absent (exit 1, 15 deselected, two dependency warnings). The same
cases passed in the analyst and revision integration selection (exit 0,
22 passed, four warnings). They check the exact paired Responses API tool
result, native completion after reading the total, row-count truncation,
oversize-row truncation, and retention of the complete artifact and registry
result. Black, Flake8, and both notebook/graph validators passed. Two patcher
runs produced identical bytes. The notebook retains 99 cells, 70 State
fields, 15 nodes, and 33 edges.

- Raw notebook SHA-256: `92595cbbc702a00ee53ae3d3ce6b41bdd0aef51d0229f5ddddce715589891310`
- LF-normalized SHA-256: `cf75cd60a987fbfe3512d04361ed7c2ffb5945a4dadf79b0f22db5399ab7a18f`

The stopped attempt's saved-response subtotal was $0.04797502, bringing the
campaign subtotal to $1.92494651. This is partial checkpoint usage; it
excludes unsaved responses, embeddings, and possible in-flight charges.
The repair and its tests require no provider calls. The PR records the exact
candidate SHA and its final validation and independent review receipts.
