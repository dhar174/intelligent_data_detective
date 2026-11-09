# CP5 CSV snapshot dimensions in tool results

Base: `31324030eb9d911f80951d1187c5e0ff5823261a` (merged PR #205).

Retail attempt `65665b5d-b1ae-4e9f-9a53-d166f44bb40d` exported a correct
180-row, 13-column CSV but completed with a 176-row description. The external
supervisor stopped it before reporting or the production validators. This was
a manual quality stop, and the original attempt remains preserved.

The native export result (`call_PsMy3HKEShJRQYEtSXi4INYQ`) recorded 180 rows
and 13 columns in its artifact. Its model-visible content was only
`Exported to retail_orders_cleaned.csv (csv).` The earlier offline replay had
already shown the right current count in the cleaner runtime context. The
missing dimensions in the final export result are a verified delivery gap;
the evidence does not establish why the live model chose the wrong count.

Only `export_dataframe` in generated Cell 32 changes. After a successful CSV
write, the result includes the existing artifact row and data-column counts.
Those counts describe the actual snapshot, including a selected column subset
and excluding the optional index. The original success prefix, native paired
ToolMessage and artifact fields are preserved. Non-CSV results, paths,
failure returns, parsing and writing policies are unchanged.

The baseline produced two intended assertion failures, with no setup errors:
the successful CSV ToolMessage lacked dimensions and the generated native
cleaner could not read them from the visible result. The regression performs
real deduplication from 184 to 180, exports one of two registered columns,
reads 180 rows and one data column from the model-visible response, and
finishes through the existing native `CleaningMetadata` schema. The actual
CSV and current-snapshot receipt are checked independently. The prior
tool-pair/checkpoint regression still verifies native delivery and round-trip.

No new schema, State field, prompt, shared hook, model setting or recovery
path is introduced. The snapshot freshness guard and production validators
remain unchanged. This does not rewrite model claims, promise a correct live
response, or address the separately documented export `na_rep` and filename
versioning defects. Both live scenarios still require real artifact review
and the unchanged **12/12 + 9/9** production gates.

Exact-HEAD local, hosted and Antigravity receipts are recorded with the PR.
No live execution during the repair. Keep #129 deferred and #165 open.
