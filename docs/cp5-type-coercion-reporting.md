# CP5 type conversion loss reporting

Base: `d1eac2ac24ce96b3eeb1e2e6b6e17e40ead359fc` (merged PR #204).

In retail attempt `8f6962c6-6ae4-4b60-b9d6-bbe28e2cbe99`, the cleaner
requested `order_id` as `int` under the existing coercion policy. The
generated tool reported `status=ok` and zero issues. The subsequent
`check_missing_values` tool reported 180 missing order IDs; the exported
180-row CSV independently contained 180 missing IDs. The external
supervisor stopped the attempt before the production validators. Original
evidence remains unchanged. This was a manual stop, not an automatic native
failure or a passing run.

The conversion policy allows invalid inputs to become missing. The defect
was the report: a successful Pandas conversion skipped the exception branch
and reported no issues even when non-missing values became missing. Dry-run
details were also confined to the tool artifact rather than its model-visible
content. The generated `convert_data_types` now labels that loss `coerced`,
reports its exact count per column, and includes affected columns in the
issue total. Both real and preview results expose the same details through
native tool content. Pre-existing missing values are excluded from the loss
count. Dry runs still leave the registered frame untouched.

Only Cell 32's conversion function changes. The registered tool, argument
defaults, parsing policies, registry write semantics, error propagation,
completion schemas, shared hook, graph, runtime settings, and production
validators are unchanged. In particular, this does not prohibit an explicit
lossy coercion or guarantee that a model acts on the reported evidence.

The original generated notebook produced **7 failing and 15 passing** dtype
tests, with two dependency warnings and no setup errors. The focused suite
then passed **22 tests**, with two dependency warnings. The native cleaner
probe receives the 180-ID loss warning from a real dry-run tool result, keeps
the identifier values as strings, exports them, and terminates through a
genuine `CleaningMetadata` response. It runs with a scripted model and no
network; its result is not a production qualification.

The related inaccurate 176-row prose seen in other attempts is separate.
Replaying the earlier recorded calls through the generated factory delivered
184 before deduplication and 180 in every effective request after it. The
scripted replay reproduced the actual correct CSV and did not demonstrate a
stale count reaching the model. This repair does not rewrite metadata or
certify arbitrary claims. The actual artifact review and **12/12 + 9/9**
production gates remain required before qualifying either scenario.

Exact-HEAD verification and independent review receipts are recorded with
the PR. Keep #129 deferred and #165 open. No live execution during the repair.
