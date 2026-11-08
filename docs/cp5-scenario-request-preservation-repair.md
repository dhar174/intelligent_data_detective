# CP5 scenario request preservation

Base: `be51d5fadebf557d04c8ef281b99b003c31f6500` (merged PR #199).

## Verified defect

Cell 48 selects a scenario-specific request. Retail requests stakeholder-ready analysis of revenue drivers, regional/channel differences, discounts, satisfaction and support-ticket risks, with at least three meaningful PNG charts. Sample_dirty explicitly requests a value histogram, category-count bar chart and score/value scatter. Cell 51 then unconditionally replaces that request with a generic analysis request. Cell 72 copies the replacement into both `initial_state.user_prompt` and the initial HumanMessage.

This is an observed deterministic overwrite, not inferred from an unused template or a private model trace. Read-only inspection of the failed retail attempt `98761fe6-83e3-415d-a3ef-c3c196c21f39` at source `31aaf6769a480dd0c6026ad45293a749d4318b38` finds the generic request in canonical SQLite state (`user_prompt` SHA-256 `e0694e92d62b7fa5ae088fc6d39f6950065bbb5fb6d85630d71536e79993b363`). Those initialization cells are unchanged at this repair's base. Offline execution of the actual generated request statements, including the real scenario branch condition, reproduces the overwrite for both scenarios. The actual generated initial-agent factory receives the generic HumanMessage under a controlled model boundary and can complete natively despite losing the more specific assignment.

This finding was documented separately from PR #199 and is not silently added to its reviewed scope. No new paid attempt has started. Historical evidence remains untouched and failed; no production qualification or paid sample_dirty run is claimed.

## Minimal correction

Through the active `_patch_notebook.py` transformation, Cell 51 keeps an existing nonblank string request unchanged, including its exact whitespace. The existing generic text remains the fallback for an absent, None, empty, whitespace-only or nonstring value. No environment variable or public API is added. HumanMessage/tuple construction, Cell 72 state initialization, RuntimeCtx and its frozen contract remain unchanged.

Only Cell 51 differs in the generated notebook. All other 98 cells, scenario texts/data generation/fixtures, factories, completion tools, wrappers, shared/Qwen hooks, graph routing, revision limits, models/settings/retries/timeouts and recursion limits are unchanged. Canonical seven modules/manifest, production proof runner and both validators are unchanged. Invariants remain 99 cells, 70 State fields, 39 reducers, 15 nodes and 33 edges. Strict completion, native failures, artifact ownership and 12/12 plus 9/9 criteria remain intact. The change supplies the intended request; it does not fabricate findings, figures or acceptance.

## Offline red/green and limits

Before production edits: **3 failed, 5 passed**, exit 1. Both scenario requests were overwritten before the real native model request, and custom user text was also lost. All five generic-fallback controls already passed.

After correction: **8 passed**, with two dependency warnings. Tests execute selected actual generated statements from Cells 48, 51 and 72, preserving branch conditions and the real state/message constructors, while bypassing client initialization, credential loaders and production execution. For both scenarios, the real generated factory, middleware and ToolStrategy receive the exact intended HumanMessage, return a genuine InitialDescription through the native response tool, and stop after one controlled model request. Custom request bytes and all fallback cases are checked separately. Only model/summary/memory boundaries use the established offline harness; outbound sockets are denied. These tests verify request delivery and native completion contracts, not live analytical accuracy.

The existing real-Jupyter proof case runs all native-completion tests in the same attested kernel, so these cases are also covered there without adding another smoke case, changing its complementary selectors or raising its 120-second per-cell bound. The full inherited completion/revision/reporting/strict-failure/SQLite suites remain required.

## Initial publication receipts

Patcher/parity/state gate: 211 passed, four warnings. Full focused, supported broad no-key, all 16 mandatory error-handling, scientific/notebook/proof-runner Jupyter, validators and deterministic regeneration are running and not yet claimed green. Fresh exact-HEAD Linux/Windows CI and actual Antigravity review are required before merge or live execution; final command/count/skip/warning/hash receipts will be posted in the PR conversation without double-counting overlapping selections.

Raw notebook SHA-256 `073d8cf9af4977e7af20a7695bbf56a0bf9bf6ba7507fc4fc8c1047796c7c3ac`; LF-normalized `b165c41178c933f7205903bdc83ed91407a96e5b6f4733d16f8cb27829073546`.

Cleaner row-count prose, numeric-conversion warnings, the REPL RuntimeCtx observation and prior report-reference failure remain separate. #129 stays deferred, #165 open, and the campaign stopped during repair. Carried saved-response usage is a partial USD 1.24853335, not a reconciled bill; existing spending limits are unchanged. Any fresh retail and sample_dirty qualification must use the same final merged source/environment, with actual production gates and artifact inspection before sample_dirty proceeds.
