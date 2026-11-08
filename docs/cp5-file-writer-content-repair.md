# CP5 file-writer prompt content contract

Base: `5ee01fa4c88db933ef51918c867e3f045278feda` (merged PR #201).

## Verified failure and narrow correction

Fresh retail attempt `7551b7c0-92b5-4969-a888-90986d0fd793`, thread `thread-e7c426dc-bebd-44cc-92ff-5c571273ceec`, exited 1 after 512.6 seconds with Cell 75 STREAM_FAILURE, native `KeyError('content')`, and CONTEXT_CLOSED cleanup. The public exception traceback reaches `file_writer_node` and `ChatPromptTemplate.format_messages`, then Python's formatter. Both validators were NOT_REACHED. The runner had already exited when an operator stop was attempted; the precondition rejected that request before creating a stop file. No manual stop, checkpoint resume, grade change or successful qualification occurred.

The cleaner now completed genuinely with accurate 180-row metadata. Its actual exported CSV has 180 rows, zero duplicates/missing values and SHA-256 `63ecc0422c67c0f512cf318950c23f82f055b3b32d6aa70a7606c0263c4971bc`. The analyst, three visualization workers, evaluator, section writers and report packager returned native structured results before the file-writer formatting failure. The actual scenario request hash remains `7ac9af4cc77491b42e25e2f61670f93a12c65e678a77cf38a36ee41c9a86b711`. These are stage receipts, not final CP5 success.

The generated Cell 25 file-writer template expects `{content}`. The existing factory and node both supply `file_content`. The node renders that template before choosing its final manifest branch, so final reporting also crashes even though that branch uses a separate native manifest prompt. Existing manifest tests substituted a constant system template and missed this boundary.

The active patcher now changes exactly that one placeholder to `{file_content}`. It scopes the replacement to the actual `file_writer_prompt_template` assignment, requires exactly one old placeholder and parses the generated cell. All surrounding prompt text and all node/factory logic are unchanged. No alias, new default, alternate completion, report generation, manifest reconciliation, file write, schema, State field, graph, tool, shared hook, model setting, retry, timeout or recovery change is introduced. All other 98 cells and seven canonical modules/manifest, proof runner and production validators remain identical. Invariants: 99 cells, 70 fields, 39 reducers, 15 nodes, 33 edges; strict 12/12 and 9/9 remain required.

## Offline regressions

Before production edits, eight new cases all fail with the actual `KeyError('content')` (exit 1). After correction, all eight pass (exit 0; two dependency warnings).

Three cases load the actual generated prompt and node, preserving empty, multiline and literal-brace content exactly up to the ordinary invocation boundary. That boundary deliberately raises; these cases do not claim that ordinary file writing completes. Five cases execute the actual generated final manifest branch with native `create_agent`, canonical State/ToolStrategy/ListOfFiles, real unknown-tool middleware, the frozen generated RuntimeCtx, isolated current-run PNGs and real Markdown/HTML/PDF fixture files. Only the chat/memory boundary is controlled. Genuine native completion succeeds; unfinished and reply-required results remain incomplete; invalid tools exhaust the existing 40-step inner budget; an unexpected native exception propagates. Existing files are hashed before/after and remain unchanged.

The existing patcher manifest helper now loads the actual Cell 25 template instead of substituting a constant template. This strengthens the existing manifest, missing-image, ownership and reconciliation controls. The inherited real-Jupyter proof smoke includes every native-completion case in its existing sequential phases, with the same kernel, source attestation and 120-second per-cell bounds. Synthetic fixture success never becomes production evidence.

The first expanded patcher run had 16 failures / 195 passes because the changed test helper omitted ChatPromptTemplate and MessagesPlaceholder from its execution namespace. That test-only setup error was corrected by binding the actual imported classes; original failed receipts are retained. It did not require another production change. Auxiliary Windows inspection commands also needed explicit UTF-8; failures were observed rather than treated as successful checks.

## Separate observations and limits

The same failed live run's scatter specification explicitly requires a linear fit and LOESS. The actual saved scatter visibly shows only the linear fit, while the genuine acceptable evaluator response claims LOWESS. This is a separate failed manual artifact inspection, retained outside this prompt repair. No sample_dirty gate can be opened from it, regardless of reporting progress. Outer checkpoints do not retain evaluator retrieval exchanges; their absence does not prove whether the inner evaluator requested image previews. No exact private-history replay or independently confirmed preview-delivery defect is claimed. Do not add a speculative evaluator or plotting correction to this PR.

There is also a source-level inconsistency on the unrelated ordinary writer path: its factory registers FileResult while the node expects ListOfFiles. This repair does not change either contract or claim to qualify that path. The failing production route uses the separate final ListOfFiles manifest agent.

Saved-response usage for the stopped attempt is USD 0.0779789, carried campaign subtotal USD 1.32724640. Nineteen saved responses versus 31 successful response log occurrences and 86 embedding occurrences make this partial, not a reconciled bill. No paid process runs during repair. Existing campaign limits remain unchanged. #129 stays deferred; #165 stays open. Historical evidence is untouched.

## Publication and verification

Target eight cases passed. Full focused, patcher/parity/state, supported broad no-key/all16 mandatory, scientific/notebook/proof-kernel, format/lint, validators and deterministic regeneration gates are running; their results are not yet claimed. Exact-HEAD hosted Linux/Windows and fresh installed Antigravity approval remain required before an authorized merge. Final commands/exits/counts/warnings/skips, hashes and CI links will be recorded in the PR conversation without double-counting repeated or overlapping cases.

Notebook raw SHA-256 `50692afbcd5d2e29875fc38b6c9152724907671342fff52a0e29d4d6cf2ad375`; LF-normalized SHA-256 `7db2204a1333f88a13196cf9e436cd6f3f1b0430b51ef6bf1221fe76f7494a2a`. Only generated Cell 25 differs from base, by one placeholder. Fresh qualification requires new preparation and fresh retail/sample attempts at the same final merged source/environment, actual validator passes and manual data/PNG/report inspection. Neither scenario is qualified yet.
