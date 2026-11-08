# CP5 evaluator image evidence

Base: `52e8d2a35296bb07c297c93bfa4cf5d2916337bb` (merged PR #202).

## Evidence and causal limits

The last fresh retail attempt `7551b7c0-92b5-4969-a888-90986d0fd793` failed with native `KeyError('content')`; PR #202 separately repaired that file-writer prompt contract. The attempt remains failed, with both validators NOT_REACHED and CONTEXT_CLOSED cleanup. Its native evaluator had returned acceptable and claimed LOWESS, but inspection of the actual current-run scatter showed only the linear fit although the specification required both. This independently failed manual artifact inspection; it cannot qualify retail or unlock sample_dirty.

Outer checkpoints do not retain that live evaluator's inner retrieval exchanges. Their absence does not prove that it failed to request previews. No private-history replay or exact live preview-delivery claim is made.

A separate controlled audit uses the actual generated evaluator factory/node, canonical State/ToolStrategy/VizFeedback and four real current-run PNGs. It reproduces a native acceptable completion in one model call with zero image blocks and zero tool results in the effective request. The current runtime context contains metadata, and images are only available if the model elects to retrieve them. The existing `get_visualization(include_preview=True)` delivery works when requested; this repair does not change that tool. The confirmed gap is that actual pixels are optional for visual grading.

## Minimal delivery correction

Immediately before native invocation, the evaluator node now validates each explicit current figure path with the existing current-run PNG ownership/integrity helper and obtains its image through the existing `_encode_preview` helper. It places a text label containing the exact figure ID/path and a PNG image_url block into the existing rendered runtime context, alongside the unchanged textual context. A model-copy preserves the original message object. The existing safe wrapper converts that SystemMessage into the same named HumanMessage runtime_context convention. The existing hosted hook protects that entire newest context under pressure. The assignment explicitly requires inspecting all supplied images against specification features rather than inferring rendered features from titles/descriptions.

This changes only generated Cell 57, through the active patcher. All other 98 cells and the remainder of Cell 57 are identical after removing precisely the new image block and inspection sentence. Both paths and pixels come from explicit current-run references, never filesystem discovery or substitutes. Missing, corrupt, stale, unreadable or unencodable PNGs fail before the model can approve. The existing preview helper's 2 MiB per-image cap is retained; oversized figures require regeneration within that limit. Files are never resized, rewritten, moved or copied by the production correction.

No factory, registered tool, schema, State field, graph, runtime type, shared/Qwen hook, model ID/effort, SDK retry, timeout, recursion budget, revision route change. Native VizFeedback parsing, grades and revision routing remain unchanged. Seven canonical modules/manifest, proof runner and both production validators remain unchanged. Invariants remain 99 cells, 70 fields, 39 reducers, 15 nodes and 33 edges. Strict 12/12 and 9/9 remain required.

The installed token estimator treats image_url blocks with its existing fixed image estimate rather than counting base64 characters. Its approximate budget and the actual provider's image-token usage are different quantities; no budget, estimator or spending limit is changed here. Under pressure the newest multimodal context remains a whole protected group. Image evidence is provided on the first request, so successful completion does not depend on extra retrieval rounds. Tools remain available for legitimate follow-up.

## Offline regression evidence

Before production edits, eight new cases fail against the real generated baseline: normal and pressured requests contain metadata alone, invalid current PNGs can reach acceptable completion, and a reused thread receives no current image bytes. The original eight-case baseline is retained. Final expanded fixtures also load the actual baseline notebook from verified Git bytes and all thirteen cases fail, without changing the implementation checkout; repetitions are not distinct tests.

After correction, thirteen cases pass (two dependency warnings). Both normal and pressured first model requests contain every figure's actual PNG bytes and ID/path association; canonical Responses API serialization emits real input_image blocks. The model terminates natively through VizFeedback. Stable checkpoint-ID caller history and PNG file hashes remain unchanged. A reused inner thread receives changed current PNG bytes and retains the original earlier request bytes. Missing, corrupt, stale, oversized and unreadable controls fail before any model request or approval with recovery set to either 0 or 1.

The first green-target invocation had six passes and two failed state-equality checks because LangGraph legitimately allocated IDs on fixture messages whose IDs were None. The fixture now supplies stable IDs matching checkpointed history; full state-equality assertions remain. This required no production change. Original failed receipts are retained. New messages without IDs still follow LangGraph's unchanged native ID-allocation behavior.

Expanded recovery controls then produced four failures / nine passes: missing, corrupt, stale and unreadable helper errors could enter legacy outer recovery with flag 1. The image block now classifies expected path-validation ValueError/OSError as evaluator-integrity errors, which the existing outer guard rethrows regardless of the flag. It never constructs feedback. The prior patcher 211 and focused 391 passes are retained; the then-running broad selection was interrupted without a result so all final gates could be rerun after this refinement. The stop command reported one unsupported descendant, and a follow-up explicitly verified all three observed process IDs absent. No paid process was involved.

Offline loaders now execute the existing pure generated preview helper where needed; no business tool is replaced. A new real-Jupyter smoke case executes all thirteen image contracts with outbound connections denied and no credentials. Existing native strict failures, real SQLite revision routing, provenance, reporting and evaluator suites remain mandatory. Test fixtures and scripted feedback are mechanism evidence, never production qualification or fabricated success.

## Verification and remaining boundaries

Target thirteen cases passed; patcher/parity/state 211 passed. Full focused/broad/all16 mandatory, scientific/notebook/proof-kernel, validators, format/lint and deterministic regeneration gates are running, not yet certified. Exact-HEAD hosted Linux/Windows and fresh installed Antigravity approval remain required before merge. Final commands/exits/counts/warnings/skips, hash receipts and CI links will be recorded in the PR conversation without counting overlapping/repeated or nested tests twice.

Notebook raw SHA-256 `3cc672bc1cf57bde059cd819200c5f09f6a92a6f4e8c09374bf73c1f0264acae`; LF-normalized `5ddcf90d4ef03f4cf0a7af328c70a9754ec3f09bca6dec481541d500e727a934`.

Providing pixels cannot guarantee a model's interpretation of every feature. Genuine evaluator feedback, strict native checks and independent actual PNG/data/report inspection are all still required. No historical grade/flag/evidence is changed, no failed checkpoint resumed, and neither scenario is qualified. The unrelated ordinary writer schema inconsistency and earlier observations remain separate. No paid process runs during repair; carried saved-response subtotal USD 1.32724640 is partial, not a reconciled bill. Existing campaign limits remain unchanged, #129 deferred and #165 open. Only fresh prepared attempts at the same final merged source/environment may qualify the two scenarios.
