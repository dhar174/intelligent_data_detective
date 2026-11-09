# CP5 worker PNG inspection

Base: `92bb72f73a42346ff71bba71cb9892391afbf315` (merged PR #209).

Retail attempt `de5877e9-8cd3-4140-ae27-7a7c6696d308` completed native
cleaning and analysis and saved four PNGs. Two workers published completed
figure records. The other two stopped at the existing unchanged retrieval
guard. The runner exited 1 at Cell 75 after 718.8 seconds and closed the
kernel. It did not reach the evaluator or production validators.

The stopped stream contained six `list_visualizations`, five
`get_visualization`, and seven `read_file` result headers. All five
detail lookups returned “Visualization not found.” Four file-read results
had PNG magic or chunk markers in text. The console does not establish the
arguments for every call or why the model chose it. The actual PNGs remain
failed-run evidence and were not substituted for native results.

A keyless reproduction used the generated worker factory, actual plotting
and read tools, the frozen runtime configuration, and a scripted chat model.
`read_file` decoded a valid current-run PNG as UTF-8 and passed unreadable
text through the Responses API. An assigned visualization ID was not in
the inventory before the worker published metadata. The exact same saved
path with `get_visualization(..., include_preview=True)` returned actual
PNG pixels as a Responses API image input.

The patch keeps canonical path resolution and ordinary text pagination.
For a resolved PNG file, `read_file` returns the actual path and directs
the model to the existing image preview tool. Explicit byte downloads
still receive the original bytes in the local artifact. PNG bytes are not
decoded into model-visible text. The worker assignment distinguishes
inspection by exact saved path before publication from use of the assigned
ID in native completion. It does not require another inspection when the
worker already has enough evidence.

Only `read_file` in Cell 32 and the visualization assignment helper in
Cell 57 change. Tool signatures, preview behavior, path ownership,
checkpoint state, worker completion and retrieval guards, evaluator
routing, and both production validators remain unchanged.

The first baseline harness run reproduced the two PNG failures and
exposed a missing injected runtime in a direct-tool control. After fixing
the test setup, the unchanged production base had two expected PNG
failures, one passing text and ownership control, and two dependency
warnings. The new regressions exercise a real saved PNG through the native
factory, paired model-visible responses, requested byte retention, actual
preview pixels, and the canonical join. The focused provenance, join, and
revision selection passed 97 tests with six warnings. Formatting, lint, and
both structural validators passed.

Two patcher runs produced byte-identical notebook output:

- Raw SHA-256: `5959e7f6d6ebde8c8a26b0aa05b17fefa5af80c24cb60423a7d6199a26bf9b3a`
- LF-normalized SHA-256: `8a99e2beeff33ce5c3d4c403389cdaf497b1b20c4abd981d9758904a4804c364`

The failed attempt recorded $0.04960213 in saved-response usage, for a
campaign subtotal of $2.01691535. The receipt records 15 unique saved
responses and 72 successful Responses API log occurrences; those counts
are not interchangeable. Embeddings and unsaved responses are excluded
from the subtotal. `sample_dirty` did not start. The PR records exact-head
validation and independent review before any merge.
