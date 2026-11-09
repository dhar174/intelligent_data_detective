# CP5 visualization worker checkpoint isolation

Base: `68e89c8f7e88dbd4edca49a47266cd4bbc91977c` (merged PR #210).

Retail attempt `1fb0eaff-a15b-4cfb-88a7-14a23663b546` completed native
cleaning and analysis and saved four PNGs. The live PNG read returned the
new inspection guidance. One worker published a completed figure; two
stopped at the existing unchanged-retrieval guard and one raised a native
`GraphRecursionError`. The runner exited 1 at Cell 75 after 572.4 seconds
and closed the kernel. No evaluator verdict or production validation was
reached. The PNGs were preserved as failed-run evidence.

The graph compiled in generated Cell 60 uses
`idd_graph.assign_viz_workers`. On the base, the canonical dispatcher
gave sibling assignments the same inner-agent checkpoint thread. A
network-denied replay used that dispatcher, the real generated worker
and native factory, and a scripted model. Starting the second assignment
after its sibling completed caused the second model request to contain
both runtime assignments and the first worker's completion receipt.
This proves cross-assignment history exposure; it does not establish
the motive for any individual live lookup.

The canonical dispatcher now copies the worker configuration and derives
its inner thread from the outer thread, revision count and assignment
index. It drops a parent checkpoint cursor from that inner copy, while
retaining the runtime and other configuration values. The parent config
and SQLite thread stay intact. Figure IDs, native completion, retrieval
guards, evaluator routing, graph topology, and production thresholds are
unchanged. No generated cell or patcher change is needed: the notebook
already imports the canonical dispatcher.

The new generated-function regression failed on the base because the
earlier assignment reached the second model request (exit 1, 229
deselected, two dependency warnings). It passed after the repair (exit
0, one passed, two warnings). Two sibling assignments in each of two
revision rounds see only their own context and reopen their own native
completion under distinct inner thread IDs. The canonical dispatch
control verifies that the parent runtime, tags, thread ID and checkpoint
cursor remain intact. The parity, patcher, state, and real SQLite revision
selection passed 215 tests with six warnings. Both structural validators,
test formatting, and lint passed.

The manifest still has seven canonical modules with matching
`files` and `modules` maps. Only the `idd_graph.py` digest was updated:
`4bd9c9892ca2722eecd66d76ae6da6d0984d649dcd4a39f36da030fd728312e8`.
The notebook is byte-identical to base and two regenerations match:

- Raw SHA-256: `5959e7f6d6ebde8c8a26b0aa05b17fefa5af80c24cb60423a7d6199a26bf9b3a`
- LF-normalized SHA-256: `8a99e2beeff33ce5c3d4c403389cdaf497b1b20c4abd981d9758904a4804c364`

The failed attempt recorded $0.03677756 in saved-response usage, for a
campaign subtotal of $2.05369291. There were 11 unique saved responses
and 99 successful Responses API log occurrences; the latter are not a
token ledger. This subtotal excludes embeddings and unsaved usage.
`sample_dirty` did not start. Exact-head checks and independent review
are recorded in the PR before any merge.
