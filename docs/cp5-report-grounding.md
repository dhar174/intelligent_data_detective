# Report figure provenance

`idd_tools.report_figure_inventory` collects canonical `DataVisualization` records
from the `visualization_results` envelope and the `viz_results` worker list.
It accepts model and serialized envelopes, rejects ambiguous IDs or paths, and
does not discover files. Runtime roots are explicit inputs from the notebook.

`idd_tools.ground_report_figures` returns a deep copy of an outline or section.
Each reference must identify exactly one inventory entry. A missing ID or path
can be filled from that entry; invented or conflicting references fail. The
canonical facade re-exports both functions with object identity preserved.

The notebook supplies runtime context and calls these canonical helpers. It
does not contain a second implementation of the validation rules. Canonical
`idd_graph.dispatch_sections` forwards both visualization channels and the
artifact root so envelope-only state survives section fan-out and checkpointing.

The seven-module provenance manifest covers these rules. Moving validation into
the canonical modules requires updating their digests, rather than retaining
stale hashes. State fields and graph topology do not change. W14J and the proof
validators retain their file, ownership, completeness and report-quality checks.

Offline regressions exercise the real outline and section factories, native
structured responses, canonical dispatch, SQLite restoration, and rejection of
invalid references. Those tests establish deterministic behavior, not live model
compliance or a production CP5 proof.
