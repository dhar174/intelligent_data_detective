<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Project Brief

## Project scope
The **Intelligent Data Detective (IDD)** is a multi-agent autonomous data analysis system. Given a CSV dataset and a natural-language request, it automatically cleans the data, performs statistical analysis, generates visualisations, and produces structured Markdown/HTML/PDF reports without human intervention. Current runnable proof work uses `IntelligentDataDetective_beta_v5_patched.ipynb`, regenerated from the source notebook by `_patch_notebook.py`.

## Primary goals
- Accept arbitrary tabular data and a user question; produce a complete analysis report.
- Coordinate specialised LangGraph agents through a supervisor-worker pattern.
- Persist intermediate findings across agent steps using a scoped memory system.

## Repository boundaries
- **No web UI** — there is no frontend, no API server, no database.
- **Notebook workflow** — edit `_patch_notebook.py`, regenerate `IntelligentDataDetective_beta_v5_patched.ipynb`, and validate the patched notebook.
- **Completion baseline** — W14 run `IDD_run_run_default_id-20260504-1338-b3079aea` passes `validate_run.py` 12/12 and `validate_artifact_quality.py` 9/9.
- **Python support files** — `memory_enhancements.py` and test files are standalone Python; they do not form a package.
- Preserve user-authored docs and existing agent assets outside managed sections.
<!-- repo-agent-bootstrap:managed:end -->
