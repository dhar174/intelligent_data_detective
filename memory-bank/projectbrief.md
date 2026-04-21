<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Project Brief

## Project scope
The **Intelligent Data Detective (IDD)** is a multi-agent autonomous data analysis system. Given a CSV dataset and a natural-language request, it automatically cleans the data, performs statistical analysis, generates visualisations, and produces a structured HTML/PDF report — without human intervention. All logic lives in a single 27-cell Jupyter notebook.

## Primary goals
- Accept arbitrary tabular data and a user question; produce a complete analysis report.
- Coordinate specialised LangGraph agents through a supervisor-worker pattern.
- Persist intermediate findings across agent steps using a scoped memory system.

## Repository boundaries
- **No web UI** — there is no frontend, no API server, no database.
- **Notebook = source of truth** — `IntelligentDataDetective_beta_v5.ipynb` is the full codebase.
- **Python support files** — `memory_enhancements.py` and test files are standalone Python; they do not form a package.
- Preserve user-authored docs and existing agent assets outside managed sections.
<!-- repo-agent-bootstrap:managed:end -->
