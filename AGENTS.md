<!-- repo-agent-bootstrap:file-kind=agents-md -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# AGENTS.md

## Project overview
The **Intelligent Data Detective (IDD)** is a multi-agent autonomous data analysis system built with LangChain and LangGraph. All logic lives inside one Jupyter notebook: `IntelligentDataDetective_beta_v5.ipynb` (27 cells, ~11 k lines). There is no separate Python package.

Agents execute a supervisor-worker pipeline:
```
Supervisor → Initial Analysis → Data Cleaner → Analyst → Visualization
          → Report Orchestrator → Section Workers → Report Packager → File Writer
```

Primary goals:
- Accept a CSV/dataset and a natural-language analysis request.
- Autonomously clean, analyse, visualise, and report on the data using LLM-driven tool calls.
- Produce structured artifacts (charts, HTML/PDF report) in the artifacts directory.

## Repo map
Key files:
- `IntelligentDataDetective_beta_v5.ipynb` — **source of truth** for the entire system
- `memory_enhancements.py` — standalone memory lifecycle/categorization module
- `memory_config.yaml` — per-agent memory namespace TTL and limits
- `idd_v4_state_graph.mmd` — authoritative LangGraph state graph topology
- `test_intelligent_data_detective.py` — 22 core tests (no API keys needed)
- `test_error_handling_framework.py` — 15/16 tests pass (1 known edge-case failure acceptable)
- `test_memory_categorization.py`, `test_memory_integration.py`, `test_memory_lifecycle.py` — memory tests

Important directories:
- `docs/` — architecture docs, ADRs, diagrams
- `.github/` — Copilot assets, instructions, agents, skills, hooks
- `memory-bank/` — resumable agent context
- `legacy_versions/` — old notebook versions; inspect before large changes
- `v5_analysis_reports/` — generated analysis reports from prior runs

## How to work in this repo
Before making changes:
1. Read `memory-bank/activeContext.md` and `memory-bank/progress.md`.
2. Check the cell map in `.github/copilot-instructions.md` when the task involves the notebook.
3. Inspect `.github/agents/` and `.github/instructions/` before adding new guidance files.

## Build, test, lint
```bash
# Install dependencies
pip install langchain langchain-core langchain-openai langchain_experimental langgraph \
    pandas numpy scipy scikit-learn matplotlib seaborn pydantic python-dotenv \
    tiktoken openpyxl tavily-python chromadb joblib pytest black flake8

# Run all tests (no API keys required)
python3 -m pytest test_intelligent_data_detective.py -v               # 22 tests
python3 -m pytest test_error_handling_framework.py -v                 # 15/16 (1 known failure OK)
python3 -m pytest test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py -v

# Format and lint
black test_intelligent_data_detective.py test_error_handling_framework.py
flake8 test_intelligent_data_detective.py --max-line-length=88 --extend-ignore=E203,E501

# Full workflow (requires API keys, 6–25 minutes — never cancel)
export OPENAI_API_KEY="your-key"
jupyter notebook IntelligentDataDetective_beta_v5.ipynb
```

## Engineering conventions
- **BaseNoExtrasModel**: all agent output models extend this; include `reply_msg_to_supervisor`, `finished_this_task`, `expect_reply`.
- **State reducers**: never assign directly to reduced State fields; use reducer semantics (see Cell 7).
- **`@handle_tool_errors`**: required on every tool function; `validate_dataframe_exists(df_id)` must be the first call for DataFrame-touching tools.
- **`MyChatOpenai`**: use everywhere in the notebook instead of `ChatOpenAI`.
- **DataFrameRegistry**: single source of truth for DataFrames; always reference by `df_id` string.
- **Memory namespaces**: `('memories', '<kind>')`; limits from `memory_config.yaml`, not hardcoded.
- **File writes**: always go through `_resolve_artifact_path()`.
- **No `src/` directory**: all Python files are at repo root.

## Constraints / do-not rules
- Do not delete or reorder notebook cells.
- Do not bypass `DataFrameRegistry` by passing DataFrames directly.
- Do not hardcode memory TTL or limits — use `memory_config.yaml`.
- Do not bypass `_resolve_artifact_path()` when writing files.
- Preserve user-authored docs and existing agent assets outside managed sections.

## Definition of done
- Relevant tests pass (see counts above).
- Notebook cell count unchanged after edits (unless new cells were appended).
- `memory-bank/` or `docs/` files refreshed when architecture or contributor expectations change.
- The final diff stays within the requested scope.
<!-- repo-agent-bootstrap:managed:end -->
