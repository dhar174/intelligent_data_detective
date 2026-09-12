<!-- repo-agent-bootstrap:file-kind=agents-md -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# AGENTS.md

## Project overview
The **Intelligent Data Detective (IDD)** is a multi-agent autonomous data analysis system built with LangChain and LangGraph. The notebook system is maintained through `_patch_notebook.py`, which regenerates the committed runnable notebook `IntelligentDataDetective_beta_v5_patched.ipynb` (99 cells in the W14 completion baseline). The original `IntelligentDataDetective_beta_v5.ipynb` remains the source notebook input, but current runnable proof work targets the patched notebook.

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
- `_patch_notebook.py` — patch source for notebook changes; regenerates the patched notebook
- `IntelligentDataDetective_beta_v5_patched.ipynb` — committed runnable W14 completion notebook
- `IntelligentDataDetective_beta_v5.ipynb` — source notebook input for the patcher
- `validate_run.py`, `validate_artifact_quality.py` — final production/artifact proof validators
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
2. For notebook behavior changes, edit `_patch_notebook.py`, regenerate `IntelligentDataDetective_beta_v5_patched.ipynb`, and run the relevant validators.
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
export IDD_NOTEBOOK="IntelligentDataDetective_beta_v5_patched.ipynb"
export IDD_SAMPLE_DATASET="retail_orders"
python run_notebook_live.py

# Final proof validators
python validate_run.py --latest --log-path notebook_run_log.txt --window 180
python validate_artifact_quality.py --latest
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
- Do not edit the generated patched notebook directly; make notebook behavior changes through `_patch_notebook.py`.
- Do not delete or reorder notebook cells.
- Do not bypass `DataFrameRegistry` by passing DataFrames directly.
- Do not hardcode memory TTL or limits — use `memory_config.yaml`.
- Do not bypass `_resolve_artifact_path()` when writing files.
- Preserve user-authored docs and existing agent assets outside managed sections.

## Definition of done
- Relevant tests pass (see counts above).
- Patched notebook regenerates successfully and keeps the expected W14 99-cell structure unless the task explicitly changes it.
- Completion-impacting notebook changes preserve the W14 baseline gates: `validate_run.py` 12/12, `validate_artifact_quality.py` 9/9, no recovery/final-hop/path-normalization markers, and complete final artifacts.
- `memory-bank/` or `docs/` files refreshed when architecture or contributor expectations change.
- The final diff stays within the requested scope.
<!-- repo-agent-bootstrap:managed:end -->
