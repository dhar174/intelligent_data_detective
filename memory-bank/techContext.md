<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Tech Context

## Stack
- Language: Python 3.10+ (validated on 3.12.3)
- LLM orchestration: LangGraph 0.1+, LangChain, `langchain_experimental`
- LLM provider: OpenAI API (`OPENAI_API_KEY` required); o-series models handled via `MyChatOpenai`
- Optional: Tavily web search (`TAVILY_API_KEY`)
- Data: pandas, numpy, scipy, scikit-learn
- Visualisation: matplotlib, seaborn
- Memory: chromadb (vector store), joblib (caching), `memory_config.yaml` (namespace config)
- Reports: openpyxl, xhtml2pdf
- Testing: pytest, black, flake8
- Notebook runtime: Jupyter / JupyterLab

## Local setup
```bash
pip install langchain langchain-core langchain-openai langchain_experimental langgraph \
    pandas numpy scipy scikit-learn matplotlib seaborn pydantic python-dotenv \
    tiktoken openpyxl xhtml2pdf tavily-python chromadb joblib \
    pytest black flake8 mypy jupyter

export OPENAI_API_KEY="your-openai-api-key"   # required
export TAVILY_API_KEY="your-tavily-api-key"   # optional
```

## Commands
| Purpose | Command |
|---------|---------|
| Run core tests | `python3 -m pytest test_intelligent_data_detective.py -v` |
| Run error handling tests | `python3 -m pytest test_error_handling_framework.py -v` |
| Run memory tests | `python3 -m pytest test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py -v` |
| Run all tests | `python3 -m pytest -v` |
| Format | `black test_intelligent_data_detective.py test_error_handling_framework.py` |
| Lint | `flake8 test_intelligent_data_detective.py --max-line-length=88 --extend-ignore=E203,E501` |
| Validate notebook | `python3 -c "import json; c=json.load(open('IntelligentDataDetective_beta_v5.ipynb'))['cells']; print(len(c),'cells')"` |

## Expected test results
- `test_intelligent_data_detective.py`: **22/22 pass**
- `test_error_handling_framework.py`: **15/16 pass** — 1 known edge-case failure is acceptable
- Memory test suites: all pass

## Full workflow execution
- Requires `OPENAI_API_KEY`
- Launch: `jupyter notebook IntelligentDataDetective_beta_v5.ipynb`
- Duration: 6–8 min (small), 12–15 min (medium), 20–25 min (large)
- **Never cancel a running workflow** — interrupting mid-run leaves state inconsistent

## CI pipeline
File: `.github/workflows/copilot-setup-steps.yml`
Triggers: `pull_request` to `main`, `push` to `main`, `workflow_dispatch`

Steps (in order):
1. **Check required project files** — asserts key test files plus both notebooks (`IntelligentDataDetective_beta_v5.ipynb` and `IntelligentDataDetective_beta_v5_patched.ipynb`) are present.
2. **Validate notebook files** — smoke-checks both notebooks by parsing JSON and confirming at least one cell exists.
3. **Install dependencies** — uses `requirements.txt` / `requirements-dev.txt` when present; falls back to an inline `pip install` list.
4. **Run no-key unit/integration checks** — `python -m pytest test_validate_run.py tests/unit tests/integration -q`.
5. **Run root regression tests** — explicitly invokes `test_intelligent_data_detective.py`, `test_memory_categorization.py`, `test_memory_integration.py`, `test_memory_lifecycle.py` (these live at repo root, outside `tests/`).
6. **Run error-handling regression tests (required)** — runs `test_error_handling_framework.py` with the one known signature edge-case deselected.
7. **Run known edge-case signature test (non-blocking)** — runs only `TestErrorHandlingFramework::test_integration_with_different_function_signatures` with `continue-on-error: true`.
8. **Check formatting** — `black --check` on the two main test files.
9. **Run flake8 on root regression test file** — `flake8 test_intelligent_data_detective.py --max-line-length=88 --extend-ignore=E203,E501`.
10. **Lint validation scripts** — `python -m flake8 validate_run.py validate_artifact_quality.py test_validate_run.py --max-line-length=120 --extend-ignore=E203,W503`.
11. **Summarize CI caveats** — always-run step noting no-key scope, explicit root-regression invocation, isolated known edge-case handling, and API-key trajectory test exclusion.

## Constraints
- No `src/` directory — Python files are at repo root.
- Preserve user-authored docs and existing agent assets outside managed sections.
<!-- repo-agent-bootstrap:managed:end -->
