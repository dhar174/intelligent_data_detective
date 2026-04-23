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

## Constraints
- No `src/` directory — Python files are at repo root.
- There is no CI pipeline (no `.github/workflows/` for automated test runs).
- Preserve user-authored docs and existing agent assets outside managed sections.
<!-- repo-agent-bootstrap:managed:end -->

<!-- session-curated:start -->
## Session-added tooling

### `validate_graph.py` (repo root)
Fast (<20s) **compile-only** graph validator. No API calls — safe to run as a pre-commit gate before launching a full notebook run.

Detects:
- Managed-channel collisions (e.g., `remaining_steps` leaking into InputSchema, source of BR-7).
- Missing reducers on collection fields.
- Unreachable nodes / dead-end nodes (e.g., EMERGENCY_MSG with no outgoing edge).
- Schema-tool name collisions across agents.

```bash
python validate_graph.py
```

### Resuming a notebook run
```bash
python run_notebook_live.py --resume
```
Resumes from `checkpoints.sqlite`, skipping completed nodes (saves 12–15 min on a typical mid-pipeline restart). **Only safe if `State` schema is unchanged since the last checkpoint.** Adding a field, changing a reducer, or renaming an annotation invalidates the checkpoint and will cause silent corruption — delete `checkpoints.sqlite` and start fresh in that case.

### Test baseline (reaffirmed this session)
- `test_intelligent_data_detective.py`: **22/22 pass**
- `test_error_handling_framework.py`: **15/16 pass** (same known edge case)
<!-- session-curated:end -->

