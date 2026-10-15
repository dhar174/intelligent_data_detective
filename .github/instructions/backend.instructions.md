---
applyTo: "*.py,test_*.py"
---

<!-- repo-agent-bootstrap:file-kind=path-instructions -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Backend instructions

- Python core modules and scripts live at the **repo root** (`idd_models.py`, `idd_registry.py`, `idd_state.py`, `idd_tools.py`, `idd_graph.py`, `idd_core.py`, `idd_bootstrap.py`, `memory_enhancements.py`). Tests live at repo root and under `tests/unit/` and `tests/integration/`.
- The canonical Python core modules are the **single source of truth** for models, registry, state schemas, reducers, tools, and graph construction. The notebook imports from them directly.
- Preserve existing interfaces and function signatures unless the task explicitly changes them.
- Add or update test coverage for any behaviour change.
- Run the relevant test suites before declaring a change done:
  - `python3 -m pytest tests/unit/test_core_parity_guard.py -v` (22 anti-drift tests)
  - `python3 -m pytest test_intelligent_data_detective.py -v` (22 core tests)
  - `python3 -m pytest test_error_handling_framework.py -v` (16/16 tests mandatory)
  - `python3 -m pytest test_validate_run.py tests/unit tests/integration -q`
- Format: `black <file>`, lint: `flake8 <file> --max-line-length=88 --extend-ignore=E203,E501`
- Reuse established canonical modules before introducing new abstractions.
<!-- repo-agent-bootstrap:managed:end -->
