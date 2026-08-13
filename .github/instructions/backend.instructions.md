---
applyTo: "*.py,test_*.py"
---

<!-- repo-agent-bootstrap:file-kind=path-instructions -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Backend instructions

- Python files are at the **repo root** — there is no `src/` or `tests/` directory.
  Relevant files: `memory_enhancements.py`, `demo_memory_enhancement.py`, `demo_memory_lifecycle.py`,
  `test_intelligent_data_detective.py`, `test_error_handling_framework.py`,
  `test_memory_categorization.py`, `test_memory_integration.py`, `test_memory_lifecycle.py`.
- Preserve existing interfaces and function signatures unless the task explicitly changes them.
- Add or update test coverage for any behaviour change.
- Run the smallest set of relevant tests before declaring a change done:
  - `python3 -m pytest test_intelligent_data_detective.py -v` (22 pass)
  - `python3 -m pytest test_error_handling_framework.py -v` (15/16 pass — 1 known edge-case)
- Format: `black <file>`, lint: `flake8 <file> --max-line-length=88 --extend-ignore=E203,E501`
- Reuse established helper modules before introducing new abstractions.
- The one source of truth for agent logic is the notebook — these Python files are support/test code only.
<!-- repo-agent-bootstrap:managed:end -->
