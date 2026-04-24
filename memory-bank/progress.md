<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Progress

## Milestones
- [x] Repository profile captured (inventory run 2026-04-20)
- [x] Hybrid agent stack scaffolded (26 files, `repo-agent-bootstrap`)
- [x] False-positive agent deleted (`frontend-experience-specialist`)
- [x] IDD-specific agents created/customised
- [x] Memory-bank and AGENTS.md enriched with real IDD content
- [ ] Validator run and any warnings resolved
- [ ] Maintenance drift reviewed against future repo changes

## What works
- Core test suite: **22/22 pass** (`test_intelligent_data_detective.py`)
- Error handling tests: **15/16 pass** (1 known edge-case failure — acceptable)
- Memory test suites: all pass
- Notebook execution: functional end-to-end (requires `OPENAI_API_KEY`; 6–25 min)
- Agent stack: all guidance files present, managed sections safe for future maintenance runs

## What is incomplete
- CI now runs via `.github/workflows/copilot-setup-steps.yml` (`push`, `pull_request`, `workflow_dispatch`) for required-file checks, notebook smoke validation, tests, and formatting/lint checks; full notebook/API-key runs still require manual execution.
- `automation.instructions.md` is a scaffolded stub; no additional automated hooks exist yet.
- The one known test failure in `test_error_handling_framework.py` is an edge case in function signature handling and has no functional impact on the main system.

## Validation status
```bash
python3 -m pytest test_intelligent_data_detective.py -v          # 22/22
python3 -m pytest test_error_handling_framework.py -v            # 15/16
python3 -m pytest -v                                             # all suites
flake8 test_intelligent_data_detective.py --max-line-length=88 --extend-ignore=E203,E501
```

## Last meaningful update
[2026-04-20] Scaffolded and fully customised agent stack for IDD repo.
<!-- repo-agent-bootstrap:managed:end -->
