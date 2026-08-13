---
name: repo-quality-gates
description: 'Run the test, lint, and typecheck commands for the repository, then summarize failures and next actions for coding agents.'
---

# Repo Quality Gates

Use this skill when you need to validate a change or interpret failing automation for this repository.

## Default validation commands
- `pip install langchain langchain-core langchain-openai langchain_experimental langgraph`
- `python -m pytest test_validate_run.py -q`
- `python validate_run.py --latest --log-path notebook_run_log.txt --window 180`
- `python validate_artifact_quality.py --latest`
- `python -m pytest tests/`
- `flake8 .`

## IDD completion gates
For notebook-completion work, the current W14 baseline is `IDD_run_run_default_id-20260504-1338-b3079aea`. A clean proof means:
- `validate_run.py` scores 12/12.
- `validate_artifact_quality.py` scores 9/9.
- The log contains native structured-output markers and 3/3 visualization fan-in.
- The log has zero recovery/final-hop/native-failure/path-normalization warnings.
- Root `final_report.html`, `final_report.md`, and `final_report.pdf` exist and embed resolving visualizations.

## Workflow
1. Run the smallest relevant validation subset first.
2. Group failures by root cause instead of by raw output order.
3. Call out missing tooling or environment issues separately from code regressions.
4. Do not relax or delete tests just to produce a green run.
