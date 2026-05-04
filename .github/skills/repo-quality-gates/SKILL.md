---
name: repo-quality-gates
description: 'Run the test, lint, and typecheck commands for the repository, then summarize failures and next actions for coding agents.'
---

# Repo Quality Gates

Use this skill when you need to validate a change or interpret failing automation for this repository.

## Default validation commands
- `pip install langchain langchain-core langchain-openai langchain_experimental langgraph`
- `pytest tests/`
- `flake8 .`

## Workflow
1. Run the smallest relevant validation subset first.
2. Group failures by root cause instead of by raw output order.
3. Call out missing tooling or environment issues separately from code regressions.
4. Do not relax or delete tests just to produce a green run.
