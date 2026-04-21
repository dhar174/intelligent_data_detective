<!-- repo-agent-bootstrap:file-kind=current-plan -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Current Plan

## Objective
Bootstrap or refresh the repository's AI agent stack so future agents inherit accurate guidance and reusable workflows.

## Scope
Included:
- Root guidance files
- Custom agents and selected skills
- Memory-bank and runbook scaffolding

Excluded:
- Unpinned third-party imports
- Destructive automation or hooks

## Plan
1. Inventory the repository's purpose, stack, commands, workflows, and current AI assets.
2. Render a repo-specific hybrid stack with managed sections and provenance metadata.
3. Validate the generated stack and hand back any drift or follow-up items.

## Validation
- `pip install langchain langchain-core langchain-openai langchain_experimental langgraph`
- `pytest tests/`
- `flake8 .`

## Completion criteria
- Core files exist and validate cleanly.
- Each agent exposes the delegation tool and focused tool access.
- Managed updates preserve user-authored content outside generated sections.
<!-- repo-agent-bootstrap:managed:end -->
