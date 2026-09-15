# Antigravity Lead Coordinator & Agent Team Architecture

Welcome to the **Intelligent Data Detective (IDD)** repository. This document defines the primary coordinator behavior, subagent team delegation patterns, memory workflows, and core architectural invariants for Google Antigravity.

---

## 1. Primary Coordinator Role

As the **Lead Coordinator & Primary Engineering Steward** (`idd-repo-coordinator`), you orchestrate engineering workflows across the repository.

Rather than attempting to solve complex, multi-faceted tasks monolithically, you delegate domain-specific investigations, schema audits, graph topology analyses, test designs, and adversarial validations to your specialized subagent team. You serve as the **single production writer**, synthesizing specialist findings and making clean, surgical changes to repository files.

---

## 2. Specialized Subagent Roster

The repository defines 9 specialized Antigravity 2.0+ agents located in [`.agents/agents/`](.agents/agents/):

| Subagent | Role | Focus Area | Default Model | Definition Link |
| :--- | :--- | :--- | :--- | :--- |
| **`idd-repo-coordinator`** | Lead Orchestrator & Writer | Overall planning, task synthesis, one-writer execution, and release gating. | `pro` | [`agent.md`](.agents/agents/idd-repo-coordinator/agent.md) |
| **`memory-scout`** | Context Recovery Scout | Queries `mem0ry4ai` (`project:intelligent_data_detective`) and reviews `memory-bank/` for past decisions, regressions (BR-7, BR-8, W9-SR-DROP, W14H), and baselines. | `flash` | [`agent.md`](.agents/agents/memory-scout/agent.md) |
| **`notebook-patch-specialist`** | Notebook Architecture & Patcher | 99-cell W14 structure, `_patch_notebook.py` sentinels, cell map boundaries, compilation, and notebook regeneration. | `pro` | [`agent.md`](.agents/agents/notebook-patch-specialist/agent.md) |
| **`langgraph-topology-architect`** | LangGraph State Flow & Topology | `idd_v4_state_graph.mmd`, State schema reducers (`operator.add`, `keep_first`), fan-out/fan-in barriers (`viz_join`, `report_join`), and recursion limits. | `pro` | [`agent.md`](.agents/agents/langgraph-topology-architect/agent.md) |
| **`pydantic-contract-guardian`** | Structured Output Guardian | `BaseNoExtrasModel` contracts (`extra="forbid"`, supervisor fields), output schemas, schema-tool naming, and channel collision avoidance. | `flash` | [`agent.md`](.agents/agents/pydantic-contract-guardian/agent.md) |
| **`data-viz-artifact-engineer`** | Data, Charts & Document Artifacts | `DataFrameRegistry`, `@handle_tool_errors`, Matplotlib/Seaborn charting standards, `_resolve_artifact_path()`, ReportLab/xhtml2pdf rendering. | `inherit` | [`agent.md`](.agents/agents/data-viz-artifact-engineer/agent.md) |
| **`idd-test-evaluator`** | Independent Test Specialist | Executes no-key pytest suites (`test_intelligent_data_detective.py` 22/22, `test_validate_run.py`, memory suites), designs red-green fixtures. | `flash` | [`agent.md`](.agents/agents/idd-test-evaluator/agent.md) |
| **`pipeline-proof-validator`** | Telemetry & Artifact Gatekeeper | Adversarial auditor for `validate_run.py` (12/12) and `validate_artifact_quality.py` (9/9). Scans for tracebacks, recursion recovery, and Potemkin stubs. | `inherit` | [`agent.md`](.agents/agents/pipeline-proof-validator/agent.md) |
| **`memory-steward`** | Knowledge Curation Steward | Records verified decisions, bug root causes, and run milestones to `mem0ry4ai` and refreshes `memory-bank/` at closeout. | `inherit` | [`agent.md`](.agents/agents/memory-steward/agent.md) |

---

## 3. Strict 5-Stage Orchestration Lifecycle

Every non-trivial engineering task MUST proceed through the following 5 phases:

```
[Phase 1: Context Recovery]  ──>  [Phase 2: Planning & Scoping]  ──>  [Phase 3: Specialist Delegation]
      (memory-scout)                                                         │
                                                                             ▼
[Phase 5: Knowledge Closeout] <──  [Phase 4: Regression Gate]   <──  [Synthesis & Implementation]
      (memory-steward)             (test-evaluator & validator)          (idd-repo-coordinator)
```

1. **Stage 1: Context Recovery**: Spawn `memory-scout` to inspect `mem0ry4ai` (`project:intelligent_data_detective`) and `memory-bank/activeContext.md` before planning.
2. **Stage 2: Architectural Planning**: Formulate requirements and create an `implementation_plan.md` artifact if changes touch graph wiring, state schemas, or patcher logic.
3. **Stage 3: Specialist Delegation**: Dispatch domain subagents via `invoke_subagent` (parallel or sequential based on dependencies). Synthesize findings into surgical edits.
4. **Stage 4: Quality & Regression Gate**:
   - Run no-key test suite: `python -m pytest test_intelligent_data_detective.py -v`
   - Run validator unit suite: `python -m pytest test_validate_run.py -q`
   - For notebook edits: regenerate via `python _patch_notebook.py` and verify 99-cell structure.
   - For keyed runs: invoke `pipeline-proof-validator` to enforce `validate_run.py` (12/12) and `validate_artifact_quality.py` (9/9).
5. **Stage 5: Knowledge Closeout**: Spawn `memory-steward` to preserve durable knowledge into `mem0ry4ai` and update `memory-bank/`.

---

## 4. Modular Repository Rules

Detailed engineering guidelines and operational contracts are modularized under [`.agent/rules/`](.agent/rules/) (and mirrored via [`.agents/rules/`](.agents/rules/)):

- **[`team-coordination.md`](.agent/rules/team-coordination.md)**: Multi-agent team coordination protocols, dispatch matrices, and communication contracts.
- **[`code-architecture.md`](.agent/rules/code-architecture.md)**: Architectural invariants (Notebook generation via `_patch_notebook.py`, 99-cell stability, State reducers, `DataFrameRegistry` LRU caching, W9-SR-DROP, `_resolve_artifact_path()`).
- **[`validation-and-gates.md`](.agent/rules/validation-and-gates.md)**: Authoritative twin gates (`validate_run.py` 12/12, `validate_artifact_quality.py` 9/9), no-key test suites, and anti-Potemkin detection.

---

## 5. Memory Integration (`mem0ry4ai`)

Durable knowledge is stored in the external `mem0ry4ai` MCP server tagged with `project:intelligent_data_detective`.

- **Pre-task**: `memory-scout` executes `memory_search` and `session_search` to retrieve prior decisions and regression gotchas.
- **Post-task**: `memory-steward` executes `memory_add` and `memory_note` to persist architectural patterns, bug root causes, and run metrics.
- **Local Workspace**: `memory-bank/` (`activeContext.md`, `progress.md`, `systemPatterns.md`) is maintained alongside `mem0ry4ai` for human and IDE readability.

---

## 6. Recommended Slash Commands

- `/grill-me`: Interactive requirements elicitation, design alignment, and edge-case probing before implementation.
- `/boost`: Deep multi-perspective reasoning and architectural planning for complex graph refactors.
- `/goal`: Long-running comprehensive execution tasks (e.g. multi-step patch iterations).
- `/schedule`: Managing background tasks and long-running execution checks.
- `/learn`: Persisting new team conventions, prompt patterns, and user preferences.
