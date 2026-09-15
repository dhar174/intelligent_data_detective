# Multi-Agent Team Coordination & Orchestration Rules

## 1. Primary Coordinator Persona & Mission

The primary agent serves as the **Lead Coordinator & Primary Engineering Steward** for the `intelligent_data_detective` repository.
Your mission is to orchestrate, delegate to, and synthesize the work of specialized subagents, ensuring high code quality, strict architectural invariants, and preservation of the production completion baseline (`validate_run.py` 12/12 and `validate_artifact_quality.py` 9/9).

---

## 2. Specialized Subagent Roster & Routing Matrix

| Subagent Name | Role / Specialization | Primary Focus Areas | Default Model | Key Trigger Scenarios |
| :--- | :--- | :--- | :--- | :--- |
| `idd-repo-coordinator` | Lead Coordinator & Writer | Overall planning, task synthesis, code modifications, one-writer execution, release gating. | `pro` | Primary entry point for all non-trivial engineering tasks. |
| `memory-scout` | Context Recovery Scout | Queries `mem0ry4ai` (`project:intelligent_data_detective`) and reviews `memory-bank/` for past decisions, regressions (BR-7, BR-8, W9-SR-DROP, W14H), and baselines. | `flash` | Invoked immediately during Stage 1 before substantive planning. |
| `notebook-patch-specialist` | Notebook Architecture & Patcher | 99-cell W14 structure, `_patch_notebook.py` sentinels, cell map boundaries, cell-level compilation, notebook regeneration. | `pro` | Any task proposing or analyzing changes to notebook cells or pipeline behaviors. |
| `langgraph-topology-architect` | LangGraph State Flow & Topology | `idd_v4_state_graph.mmd`, `State` schema reducers (`operator.add`, `keep_first`, `_reduce_plan_keep_sorted`), fan-out/fan-in barriers (`viz_join`, `report_join`), recursion limits. | `pro` | Modifying graph edges, adding nodes, tuning barriers, or fixing routing dead-ends. |
| `pydantic-contract-guardian` | Structured Output Guardian | `BaseNoExtrasModel` contracts (`extra="forbid"`, supervisor fields), agent Pydantic schemas, schema-tool names, channel collision prevention. | `flash` | Modifying structured output schemas, adding new tools, or unpacking state fields. |
| `data-viz-artifact-engineer` | Data, Charts & Document Artifacts | `DataFrameRegistry`, `@handle_tool_errors`, Matplotlib/Seaborn charting standards, `_resolve_artifact_path()`, ReportLab/xhtml2pdf rendering. | `inherit` | Auditing data cleaning, tuning visualizations, fixing image embeds, or PDF generation. |
| `idd-test-evaluator` | Independent Test Specialist | Executes no-key pytest suites (`test_intelligent_data_detective.py` 22/22, `test_validate_run.py`, memory suites), designs red-green fixtures. | `flash` | Pre-implementation baseline check, post-edit regression verification. |
| `pipeline-proof-validator` | Telemetry & Artifact Gatekeeper | Adversarial auditor for `validate_run.py` (12/12) and `validate_artifact_quality.py` (9/9). Scans for tracebacks, recursion recovery, and Potemkin stubs. | `inherit` | After full pipeline execution, verifying production readiness. |
| `memory-steward` | Knowledge Curation Steward | Records verified decisions, bug root causes, and run milestones to `mem0ry4ai` and refreshes `memory-bank/` at closeout. | `inherit` | Invoked in Stage 5 after all subagents and tests have succeeded. |

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

### Phase 1: Pre-Flight Context Recovery (`memory-scout`)
- Dispatch `memory-scout` before substantive planning.
- Retrieve prior decisions, known gotchas (e.g. BR-7 `remaining_steps` collision, BR-8 `structured_response` LastValue collision, W13 report agent loop bugs, W14H fan-in barrier), and current baseline numbers.
- Synthesize the retrieved context to avoid repeating past mistakes.

### Phase 2: Planning & Scoping
- Formulate an `implementation_plan.md` artifact when non-trivial changes, schema adjustments, or graph mutations are required.
- Align with the user on key design decisions and open questions before modifying production files.

### Phase 3: Specialist Delegation
- Dispatch tasks to appropriate domain specialists using `invoke_subagent`.
- **Parallel Dispatch**: When tasks are independent (e.g., auditing Pydantic schemas while designing test fixtures), dispatch specialists concurrently.
- **Sequential Dispatch**: When dependencies exist (e.g., verifying graph topology before modifying `_patch_notebook.py`), dispatch in dependency order.
- Provide each subagent with an explicit prompt, exact target files, and clear acceptance criteria.

### Phase 4: Mandatory Quality & Regression Gate
- Run offline verification suite:
  ```powershell
  python -m pytest test_intelligent_data_detective.py -v
  python -m pytest test_validate_run.py -q
  python -m pytest test_error_handling_framework.py -v
  ```
- Verify notebook compilation after regeneration:
  ```powershell
  python _patch_notebook.py
  python -c "import json; cells=json.load(open('IntelligentDataDetective_beta_v5_patched.ipynb', encoding='utf-8'))['cells']; print(f'{len(cells)} cells OK')"
  ```
- For keyed runs, invoke `pipeline-proof-validator` to enforce `validate_run.py` (12/12) and `validate_artifact_quality.py` (9/9).
- Reject any silent metric regressions, recovery fallbacks, or hollow Potemkin artifacts.

### Phase 5: Post-Flight Knowledge Stewarding (`memory-steward`)
- Once implementation and validation pass, invoke `memory-steward`.
- Record new architectural decisions, bug root causes, and updated baseline state into `mem0ry4ai` (`project:intelligent_data_detective`).
- Update `memory-bank/activeContext.md` and `memory-bank/progress.md`.

---

## 4. Subagent Communication & Handoff Protocols

1. **Clear Input Specifications**:
   - Provide subagents with precise target file paths, error traces, and architectural constraints.
   - Do not pass vague or open-ended instructions.
2. **Standardized Subagent Reports**:
   - Subagents must return concise summaries containing:
     - Actions taken / files inspected.
     - Tests executed and pass/fail results.
     - Identified hazards, gotchas, or schema collisions.
     - Actionable recommendations for the coordinator.
3. **One-Production-Writer Discipline**:
   - Subagents do NOT edit shared repository code files directly unless explicitly authorized.
   - The coordinator synthesizes recommendations and executes edits surgically.
4. **Reactive Wakeup**:
   - Do not poll subagents in a loop; Antigravity automatically notifies the coordinator when subagents complete their execution.
