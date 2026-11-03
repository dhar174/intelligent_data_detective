# Antigravity Preflight Configuration & Validation Architecture

This document records the canonical preflight configuration, rule-loading semantics, fail-closed notebook integrity gating, and CI-aligned regression baselines for Google Antigravity in the `intelligent_data_detective` repository (Issue #152).

## Lifecycle investigation and repair (2026-11-02)

The earlier preflight verified agent invocation in a particular Antigravity
session, but deferred lifecycle hooks (see section 5). It did not establish
automatic memory recovery or closeout in a new chat.

Comparison with the default branch of
[`backrooms_lore_scraper`](https://github.com/dhar174/backrooms_lore_scraper/blob/main/.agents/hooks.json)
found these concrete differences:

- IDD had no `.agents/hooks.json`, no hook adapters, and no bundled memory
  plugin. Backrooms registered mem0ry4ai `PreInvocation` and `Stop` handlers,
  translating Antigravity payloads to its shared local memory installation.
- IDD's team-coordination rule used `model_decision`. That exposes a description
  for optional retrieval, rather than continuously loading the full lifecycle.
  Dedicated memory workflow proposals were also explicitly inactive under
  `docs/evidence/proposals/`.
- IDD's `.gitignore` ignored every `*.json`, including new hook registrations.
- Backrooms uses machine-specific Windows interpreter and checkout paths. Copying
  those commands into IDD would target the wrong repository.

The repair registers `idd-memory-lifecycle` in `.agents/hooks.json`, explicitly
unignores that file, and changes team coordination to `always_on`. The adapter
`.agents/hooks/idd_memory.py` runs `hooks/session_start.py` from the shared
mem0ry4ai installation on `PreInvocation` **only when `invocationNum == 0`**.
It injects recalled context and the primary coordinator's memory checkpoints.
`Stop` runs shared `hooks/capture.py` when `fullyIdle` is true. Child processes
have a 25-second timeout inside the 30-second hook timeout. Diagnostics go to
stderr; stdout contains only Antigravity JSON. Missing/failed recovery is also
reported in the injected context, with a local memory-bank fallback.

PR #178 review also exposed an input-encoding gap: the adapter now reads JSON
from binary stdin, so valid UTF-8 paths survive ASCII and CP1252 launchers.
Tests send unescaped UTF-8 bytes and assert exact transcript-path preservation
for both startup and idle capture, including accented and Chinese folder names.
Invalid UTF-8 returns the existing safe fallback without invoking the backend.

The adapter derives project identity from its own location, so another folder
listed first in `workspacePaths` cannot redirect IDD's memory scope. It does not
change production IDD code or the notebook. Primary-only orchestration avoids
recursive memory-agent bookends in delegated sessions.

### Local prerequisites and runtime verification

1. Open **`intelligent_data_detective` itself** as the Antigravity project folder.
   Opening only its parent does not discover customizations in child folders.
2. Ensure `python` resolves to Python 3.10+ with the dependencies used by your
   mem0ry4ai installation. Antigravity CLI 1.4.12 on Windows launches commands
   from the directory containing `.agents/hooks.json`. The script path is
   `hooks/idd_memory.py` relative to `.agents`. The adapter derives the
   repository root from its own path before invoking the shared memory scripts.
3. Shared memory scripts must exist at `~/mem0ry4ai/hooks/session_start.py` and
   `~/mem0ry4ai/hooks/capture.py`, or set `MEM0RY4AI_HOME` to the shared engine's
   directory **before launching Antigravity**. The bridge reuses the existing
   store; it neither installs the engine nor configures its MCP server.
4. Enable the `mem0ry4ai` MCP server in Antigravity independently. Verify the
   actual tools in the parent and a memory subagent. The existing `inheritMcp`
   metadata and historical successful query below are not evidence of current
   server availability. Do not re-add `call_mcp_tool` to built-in tool lists.
5. Start a fresh chat. Check Hooks in Customizations (CLI: `/hooks`), the custom
   agent list (CLI: `/agents`), and MCP availability (CLI: `/mcp`). Expect
   `idd-memory-lifecycle`, all nine repo agents, and mem0ry4ai tools. The current
   custom-subagent documentation lists **App and CLI** support; verify capability
   separately if using the standalone IDE or an older version.
6. In a small non-trivial task, observe actual `memory-scout` invocation before
   planning, relevant specialist delegation, and `memory-steward` before final
   closeout. Confirm successful MCP recall and search-before-save, or an explicit
   unavailability report and local fallback. An `action: none` closeout is valid.
7. At idle stop, check hook diagnostics and the shared engine's staging output.
   `capture.py` stages transcripts; **it does not itself certify a curated memory
   save**. A zero child exit reports completion of the script, not a count of
   newly saved memories. Verify semantic saves through MCP results.

Offline checks exercise the registered commands from the hook configuration
directory with temporary stand-in engine scripts. They cover paths with spaces,
multi-root scope, backend failure, idle capture, and raw UTF-8 input under
non-UTF-8 launchers.

### Runtime evidence checklist

The initial Windows run on CLI 1.4.12 reproduced launch failures for both
handlers because the commands included an extra `.agents` path segment. A
config-relative probe recovered context and staged a real transcript. See the
[PR #178 smoke-test report](https://github.com/dhar174/intelligent_data_detective/pull/178#issuecomment-6592496756)
for the results and coverage limits. Use this checklist for fresh runtime runs.

| Check | Evidence to record in the PR after the local smoke test |
| --- | --- |
| Actual environment | Antigravity surface/version, OS, tested commit, workspace path, and resolved Python interpreter. |
| Discovery and launch | Loaded `idd-memory-lifecycle` handlers and all nine custom agents; actual hook invocation with successful script diagnostics. |
| Parent/subagent MCP access | Successful project-scoped recall in the coordinator and an invoked `memory-scout`; report missing tools explicitly. |
| Team coordination | `memory-scout` before planning, relevant specialist invocation, and `memory-steward` after verification without recursive bookends. |
| Semantic memory closeout | Search-before-write and a successful curated save/read-back, or an explained `action: none` when nothing merits saving. |
| Idle capture | Actual `capture.py` invocation and the shared engine's captured-session/staging record for the conversation. |

Use a small non-trivial repository task and record redacted tool results or log
references. A successful pytest run, a historical discovery result, and a child
process exit code alone do not satisfy these checks. This is an evidence
checklist, not a record that any local runtime checks have passed.

### Copilot setup activation and hosted verification

CI provenance tests additionally require historical Git objects. Copilot's
generated review job used checkout depth 2 even though this repository requests
full history. The explicit post-checkout history step restores shallow clones
before validation, preserving the evidence tests' fail-closed assertions.

Follow-up run `41547684581` checked out `e883ce6` but its generated setup omitted
`Ensure full history for provenance checks` entirely. It repeated the same four
missing-history fixture errors (950 passed, 3 skipped). The step is absent on
`main` and present on the PR branch: this run did not exercise the proposed fix.
GitHub documents the setup workflow's default-branch requirement and its
override of checkout depth in
[Configure the development environment for agents on GitHub Actions](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/customize-the-agent-environment).
Do not repeatedly rerun the unchanged generated setup or weaken provenance
assertions. Deploy the setup change through the normal reviewed merge, then
check that a fresh Copilot run includes and executes the history step.

The branch's workflow can also be tested independently, without merging:

```bash
gh workflow run copilot-setup-steps.yml \
  --repo dhar174/intelligent_data_detective \
  --ref codex/fix-antigravity-memory-hooks
```

Record that run's tested SHA and result separately from generated Copilot setup
and from the local Antigravity smoke test. They validate different environments.

[Hosted branch run `41553081052`](https://github.com/dhar174/intelligent_data_detective/actions/runs/41553081052)
completed successfully on `e883ce6e915fa70355e7e725c75ff3120eb97728`:
unit/integration **954 passed, 3 skipped** (all five provenance tests passed),
scientific kernel **3 passed**, SQLite-present notebook **21 passed**, synthetic
proof runner **27 passed**, SQLite-uninstalled notebook **1 passed** (20
intentionally deselected), root regressions **110 passed**, prompt regressions
**17 passed**, and mandatory error handling **16 passed**. Notebook/graph
validation, formatting, and lint also passed. The hosted checkout used full
history, so the history step was a no-op; the earlier disposable depth-2 clone
separately established actual restoration. This green Ubuntu run does not
complete the Antigravity runtime evidence checklist above.

Sources checked on 2026-11-02:
[Hooks](https://antigravity.google/docs/hooks/),
[Rules](https://antigravity.google/docs/rules/),
[Customization discovery](https://antigravity.google/docs/customization-discovery/),
and [Custom subagents](https://antigravity.google/docs/subagents/).

---

## 1. Current Antigravity Configuration

### Historical agent discovery (earlier preflight session)

The statuses below describe the earlier preflight session only. They do not
certify discovery or invocation for this repair; use the runtime evidence
checklist above to record current results.
Antigravity automatically discovers custom agents defined under `.agents/agents/<name>/agent.md`:

| Agent Name | Persona / Role | Configuration Type | Default Model | Discoverable Status |
| :--- | :--- | :--- | :--- | :--- |
| `idd-repo-coordinator` | Primary Engineering Coordinator & Lead Writer | Main Agent (`mainAgent: true`, `subagent: false`) | `pro` | **Verified** (Primary session agent) |
| `idd-test-evaluator` | Independent Test Specialist (no-key offline suites) | Subagent (`subagent: true`) | `flash` | **Verified** (Invoked & verified in runtime smoke check) |
| `notebook-patch-specialist` | Notebook Architecture & Patch Engine Specialist | Subagent (`subagent: true`) | `pro` | **Verified** |
| `langgraph-topology-architect`| State Flow, Graph Topology & Barrier Specialist | Subagent (`subagent: true`) | `pro` | **Verified** |
| `pydantic-contract-guardian` | Structured Output & Schema Specialist | Subagent (`subagent: true`) | `flash` | **Verified** |
| `data-viz-artifact-engineer` | Data Cleaning, Visualizations & PDF Artifacts | Subagent (`subagent: true`) | `inherit` | **Verified** |
| `pipeline-proof-validator` | Telemetry & Artifact Gatekeeper (12/12 + 9/9) | Subagent (`subagent: true`) | `inherit` | **Verified** |
| `memory-scout` | Pre-flight Context Recovery Scout | Subagent (`subagent: true`) | `flash` | **Verified** (Tool declaration repaired) |
| `memory-steward` | Post-flight Knowledge Curation Steward | Subagent (`subagent: true`) | `inherit` | **Verified** (Tool declaration repaired) |

### Rules Discovered & Triggers
The repository provides modular rules under `.agents/rules/` configured with valid YAML frontmatter:

| Rule File | Trigger Mode | Trigger Rationale | Description |
| :--- | :--- | :--- | :--- |
| `.agents/rules/code-architecture.md` | `always_on` | Truly universal repository constraints (never edit patched notebook, 99-cell invariant, channel collision avoidance, `DataFrameRegistry`, no `src/` directory). Active on every turn. | Non-negotiable architectural invariants, notebook generation rules, state reducers, Pydantic contracts, and DataFrameRegistry conventions. |
| `.agents/rules/team-coordination.md` | `always_on` | Required memory checkpoints and primary-coordinator lifecycle must be present before planning, rather than selected on demand. | Multi-agent team coordination protocols, subagent dispatch matrix, 5-stage orchestration lifecycle, and one-writer discipline for IDD engineering tasks. |
| `.agents/rules/validation-and-gates.md`| `model_decision` | Task-specific gating: activated dynamically during testing, validation, pre-commit checks, or release gating. | Validation criteria, test execution suites, notebook integrity checks, and release quality gates for code modifications and pipeline runs. |

*Note on Path Corrections*: Stale links referencing `.agent/rules/` (singular) have been corrected to `.agents/rules/` across all documentation (`GEMINI.md`, agent manifests, and rules).

### MCP Inheritance & Memory Agents Status

The following success claims record the earlier preflight session; they are not
a certification of the current user's Antigravity/MCP environment.
- **`inheritMcp: true`**: Successfully recognized by Antigravity. When enabled on subagents, MCP tools (including `mem0ry4ai` and `github-mcp-server`) are inherited by the agent execution context.
- **`call_mcp_tool` Resolution Fix**:
  - *Root Cause*: Previous frontmatter in `memory-scout/agent.md` and `memory-steward/agent.md` listed `call_mcp_tool` inside the `tools:` list. In Antigravity's component architecture, `tools:` resolves built-in CLI tools (e.g., `view_file`, `list_dir`), whereas `call_mcp_tool` is an internal MCP dispatcher. This caused `failed to construct executor: unknown component: tool "call_mcp_tool" not found in registry`.
  - *Fix*: Removed `call_mcp_tool` from the `tools:` list. With `inheritMcp: true`, MCP tools are made available automatically.
- **`mem0ry4ai` MCP Availability**:
  - The `mem0ry4ai` MCP server is active in the environment with lazy tools: `memory_search`, `memory_get`, `memory_list`, `memory_resume`, `memory_add`, `memory_note`, `memory_promote`, `session_search`.
  - Empirically verified: read-only query `memory_search(query="BR-7")` executed successfully and returned contextual records.
  - Local repository memory-bank recovery (`memory-bank/activeContext.md`) is verified and functional as fallback and immediate context.

---

## 2. Canonical Notebook Validation Gate

### Exact Command
```powershell
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb
```

And for full static topology validation (guarded harness invariant automatically suppresses in-notebook pip/installer calls under all caller environments):
```powershell
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
```

### Exact Invariants Checked by `validate_notebook_integrity.py`
1. **File Existence & JSON Validity**: Fails nonzero (exit 1) if the target notebook file does not exist or contains invalid JSON.
2. **Top-Level & Cell Structure**: Fails nonzero (exit 1) if the `cells` key is missing or not a JSON list, if any cell lacks or has an unknown `cell_type` (must be `code`, `markdown`, or `raw`), or if `source` is missing, not a string/list, or contains non-string elements.
3. **Exact Cell Count Invariant (99 cells)**: Enforces that the notebook contains exactly 99 cells (default for W14 baseline). Fails nonzero (exit 1) on 0, 98, 100, or any mismatch.
4. **Code Cell Code-Object Compilation**: Compiles every code cell with real code-object compilation semantics (`compile(..., mode='exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)`). Rejects module-level `return 42`, `break`, `continue`, duplicate function formal parameters, `nonlocal`, and `yield` that falsely pass AST-only parsing. Preserves valid top-level `await` without executing runtime code.
5. **Safe Magic, Payload & Statement Structure Handling**:
   - **Python-bearing cell magics** (`%%python`, `%%time`, `%%timeit`, `%%capture`, `%%prun`): Leading directive is handled while validating underlying Python code. For `%%timeit` and `%timeit`, options are strictly parsed and validated (`-n <N>` accepts non-negative integers, with zero selecting automatic loop count; `-r <R>` requires a positive integer; `-p <P>` requires a non-negative integer; supported flags are `-t`, `-c`, `-o`, and `-q`). Invalid options (e.g. `-n banana`, negative `-n`, missing arguments, non-integer values, unrecognized flags such as `--quiet`) fail validation with actionable diagnostics. Remaining Python setup and statement bodies are preserved and compiled. Broken setup code (e.g. `%%timeit x = (`) fails compilation.
   - **Recognized non-Python cell magics** (`%%bash`, `%%sh`, `%%html`, `%%javascript`, `%%js`, `%%latex`, `%%writefile`, `%%svg`, `%%cmd`, `%%ruby`, `%%perl`): Entire cell is safely excluded from Python compilation, explicitly reported in validation diagnostics, and not counted as compiled Python code cells in accounting.
   - **Unknown / unsupported cell magics**: Fail validation with nonzero exit code and an actionable diagnostic identifying the unsupported magic.
   - **Python-bearing line magics** (`%time`, `%timeit`, `%prun`): Validate Python expression payloads and options. Broken expressions or invalid options fail compilation.
   - **Statement and block structure preservation**: Standalone shell commands (`!cmd`) and line magics (`%pwd`) inside control-flow blocks (`if True:\n    !echo ok`) transform to `pass` at identical indentation, preventing `IndentationError` while preserving block syntax.
   - **Shell and magic assignments**: Assignments (`files = !echo ok`, `res = %pwd`) transform to valid Python (`files = []`, `res = None` or evaluated expressions) without loss of statement structure.
   - **Multiline modulo expressions**: Multi-line expressions with continuation lines starting with `%` (e.g. modulo arithmetic inside parentheses) are preserved intact and not misidentified as line magics.
   - **Mid-cell `%%` placement**: Left intact so invalid mid-cell directives trigger `SyntaxError`.
   - **Multiline string literals & delimiter tracking**: Unified delimiter tracking ensures characters inside multiline strings and comments do not alter open delimiter depth, and trailing delimiters/expressions on boundary lines (e.g. `x = ("""alpha\nbeta""")\n%pwd`) correctly restore delimiter depth so subsequent line magics and Python code compile cleanly without false rejections. String literals containing `?`, `%`, or `!` are preserved intact.
6. **Actionable Diagnostics**: When a failure occurs, outputs:
   - Cell index (e.g. `Cell 4`)
   - Cell ID if present (e.g. `(id: RCmRvBsV-i4t)`)
   - Exception type (`SyntaxError`, `IndentationError`)
   - Line number and column offset within the cell
   - Offending source code line
7. **Zero Runtime Side Effects**: Never executes code cells, never imports runtime LLM models, never calls the network, and never mutates the file.

### Expected Success Output
```text
PASSED: Notebook 'IntelligentDataDetective_beta_v5_patched.ipynb' verified (99 cells, all code cells compiled cleanly).
```
With `-v`:
```text
PASSED: Notebook 'IntelligentDataDetective_beta_v5_patched.ipynb' verified (99 cells, all code cells compiled cleanly).
  Successfully validated 99 cells (42 code cells compiled cleanly).
```

### Expected Failure Behavior
- Nonexistent file:
  ```text
  FAILED: Notebook integrity check failed for 'non_existent.ipynb':
    - Notebook file not found: non_existent.ipynb
  ```
  Exit code: `1`.
- Cell count mismatch (e.g. unpatched 98-cell notebook):
  ```text
  FAILED: Notebook integrity check failed for 'IntelligentDataDetective_beta_v5.ipynb':
    - Cell count mismatch in IntelligentDataDetective_beta_v5.ipynb: expected exactly 99 cells, found 98
  ```
  Exit code: `1`.
- Syntax error in code cell:
  ```text
  FAILED: Notebook integrity check failed for 'bad_syntax.ipynb':
    - Cell 42 (id: bad_cell_42) syntax compilation failed: SyntaxError: invalid syntax at line 2, column 5
        Source: if x = 1:
  ```
  Exit code: `1`.

---

## 3. Canonical No-Key Regression Gate

The canonical no-key test commands align directly with current CI (`.github/workflows/copilot-setup-steps.yml`):

### 1. Validator, Unit & Integration Suites
```powershell
python -m pytest test_validate_run.py tests/unit tests/integration -q
```
- **Standard**: All tests must PASS (baseline reference: 463 passed, 9 skipped; includes `test_graph_validation_safety.py` 12 tests).
- **Scope**: Covers agent messages, artifact paths, BaseNoExtrasModel contracts, tool error handling, model configs, patcher integrity, reducers, DataFrameRegistry, supervisor routing edge cases, and graph harness installation suppression safety.

### 2. Core Pipeline & Memory Enhancement Suites
```powershell
python -m pytest test_intelligent_data_detective.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py -v
```
- **Standard**: All tests must PASS (baseline reference: 77 passed: 22 core + 55 memory).
- **Scope**: Core DataFrameRegistry caching, State reducers, tool decorators, prompt rendering, Pydantic model schemas, memory namespaces, and TTL lifecycle.

### 3. Prompt Template Formatting Suites
```powershell
python -m pytest test_prompt_formatting.py test_prompt_template_fixes.py -v
```
- **Standard**: All tests must PASS (baseline reference: 17 passed).
- **Scope**: Prompt brace escaping and template validation logic.

### 4. Notebook Integrity Test Suite
```powershell
python -m pytest test_validate_notebook_integrity.py -v
```
- **Standard**: All 33 tests must PASS.
- **Scope**: Validates 99-cell exact match, 0/98/100 cell rejection, malformed JSON, missing cells, bad cell types, real code-object compilation errors (return, break, continue, duplicate args, nonlocal, yield), top-level await, zero runtime side effects, %%timeit setup parsing and broken setup rejection, %time/%timeit/%prun expressions, unsupported cell magics, control-flow block indentation preservation, shell/magic assignments, multiline modulo expressions, %%timeit and %timeit option validation (F5), and multiline string delimiter boundary and open delimiter restoration (F6).

### 5. Error Handling Framework Suite & CI Exception Policy
```powershell
python -m pytest test_error_handling_framework.py -v --deselect test_error_handling_framework.py::TestErrorHandlingFramework::test_integration_with_different_function_signatures
```
- **CI Parity Policy**:
  - All tests in `test_error_handling_framework.py` are **strictly blocking**, with exactly ONE isolated exception:
    `test_error_handling_framework.py::TestErrorHandlingFramework::test_integration_with_different_function_signatures`
  - In CI, this specific test is run as a non-blocking step (`continue-on-error: true`).
  - Any other failure in `test_error_handling_framework.py` is strictly blocking.
  - Generic policies like "15/16 passing is acceptable" are prohibited; the exception is test-specific and identified by exact test ID.

---

## 4. Scope Boundary

This preflight task establishes:
1. Valid rule activation metadata and working link resolution.
2. A fail-closed notebook integrity compiler gate (`validate_notebook_integrity.py`).
3. Alignment of Antigravity instructions with repository CI.
4. Empirical verification of subagent discovery, invocation, and MCP memory tool behavior.

**Explicit Scope Limitation**:
- This preflight does **NOT** certify IDD production runtime behavior.
- It does **NOT** replace the live, keyed production proof gates (`validate_run.py` 12/12 and `validate_artifact_quality.py` 9/9).
- It does **NOT** authorize or implement changes to LangGraph topology, reducers, prompts, supervisor routing, report pipelines, or canonical-core architectures.

---

## 5. Follow-ups & Recommendations

1. **Lifecycle Hooks (`.agents/hooks.json`)**:
   - Memory lifecycle hooks were added in the 2026-11-02 repair described above.
   - A `PreToolUse` hook could be configured to prevent write tools from directly mutating `IntelligentDataDetective_beta_v5_patched.ipynb`, enforcing the one-way generation model through `_patch_notebook.py`.
   - A `PostToolUse` or `Stop` hook could automatically run `validate_notebook_integrity.py` whenever `_patch_notebook.py` is edited.
   - The notebook write/validation hooks remain separate follow-up work; the
     original Issue #152 deferral no longer describes the memory lifecycle hooks.
2. **Subagent Session Caching**:
   - Custom subagent manifests discovered at Antigravity session startup are cached in memory for the duration of the conversation. Modifying `.agents/agents/<name>/agent.md` on disk takes full effect in new conversation sessions or via programmatic definition (`define_subagent`).
