# Antigravity Preflight Configuration & Validation Architecture

This document records the canonical preflight configuration, rule-loading semantics, fail-closed notebook integrity gating, and CI-aligned regression baselines for Google Antigravity in the `intelligent_data_detective` repository (Issue #152).

---

## 1. Current Antigravity Configuration

### Agents Discovered
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
| `.agents/rules/team-coordination.md` | `model_decision` | Task-specific orchestration: activated dynamically when delegating, coordinating subagents, or managing the 5-stage lifecycle. | Multi-agent team coordination protocols, subagent dispatch matrix, 5-stage orchestration lifecycle, and one-writer discipline for IDD engineering tasks. |
| `.agents/rules/validation-and-gates.md`| `model_decision` | Task-specific gating: activated dynamically during testing, validation, pre-commit checks, or release gating. | Validation criteria, test execution suites, notebook integrity checks, and release quality gates for code modifications and pipeline runs. |

*Note on Path Corrections*: Stale links referencing `.agent/rules/` (singular) have been corrected to `.agents/rules/` across all documentation (`GEMINI.md`, agent manifests, and rules).

### MCP Inheritance & Memory Agents Status
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
   - A `PreToolUse` hook could be configured to prevent write tools from directly mutating `IntelligentDataDetective_beta_v5_patched.ipynb`, enforcing the one-way generation model through `_patch_notebook.py`.
   - A `PostToolUse` or `Stop` hook could automatically run `validate_notebook_integrity.py` whenever `_patch_notebook.py` is edited.
   - Per Issue #152 boundaries, these lifecycle hooks remain deferred to a separate issue.
2. **Subagent Session Caching**:
   - Custom subagent manifests discovered at Antigravity session startup are cached in memory for the duration of the conversation. Modifying `.agents/agents/<name>/agent.md` on disk takes full effect in new conversation sessions or via programmatic definition (`define_subagent`).
