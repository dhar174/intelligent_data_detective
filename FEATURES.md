# Intelligent Data Detective: Features, Tools, Architecture, and Innovations

> **Feature inventory | v5 canonical-core iteration | Updated October 20, 2026**
>
> **Status:** CP0–CP4 are merged into `main` (CP4: [PR #163](https://github.com/dhar174/intelligent_data_detective/pull/163), merge commit `08bdc2b9e61fe67423dcc4528895bedd25c6d209`). The current revision has **not** completed the post-migration, two-run production proof in [#165](https://github.com/dhar174/intelligent_data_detective/issues/165). This catalog distinguishes implemented code from empirically validated behavior, optional functionality, and roadmap proposals.

Intelligent Data Detective (IDD) is a LangChain/LangGraph-based **autonomous multi-agent data investigation and reporting system**. Specialized agents inspect datasets, clean and transform data, compute statistics, investigate patterns, generate visualizations, and produce multi-format analytical reports. A structured supervisor, typed shared state, memory/retrieval subsystems, executable data tools, and independent validation gates coordinate and constrain the workflow.

This file is the maintained **feature and tooling catalog**. For an introduction and running instructions, see [README.md](README.md); for implementation boundaries see [the canonical-core migration plan](docs/plans/idd-canonical-core-migration.md) and [the runtime notebook](IntelligentDataDetective_beta_v5_patched.ipynb).

## 1. Capability at a glance

| Dimension | Current implementation | Evidence / qualification |
| --- | --- | --- |
| Agent workflow | **15 LangGraph nodes**, **33 edges** (21 conditional), parallel visualization and report-section stages | Canonical `idd_graph.py`; graph/integrity tests |
| Shared workflow state | **70 state fields**, custom reducers, typed agent artifacts and messages | `idd_state.py` |
| Analytical tools | **40 decorated tools** in the v5 notebook tool ecosystem, plus LangChain memory tools | Inventory below; tool availability varies by agent; not a claim that all 40 have fresh end-to-end coverage |
| Data management | DataFrame registry, ID-based access, thread-safe LRU, reload/persistence, context-local overrides | `idd_registry.py` |
| Analytics | Data profiling, data-quality operations, statistics, basic ML, user-directed Python execution | Tool inventory below |
| Figures | Histogram, scatter, heatmap, box and violin plotting; parallel workers; feedback | Plotting functions and visualization graph nodes |
| Reports | HTML, Markdown and PDF; report outline, section fan-out, join and packaging | Historical successful W14 proof; current-head live proof pending |
| Runtime memory | LangMem tools, scoped memory namespaces, in-process indexed store and selected notebook-local retrieval helpers | The notebook does **not** import the standalone lifecycle or adaptive retrieval modules; retention enforcement is not uniform |
| Standalone memory policies | Six-category TTL, ranking and pruning implementation | `memory_enhancements.py` / `memory_config.yaml` plus related legacy notebook helpers; not a globally enforced rule for every runtime memory write |
| Adaptive retrieval | Agent/phase-specific token-aware retrieval implemented in separate modules | `adaptive_retrieval.py` / `adaptive_integration.py` are **not wired into the supported notebook**; the feature flag alone has no notebook effect |
| Model support | OpenAI Responses API model configurations; optional local OpenAI-compatible model path | Model-specific adapters exist; universal provider compatibility not certified |
| Engineering correctness | Canonical imports, SHA-256 manifest, no-key regression suites, anti-drift tests | CP4 merged; hosted head CI reviewed |
| Full production quality | **Not yet established for the current post-CP4 revision** | [#164](https://github.com/dhar174/intelligent_data_detective/issues/164), [#165](https://github.com/dhar174/intelligent_data_detective/issues/165), [#167](https://github.com/dhar174/intelligent_data_detective/issues/167) |

### Status terminology

- **Implemented:** code or a configured workflow exists; it may still require dependencies, API credentials, a supported execution environment, or further integration tests.
- **Validated (no-key):** checked by automated offline, mocked, structural, or actual-kernel preflight tests without paid model inference.
- **Historically proven:** demonstrated on an earlier notebook/revision but **not** a pass claim for current `main`.
- **Opt-in / experimental:** implemented behind a switch or an alternate configuration; broader compatibility is not guaranteed.
- **Standalone / not runtime-integrated:** a module or demonstration exists, but the supported patched notebook does not invoke it automatically.
- **Planned / unproven:** an issue, design or future direction, not a current shipped product capability.

## 2. Multi-agent runtime and graph orchestration

The executable workflow uses the canonical graph factory in `idd_graph.py`. The source notebook is processed by `_patch_notebook.py` to create the 99-cell runnable `IntelligentDataDetective_beta_v5_patched.ipynb`; the generated notebook imports canonical models, registry, state, tools and graph logic instead of maintaining duplicate definitions.

### All 15 production graph nodes

| Node | Purpose |
| --- | --- |
| `supervisor` | Plans, assesses progress, selects the next agent and provides task instructions |
| `initial_analysis` | Profiles the dataset, its schema and samples; establishes initial context |
| `data_cleaner` | Cleans/transforms datasets and records the operations performed |
| `analyst` | Performs exploratory/statistical analysis and produces structured findings |
| `visualization` | Produces visualization specifications and dispatches chart tasks |
| `viz_worker` | Creates individual chart artifacts from a specification |
| `viz_join` | Collects parallel visualization worker outputs |
| `viz_evaluator` | Reviews figures and produces accept/revise feedback |
| `report_orchestrator` | Plans report sections, goals and required evidence |
| `report_section_worker` | Produces a report section using available analytical context |
| `report_join` | Synchronizes completed report sections |
| `report_packager` | Assembles sections and report-format deliverables |
| `file_writer` | Persists final content and records file metadata |
| `EMERGENCY_MSG` | Handles exceptional communication and rerouting |
| `FINISH` | Applies terminal reporting/writing logic and ends or redirects execution |

Key orchestration capabilities:

- **Supervisor-worker delegation:** structured routing decisions in a validated `Router` object specify the destination, next-agent prompt and optional metadata.
- **State-driven conditional routing:** agents and graph barriers are selected according to existing state and completion evidence.
- **Parallel map/join for visualization:** LangGraph `Send` distributes visualization specifications across `viz_worker` branches, with reducer-aware fan-in through `viz_join`.
- **Parallel report-section production:** report outlines generate `Send` calls for `report_section_worker`; `report_join` precedes packaging.
- **Feedback and bounded retry paths:** `VizFeedback` records acceptance/revision requests; graph routing includes a retry counter and accepts partial outputs in defined circumstances. This is **not** proof of semantic image quality.
- **Report completion gating:** `route_to_writer` and `FINISH` inspect report flags, expected/written sections, draft content and writer status instead of treating every agent response as completion.
- **Emergency handoffs:** typed agent messages and `EMERGENCY_MSG` support exceptional routing.
- **Checkpointing and cache policies:** compiled graphs accept a checkpointer, store and cache. CP4 preserves the notebook's node cache policy (`CachePolicy(ttl=120)`). Durable resumption depends on the configured checkpointer.
- **Structural vs. executable graph construction:** `build_graph(structural_only=True)` permits non-executing topology inspection; executable mode rejects missing production worker callables.
- **Topology constraints:** 15 nodes, 33 edges, 70-field state, no unreachable nodes or dead ends in the canonical validation baseline.

### Supervisor planning and inter-agent communication

The supervisor separates logical functions for broad planning, route selection, replies, new plans, replanning, progress accounting, to-do generation and conversational interaction. Contracts include `Plan`, `PlanStep`, `CompletedStepsAndTasks`, `ProgressReport`, `ToDoList`, `NextAgentMetadata`, `SendAgentMessage` and `MessagesToAgentsList`.

Features include plan version/lifecycle handling, step ordering and duplicate-number protection, progress tracking, task completion bookkeeping, explicit metadata handoff, and a structured state update for agent results. The **70-field `State`** uses purpose-specific reducers for messages, lists, counters, report artifacts, paths and completion flags. It deliberately omits a supervisor-level `structured_response` channel to avoid known LangGraph state-channel collisions.

## 3. Data ingestion, dataset registration and file formats

### Multi-dataset handling

- Register DataFrames under stable `df_id` keys; pass identifiers rather than copying DataFrame payloads between agents.
- Inspect schemas, column names, samples and descriptive statistics.
- Load multiple **CSV and JSON** files through the dedicated `load_multiple_files` tool.
- Query a registered DataFrame with selected columns, optional filtering and aggregate operations.
- Merge registered DataFrames with inner, outer, left, right or cross join semantics and configurable join keys.
- Export processed DataFrames to **CSV, Excel, JSON or Parquet**, with configurable serialization and overwrite/versioning options.
- Persist/reload registry data as **CSV, JSON, Parquet or trusted Pickle** snapshots. The dedicated multi-file import tool and the lower-level registry do **not** have identical accepted-format lists.
- Retrieve, remove, enumerate or clear registrations and inspect original-source vs. owned-backing paths.

### Canonical DataFrame registry

`idd_registry.py` implements:

- A thread-safe, configurable **LRU cache** (default capacity: 20).
- Disk-backed lazy reload after cache eviction, using the recorded file format rather than assuming CSV.
- Atomic owned snapshot writes with exclusive temporary creation and replacement; POSIX file mode restrictions where supported.
- Separate source-file paths and registry-owned reload snapshots.
- Path containment, filename normalization, and an explicit trust decision for untrusted external Pickle files.
- A process-default registry plus `ContextVar`-based `override_global_registry(...)` to isolate concurrent or test-scoped accesses.
- Dynamic `get_global_registry()` / `get_global_df_registry()` resolution while preserving canonical notebook object identity.
- Explicit `clear()` semantics that clear registration state without deleting underlying files.

These controls improve reproducibility and isolation; they are not substitutes for user authentication or a hardened multi-tenant execution sandbox.

## 4. Data cleaning and transformations

Available operations include:

- Missing-value counts and summaries.
- Numeric median imputation.
- Duplicate inspection and removal with subset keys, first/last retention, case/whitespace normalization, sampled duplicate previews and dry-run mode.
- Row deletion using conditions; removal of unwanted columns.
- Data-type coercion for numeric, string, Boolean, categorical and datetime values, including nullable types, per-column date formats, locale separators and downcasting.
- Column-name standardization (including snake_case).
- Label encoding and one-hot encoding for categorical features.
- General dataset quality assessment.
- Structured cleaning reports (`CleaningMetadata`) that describe completed transformations and the resulting dataset.
- Re-export or continued analysis of a cleaned/derived DataFrame.

Agents may propose and execute cleaning actions. As with any automated data transformation, users should review the decisions and retained/removed rows when the dataset is important.

## 5. Analysis, statistics and basic machine learning

### Exploratory and inferential analyses

- Descriptive statistics: counts, means, standard deviations, extrema, quartiles and distribution summaries.
- Pearson pairwise correlation and full correlation matrices.
- One-sample Student t-tests.
- Shapiro-Wilk normality tests.
- IQR-based outlier detection.
- Missingness, duplicates, type checks and dataset profiling.
- LLM-assisted pattern/anomaly interpretation and suggested next investigations.
- Optional Tavily web search for supporting external domain context (when configured).

### ML tool

`train_ml_model` supports **linear regression** (RMSE scoring) and **logistic regression** (accuracy scoring), configurable feature/target columns, train/test split, random seed, missing-row handling and optional Joblib serialization. Separate categorical encoding tools can prepare features.

**Boundary:** IDD does **not** yet implement a general AutoML search system, automated hyperparameter tuning, deployed inference endpoints or a validated forecasting product.

## 6. Visualization engine

Five dedicated plotting tools are exposed in the v5 source, in addition to Python-based custom plotting:

| Tool | Visual purpose | Selected controls |
| --- | --- | --- |
| `create_histogram` | Univariate and multi-series distribution views | Bins, bin width/range, normalization, weights, overlays, hue, sampling |
| `create_scatter_plot` | Numeric relationships | One/multiple Y variables, hue, point size/style, sampling and filters |
| `create_correlation_heatmap` | Correlation matrix display | Pearson/Spearman/Kendall, masks, annotations, order, clustering |
| `create_box_plot` | Distribution and outlier comparisons | Category grouping, hue, multiple values, whiskers, overlays |
| `create_violin_plot` | Distribution-density comparisons | Grouping, split plots, inner summaries, multiple values |

Further features:

- Typed `VizSpec` objects carry chart IDs, titles, types, dataset references, X/Y/hue, filters, styling and instructions.
- Independent worker tasks execute in parallel; `viz_join` consolidates the results.
- `VizFeedback` holds an accept/revise decision, comments and a redo list.
- `list_visualizations` / `get_visualization` discover and retrieve figures by ID, path, title or description, with optional Base64 preview.
- Figures can be referenced in HTML, Markdown and PDF reports with descriptions and metadata.
- Plotters use Matplotlib/Seaborn and selected backend fallbacks.

**Boundary:** generated PNG figures and Base64 previews are not the same as a fully interactive browser dashboard; the latter remains roadmap work. An automated quality verdict does not yet establish chart-to-source numerical fidelity; see [#167](https://github.com/dhar174/intelligent_data_detective/issues/167).

## 7. Complete v5 analytical tool catalog (40)

The v5 notebook tool ecosystem defines **40 decorated tools**. These are grouped by function below; the runtime grants a selected subset to each agent. Additional memory tools are built dynamically and are **not** included in the number 40.

| Category (count) | Tools |
| --- | --- |
| **Discovery / querying (5)** | `get_dataframe_schema`, `get_column_names`, `query_dataframe`, `get_descriptive_statistics`, `assess_data_quality` |
| **Cleaning / transformation (8)** | `check_missing_values`, `drop_column`, `delete_rows`, `fill_missing_median`, `detect_and_remove_duplicates`, `convert_data_types`, `standardize_column_names`, `handle_categorical_encoding` |
| **Statistics / ML (6)** | `calculate_correlation`, `calculate_correlation_matrix`, `perform_hypothesis_test`, `perform_normality_test`, `detect_outliers`, `train_ml_model` |
| **Visualization (7)** | `create_histogram`, `create_scatter_plot`, `create_correlation_heatmap`, `create_box_plot`, `create_violin_plot`, `list_visualizations`, `get_visualization` |
| **Data / file operations (8)** | `load_multiple_files`, `merge_dataframes`, `export_dataframe`, `list_available_files`, `read_file`, `write_file`, `edit_file`, `create_sample` |
| **Code / external research (2)** | `python_repl_tool`, `search_web_for_context` |
| **Reporting (3)** | `generate_html_report`, `format_markdown_report`, `create_pdf_report` |
| **Progress (1)** | `report_intermediate_progress` |
| **Total** | **40** |

Separate tool/middleware helpers include `create_manage_memory_tool`, `create_search_memory_tool`, `cap_output`, `@handle_tool_errors`, `validate_dataframe_exists` and `_resolve_artifact_path`.

### Tool safety, execution, and response controls

- Canonical `idd_tools.handle_tool_errors` performs **signature-aware argument binding**, rather than treating any first-position string as a DataFrame ID.
- Structured tool failures follow the **`status`, `operation`, `reason`, `action`** schema and sanitize unexpected exception details.
- Dataset-touching functions validate IDs and use the active context-local registry.
- Artifact writes are resolved against allowed roots; traversal is rejected **before** directory creation.
- Tool outputs can be capped by characters, bytes, lines and preview limits to avoid flooding model context.
- `python_repl_tool` supports computed analysis with access to the registry and guarded filesystem operations. **Do not treat this Python execution facility as a hostile-code security sandbox.**
- Visualization lookup prioritizes graph state and falls back to artifact discovery.

## 8. Memory, retrieval, and long-context capabilities

### Memory actually wired into the supported notebook

The patched v5 notebook configures a LangGraph `InMemoryStore` (with an embedding index where available) and adds LangMem `create_manage_memory_tool` / `create_search_memory_tool` to specialist tool lists. Agent-specific namespaces include `("memories", "cleaning")`, `("memories", "initial_description")`, `("memories", "analysis")`, `("memories", "reports")`, `("memories", "files")`, and `("memories", "visualization")`.

The notebook also retains **legacy/inlined** `MemoryPolicyEngine`, `retrieve_memories`, `put_memory`, `enhanced_retrieve_mem`, `enhanced_mem_text` and `update_memory_with_kind` helper code. Some agent-node prompts and supervisor operations call these helpers, and their policy path can consult `memory_config.yaml` for supported kinds. However, **the notebook does not import `memory_enhancements.py`**, and the direct LangMem manage/search tools are **not universally routed through these helpers**. Therefore, notebook-wide TTL expiration, ranking weights, pruning and six-kind policy compliance **must not be claimed as uniformly enforced runtime features**. The legacy/inlined behavior has not been freshly certified end-to-end on the post-CP4 revision.

### Standalone memory categorization and lifecycle engine

The separately maintained `memory_enhancements.py` module defines an explicit six-kind policy layer:

| Kind | Intended content | Configured policy TTL (not a notebook-wide guarantee) |
| --- | --- | --- |
| `conversation` | User prompts and exchanges | 3 days |
| `analysis` | Statistics and analysis results | 14 days |
| `cleaning` | Preprocessing actions and decisions | 14 days |
| `visualization` | Figure-related context | 7 days |
| `insights` | Consolidated findings | 21 days |
| `errors` | Failure modes and troubleshooting records | 7 days |

`memory_config.yaml` configures policy-engine behavior, including:

- Namespace-aware writes and filtered retrieval.
- Importance/relevance scoring, recency decay and usage weighting.
- TTL-based, size-based and importance-based pruning.
- Near-duplicate/superseded memory handling.
- Per-kind and global retention limits.
- Usage counters, importance recalculation and diagnostic reports.
- Compatibility fallback to older generic/user namespaces.
- Embedding-based lookup when an indexed store is configured.

The policy's default ranking weights are **similarity 0.55, importance 0.25, recency 0.15, usage 0.05**. These are values of the policy implementation, **not universal defaults for every direct LangMem tool call**.

### Adaptive context retrieval (standalone; not notebook-integrated)

`adaptive_retrieval.py` and `adaptive_integration.py` implement:

- Per-agent retrieval profiles, memory categories and target item counts.
- Exploration/analysis/reporting phase overrides.
- Query-intent adjustments for analytical, cleaning and visualization requests.
- Candidate expansion, multi-factor ranking and token-aware packing.
- Soft token budgets and context-window-based hard limits.
- Heuristic token estimation, diagnostic metrics and legacy fallback.
- Specialized wrappers for initial analysis, data cleaning, analysis, visualization, reporting, file writing and supervision.

**Integration boundary:** neither the supported patched notebook nor `_patch_notebook.py` invokes/imports these modules. Setting `ADAPTIVE_RETRIEVAL_ENABLED=true` **does not activate adaptive retrieval in the supported notebook**; it only controls code paths that explicitly call the standalone integration module. Connecting it to the production agent nodes and proving actual token-budget behavior requires separate implementation and verification.

### Embeddings and history controls

The notebook includes cloud embedding configuration and an optional Hugging Face local embedding path, helpers for E5-style document/query prefixes, LangGraph indexed in-memory stores, conversation summarization and protected-message/token-budget trimming.

**Durability qualification:** an `InMemoryStore` is not a durable, cross-process knowledge service. Long-term retention requires an appropriately configured storage backend and validation; full enterprise RAG and persistent cross-deployment memory remain future work.

## 9. Model configuration, reasoning, and structured contracts

- A role-aware **multi-model supervisor** separates big-picture evaluation, route selection, replies, planning, replanning, progress, to-dos and conversational functions.
- The checked-in default v5 notebook configuration uses **GPT-5 mini/nano model roles** through OpenAI's Responses API (subject to model/API availability).
- Reasoning-effort and verbosity are configured by task, instead of assuming every agent needs the same budget.
- `MyChatOpenai` provides compatibility handling for Responses API payloads, structured outputs, deprecated parameters and some model-specific request rules.
- Optional `use_local_llm` paths include OpenAI-compatible custom endpoints, Hugging Face local embeddings and Qwen3 message/tool-response middleware.
- System prompts, task prompts and tool descriptions are tailored to worker roles.
- Pydantic output schemas use `BaseNoExtrasModel` with strict extra-field handling and worker-to-supervisor progress fields.
- Structured contracts include `InitialDescription`, `CleaningMetadata`, `AnalysisInsights`, `VizSpec`, `DataVisualization`, `VisualizationResults`, `VizFeedback`, `ReportOutline`, `Section`, `ReportResults`, `FileResult` and `Router`.
- Configurable message-history summarization and tool-response trimming control long contexts.

**Boundaries:** this architecture does not demonstrate a provider-agnostic compatibility test matrix, a rigorous explanation-of-reasoning guarantee, or exposure of a model's private internal reasoning. IDD provides observable plans, decisions, tool calls, state and artifacts.

## 10. Reporting and artifact delivery

- The report orchestrator defines an outline with titles, goals, required signals, figures and section word targets.
- Multiple report-section workers can compose sections in parallel.
- Report join and packager stages combine written sections, figures and analytical context.
- File-writing tools and post-processing prepare **HTML, Markdown and PDF**.
- Figures can be embedded or referenced in generated documents.
- The system tracks result paths in `ReportResults` (`html_report_path`, `markdown_report_path`, `pdf_report_path`) and file-level metadata in `FileResult`.
- Output handling includes run-specific directories, scoped artifact paths, image materialization and file versioning in selected tools.
- A headless runner supports fresh or checkpoint-resumed notebook execution and collects logs/artifact summaries.

A generated report's existence is **not** treated as sufficient proof of quality.

## 11. Verification, provenance, and failure detection

### Canonical-core engineering (CP1–CP4 merged)

The canonical implementation is split across these modules:

| File | Authority |
| --- | --- |
| `idd_models.py` | Pydantic base, plans and completed-step lifecycle |
| `idd_registry.py` | Registry, caching, isolation and owned data snapshots |
| `idd_state.py` | 70-field state, reducers and typed domain results |
| `idd_tools.py` | Tool error contracts, dataset validation and artifact paths |
| `idd_graph.py` | Graph builder, routes, barriers and completion rules |
| `idd_core.py` | Backward-compatible re-export façade |
| `idd_bootstrap.py` | Fail-closed canonical module startup verification |
| `idd_manifest.json` | SHA-256 digests for seven canonical Python modules |
| `_patch_notebook.py` | Authoritative source for regenerated 99-cell runnable notebook |

CP4's anti-drift tests check static imports, protected symbol shadowing, runtime object identity, behavioral tool parity, actual generated-cell execution in controlled modes, malicious/mismatched module origins, changed patch anchors and mutated manifests. Canonical notebook regeneration is **not** done by hand-editing the patched notebook.

The manifest establishes byte consistency against known expectations. A co-distributed manifest **alone** does not independently authenticate the entire distribution. A standalone Colab trust anchor is undecided under [#166](https://github.com/dhar174/intelligent_data_detective/issues/166).

### Validation and diagnostic tools

| Utility | Function |
| --- | --- |
| `validate_notebook_integrity.py` | JSON/cell-count checks, actual code-object compilation and controlled magic handling |
| `validate_graph.py` | No-key structural graph validation and negative topology probes |
| `tests/unit/test_core_parity_guard.py` | Canonical identity, notebook anti-drift, bootstrap and negative-control tests |
| `tools/diagnostics/reproduce_drift.py` | Reproducible checks for known notebook/core drift findings |
| `run_notebook_live.py` | Headless Jupyter execution, optional resume, logs and artifacts |
| `tests/smoke/test_scientific_kernel.py` | Opt-in selected-kernel scientific import probe and `IDD_SKIP_NOTEBOOK_INSTALLS` environment-variable inheritance check; **does not install packages or execute notebook cells** |
| `validate_run.py` | Twelve-point execution + artifact completeness gate |
| `validate_artifact_quality.py` | Nine-point document usability/artifact gate |
| `validate_report_quality.py` | Rejects scaffold prose, inappropriate dumps and missing report-agent evidence |
| `pytest` suites / GitHub Actions | No-key unit/integration/regression checks and linting |

**Preflight boundary:** `run_notebook_live.py --preflight` and the kernel smoke test check scientific imports and the install-flag value in a child Jupyter kernel only. They do **not** install dependencies, construct or execute the notebook's graph, call an LLM, or prove report output. Actual effective-notebook, cross-platform no-key execution is the separate acceptance work in [#164](https://github.com/dhar174/intelligent_data_detective/issues/164).

### 12-point production acceptance

`validate_run.py` checks:

1. A final completion marker.
2. Visualization/report completion flags.
3. No problematic recovery/final-hop warning markers.
4. No tracebacks.
5. Native structured agent outputs.
6. A PDF in the run-specific report directory.
7. HTML report text of at least 3,000 characters.
8. At least four distinct report sections.
9. At least three genuinely distinct chart files.
10. A PDF of at least 30 KB.
11. No placeholder/stub-marker report files.
12. Report text incorporating actual correlation/anomaly findings.

### 9-point artifact usability checks

`validate_artifact_quality.py` checks:

1. PDF parseability and extracted text.
2. Resolvable HTML figure links.
3. Resolvable Markdown figure links.
4. Resolvable figure links in root HTML.
5. Low duplicate-paragraph rate.
6. Heading cleanliness.
7. Stakeholder-readable narrative rather than scaffolding or raw dumps.
8. Absence of stray text artifacts.
9. Non-trivial chart selection.

`validate_report_quality.py` independently examines report-authoring authenticity and stakeholder prose.

**These gates measure execution, content presence and usability, not the factual correctness of every chart or natural-language claim.** Objective analytical-fidelity benchmarks are planned in [#167](https://github.com/dhar174/intelligent_data_detective/issues/167).

### Exact review evidence and historical boundary

For the final CP4 review of PR #163, the hosted Actions job on reviewed head `21564b625ca41f5e2ebdb8fa4411c8c29bf3c623` reported:

- **695 passed / 3 skipped** in unit and integration tests.
- **3 passed** scientific-kernel preflight tests.
- **110 passed** additional root/memory/notebook tests.
- **17 passed** prompt regressions.
- **16/16 passed** mandatory error-framework tests.
- All 19 inline CP4 review threads resolved.

See [Actions run #39465912223](https://github.com/dhar174/intelligent_data_detective/actions/runs/39465912223) and [PR #163](https://github.com/dhar174/intelligent_data_detective/pull/163). Test suites above overlap and must not be added as a unique combined test total.

A **May 4, 2026 historical W14** notebook run met **12/12** and **9/9** on a retail-order fixture; a **September 29, 2026** run failed with **5/12**, including missing report content. Neither is a fresh post-CP4 proof. Passing structural/no-key CI does **not** establish current-head production or analytical correctness.

## 12. AI-assisted development tooling

IDD also includes a distinct *repository-development* agent stack, separate from its runtime data-analysis agents.

### Nine repository development-agent configurations

- `idd-repo-coordinator`: work coordination and one-writer scope.
- `notebook-patch-specialist`: notebook generator/cell integrity.
- `langgraph-topology-architect`: graph topology and reducers.
- `pydantic-contract-guardian`: typed output contracts.
- `pipeline-proof-validator`: execution and artifact evidence.
- `idd-test-evaluator`: no-key regression evaluation.
- `data-viz-artifact-engineer`: plotting and document artifacts.
- `memory-scout`: historical context recovery.
- `memory-steward`: maintained technical knowledge.

### Nineteen bundled development skills

The `.agents/skills/` directory includes `accesslint-audit`, `agent-evaluation`, `agent-memory-systems`, `agent-tool-builder`, `agents-generator`, `agenttrace-session-audit`, `andrej-karpathy`, `debugging-and-error-recovery`, `error-detective`, `langgraph`, `llm-structured-output`, `matplotlib`, `multi-agent-architect`, `pdf-official`, `python-pro`, `python-testing-patterns`, `seaborn`, `tavily-web` and `tree-ring-memory`.

Repository-specific instructions are distributed across `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.github/`, `.agents/` and `memory-bank/`. These support software-development workflows; they are **not automatically user-facing IDD runtime features**.

## 13. Distinctive engineering innovations

The most compelling project-level contributions are combinations of already established technologies:

1. **Constrained autonomy:** LLM-guided supervisor decisions within a deterministic, type-checked LangGraph topology.
2. **Parallel but synchronized analytical production:** concurrent figure and report writers with explicit fan-in and reducer behavior.
3. **Memory-aware agent specialization:** runtime agents receive scoped LangMem tools and selected notebook-local retrieval helpers; standalone lifecycle-policy and adaptive context-selection modules provide additional design work **not yet comprehensively integrated** into the supported notebook.
4. **Canonical data ownership:** registry-held DataFrames, context-local overrides, multi-format backing and controlled tool access.
5. **Typed handoffs and progress:** plans, tasks, structured insights and validation-rich agent-to-agent state transitions.
6. **Anti-Potemkin validation:** detect a completed workflow that produced hollow text, duplicate visuals or placeholder artifacts.
7. **Executable notebook/runtime parity:** module-level single source of truth, byte-integrity verification and automated tests that detect notebook shadowing.
8. **Model-role specialization:** configure smaller/faster vs. more analytical model roles according to the task.

These are IDD architectural differentiators, **not** claims that any individual technique is unique or patented.

## 14. Incomplete, optional and proposed features

| Feature | Accurate current status |
| --- | --- |
| Live end-to-end success on current merged head | **Pending** [#165](https://github.com/dhar174/intelligent_data_detective/issues/165) |
| Linux/Windows/cloned-Colab real-kernel compatibility matrix | **Pending** [#164](https://github.com/dhar174/intelligent_data_detective/issues/164) |
| Ground-truth numerically faithful analysis/reports | **Unproven; benchmark proposed** [#167](https://github.com/dhar174/intelligent_data_detective/issues/167) |
| Standalone Colab with immutable external trust anchor | **Decision pending** [#166](https://github.com/dhar174/intelligent_data_detective/issues/166) |
| Adaptive memory retrieval | **Standalone implementation only**; feature flag alone has no effect on the supported notebook, and runtime wiring/proof remain open |
| Notebook-wide memory TTL/ranking/pruning | **Not established**; some legacy notebook helpers have policy logic, but direct LangMem tools do not uniformly enforce it |
| Local OpenAI-compatible LLM endpoint | **Optional/experimental; cross-provider proof missing** |
| Durable cross-process memory/RAG service | **Not productionized**; in-memory/embedding foundations exist |
| Interactive dashboard and chart drill-down | **Roadmap** |
| Full AutoML, advanced prediction and forecasting | **Roadmap**; only basic linear/logistic regression currently |
| Real-time ingestion and recurring scheduled analyses | **Roadmap** |
| Multi-user auth, access control, masking and enterprise isolation | **Roadmap** |
| User-defined specialized agents/plugin builder | **Roadmap** |
| Enterprise monitoring/dashboard and deployment packaging | **Roadmap** |

### Recommended next evidence gates

1. [#164](https://github.com/dhar174/intelligent_data_detective/issues/164): execute effective generated notebook code in real kernels without keys, recording an explicit PASS/FAIL/NOT EXECUTED matrix.
2. [#165](https://github.com/dhar174/intelligent_data_detective/issues/165): after explicit authorization for paid inference, run **two separate fresh full analyses** on meaningfully different inputs; require **12/12 and 9/9** on *each*, with source SHA, logs and inspected final artifacts.
3. [#167](https://github.com/dhar174/intelligent_data_detective/issues/167): build reference-dataset, numerical-claim, figure-fidelity and unsupported-conclusion tests independent of LLM-generated answers.
4. [#166](https://github.com/dhar174/intelligent_data_detective/issues/166): record the standalone Colab support/trust-anchor decision and document supported distribution modes.

## 15. Maintainer notes / authoritative references

- **Runtime notebook:** [`IntelligentDataDetective_beta_v5_patched.ipynb`](IntelligentDataDetective_beta_v5_patched.ipynb)
- **Canonical topology:** [`idd_graph.py`](idd_graph.py)
- **Shared state:** [`idd_state.py`](idd_state.py)
- **Core models:** [`idd_models.py`](idd_models.py)
- **Dataset management:** [`idd_registry.py`](idd_registry.py)
- **Tool error/path contracts:** [`idd_tools.py`](idd_tools.py)
- **Bootstrap/manifest:** [`idd_bootstrap.py`](idd_bootstrap.py) and [`idd_manifest.json`](idd_manifest.json)
- **Notebook memory:** [source notebook](IntelligentDataDetective_beta_v5.ipynb) and [generated notebook](IntelligentDataDetective_beta_v5_patched.ipynb) (LangMem tool wiring and legacy/inlined helpers)
- **Standalone memory lifecycle module:** [`memory_enhancements.py`](memory_enhancements.py), [`memory_config.yaml`](memory_config.yaml)
- **Standalone adaptive retrieval (not runtime-wired):** [`adaptive_retrieval.py`](adaptive_retrieval.py), [`adaptive_integration.py`](adaptive_integration.py), [`adaptive_memory_config.yaml`](adaptive_memory_config.yaml)
- **Report quality:** [`validate_run.py`](validate_run.py), [`validate_artifact_quality.py`](validate_artifact_quality.py), [`validate_report_quality.py`](validate_report_quality.py)
- **Engineering plan:** [`docs/plans/idd-canonical-core-migration.md`](docs/plans/idd-canonical-core-migration.md)
- **Review/evidence:** [PR #163](https://github.com/dhar174/intelligent_data_detective/pull/163), [#140](https://github.com/dhar174/intelligent_data_detective/issues/140), [#164](https://github.com/dhar174/intelligent_data_detective/issues/164), [#165](https://github.com/dhar174/intelligent_data_detective/issues/165), [#166](https://github.com/dhar174/intelligent_data_detective/issues/166), [#167](https://github.com/dhar174/intelligent_data_detective/issues/167)

**Maintenance rule:** update this document when implementation capabilities change, not just when a feature is proposed. Keep **implemented**, **historically proven**, **currently verified**, **opt-in** and **planned** claims distinct; refresh dated validation evidence rather than carrying it forward as current-head proof.
