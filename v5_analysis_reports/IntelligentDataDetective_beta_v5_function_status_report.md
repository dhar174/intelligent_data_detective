# IntelligentDataDetective_beta_v5.ipynb — Function & Cell Status Report

Comprehensive review of the notebook at `IntelligentDataDetective_beta_v5.ipynb`. All code cells currently have `execution_count=None` (no evidence of execution). PIP installs and shell magics are present but have not been run in this snapshot.

## Notebook Overview
- **Total cells:** 94 (markdown + code).
- **Primary goals:** multi-agent, LangGraph-driven “Intelligent Data Detective” workflow with custom ChatOpenAI client, robust Pydantic data models, dataframe registry, rich toolset (analysis, cleaning, visualization, file ops, ML training), streaming execution, memory integration, and checkpoint migration.
- **Key themes:** environment/bootstrap, extensive prompts, reducer/state wiring, tool libraries, memory + embeddings, model hooks, agent/node definitions, orchestration graph, streaming display utilities, and checkpoint persistence.

## Cell-by-Cell Breakdown
1. **Cell 0 (MD)** – Notebook title: Intelligent Data Detective introduction.
2. **Cell 1 (Code)** – Flag `use_local_llm=False` to switch to local llama.cpp server when desired.
3. **Cell 2 (MD)** – Blank spacer.
4. **Cell 3 (MD)** – Section header: environment setup and dependency management.
5. **Cell 4 (Code)** – Environment bootstrap: imports (`os`, `subprocess`, `PathlibPath`, `contextmanager`, `logging`, etc.), Colab detection, API key retrieval, conditional local LLM installs, and bulk `pip install` for LangChain/LangGraph/tavily/scikit-learn stack; sets `TAVILY_API_KEY`/`OPENAI_API_KEY` if available.
6. **Cell 5 (MD)** – Narrative explaining environment detection, key management, dependency installs, and fallbacks.
7. **Cell 6 (MD)** – “New Section” marker.
8. **Cell 7 (MD)** – Header for core imports/type system.
9. **Cell 8 (Code)** – Core imports and utilities: future annotations, langchain_huggingface embeddings, extensive stdlib/scientific stack, plotting, Tavily/Kaggle, LangGraph/LC components, `ChatOpenAI`, toolkits, working directory helpers (`_is_colab`, `_make_idd_results_dir`, `persist_to_drive`), reducers (`keep_first`, `dict_merge_shallow`), Pydantic setup, vector validation `is_1d_vector`, agent member classes (`AgentMembers` + subclasses), agent ID aliases, and `agent_list_default_generator()`.
10. **Cell 9 (MD)** – Summary of foundational imports and typing focus.
11. **Cell 10 (MD)** – Header for OpenAI API integration.
12. **Cell 11 (Code)** – Custom responses API payload builder `_construct_responses_api_payload()`; subclass `MyChatOpenai` overriding `_get_request_payload_mod/_get_request_payload` for responses API, o-series role remapping, legacy token handling, tool/structured-output normalization.
13. **Cell 12 (MD)** – Explains custom ChatOpenAI enhancements and GPT‑5/o-series readiness.
14. **Cell 13 (MD)** – Header for dependency verification.
15. **Cell 14 (Code)** – `!pip show --verbose langchain_experimental` (version check).
16. **Cell 15 (MD)** – Describes version validation intent.
17. **Cell 16 (MD)** – Header for data models/state architecture.
18. **Cell 17 (Code)** – Extensive Pydantic models: `BaseNoExtrasModel`, `AnalysisConfig`, `CleaningMetadata`, `InitialDescription`, `VizSpec`, `AnalysisInsights`, `ImagePayload`, `DataVisualization`, `VisualizationResults`, `ReportResults`, `DataQueryParams`/`QueryDataframeInput`, `FileResult`, `ListOfFiles`, `DataFrameRegistryError`, `ProgressReport`, plan models (`PlanStep`, `Plan`, `CompletedStepsAndTasks`, `ToDoList`), routing metadata (`NextAgentMetadata`, `SendAgentMessage`, `MessagesToAgentsList`), helpers (`_sort_plan_steps`, `_assert_sorted_completed_no_dups`, `_triplet_from_raw`, etc.).
19. **Cell 18 (MD)** – Notes core data structures and state management purpose.
20. **Cell 19 (MD)** – Header for dataframe registry.
21. **Cell 20 (Code)** – `DataFrameRegistry` class with thread-safe LRU caching, IO helpers, register/load/remove APIs; global registry; section/report data models (`Section`, `SectionOutline`, `ReportOutline`); feedback/response models (`VizFeedback`, `ConversationalResponse`); helper `get_global_df_registry`.
22. **Cell 21 (MD)** – Summarizes registry features (LRU, auto-ID, thread safety, file integration).
23. **Cell 22 (MD)** – Header for state reducers.
24. **Cell 23 (Code)** – Reducers (`merge_lists`, `merge_unique`, `merge_int_sum`, `merge_dicts`, `merge_dict`, `any_true`, `last_wins`, `_reduce_plan_keep_sorted`); main `State` TypedDict defining workflow fields (routing, plan, agents, viz/report orchestration, artifacts, flags); `VizWorkerState`. Default `AnalysisConfig` instance and `CLASS_TO_AGENT` mapping.
25. **Cell 24 (MD)** – Describes reducer/state merge behavior.
26. **Cell 25 (MD)** – Header for prompt templates.
27. **Cell 26 (Code)** – Extensive prompt templates and policies: shared tooling guidelines, data cleaner prompts (full + mini), initial analyst prompts (full + mini), main analyst prompts (full + mini), file writer prompts, visualization prompts, report orchestrator/section worker prompts, visualization evaluator, supervisor/system instructions, and supporting constants/placeholders.
28. **Cell 27 (MD)** – Summary of comprehensive agent prompts and guidance.
29. **Cell 28 (MD)** – Header for agent construction and orchestration intro.
30. **Cell 29 (Code)** – Imports/aliases for LangGraph compilation, ChatOpenAI instantiation shortcuts, human system prompts, and graph wiring placeholders (no new defs).
31. **Cell 30 (MD)** – Describes routing/graph orchestration scope.
32. **Cell 31 (MD)** – Header for runtime configuration.
33. **Cell 32 (Code)** – Output capping utilities: `_is_df`, `_to_jsonable`, `_pretty`, decorator `cap_output` to truncate tool output safely with logging hooks.
34. **Cell 33 (Code)** – Core toolkit definitions:
    - Error-handling decorator `handle_tool_errors` and validator `validate_dataframe_exists`.
    - Dataframe inspection/manipulation tools (`get_dataframe_schema`, `get_column_names`, `check_missing_values`, `drop_column`, `delete_rows`, `fill_missing_median`, `query_dataframe`, `get_descriptive_statistics`, `calculate_correlation`, `perform_hypothesis_test`, etc.).
    - File utilities (`create_sample`, `read_file`, `write_file`, `edit_file`, `sandbox_filesystem`, `python_repl_tool`), histogram/scatter/heatmap/box/violin plotting helpers, export/merge/cleaning/conversion helpers, reporting (markdown/PDF), ML training (`train_ml_model`), encoding helpers, visualization listing (`list_visualizations`, `get_visualization`), and progress logging tool `report_intermediate_progress`. Includes helper classes `NewPythonInputs`, `NpEncoder`.
    - Tool collections (e.g., `data_cleaning_tools`, `analyst_tools`, `visualization_tools`, `report_generator_tools`, `file_writer_tools`).
35. **Cell 34 (Code)** – Partial/placeholder start of an additional tool (truncated call_file_w… comment).
36. **Cell 35 (MD)** – Narrative of complete toolkit coverage (analysis, cleaning, visualization, file management, REPL, web search, error handling).
37. **Cell 36 (MD)** – Blank spacer.
38. **Cell 37 (Code)** – Memory and embedding utilities: identity helpers, e5 embedding wrappers, memory policy models (`MemoryRecord`, `MemoryPolicy`, `RankingWeights`, `PruneReport`, `MemoryPolicyEngine`), importance/similarity scoring, memory CRUD (`put_memory`, `retrieve_memories`, `enhanced_retrieve_mem`, `put_memory_with_policy`, `prune_memories`, `update_memory_with_kind`, `store_categorized_memory`), metrics tracking/reset.
39. **Cell 38 (MD)** – Memory system summary and policy-driven storage.
40. **Cell 39 (MD)** – Header for model hook utilities.
41. **Cell 40 (Code)** – Tool-call extraction/formatting helpers for Qwen/OpenAI workflows (`_safe_json_loads`, `extract_tool_calls`, `retry_extract_tool_calls`, `format_tool_responses_for_qwen3`, `qwen3_pre_model_hook`, `qwen3_post_model_hook`, `is_final_answer`, `extract_final_text`, formatting utilities for model docs and strict-final wrappers).
42. **Cell 41 (Code)** – Imports/setup for model docs and hook chaining (supporting Cell 40 functions).
43. **Cell 42 (Code)** – `_dedupe_tools` helper to merge tool specs.
44. **Cell 43 (Code)** – Summarization/collapse utilities for conversation history (`_msg_has_tool_invocation`, `_extract_message_id`, `_new_id`, `_rebuild_plain_message`, `strip_tools_for_summary_hardened`, token counting, protection windows, snapshot collapsing, `make_pre_model_hook`).
45. **Cell 44 (Code)** – Agent factory functions (`create_data_cleaner_agent`, `create_initial_analysis_agent`, `create_analyst_agent`, `create_file_writer_agent`, `create_visualization_agent`, `create_viz_evaluator_agent`, `create_report_generator_agent`), memory updater, and supervisor node builder.
46. **Cell 45 (MD)** – Describes agent creation/supervisor setup.
47. **Cell 46 (Code)** – Graph wiring setup for state graph (imports LangGraph builder, checkpoint store placeholders) without new defs.
48. **Cell 47 (MD)** – Notes transition into state graph assembly.
49. **Cell 48 (MD)** – Header for runtime context and directories.
50. **Cell 49 (Code)** – `RuntimeCtx` class managing directories (`artifacts_dir`, `reports_dir`, etc.), helper `_touch_dir`, `make_runtime_ctx` call to instantiate `RUNTIME`.
51. **Cell 50 (MD)** – Explains runtime context roles.
52. **Cell 51 (MD)** – Header for artifact handling.
53. **Cell 52 (Code)** – File persistence helpers: hashing/encoding detection (`_sha256_bytes`, `_detect_mime_and_encoding`), atomic write helpers, visualization materialization (`save_viz_for_state`), artifact manifest builder (`_manifest_from_path`), versioned path generator (`_next_version_path`).
54. **Cell 53 (MD)** – Notes artifact persistence/manifest utilities.
55. **Cell 54 (MD)** – Header for graph node definitions.
56. **Cell 55 (Code)** – Core LangGraph nodes and routing logic: `initial_analysis_node`, `data_cleaner_node`, `analyst_node`, file writer helpers (`_normalize_meta`, `file_writer_node`), visualization orchestrator/worker/evaluator (`_guess_viz_type`, `_normalize_viz_spec`, `visualization_orchestrator`, `viz_worker`, `assign_viz_workers`, `viz_join`, `viz_evaluator_node`, `route_viz`), reporting orchestrator/section workflow (`report_orchestrator`, `section_worker`, `dispatch_sections`, `assign_section_workers`, `report_join`, `report_packager_node`), emergency routing `emergency_correspondence_node`.
57. **Cell 56 (MD)** – Describes node behaviors and orchestration responsibilities.
58. **Cell 57 (MD)** – Header for routing functions.
59. **Cell 58 (Code)** – Router helpers: `write_output_to_file`, `route_to_writer`, `route_from_supervisor` for dynamic edge selection.
60. **Cell 59 (MD)** – Summarizes routing strategies and triggers.
61. **Cell 60 (MD)** – Header for graph compilation.
62. **Cell 61 (Code)** – Graph assembly: builds `StateGraph`, adds nodes/edges, conditional visualization/reporting branches, memory saver, global `data_detective_graph`, `graph_cfg` presets, `initial_state`.
63. **Cell 62 (MD)** – Notes full graph construction and compiled state graph availability.
64. **Cell 63 (MD)** – Header for debugging/inspection utilities.
65. **Cell 64 (Code)** – Debug helpers for graph visualization and state snapshots (uses `plot_graph` placeholders, printing structure).
66. **Cell 65 (MD)** – Describes debugging graph layout.
67. **Cell 66 (MD)** – Header for state/key-path utilities.
68. **Cell 67 (Code)** – Path search/introspection helpers (`find_key_paths`, `find_key_paths_list`, `get_by_path`, `pick_tool_messages`, `extract_handles_from_tools`) for nested structures and tool metadata.
69. **Cell 68 (MD)** – Summarizes debugging utilities for nested data.
70. **Cell 69 (MD)** – Header for streaming workflow execution.
71. **Cell 70 (Code)** – Streaming run setup: imports, `received_steps`, `thread_id`/`user_id` config, `RunnableConfig`, callback placeholders for printing streamed events (truncated).
72. **Cell 71 (MD)** – Describes streaming engine, progress monitoring, and error handling intent.
73. **Cell 72 (Code)** – Blank code cell (no content).
74. **Cell 73 (MD)** – Header for extended streaming utilities.
75. **Cell 74 (Code)** – Jupyter display CSS tweaks for wrapped output; helper imports; text extraction utilities for OpenAI content blocks (`iter_allowed_text_blocks`, `content_to_text`, `strip_tool_calls`, `pretty_print_wrapped`, etc.).
76. **Cell 75 (Code)** – Sample accumulation loop over `received_steps` to collect streaming messages with metadata.
77. **Cell 76 (MD)** – Summarizes streaming utilities and content extraction.
78. **Cell 77 (Code)** – Drive persistence: fetches final state from graph, persists working directory via `persist_to_drive`, ensures final report packaging.
79. **Cell 78 (MD)** – Header for final state inspection.
80. **Cell 79 (Code)** – Final state reporter: lists figures/reports, prints key flags from checkpointer, iterates tool messages and plan/viz/report summaries.
81. **Cell 80 (MD)** – Narrative on comprehensive inspection of workflow outputs.
82. **Cell 81 (Code)** – Prints working directory path.
83. **Cell 82 (MD)** – Header for function-calling utilities.
84. **Cell 83 (Code)** – Imports `convert_to_openai_tool` from LC function-calling utilities.
85. **Cell 84 (MD)** – Notes tool conversion/schema validation purpose.
86. **Cell 85 (MD)** – Header for advanced testing/validation narrative.
87. **Cell 86 (MD)** – Header for final model validation/QA.
88. **Cell 87 (Code)** – Commented-out Pydantic validation test snippets for `InitialDescription`.
89. **Cell 88 (MD)** – Describes final validation and QA intent.
90. **Cell 89 (MD)** – Header for checkpoint migration.
91. **Cell 90 (Code)** – SQLite checkpointer migration: uses `SqliteSaver` to migrate histories from in-memory graph to SQLite, defines `migrate_thread` helper.
92. **Cell 91 (MD)** – Header for restoring checkpoint.
93. **Cell 92 (Code)** – Restores graph with `SqliteSaver` from `checkpoints.sqlite`, compiling `data_detective_graph` with checkpoint/cache.
94. **Cell 93 (Code)** – Prints `run_config` to show current configuration snapshot.

## Status Notes
- **Execution:** All `execution_count` fields are `null`; cells contain definitions and shell magics but were not executed in this snapshot.
- **Dependencies:** Requires LangChain/LangGraph stack, scikit-learn, seaborn (optional), Tavily, kagglehub, joblib, matplotlib, pandas, numpy, pydantic v2, etc. Shell installs appear in Cell 4 and version check in Cell 14.
- **Artifacts & Paths:** Working directory generated via `TemporaryDirectory`; helper `persist_to_drive` copies artifacts to Google Drive or local `IDD_results` with run IDs.
- **Graph/Agents:** Multi-agent workflow comprises initial analyst → cleaner → analyst → visualization orchestration/evaluator → report orchestration/packager with supervisor routing and memory/tool hooks.
- **Tooling Coverage:** Data access/manipulation, profiling, cleaning, viz generation, file management, PDF/HTML/MD reporting, ML training, Tavily web search, Python REPL, visualization cataloging, and progress logging.
- **State & Persistence:** Rich `State` schema with reducers; memory saver; optional SQLite checkpoint migration/restoration; runtime directories for artifacts/reports.

## Open Considerations
- Shell magics (`pip install`, `pip show`) are present but unrun; environment readiness should be confirmed before execution.
- Several placeholders/partials (e.g., Cell 34 stub) indicate potential unfinished tool wiring.
- Visualization/report prompts and routing depend on properly populated `available_df_ids`, `analysis_config`, and tool registries.
