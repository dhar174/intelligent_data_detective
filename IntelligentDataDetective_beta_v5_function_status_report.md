# IntelligentDataDetective_beta_v5 Function & Status Report

Generated automatically from notebook contents. Covers every cell with imports, functions, classes, assignments, and notable behaviors.

- Total cells: 94
- Code cells: 39
- Markdown cells: 55

---

## Parsing limitations

- Cell 75 mixes embedded HTML/JS with Python and inconsistent indentation, so AST extraction is skipped there. The raw preview remains for reference.

---

## Cell 1: # Intelligent Data Detective Introduction
- Type: markdown
- Preview:
  # Intelligent Data Detective Introduction


## Cell 2: #This variable is to enable custom llama.cpp llama-server connections for using local small models instead.
- Type: code
- Line count: 2
- Imports (0): None
- Classes: none
- Functions: none
- Assignments (1): use_local_llm
- Preview:
  #This variable is to enable custom llama.cpp llama-server connections for using local small models instead.
  use_local_llm = False


## Cell 3: (empty markdown cell)
- Type: markdown
- Preview:
  (empty markdown)


## Cell 4: # 🔧 Environment Setup and Dependency Management
- Type: markdown
- Preview:
  # 🔧 Environment Setup and Dependency Management


## Cell 5: # Import the standard library module for interacting with the operating system (env vars, paths, processes).
- Type: code
- Line count: 80
- Magics / shell cmds:
  !pip install -U langchain_huggingface sentence_transformers
  !pip install -U  langmem langchain-community tavily-python scikit-learn xhtml2pdf joblib langchain langchain-core langchain-openai langchain_experimental langgraph chromadb pydantic python-dotenv tiktoken openpyxl scipy openai langgraph-checkpoint-sqlite
- Imports (11): ast, builtins, contextlib.contextmanager, functools.lru_cache, google.colab.userdata, io.StringIO, logging, os, pathlib.Path, subprocess, sys
- Classes: none
- Functions: none
- Assignments (3): is_colab, oai_key, tavily_key
- Notable calls (5 unique): get_ipython, os.environ.get, print, str, userdata.get
- Preview:
  # Import the standard library module for interacting with the operating system (env vars, paths, processes).
  import os

  # Import utilities to spawn and manage child processes (not used in this snippet, but common in notebooks).
  import subprocess

  # This commented import would load environment variables from a .env file; left disabled because keys are fetched differently below.
  # from dotenv import load_dotenv


## Cell 6: This section handles the initial environment setup, including:
- Type: markdown
- Preview:
  This section handles the initial environment setup, including:
  - **Environment Detection**: Automatically detects if running in Google Colab or local environment
  - **API Key Management**: Securely retrieves OpenAI and Tavily API keys from environment variables or Colab userdata
  - **Package Installation**: Installs all required dependencies including LangChain, LangGraph, and data science libraries
  - **Error Handling**: Provides fallback mechanisms for API key retrieval


## Cell 7: # New Section
- Type: markdown
- Preview:
  # New Section


## Cell 8: # 📚 Core Imports and Type System Foundation
- Type: markdown
- Preview:
  # 📚 Core Imports and Type System Foundation


## Cell 9: # MUST be first in the file/notebook once; do NOT re-import later cells
- Type: code
- Line count: 318
- Imports (126): IPython.display.Image, IPython.display.display, __future__.annotations, ast, base64, collections.OrderedDict, collections.abc.Sequence, datetime.datetime, functools, functools.lru_cache, functools.wraps, google.colab, google.colab.drive, hashlib, inspect, io, io.BytesIO, io.StringIO, itertools, json, kagglehub, langchain_community.agent_toolkits.FileManagementToolkit, langchain_core.embeddings.Embeddings, langchain_core.language_models.chat_models.BaseChatModel, langchain_core.messages.AIMessage, langchain_core.messages.AIMessageChunk, langchain_core.messages.BaseMessage, langchain_core.messages.BaseMessageChunk, langchain_core.messages.ChatMessage, langchain_core.messages.HumanMessage, langchain_core.messages.HumanMessageChunk, langchain_core.messages.RemoveMessage, langchain_core.messages.SystemMessage, langchain_core.messages.SystemMessageChunk, langchain_core.messages.ToolCall, langchain_core.messages.ToolMessage, langchain_core.messages.ToolMessageChunk, langchain_core.messages.trim_messages, langchain_core.messages.utils.message_chunk_to_message, langchain_core.prompts.ChatPromptTemplate, langchain_core.prompts.MessagesPlaceholder, langchain_core.prompts.PromptTemplate, langchain_core.runnables.RunnableLambda, langchain_core.runnables.config.RunnableConfig, langchain_core.tools.InjectedToolArg, langchain_core.tools.InjectedToolCallId, langchain_core.tools.Tool, langchain_core.tools.tool, langchain_experimental.tools.python.tool.PythonAstREPLTool, langchain_huggingface.HuggingFaceEmbeddings, langchain_openai.ChatOpenAI, langgraph.cache.memory.InMemoryCache, langgraph.checkpoint.memory.InMemorySaver, langgraph.checkpoint.memory.MemorySaver, langgraph.graph.END, langgraph.graph.MessagesState, langgraph.graph.START, langgraph.graph.StateGraph, langgraph.graph.state.CompiledStateGraph, langgraph.prebuilt.InjectedState, langgraph.prebuilt.InjectedStore, langgraph.prebuilt.chat_agent_executor.AgentState, langgraph.prebuilt.create_react_agent, langgraph.store.base.BaseStore, langgraph.store.memory.InMemoryStore, langgraph.types.CachePolicy, langgraph.types.Command, langgraph.types.Send, langgraph.utils.config.get_store, langmem.create_manage_memory_tool, langmem.create_search_memory_tool, logging, math, math.nan, matplotlib.figure.Figure, matplotlib.pyplot, numpy, numpy.ndarray, numpy.typing.ArrayLike, operator, operator.add, operator.or_, os, pandas, pandas.Index, pandas.api.types.is_list_like, pprint.pprint, pydantic.AfterValidator, pydantic.BaseModel, pydantic.ConfigDict, pydantic.Field, pydantic.PrivateAttr, pydantic.ValidationError, pydantic.ValidationInfo, pydantic.field_validator, pydantic.model_validator, re, scipy.stats, seaborn, shutil, sklearn.preprocessing.LabelEncoder, sklearn.preprocessing.OneHotEncoder, sys, tavily.TavilyClient, tempfile.TemporaryDirectory, threading, typing.Any, typing.Callable, typing.Dict, typing.Iterable, typing.List, typing.Literal, typing.Mapping, typing.MutableMapping, typing.Optional, typing.Sequence, typing.Set, typing.Tuple, typing.TypeGuard, typing.Union, typing.cast, typing_extensions.Annotated, typing_extensions.NotRequired, typing_extensions.TypeAlias, typing_extensions.TypedDict, uuid
- Classes:
  AgentMembers (methods: log_failed_validation)
  InitialAnalysis (methods: none)
  DataCleaner (methods: none)
  Analyst (methods: none)
  FileWriter (methods: none)
  Visualization (methods: none)
  ReportGenerator (methods: none)
  SuperVisor (methods: none)
  ReportOrchestrator (methods: none)
  ReportSection (methods: none)
- Functions:
  _is_colab()
  _make_idd_results_dir()
  _is_relative_to(a, b)
  persist_to_drive(src, run_id, dst_root, ignore_names) — Copy a file or directory into the IDD_results/IDD_run_<date>_<run_id> folder.
  keep_first(a, b) — Reducer to preserve the first non-null value.
  dict_merge_shallow(old, new) — Merge two dicts shallowly (one level).
  is_1d_vector(x) — Return True if x is a 1-D numeric-like sequence:
  agent_list_default_generator()
- Assignments (29): AgentId, AgentOrSupervisor, Array1D, BinSpec, BinWidthSpec, ColumnSelector, Estimator, Number, RangeSpec, ScalarNum, Supervisor, WORKING_DIRECTORY, _, _HAS_SNS, _TEMP_DIRECTORY, agent_type, ar, arr, base, base_env, br, description, dst_root, file_tools, run_id, src, target, toolkit, ts
- Notable calls (49 unique; showing first 20, 29 omitted): Analyst, DataCleaner, Field, FileManagementToolkit, FileNotFoundError, FileWriter, InitialAnalysis, PathlibPath, PathlibPath.cwd, ReportGenerator, SuperVisor, TemporaryDirectory, ValueError, Visualization, _is_colab, _is_relative_to, _make_idd_results_dir, _should_ignore, a.resolve, ar.startswith
- Preview:
  # MUST be first in the file/notebook once; do NOT re-import later cells
  from __future__ import annotations
  import json, math, inspect
  from functools import wraps

  from langchain_huggingface import HuggingFaceEmbeddings
  from langchain_core.embeddings import Embeddings



## Cell 10: Establishes the foundational imports and type system for the entire notebook:
- Type: markdown
- Preview:
  Establishes the foundational imports and type system for the entire notebook:
  - **Advanced Typing**: Comprehensive type annotations using Python 3.12+ features
  - **LangChain/LangGraph Stack**: Core imports for the multi-agent framework
  - **Data Science Libraries**: Pandas, NumPy, Matplotlib, Seaborn for data analysis
  - **Type Safety**: Extensive use of TypedDict, Literal types, and generic annotations


## Cell 11: # 🤖 OpenAI API Integration and Model Customization
- Type: markdown
- Preview:
  # 🤖 OpenAI API Integration and Model Customization


## Cell 12: from langchain_core.language_models import LanguageModelInput
- Type: code
- Line count: 177
- Imports (7): langchain_core.language_models.LanguageModelInput, langchain_core.messages.BaseMessage, langchain_openai.chat_models.base._construct_responses_api_input, langchain_openai.chat_models.base._convert_message_to_dict, langchain_openai.chat_models.base._convert_to_openai_response_format, langchain_openai.chat_models.base._get_last_messages, langchain_openai.chat_models.base._is_pydantic_class
- Classes:
  MyChatOpenai (methods: _get_request_payload_mod, _get_request_payload)
- Functions:
  _construct_responses_api_payload(messages, payload)
- Assignments (14): existing_text, last_messages, merged_text, messages, model, new_tools, payload, payload_to_use, previous_response_id, schema_dict, strict, structured_text, text_content, verbosity
- Notable calls (20 unique): NotImplementedError, _construct_responses_api_input, _construct_responses_api_payload, _convert_message_to_dict, _convert_to_openai_response_format, _get_last_messages, _is_pydantic_class, existing_text.copy, isinstance, merged_text.update, model.startswith, new_tools.append, payload.get, payload.pop, re.match, schema.model_json_schema, self._convert_input, self._get_request_payload_mod, self._use_responses_api, to_messages
- Preview:
  from langchain_core.language_models import LanguageModelInput
  from langchain_openai.chat_models.base import _construct_responses_api_input, _is_pydantic_class, _convert_message_to_dict, _convert_to_openai_response_format, _get_last_messages
  from langchain_core.messages import BaseMessage

  def _construct_responses_api_payload(
      messages: Sequence[BaseMessage], payload: dict
  ) -> dict:
      # Rename legacy parameters


## Cell 13: Custom ChatOpenAI implementation with advanced features:
- Type: markdown
- Preview:
  Custom ChatOpenAI implementation with advanced features:
  - **GPT-5 Support**: Forward-compatible implementation for o-series models
  - **Responses API**: Handles transition from legacy to new OpenAI API patterns
  - **Model-Specific Logic**: Adapts behavior based on model capabilities and limitations
  - **Parameter Mapping**: Proper handling of deprecated and new API parameters


## Cell 14: # 📋 Dependency Version Verification
- Type: markdown
- Preview:
  # 📋 Dependency Version Verification


## Cell 15: !pip show --verbose langchain_experimental
- Type: code
- Line count: 1
- Magics / shell cmds:
  !pip show --verbose langchain_experimental
- Imports (0): None
- Classes: none
- Functions: none
- Preview:
  !pip show --verbose langchain_experimental


## Cell 16: Quick verification of installed package versions:
- Type: markdown
- Preview:
  Quick verification of installed package versions:
  - **LangChain Experimental**: Checks the version of experimental features being used
  - **Compatibility Validation**: Ensures correct versions are installed for the workflow


## Cell 17: # 🏗️ Data Models and State Architecture
- Type: markdown
- Preview:
  # 🏗️ Data Models and State Architecture


## Cell 18: # Support models — keep this cell before nodes/supervisor/graph
- Type: code
- Line count: 311
- Imports (2): typing.ClassVar, typing.List
- Classes:
  BaseNoExtrasModel (methods: none)
  AnalysisConfig (methods: none) — User-configurable settings for the data analysis workflow.
  CleaningMetadata (methods: none) — Metadata about the data cleaning actions taken.
  InitialDescription (methods: none) — Initial description of the dataset.
  VizSpec (methods: none)
  AnalysisInsights (methods: none) — Insights from the exploratory data analysis.
  ImagePayload (methods: ensure_b64, enforce_size) — Wrap both the image bytes and its declared MIME-type.
  DataVisualization (methods: none) — Individual visualizations generated
  VisualizationResults (methods: none) — Results from the visualization generation.
  ReportResults (methods: none) — Results from the report generation.
  DataQueryParams (methods: none) — Parameters for querying the DataFrame.
  QueryDataframeInput (methods: none)
  FileResult (methods: none) — Results object storing metadata from the file generation or editing. The fields include the following:
  ListOfFiles (methods: none) — List of metadata as FileResult objects for the files generated.
  DataFrameRegistryError (methods: __init__, __str__, __repr__, to_dict) — Exception raised for errors in the DataFrameRegistry.
  ProgressReport (methods: none)
  PlanStep (methods: none)
  Plan (methods: _sync_step_versions_on_assignment, _sync_steps_and_assert_increasing)
  CompletedStepsAndTasks (methods: _inject_and_dedupe, _sorted_no_dups_and_subset)
  ToDoList (methods: none)
  NextAgentMetadata (methods: none)
  SendAgentMessage (methods: none)
  MessagesToAgentsList (methods: none)
- Functions:
  _sort_plan_steps(steps)
  _assert_sorted_completed_no_dups(steps)
  _norm(s)
  _triplet_from_raw(d)
- Assignments (100): Triplet, _lock, _next, _ver_assigned, agent_obj_needs_recreated_bool, agg, allowed, anomaly_insights, bins, cand_score, category_tag, columns, completed_steps, correlation_insights, d, data_description_after_cleaning, data_sample, dataset_description, datetime_format_preference, dedup_list, default_visualization_style, delivery_status, description, df_id, expect_reply, file_content, file_description, file_name, file_path, file_type, files, filter_column, filter_value, finished_tasks, finished_this_task, html_report_path, hue, immediate_emergency_reroute_to_recipient, is_final_report, is_message_critical, is_step_complete, k, key, large_dataframe_preview_rows, latest_progress, limit, markdown_report_path, max_bytes, message, messages_to_agents, mime, model_config, norm, notes, nums, operation, path, payload, pdf_report_path, plan, plan_steps, plan_summary, plan_title, plan_version, prev, prev_score, progress_report, pv, query, recipient, recommended_next_steps, recommended_visualizations, reply_msg_to_supervisor, report_author, section_name, seen, step_description, step_name, step_number, steps, steps_taken, style, summary, t, title, to_do_list, v, visualization_description, visualization_id, visualization_style, visualization_title, visualization_type, visualizations, viz_id, viz_instructions, viz_spec, viz_type, write_success, x, y
- Notable calls (39 unique; showing first 20, 19 omitted): AfterValidator, ConfigDict, Field, PlanStep.model_validate, PrivateAttr, ValueError, __init__, _norm, _triplet_from_raw, any, base64.b64decode, bool, d.get, dedup_list.sort, dict, field_validator, get, hasattr, info.data.get, int
- Preview:
  # Support models — keep this cell before nodes/supervisor/graph
  class BaseNoExtrasModel(BaseModel):
      model_config = ConfigDict(extra="forbid",json_schema_extra={"additionalProperties": False}) # -> additionalProperties: false
      reply_msg_to_supervisor: str = Field(...,description="Message to send to the supervisor. Can be a simple message stating completion of the task, or it can be detailed information about the result, or you can put any questions for the supervisor here as well. This is ONLY for sending messages to the supervisor, NOT to worker agents. If you are the/a supervisor (or the router, planner, or progress reporter), this field should be empty unless you are expecting a reply from the main supervisor, NOT from a worker agent.")
      finished_this_task: bool = Field(...,description="Whether this assigned task represented by this object has been completed. For example, if it is a Router object, this field should be True if the route decision has been made. Another example, if it is a CleaningMetadata object, this field should be True if the cleaning has been completed.")
      expect_reply: bool = Field(...,description="Whether you expect a reply from the supervisor based on content of 'reply_msg_to_supervisor'. This is ONLY for receiving replies from the supervisor, not from worker agents. If you are the/a supervisor (or the router, planner, or progress reporter), only set this to True if you are expecting a reply from the main supervisor, NOT from a worker agent. Worker agents will always reply to 'next_agent_prompt' when routed to.")




## Cell 19: Defines the core data structures and state management system:
- Type: markdown
- Preview:
  Defines the core data structures and state management system:
  - **Pydantic Models**: Type-safe data models with validation for all workflow stages
  - **State Definition**: Central state object managing the entire multi-agent workflow
  - **Configuration Models**: User settings and analysis configuration structures
  - **Result Models**: Structured outputs for each analysis phase (cleaning, analysis, visualization, reporting)


## Cell 20: # 📊 DataFrame Registry and Data Management
- Type: markdown
- Preview:
  # 📊 DataFrame Registry and Data Management


## Cell 21: import threading
- Type: code
- Line count: 249
- Imports (1): threading
- Classes:
  DataFrameRegistry (methods: __init__, _norm_path, _write_df, _read_df, _touch_cache, write_dataframe_to_csv_file, write_dataframe_to_parquet_file, write_dataframe_to_pickle_file, write_dataframe_to_json_file, write_dataframe_to_file, register_dataframe, get_dataframe, remove_dataframe, get_raw_path_from_id, get_id_from_raw_path, has_df, ids, size)
  Section (methods: none)
  SectionOutline (methods: none)
  ReportOutline (methods: none)
  VizFeedback (methods: none)
  ConversationalResponse (methods: none)
- Functions:
  get_global_df_registry()
- Assignments (27): _, content, data_signals, data_signals_available, data_signals_needed, description, df, df_id, evicted_id, expected_figures, feedback, global_df_registry, goals, grade, info, loaded, name, path, raw_path, redo_list, response, section_num, sections, suf, target, title, word_target
- Notable calls (42 unique; showing first 20, 22 omitted): DataFrameRegistry, Field, OrderedDict, PathlibPath, df.to_csv, df.to_json, df.to_parquet, df.to_pickle, expanduser, get_global_df_registry, info.get, isinstance, len, list, os.path.exists, path.exists, path.is_file, path.parent.exists, path.parent.mkdir, path.suffix.lower
- Preview:
  import threading

  class DataFrameRegistry:
      def __init__(self, capacity=20):
          self._lock = threading.RLock()
          self.registry: Dict[str, dict] = {}
          self.df_id_to_raw_path: Dict[str, str] = {}
          self.cache = OrderedDict()


## Cell 22: Centralized DataFrame management system with caching:
- Type: markdown
- Preview:
  Centralized DataFrame management system with caching:
  - **LRU Cache**: Efficient memory management for large datasets
  - **Auto-ID Generation**: Automatic unique identifier assignment for DataFrames
  - **Thread Safety**: Concurrent access protection for multi-agent operations
  - **File Integration**: Seamless loading and registration of datasets from file paths


## Cell 23: # 🔄 State Management and Reducer Functions
- Type: markdown
- Preview:
  # 🔄 State Management and Reducer Functions


## Cell 24: from langchain_core.messages import AnyMessage
- Type: code
- Line count: 194
- Imports (3): langchain_core.messages.AnyMessage, langgraph.graph.message.REMOVE_ALL_MESSAGES, langgraph.graph.message.add_messages
- Classes:
  State (methods: none)
  VizWorkerState (methods: none)
- Functions:
  merge_lists(a, b)
  merge_unique(a, b)
  merge_int_sum(a, b)
  merge_dicts(a, b)
  merge_dict(a, b)
  any_true(a, b)
  last_wins(a, b)
  _reduce_plan_keep_sorted(a, b)
- Assignments (74): CLASS_TO_AGENT, _config, _count_, _id_, analysis_config, analysis_insights, analyst_agent, analyst_complete, artifacts_path, available_df_ids, by_num, cleaned_dataset_description, cleaning_metadata, completed_plan_steps, completed_tasks, current_dataframe, current_dataframe_id, current_plan, current_turn_agent_id, d, data_cleaner_agent, data_cleaning_complete, default_an_config, emergency_reroute, file_content, file_results, file_writer_complete, final_report_path, final_turn_msgs_list, individual_viz_task, initial_analysis_agent, initial_analysis_complete, initial_description, last_agent_expects_reply, last_agent_finished_this_task, last_agent_id, last_agent_message, last_agent_reply_msg, last_created_obj, latest_progress, logs_path, merged, merged_sorted_steps, next, next_agent_metadata, next_agent_prompt, norm, progress_reports, report_draft, report_generator_complete, report_outline, report_paths, report_results, reports_path, run_id, section, sections, steps, structured_response, supervisor_to_agent_msgs, task, to_do_list, user_prompt, visualization_complete, visualization_results, viz_eval_result, viz_feedback, viz_grade, viz_paths, viz_results, viz_spec, viz_specs, viz_tasks, written_sections
- Notable calls (14 unique): AfterValidator, AnalysisConfig, Plan.model_validate, PlanStep.model_validate, a.model_dump, b.model_dump, bool, d.update, dict.fromkeys, int, isinstance, list, sorted, steps.extend
- Preview:
  from langchain_core.messages import AnyMessage
  from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES


  # --- custom reducers ---
  def merge_lists(a: list | None, b: list | None) -> list:
      return (a or []) + (b or [])



## Cell 25: Custom reducer functions for state merging and management:
- Type: markdown
- Preview:
  Custom reducer functions for state merging and management:
  - **Message Handling**: Manages conversation history and agent communications
  - **List Merging**: Intelligent merging of analysis results and metadata
  - **Unique Merging**: Deduplication strategies for accumulated data
  - **State Persistence**: Ensures proper state transitions across agent workflows


## Cell 26: # 💬 Agent Prompt Templates and Instructions
- Type: markdown
- Preview:
  # 💬 Agent Prompt Templates and Instructions


## Cell 27: # === Agent Prompt Templates (ChatPromptTemplate) =================================
- Type: code
- Line count: 1077
- Imports (2): langchain_core.prompts.ChatPromptTemplate, langchain_core.prompts.MessagesPlaceholder
- Classes: none
- Functions: none
- Assignments (19): DEFAULT_TOOLING_GUIDELINES, DEFAULT_TOOLING_GUIDELINES_MINI, DEFAULT_TOOLING_GUIDELINES_MINI_V2, analyst_prompt_template_initial, analyst_prompt_template_initial_mini, analyst_prompt_template_main, analyst_prompt_template_main_mini, data_cleaner_prompt_template, data_cleaner_prompt_template_mini, file_writer_prompt_template, file_writer_prompt_template_mini, replan_str, report_generator_prompt_template, report_generator_prompt_template_mini, todo_str, visualization_prompt_template, visualization_prompt_template_mini, viz_evaluator_prompt_template, viz_evaluator_prompt_template_mini
- Notable calls (3 unique): ChatPromptTemplate.from_messages, MessagesPlaceholder, partial
- Preview:
  # === Agent Prompt Templates (ChatPromptTemplate) =================================
  from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

  # Shared guidance injected into each system prompt
  DEFAULT_TOOLING_GUIDELINES = (
      """
      <tool_preambles>
      Tool-use policy:


## Cell 28: Comprehensive prompt templates for each specialized agent:
- Type: markdown
- Preview:
  Comprehensive prompt templates for each specialized agent:
  - **Role-Specific Prompts**: Tailored instructions for data cleaner, analyst, visualizer, and report generator
  - **Tool Integration Guidelines**: Clear guidance on tool usage patterns and best practices
  - **Output Format Specifications**: Structured JSON schema compliance requirements
  - **Contextual Instructions**: Dynamic prompt adaptation based on data characteristics


## Cell 29: # 📝 Supervisor and Planning Prompt Templates
- Type: markdown
- Preview:
  # 📝 Supervisor and Planning Prompt Templates


## Cell 30: from langchain_core.prompts import ChatPromptTemplate
- Type: code
- Line count: 194
- Imports (1): langchain_core.prompts.ChatPromptTemplate
- Classes: none
- Functions: none
- Assignments (3): plan_prompt, replan_prompt, todo_prompt
- Notable calls (3 unique): ChatPromptTemplate.from_messages, MessagesPlaceholder, partial
- Preview:
  from langchain_core.prompts import ChatPromptTemplate

  plan_prompt = ChatPromptTemplate.from_messages(
      [("system",
  """For the given objective, produce a concise, numbered plan with only the remaining steps needed to reach the final answer.

  <persistence>
     - Please keep thinking until the plan is finalized, then ending your turn and yielding your final output.


## Cell 31: Advanced prompt templates for workflow orchestration:
- Type: markdown
- Preview:
  Advanced prompt templates for workflow orchestration:
  - **Supervisor Prompts**: Templates for the main coordinator agent
  - **Planning Templates**: Strategic analysis and task distribution prompts
  - **Decision Logic**: Routing and workflow control instructions


## Cell 32: # 🛠️ Comprehensive Tool Ecosystem and Error Handling
- Type: markdown
- Preview:
  # 🛠️ Comprehensive Tool Ecosystem and Error Handling


## Cell 33: import json
- Type: code
- Line count: 169
- Imports (10): functools.wraps, json, logging, math, pandas, pydantic.BaseModel, typing.Any, typing.Callable, typing.Optional, typing.Sequence
- Classes: none
- Functions:
  _is_df(x)
  _to_jsonable(x) — Safely convert to something json.dumps can handle.
  _pretty(obj, *, minify_json) — Human-friendly string with JSON where possible.
  cap_output(max_chars, max_bytes, max_lines, *, minify_json, add_footer, footer_prefix)
- Assignments (21): _LOG, b, df, ell, ell_b, enc, head, keep_head, keep_tail, lines, new_bytes, new_chars, new_lines, orig_bytes, orig_chars, orig_lines, out, s, sig, tail, truncated
- Notable calls (28 unique; showing first 20, 8 omitted): _LOG.info, _is_df, _to_jsonable, decode, df.head, fn, getattr, inspect.signature, int, isinstance, join, json.dumps, len, list, logging.getLogger, map, math.ceil, max, repr, s.count
- Preview:
  import json
  import math
  import logging
  from functools import wraps
  from typing import Any, Callable, Optional, Sequence

  # Optional: if you use pandas or pydantic v2
  import pandas as pd  # type: ignore


## Cell 34: # Filename helpers for saving files
- Type: code
- Line count: 5146
- Imports (19): base64, binascii, csv, html, joblib, langchain_core.runnables.config.RunnableConfig, langchain_core.tools.InjectedToolArg, numpy, pandas, scipy.cluster.hierarchy, scipy.spatial.distance.squareform, shutil, sklearn.linear_model.LinearRegression, sklearn.linear_model.LogisticRegression, sklearn.metrics.accuracy_score, sklearn.metrics.mean_squared_error, sklearn.model_selection.train_test_split, typing.Optional, xhtml2pdf.pisa
- Classes:
  NewPythonInputs (methods: none) — Python inputs.
  NpEncoder (methods: default)
- Functions:
  validate_dataframe_exists(df_id) — Validates the existence and validity of a dataframe by its ID.
  handle_tool_errors(func) — Decorator for consistent error handling across tool functions.
  get_dataframe_schema(df_id) — Return a summary of the DataFrame's schema and sample data.
  get_column_names(df_id) — Useful to get the names of the columns in the current DataFrame.
  check_missing_values(df_id) — Checks for missing values in a pandas DataFrame and returns a summary.
  drop_column(df_id, column_name) — Drops a specified column from the DataFrame.
  delete_rows(df_id, conditions, inplace) — Deletes rows from the DataFrame based on specified conditions.
  fill_missing_median(df_id, column_name) — Fills missing values in a specified column with the median.
  query_dataframe(params) — Query a registered DataFrame by columns, optional equality filter, and an operation.
  get_descriptive_statistics(df_id, column_names) — Calculates descriptive statistics for specified columns in the DataFrame.
  calculate_correlation(df_id, column1_name, column2_name) — Calculates the Pearson correlation coefficient between two columns.
  perform_hypothesis_test(df_id, column_name, value) — Performs a one-sample t-test.
  _get_artifacts_base(config) — Resolve the base directory for artifacts. Priority:
  _is_subpath(path, parent)
  _resolve_artifact_path(file_name, *, config, subdir, create_parents) — If `file_name` is relative -> resolve under artifacts_dir[/subdir].
  create_sample(points, file_name) — Create and save a data sample.
  read_file(file_name, start, end, return_bytes, *, config) — Read a text file safely. If `file_name` is relative, it is resolved under
  write_file(content, file_name, sub_dir) — Write UTF‑8 text to a secure path.
  edit_file(file_name, inserts, return_file, return_file_type, *, config) — Usefule for editing structured text files.
  get_df_from_registry(df_id_local)
  save_df_to_registry(df_id_local, df)
  strip_code_fences(code)
  sanitize_code(code)
  ensure_last_fn_call(code) — If the last top-level statement is a FunctionDef, append a no-arg call
  _resolve_sandbox_root(globs)
  _inside(root, p)
  sandbox_filesystem(root, *, block_chdir) — Temporarily:
  python_repl_tool(code, df_id) — Executes Python code within a Python REPL with access to the global registry, and the current DataFrame if df_id is provided, with AST-based execution.
  _as_number_or_list(x) — Return list[float] or None. Accepts scalars/iterables; parses str via float.
  _as_int_or_list(x) — Return list[int] or None. Accepts scalars/iterables; parses str via int.
  _encode_png(fig, return_bytes)
  _resolve_columns(df, column_name, allow_overlay, max_overlay)
  _normalize_bins(bins) — Return one of: 'auto' | int | list[int].
  _assert_bin_var_typesafety(bins) — Assert bins variable is of BinSpec
  _align_vector(v, base_index, df_index) — Return a Series aligned to `base_index` for weights/hue-like vectors.
  _normalize(counts, edges, stat)
  create_histogram(df_id, *, columns, rows, hue, weights, bins, binwidth, binrange, overlay, max_overlay, discrete, common_bins, common_norm, stat, multiple, element, fill, shrink, cumulative, kde, density, dropna, coerce_numeric, x_range, sample_n, sample_frac, legend, return_bytes) — Generates a 1-D or multi-overlay histogram from a registered ``DataFrame``.
  create_scatter_plot(df_id, *, x, y, overlay_y, max_overlay, hue, style, size, point_sizes, alpha, rows, dropna, coerce_numeric, x_range, y_range, sample_n, sample_frac, legend, marker, edgecolor, linewidth, return_bytes) — Generate a scatter plot for one X against one or more Y series, with optional
  create_correlation_heatmap(df_id, *, columns, rows, sample_n, sample_frac, dropna, coerce_numeric, method, min_periods, absolute, cluster, order, mask_upper, mask_diagonal, annot, fmt, cmap, cbar, linewidths, linecolor, figsize, vmin, vmax, center, return_bytes) — Generate a correlation heatmap over selected numeric columns with optional
  create_box_plot(df_id, *, values, group, hue, overlay_values, max_overlay, order, hue_order, orient, width, whis, notch, showcaps, showfliers, linewidth, rows, dropna, coerce_numeric, y_range, sample_n, sample_frac, rotate_xticks, tight_layout, legend, return_bytes) — Generate a box plot for one or more value columns, optionally grouped by a primary
  create_violin_plot(df_id, *, values, group, hue, overlay_values, max_overlay, order, hue_order, split, orient, width, inner, gridsize, cut, linewidth, rows, dropna, coerce_numeric, y_range, sample_n, sample_frac, rotate_xticks, tight_layout, legend, return_bytes) — Generate a violin plot for one or more value columns, optionally grouped by a primary
  export_dataframe(df_id, *, file_name, file_format, columns, include_index, overwrite, sep, encoding, na_rep, float_format, date_format, quoting, compression, sheet_name, json_orient, json_lines, indent, parquet_engine) — Export a registered DataFrame to disk with safe file naming, optional
  detect_and_remove_duplicates(df_id, *, subset, keep, casefold, normalize_ws, dry_run, sample_duplicates) — Detect (and optionally remove) duplicate rows with flexible subset selection,
  convert_data_types(df_id, *, column_types, errors, prefer_nullable, datetime_formats, to_category, numeric_locale, downcast, dry_run) — Convert specified columns to target dtypes with clear reporting and policies
  generate_html_report(report_title, text_sections, image_sections) — Generates an HTML report from text and image sections and saves it to a file.
  calculate_correlation_matrix(df_id, column_names) — Calculates the correlation matrix for numeric columns in a DataFrame.
  detect_outliers(df_id, column_name) — Detects outliers in a numeric column of a DataFrame using the IQR method.
  perform_normality_test(df_id, column_name) — Performs a Shapiro-Wilk normality test on a numeric column.
  assess_data_quality(df_id) — Provides a comprehensive data quality assessment for a DataFrame.
  search_web_for_context(query, max_results) — Performs a web search using Tavily API to find external context or insights.
  load_multiple_files(file_paths, file_type) — Loads multiple data files (e.g., CSVs, JSONs) into DataFrames.
  merge_dataframes(left_df_id, right_df_id, how, on, left_on, right_on) — Merges two DataFrames based on specified keys and join type.
  standardize_column_names(df_id, rule) — Standardizes column names of a DataFrame.
  format_markdown_report(report_title, text_sections, image_sections) — Formats a report from text and image sections into a Markdown file.
  create_pdf_report(html_file_path_str) — Converts a given HTML file (in the working directory) to a PDF report.
  train_ml_model(df_id, feature_columns, target_column, model_type, test_size, random_state, save_model) — Trains a specified ML model on the DataFrame.
  handle_categorical_encoding(df_id, column_name, strategy) — Applies categorical encoding to a specified column.
  report_intermediate_progress(progress_message, state, tool_call_id, **kwargs) — Use this tool every several turns to continuously and repeatedly report on your step-by-step progress to your supervisor and directly to the user.
  _hash_id(s)
  _coerce_viz_dict(path, vtype, title, style, desc) — Turn a file path into a DataVisualization-like dict; fill what we can.
  _gather_from_state(state) — Prefer state.visualization_results; else viz_results/viz_paths; else empty.
  _scan_artifacts_dir(artifacts_dir)
  _encode_preview(path, max_bytes)
  list_visualizations(query, viz_type, limit, start, include_previews, artifacts_dir, state) — List available visualizations known to the current run.
  get_visualization(visualization_id, path, include_preview, artifacts_dir, state) — Fetch a single visualization by id or path. You may pass either id or path.
- Assignments (326): ConfigParam, IQR, Q1, Q3, X, X_test, X_train, _, _CODE_FENCE_RE, _IMAGE_EXTS, after, all_conditions, all_vals, allowed, allowed_how_types, alpha, alt, analyst_tools, arr, arrays, artifact, artifact_string, artifacts_dir, assets_dir, ax, b64, bad, base, base_candidate, base_dir, before, bin_edges, binrange, bins, binwidth, br, buf, cand, candidate, cats, cfg, cleaned, client, cmap, code, code_to_run, cols, cols_in, cols_ordered, column_data, column_name, columns, columns_added, columns_removed, columns_to_describe, comp_arg, content, corr, corr_matrix, correlation, counts, csv_kwargs, cut, data, data_cleaning_tools, dec, deduped, dend, desc_stats, detail, df, df_cleaned, df_copy, df_id, df_numeric, df_to_correlate, dist, dropped, dst, dst_name, dup_mask, dvec, e, e_idx, edges, effective_shrink, effective_stat, encoded_data, encoded_df, encoder, engine, error_msg, ext, ext_map, f, fig, fig_path, fig_save_dir, file_name, file_path, file_writer_tools, filtered, filtered_df, final_text, fmt, fn_name, fname, formatted_results, full_path, g, group_name, groups, h, hay, hi, how, html_content, html_str, hue, hue_name, i, im, image_base64, image_bytes, init_analyst_tools, inner, interpretation, is_b64, is_base64, is_normal, items, k, kept, key, key_cols, kind, kwargs, last_stmt, latest_progress, left_df, left_on, legend, linecolor, lines, link, lo, long_df, lower, lower_bound, mask, md_content, md_parts, median_value, memory_usage_bytes, merged_df, message, meta, metric_name, metric_value, mime, min_periods, missing, missing_cols, missing_features, missing_info, model, model_filename, model_full_path, model_path, msg, n_equal_bins, n_points, new_col_name, new_cols, new_columns, new_df_id, new_merged_df_id, non_numeric, note, note2, np_path, num_dup_rows, num_duplicates, numeric, numeric_candidates, numeric_cols, obj_cols, ok, on, opt, order_applied, order_idx, ordered, orient_resolved, orig_chdir, orig_cwd, orig_open, original_columns, original_raw_path, out, out_path, outliers, overlay, overlay_eff, p, p_value, page, parsed, parts, path, paths, pdf_file_path, per_counts, per_edges, per_norm, pick, pisa_status, plot_df, plot_mat, plot_sizes, possible_bin_strs, preview, progress_message_final, python_repl, quality_report, query, query_str, raw, raw_path, raw_path_info, rel, relevant_columns, report_generator_tools, requested, resolved, response, result, results, results_summary, right_df, right_on, root, rows_removed, rows_to_drop, rt, runtime, s, s1, s2, s3, s4, s_idx, safe_cols, safe_title, safe_title_for_file, sample, sandbox_root, schema, schema_string, sd_lower, sd_path, seen, size_key, sizes, snippet, sorted_edits, source_html_path, src, st, stat, status, stem, sz, t_statistic, target, tavily_api_key, text, tgt, th, tid, title_bits, title_cols, title_escaped, title_vals, title_y, total, total_cells, tree, types, uid, update, upper_bound, v, val, vals, value_names_raw, value_numeric, variances, visr, visualization_tools, vp, w, w_base, w_np, w_series, wd, whis, widths, work, write_root, write_root_candidate, x, x_filter, x_name, xcol, xvals, y, y_names_raw, y_numeric, y_pred, y_test, y_train, ycol, ylabel_map, ymask, yvals
- Notable calls (438 unique; showing first 20, 418 omitted): Field, InjectedToolArg, LabelEncoder, LinearRegression, LogisticRegression, OneHotEncoder, PathlibPath, PathlibPath.cwd, PermissionError, PythonAstREPLTool, RunnableConfig, StringIO, TavilyClient, ToolMessage, TypeError, ValueError, WORKING_DIRECTORY.resolve, _CODE_FENCE_RE.sub, _align_vector, _as_int_or_list
- Preview:
  # Filename helpers for saving files
  # ---- shared helpers (place once) ----
  import base64, binascii, html, shutil


  # Error Handling and Validation Framework
  def validate_dataframe_exists(df_id: str) -> bool:
      """Validates the existence and validity of a dataframe by its ID.


## Cell 35: # @tool("call_file_w
- Type: code
- Line count: 1
- Imports (0): None
- Classes: none
- Functions: none
- Preview:
  # @tool("call_file_w


## Cell 36: The complete toolkit for data analysis operations:
- Type: markdown
- Preview:
  The complete toolkit for data analysis operations:
  - **Data Analysis Tools**: Statistical analysis, correlation, hypothesis testing, outlier detection
  - **Data Cleaning Tools**: Missing value handling, duplicate removal, type conversion
  - **Visualization Engine**: Chart generation (histograms, scatter plots, heatmaps, box plots)
  - **File Management**: Read/write operations for multiple formats (CSV, Excel, JSON, HTML, PDF)
  - **Python REPL Integration**: Dynamic code execution capabilities


## Cell 37: # 🔧 Extended Imports and Memory Integration
- Type: markdown
- Preview:
  # 🔧 Extended Imports and Memory Integration


## Cell 38: from langchain.embeddings import init_embeddings
- Type: code
- Line count: 1536
- Imports (22): builtins, dataclasses.dataclass, functools.lru_cache, functools.partial, langchain.embeddings.init_embeddings, langchain_core.embeddings.Embeddings, langchain_core.messages.HumanMessage, langchain_core.messages.SystemMessage, langgraph.checkpoint.memory.InMemorySaver, langgraph.store.memory.InMemoryStore, langgraph.types.Command, langgraph.types.Send, langgraph.utils.config.get_store, logging, math, os, time, typing.Callable, typing.List, typing.Union, uuid, yaml
- Classes:
  MemoryRecord (methods: __post_init__) — Enhanced memory record with lifecycle management fields.
  MemoryPolicy (methods: none) — Memory lifecycle policy configuration.
  RankingWeights (methods: none) — Weights for memory retrieval ranking.
  PruneReport (methods: none) — Report from pruning operations.
  MemoryPolicyEngine (methods: __init__, insert, retrieve, prune, recalc_importance, _check_duplicates, _rank_memories, _calculate_score, _calculate_recency_factor, _update_usage, _maybe_prune) — Core engine for memory lifecycle management with policy-driven operations.
- Functions:
  identity(xs)
  e5_docs(xs)
  e5_query(q)
  make_doc_embedder(embeddings, preproc) — Returns a function(texts) -> list[list[float]] bound to this embeddings instance.
  make_query_embedder(embeddings, preproc) — Returns a function(query: str) -> list[float] bound to the given embeddings.
  load_memory_policy(config_path) — Load memory policy configuration from YAML file.
  estimate_importance(kind, text) — Estimate base importance score for a memory item based on content heuristics.
  calculate_similarity(text1, text2) — Simple similarity calculation (can be enhanced with embeddings later).
  put_memory(store, kind, text, meta, user_id, use_policy_engine) — Store a memory item with categorization and optional policy enforcement.
  retrieve_memories(store, query, kinds, limit, user_id, use_policy_engine) — Retrieve memories with optional kind filtering, fallback, and enhanced ranking.
  format_memories_by_kind(memories) — Format memories grouped by kind for prompt inclusion.
  enhanced_retrieve_mem(state, kinds, limit) — Enhanced memory retrieval function for use in agent nodes.
  enhanced_mem_text(query, kinds, limit, store) — Enhanced version of _mem_text with kind support.
  get_memory_metrics() — Get current memory metrics for monitoring and debugging.
  reset_memory_metrics() — Reset memory metrics counters.
  memory_policy_report(store) — Generate a diagnostic report of memory status and policy compliance.
  put_memory_with_policy(store, kind, text, meta, user_id) — Store memory with full policy engine features enabled.
  retrieve_memories_with_ranking(store, query, kinds, limit, user_id) — Retrieve memories with enhanced ranking enabled.
  prune_memories(store, reason) — Manually trigger memory pruning.
  recalculate_importance(store, kinds) — Recalculate dynamic importance for stored memories.
  update_memory_with_kind(state, config, kind, memstore, text) — Enhanced update_memory function with memory kind categorization.
  categorize_memory_by_context(state, last_agent_id, error) — NOTE: Not yet in use.
  store_categorized_memory(state, config, memstore) — NOTE: Not yet in use.
- Assignments (122): DIM, MEMORY_CONFIG, MEMORY_METRICS, MEMORY_POLICIES, MEM_EMBEDDINGS, MODEL_NAME, MemoryKind, RANKING_WEIGHTS, age, agent_memory_mapping, ages, analytical_keywords, avg_age, base_importance, candidates, config, config_path, config_str, count, created_at, current_dir, current_time, decay_floor, decay_half_life_seconds, default_policy, degraded, doc_embed_func, doc_pre_arg, dynamic_importance, engine, existing_text, expired_count, fallback_limit, generic_namespace, grouped, hfembeddings, id, importance, in_memory_store, intersection, item, items, items_to_keep, items_to_remove, keep_score, keyword_count, keyword_score, kind, kind_config, kind_info, kind_keys, kind_limit, kind_max_items, kind_pruned, kinds_config, kinds_to_process, last_agent_id, last_message, last_used_at, length_score, low_importance_count, max_items, max_items_per_kind, mem_ids_str, mem_tools, memids, memories, memory_id, memory_kind, memstore, meta, min_importance, namespace, next_agent, oai_embeds, policies, policy, policy_defaults, prep, progress_tool, query, query_embed_func, query_pre_arg, ranked, ranking_config, recency, recency_factor, record, records, remaining_count, remaining_items, report, results, role_weight, role_weights, score, scored, scored_items, section_content, section_title, sections, similarity, size_pruned_count, store, superseded_by, superseded_count, text, to_delete, total_pruned, total_updated, ttl_seconds, union, usage, usage_count, usage_factor, use_policy_engine, user_id, user_namespace, vector, weights, words1, words2
- Notable calls (109 unique; showing first 20, 89 omitted): Field, HuggingFaceEmbeddings, InMemoryStore, MEMORY_METRICS.copy, MEMORY_POLICIES.get, MemoryPolicy, MemoryPolicyEngine, MemoryRecord, PruneReport, RankingWeights, agent_memory_mapping.get, agent_memory_mapping.keys, ages.append, append, calculate_similarity, candidates.append, categorize_memory_by_context, config.get, create_manage_memory_tool, create_search_memory_tool
- Preview:
  from langchain.embeddings import init_embeddings
  from langchain_core.embeddings import Embeddings
  from langchain_core.messages import SystemMessage, HumanMessage
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.store.memory import InMemoryStore
  from langgraph.types import Command, Send
  from functools import lru_cache
  from typing import List, Union


## Cell 39: Additional imports and setup for advanced features:
- Type: markdown
- Preview:
  Additional imports and setup for advanced features:
  - **Memory Systems**: Integration with LangGraph memory and checkpointing
  - **Embedding Models**: Setup for vector storage and retrieval
  - **Store Integration**: Advanced state management with persistent storage
  - **Command Types**: Support for complex workflow commands and parallel processing


## Cell 40: # LLM Initialization and Agent Factories
- Type: markdown
- Preview:
  # LLM Initialization and Agent Factories


## Cell 41: # Global toggles
- Type: code
- Line count: 699
- Imports (16): enum.Enum, json, langchain_core.messages.AIMessage, langchain_core.messages.BaseMessage, pydantic.BaseModel, pydantic.fields.FieldInfo, pydantic_core.PydanticUndefined, re, typing.Any, typing.Dict, typing.List, typing.Optional, typing.Type, typing.get_args, typing.get_origin, uuid
- Classes: none
- Functions:
  _safe_json_loads(s)
  extract_tool_calls(text) — Extract tool calls from model response
  retry_extract_tool_calls(text) — Returns a list of {"name": str, "arguments": dict} from assistant text.
  format_tool_responses_for_qwen3(tool_messages) — Given a list of (tool_name, tool_content) -> build a single assistant content
  _collect_trailing_tool_messages(msgs) — Walk backwards until we hit a non-ToolMessage; return [(tool_name, content)], and the index
  qwen3_pre_model_hook(state) — - Convert trailing ToolMessages -> single assistant message with <tool_response> blocks
  _parse_inline_tool_calls(text) — Accepts either:
  getnestedattr(obj, attr, default)
  remove_unwanted_tags(text)
  qwen3_post_model_hook(state) — Post-model hook for create_react_agent:
  is_final_answer(messages)
  extract_final_text(messages)
  _is_required(fi)
  _type_name(tp) — Turn typing/Annotated/Union/etc. into a readable type string.
  _enum_values(tp)
  collect_model_docs(Model) — Returns a dict you can format however you like:
  format_model_for_prompt(Model) — Produces a concise, prompt-ready string block describing the model.
  count_last_cycle_tool_calls(messages)
  _final_llm_for_model(base_llm, pyd_model) — Returns an LLM that enforces pyd_model via JSON Schema for the *final* hop.
  _strict_final_wrapper(agent, base_llm, pyd_model) — Wrap a ReAct agent so that after it finishes tool use, we do a *separate*
  chain_pre_hooks(*hooks) — Run multiple pre_model_hooks left→right.
  chain_post_hooks(*hooks) — Run multiple post_model_hooks left→right.
  as_post_runnable(hook) — Wrap a (state, response) -> response function into a Runnable that accepts
- Assignments (91): CONTEXT_HEADROOM, DEFAULT_TOOLING_GUIDELINES, LOCAL_LLM_MAX_CONTEXT, MAX_CONTEXT, MAX_TOOL_TURNS, Post2Arg, PostHook, PreHook, USE_MANUAL_SCHEMA_BINDING, USE_STRICT_JSON_SCHEMA_FINAL_HOP, _JSON_ARRAY_RE, _TOOL_BLOCK_RE, _TOOL_CALL_TAG_RE, _args, _body, _id, _id_match, _name, analyst_prompt_template_initial, analyst_prompt_template_main, ann, args, arr, blocks, calls, collected, content, content_json, content_str, data, data_cleaner_prompt_template, desc, end_index, end_match, enum_str, enum_vals, final_llm, final_text, final_text_msg, finalize_msg, found, info, json_pattern, k, l_status, last, last_id, last_msg, lines, m, matches, messages, model_kwargs, msgs, n, new_last, new_msgs, ngrok_url, norm, obj_match, origin, out, parsed, raw, report_generator_prompt_template, req, resp, response, result, schema, schema_fmt_str, start_idx, start_index, start_match, state, stops, structured, sys_guard, t_id, text, tool_calls, tool_name, tool_status, tool_status_str, tp, trailing, type_str, v, visualization_prompt_template, viz_evaluator_prompt_template, wrapped
- Notable calls (92 unique; showing first 20, 72 omitted): AIMessage, Model.model_fields.items, RemoveMessage, RunnableLambda, RuntimeError, SystemMessage, _JSON_ARRAY_RE.findall, _JSON_ARRAY_RE.search, _TOOL_BLOCK_RE.findall, _TOOL_CALL_TAG_RE.findall, _collect_trailing_tool_messages, _enum_values, _final_llm_for_model, _id_match.group, _is_required, _parse_inline_tool_calls, _safe_json_loads, _type_name, agent.invoke, all
- Preview:
  # Global toggles
  USE_STRICT_JSON_SCHEMA_FINAL_HOP = use_local_llm     # gate: True = strict JSON-Schema final hop; False = your current path
  USE_MANUAL_SCHEMA_BINDING = False           # fallback if your LC build doesn’t honor method="json_schema"
  #run your llama.cpp server
  # ./llama-server -m /mnt/d/agent_models/Qwen3-4B-Function-Calling-Pro.gguf -c 65536 --n-gpu-layers -1 --host 0.0.0.0 --mlock /
  # --no-context-shift --jinja --chat-template-file ./qwen3chat_template.tmpl --reasoning-budget -1 --verbose /
  # --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 0.5 --repeat-penalty 1.05 /
  # --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 32768 --keep -1 /


## Cell 42: # --- LLM ---
- Type: code
- Line count: 35
- Imports (0): None
- Classes: none
- Functions: none
- Assignments (25): analyst_llm, big_picture_llm, complex_summary_llm, critical_complex_summary_llm, data_cleaner_llm, file_writer_llm, initial_analyst_llm, low_reasoning_llm, memsearch_query_llm, mid_substep_llm, plan_llm, progress_llm, quick_summary_llm, replan_llm, reply_llm, report_orchestrator_llm, report_packager_llm, report_section_worker_llm, router_llm, small_detail_llm, summary_llm, todo_llm, visualization_orchestrator_llm, viz_evaluator_llm, viz_worker_llm
- Notable calls (1 unique): ChatOpenAI
- Preview:
  # --- LLM ---
  big_picture_llm = ChatOpenAI(model="gpt-5-mini",use_responses_api=True, api_key=oai_key,output_version="responses/v1",reasoning={'effort': 'high'}, model_kwargs={'text': {'verbosity': 'low'}}) if not use_local_llm else ChatOpenAI(base_url=f"{ngrok_url}/v1",api_key="[REDACTED_LOCAL_API_KEY]",  model="qwen3-4b-local",temperature=0.5,extra_body={"top_p": 0.95,"repeat_penalty": 0.5,})
  router_llm = ChatOpenAI(model="gpt-5-mini",use_responses_api=True, api_key=oai_key,output_version="responses/v1",reasoning={'effort': 'medium'}, model_kwargs={'text': {'verbosity': 'low'}}) if not use_local_llm else ChatOpenAI(base_url=f"{ngrok_url}/v1",api_key="[REDACTED_LOCAL_API_KEY]",  model="qwen3-4b-local",temperature=0.5,extra_body={"top_p": 0.95,"repeat_penalty": 0.5,})
  reply_llm = ChatOpenAI(model="gpt-5-nano",use_responses_api=True, api_key=oai_key,output_version="responses/v1",reasoning={'effort': 'medium'}, model_kwargs={'text': {'verbosity': 'low'}}) if not use_local_llm else ChatOpenAI(base_url=f"{ngrok_url}/v1",api_key="[REDACTED_LOCAL_API_KEY]",  model="qwen3-4b-local",temperature=0.5,extra_body={"top_p": 0.95,"repeat_penalty": 0.5,})
  plan_llm = ChatOpenAI(model="gpt-5-mini",use_responses_api=True, api_key=oai_key,output_version="responses/v1",reasoning={'effort': 'high'}, model_kwargs={'text': {'verbosity': 'medium'}}) if not use_local_llm else ChatOpenAI(base_url=f"{ngrok_url}/v1",api_key="[REDACTED_LOCAL_API_KEY]",  model="qwen3-4b-local",temperature=0.5,extra_body={"top_p": 0.95,"repeat_penalty": 0.5,})
  replan_llm = ChatOpenAI(model="gpt-5-mini",use_responses_api=True, api_key=oai_key,output_version="responses/v1",reasoning={'effort': 'medium'}, model_kwargs={'text': {'verbosity': 'low'}}) if not use_local_llm else ChatOpenAI(base_url=f"{ngrok_url}/v1",api_key="[REDACTED_LOCAL_API_KEY]",  model="qwen3-4b-local",temperature=0.5,extra_body={"top_p": 0.95,"repeat_penalty": 0.5,})
  todo_llm = ChatOpenAI(model="gpt-5-mini",use_responses_api=True, api_key=oai_key,output_version="responses/v1",reasoning={'effort': 'medium'}, model_kwargs={'text': {'verbosity': 'low'}}) if not use_local_llm else ChatOpenAI(base_url=f"{ngrok_url}/v1",api_key="[REDACTED_LOCAL_API_KEY]",  model="qwen3-4b-local",temperature=0.5,extra_body={"top_p": 0.95,"repeat_penalty": 0.5,})
  progress_llm = ChatOpenAI(model="gpt-5-mini",use_responses_api=True, api_key=oai_key,output_version="responses/v1",reasoning={'effort': 'low'}, model_kwargs={'text': {'verbosity': 'high'}}) if not use_local_llm else ChatOpenAI(base_url=f"{ngrok_url}/v1",api_key="[REDACTED_LOCAL_API_KEY]",  model="qwen3-4b-local",temperature=0.5,extra_body={"top_p": 0.95,"repeat_penalty": 0.5,})


## Cell 43: # "conversation",
- Type: code
- Line count: 52
- Imports (0): None
- Classes: none
- Functions:
  _dedupe_tools(tools)
- Assignments (8): analyst_tools, data_cleaning_tools, init_analyst_tools, name, out, report_generator_tools, seen, visualization_tools
- Notable calls (14 unique): _dedupe_tools, analyst_tools.append, create_manage_memory_tool, create_search_memory_tool, data_cleaning_tools.append, file_writer_tools.append, getattr, init_analyst_tools.append, out.append, report_generator_tools.append, repr, seen.add, set, visualization_tools.append
- Preview:

  # "conversation",
  # "analysis",
  # "progress",
  # "routes",
  # "replies",
  # "plans",
  # "todos",


## Cell 44: # --- pre_model_hook_summarize_then_trim.py ---
- Type: code
- Line count: 725
- Imports (3): langchain_core.messages.utils.count_tokens_approximately, langmem.short_term.RunningSummary, langmem.short_term.summarize_messages
- Classes: none
- Functions:
  _msg_has_tool_invocation(msg) — Detect assistant tool invocation across common encodings.
  _extract_message_id(m)
  _new_id(prefix)
  _with_id(msg, mid) — Attach an id to a message, using constructor if supported; fallback to setattr.
  _rebuild_plain_message(m, prefix) — Return a message of the same role with plain text content AND a stable id.
  _make_state_snapshot_message(text)
  strip_tools_for_summary_hardened(msgs) — Hardened version of your strip_tools_for_summary:
  _count_tokens(msgs)
  _find_indices(msgs) — Return (first_system_idx, last_idx, last_tool_idx).
  _summarize_span(msgs, idxs_to_summarize, *, summary_llm, max_summary_tokens)
  _retrim_with_current_indices(messages, *, protected_tool_idxs, protected_ai_idxs, budget_tokens)
  _is_role(m, role)
  _last_k_ai_indices(messages, k)
  _last_supervisor_index(messages)
  choose_protected_ai_indices(messages, recency_k_ai, include_supervisor)
  _last_k_tool_indices(messages, k)
  _extract_refs_from_text(text)
  _referenced_tool_indices(messages) — Find tools referenced by the most recent non-tool message (AI or Human).
  _no_summary_tools(messages)
  _token_cost(encounter, idxs)
  choose_protected_tool_indices(messages, recency_k, model_budget_tokens, cap_fraction, min_recency)
  _shrink_protection_windows_if_needed(messages, *, protected_tool_idxs, protected_ai_idxs, model_budget_tokens, tools_cap_fraction, ai_cap_fraction, min_tool_recency, min_ai_recency, recency_k_tools, recency_k_ai) — If protected sets consume too many tokens, shrink their recency windows (not below mins).
  _state_snapshot_collapse(messages, *, summarizer, keep_sys_idx, keep_last_idx, summary_tokens) — Replace the middle span with a single compact 'state_snapshot' message.
  _hard_trim_oldest_non_human(msgs, *, protect_sys_idx, protect_last_idx, protected_tool_idxs, protected_ai_idxs, budget_tokens)
  make_pre_model_hook(summary_llm, model_budget_tokens, max_summary_tokens) — Returns a pre_model_hook(state) callable you can pass to create_react_agent(..., pre_model_hook=...).
- Assignments (93): TARGET_BUDGET, _, _CALLID_RE, _DF_ID_RE, _PATH_RE, addl, ai_cap, ai_cost, ak, all_tools, base, c, callid, calls_txt, cap_tokens, clean_segment, cls, collapsed, content, content_str, cur_ai, cur_tools, df_id, fc, hits, i, idxs, idxs_to_summarize, keep, keep_set, last_i, last_idx, last_idx2, last_idx3, last_idx4, last_text, last_tool_idx, last_tool_idx2, left, m, messages, meta, mid, mid_segment, msg, n, new, newer_ai, newer_set, obj, out, path, prefix, preleft, preright, protected, protected_ai, protected_ai_idxs, protected_cost, protected_tool_idxs, protected_tools, r, recent, refs, res, right, rm, role, segment_base, sl, snap_idx, snapshot, snapshot_text, span_end, span_start, start, summarized_msgs, summarized_segment, summarizer, sup, sys_i, sys_idx, sys_idx2, sys_idx3, sys_idx4, tc, tc1, tool_cap, tool_cost, tool_idxs, trimmed_msgs, truncated, working
- Notable calls (78 unique; showing first 20, 58 omitted): AIMessage, ToolMessage, _CALLID_RE.findall, _DF_ID_RE.findall, _PATH_RE.findall, _construct, _count_tokens, _extract_message_id, _extract_refs_from_text, _find_indices, _hard_trim_oldest_non_human, _is_role, _last_k_ai_indices, _last_k_tool_indices, _last_supervisor_index, _make_state_snapshot_message, _msg_has_tool_invocation, _new_id, _no_summary_tools, _pointerize_all_but_last_tool_safe
- Preview:
  # --- pre_model_hook_summarize_then_trim.py ---

  from langchain_core.messages.utils import count_tokens_approximately
  from langmem.short_term import summarize_messages, RunningSummary  # pip install langmem

  TARGET_BUDGET = 250_000  # leave padding vs ~276k ctx window you mentioned




## Cell 45: if use_local_llm:
- Type: code
- Line count: 2531
- Imports (3): pydantic.BaseModel, pydantic.Field, typing.Literal
- Classes: none
- Functions:
  create_data_cleaner_agent(initial_description, df_ids)
  create_initial_analysis_agent(user_prompt, df_ids)
  create_analyst_agent(initial_description, df_ids)
  create_file_writer_agent(df_ids)
  create_visualization_agent(df_ids)
  create_viz_evaluator_agent()
  create_report_generator_agent(df_ids, rg_agent_task)
  update_memory(state, config, *, memstore)
  make_supervisor_node(supervisor_llms, members, user_prompt)
- Assignments (262): Key, PROGRESS_ACCOUNTING_STR, PlanStepIdentity, _count, _post_model_hook, _prehook, _word_re, a_np, ab, agent_output_map, agent_outputs_objs, agent_rq_msgs, all_flags, allowed_anyof, b_np, base, base_agent, base_replan_prompt, base_todo_prompt, best_versions, by_index, by_number, checkpointer, chosen, complete_map, completed_agents, completion_order, conv_resp, conv_routing_llm, conversation_result, corresponding_agent_msg, cos, cos_bucket, cos_thresholds, cps, critical_flags, cross_pairs, cross_score, cst_llm, cst_llmb, cst_schema, curr_plan, d_unit, decisive_hits, dedup_done, dedup_plan_steps, den, desc_len, desc_score, details, didcomplete, dim, done_descs, done_names, done_nums, done_steps, done_tasks, dot_bucket, dot_raw, dot_thresholds, downcount, e_bins, e_dd_a, e_dd_b, e_nn_a, e_nn_b, emergency_flags, euclid_bucket, euclid_raw, euclid_unit, euclid_unit_thresholds, existing_step, final_base_list, final_base_str, final_turn_msgs_list, fused, fused_lex, goto, init_analyst_vars, init_dc_vars, init_df_id_str, init_fw_vars, init_ia_vars, init_rg_vars, init_vis_vars, init_viz_vars, k, key, l1_scaled_similarity, l1s, l1s_thresholds, last_agent_expects_reply, last_agent_finished, last_agent_id, last_agent_prompt, last_agent_reply_msg, last_count, last_known, last_message_text, last_output_obj, latest_message, latest_progress, left, lex, lex_cross, lex_desc, lex_name, lm_name, manhattan_bucket, manhattan_raw, manhattan_unit, manhattan_unit_max, map_key, map_list, match, match_sync, max_done_plus1, memory_id, mems, message_history, metric_weights, msg, n, na, name_len, name_or_desc_hit, name_or_desc_ok, name_score, namespace, nap, nb, needs_recreated_flags, needs_replies, new_cst_schema, new_done_steps, new_messages, new_plan, new_plan_steps, next, next_agent_metadata, next_agent_prompt, num, num_ok, num_window_hit, options, out, output_format, output_format_map, p_found, pair_weights, plan_prompt_key, plan_steps, plan_supervisor_expects_reply, plan_txt, planning_llm, planning_supervisor_llm, prehook, prehook_complex, prehook_critical_complex, prehook_quick, prev_plan, priority_sorted_reply_keys, progress, progress_account_str, progress_account_str_b, progress_llm, progress_llm_b, progress_llm_conv, progress_prompt, progress_prompt_b, progress_prompt_c, progress_report, progress_result, progress_result_conv, progress_resultb, progress_str, progress_supervisor_expects_reply, progress_vars, progress_varsb, progress_varsc, prompt, prompt_for_planning, pstep_c, pv, pw_cross, pw_desc, pw_name, remaining_agents, rendered_new_plan_prompt, rendered_progress_prompt, rendered_progress_promptb, rendered_progress_promptc, rendered_reply_prompt, rendered_routing_prompt, rendered_sp_routing_prompt, rendered_todo_prompt, rep, replace_step, replaced, replan_vars, replies_map_bools, replies_order, reply_ctx_str, reply_msgs, reply_obj, reply_objs, reply_prompt, reply_result, reply_str_map, replying_supervisor_llm, report_task, result, right, rng, routing, routing_llm, routing_state_vars, routing_supervisor_expects_reply, sa, sb, score_, scores_for_fusion, se, second_supervsr_prompt_str, secondary_transition_map, seen, seen_nums, seq, sim_desc, sim_name, special_reroute_str, supervisor_msgs, supervisor_prompt, supervisor_replies, sv_roles, system_prompt, task_fin_str_map, task_one_desc, task_two_desc, temp_sorted, temp_sorted_list, this_last_agent_finished, this_last_agent_id, this_last_agent_reply_msg, this_nap, todo_list, todo_llm, todo_results, todo_supervisor_expects_reply, todo_vars, tool_descrips_mini, tool_descriptions, ua, ub, updated_progress_prompt, updated_progress_promptb, updated_progress_promptc, updated_replan_prompt, updated_todo_prompt, user_id, user_prompt, weights_for_fusion
- Notable calls (206 unique; showing first 20, 186 omitted): AIMessage, AnalysisInsights.model_json_schema, ChatPromptTemplate.from_messages, CleaningMetadata.model_json_schema, CompletedStepsAndTasks, CompletedStepsAndTasks.model_json_schema, CompletedStepsAndTasks.model_validate, CompletedStepsAndTasks.model_validate_json, ConversationalResponse.model_json_schema, ConversationalResponse.model_validate, DataVisualization.model_json_schema, Field, FileResult.model_json_schema, HumanMessage, InMemorySaver, InitialDescription.model_json_schema, MessagesPlaceholder, PROGRESS_ACCOUNTING_STR.format, Plan, Plan.model_json_schema
- Preview:


  if use_local_llm:
      print("Using local LLM")


  prehook = make_pre_model_hook(summary_llm, model_budget_tokens=200_000, max_summary_tokens=2048) if not use_local_llm else make_pre_model_hook(summary_llm, model_budget_tokens=40000, max_summary_tokens=4096)
  prehook_quick = make_pre_model_hook(quick_summary_llm, model_budget_tokens=180_000, max_summary_tokens=512) if not use_local_llm else make_pre_model_hook(quick_summary_llm, model_budget_tokens=32000, max_summary_tokens=1024)


## Cell 46: # 📂 Sample Dataset Loading and Registration
- Type: markdown
- Preview:
  # 📂 Sample Dataset Loading and Registration


## Cell 47: # Download & prepare sample dataset from KaggleHub (robust)
- Type: code
- Line count: 79
- Imports (1): glob
- Classes: none
- Functions: none
- Assignments (21): analyst_agent, chosen, csv_candidates, data_cleaner_agent, df, df_id, df_name, file_writer_agent, initial_analysis_agent, initial_description, load_errors, path, preferred, raw_path_str, report_generator_agent, report_packager_agent, report_section_agent, sample_prompt_text, sample_prompt_tuple, visualization_agent, viz_evaluator_agent
- Notable calls (27 unique; showing first 20, 7 omitted): FileNotFoundError, InitialDescription, PathlibPath, RuntimeError, create_analyst_agent, create_data_cleaner_agent, create_file_writer_agent, create_initial_analysis_agent, create_report_generator_agent, create_visualization_agent, create_viz_evaluator_agent, df.head, dict, glob.glob, global_df_registry.register_dataframe, kagglehub.dataset_download, load_errors.append, max, os.path.join, pd.read_csv
- Preview:
  # Download & prepare sample dataset from KaggleHub (robust)
  # Assumes: pprint, os, pandas as pd, kagglehub, global_df_registry,
  #          InitialDescription, and the agent factory fns are imported.

  import glob

  # Download (cached by kagglehub if already present)
  path = kagglehub.dataset_download("datafiniti/consumer-reviews-of-amazon-products")


## Cell 48: Automated dataset acquisition and registration:
- Type: markdown
- Preview:
  Automated dataset acquisition and registration:
  - **KaggleHub Integration**: Downloads sample dataset from Kaggle
  - **Data Registration**: Automatic registration in the DataFrame registry
  - **Initial Analysis**: Basic dataset inspection and metadata extraction
  - **Path Management**: Robust file handling and path resolution


## Cell 49: # ⚙️ Runtime Context and Configuration
- Type: markdown
- Preview:
  # ⚙️ Runtime Context and Configuration


## Cell 50: import uuid
- Type: code
- Line count: 73
- Imports (5): dataclasses.dataclass, datetime.UTC, datetime.datetime, os, uuid
- Classes:
  RuntimeCtx (methods: none)
- Functions: none
- Assignments (27): RUNTIME, _config, _config_obj, _default_cfg, analyst_agent, artifacts, artifacts_dir, base_dir, data, data_cleaner_agent, data_dir, file_writer_agent, initial_analysis_agent, logs, logs_dir, report_generator_agent, reports, reports_dir, run_id, runtime_toolkit, sample_prompt_final_human, sample_prompt_text, sample_prompt_tuple, visualization_agent, viz, viz_dir, viz_evaluator_agent
- Notable calls (11 unique): FileManagementToolkit, HumanMessage, PathlibPath, RunnableConfig, RuntimeCtx, dataclass, datetime.now, p.mkdir, str, strftime, uuid.uuid4
- Preview:
  import uuid

  sample_prompt_text = f"Please analyze the dataset named {df_name}. You have tools available to you for accessing the data using the following str as the df_id parameter: `{df_id}`. A full analysis will be performed on the dataset. Then, relevant and meaningful visualizations will need to be chosen and produced with the data, after which a full report will be generated in several formats, including PDF, Markdown, and HTML."

  sample_prompt_final_human = HumanMessage(content=sample_prompt_text, name="user") # Ensure it's a HumanMessage
  sample_prompt_tuple = ("user", sample_prompt_text)
  # --- runtime_ctx.py (put this near your imports or in a small cell) ---
  from dataclasses import dataclass


## Cell 51: Runtime configuration and context management:
- Type: markdown
- Preview:
  Runtime configuration and context management:
  - **Working Directories**: Setup of output directories for reports and visualizations
  - **Sample Prompts**: Default user prompts for testing and demonstration
  - **UUID Generation**: Unique identifiers for tracking analysis sessions
  - **Configuration Objects**: Runtime context for workflow execution


## Cell 52: # 📋 Report Generation Utilities and Packaging
- Type: markdown
- Preview:
  # 📋 Report Generation Utilities and Packaging


## Cell 53: # report_packager_node helpers
- Type: code
- Line count: 408
- Imports (7): base64, hashlib, mimetypes, os, tempfile, textwrap, uuid
- Classes: none
- Functions:
  _sha256_bytes(data)
  _detect_mime_and_encoding(path, default_mime)
  _atomic_write_bytes(p, data)
  _atomic_write_text(p, txt, encoding)
  _write_bytes(p, data)
  _write_text(p, txt)
  _materialize_images(viz_artifacts, out_dir) — Return {fig_name: file_path}. Converts base64 to PNG files (or writes bytes).
  _render_markdown(sections, fig_paths, meta) — Assemble a clean Markdown report.
  _markdown_to_html(md)
  _ensure_list_str(x)
  _safe_copy(src, dst, mode)
  _resolve_artifacts_root(state, *, into, artifacts_key, run_id_key) — Decide artifacts root and run_id without mutating the input `state`.
  save_viz_for_state(state, viz_results, *, into, artifacts_key, run_id_key, copy_mode, make_relative) — Normalize & persist visualization files (VisualizationResults), then return a
  _manifest_from_path(p)
  _next_version_path(p)
- Assignments (48): ALIASES, WORK_DIR, artifacts_root, base, body_parts, candidate, data, dest, dest_dir, digest, enc, errors, existing_root, first_data_viz, header, i, img_path, mapping, md, merged_results, mime, name, normalized, num_viz, out, pr, prev_paths, prev_results, prev_viz_results_list, root, root_updates, run_id, saved_paths, saved_visualizations, src, stem, stored_path, suffix, title, tmp_name, update, update_bits, vis_id, viz_, viz_list_as_dict, viz_results, viz_results_b, viz_resultsb
- Notable calls (73 unique; showing first 20, 53 omitted): ALIASES.get, DataVisualization, DataVisualization.model_fields.items, DataVisualization.model_validate, PathlibPath, PathlibPath.cwd, VisualizationResults, VisualizationResults.model_validate, _atomic_write_bytes, _atomic_write_text, _detect_mime_and_encoding, _ensure_list_str, _resolve_artifacts_root, _safe_copy, _sha256_bytes, _write_bytes, all, art.get, base64.b64decode, body_parts.append
- Preview:
  # report_packager_node helpers
  import textwrap

  WORK_DIR = WORKING_DIRECTORY  # you already set this globally

  # --- Add near top of the helpers cell (report_packager_node helpers) ---
  import tempfile, hashlib, mimetypes



## Cell 54: Helper functions for report generation and file management:
- Type: markdown
- Preview:
  Helper functions for report generation and file management:
  - **File Writing Utilities**: Safe file operations with error handling
  - **Report Packaging**: Multi-format report generation (HTML, Markdown, PDF)
  - **Template Management**: Report template processing and customization
  - **Output Organization**: Structured file output with proper naming conventions


## Cell 55: # **🤖 Agent Node Implementation and Workflow Logic**
- Type: markdown
- Preview:
  # **🤖 Agent Node Implementation and Workflow Logic**


## Cell 56: # Node Functions (revised)
- Type: code
- Line count: 1794
- Imports (0): None
- Classes: none
- Functions:
  initial_analysis_node(state)
  data_cleaner_node(state)
  analyst_node(state)
  _normalize_meta(meta_obj)
  file_writer_node(state)
  _guess_viz_type(name_or_desc)
  _norm_title(s)
  _normalize_viz_spec(raw, *, default_df_id, fallback_title) — Return a clean dict spec with required keys and safe defaults.
  visualization_orchestrator(state) — Prepare `viz_tasks` and `viz_specs` for fan-out.
  viz_worker(state)
  assign_viz_workers(state)
  viz_join(state)
  viz_evaluator_node(state)
  route_viz(state)
  report_orchestrator(state)
  section_worker(state)
  dispatch_sections(state) — Emit Send events to run one section_worker per section in the outline.
  assign_section_workers(state)
  report_join(state)
  report_packager_node(state)
  emergency_correspondence_node(state)
- Assignments (143): ALIASES, ALLOWED, ALLOWED_SPEC_KEYS, MANDATORY_SPEC_KEYS, _all_viz, _df0, _msgs, all_viz, analyst_vars, available, base_prompt, candidate, cleaning, cleaning_metadata, cm, completed_str, content, dc_vars, default_content_str, default_df_id, default_instruction, default_instruction_b, default_instruction_supervisor, df_id, df_id0, df_id_str, draft, emer_msg_txt, emerg_msg, expect_reply, expected_viz, expected_viz_str, expects_reply, fb, feedback, file_name, file_results, file_type, file_type_str, filetype_str_lst, fin_str, final_grade, final_msgs, final_report_path, final_report_str, finished_this_task, fit_last_obj, fw_vars, global_df_registry, grade, html_exists, ia_vars, initial_description, insights, invoke_state, is_final, last_known, main_emer_msg, markdown_exists, memory_text, mems, messages, meta, missing, msg, msg_key, msg_obj, msg_obj_candidates, msg_text, msgs_tmp, n, newest_msg, norm_specs, nxt, obj, orig_agent_msg, outline, outline_response, output_format, parsed, parts, payload, pdf_exists, plan_preview, pr, progress_key, progress_reports, r, raw_spec, recs, registry, rendered, reply_msg_to_supervisor, report, report_paths, result, result_spec_map, result_task_map, results, resultsa, rg_vars, rr, s, sample, secs, section, section_text, sections, sends, spec, spec_result_map, spec_task_map, specs, spvsr_to_agent_msgs, sr, steps_str, store, structured, summary_lines, supervisor_message, system_message_content, target, task, task_result_map, task_spec_map, task_vizid, tasks, title, tmp_basemsgs, tool_descriptions, topic, update, user_prompt, user_t, v, vis_vars, viz, viz_paths, viz_specs, viz_tasks, vr_results, warnings, written_sections
- Notable calls (140 unique; showing first 20, 120 omitted): AIMessage, AnalysisInsights.model_json_schema, CLASS_TO_AGENT.items, ChatPromptTemplate.from_messages, CleaningMetadata.model_json_schema, Command, DataVisualization, DataVisualization.model_fields.items, DataVisualization.model_json_schema, DataVisualization.model_validate, FileResult.model_json_schema, HumanMessage, InitialDescription, InitialDescription.model_json_schema, ListOfFiles.model_json_schema, MessagesPlaceholder, NextAgentMetadata, ReportOutline.model_json_schema, ReportResults, ReportResults.model_json_schema
- Preview:
  # Node Functions (revised)


  def initial_analysis_node(state: State):
      user_prompt = state.get("user_prompt", sample_prompt_text)

      global_df_registry = get_global_df_registry()
      initial_description = state.get("initial_description") or InitialDescription(dataset_description="No description yet", data_sample="No sample available",notes="None yet", expect_reply=False, reply_msg_to_supervisor="No reply yet", finished_this_task=False)


## Cell 57: Core agent node implementations for the multi-agent workflow:
- Type: markdown
- Preview:
  Core agent node implementations for the multi-agent workflow:
  - **Initial Analysis Node**: Dataset inspection and metadata extraction
  - **Data Cleaner Node**: Automated data cleaning and preprocessing
  - **Analyst Node**: Statistical analysis and pattern detection
  - **Visualization Node**: Chart and graph generation
  - **Report Generator Node**: Comprehensive report compilation


## Cell 58: # 🌐 Workflow Graph Compilation and Configuration
- Type: markdown
- Preview:
  # 🌐 Workflow Graph Compilation and Configuration


## Cell 59: #Graph compile (revised)
- Type: code
- Line count: 199
- Imports (0): None
- Classes: none
- Functions:
  write_output_to_file(state, config)
  route_to_writer(state)
  route_from_supervisor(state)
- Assignments (11): allowed, already_wrote, checkpointer, coordinator_node, data_analysis_team_builder, data_detective_graph, finished_secs_count, nxt, report_done, report_outline_secs_count, report_ready
- Notable calls (15 unique): CachePolicy, Command, HumanMessage, InMemoryCache, MemorySaver, StateGraph, bool, data_analysis_team_builder.add_conditional_edges, data_analysis_team_builder.add_edge, data_analysis_team_builder.add_node, data_analysis_team_builder.compile, isinstance, len, make_supervisor_node, state.get
- Preview:
   #Graph compile (revised)

  coordinator_node = make_supervisor_node(
      [big_picture_llm,router_llm, reply_llm, plan_llm, replan_llm, progress_llm, todo_llm,low_reasoning_llm],
      ["initial_analysis", "data_cleaner", "analyst", "file_writer", "visualization", "report_orchestrator"],
      sample_prompt_text,
  )



## Cell 60: LangGraph workflow compilation and supervisor integration:
- Type: markdown
- Preview:
  LangGraph workflow compilation and supervisor integration:
  - **Multi-LLM Supervisor**: Advanced coordinator with specialized sub-models
  - **Graph Construction**: Complete workflow graph with all nodes and edges
  - **Parallel Processing**: Support for concurrent analysis operations
  - **Error Recovery**: Graceful handling of node failures and timeouts
  - **Checkpointing**: Workflow state persistence and recovery capabilities


## Cell 61: # 📊 Workflow Graph Visualization
- Type: markdown
- Preview:
  # 📊 Workflow Graph Visualization


## Cell 62: try:
- Type: code
- Line count: 5
- Imports (0): None
- Classes: none
- Functions: none
- Notable calls (5 unique): Image, data_detective_graph.get_graph, display, draw_mermaid_png, print
- Preview:
  try:
      display(Image(data_detective_graph.get_graph().draw_mermaid_png(max_retries=2, retry_delay=2.0)))
  except Exception as e:
      print(f"Error drawing graph: {e}")
      pass


## Cell 63: Visual representation of the compiled workflow graph:
- Type: markdown
- Preview:
  Visual representation of the compiled workflow graph:
  - **Mermaid Diagram**: Interactive workflow visualization
  - **Node Relationships**: Clear display of agent interactions and data flow
  - **Debugging Aid**: Visual debugging tool for workflow understanding


## Cell 64: # ✅ Schema Validation and Testing
- Type: markdown
- Preview:
  # ✅ Schema Validation and Testing


## Cell 65: print(InitialDescription.model_json_schema())
- Type: code
- Line count: 6
- Imports (0): None
- Classes: none
- Functions: none
- Assignments (1): initial_test
- Notable calls (5 unique): InitialDescription, InitialDescription.model_json_schema, initial_analysis_agent.get_output_schema, initial_test.model_dump_json, print
- Preview:
  print(InitialDescription.model_json_schema())
  initial_test = InitialDescription(dataset_description="test", data_sample="test", notes="test notes", reply_msg_to_supervisor="test", finished_this_task=False, expect_reply=False)
  print(initial_test.model_dump_json())
  print(initial_analysis_agent.get_output_schema())
  print(big_picture_llm.get_output_schema)
  # print(initial_analysis_agent.invoke(


## Cell 66: Validation of data models and schema compliance:
- Type: markdown
- Preview:
  Validation of data models and schema compliance:
  - **Pydantic Schema Validation**: Ensures proper model structure
  - **JSON Schema Generation**: Validates serialization/deserialization
  - **Type Safety Testing**: Confirms type annotations and constraints


## Cell 67: # 🔍 Advanced Debugging and Introspection Tools
- Type: markdown
- Preview:
  # 🔍 Advanced Debugging and Introspection Tools


## Cell 68: #These are only helpers for accessing or checking keys nested within variable iterables - do not worry about or focus on these, they are non-critical print helpers
- Type: code
- Line count: 125
- Imports (11): collections.abc.Mapping, collections.abc.Sequence, langchain_core.messages.ToolMessage, typing.Any, typing.Iterable, typing.Literal, typing.Optional, typing.Set, typing.Tuple, typing.TypeAlias, typing.Union
- Classes: none
- Functions:
  find_key_paths(obj, target_key, *, to_value) — Yield paths to each occurrence of `target_key` inside any dict at any depth.
  find_key_paths_list(obj, target_key, *, to_value) — Materialize all paths into a list (tiny convenience wrapper).
  get_by_path(obj, path, *, just_value) — Follow a path (as emitted by find_key_paths) and return:
  pick_tool_messages(messages)
  extract_handles_from_tools(messages, durable_keys)
- Assignments (16): ARTIFACT_TOOLS, DURABLE_KEYS, Path, PathStep, all_e_values, containers, cur, data, dct, handles, is_last, keep, kept, payload, value_paths, values
- Notable calls (17 unique): KeyError, ValueError, _walk, append, enumerate, find_key_paths, find_key_paths_list, get_by_path, getattr, handles.setdefault, isinstance, keep.append, kept.append, len, list, print, x.items
- Preview:
  #These are only helpers for accessing or checking keys nested within variable iterables - do not worry about or focus on these, they are non-critical print helpers

  from collections.abc import Mapping, Sequence
  from typing import Iterable, Tuple, Union, Any, TypeAlias, Literal, Optional, Set

  PathStep:TypeAlias = Tuple[str, Any]   # ('key', k) | ('idx', i) | ('item', v)
  Path:TypeAlias = Tuple[PathStep, ...]



## Cell 69: Sophisticated debugging utilities for complex data structures:
- Type: markdown
- Preview:
  Sophisticated debugging utilities for complex data structures:
  - **Path Finding**: Navigate nested data structures and find specific keys/values
  - **Deep Inspection**: Analyze complex nested objects and state structures
  - **Type Analysis**: Runtime type checking and validation
  - **Search Utilities**: Locate specific data within large state objects


## Cell 70: # 🚀 Streaming Workflow Execution and Real-time Processing
- Type: markdown
- Preview:
  # 🚀 Streaming Workflow Execution and Real-time Processing


## Cell 71: # Streaming run (clean + robust)
- Type: code
- Line count: 212
- Imports (4): langchain_core.messages.HumanMessage, langchain_core.runnables.config.RunnableConfig, traceback, uuid
- Classes: none
- Functions: none
- Assignments (11): RUNTIME, current_step, empty_message_count, initial_state, most_recent_label, previous_name, received_steps, run_config, runtime_fields, thread_id, user_id_str
- Notable calls (7 unique): HumanMessage, RUNTIME.artifacts_dir.resolve, RunnableConfig, RuntimeCtx, pprint, print, uuid.uuid4
- Preview:
  # Streaming run (clean + robust)

  from langchain_core.runnables.config import RunnableConfig
  from langchain_core.messages import HumanMessage
  import uuid
  import traceback

  # print langchain_openai version for debugging


## Cell 72: Main execution engine for the data analysis workflow:
- Type: markdown
- Preview:
  Main execution engine for the data analysis workflow:
  - **Stream Processing**: Real-time execution with live updates
  - **Progress Monitoring**: Track workflow progress and intermediate results
  - **Error Handling**: Robust error recovery and graceful degradation
  - **Result Streaming**: Live display of analysis results as they're generated


## Cell 73: (empty markdown cell)
- Type: code
- Line count: 0
- Imports (0): None
- Classes: none
- Functions: none
- Preview:
  (empty code cell)


## Cell 74: # 📡 Extended Streaming Utilities and Text Processing
- Type: markdown
- Preview:
  # 📡 Extended Streaming Utilities and Text Processing


## Cell 75: from IPython.display import HTML, display
- Type: code
- Line count: 665
- Parsing note: Mixed indentation and notebook-format HTML injections prevent reliable AST parsing; retained raw preview for reference.
- Preview:
  from IPython.display import HTML, display
  display(HTML("""
  <style>
  /* Wrap anything printed into output areas (Colab + Jupyter) */
  .output_subarea pre, .output-area pre, .output pre, div.rich pre, pre {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;


## Cell 76: collected = {"test":{"langgraph_step": 0, "msg": SystemMessage(content="test message")}}
- Type: code
- Line count: 44
- Imports (0): None
- Classes: none
- Functions: none
- Assignments (5): collected, l_msg, lstep, rstep, unparsed_steps
- Notable calls (11 unique): SystemMessage, collected.items, details.items, get, getattr, int, isinstance, pprint, print, sorted, unparsed_steps.append
- Preview:
  collected = {"test":{"langgraph_step": 0, "msg": SystemMessage(content="test message")}}
  unparsed_steps = []
  for rstep_ in received_steps:
      rstep = rstep_[0]
      pprint(rstep_)
      # print(getattr(rstep, "id", "fart"))
      if getattr(rstep, "id", None) is not None:



## Cell 77: Advanced streaming utilities and content extraction:
- Type: markdown
- Preview:
  Advanced streaming utilities and content extraction:
  - **Text Extraction**: Extract text from various content formats and structures
  - **Content Processing**: Handle OpenAI-style content blocks and nested structures
  - **Stream Utilities**: Additional helpers for streaming operations
  - **Format Handling**: Support for multiple content types and formats


## Cell 78: # save temp dir to gdrive
- Type: code
- Line count: 62
- Imports (0): None
- Classes: none
- Functions:
  _dir_has_any_files(p) — Return True if directory contains at least one file anywhere under it.
- Assignments (17): candidate_outputs_dirs, dst, file_dst, file_results, final_dst, final_state, html_dst, html_path, md_dst, md_path, pdf_dst, pdf_path, report_results, state_vals, vis, visualization_dst, visualization_results
- Notable calls (15 unique): PathlibPath, _dir_has_any_files, all, child.is_file, data_detective_graph.get_state, get, isinstance, out_dir.exists, out_dir.is_dir, p.rglob, persist_to_drive, print, run_config.get, state_vals.get, str
- Preview:
  # save temp dir to gdrive
  final_state = data_detective_graph.get_state(run_config)
  if final_state and final_state.values:
      state_vals = final_state.values
      final_dst = persist_to_drive(WORKING_DIRECTORY, run_id = str(run_config.get("run_id") or state_vals.get("run_id") or state_vals.get("_config", {}).get("run_id", run_id)))

      if state_vals.get("final_report") is not None and isinstance(state_vals.get("final_report"), ReportResults):
          assert state_vals.get("final_report") is not None


## Cell 79: # 🔎 Final State Inspection and Results Review
- Type: markdown
- Preview:
  # 🔎 Final State Inspection and Results Review


## Cell 80: print("Figures:", list(RUNTIME.viz_dir.glob("*.png")))
- Type: code
- Line count: 90
- Imports (0): None
- Classes: none
- Functions: none
- Assignments (5): I_d, c_m, final_state, sk, state_vals
- Notable calls (17 unique): I_d.model_dump_json, RUNTIME.reports_dir.glob, RUNTIME.viz_dir.glob, _print_new_suffix_wrapped, c_m.model_dump_json, data_detective_graph.get_state, enumerate, isinstance, list, msg.text, pprint, pretty_print, print, state_vals.get, state_vals.items, str, v.model_dump_json
- Preview:
  print("Figures:", list(RUNTIME.viz_dir.glob("*.png")))
  print("Reports:", list(RUNTIME.reports_dir.glob("*.*")))
  # Inspect final state from the checkpointer (since we used MemorySaver + thread_id)
  try:
      final_state = data_detective_graph.get_state(run_config)
      if final_state and final_state.values:
          state_vals = final_state.values
          print("— Final state summary —")


## Cell 81: Comprehensive inspection of workflow results:
- Type: markdown
- Preview:
  Comprehensive inspection of workflow results:
  - **State Analysis**: Examine final workflow state and generated artifacts
  - **File Listing**: Review generated reports, visualizations, and data files
  - **Checkpointer Access**: Retrieve and analyze saved workflow checkpoints
  - **Results Summary**: Overview of completed analysis and generated outputs


## Cell 82: pprint(WORKING_DIRECTORY)
- Type: code
- Line count: 1
- Imports (0): None
- Classes: none
- Functions: none
- Notable calls (1 unique): pprint
- Preview:
  pprint(WORKING_DIRECTORY)


## Cell 83: # 🔧 Function Calling Utilities and Tool Conversion
- Type: markdown
- Preview:
  # 🔧 Function Calling Utilities and Tool Conversion


## Cell 84: from langchain_core.utils.function_calling import convert_to_openai_tool
- Type: code
- Line count: 1
- Imports (1): langchain_core.utils.function_calling.convert_to_openai_tool
- Classes: none
- Functions: none
- Preview:
  from langchain_core.utils.function_calling import convert_to_openai_tool


## Cell 85: Utilities for OpenAI function calling and tool conversion:
- Type: markdown
- Preview:
  Utilities for OpenAI function calling and tool conversion:
  - **Tool Conversion**: Convert Pydantic models to OpenAI tool format
  - **Schema Validation**: Ensure proper function calling schema compliance
  - **API Compatibility**: Support for different OpenAI API versions and formats


## Cell 86: Advanced testing and validation of data models:
- Type: markdown
- Preview:
  Advanced testing and validation of data models:
  - **Schema Comparison**: Compare different schema generation methods
  - **Strict Validation**: Test strict vs. lenient validation modes
  - **Alias Testing**: Validate field aliases and serialization options
  - **Compatibility Testing**: Ensure backward compatibility with different versions


## Cell 87: # 🎯 Final Model Validation and Quality Assurance
- Type: markdown
- Preview:
  # 🎯 Final Model Validation and Quality Assurance


## Cell 88: # initial_test = InitialDescription(dataset_description="test", data_sample="test")
- Type: code
- Line count: 10
- Imports (0): None
- Classes: none
- Functions: none
- Preview:
  # initial_test = InitialDescription(dataset_description="test", data_sample="test")
  # print(initial_test.model_dump_json())
  # print(InitialDescription.model_validate(initial_test, strict=True,from_attributes=True))
  # print("\n")
  # print(initial_test.model_json_schema().__str__())
  # print("\n")
  # # print(initial_test.model_validate(initial_test.model_json_schema(), strict=True,from_attributes=True))
  # print("\n")


## Cell 89: Final validation steps and quality assurance checks:
- Type: markdown
- Preview:
  Final validation steps and quality assurance checks:
  - **Model Compliance**: Final verification of all data models
  - **Serialization Testing**: Validate JSON serialization and deserialization
  - **Schema Output**: Generate and verify final schema documentation
  - **Quality Checks**: Comprehensive validation of the entire system


## Cell 90: # Save the InMemorySaver checkpointer to a SQL database file on disk
- Type: markdown
- Preview:
  # Save the InMemorySaver checkpointer to a SQL database file on disk


## Cell 91: import sqlite3
- Type: code
- Line count: 24
- Imports (2): langgraph.checkpoint.sqlite.SqliteSaver, sqlite3
- Classes: none
- Functions: none
- Assignments (7): cfg, dst_graph, last_writer, seq, snaps, src_graph, writes
- Notable calls (9 unique): SqliteSaver.from_conn_string, data_analysis_team_builder.compile, dst_graph.update_state, get, list, migrate_thread, reversed, src_graph.get_state_history, writes.keys
- Preview:
  import sqlite3
  from langgraph.checkpoint.sqlite import SqliteSaver
  # src graph was compiled with InMemorySaver
  src_graph = data_detective_graph

  # destination graph with SQLite persistence
  with SqliteSaver.from_conn_string("checkpoints.sqlite") as dst_cp:
    dst_graph = data_analysis_team_builder.compile(checkpointer=dst_cp)


## Cell 92: # To restore a previous checkpointer state from an SQL database file on disk
- Type: markdown
- Preview:
  # To restore a previous checkpointer state from an SQL database file on disk


## Cell 93: import sqlite3
- Type: code
- Line count: 8
- Imports (2): langgraph.checkpoint.sqlite.SqliteSaver, sqlite3
- Classes: none
- Functions: none
- Assignments (3): conn, cp, data_detective_graph
- Notable calls (4 unique): InMemoryCache, SqliteSaver, data_analysis_team_builder.compile, sqlite3.connect
- Preview:
  import sqlite3
  from langgraph.checkpoint.sqlite import SqliteSaver

  conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
  cp = SqliteSaver(conn)
  data_detective_graph = data_analysis_team_builder.compile(checkpointer=cp,    store=in_memory_store, cache=InMemoryCache())

  # conn.close() #Use conn.close() when finished


## Cell 94: print(run_config)
- Type: code
- Line count: 1
- Imports (0): None
- Classes: none
- Functions: none
- Notable calls (1 unique): print
- Preview:
  print(run_config)
