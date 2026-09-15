# IntelligentDataDetective_beta_v5 – Function and Status Report

This report documents every cell in `IntelligentDataDetective_beta_v5.ipynb`, including execution status, intents, and structural elements (functions, classes, imports, and notable behaviors). Generated automatically for comprehensive traceability.

## Cell 0 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# Intelligent Data Detective Introduction
````
- **Markdown content summary**:

# Intelligent Data Detective Introduction


## Cell 1 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 2
- **Leading content**:

````
#This variable is to enable custom llama.cpp llama-server connections for using local small models instead.
use_local_llm = False
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  - #This variable is to enable custom llama.cpp llama-server connections for using local small models instead.

## Cell 2 – Markdown
- **Execution count**: None
- **Source lines**: 0

## Cell 3 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🔧 Environment Setup and Dependency Management
````
- **Markdown content summary**:

# 🔧 Environment Setup and Dependency Management


## Cell 4 – Code
- **Execution count**: None
- **Stored outputs**: 1
- **Source lines**: 80
- **Leading content**:

````
# Import the standard library module for interacting with the operating system (env vars, paths, processes).
import os

````
- **AST parse**: failed (`invalid syntax (<unknown>, line 56)`); code likely contains magics or shell commands. Function/class inventory may be incomplete.
- **Commentary / intent hints**:
  - # Import the standard library module for interacting with the operating system (env vars, paths, processes).
  - # Import utilities to spawn and manage child processes (not used in this snippet, but common in notebooks).
  - # This commented import would load environment variables from a .env file; left disabled because keys are fetched differently below.
  - # from dotenv import load_dotenv
  - # Import an object-oriented filesystem path API and alias it for clarity in larger projects.
  - # Import helper to create context managers using the @contextmanager decorator (not used yet in this snippet).
  - # Import several modules in one line:
  - # - builtins: access to Python’s built-in functions/types (rarely needed directly)
  - # - os: duplicated from above; harmless but redundant
  - # - sys: access to interpreter-level details (argv, stdout, path, etc.)

## Cell 5 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
This section handles the initial environment setup, including:
- **Environment Detection**: Automatically detects if running in Google Colab or local environment
- **API Key Management**: Securely retrieves OpenAI and Tavily API keys from environment variables or Colab userdata
````
- **Markdown content summary**:

This section handles the initial environment setup, including:
- **Environment Detection**: Automatically detects if running in Google Colab or local environment
- **API Key Management**: Securely retrieves OpenAI and Tavily API keys from environment variables or Colab userdata
- **Package Installation**: Installs all required dependencies including LangChain, LangGraph, and data science libraries
- **Error Handling**: Provides fallback mechanisms for API key retrieval


## Cell 6 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# New Section
````
- **Markdown content summary**:

# New Section


## Cell 7 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 📚 Core Imports and Type System Foundation
````
- **Markdown content summary**:

# 📚 Core Imports and Type System Foundation


## Cell 8 – Code
- **Execution count**: None
- **Stored outputs**: 1
- **Source lines**: 318
- **Leading content**:

````
# MUST be first in the file/notebook once; do NOT re-import later cells
from __future__ import annotations
import json, math, inspect
````
- **Functions defined (10)**:
  - _is_colab
  - _make_idd_results_dir
  - _is_relative_to
  - persist_to_drive
  - keep_first
  - dict_merge_shallow
  - is_1d_vector
  - agent_list_default_generator
  - _should_ignore
  - log_failed_validation
- **Classes defined (10)**:
  - AgentMembers
  - InitialAnalysis
  - DataCleaner
  - Analyst
  - FileWriter
  - Visualization
  - ReportGenerator
  - SuperVisor
  - ReportOrchestrator
  - ReportSection
- **Imports (70)**:
  - from __future__ import annotations
  - import json, math, inspect
  - from functools import wraps
  - from langchain_huggingface import HuggingFaceEmbeddings
  - from langchain_core.embeddings import Embeddings
  - import os
  - import sys
  - import ast
  - import io
  - from io import StringIO, BytesIO
  - import re
  - import json
  - import uuid
  - import hashlib
  - import shutil
  - import logging
  - import functools
  - from functools import lru_cache
  - from math import nan
  - from pprint import pprint
  - from collections import OrderedDict
  - from collections.abc import Sequence
  - from typing import Dict, Optional, List, Tuple, Union, Literal, Any, Mapping, MutableMapping, cast, TypeGuard, Iterable, Callable
  - from typing_extensions import TypedDict, NotRequired, Annotated, TypeAlias
  - from tempfile import TemporaryDirectory
  - import itertools, threading
  - import numpy
  - from numpy.typing import ArrayLike
  - from numpy import ndarray
  - import pandas
  - from pandas import Index
  - from pandas.api.types import is_list_like
  - from scipy import stats
  - from sklearn.preprocessing import LabelEncoder, OneHotEncoder
  - import base64
  - import matplotlib.pyplot
  - from matplotlib.figure import Figure
  - from IPython.display import Image, display
  - import kagglehub
  - from tavily import TavilyClient
  - from langgraph.store.base import BaseStore
  - from langgraph.store.memory import InMemoryStore
  - from langgraph.cache.memory import InMemoryCache
  - from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
  - from langgraph.graph import StateGraph, MessagesState, START, END
  - from langgraph.graph.state import CompiledStateGraph
  - from langgraph.types import Command, CachePolicy, Send
  - from langgraph.prebuilt import create_react_agent, InjectedState, InjectedStore
  - from langgraph.utils.config import get_store
  - from langchain_openai import ChatOpenAI
  - from langchain_core.language_models.chat_models import BaseChatModel
  - from langchain_core.runnables import RunnableLambda
  - from langchain_core.runnables.config import RunnableConfig
  - from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
  - from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, ToolMessageChunk, ToolCall, trim_messages, AIMessageChunk, ChatMessage, BaseMessage, RemoveMessage, BaseMessageChunk, SystemMessageChunk, HumanMessageChunk
  - from typing import Sequence, List, Any, Set
  - from langchain_core.messages.utils import message_chunk_to_message
  - from langchain_core.tools import tool, InjectedToolArg, InjectedToolCallId, Tool
  - from langgraph.prebuilt.chat_agent_executor import AgentState
  - from langchain_experimental.tools.python.tool import PythonAstREPLTool
  - from langmem import create_manage_memory_tool, create_search_memory_tool
  - from langchain_community.agent_toolkits import FileManagementToolkit
  - from datetime import datetime
  - import shutil, hashlib
  - import operator
  - from operator import add, or_
  - from pydantic import BaseModel, Field, model_validator, field_validator, ValidationError, ConfigDict, AfterValidator, ValidationInfo, PrivateAttr
  - import seaborn
  - import google.colab
  - from google.colab import drive
- **Commentary / intent hints**:
  - # MUST be first in the file/notebook once; do NOT re-import later cells
  - # --- Stdlib ---

## Cell 9 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Establishes the foundational imports and type system for the entire notebook:
- **Advanced Typing**: Comprehensive type annotations using Python 3.12+ features
- **LangChain/LangGraph Stack**: Core imports for the multi-agent framework
````
- **Markdown content summary**:

Establishes the foundational imports and type system for the entire notebook:
- **Advanced Typing**: Comprehensive type annotations using Python 3.12+ features
- **LangChain/LangGraph Stack**: Core imports for the multi-agent framework
- **Data Science Libraries**: Pandas, NumPy, Matplotlib, Seaborn for data analysis
- **Type Safety**: Extensive use of TypedDict, Literal types, and generic annotations


## Cell 10 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🤖 OpenAI API Integration and Model Customization
````
- **Markdown content summary**:

# 🤖 OpenAI API Integration and Model Customization


## Cell 11 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 177
- **Leading content**:

````
from langchain_core.language_models import LanguageModelInput
from langchain_openai.chat_models.base import _construct_responses_api_input, _is_pydantic_class, _convert_message_to_dict, _convert_to_openai_response_format, _get_last_messages
from langchain_core.messages import BaseMessage
````
- **Functions defined (3)**:
  - _construct_responses_api_payload
  - _get_request_payload_mod
  - _get_request_payload
- **Classes defined (1)**:
  - MyChatOpenai
- **Imports (3)**:
  - from langchain_core.language_models import LanguageModelInput
  - from langchain_openai.chat_models.base import _construct_responses_api_input, _is_pydantic_class, _convert_message_to_dict, _convert_to_openai_response_format, _get_last_messages
  - from langchain_core.messages import BaseMessage
- **Commentary / intent hints**:
  -     # Rename legacy parameters
  -     # Remove temperature parameter for models that don't support it in responses API

## Cell 12 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Custom ChatOpenAI implementation with advanced features:
- **GPT-5 Support**: Forward-compatible implementation for o-series models
- **Responses API**: Handles transition from legacy to new OpenAI API patterns
````
- **Markdown content summary**:

Custom ChatOpenAI implementation with advanced features:
- **GPT-5 Support**: Forward-compatible implementation for o-series models
- **Responses API**: Handles transition from legacy to new OpenAI API patterns
- **Model-Specific Logic**: Adapts behavior based on model capabilities and limitations
- **Parameter Mapping**: Proper handling of deprecated and new API parameters


## Cell 13 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 📋 Dependency Version Verification
````
- **Markdown content summary**:

# 📋 Dependency Version Verification


## Cell 14 – Code
- **Execution count**: None
- **Stored outputs**: 1
- **Source lines**: 1
- **Leading content**:

````
!pip show --verbose langchain_experimental
````
- **AST parse**: failed (`invalid syntax (<unknown>, line 1)`); code likely contains magics or shell commands. Function/class inventory may be incomplete.

## Cell 15 – Markdown
- **Execution count**: None
- **Source lines**: 3
- **Leading content**:

````
Quick verification of installed package versions:
- **LangChain Experimental**: Checks the version of experimental features being used
- **Compatibility Validation**: Ensures correct versions are installed for the workflow
````
- **Markdown content summary**:

Quick verification of installed package versions:
- **LangChain Experimental**: Checks the version of experimental features being used
- **Compatibility Validation**: Ensures correct versions are installed for the workflow


## Cell 16 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🏗️ Data Models and State Architecture
````
- **Markdown content summary**:

# 🏗️ Data Models and State Architecture


## Cell 17 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 311
- **Leading content**:

````
# Support models — keep this cell before nodes/supervisor/graph
class BaseNoExtrasModel(BaseModel):
    model_config = ConfigDict(extra="forbid",json_schema_extra={"additionalProperties": False}) # -> additionalProperties: false
````
- **Functions defined (14)**:
  - _sort_plan_steps
  - _assert_sorted_completed_no_dups
  - _norm
  - _triplet_from_raw
  - ensure_b64
  - enforce_size
  - __init__
  - __str__
  - __repr__
  - to_dict
  - _sync_step_versions_on_assignment
  - _sync_steps_and_assert_increasing
  - _inject_and_dedupe
  - _sorted_no_dups_and_subset
- **Classes defined (23)**:
  - BaseNoExtrasModel
  - AnalysisConfig
  - CleaningMetadata
  - InitialDescription
  - VizSpec
  - AnalysisInsights
  - ImagePayload
  - DataVisualization
  - VisualizationResults
  - ReportResults
  - DataQueryParams
  - QueryDataframeInput
  - FileResult
  - ListOfFiles
  - DataFrameRegistryError
  - ProgressReport
  - PlanStep
  - Plan
  - CompletedStepsAndTasks
  - ToDoList
  - NextAgentMetadata
  - SendAgentMessage
  - MessagesToAgentsList
- **Imports (1)**:
  - from typing import List, ClassVar
- **Commentary / intent hints**:
  - # Support models — keep this cell before nodes/supervisor/graph
  -     # default_correlation_method: str = Field("pearson", description="Default method for correlation.")
  -     # automatic_outlier_removal: bool = Field(False, description="Whether to automatically remove outliers found.")

## Cell 18 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Defines the core data structures and state management system:
- **Pydantic Models**: Type-safe data models with validation for all workflow stages
- **State Definition**: Central state object managing the entire multi-agent workflow
````
- **Markdown content summary**:

Defines the core data structures and state management system:
- **Pydantic Models**: Type-safe data models with validation for all workflow stages
- **State Definition**: Central state object managing the entire multi-agent workflow
- **Configuration Models**: User settings and analysis configuration structures
- **Result Models**: Structured outputs for each analysis phase (cleaning, analysis, visualization, reporting)


## Cell 19 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 📊 DataFrame Registry and Data Management
````
- **Markdown content summary**:

# 📊 DataFrame Registry and Data Management


## Cell 20 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 249
- **Leading content**:

````
import threading

class DataFrameRegistry:
````
- **Functions defined (19)**:
  - get_global_df_registry
  - __init__
  - _norm_path
  - _write_df
  - _read_df
  - _touch_cache
  - write_dataframe_to_csv_file
  - write_dataframe_to_parquet_file
  - write_dataframe_to_pickle_file
  - write_dataframe_to_json_file
  - write_dataframe_to_file
  - register_dataframe
  - get_dataframe
  - remove_dataframe
  - get_raw_path_from_id
  - get_id_from_raw_path
  - has_df
  - ids
  - size
- **Classes defined (6)**:
  - DataFrameRegistry
  - Section
  - SectionOutline
  - ReportOutline
  - VizFeedback
  - ConversationalResponse
- **Imports (1)**:
  - import threading
- **Commentary / intent hints**:
  -     # ---------- internal helpers ----------

## Cell 21 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Centralized DataFrame management system with caching:
- **LRU Cache**: Efficient memory management for large datasets
- **Auto-ID Generation**: Automatic unique identifier assignment for DataFrames
````
- **Markdown content summary**:

Centralized DataFrame management system with caching:
- **LRU Cache**: Efficient memory management for large datasets
- **Auto-ID Generation**: Automatic unique identifier assignment for DataFrames
- **Thread Safety**: Concurrent access protection for multi-agent operations
- **File Integration**: Seamless loading and registration of datasets from file paths


## Cell 22 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🔄 State Management and Reducer Functions
````
- **Markdown content summary**:

# 🔄 State Management and Reducer Functions


## Cell 23 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 194
- **Leading content**:

````
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES

````
- **Functions defined (8)**:
  - merge_lists
  - merge_unique
  - merge_int_sum
  - merge_dicts
  - merge_dict
  - any_true
  - last_wins
  - _reduce_plan_keep_sorted
- **Classes defined (2)**:
  - State
  - VizWorkerState
- **Imports (2)**:
  - from langchain_core.messages import AnyMessage
  - from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
- **Commentary / intent hints**:
  - # --- custom reducers ---

## Cell 24 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Custom reducer functions for state merging and management:
- **Message Handling**: Manages conversation history and agent communications
- **List Merging**: Intelligent merging of analysis results and metadata
````
- **Markdown content summary**:

Custom reducer functions for state merging and management:
- **Message Handling**: Manages conversation history and agent communications
- **List Merging**: Intelligent merging of analysis results and metadata
- **Unique Merging**: Deduplication strategies for accumulated data
- **State Persistence**: Ensures proper state transitions across agent workflows


## Cell 25 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 💬 Agent Prompt Templates and Instructions
````
- **Markdown content summary**:

# 💬 Agent Prompt Templates and Instructions


## Cell 26 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 1077
- **Leading content**:

````
# === Agent Prompt Templates (ChatPromptTemplate) =================================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports (4)**:
  - from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
  - from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
  - from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
  - from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
- **Commentary / intent hints**:
  - # === Agent Prompt Templates (ChatPromptTemplate) =================================
  - # Shared guidance injected into each system prompt
  - # Short tooling guidance (keeps your policy, cuts fluff)

## Cell 27 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Comprehensive prompt templates for each specialized agent:
- **Role-Specific Prompts**: Tailored instructions for data cleaner, analyst, visualizer, and report generator
- **Tool Integration Guidelines**: Clear guidance on tool usage patterns and best practices
````
- **Markdown content summary**:

Comprehensive prompt templates for each specialized agent:
- **Role-Specific Prompts**: Tailored instructions for data cleaner, analyst, visualizer, and report generator
- **Tool Integration Guidelines**: Clear guidance on tool usage patterns and best practices
- **Output Format Specifications**: Structured JSON schema compliance requirements
- **Contextual Instructions**: Dynamic prompt adaptation based on data characteristics


## Cell 28 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 📝 Supervisor and Planning Prompt Templates
````
- **Markdown content summary**:

# 📝 Supervisor and Planning Prompt Templates


## Cell 29 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 194
- **Leading content**:

````
from langchain_core.prompts import ChatPromptTemplate

plan_prompt = ChatPromptTemplate.from_messages(
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports (1)**:
  - from langchain_core.prompts import ChatPromptTemplate

## Cell 30 – Markdown
- **Execution count**: None
- **Source lines**: 4
- **Leading content**:

````
Advanced prompt templates for workflow orchestration:
- **Supervisor Prompts**: Templates for the main coordinator agent
- **Planning Templates**: Strategic analysis and task distribution prompts
````
- **Markdown content summary**:

Advanced prompt templates for workflow orchestration:
- **Supervisor Prompts**: Templates for the main coordinator agent
- **Planning Templates**: Strategic analysis and task distribution prompts
- **Decision Logic**: Routing and workflow control instructions


## Cell 31 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🛠️ Comprehensive Tool Ecosystem and Error Handling
````
- **Markdown content summary**:

# 🛠️ Comprehensive Tool Ecosystem and Error Handling


## Cell 32 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 169
- **Leading content**:

````
import json
import math
import logging
````
- **Functions defined (6)**:
  - _is_df
  - _to_jsonable
  - _pretty
  - cap_output
  - deco
  - wrapper
- **Classes defined (0)**:None
- **Imports (7)**:
  - import json
  - import math
  - import logging
  - from functools import wraps
  - from typing import Any, Callable, Optional, Sequence
  - import pandas
  - from pydantic import BaseModel
- **Commentary / intent hints**:
  - # Optional: if you use pandas or pydantic v2

## Cell 33 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 5146
- **Leading content**:

````
# Filename helpers for saving files
# ---- shared helpers (place once) ----
import base64, binascii, html, shutil
````
- **Functions defined (100)**:
  - validate_dataframe_exists
  - handle_tool_errors
  - get_dataframe_schema
  - get_column_names
  - check_missing_values
  - drop_column
  - delete_rows
  - fill_missing_median
  - query_dataframe
  - get_descriptive_statistics
  - calculate_correlation
  - perform_hypothesis_test
  - _get_artifacts_base
  - _is_subpath
  - _resolve_artifact_path
  - create_sample
  - read_file
  - write_file
  - edit_file
  - get_df_from_registry
  - save_df_to_registry
  - strip_code_fences
  - sanitize_code
  - ensure_last_fn_call
  - _resolve_sandbox_root
  - _inside
  - sandbox_filesystem
  - python_repl_tool
  - _as_number_or_list
  - _as_int_or_list
  - _encode_png
  - _resolve_columns
  - _normalize_bins
  - _assert_bin_var_typesafety
  - _align_vector
  - _normalize
  - create_histogram
  - create_scatter_plot
  - create_correlation_heatmap
  - create_box_plot
  - create_violin_plot
  - export_dataframe
  - detect_and_remove_duplicates
  - convert_data_types
  - generate_html_report
  - calculate_correlation_matrix
  - detect_outliers
  - perform_normality_test
  - assess_data_quality
  - search_web_for_context
  - load_multiple_files
  - merge_dataframes
  - standardize_column_names
  - format_markdown_report
  - create_pdf_report
  - train_ml_model
  - handle_categorical_encoding
  - report_intermediate_progress
  - _hash_id
  - _coerce_viz_dict
  - _gather_from_state
  - _scan_artifacts_dir
  - _encode_preview
  - list_visualizations
  - get_visualization
  - wrapper
  - _workdir
  - _resolve_in_workdir
  - _guarded_open
  - _guarded_chdir
  - to_float
  - _validate_range
  - get_norm_by_stat
  - _shared_edges
  - _prep_weights_for_index
  - _resolve_name
  - _resolve_names
  - _resolve_name
  - _resolve_name
  - to_snake_case
  - default
  - _resolve_many
  - _resolve_many
  - _workdir
  - _sanitize_filename
  - _resolve_in_workdir
  - _decode_base64
  - _esc
  - _workdir
  - _sanitize_filename
  - _resolve_in_workdir
  - _md_escape_alt
  - _decode_base64
  - _workdir
  - _resolve_in_workdir
  - _x2p_link_callback
  - _match
  - _mpl_violin
  - _norm
  - _vals_w
- **Classes defined (2)**:
  - NewPythonInputs
  - NpEncoder
- **Imports (15)**:
  - import base64, binascii, html, shutil
  - from typing import Optional
  - from langchain_core.runnables.config import RunnableConfig
  - from langchain_core.tools import InjectedToolArg
  - from xhtml2pdf import pisa
  - import pandas
  - from xhtml2pdf import pisa
  - from sklearn.model_selection import train_test_split
  - from sklearn.linear_model import LogisticRegression, LinearRegression
  - from sklearn.metrics import accuracy_score, mean_squared_error
  - import joblib
  - import numpy
  - import scipy.cluster.hierarchy
  - from scipy.spatial.distance import squareform
  - import csv
- **Commentary / intent hints**:
  - # Filename helpers for saving files
  - # ---- shared helpers (place once) ----
  - # Error Handling and Validation Framework

## Cell 34 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 1
- **Leading content**:

````
# @tool("call_file_w
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  - # @tool("call_file_w

## Cell 35 – Markdown
- **Execution count**: None
- **Source lines**: 8
- **Leading content**:

````
The complete toolkit for data analysis operations:
- **Data Analysis Tools**: Statistical analysis, correlation, hypothesis testing, outlier detection
- **Data Cleaning Tools**: Missing value handling, duplicate removal, type conversion
````
- **Markdown content summary**:

The complete toolkit for data analysis operations:
- **Data Analysis Tools**: Statistical analysis, correlation, hypothesis testing, outlier detection
- **Data Cleaning Tools**: Missing value handling, duplicate removal, type conversion
- **Visualization Engine**: Chart generation (histograms, scatter plots, heatmaps, box plots)
- **File Management**: Read/write operations for multiple formats (CSV, Excel, JSON, HTML, PDF)
- **Python REPL Integration**: Dynamic code execution capabilities
- **Web Search Integration**: Tavily API integration for external research
- **Error Handling Framework**: Robust validation and error recovery mechanisms


## Cell 36 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🔧 Extended Imports and Memory Integration
````
- **Markdown content summary**:

# 🔧 Extended Imports and Memory Integration


## Cell 37 – Code
- **Execution count**: None
- **Stored outputs**: 1
- **Source lines**: 1536
- **Leading content**:

````
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.messages import SystemMessage, HumanMessage
````
- **Functions defined (39)**:
  - identity
  - e5_docs
  - e5_query
  - make_doc_embedder
  - make_query_embedder
  - load_memory_policy
  - estimate_importance
  - calculate_similarity
  - put_memory
  - retrieve_memories
  - format_memories_by_kind
  - enhanced_retrieve_mem
  - enhanced_mem_text
  - get_memory_metrics
  - reset_memory_metrics
  - memory_policy_report
  - put_memory_with_policy
  - retrieve_memories_with_ranking
  - prune_memories
  - recalculate_importance
  - update_memory_with_kind
  - categorize_memory_by_context
  - store_categorized_memory
  - _identity
  - _embed_docs
  - _identity
  - query_embed_func
  - __post_init__
  - __init__
  - insert
  - retrieve
  - prune
  - recalc_importance
  - _check_duplicates
  - _rank_memories
  - _calculate_score
  - _calculate_recency_factor
  - _update_usage
  - _maybe_prune
- **Classes defined (5)**:
  - MemoryRecord
  - MemoryPolicy
  - RankingWeights
  - PruneReport
  - MemoryPolicyEngine
- **Imports (21)**:
  - from langchain.embeddings import init_embeddings
  - from langchain_core.embeddings import Embeddings
  - from langchain_core.messages import SystemMessage, HumanMessage
  - from langgraph.checkpoint.memory import InMemorySaver
  - from langgraph.store.memory import InMemoryStore
  - from langgraph.types import Command, Send
  - from functools import lru_cache
  - from typing import List, Union
  - import time
  - import uuid
  - import math
  - import logging
  - import yaml
  - import os
  - from dataclasses import dataclass
  - from typing import Callable
  - from functools import partial
  - from langgraph.utils.config import get_store
  - from langgraph.utils.config import get_store
  - import builtins
  - import builtins
- **Commentary / intent hints**:
  - #import Callable
  - # -------------------------

## Cell 38 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Additional imports and setup for advanced features:
- **Memory Systems**: Integration with LangGraph memory and checkpointing
- **Embedding Models**: Setup for vector storage and retrieval
````
- **Markdown content summary**:

Additional imports and setup for advanced features:
- **Memory Systems**: Integration with LangGraph memory and checkpointing
- **Embedding Models**: Setup for vector storage and retrieval
- **Store Integration**: Advanced state management with persistent storage
- **Command Types**: Support for complex workflow commands and parallel processing


## Cell 39 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# LLM Initialization and Agent Factories
````
- **Markdown content summary**:

# LLM Initialization and Agent Factories


## Cell 40 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 699
- **Leading content**:

````
# Global toggles
USE_STRICT_JSON_SCHEMA_FINAL_HOP = use_local_llm     # gate: True = strict JSON-Schema final hop; False = your current path
USE_MANUAL_SCHEMA_BINDING = False           # fallback if your LC build doesn’t honor method="json_schema"
````
- **Functions defined (27)**:
  - _safe_json_loads
  - extract_tool_calls
  - retry_extract_tool_calls
  - format_tool_responses_for_qwen3
  - _collect_trailing_tool_messages
  - qwen3_pre_model_hook
  - _parse_inline_tool_calls
  - getnestedattr
  - remove_unwanted_tags
  - qwen3_post_model_hook
  - is_final_answer
  - extract_final_text
  - _is_required
  - _type_name
  - _enum_values
  - collect_model_docs
  - format_model_for_prompt
  - count_last_cycle_tool_calls
  - _final_llm_for_model
  - _strict_final_wrapper
  - chain_pre_hooks
  - chain_post_hooks
  - as_post_runnable
  - _run
  - _run
  - _run
  - _run
- **Classes defined (0)**:None
- **Imports (13)**:
  - from typing import Type
  - import json, re, uuid
  - import json, re, uuid
  - from typing import Any, Dict, List, Optional
  - from langchain_core.messages import AIMessage
  - from typing import Any, Dict, List, Type
  - from langchain_core.messages import AIMessage, BaseMessage
  - from pydantic import BaseModel
  - from typing import get_origin, get_args, Any, Type
  - from enum import Enum
  - from pydantic import BaseModel
  - from pydantic.fields import FieldInfo
  - from pydantic_core import PydanticUndefined
- **Commentary / intent hints**:
  - # Global toggles
  - #run your llama.cpp server
  - # ./llama-server -m /mnt/d/agent_models/Qwen3-4B-Function-Calling-Pro.gguf -c 65536 --n-gpu-layers -1 --host 0.0.0.0 --mlock /
  - # --no-context-shift --jinja --chat-template-file ./qwen3chat_template.tmpl --reasoning-budget -1 --verbose /
  - # --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 0.5 --repeat-penalty 1.05 /
  - # --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 32768 --keep -1 /
  - # --prio 1 -ub 896 -b 2048 --cache-reuse 1024 -n 1024
  - # Now, your LangGraph code in Colab will send requests to the ngrok URL,
  - # which will forward them to your local llama.cpp server.
  - # Copy the "Forwarding" URL from your ngrok terminal
  - # Initialize these ones like this:
  - # llm = ChatOpenAI(
  - #     # Append the /v1 endpoint to the ngrok URL

## Cell 41 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 35
- **Leading content**:

````
# --- LLM ---
big_picture_llm = ChatOpenAI(model="gpt-5-mini",use_responses_api=True, api_key=oai_key,output_version="responses/v1",reasoning={'effort': 'high'}, model_kwargs={'text': {'verbosity': 'low'}}) if not use_local_llm else ChatOpenAI(base_url=f"{ngrok_url}/v1",api_key="337jQ3UhKyoRJVafubzVUjeippe_4Niq48FtneDTEjc5GF2bB",  model="qwen3-4b-local",temperature=0.5,extra_body={"top_p": 0.95,"repeat_penalty": 0.5,})
router_llm = ChatOpenAI(model="gpt-5-mini",use_responses_api=True, api_key=oai_key,output_version="responses/v1",reasoning={'effort': 'medium'}, model_kwargs={'text': {'verbosity': 'low'}}) if not use_local_llm else ChatOpenAI(base_url=f"{ngrok_url}/v1",api_key="337jQ3UhKyoRJVafubzVUjeippe_4Niq48FtneDTEjc5GF2bB",  model="qwen3-4b-local",temperature=0.5,extra_body={"top_p": 0.95,"repeat_penalty": 0.5,})
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  - # --- LLM ---

## Cell 42 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 52
- **Leading content**:

````

# "conversation",
# "analysis",
````
- **Functions defined (1)**:
  - _dedupe_tools
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  - # "conversation",
  - # "analysis",
  - # "progress",
  - # "routes",
  - # "replies",
  - # "plans",
  - # "todos",
  - # "initial_description",
  - # "cleaning",
  - # "visualization",
  - # "insights",
  - # "reports",
  - # "files",
  - # "errors"

## Cell 43 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 725
- **Leading content**:

````
# --- pre_model_hook_summarize_then_trim.py ---

from langchain_core.messages.utils import count_tokens_approximately
````
- **Functions defined (31)**:
  - _msg_has_tool_invocation
  - _extract_message_id
  - _new_id
  - _with_id
  - _rebuild_plain_message
  - _make_state_snapshot_message
  - strip_tools_for_summary_hardened
  - _count_tokens
  - _find_indices
  - _summarize_span
  - _retrim_with_current_indices
  - _is_role
  - _last_k_ai_indices
  - _last_supervisor_index
  - choose_protected_ai_indices
  - _last_k_tool_indices
  - _extract_refs_from_text
  - _referenced_tool_indices
  - _no_summary_tools
  - _token_cost
  - choose_protected_tool_indices
  - _shrink_protection_windows_if_needed
  - _state_snapshot_collapse
  - _hard_trim_oldest_non_human
  - make_pre_model_hook
  - _construct
  - _tok_cost
  - protected
  - pre_model_hook
  - _safe_pointerize_tool_message
  - _pointerize_all_but_last_tool_safe
- **Classes defined (0)**:None
- **Imports (2)**:
  - from langchain_core.messages.utils import count_tokens_approximately
  - from langmem.short_term import summarize_messages, RunningSummary
- **Commentary / intent hints**:
  - # --- pre_model_hook_summarize_then_trim.py ---
  -     # 1) Newer LC encodes tool calls here
  -     # 2) Older/alt encodings live in additional_kwargs

## Cell 44 – Code
- **Execution count**: None
- **Stored outputs**: 1
- **Source lines**: 2531
- **Leading content**:

````


if use_local_llm:
````
- **Functions defined (39)**:
  - create_data_cleaner_agent
  - create_initial_analysis_agent
  - create_analyst_agent
  - create_file_writer_agent
  - create_visualization_agent
  - create_viz_evaluator_agent
  - create_report_generator_agent
  - update_memory
  - make_supervisor_node
  - _dedup
  - _get_step_identity
  - dedup_steps
  - _parse_cst_with_plan
  - schema_for_completed_steps
  - _cosine_similarity
  - _euclidean
  - _manhattan
  - _unit
  - _bucket_from_thresholds
  - embedding_similarity_report
  - _tokens
  - _jaccard
  - _pair_similarity
  - _weighted_mean
  - same_task
  - _norm
  - _key
  - _name_or_desc_match
  - _same_or_fuzzy
  - consolidate_plan_with_completed_steps
  - supervisor_node
  - _inner
  - d_from_cos
  - _sync_ver
  - _complete
  - _find_matching_done
  - _found_in_done
  - _has_completed_neighbor
  - _replace_in_done
- **Classes defined (1)**:
  - Router
- **Imports (2)**:
  - from pydantic import BaseModel, Field
  - from typing import Literal
- **Commentary / intent hints**:
  - # If you prefer to overwrite the graph state's messages entirely, return:
  - # return {"messages": [RemoveMessage(REMOVE_ALL_MESSAGES), *trimmed]}
  - # -------------------------
  - # Agent factories
  - # -------------------------

## Cell 45 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 📂 Sample Dataset Loading and Registration
````
- **Markdown content summary**:

# 📂 Sample Dataset Loading and Registration


## Cell 46 – Code
- **Execution count**: None
- **Stored outputs**: 1
- **Source lines**: 79
- **Leading content**:

````
# Download & prepare sample dataset from KaggleHub (robust)
# Assumes: pprint, os, pandas as pd, kagglehub, global_df_registry,
#          InitialDescription, and the agent factory fns are imported.
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports (1)**:
  - import glob
- **Commentary / intent hints**:
  - # Download & prepare sample dataset from KaggleHub (robust)
  - # Assumes: pprint, os, pandas as pd, kagglehub, global_df_registry,
  - #          InitialDescription, and the agent factory fns are imported.
  - # Download (cached by kagglehub if already present)
  - # Pick the most appropriate CSV:
  - # 1) Prefer files starting with the canonical prefix
  - # 2) Otherwise, take the largest CSV

## Cell 47 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Automated dataset acquisition and registration:
- **KaggleHub Integration**: Downloads sample dataset from Kaggle
- **Data Registration**: Automatic registration in the DataFrame registry
````
- **Markdown content summary**:

Automated dataset acquisition and registration:
- **KaggleHub Integration**: Downloads sample dataset from Kaggle
- **Data Registration**: Automatic registration in the DataFrame registry
- **Initial Analysis**: Basic dataset inspection and metadata extraction
- **Path Management**: Robust file handling and path resolution


## Cell 48 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# ⚙️ Runtime Context and Configuration
````
- **Markdown content summary**:

# ⚙️ Runtime Context and Configuration


## Cell 49 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 73
- **Leading content**:

````
import uuid

sample_prompt_text = f"Please analyze the dataset named {df_name}. You have tools available to you for accessing the data using the following str as the df_id parameter: `{df_id}`. A full analysis will be performed on the dataset. Then, relevant and meaningful visualizations will need to be chosen and produced with the data, after which a full report will be generated in several formats, including PDF, Markdown, and HTML."
````
- **Functions defined (0)**:None
- **Classes defined (1)**:
  - RuntimeCtx
- **Imports (5)**:
  - import uuid
  - from dataclasses import dataclass
  - import uuid
  - import os
  - from datetime import datetime, UTC
- **Commentary / intent hints**:
  - # --- runtime_ctx.py (put this near your imports or in a small cell) ---

## Cell 50 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Runtime configuration and context management:
- **Working Directories**: Setup of output directories for reports and visualizations
- **Sample Prompts**: Default user prompts for testing and demonstration
````
- **Markdown content summary**:

Runtime configuration and context management:
- **Working Directories**: Setup of output directories for reports and visualizations
- **Sample Prompts**: Default user prompts for testing and demonstration
- **UUID Generation**: Unique identifiers for tracking analysis sessions
- **Configuration Objects**: Runtime context for workflow execution


## Cell 51 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 📋 Report Generation Utilities and Packaging
````
- **Markdown content summary**:

# 📋 Report Generation Utilities and Packaging


## Cell 52 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 408
- **Leading content**:

````
# report_packager_node helpers
import textwrap

````
- **Functions defined (15)**:
  - _sha256_bytes
  - _detect_mime_and_encoding
  - _atomic_write_bytes
  - _atomic_write_text
  - _write_bytes
  - _write_text
  - _materialize_images
  - _render_markdown
  - _markdown_to_html
  - _ensure_list_str
  - _safe_copy
  - _resolve_artifacts_root
  - save_viz_for_state
  - _manifest_from_path
  - _next_version_path
- **Classes defined (0)**:None
- **Imports (4)**:
  - import textwrap
  - import tempfile, hashlib, mimetypes
  - import base64, uuid, os
  - import os
- **Commentary / intent hints**:
  - # report_packager_node helpers
  - # --- Add near top of the helpers cell (report_packager_node helpers) ---
  -     # sensible fallbacks

## Cell 53 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Helper functions for report generation and file management:
- **File Writing Utilities**: Safe file operations with error handling
- **Report Packaging**: Multi-format report generation (HTML, Markdown, PDF)
````
- **Markdown content summary**:

Helper functions for report generation and file management:
- **File Writing Utilities**: Safe file operations with error handling
- **Report Packaging**: Multi-format report generation (HTML, Markdown, PDF)
- **Template Management**: Report template processing and customization
- **Output Organization**: Structured file output with proper naming conventions


## Cell 54 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# **🤖 Agent Node Implementation and Workflow Logic**
````
- **Markdown content summary**:

# **🤖 Agent Node Implementation and Workflow Logic**


## Cell 55 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 1794
- **Leading content**:

````
# Node Functions (revised)


````
- **Functions defined (22)**:
  - initial_analysis_node
  - data_cleaner_node
  - analyst_node
  - _normalize_meta
  - file_writer_node
  - _guess_viz_type
  - _norm_title
  - _normalize_viz_spec
  - visualization_orchestrator
  - viz_worker
  - assign_viz_workers
  - viz_join
  - viz_evaluator_node
  - route_viz
  - report_orchestrator
  - section_worker
  - dispatch_sections
  - assign_section_workers
  - report_join
  - report_packager_node
  - emergency_correspondence_node
  - enhanced_retrieve_mem
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  - # Node Functions (revised)

## Cell 56 – Markdown
- **Execution count**: None
- **Source lines**: 7
- **Leading content**:

````
Core agent node implementations for the multi-agent workflow:
- **Initial Analysis Node**: Dataset inspection and metadata extraction
- **Data Cleaner Node**: Automated data cleaning and preprocessing
````
- **Markdown content summary**:

Core agent node implementations for the multi-agent workflow:
- **Initial Analysis Node**: Dataset inspection and metadata extraction
- **Data Cleaner Node**: Automated data cleaning and preprocessing
- **Analyst Node**: Statistical analysis and pattern detection
- **Visualization Node**: Chart and graph generation
- **Report Generator Node**: Comprehensive report compilation
- **Memory Integration**: Persistent state management across workflow stages


## Cell 57 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🌐 Workflow Graph Compilation and Configuration
````
- **Markdown content summary**:

# 🌐 Workflow Graph Compilation and Configuration


## Cell 58 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 199
- **Leading content**:

````
 #Graph compile (revised)

coordinator_node = make_supervisor_node(
````
- **Functions defined (3)**:
  - write_output_to_file
  - route_to_writer
  - route_from_supervisor
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  -  #Graph compile (revised)
  -     # Route to file_writer if reports exist and file writing hasn't been done yet

## Cell 59 – Markdown
- **Execution count**: None
- **Source lines**: 6
- **Leading content**:

````
LangGraph workflow compilation and supervisor integration:
- **Multi-LLM Supervisor**: Advanced coordinator with specialized sub-models
- **Graph Construction**: Complete workflow graph with all nodes and edges
````
- **Markdown content summary**:

LangGraph workflow compilation and supervisor integration:
- **Multi-LLM Supervisor**: Advanced coordinator with specialized sub-models
- **Graph Construction**: Complete workflow graph with all nodes and edges
- **Parallel Processing**: Support for concurrent analysis operations
- **Error Recovery**: Graceful handling of node failures and timeouts
- **Checkpointing**: Workflow state persistence and recovery capabilities


## Cell 60 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 📊 Workflow Graph Visualization
````
- **Markdown content summary**:

# 📊 Workflow Graph Visualization


## Cell 61 – Code
- **Execution count**: None
- **Stored outputs**: 1
- **Source lines**: 5
- **Leading content**:

````
try:
    display(Image(data_detective_graph.get_graph().draw_mermaid_png(max_retries=2, retry_delay=2.0)))
except Exception as e:
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None

## Cell 62 – Markdown
- **Execution count**: None
- **Source lines**: 4
- **Leading content**:

````
Visual representation of the compiled workflow graph:
- **Mermaid Diagram**: Interactive workflow visualization
- **Node Relationships**: Clear display of agent interactions and data flow
````
- **Markdown content summary**:

Visual representation of the compiled workflow graph:
- **Mermaid Diagram**: Interactive workflow visualization
- **Node Relationships**: Clear display of agent interactions and data flow
- **Debugging Aid**: Visual debugging tool for workflow understanding


## Cell 63 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# ✅ Schema Validation and Testing
````
- **Markdown content summary**:

# ✅ Schema Validation and Testing


## Cell 64 – Code
- **Execution count**: None
- **Stored outputs**: 1
- **Source lines**: 6
- **Leading content**:

````
print(InitialDescription.model_json_schema())
initial_test = InitialDescription(dataset_description="test", data_sample="test", notes="test notes", reply_msg_to_supervisor="test", finished_this_task=False, expect_reply=False)
print(initial_test.model_dump_json())
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  - # print(initial_analysis_agent.invoke(

## Cell 65 – Markdown
- **Execution count**: None
- **Source lines**: 4
- **Leading content**:

````
Validation of data models and schema compliance:
- **Pydantic Schema Validation**: Ensures proper model structure
- **JSON Schema Generation**: Validates serialization/deserialization
````
- **Markdown content summary**:

Validation of data models and schema compliance:
- **Pydantic Schema Validation**: Ensures proper model structure
- **JSON Schema Generation**: Validates serialization/deserialization
- **Type Safety Testing**: Confirms type annotations and constraints


## Cell 66 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🔍 Advanced Debugging and Introspection Tools
````
- **Markdown content summary**:

# 🔍 Advanced Debugging and Introspection Tools


## Cell 67 – Code
- **Execution count**: None
- **Stored outputs**: 1
- **Source lines**: 125
- **Leading content**:

````
#These are only helpers for accessing or checking keys nested within variable iterables - do not worry about or focus on these, they are non-critical print helpers

from collections.abc import Mapping, Sequence
````
- **Functions defined (6)**:
  - find_key_paths
  - find_key_paths_list
  - get_by_path
  - pick_tool_messages
  - extract_handles_from_tools
  - _walk
- **Classes defined (0)**:None
- **Imports (3)**:
  - from collections.abc import Mapping, Sequence
  - from typing import Iterable, Tuple, Union, Any, TypeAlias, Literal, Optional, Set
  - from langchain_core.messages import ToolMessage
- **Commentary / intent hints**:
  - #These are only helpers for accessing or checking keys nested within variable iterables - do not worry about or focus on these, they are non-critical print helpers

## Cell 68 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Sophisticated debugging utilities for complex data structures:
- **Path Finding**: Navigate nested data structures and find specific keys/values
- **Deep Inspection**: Analyze complex nested objects and state structures
````
- **Markdown content summary**:

Sophisticated debugging utilities for complex data structures:
- **Path Finding**: Navigate nested data structures and find specific keys/values
- **Deep Inspection**: Analyze complex nested objects and state structures
- **Type Analysis**: Runtime type checking and validation
- **Search Utilities**: Locate specific data within large state objects


## Cell 69 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🚀 Streaming Workflow Execution and Real-time Processing
````
- **Markdown content summary**:

# 🚀 Streaming Workflow Execution and Real-time Processing


## Cell 70 – Code
- **Execution count**: None
- **Stored outputs**: 1
- **Source lines**: 212
- **Leading content**:

````
# Streaming run (clean + robust)

from langchain_core.runnables.config import RunnableConfig
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports (4)**:
  - from langchain_core.runnables.config import RunnableConfig
  - from langchain_core.messages import HumanMessage
  - import uuid
  - import traceback
- **Commentary / intent hints**:
  - # Streaming run (clean + robust)
  - # print langchain_openai version for debugging
  - # !pip show langchain
  - # !pip show langchain_openai
  - # !pip show langchain_core
  - # !pip show langgraph
  - # One config to rule them all

## Cell 71 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Main execution engine for the data analysis workflow:
- **Stream Processing**: Real-time execution with live updates
- **Progress Monitoring**: Track workflow progress and intermediate results
````
- **Markdown content summary**:

Main execution engine for the data analysis workflow:
- **Stream Processing**: Real-time execution with live updates
- **Progress Monitoring**: Track workflow progress and intermediate results
- **Error Handling**: Robust error recovery and graceful degradation
- **Result Streaming**: Live display of analysis results as they're generated


## Cell 72 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 0
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None

## Cell 73 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 📡 Extended Streaming Utilities and Text Processing
````
- **Markdown content summary**:

# 📡 Extended Streaming Utilities and Text Processing


## Cell 74 – Code
- **Execution count**: None
- **Stored outputs**: 2
- **Source lines**: 665
- **Leading content**:

````
from IPython.display import HTML, display
display(HTML("""
<style>
````
- **AST parse**: failed (`expected an indented block after function definition on line 229 (<unknown>, line 230)`); code likely contains magics or shell commands. Function/class inventory may be incomplete.
- **Commentary / intent hints**:
  - # --- helper: fallback text extractor when your get_by_path/find_key_paths aren't present ---

## Cell 75 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 44
- **Leading content**:

````
collected = {"test":{"langgraph_step": 0, "msg": SystemMessage(content="test message")}}
unparsed_steps = []
for rstep_ in received_steps:
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  -     # print(getattr(rstep, "id", "fart"))
  -             # collected[rstep.id] = rstep.id

## Cell 76 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Advanced streaming utilities and content extraction:
- **Text Extraction**: Extract text from various content formats and structures
- **Content Processing**: Handle OpenAI-style content blocks and nested structures
````
- **Markdown content summary**:

Advanced streaming utilities and content extraction:
- **Text Extraction**: Extract text from various content formats and structures
- **Content Processing**: Handle OpenAI-style content blocks and nested structures
- **Stream Utilities**: Additional helpers for streaming operations
- **Format Handling**: Support for multiple content types and formats


## Cell 77 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 62
- **Leading content**:

````
# save temp dir to gdrive
final_state = data_detective_graph.get_state(run_config)
if final_state and final_state.values:
````
- **Functions defined (1)**:
  - _dir_has_any_files
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  - # save temp dir to gdrive

## Cell 78 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🔎 Final State Inspection and Results Review
````
- **Markdown content summary**:

# 🔎 Final State Inspection and Results Review


## Cell 79 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 90
- **Leading content**:

````
print("Figures:", list(RUNTIME.viz_dir.glob("*.png")))
print("Reports:", list(RUNTIME.reports_dir.glob("*.*")))
# Inspect final state from the checkpointer (since we used MemorySaver + thread_id)
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  - # Inspect final state from the checkpointer (since we used MemorySaver + thread_id)
  -         # Peek at structured products if present

## Cell 80 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Comprehensive inspection of workflow results:
- **State Analysis**: Examine final workflow state and generated artifacts
- **File Listing**: Review generated reports, visualizations, and data files
````
- **Markdown content summary**:

Comprehensive inspection of workflow results:
- **State Analysis**: Examine final workflow state and generated artifacts
- **File Listing**: Review generated reports, visualizations, and data files
- **Checkpointer Access**: Retrieve and analyze saved workflow checkpoints
- **Results Summary**: Overview of completed analysis and generated outputs


## Cell 81 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 1
- **Leading content**:

````
pprint(WORKING_DIRECTORY)
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None

## Cell 82 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🔧 Function Calling Utilities and Tool Conversion
````
- **Markdown content summary**:

# 🔧 Function Calling Utilities and Tool Conversion


## Cell 83 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 1
- **Leading content**:

````
from langchain_core.utils.function_calling import convert_to_openai_tool
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports (1)**:
  - from langchain_core.utils.function_calling import convert_to_openai_tool

## Cell 84 – Markdown
- **Execution count**: None
- **Source lines**: 4
- **Leading content**:

````
Utilities for OpenAI function calling and tool conversion:
- **Tool Conversion**: Convert Pydantic models to OpenAI tool format
- **Schema Validation**: Ensure proper function calling schema compliance
````
- **Markdown content summary**:

Utilities for OpenAI function calling and tool conversion:
- **Tool Conversion**: Convert Pydantic models to OpenAI tool format
- **Schema Validation**: Ensure proper function calling schema compliance
- **API Compatibility**: Support for different OpenAI API versions and formats


## Cell 85 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Advanced testing and validation of data models:
- **Schema Comparison**: Compare different schema generation methods
- **Strict Validation**: Test strict vs. lenient validation modes
````
- **Markdown content summary**:

Advanced testing and validation of data models:
- **Schema Comparison**: Compare different schema generation methods
- **Strict Validation**: Test strict vs. lenient validation modes
- **Alias Testing**: Validate field aliases and serialization options
- **Compatibility Testing**: Ensure backward compatibility with different versions


## Cell 86 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# 🎯 Final Model Validation and Quality Assurance
````
- **Markdown content summary**:

# 🎯 Final Model Validation and Quality Assurance


## Cell 87 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 10
- **Leading content**:

````
# initial_test = InitialDescription(dataset_description="test", data_sample="test")
# print(initial_test.model_dump_json())
# print(InitialDescription.model_validate(initial_test, strict=True,from_attributes=True))
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None
- **Commentary / intent hints**:
  - # initial_test = InitialDescription(dataset_description="test", data_sample="test")
  - # print(initial_test.model_dump_json())
  - # print(InitialDescription.model_validate(initial_test, strict=True,from_attributes=True))
  - # print("\n")
  - # print(initial_test.model_json_schema().__str__())
  - # print("\n")
  - # # print(initial_test.model_validate(initial_test.model_json_schema(), strict=True,from_attributes=True))
  - # print("\n")
  - # initial_test.model_json_schema()

## Cell 88 – Markdown
- **Execution count**: None
- **Source lines**: 5
- **Leading content**:

````
Final validation steps and quality assurance checks:
- **Model Compliance**: Final verification of all data models
- **Serialization Testing**: Validate JSON serialization and deserialization
````
- **Markdown content summary**:

Final validation steps and quality assurance checks:
- **Model Compliance**: Final verification of all data models
- **Serialization Testing**: Validate JSON serialization and deserialization
- **Schema Output**: Generate and verify final schema documentation
- **Quality Checks**: Comprehensive validation of the entire system


## Cell 89 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# Save the InMemorySaver checkpointer to a SQL database file on disk
````
- **Markdown content summary**:

# Save the InMemorySaver checkpointer to a SQL database file on disk


## Cell 90 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 24
- **Leading content**:

````
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
# src graph was compiled with InMemorySaver
````
- **Functions defined (1)**:
  - migrate_thread
- **Classes defined (0)**:None
- **Imports (2)**:
  - import sqlite3
  - from langgraph.checkpoint.sqlite import SqliteSaver
- **Commentary / intent hints**:
  - # src graph was compiled with InMemorySaver
  - # destination graph with SQLite persistence
  -           # choose the last writer for correct "what runs next"

## Cell 91 – Markdown
- **Execution count**: None
- **Source lines**: 1
- **Leading content**:

````
# To restore a previous checkpointer state from an SQL database file on disk
````
- **Markdown content summary**:

# To restore a previous checkpointer state from an SQL database file on disk


## Cell 92 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 8
- **Leading content**:

````
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports (2)**:
  - import sqlite3
  - from langgraph.checkpoint.sqlite import SqliteSaver
- **Commentary / intent hints**:
  - # conn.close() #Use conn.close() when finished

## Cell 93 – Code
- **Execution count**: None
- **Stored outputs**: 0
- **Source lines**: 1
- **Leading content**:

````
print(run_config)
````
- **Functions defined (0)**:None
- **Classes defined (0)**:None
- **Imports**: None
