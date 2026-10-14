"""
idd_core.py — Transitional compatibility façade for Intelligent Data Detective.

Historically extracted from IntelligentDataDetective_beta_v5.ipynb for testing.
During the canonical core migration (Issue #140 / Issue #156):
- Planning and step synchronization models have been extracted to `idd_models.py`.
- DataFrameRegistry and context isolation helpers have been extracted to `idd_registry.py`.
- This module serves as a backward-compatible façade re-exporting canonical symbols
  alongside transitional tools, adapters, and helper functions until subsequent checkpoints
  (CP2-CP5) establish dedicated canonical modules.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path as PathlibPath
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Tuple,
    TypeGuard,
    Union,
)
from typing_extensions import Annotated
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas.api.types import is_list_like
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Canonical models and registry imports (Checkpoint 1)
from idd_models import (
    BaseNoExtrasModel,
    ProgressReport,
    PlanStep,
    Plan,
    CompletedStepsAndTasks,
    Triplet,
    _norm,
    _triplet_from_raw,
    _sort_plan_steps,
    _assert_sorted_completed_no_dups,
    _reduce_plan_keep_sorted,
)
from idd_registry import (
    DataFrameRegistry,
    DataFrameRegistryError,
    get_global_registry,
    set_global_registry,
    override_global_registry,
)

# Canonical tools and error handling (Checkpoint 2)
from idd_tools import (
    WORKING_DIRECTORY,
    _get_artifacts_base,
    _is_subpath,
    _resolve_artifact_path,
    _tool_error,
    _tool_failure,
    handle_tool_errors,
    is_safe_artifact_path,
    validate_dataframe_exists,
)

# Canonical state, reducers, and domain schemas (Checkpoint 3)
from idd_state import (
    State,
    VizWorkerState,
    keep_first,
    _sr_reducer,
    _reduce_strs_resettable,
    _reduce_viz_results_resettable,
    _keep_last_or_clear,
    merge_int_sum,
    bool_or,
    dict_merge_shallow,
    merge_lists,
    merge_unique,
    merge_dicts,
    merge_dict,
    any_true,
    last_wins,
    AgentId,
    Supervisor,
    AgentOrSupervisor,
    AgentMembers,
    AnalysisConfig,
    CleaningMetadata,
    InitialDescription,
    VizSpec,
    AnalysisInsights,
    ImagePayload,
    DataVisualization,
    VisualizationResults,
    ReportResults,
    DataQueryParams,
    QueryDataframeInput,
    FileResult,
    ListOfFiles,
    ToDoList,
    NextAgentMetadata,
    SendAgentMessage,
    MessagesToAgentsList,
    Section,
    SectionOutline,
    ReportOutline,
    VizFeedback,
    ConversationalResponse,
)

# Canonical graph assembly and routing (Checkpoint 3)
from idd_graph import (
    build_graph,
    AGENT_MEMBERS,
    AGENT_OPTIONS,
    options,
    PRODUCTION_NODE_NAMES,
    Router,
    route_from_supervisor,
    route_to_writer,
    route_viz,
    write_output_to_file,
    assign_viz_workers,
    dispatch_sections,
)

# Optional LangChain imports (needed for MyChatOpenai and State only)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        HumanMessage,
        AIMessage,
        SystemMessage,
        ToolMessage,
        BaseMessage,
        AnyMessage,
    )
    from langchain_core.runnables.config import RunnableConfig

    try:
        from langchain.agents import AgentState
    except ImportError:
        # AgentState was removed from langchain.agents in newer versions
        AgentState = dict
    from langgraph.graph.message import add_messages

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseChatModel = object
    RunnableConfig = dict
    AnyMessage = Any
    AgentState = dict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _is_colab() -> bool:
    """Always returns False in idd_core — no Colab dependency."""
    return False


def _make_idd_results_dir() -> PathlibPath:
    base_env = os.environ.get("GDRIVE_BASE", "")
    base = (
        PathlibPath(base_env).expanduser()
        if base_env
        else PathlibPath.cwd() / "IDD_results"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def _is_relative_to(a: PathlibPath, b: PathlibPath) -> bool:
    try:
        return a.resolve().is_relative_to(b.resolve())  # py>=3.9
    except AttributeError:
        ar, br = str(a.resolve()), str(b.resolve())
        return ar.startswith(br)


# ---------------------------------------------------------------------------
# Type aliases and helpers
# ---------------------------------------------------------------------------

Array1D = Union[
    Sequence[float],
    Sequence[int],
    np.ndarray,
    pd.Series,
]


def is_1d_vector(x: object) -> TypeGuard[Array1D]:
    """Return True if x is a 1-D numeric-like sequence."""
    try:
        if isinstance(x, (str, bytes)):
            return False
        if isinstance(x, pd.Series):
            return True
        if isinstance(x, np.ndarray):
            return x.ndim == 1
        if is_list_like(x):
            try:
                arr = np.asarray(x)
            except Exception:
                return False
            return arr.ndim == 1
        if isinstance(x, Sequence):
            try:
                arr = np.asarray(x, dtype=float)
            except (ValueError, TypeError):
                return False
            return arr.ndim == 1
        return False
    except Exception as e:
        raise e


Number = Union[int, float]
ScalarNum = Annotated[Number, "Scalar number (int | float)"]
Estimator = Literal["auto", "fd", "doane", "scott", "sturges", "sqrt", "stone", "rice"]

BinSpec = Union[
    int,
    Estimator,
    Tuple[Union[int, Estimator, Sequence[int]], Union[int, Estimator, Sequence[int]]],
    ArrayLike,
    None,
]

BinWidthSpec = Annotated[
    Union[Number, Sequence[Number], np.ndarray, pd.Series, None],
    "A scalar or sequence of widths",
]
RangeSpec = Annotated[
    Optional[Tuple[Number, Number]],
    "(lo, hi) numeric tuple",
]
ColumnSelector = Annotated[
    Optional[Union[str, int, Sequence[str], Sequence[int], Literal["all"]]],
    "Column(s) to select",
]


class InitialAnalysis(AgentMembers):
    agent_type: Literal["initial_analysis"] = "initial_analysis"


class DataCleaner(AgentMembers):
    agent_type: Literal["data_cleaner"] = "data_cleaner"


class Analyst(AgentMembers):
    agent_type: Literal["analyst"] = "analyst"


class FileWriter(AgentMembers):
    agent_type: Literal["file_writer"] = "file_writer"


class Visualization(AgentMembers):
    agent_type: Literal["visualization"] = "visualization"


class ReportGenerator(AgentMembers):
    agent_type: Literal["report_packager"] = "report_packager"


class SuperVisor(AgentMembers):
    agent_type: Literal["supervisor"] = "supervisor"


class ReportOrchestrator(AgentMembers):
    agent_type: Literal["report_orchestrator"] = "report_orchestrator"


class ReportSection(AgentMembers):
    agent_type: Literal["report_section_worker"] = "report_section_worker"


def agent_list_default_generator() -> List[AgentMembers]:
    return [
        InitialAnalysis(),
        DataCleaner(),
        Analyst(),
        FileWriter(),
        Visualization(),
        ReportGenerator(),
        SuperVisor(),
    ]


# ---------------------------------------------------------------------------
# LangChain-dependent helpers
# ---------------------------------------------------------------------------

if HAS_LANGCHAIN:
    from langchain_openai.chat_models.base import (
        _construct_responses_api_input,
        _is_pydantic_class,
        _convert_message_to_dict,
        _convert_to_openai_response_format,
        _get_last_messages,
    )

    def _construct_responses_api_payload(
        messages: Sequence[BaseMessage], payload: dict
    ) -> dict:
        # Rename legacy parameters
        for legacy_token_param in ["max_tokens", "max_completion_tokens"]:
            if legacy_token_param in payload:
                payload["max_output_tokens"] = payload.pop(legacy_token_param)
        if "reasoning_effort" in payload and "reasoning" not in payload:
            payload["reasoning"] = {"effort": payload.pop("reasoning_effort")}

        model = payload.get("model", "")
        if model.startswith("gpt-5"):
            payload.pop("temperature", None)

        payload["input"] = _construct_responses_api_input(messages)
        if tools := payload.pop("tools", None):
            new_tools: list = []
            for tool in tools:
                if tool["type"] == "function" and "function" in tool:
                    new_tools.append({"type": "function", **tool["function"]})
                else:
                    if tool["type"] == "image_generation":
                        if "partial_images" in tool:
                            raise NotImplementedError(
                                "Partial image generation is not yet supported "
                                "via the LangChain ChatOpenAI client. Please "
                                "drop the 'partial_images' key from the image_generation "
                                "tool."
                            )
                        elif payload.get("stream") and "partial_images" not in tool:
                            tool["partial_images"] = 1
                        else:
                            pass
                    new_tools.append(tool)
            payload["tools"] = new_tools

        if tool_choice := payload.pop("tool_choice", None):
            if (
                isinstance(tool_choice, dict)
                and tool_choice["type"] == "function"
                and "function" in tool_choice
            ):
                payload["tool_choice"] = {"type": "function", **tool_choice["function"]}
            else:
                payload["tool_choice"] = tool_choice

        if schema := payload.pop("response_format", None):
            existing_text = payload.pop("text", None)
            strict = payload.pop("strict", None)
            if not payload.get("stream") and _is_pydantic_class(schema):
                verbosity = payload.pop("verbosity", None)
                payload["text_format"] = schema
                text_content = (
                    existing_text.copy() if isinstance(existing_text, dict) else {}
                )
                if verbosity is not None:
                    text_content["verbosity"] = verbosity
                if text_content and "format" not in text_content:
                    payload["text"] = text_content
            else:
                if _is_pydantic_class(schema):
                    schema_dict = schema.model_json_schema()
                    strict = True
                else:
                    schema_dict = schema
                if schema_dict == {"type": "json_object"}:
                    structured_text = {"format": {"type": "json_object"}}
                elif (
                    (
                        response_format := _convert_to_openai_response_format(
                            schema_dict, strict=strict
                        )
                    )
                    and (isinstance(response_format, dict))
                    and (response_format["type"] == "json_schema")
                ):
                    structured_text = {
                        "format": {
                            "type": "json_schema",
                            **response_format["json_schema"],
                        }
                    }
                else:
                    structured_text = {}

                if existing_text or structured_text:
                    merged_text = {}
                    if existing_text and isinstance(existing_text, dict):
                        merged_text.update(existing_text)
                    if structured_text:
                        merged_text.update(structured_text)
                    payload["text"] = merged_text

                verbosity = payload.pop("verbosity", None)
                if verbosity is not None:
                    if "text" not in payload:
                        payload["text"] = {"format": {"type": "text"}}
                    payload["text"]["verbosity"] = verbosity
        else:
            verbosity = payload.pop("verbosity", None)
            if verbosity is not None:
                if "text" not in payload:
                    payload["text"] = {"format": {"type": "text"}}
                payload["text"]["verbosity"] = verbosity

        return payload

    class MyChatOpenai(ChatOpenAI):

        def _get_request_payload_mod(
            self,
            input_: LanguageModelInput,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
        ) -> dict:
            messages = self._convert_input(input_).to_messages()
            if stop is not None:
                kwargs["stop"] = stop

            payload = {**self._default_params, **kwargs}

            if self._use_responses_api(payload):
                if self.use_previous_response_id:
                    last_messages, previous_response_id = _get_last_messages(messages)
                    payload_to_use = last_messages if previous_response_id else messages
                    if previous_response_id:
                        payload["previous_response_id"] = previous_response_id
                    payload = _construct_responses_api_payload(payload_to_use, payload)
                else:
                    payload = _construct_responses_api_payload(messages, payload)
            else:
                payload["messages"] = [_convert_message_to_dict(m) for m in messages]
            return payload

        def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: Optional[list[str]] = None,
            **kwargs: Any,
        ) -> dict:
            payload = self._get_request_payload_mod(input_, stop=stop, **kwargs)
            if "max_tokens" in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")

            if self.model_name and re.match(r"^o\d", self.model_name):
                for message in payload.get("messages", []):
                    if message["role"] == "system":
                        message["role"] = "developer"
            return payload


# ---------------------------------------------------------------------------
# DataFrame Registry (Canonical implementation in idd_registry.py)
# ---------------------------------------------------------------------------


def get_global_df_registry() -> DataFrameRegistry:
    """Return the active DataFrameRegistry instance."""
    return get_global_registry()


# ---------------------------------------------------------------------------
# Default config and class-to-agent mapping
# ---------------------------------------------------------------------------

default_an_config = AnalysisConfig(
    default_visualization_style="seaborn-v0_8-whitegrid",
    report_author="Your Name",
    datetime_format_preference="%Y-%m-%d %H:%M:%S",
    large_dataframe_preview_rows=5,
    expect_reply=False,
    reply_msg_to_supervisor="No message",
    finished_this_task=True,
)

CLASS_TO_AGENT: dict[type, AgentId] = {
    InitialDescription: "initial_analysis",
    CleaningMetadata: "data_cleaner",
    AnalysisInsights: "analyst",
    VisualizationResults: "viz_join",
    DataVisualization: "viz_worker",
    VizFeedback: "viz_evaluator",
    SectionOutline: "report_orchestrator",
    Section: "report_section_worker",
    ReportOutline: "report_orchestrator",
    ReportResults: "report_packager",
    FileResult: "file_writer",
}

# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Framework / environment
    "HAS_LANGCHAIN",
    "WORKING_DIRECTORY",
    "default_an_config",
    "CLASS_TO_AGENT",
    # Local types and classes
    "Literal",
    "Field",
    "BaseModel",
    "ConfigDict",
    "field_validator",
    "model_validator",
    "ValidationError",
    "Array1D",
    "is_1d_vector",
    "Number",
    "ScalarNum",
    "Estimator",
    "BinSpec",
    "BinWidthSpec",
    "RangeSpec",
    "ColumnSelector",
    "InitialAnalysis",
    "DataCleaner",
    "Analyst",
    "FileWriter",
    "Visualization",
    "ReportGenerator",
    "SuperVisor",
    "ReportOrchestrator",
    "ReportSection",
    "agent_list_default_generator",
    "MyChatOpenai",
    # Re-exports: idd_models (CP1)
    "BaseNoExtrasModel",
    "ProgressReport",
    "PlanStep",
    "Plan",
    "CompletedStepsAndTasks",
    "Triplet",
    "_norm",
    "_triplet_from_raw",
    "_sort_plan_steps",
    "_assert_sorted_completed_no_dups",
    "_reduce_plan_keep_sorted",
    # Re-exports: idd_registry (CP1)
    "DataFrameRegistry",
    "DataFrameRegistryError",
    "get_global_registry",
    "set_global_registry",
    "override_global_registry",
    # Re-exports: idd_tools (CP2)
    "_get_artifacts_base",
    "_is_subpath",
    "_resolve_artifact_path",
    "_tool_error",
    "_tool_failure",
    "handle_tool_errors",
    "is_safe_artifact_path",
    "validate_dataframe_exists",
    # Re-exports: idd_state (CP3)
    "State",
    "VizWorkerState",
    "keep_first",
    "_sr_reducer",
    "_reduce_strs_resettable",
    "_reduce_viz_results_resettable",
    "_keep_last_or_clear",
    "merge_int_sum",
    "bool_or",
    "dict_merge_shallow",
    "merge_lists",
    "merge_unique",
    "merge_dicts",
    "merge_dict",
    "any_true",
    "last_wins",
    "AgentId",
    "Supervisor",
    "AgentOrSupervisor",
    "AgentMembers",
    "AnalysisConfig",
    "CleaningMetadata",
    "InitialDescription",
    "VizSpec",
    "AnalysisInsights",
    "ImagePayload",
    "DataVisualization",
    "VisualizationResults",
    "ReportResults",
    "DataQueryParams",
    "QueryDataframeInput",
    "FileResult",
    "ListOfFiles",
    "ToDoList",
    "NextAgentMetadata",
    "SendAgentMessage",
    "MessagesToAgentsList",
    "Section",
    "SectionOutline",
    "ReportOutline",
    "VizFeedback",
    "ConversationalResponse",
    # Re-exports: idd_graph (CP3)
    "build_graph",
    "AGENT_MEMBERS",
    "AGENT_OPTIONS",
    "options",
    "PRODUCTION_NODE_NAMES",
    "Router",
    "route_from_supervisor",
    "route_to_writer",
    "route_viz",
    "write_output_to_file",
    "assign_viz_workers",
    "dispatch_sections",
    # LangChain message classes
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
    "BaseMessage",
    "AnyMessage",
    "add_messages",
]


# ---------------------------------------------------------------------------
# Module getattr for backward-compatible dynamic registry resolution
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    if name == "global_df_registry":
        return get_global_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + ["global_df_registry"])
