"""
idd_core.py — Sanitized, importable core extracted from
IntelligentDataDetective_beta_v5.ipynb for testing purposes.
Shell magics and Colab-specific code are removed.
Bug fix: CompletedStepsAndTasks._inject_and_dedupe now returns dedup_list (sorted)
instead of list(seen.values()) (unsorted).
"""
from __future__ import annotations
import os, sys, re, json, uuid, hashlib, shutil, logging, functools
import itertools, threading, operator, base64, tempfile, math
from functools import wraps, lru_cache
from io import StringIO, BytesIO
from pathlib import Path as PathlibPath
from collections import OrderedDict
from collections.abc import Sequence
from typing import (
    Dict, Optional, List, Tuple, Union, Literal, Any, Mapping, MutableMapping,
    cast, TypeGuard, Iterable, Callable
)
from typing_extensions import TypedDict, NotRequired, Annotated, TypeAlias
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas.api.types import is_list_like
from pydantic import (
    BaseModel, Field, model_validator, field_validator, ValidationError,
    ConfigDict, AfterValidator, ValidationInfo, PrivateAttr
)
from typing import List, ClassVar
from operator import add, or_ as bool_or

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

# Optional LangChain imports (needed for MyChatOpenai and State only)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage,
        AnyMessage
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
# Working directory
# ---------------------------------------------------------------------------
import tempfile as _tempfile
_WORKING_DIR_PATH = PathlibPath(_tempfile.mkdtemp(prefix="idd_core_"))
WORKING_DIRECTORY = _WORKING_DIR_PATH

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
    base = PathlibPath(base_env).expanduser() if base_env else PathlibPath.cwd() / "IDD_results"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _is_relative_to(a: PathlibPath, b: PathlibPath) -> bool:
    try:
        return a.resolve().is_relative_to(b.resolve())  # py>=3.9
    except AttributeError:
        ar, br = str(a.resolve()), str(b.resolve())
        return ar.startswith(br)

# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------

def keep_first(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    """Reducer to preserve the first non-null value."""
    return a if a is not None else b


def dict_merge_shallow(old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge two dicts shallowly (one level)."""
    if old is None and new is None:
        return {}
    if old is None:
        return dict(new)
    if new is None:
        return dict(old)
    return {**old, **new}

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


Number    = Union[int, float]
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
    "A scalar or sequence of widths"
]
RangeSpec = Annotated[
    Optional[Tuple[Number, Number]],
    "(lo, hi) numeric tuple",
]
ColumnSelector = Annotated[
    Optional[Union[str, int, Sequence[str], Sequence[int], Literal["all"]]],
    "Column(s) to select",
]


class AgentMembers(BaseModel):
    description: str = Field(default="Members of an agent list.")
    agent_type: Literal[
        "initial_analysis", "data_cleaner", "analyst", "file_writer",
        "visualization", "report_orchestrator", "report_section_worker",
        "report_packager", "supervisor"
    ]

    @model_validator(mode="wrap")
    @classmethod
    def log_failed_validation(cls, data, handler):
        try:
            return handler(data)
        except ValidationError:
            print(f"[AgentMembers] validation failed: {data}")
            raise


class InitialAnalysis(AgentMembers):   agent_type: Literal["initial_analysis"]    = "initial_analysis"
class DataCleaner(AgentMembers):        agent_type: Literal["data_cleaner"]        = "data_cleaner"
class Analyst(AgentMembers):            agent_type: Literal["analyst"]             = "analyst"
class FileWriter(AgentMembers):         agent_type: Literal["file_writer"]         = "file_writer"
class Visualization(AgentMembers):      agent_type: Literal["visualization"]       = "visualization"
class ReportGenerator(AgentMembers):    agent_type: Literal["report_packager"]     = "report_packager"
class SuperVisor(AgentMembers):         agent_type: Literal["supervisor"]          = "supervisor"
class ReportOrchestrator(AgentMembers): agent_type: Literal["report_orchestrator"] = "report_orchestrator"
class ReportSection(AgentMembers):      agent_type: Literal["report_section_worker"] = "report_section_worker"


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


AgentId: TypeAlias = Literal[
    "initial_analysis", "data_cleaner", "analyst",
    "viz_worker", "viz_join", "viz_evaluator", "visualization",
    "report_orchestrator", "report_section_worker", "report_join",
    "report_packager", "file_writer", "END", "FINISH"
]
Supervisor = Literal["supervisor"]
AgentOrSupervisor = Union[AgentId, Supervisor]

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
                        "format": {"type": "json_schema", **response_format["json_schema"]}
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
# Pydantic models
# ---------------------------------------------------------------------------

class AnalysisConfig(BaseNoExtrasModel):
    """User-configurable settings for the data analysis workflow."""
    default_visualization_style: str = Field(..., description="Default style for matplotlib/seaborn visualizations. seaborn-v0_8-whitegrid is a decent choice if unsure.")
    report_author: str = Field(..., description="Author name to include in generated reports.")
    datetime_format_preference: str = Field(..., description="Preferred format for datetime string representations. If unsure, use %Y-%m-%d %H:%M:%S")
    large_dataframe_preview_rows: int = Field(..., description="Number of rows for previewing large dataframes. Use 5 if unsure.")


class CleaningMetadata(BaseNoExtrasModel):
    """Metadata about the data cleaning actions taken."""
    steps_taken: list[str] = Field(..., description="List of cleaning steps performed.")
    data_description_after_cleaning: str = Field(..., description="Brief description of the dataset after cleaning.")


class InitialDescription(BaseNoExtrasModel):
    """Initial description of the dataset."""
    dataset_description: str = Field(..., description="Brief description of the dataset.")
    data_sample: str = Field(..., description="Sample of the dataset.")
    notes: str = Field(..., description="Notes about the dataset.")


class VizSpec(BaseNoExtrasModel):
    title: Optional[Union[str, None]] = Field(..., description="Title of the visualization.")
    viz_type: Optional[Union[Literal["histogram", "scatter", "bar", "line", "box", "auto"], None]] = Field(..., description="Type of the visualization. Set to 'auto' to let the agent decide.")
    df_id: Optional[Union[str, None]] = Field(..., description="ID of the DataFrame to visualize.")
    viz_instructions: Optional[Union[str, None]] = Field(..., description="Instructions to the next agent for the visualization.")
    viz_id: Optional[Union[str, None]] = Field(..., description="ID of the visualization. If not provided, the agent will generate one. Must be unique.")
    columns: Optional[Union[List[str], None]] = Field(..., description="Optional list of columns to visualize.")
    x: Optional[Union[str, None]] = Field(..., description="Optional column to use for the x-axis.")
    y: Optional[Union[str, None]] = Field(..., description="Optional column to use for the y-axis.")
    hue: Optional[str] = Field(..., description="Optional column to use for the hue.")
    bins: Optional[Union[int, str, None]] = Field(..., description="Optional number of bins or method for binning.")
    agg: Optional[Union[str, None]] = Field(..., description="Optional aggregation method.")
    query: Optional[Union[str, None]] = Field(..., description="Optional query to filter the data.")
    description: Optional[Union[str, None]] = Field(..., description="Optional description of the visualization.")
    limit: Optional[Union[int, None]] = Field(..., description="Optional limit of rows to visualize.")
    style: Optional[Union[str, None]] = Field(..., description="Optional style of the visualization. This should be a matplotlib/seaborn style.")


class AnalysisInsights(BaseNoExtrasModel):
    """Insights from the exploratory data analysis."""
    summary: str = Field(..., description="Overall summary of EDA findings.")
    correlation_insights: str = Field(..., description="Key correlation insights identified.")
    anomaly_insights: str = Field(..., description="Anomalies or interesting patterns detected.")
    recommended_visualizations: List[VizSpec] = Field(...)
    recommended_next_steps: List[str] = Field(..., description="List of recommended next analysis steps or questions to investigate based on the findings.")


class ImagePayload(BaseNoExtrasModel):
    """Wrap both the image bytes and its declared MIME-type."""
    mime: Literal["image/png", "image/jpeg"]
    payload: bytes

    @field_validator("payload", mode="before")
    def ensure_b64(cls, v: str | bytes):
        if isinstance(v, bytes):
            return v
        try:
            return base64.b64decode(v, validate=True)
        except Exception as e:
            raise ValueError("Invalid Base-64") from e

    @field_validator("payload")
    def enforce_size(cls, v: bytes):
        max_bytes = 2 * 1024 * 1024
        if len(v) > max_bytes:
            raise ValueError(f"Image is too large ({len(v)} bytes > {max_bytes})")
        return v


class DataVisualization(BaseNoExtrasModel):
    """Individual visualizations generated"""
    path: str = Field(description="Path to the generated visualization.")
    visualization_id: str = Field(..., description="Unique ID for the visualization.")
    visualization_type: str = Field(..., description="Type of the visualization.")
    visualization_description: str = Field(..., description="Description of the visualization.")
    visualization_style: str = Field(..., description="Style of the visualization.")
    visualization_title: str = Field(..., description="Title of the visualization.")


class VisualizationResults(BaseNoExtrasModel):
    """Results from the visualization generation."""
    visualizations: List[DataVisualization] = Field(...)


class ReportResults(BaseNoExtrasModel):
    """Results from the report generation."""
    pdf_report_path: str = Field(..., description="Path to the generated PDF report.")
    html_report_path: str = Field(..., description="Path to the generated HTML report.")
    markdown_report_path: str = Field(..., description="Path to the generated Markdown report.")


class DataQueryParams(BaseModel):
    """Parameters for querying the DataFrame."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"additionalProperties": False})
    columns: List[str] = Field(..., description="List of columns to include in the output")
    filter_column: str = Field(None, description="Column to apply the filter on")
    filter_value: str = Field(None, description="Value to filter the rows by")
    operation: str = Field(..., description="Operation to perform: 'select', 'sum', 'mean', 'count', 'max', 'min', 'median', etc.")


class QueryDataframeInput(BaseModel):
    """Args schema to query a registered DataFrame by columns, optional equality filter, and an operation."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"additionalProperties": False})
    params: DataQueryParams
    df_id: str = Field(..., description="ID of the DataFrame in the registry")


class FileResult(BaseNoExtrasModel):
    """Results object storing metadata from the file generation or editing."""
    write_success: bool = Field(..., description="Whether the file was written to disk successfully.")
    file_path: str = Field(..., description="Path to the generated file.")
    file_type: str = Field(..., description="Type of the generated file.")
    file_name: str = Field(..., description="Name of the generated file.")
    file_description: str = Field(..., description="Description of the generated file.")
    is_final_report: bool = Field(..., description="Whether the file is the finalized report.")
    category_tag: str = Field(..., description="Type of the resulting file: 'report', 'data', 'visualization', or 'other'.")


class ListOfFiles(BaseNoExtrasModel):
    """List of metadata as FileResult objects for the files generated."""
    files: List[FileResult] = Field(...)


# Core planning models (PlanStep, Plan, CompletedStepsAndTasks, ProgressReport,
# DataFrameRegistryError) are canonical in idd_models.py and idd_registry.py.



class ToDoList(BaseNoExtrasModel):
    to_do_list: List[str] = Field(..., description="List of tasks to be done based on the steps of the Plan")


class NextAgentMetadata(BaseNoExtrasModel):
    df_id: Optional[str] = Field(..., description="Optional DataFrame ID to supply to the next agent.")
    file_type: Optional[str] = Field(..., description="Optional field for specifying a file type.")
    file_name: Optional[str] = Field(..., description="Optional field for communicating a file name.")
    section_name: Optional[str] = Field(..., description="Optional field for specifying a section name.")
    viz_spec: Optional[Union[VizSpec, None]] = Field(...)
    notes: Optional[str] = Field(..., description="Optional field for communicating notes.")
    file_content: Optional[str] = Field(..., description="Optional field for communicating file content.")


class SendAgentMessage(BaseNoExtrasModel):
    recipient: AgentOrSupervisor = Field(...)
    message: str = Field(..., description="Message to send to recipient agent")
    delivery_status: bool = Field(..., description="Whether the message was successfully delivered.")
    agent_obj_needs_recreated_bool: bool = Field(..., description="Whether the object needs to be recreated by the recipient agent.")
    is_message_critical: bool = Field(..., description="Whether the message is critical.")
    immediate_emergency_reroute_to_recipient: bool = Field(..., description="Whether to immediately reroute the message.")


class MessagesToAgentsList(BaseNoExtrasModel):
    messages_to_agents: List[SendAgentMessage] = Field(...)


class Section(BaseNoExtrasModel):
    name: str = Field(..., description="Section name")
    section_num: int = Field(..., description="Section number")
    description: str = Field(..., description="What this section covers")
    goals: List[str] = Field(..., description="List of goals for this section")
    data_signals: List[str] = Field(..., description="List of data signals used for this section")
    expected_figures: List[DataVisualization] = Field(...)
    content: str = Field(..., description="Content of the section")


class SectionOutline(BaseNoExtrasModel):
    name: str = Field(..., description="Section name/title")
    section_num: int = Field(..., description="Section number")
    description: str = Field(..., description="What this section covers")
    goals: List[str] = Field(..., description="List of goals for this section.")
    data_signals_needed: Dict[str, str] = Field(..., description="List of data signals needed for this section.")
    data_signals_available: List[str] = Field(..., description="List of data signals available for this section.")
    expected_figures: List[DataVisualization] = Field(...)
    word_target: int = Field(..., description="Approx length target. 300 is a good standard length target per section")


class ReportOutline(SectionOutline):
    title: str = Field(..., description="Title of the report")
    description: str = Field(..., description="Description of the report")
    goals: List[str] = Field(..., description="List of goals for the report")
    sections: List[SectionOutline] = Field(...)


class VizFeedback(BaseNoExtrasModel):
    grade: Literal["acceptable", "revise"] = Field(..., description="Overall judgment of visualizations")
    feedback: str = Field(..., description="Concrete advice if 'revise'")
    redo_list: List[str] = Field(..., description="List of visualizations to redo")


class ConversationalResponse(BaseNoExtrasModel):
    response: str = Field(..., description="A conversational response to the supervisors message.")

# ---------------------------------------------------------------------------
# DataFrame Registry (Canonical implementation in idd_registry.py)
# ---------------------------------------------------------------------------

def get_global_df_registry() -> DataFrameRegistry:
    """Return the active DataFrameRegistry instance."""
    return get_global_registry()


# ---------------------------------------------------------------------------
# Additional reducers (depend on Plan/PlanStep, so placed after them)
# ---------------------------------------------------------------------------

def merge_lists(a: list | None, b: list | None) -> list:
    return (a or []) + (b or [])


def merge_unique(a: list[str] | None, b: list[str] | None) -> list[str]:
    return list(dict.fromkeys((a or []) + (b or [])))


def merge_int_sum(a: int | None, b: int | None) -> int:
    return int(a or 0) + int(b or 0)


def merge_dicts(a: Dict | None, b: Dict | None) -> Dict:
    d = {}
    if a: d.update(a)
    if b: d.update(b)
    return d


def merge_dict(a: Optional[dict], b: Optional[dict]) -> dict:
    return {**(a or {}), **(b or {})}


def any_true(a: Optional[bool], b: Optional[bool]) -> bool:
    return bool(a) or bool(b)


def last_wins(a, b):
    return b


# _reduce_plan_keep_sorted is canonical in idd_models.py

# State TypedDict is intentionally excluded from idd_core.
# It requires langgraph.graph.message.add_messages and inherits from AgentState,
# which makes it incompatible with unit test imports.
# Import State directly from the production script when needed for integration tests.

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
# Path helper functions
# ---------------------------------------------------------------------------

if HAS_LANGCHAIN:
    def _get_artifacts_base(config: Optional[RunnableConfig]) -> PathlibPath:
        """Resolve the base directory for artifacts."""
        try:
            cfg = getattr(config, "configurable", None) or {}
            runtime = cfg.get("runtime")
            if runtime is not None and getattr(runtime, "artifacts_dir", None):
                base = PathlibPath(runtime.artifacts_dir)
                base.mkdir(parents=True, exist_ok=True)
                return base
        except Exception:
            pass

        try:
            if "RUNTIME" in globals() and getattr(globals()["RUNTIME"], "artifacts_dir", None):
                base = PathlibPath(globals()["RUNTIME"].artifacts_dir)
                base.mkdir(parents=True, exist_ok=True)
                return base
        except Exception:
            pass

        env_artifacts = os.environ.get("IDD_ARTIFACTS_DIR")
        if env_artifacts:
            base = PathlibPath(env_artifacts)
            base.mkdir(parents=True, exist_ok=True)
            return base

        base = PathlibPath(WORKING_DIRECTORY) / "artifacts"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _resolve_artifact_path(
        file_name: str,
        *,
        config: Optional[RunnableConfig],
        subdir: Optional[str] = None,
        create_parents: bool = True,
    ) -> PathlibPath:
        """Resolve file_name relative to (or validated within) the artifacts dir."""
        if not file_name or not isinstance(file_name, str):
            raise ValueError("file_name must be a non-empty string.")

        base = _get_artifacts_base(config)
        if subdir:
            base = (base / subdir).resolve()

        base.mkdir(parents=True, exist_ok=True)

        candidate = PathlibPath(file_name)
        path = (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

        if not _is_subpath(path, base):
            raise ValueError(f"Refusing to access path outside artifacts root: {path}")

        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _is_subpath(path: PathlibPath, parent: PathlibPath) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def validate_dataframe_exists(df_id: str) -> bool:
    """Validates the existence and validity of a dataframe by its ID."""
    if not df_id or not isinstance(df_id, str):
        return False

    try:
        reg = get_global_registry()
        df = reg.get_dataframe(df_id)
        if df is not None:
            return not df.empty

        raw_path = reg.get_raw_path_from_id(df_id)
        if raw_path and os.path.exists(raw_path):
            try:
                df = pd.read_csv(raw_path)
                if df is not None and not df.empty:
                    reg.register_dataframe(df, df_id, raw_path)
                    return True
            except Exception:
                return False

        return False
    except Exception:
        return False


def handle_tool_errors(func):
    """Decorator for consistent error handling across tool functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            df_id = None

            if args and isinstance(args[0], str):
                df_id = args[0]
            elif 'df_id' in kwargs:
                df_id = kwargs['df_id']
            elif args and hasattr(args[0], 'df_id'):
                df_id = args[0].df_id

            if df_id and not validate_dataframe_exists(df_id):
                error_msg = f"Error: DataFrame with ID '{df_id}' not found or is invalid."
                logging.error(f'{func.__name__}: {error_msg}')
                return error_msg

            return func(*args, **kwargs)

        except FileNotFoundError as e:
            error_msg = f"Error: File not found - {str(e)}"
            logging.error(f"{func.__name__}: {error_msg}")
            return error_msg

        except KeyError as e:
            error_msg = f"Error: Column or key '{str(e)}' not found"
            logging.error(f"{func.__name__}: {error_msg}")
            return error_msg

        except pd.errors.EmptyDataError:
            error_msg = "Error: No data - the DataFrame or file is empty"
            logging.error(f"{func.__name__}: {error_msg}")
            return error_msg

        except pd.errors.ParserError as e:
            error_msg = f"Error: Failed to parse data - {str(e)}"
            logging.error(f"{func.__name__}: {error_msg}")
            return error_msg

        except pd.errors.DtypeWarning as e:
            error_msg = f"Error: Data type mismatch - {str(e)}"
            logging.error(f"{func.__name__}: {error_msg}")
            return error_msg

        except ValueError as e:
            error_msg = f"Error: Invalid value - {str(e)}"
            logging.error(f"{func.__name__}: {error_msg}")
            return error_msg

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logging.error(f"{func.__name__}: {error_msg}")
            return error_msg

    return wrapper


# ---------------------------------------------------------------------------
# Module getattr for backward-compatible dynamic registry resolution
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    if name == "global_df_registry":
        return get_global_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + ["global_df_registry"])
