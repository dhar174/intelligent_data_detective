"""
idd_state.py — Canonical LangGraph State Schema and Reducers for Intelligent Data Detective.

Home of:
- Canonical 70-field State(TypedDict, total=False)
- VizWorkerState(TypedDict)
- Custom and standard reducers:
  keep_first, _reduce_plan_keep_sorted, _sr_reducer, _reduce_strs_resettable,
  _reduce_viz_results_resettable, _keep_last_or_clear, merge_int_sum, bool_or,
  dict_merge_shallow, merge_lists, merge_unique, merge_dicts, merge_dict,
  any_true, last_wins
- Agent typing: AgentId, Supervisor, AgentOrSupervisor, AgentMembers
- Domain schemas: AnalysisConfig, CleaningMetadata, InitialDescription,
  VizSpec, DataVisualization, VisualizationResults, ReportResults,
  DataQueryParams, QueryDataframeInput, FileResult, ListOfFiles,
  ToDoList, NextAgentMetadata, SendAgentMessage, MessagesToAgentsList,
  Section, SectionOutline, ReportOutline, VizFeedback, ConversationalResponse

Architecture: Checkpoint 3 (Issue #160 / Refs #140, #144, #145).
"""

from __future__ import annotations

import base64
import operator
from pathlib import Path as PathlibPath
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypeAlias,
    Union,
)
from typing_extensions import Annotated, TypedDict

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Optional LangChain / LangGraph imports with graceful fallbacks
try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AnyMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.runnables.config import RunnableConfig
    from langgraph.graph.message import add_messages

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseChatModel = object  # type: ignore[misc,assignment]
    AIMessage = Any  # type: ignore[misc,assignment]
    AnyMessage = Any  # type: ignore[misc,assignment]
    BaseMessage = Any  # type: ignore[misc,assignment]
    HumanMessage = Any  # type: ignore[misc,assignment]
    SystemMessage = Any  # type: ignore[misc,assignment]
    ToolMessage = Any  # type: ignore[misc,assignment]
    RunnableConfig = dict  # type: ignore[misc,assignment]

    def add_messages(a: Any, b: Any) -> list:  # type: ignore[misc]
        return (a or []) + (b or [])


# Canonical planning models and step synchronization from idd_models.py (CP1)
from idd_models import (
    BaseNoExtrasModel,
    Plan,
    PlanStep,
    _assert_sorted_completed_no_dups,
    _reduce_plan_keep_sorted,
)


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def keep_first(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    """Reducer to preserve the first non-null value."""
    return a if a is not None else b


def _reduce_strs_resettable(
    a: Optional[List[str]], b: Optional[List[str]]
) -> List[str]:
    """Append-by-default; pass None to clear."""
    if b is None:
        return []
    return (a or []) + (b or [])


def _reduce_viz_results_resettable(
    a: Optional[List[dict]], b: Optional[List[dict]]
) -> List[dict]:
    """Append-by-default; pass None to clear."""
    if b is None:
        return []
    return (a or []) + (b or [])


def _keep_last_or_clear(a: Any, b: Any) -> Any:
    """Always overwrite; b=None acts as explicit clear."""
    return b


def _sr_reducer(left: Any, right: Any) -> Any:
    """Last-writer-wins; preserve prior non-None when right is None."""
    if right is None and left is not None:
        return left
    return right


def merge_int_sum(a: Optional[int], b: Optional[int]) -> int:
    """Sum integers; treat None as 0."""
    return int(a or 0) + int(b or 0)


def bool_or(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    """Safe boolean-OR reducer: treats None as False, avoids TypeError."""
    return b if a is None else (a if b is None else bool(a | b))


def dict_merge_shallow(
    old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge two dicts shallowly (one level); new keys override old."""
    if old is None and new is None:
        return {}
    if old is None:
        return dict(new)
    if new is None:
        return dict(old)
    return {**old, **new}


def merge_lists(a: Optional[list], b: Optional[list]) -> list:
    """Concatenate two lists, treating None as empty list."""
    return (a or []) + (b or [])


def merge_unique(a: Optional[List[str]], b: Optional[List[str]]) -> List[str]:
    """Concatenate two lists and deduplicate preserving order."""
    return list(dict.fromkeys((a or []) + (b or [])))


def merge_dicts(a: Optional[Dict], b: Optional[Dict]) -> Dict:
    """Merge two dicts into a new dict."""
    d: Dict = {}
    if a:
        d.update(a)
    if b:
        d.update(b)
    return d


def merge_dict(a: Optional[dict], b: Optional[dict]) -> dict:
    """Merge two dicts shallowly."""
    return {**(a or {}), **(b or {})}


def any_true(a: Optional[bool], b: Optional[bool]) -> bool:
    """Return True if either argument is truthy."""
    return bool(a) or bool(b)


def last_wins(a: Any, b: Any) -> Any:
    """Always return the second value."""
    return b


# ---------------------------------------------------------------------------
# Agent Typing & Members
# ---------------------------------------------------------------------------

AgentId: TypeAlias = Literal[
    "initial_analysis",
    "data_cleaner",
    "analyst",
    "viz_worker",
    "viz_join",
    "viz_evaluator",
    "visualization",
    "report_orchestrator",
    "report_section_worker",
    "report_join",
    "report_packager",
    "file_writer",
    "END",
    "FINISH",
]

Supervisor = Literal["supervisor"]
AgentOrSupervisor = Union[AgentId, Supervisor]


class AgentMembers(BaseModel):
    """Pydantic model representing agent membership."""

    description: str = Field(default="Members of an agent list.")
    agent_type: Literal[
        "initial_analysis",
        "data_cleaner",
        "analyst",
        "file_writer",
        "visualization",
        "report_orchestrator",
        "report_section_worker",
        "report_packager",
        "supervisor",
    ]

    @model_validator(mode="wrap")
    @classmethod
    def log_failed_validation(cls, data, handler):
        try:
            return handler(data)
        except Exception:
            raise


# ---------------------------------------------------------------------------
# Domain Schemas & Structured Response Contracts
# ---------------------------------------------------------------------------


class AnalysisConfig(BaseNoExtrasModel):
    default_visualization_style: str = Field(
        ...,
        description="The default style to use for matplotlib/seaborn plots.",
    )
    report_author: str = Field(
        ..., description="Author name to include in generated reports."
    )
    datetime_format_preference: str = Field(
        ...,
        description="Preferred format for datetime string representations. If unsure, use %Y-%m-%d %H:%M:%S",
    )
    large_dataframe_preview_rows: int = Field(
        ...,
        description="Number of rows for previewing large dataframes. Use 5 if unsure.",
    )


class CleaningMetadata(BaseNoExtrasModel):
    """Metadata about the data cleaning actions taken."""

    steps_taken: list[str] = Field(..., description="List of cleaning steps performed.")
    data_description_after_cleaning: str = Field(
        ..., description="Brief description of the dataset after cleaning."
    )


class InitialDescription(BaseNoExtrasModel):
    """Initial description of the dataset."""

    dataset_description: str = Field(
        ..., description="Brief description of the dataset."
    )
    data_sample: str = Field(..., description="Sample of the dataset.")
    notes: str = Field(..., description="Notes about the dataset.")


class VizSpec(BaseNoExtrasModel):
    title: Optional[Union[str, None]] = Field(
        ..., description="Title of the visualization."
    )
    viz_type: Optional[
        Union[Literal["histogram", "scatter", "bar", "line", "box", "auto"], None]
    ] = Field(
        ...,
        description="Type of the visualization. Set to 'auto' to let the agent decide.",
    )
    df_id: Optional[Union[str, None]] = Field(
        ..., description="ID of the DataFrame to visualize."
    )
    viz_instructions: Optional[Union[str, None]] = Field(
        ...,
        description="Instructions to the next agent for the visualization.",
    )
    viz_id: Optional[Union[str, None]] = Field(
        ...,
        description="ID of the visualization. If not provided, the agent will generate one. Must be unique.",
    )
    columns: Optional[Union[List[str], None]] = Field(
        ..., description="Optional list of columns to visualize."
    )
    x: Optional[Union[str, None]] = Field(
        ..., description="Optional column to use for the x-axis."
    )
    y: Optional[Union[str, None]] = Field(
        ..., description="Optional column to use for the y-axis."
    )
    hue: Optional[str] = Field(..., description="Optional column to use for the hue.")
    bins: Optional[Union[int, str, None]] = Field(
        ..., description="Optional number of bins or method for binning."
    )
    agg: Optional[Union[str, None]] = Field(
        ..., description="Optional aggregation method."
    )
    query: Optional[Union[str, None]] = Field(
        ..., description="Optional query to filter the data."
    )
    description: Optional[Union[str, None]] = Field(
        ..., description="Optional description of the visualization."
    )
    limit: Optional[Union[int, None]] = Field(
        ..., description="Optional limit of rows to visualize."
    )
    style: Optional[Union[str, None]] = Field(
        ...,
        description="Optional style of the visualization. This should be a matplotlib/seaborn style.",
    )


class AnalysisInsights(BaseNoExtrasModel):
    """Insights from the exploratory data analysis."""

    summary: str = Field(..., description="Overall summary of EDA findings.")
    correlation_insights: str = Field(
        ..., description="Key correlation insights identified."
    )
    anomaly_insights: str = Field(
        ..., description="Anomalies or interesting patterns detected."
    )
    recommended_visualizations: List[VizSpec] = Field(...)
    recommended_next_steps: List[str] = Field(
        ...,
        description="List of recommended next analysis steps or questions to investigate based on the findings.",
    )


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
    """Individual visualizations generated."""

    path: str = Field(description="Path to the generated visualization.")
    visualization_id: str = Field(..., description="Unique ID for the visualization.")
    visualization_type: str = Field(..., description="Type of the visualization.")
    visualization_description: str = Field(
        ..., description="Description of the visualization."
    )
    visualization_style: str = Field(..., description="Style of the visualization.")
    visualization_title: str = Field(..., description="Title of the visualization.")


class VisualizationResults(BaseNoExtrasModel):
    """Results from the visualization generation."""

    visualizations: List[DataVisualization] = Field(...)


class ReportResults(BaseNoExtrasModel):
    """Results from the report generation."""

    pdf_report_path: str = Field(..., description="Path to the generated PDF report.")
    html_report_path: str = Field(..., description="Path to the generated HTML report.")
    markdown_report_path: str = Field(
        ..., description="Path to the generated Markdown report."
    )


class DataQueryParams(BaseModel):
    """Parameters for querying the DataFrame."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"additionalProperties": False}
    )
    columns: List[str] = Field(
        ..., description="List of columns to include in the output"
    )
    filter_column: Optional[str] = Field(
        default=None, description="Column to apply the filter on"
    )
    filter_value: Optional[str] = Field(
        default=None, description="Value to filter the rows by"
    )
    operation: str = Field(
        ...,
        description="Operation to perform: 'select', 'sum', 'mean', 'count', 'max', 'min', 'median', etc.",
    )


class QueryDataframeInput(BaseModel):
    """Args schema to query a registered DataFrame by columns, optional equality filter, and an operation."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"additionalProperties": False}
    )
    params: DataQueryParams
    df_id: str = Field(..., description="ID of the DataFrame in the registry")


class FileResult(BaseNoExtrasModel):
    """Results object storing metadata from the file generation or editing."""

    write_success: bool = Field(
        ..., description="Whether the file was written to disk successfully."
    )
    file_path: str = Field(..., description="Path to the generated file.")
    file_type: str = Field(..., description="Type of the generated file.")
    file_name: str = Field(..., description="Name of the generated file.")
    file_description: str = Field(..., description="Description of the generated file.")
    is_final_report: bool = Field(
        ..., description="Whether the file is the finalized report."
    )
    category_tag: str = Field(
        ...,
        description="Type of the resulting file: 'report', 'data', 'visualization', or 'other'.",
    )


class ListOfFiles(BaseNoExtrasModel):
    """List of metadata as FileResult objects for the files generated."""

    files: List[FileResult] = Field(...)


class ToDoList(BaseNoExtrasModel):
    to_do_list: List[str] = Field(
        ...,
        description="List of tasks to be done based on the steps of the Plan",
    )


class NextAgentMetadata(BaseNoExtrasModel):
    df_id: Optional[str] = Field(
        ..., description="Optional DataFrame ID to supply to the next agent."
    )
    file_type: Optional[str] = Field(
        ..., description="Optional field for specifying a file type."
    )
    file_name: Optional[str] = Field(
        ..., description="Optional field for communicating a file name."
    )
    section_name: Optional[str] = Field(
        ..., description="Optional field for specifying a section name."
    )
    viz_spec: Optional[Union[VizSpec, None]] = Field(...)
    notes: Optional[str] = Field(
        ..., description="Optional field for communicating notes."
    )
    file_content: Optional[str] = Field(
        ..., description="Optional field for communicating file content."
    )


class SendAgentMessage(BaseNoExtrasModel):
    recipient: AgentOrSupervisor = Field(...)
    message: str = Field(..., description="Message to send to recipient agent")
    delivery_status: bool = Field(
        ..., description="Whether the message was successfully delivered."
    )
    agent_obj_needs_recreated_bool: bool = Field(
        ...,
        description="Whether the object needs to be recreated by the recipient agent.",
    )
    is_message_critical: bool = Field(
        ..., description="Whether the message is critical."
    )
    immediate_emergency_reroute_to_recipient: bool = Field(
        ..., description="Whether to immediately reroute the message."
    )


class MessagesToAgentsList(BaseNoExtrasModel):
    messages_to_agents: List[SendAgentMessage] = Field(...)


class Section(BaseNoExtrasModel):
    name: str = Field(..., description="Section name")
    section_num: int = Field(..., description="Section number")
    description: str = Field(..., description="What this section covers")
    goals: List[str] = Field(..., description="List of goals for this section")
    data_signals: List[str] = Field(
        ..., description="List of data signals used for this section"
    )
    expected_figures: List[DataVisualization] = Field(...)
    content: str = Field(..., description="Content of the section")


class SectionOutline(BaseNoExtrasModel):
    name: str = Field(..., description="Section name/title")
    section_num: int = Field(..., description="Section number")
    description: str = Field(..., description="What this section covers")
    goals: List[str] = Field(..., description="List of goals for this section.")
    data_signals_needed: Dict[str, str] = Field(
        ..., description="List of data signals needed for this section."
    )
    data_signals_available: List[str] = Field(
        ..., description="List of data signals available for this section."
    )
    expected_figures: List[DataVisualization] = Field(...)
    word_target: int = Field(
        ...,
        description="Approx length target. 300 is a good standard length target per section",
    )


class ReportOutline(SectionOutline):
    title: str = Field(..., description="Title of the report")
    description: str = Field(..., description="Description of the report")
    goals: List[str] = Field(..., description="List of goals for the report")
    sections: List[SectionOutline] = Field(...)


class VizFeedback(BaseNoExtrasModel):
    grade: Literal["acceptable", "revise"] = Field(
        ..., description="Overall judgment of visualizations"
    )
    feedback: str = Field(..., description="Concrete advice if 'revise'")
    redo_list: List[str] = Field(..., description="List of visualizations to redo")


class ConversationalResponse(BaseNoExtrasModel):
    response: str = Field(
        ...,
        description="A conversational response to the supervisor's message.",
    )


# ---------------------------------------------------------------------------
# Canonical LangGraph State Schema (70 fields)
# ---------------------------------------------------------------------------


class State(TypedDict, total=False):
    """Authoritative 70-field State schema for Intelligent Data Detective.

    Invariants:
    - messages: Annotated[list[AnyMessage], add_messages] (declared directly,
      NEVER inheriting from LangChain AgentState base to prevent remaining_steps
      channel rejection - BR-7).
    - structured_response: Intentionally NOT a channel on supervisor State
      (W9-SR-DROP channel collision prevention). Subgraphs manage their own
      response schema.
    """

    # 1. Chat history & messages
    messages: Annotated[list[AnyMessage], add_messages]

    # 2. Routing
    next: Optional[AgentId]

    # 3-10. Turn accounting & handoffs (use-last reducers)
    last_agent_id: Annotated[Optional[AgentId], lambda a, b: b]
    _viz_retry_count: Optional[int]
    _report_dispatched: Annotated[Optional[bool], bool_or]
    current_turn_agent_id: Annotated[Optional[AgentId], lambda a, b: b]
    last_agent_message: Annotated[
        Optional[Union[AIMessage, ToolMessage]], lambda a, b: b
    ]
    last_agent_expects_reply: Annotated[Optional[bool], lambda a, b: b]
    last_agent_reply_msg: Annotated[Optional[str], lambda a, b: b]
    last_agent_finished_this_task: Annotated[Optional[bool], lambda a, b: b]

    # 11-13. Messages & objects
    last_created_obj: Annotated[Optional[str], lambda a, b: b]
    final_turn_msgs_list: Annotated[
        Optional[list[Union[AIMessage, ToolMessage]]],
        lambda a, b: add_messages(a or [], b or []),
    ]
    supervisor_to_agent_msgs: Annotated[
        Optional[list[SendAgentMessage]], lambda a, b: (a or []) + (b or [])
    ]

    # 14-16. Emergency & prompt
    emergency_reroute: Annotated[Optional[AgentId], _keep_last_or_clear]
    __active_response_tool__: Optional[str]
    user_prompt: str

    # 17. Plan (sorted step deduplication reducer)
    current_plan: Annotated[Optional[Plan], _reduce_plan_keep_sorted]

    # 18-20. DataFrame tracking
    current_dataframe: Optional[str]
    current_dataframe_id: Optional[str]
    available_df_ids: List[str]

    # 21-23. Visualization tasks & results
    viz_tasks: Annotated[List[str], _keep_last_or_clear]
    individual_viz_task: Optional[str]
    viz_results: Annotated[List[dict], _reduce_viz_results_resettable]

    # 24-30. Visualization evaluation & specs
    viz_revise_count: int
    _supervisor_turn_count: Annotated[int, operator.add]
    viz_eval_result: Optional[VizFeedback]
    viz_grade: Optional[str]
    viz_feedback: Optional[str]
    viz_specs: list[VizSpec]
    viz_spec: VizSpec

    # 31-35. Reporting sections & draft
    section: Optional[SectionOutline]
    report_outline: Optional[ReportOutline]
    sections: Annotated[List[Section], operator.add]
    written_sections: Annotated[List[str], _reduce_strs_resettable]
    report_draft: Annotated[Optional[str], lambda a, b: b if b is not None else a]

    # 36-39. Plan steps & tasks
    completed_plan_steps: Annotated[
        List[PlanStep], AfterValidator(_assert_sorted_completed_no_dups)
    ]
    to_do_list: List[str]
    progress_reports: Annotated[List[str], operator.add]
    completed_tasks: Annotated[List[str], operator.add]

    # 40-45. Agents & config
    latest_progress: Optional[str]
    initial_analysis_agent: Optional[BaseChatModel]
    data_cleaner_agent: Optional[BaseChatModel]
    analyst_agent: Optional[BaseChatModel]
    analysis_config: Optional[AnalysisConfig]
    _config: Optional[RunnableConfig]

    # 46-49. Paths & content
    viz_paths: Annotated[Optional[list[str]], operator.add]
    report_paths: Annotated[
        Optional[dict[str, str]],
        lambda _a3, _b3: {
            **(_a3 if isinstance(_a3, dict) else {}),
            **(_b3 if isinstance(_b3, dict) else {}),
        },
    ]
    run_id: Annotated[Optional[str], lambda a, b: b if b is not None else a]
    file_content: Optional[dict[str, str]]

    # 50-51. Supervisor handoffs
    next_agent_prompt: Optional[str]
    next_agent_metadata: Optional[NextAgentMetadata]

    # 52-58. Artifacts & structured results
    initial_description: Optional[InitialDescription]
    cleaning_metadata: Optional[CleaningMetadata]
    analysis_insights: Optional[AnalysisInsights]
    visualization_results: Annotated[Optional[VisualizationResults], lambda a, b: b]
    report_results: Annotated[Optional[ReportResults], _sr_reducer]
    cleaned_dataset_description: Optional[str]
    file_results: Annotated[Optional[List[FileResult]], operator.add]

    # 59-64. Completion flags (boolean OR preserves True once set)
    initial_analysis_complete: Annotated[Optional[bool], bool_or]
    data_cleaning_complete: Annotated[Optional[bool], bool_or]
    analyst_complete: Annotated[Optional[bool], bool_or]
    file_writer_complete: Annotated[Optional[bool], bool_or]
    visualization_complete: Annotated[Optional[bool], bool_or]
    report_generator_complete: Annotated[Optional[bool], bool_or]

    # 65-66. Misc counters & IDs
    _count_: Annotated[int, merge_int_sum]
    _id_: Optional[str]

    # 67-70. Persistent filesystem paths (keep first non-null)
    artifacts_path: Annotated[Optional[Union[str, PathlibPath]], keep_first]
    reports_path: Annotated[Optional[Union[str, PathlibPath]], keep_first]
    logs_path: Annotated[Optional[Union[str, PathlibPath]], keep_first]
    final_report_path: Annotated[Optional[Union[str, PathlibPath]], keep_first]


class VizWorkerState(TypedDict):
    """Sub-state for parallel visualization workers dispatched via Send."""

    task: str
    viz_results: Annotated[List[dict], _reduce_viz_results_resettable]
