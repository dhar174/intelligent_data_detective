"""
idd_graph.py — Canonical LangGraph Assembly and Routing for Intelligent Data Detective.

Home of:
- Canonical 15-node StateGraph assembly: build_graph()
- Public agent membership and option constants:
  AGENT_MEMBERS, AGENT_OPTIONS, options
- Routing decision schemas: Router
- Authoritative routing and gate functions:
  route_from_supervisor, route_to_writer, route_viz, write_output_to_file,
  assign_viz_workers, dispatch_sections

Architecture: Checkpoint 3 (Issue #160 / Refs #140, #144, #145).
"""

from __future__ import annotations

import uuid
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    TypeAlias,
    Union,
)

from pydantic import Field

# Optional LangChain / LangGraph imports
try:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables.config import RunnableConfig
    from langgraph.constants import END, START
    from langgraph.graph import StateGraph
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import CachePolicy, Command, Send

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    START = "__start__"
    END = "__end__"
    StateGraph = object  # type: ignore[misc,assignment]
    CompiledStateGraph = object  # type: ignore[misc,assignment]
    Command = Any  # type: ignore[misc,assignment]
    Send = Any  # type: ignore[misc,assignment]
    CachePolicy = Any  # type: ignore[misc,assignment]
    RunnableConfig = dict  # type: ignore[misc,assignment]
    AIMessage = Any  # type: ignore[misc,assignment]
    HumanMessage = Any  # type: ignore[misc,assignment]


from idd_models import BaseNoExtrasModel
from idd_state import (
    NextAgentMetadata,
    ReportOutline,
    ReportResults,
    State,
    VizSpec,
)


# ---------------------------------------------------------------------------
# Membership and Option Constants
# ---------------------------------------------------------------------------

AGENT_MEMBERS: List[str] = [
    "initial_analysis",
    "data_cleaner",
    "analyst",
    "visualization",
    "report_orchestrator",
    "report_section_worker",
    "report_packager",
    "file_writer",
    "viz_worker",
    "viz_evaluator",
]

AGENT_OPTIONS: List[str] = AGENT_MEMBERS + ["FINISH", "EMERGENCY_MSG"]

# Backward-compatibility alias
options: List[str] = AGENT_OPTIONS

PRODUCTION_NODE_NAMES: List[str] = [
    "supervisor",
    "initial_analysis",
    "data_cleaner",
    "analyst",
    "viz_worker",
    "viz_join",
    "viz_evaluator",
    "report_orchestrator",
    "report_section_worker",
    "report_join",
    "report_packager",
    "file_writer",
    "visualization",
    "EMERGENCY_MSG",
    "FINISH",
]


# ---------------------------------------------------------------------------
# Router Model
# ---------------------------------------------------------------------------


SupervisorRouteTarget: TypeAlias = Literal[
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
    "EMERGENCY_MSG",
]


class Router(BaseNoExtrasModel):
    """Structured output contract for supervisor routing decisions."""

    next: SupervisorRouteTarget = Field(
        ...,
        description="The next agent to route to.",
    )
    next_agent_prompt: str = Field(
        ...,
        description="Instructions for the next agent.",
    )
    next_agent_metadata: Optional[NextAgentMetadata] = Field(
        ...,
        description="Metadata for the next agent.",
    )


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------


def route_from_supervisor(
    state: State,
) -> str:
    """Route from supervisor to the chosen worker agent, barrier, or terminal.

    Raises:
        ValueError: If destination 'next' is missing/empty, or not in the allowed
            14 production destinations.
    """
    # W13N-RT: final supervisor route to file_writer/FINISH
    if bool(state.get("report_generator_complete")) and state.get("report_results"):
        return "FINISH" if bool(state.get("file_writer_complete")) else "file_writer"

    nxt = state.get("next")
    if not nxt:
        raise ValueError(
            "Supervisor routing failed: destination 'next' is missing or empty in State."
        )

    # Allow "END" token to route to "FINISH" transition node
    if nxt == "END":
        nxt = "FINISH"

    allowed: Set[str] = {
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
        "FINISH",
        "EMERGENCY_MSG",
    }
    if nxt not in allowed:
        raise ValueError(
            f"Supervisor routing failed: destination {nxt!r} is invalid. "
            f"Allowed destinations: {sorted(allowed)}"
        )
    return nxt


def route_to_writer(
    state: State,
) -> Literal["file_writer", "supervisor", "END"]:
    """Conditional gate after report packaging: routes to file_writer, supervisor, or END."""
    report_done = bool(state.get("report_generator_complete"))
    report_results = state.get("report_results")
    outline = state.get("report_outline")
    expected_sections = (
        len(outline.sections)
        if (isinstance(outline, ReportOutline) or hasattr(outline, "sections"))
        else 0
    )
    written_sections = state.get("written_sections", []) or []
    report_ready = bool(
        report_done
        and (
            isinstance(report_results, ReportResults)
            or (
                report_results is not None
                and type(report_results).__name__ == "ReportResults"
            )
        )
        and expected_sections > 0
        and len(written_sections) >= expected_sections
        and state.get("report_draft")
    )
    already_wrote = bool(state.get("file_writer_complete"))
    if already_wrote:
        return "END"
    if report_ready:
        return "file_writer"
    return "supervisor"


def route_viz(state: State) -> Literal["Accepted", "Revise"]:
    """Evaluate visualization quality: accepts on threshold or partial completion."""
    # After first viz round, accept partial results to prevent analyst re-runs
    if int(state.get("_viz_retry_count") or 0) >= 1:
        return "Accepted"
    # Also accept if viz_join completed and we have any results
    if state.get("visualization_complete") and state.get("viz_results"):
        return "Accepted"
    return "Accepted" if state.get("viz_grade") == "acceptable" else "Revise"


def write_output_to_file(
    state: State, config: Optional[RunnableConfig] = None
) -> Command[Union[Literal["supervisor"], Literal["file_writer"], Literal["__end__"]]]:
    """Terminal output node logic executed by FINISH."""
    if (
        state.get("report_generator_complete", False)
        and state.get("report_results")
        and not state.get("file_writer_complete", False)
    ):
        return Command(
            goto="file_writer",
            update={
                "messages": [
                    HumanMessage(
                        content="Please write the report to the appropriate files, as well as any visualizations. "
                    )
                ]
            },
        )

    if state.get("file_writer_complete") or (
        state.get("report_results") and state.get("report_generator_complete")
    ):
        return Command(goto=END)
    return Command(goto="supervisor")


def assign_viz_workers(state: State):
    """Fan out visualization tasks to parallel viz_worker branches."""
    tasks = state.get("viz_tasks", []) or []
    viz_specs = state.get("viz_specs", []) or []
    _safe_specs = []
    for sp in viz_specs:
        if isinstance(sp, dict):
            try:
                _safe_specs.append(VizSpec(**sp))
            except Exception:
                pass
        elif isinstance(sp, VizSpec):
            _safe_specs.append(sp)
    viz_specs = _safe_specs

    if not tasks or not viz_specs:
        return Send(
            "report_orchestrator",
            {
                "messages": [
                    AIMessage(
                        content="No viz tasks or specs to assign. Proceeding to report generation.",
                        name="visualization",
                    )
                ]
            },
        )

    for sp in viz_specs:
        if not getattr(sp, "viz_id", None):
            sp.viz_id = uuid.uuid4().hex

    _vw_cm = state.get("cleaning_metadata")
    _vw_ai = state.get("analysis_insights")
    _vw_df_ids = state.get("available_df_ids") or []
    _vw_cfg = state.get("_config")
    _vw_ftml = state.get("final_turn_msgs_list") or []
    _vw_msgs = state.get("messages") or []

    return [
        Send(
            "viz_worker",
            {
                "task": t,
                "individual_viz_task": t,
                "viz_spec": sp,
                "cleaning_metadata": _vw_cm,
                "analysis_insights": _vw_ai,
                "available_df_ids": _vw_df_ids,
                "_config": _vw_cfg,
                "final_turn_msgs_list": _vw_ftml,
                "messages": _vw_msgs,
            },
        )
        for t, sp in zip(tasks, viz_specs)
    ]


def dispatch_sections(state: State):
    """Emit Send events to run one report_section_worker per section in outline."""
    outline = state.get("report_outline", None)
    if not isinstance(outline, ReportOutline) or not outline or not outline.sections:
        return []

    _sec_cm = state.get("cleaning_metadata")
    _sec_ai = state.get("analysis_insights")
    _sec_df_ids = state.get("available_df_ids") or []
    _sec_cfg = state.get("_config")
    _sec_prompt = state.get("user_prompt")
    _sec_viz = state.get("viz_results")
    _sec_cdd = state.get("cleaned_dataset_description")
    _sec_ftml = state.get("final_turn_msgs_list") or []
    _sec_msgs = state.get("messages") or []

    sends = []
    for s in outline.sections:
        sec_payload = (
            s.model_dump(mode="json", exclude_none=True)
            if hasattr(s, "model_dump")
            else (dict(s) if isinstance(s, dict) else s)
        )
        payload = {
            "section": sec_payload,
            "cleaning_metadata": _sec_cm,
            "analysis_insights": _sec_ai,
            "available_df_ids": _sec_df_ids,
            "_config": _sec_cfg,
            "user_prompt": _sec_prompt,
            "viz_results": _sec_viz,
            "cleaned_dataset_description": _sec_cdd,
            "final_turn_msgs_list": _sec_ftml,
            "messages": _sec_msgs,
        }
        sends.append(Send("report_section_worker", payload))
    return sends


# ---------------------------------------------------------------------------
# Default Pass-Through Node Factory
# ---------------------------------------------------------------------------


def _make_pass_through_node(name: str) -> Callable[[State], dict]:
    """Create a placeholder pass-through node for structural-only graph assembly.

    If invoked at runtime, raises RuntimeError to prevent silent execution of placeholder nodes.
    """

    def pass_through_node(state: State) -> dict:
        raise RuntimeError(
            f"Cannot execute placeholder node '{name}'. "
            "Graph was assembled with structural_only=True. "
            "To execute the graph, provide node_overrides and compile with structural_only=False."
        )

    pass_through_node.__name__ = f"{name}_node"
    return pass_through_node


# ---------------------------------------------------------------------------
# Graph Construction & Compilation
# ---------------------------------------------------------------------------


def build_graph(
    node_overrides: Optional[Dict[str, Callable]] = None,
    checkpointer: Any = None,
    store: Any = None,
    cache: Any = None,
    structural_only: bool = True,
    node_cache_policies: Optional[Union[Any, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """Construct, wire, and compile the authoritative 15-node IDD StateGraph.

    Args:
        node_overrides: Optional mapping of node name -> callable implementation.
            Allows injecting mock or production agent nodes.
        checkpointer: Checkpointer instance (e.g. MemorySaver, SqliteSaver).
            Defaults to None to avoid side-effect SQLite database file creation.
        store: LangGraph store instance (e.g. InMemoryStore).
        cache: Cache policy or instance (e.g. InMemoryCache).
        structural_only: If True (default for backward compatibility), un-overridden
            nodes get placeholder functions that raise RuntimeError if invoked.
            If False, fails closed with ValueError unless all 14 production worker
            and supervisor nodes are provided in node_overrides (FINISH defaults to
            write_output_to_file).
        node_cache_policies: Optional CachePolicy or dict of node name -> CachePolicy
            attached to registered nodes during graph assembly.
        **kwargs: Accepted for backward compatibility (e.g. api_key=...).

    Returns:
        CompiledStateGraph with exactly 15 nodes and 33 edges (21 conditional),
        0 unreachable nodes, and 0 dead ends. Sets compiled.is_structural_only = structural_only.

    Raises:
        ValueError: If node_overrides contains unknown node names, or if
            structural_only is False and required node implementations are missing.
    """
    if not HAS_LANGGRAPH:
        raise RuntimeError("LangGraph is required to build the state graph.")

    overrides = dict(node_overrides or {})

    # 1. Validate override names: reject unknown nodes
    unknown = set(overrides.keys()) - set(PRODUCTION_NODE_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown node override(s): {sorted(unknown)}. "
            f"Allowed production nodes: {sorted(PRODUCTION_NODE_NAMES)}"
        )

    # 2. In executable mode, fail closed if any required node implementations are missing
    if not structural_only:
        required_nodes = set(PRODUCTION_NODE_NAMES) - {"FINISH"}
        missing = required_nodes - set(overrides.keys())
        if missing:
            raise ValueError(
                f"Cannot build executable graph (structural_only=False): "
                f"missing node override(s): {sorted(missing)}. "
                "All 14 production worker/supervisor nodes must be provided. "
                "For offline compilation and topology inspection, pass structural_only=True."
            )

    builder = StateGraph(State)

    # 3. Register all 15 nodes
    for name in PRODUCTION_NODE_NAMES:
        if name in overrides:
            node_fn = overrides[name]
        elif name == "FINISH":
            node_fn = write_output_to_file
        else:
            node_fn = _make_pass_through_node(name)

        policy = (
            node_cache_policies.get(name)
            if isinstance(node_cache_policies, dict)
            else node_cache_policies
        )
        if policy is not None:
            builder.add_node(name, node_fn, cache_policy=policy)
        else:
            builder.add_node(name, node_fn)

    # 2. Entry edge
    builder.add_edge(START, "initial_analysis")

    # 3. Supervisor conditional routing (14 destinations)
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "initial_analysis": "initial_analysis",
            "data_cleaner": "data_cleaner",
            "analyst": "analyst",
            "viz_worker": "viz_worker",
            "viz_join": "viz_join",
            "viz_evaluator": "viz_evaluator",
            "report_orchestrator": "report_orchestrator",
            "report_section_worker": "report_section_worker",
            "report_join": "report_join",
            "report_packager": "report_packager",
            "file_writer": "file_writer",
            "visualization": "visualization",
            "FINISH": "FINISH",
            "EMERGENCY_MSG": "EMERGENCY_MSG",
        },
    )

    # 4. Pipeline-entry nodes and emergency fan-in return to supervisor
    for src in [
        "initial_analysis",
        "data_cleaner",
        "analyst",
        "EMERGENCY_MSG",  # W2-EMERGENCY: fan-in edge prevents dead-end
    ]:
        builder.add_edge(src, "supervisor")

    # 5. Visualization sub-pipeline (fan-out, barrier, evaluator)
    builder.add_conditional_edges(
        "visualization",
        assign_viz_workers,
        ["viz_worker", "report_orchestrator"],
    )
    builder.add_edge("viz_worker", "viz_join")
    builder.add_edge("viz_join", "viz_evaluator")
    builder.add_conditional_edges(
        "viz_evaluator",
        route_viz,
        {"Accepted": "supervisor", "Revise": "supervisor"},
    )

    # 6. Reporting sub-pipeline (fan-out, barrier, packager, file_writer)
    builder.add_conditional_edges(
        "report_orchestrator",
        dispatch_sections,
        ["report_section_worker"],
    )
    builder.add_edge("report_section_worker", "report_join")
    builder.add_edge("report_orchestrator", "report_join")
    builder.add_edge("report_join", "report_packager")

    # 7. Report packager conditional gate (file_writer, supervisor, or END)
    builder.add_conditional_edges(
        "report_packager",
        route_to_writer,
        {
            "file_writer": "file_writer",
            "supervisor": "supervisor",
            "END": END,
        },
    )
    builder.add_edge("file_writer", "supervisor")

    # 8. Compile graph
    compile_kwargs: Dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if store is not None:
        compile_kwargs["store"] = store
    if cache is not None:
        compile_kwargs["cache"] = cache

    compiled = builder.compile(**compile_kwargs)
    compiled.is_structural_only = structural_only
    return compiled
