# IDD v4 Memory Integration Analysis Report

## Part 1: Memory Functions in the Workflow - Step-by-Step Analysis

This section provides a comprehensive analysis of all memory functions in the IntelligentDataDetective v4 notebook and explains how they integrate into the overall workflow.

### 1.1 Memory Architecture Overview

The IDD v4 system employs a sophisticated dual-layer memory architecture that combines:
- **LangMem Tools**: High-level memory management and search tools from the langmem library
- **LangGraph InMemoryStore**: Low-level vector storage with semantic search capabilities
- **Universal Retrieval Pattern**: Standardized memory access across all agent nodes
- **Checkpointing**: State persistence at both agent and graph levels

### 1.2 Core Memory Components

#### 1.2.1 Embedding Function Setup (Cell 13: ff7w7v0dtWBy)

```python
def _embed_docs(texts: List[str]) -> List[List[float]]:
    # LangGraph store expects a callable that returns list[list[float]]
    return embeddings.embed_documents(texts)

in_memory_store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": _embed_docs,
    }
)

# --- Tools: add memory tools once and de-dupe by name across lists ---
mem_manage = create_manage_memory_tool(namespace=("memories",))
mem_search = create_search_memory_tool(namespace=("memories",))
progress_tool = report_intermediate_progress
init_analyst_tools = analyst_tools.copy()

mem_tools = [create_manage_memory_tool(namespace=("memories",)),
             create_search_memory_tool(namespace=("memories",)),
             report_intermediate_progress]
for tool in mem_tools:
    data_cleaning_tools.append(tool)
    init_analyst_tools.append(tool)
    analyst_tools.append(tool)
    report_generator_tools.append(tool)
    file_writer_tools.append(tool)
    visualization_tools.append(tool)
def _dedupe_tools(tools):
    seen = set()
    out = []
    for t in tools:
        name = getattr(t, "name", None) or repr(t)
        if name in seen:
            continue
        seen.add(name)
        out.append(t)
    return out

init_analyst_tools = _dedupe_tools(init_analyst_tools)
analyst_tools = _dedupe_tools(analyst_tools)
data_cleaning_tools = _dedupe_tools(data_cleaning_tools)
report_generator_tools = _dedupe_tools(report_generator_tools)
visualization_tools = _dedupe_tools(visualization_tools)

# Pull in the file management toolkit only once for file_writer
toolkit_tools = toolkit.get_tools()
assert toolkit_tools, "No tools found in toolkit"
for tool in toolkit_tools:
    file_writer_tools.append(tool)
file_writer_tools = _dedupe_tools(file_writer_tools)

# --- Small helper to fetch “memories” text for prompts (avoid passing a function) ---
def _mem_text(query: str, limit: int = 5) -> str:
    try:
        items = in_memory_store.search(("memories",), query=query, limit=limit)
        if not items:
            return "None."
        # items are dict-like; stringify safely
        return "\n".join(str(it) for it in items)
    except Exception:
        return "None."

# -------------------------
# Agent factories
# -------------------------
def create_data_cleaner_agent(initial_description: InitialDescription, df_ids: List[str] = []):

    checkpointer = InMemorySaver()
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in data_cleaning_tools)
    init_df_id_str = ", /n".join(df_ids)
    init_dc_vars = {"available_df_ids":init_df_id_str,"dataset_description":initial_description.dataset_description,
                    "tool_descriptions":tool_descriptions,"tooling_guidelines" : DEFAULT_TOOLING_GUIDELINES,"output_format" : CleaningMetadata.model_json_schema(),"memories" : "No memories yet",
                    "data_sample":initial_description.data_sample}
    prompt = data_cleaner_prompt_template.partial(**init_dc_vars)
    # NOTE: response_format prefers a Pydantic model class, not a JSON schema string
    return create_react_agent(
        data_cleaner_llm,
        tools=data_cleaning_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=CleaningMetadata,
        prompt=prompt,
        name="data_cleaner",
        version="v2",
    )

def create_initial_analysis_agent(user_prompt: str, df_ids: List[str] = []):
    init_df_id_str = ", /n".join(df_ids)

    checkpointer = InMemorySaver()
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in init_analyst_tools)
    init_ia_vars = {"available_df_ids":init_df_id_str,"dataset_description":initial_description.dataset_description,
                    "tool_descriptions":tool_descriptions,"tooling_guidelines" : DEFAULT_TOOLING_GUIDELINES,"output_format" : InitialDescription.model_json_schema(),"memories" : "No memories yet",
                    "data_sample":initial_description.data_sample,"user_prompt":user_prompt}
    prompt = analyst_prompt_template_initial.partial(**init_ia_vars)

    return create_react_agent(
        initial_analyst_llm,
        tools=init_analyst_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=InitialDescription,
        prompt=prompt,
        name="initial_analysis",
        version="v2",
    )

def create_analyst_agent(initial_description: InitialDescription, df_ids: List[str] = []):
    init_df_id_str = ", /n".join(df_ids)
    checkpointer = InMemorySaver()
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in analyst_tools)
    init_analyst_vars = {"available_df_ids":init_df_id_str,"cleaned_dataset_description":initial_description.dataset_description,
                    "tool_descriptions":tool_descriptions,"output_format" : AnalysisInsights.model_json_schema(),"memories" : "No memories yet",
                    "data_sample":initial_description.data_sample}
    prompt = analyst_prompt_template_main.partial(**init_analyst_vars)
    return create_react_agent(
        analyst_llm,
        tools=analyst_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=AnalysisInsights,
        prompt=prompt,
        name="analyst",
        version="v2",
    )

def create_file_writer_agent(df_ids: List[str] = []):
    init_df_id_str = ", /n".join(df_ids)
    checkpointer = InMemorySaver()
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in file_writer_tools)
    init_fw_vars = {"available_df_ids":init_df_id_str,"tool_descriptions":tool_descriptions,"tooling_guidelines" : DEFAULT_TOOLING_GUIDELINES, "output_format" : FileResults.model_json_schema(),
                    "memories" : "No memories yet", "file_content": "No content yet","file_name": "No file name yet", "file_type": "No file type yet"}
    # NOTE: response_format prefers a Pydantic model class, not a JSON schema string
    prompt = file_writer_prompt_template.partial(**init_fw_vars)
    return create_react_agent(
        file_writer_llm,
        tools=file_writer_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        prompt=prompt,
        response_format=FileResults.model_json_schema(),   # 👈
        name="file_writer",
        version="v2",
    )

def create_visualization_agent(df_ids: List[str] = []):
    init_df_id_str = ", /n".join(df_ids)
    checkpointer = InMemorySaver()
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in visualization_tools)
    init_vis_vars = {"available_df_ids":init_df_id_str,"tool_descriptions":tool_descriptions,"tooling_guidelines" : DEFAULT_TOOLING_GUIDELINES, "output_format" : VisualizationResults.model_json_schema(),
                    "memories" : "No memories yet", "analysis_insights": "No analysis insights yet","cleaned_dataset_description": "No cleaned dataset description yet"}

    prompt = visualization_prompt_template.partial(**init_vis_vars)
    return create_react_agent(
        visualization_orchestrator_llm,
        tools=visualization_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=VisualizationResults,
        prompt=prompt,
        name="visualization",
        version="v2",
    )

def create_viz_evaluator_agent():
    checkpointer = InMemorySaver()

    init_viz_vars = {"output_format" : VizFeedback.model_json_schema(), "memories" : "No memories yet", "analysis_insights": "No analysis insights yet","cleaned_dataset_description": "No cleaned dataset description yet",
                    "visualization_results": "No visualization results yet"}
    prompt = viz_evaluator_prompt_template.partial(**init_viz_vars)
    return create_react_agent(
        viz_evaluator_llm,
        tools=[list_visualizations, get_visualization],
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=VizFeedback,
        prompt=prompt,
        name="viz_evaluator",
        version="v2",
    )

def create_report_generator_agent(df_ids: List[str] = [], rg_agent_task : Literal["outline","section","package"] = "outline"):
    init_df_id_str = ", /n".join(df_ids)
    checkpointer = InMemorySaver()
    output_format_map = {"outline" : {"output_format" : ReportOutline.model_json_schema(), "report_task": "generate a report outline", "name": "report_orchestrator","llm": report_orchestrator_llm},
                    "section" : {"output_format" : Section.model_json_schema(), "report_task": "generate a section of the report", "name": "report_section_worker","llm": report_section_worker_llm},
                    "package" : {"output_format" : ReportResults.model_json_schema(), "report_task": "generate a full report package in PDF, Markdown, and HTML", "name": "report_packager","llm": report_packager_llm}}
    output_format = output_format_map[rg_agent_task]
    report_task = output_format["report_task"]
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in report_generator_tools)
    init_rg_vars = {"available_df_ids":init_df_id_str,"tool_descriptions":tool_descriptions,"tooling_guidelines" : DEFAULT_TOOLING_GUIDELINES, "output_format" : output_format,
                    "memories" : "No memories yet", "analysis_insights": "No analysis insights yet", "cleaned_dataset_description": "No cleaned dataset description yet",
                    "visualization_results": "No visualization results yet", "report_task": report_task}

    prompt = report_generator_prompt_template.partial(**init_rg_vars)
    return create_react_agent(
        output_format_map[rg_agent_task]["llm"],
        tools=report_generator_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=ReportOutline,
        prompt=prompt,
        name=output_format_map[rg_agent_task]["name"],
        version="v2",
    )

# (optional) simple memory write helper
@lru_cache(maxsize=128)
def update_memory(state: Union[MessagesState, State], config: RunnableConfig, *, memstore: InMemoryStore):
    user_id = str(config.get("configurable", {}).get("user_id", "user"))
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    memstore.put(namespace, memory_id, {"memory": state["messages"][-1].text()})


# -------------------------
# Supervisor
# -------------------------
from pydantic import BaseModel, Field
from typing import Literal

def make_supervisor_node(supervisor_llms: List[BaseChatModel], members: list[str], user_prompt: str):
    #    [big_picture_llm,router_llm, reply_llm, plan_llm, replan_llm, progress_llm, todo_llm],

    options = list(dict.fromkeys(members + ["FINISH"]))  # keep order, dedupe

    system_prompt = ChatPromptTemplate.from_messages(
    [ SystemMessage(content= (
        "You are a supervisor managing these workers: {members}.\n"
        "User request: {user_prompt}\n\n"
        "Goal: a thorough EDA + strong visuals + a final report (Markdown, PDF, HTML) saved to disk.\n"
        "Before each handoff, think step-by-step and maintain (1) the Plan and (2) a To-Do list.\n"
        "Only route to workers that still have work; FINISH when everything is done."
        "The Initial Analysis agent (aka 'initial_analysis') simply produces an initial description of the dataset and a data sample in the form of the 'InititialDescription' class found keyed as 'initial_description'. \n"
        "The Initial Analysis agent should be finished before your first turn, only route back to Initial Analysis agent if it did not finish. The Initial Analysis agent MUST be finished before any other agents can begin. \n"
        "The Data Cleaner (aka 'data_cleaner') needs to save the cleaned data and provide a way to access the newly cleaned dataset. The Data Cleaner returns the cleaned data in the form of the 'CleaningMetadata' class found keyed as 'cleaning_metadata'.\n"
        "The Data Cleaner MUST also be finished before Analyst or other agents (besides Initial Analysis) can begin. \n"
        "The Analyst (aka 'analyst') produces insights from the data. The Analyst returns the insights in the form of the 'AnalysisInsights' class found keyed as 'analysis_insights'.\n"
        "If the visualization agent produces images (keyed as 'visualization_results), ensure they are saved to disk (directly or via FileWriter, aka 'file_writer'). FileWriter returns the file metadata in the form of the 'FileResults' class found keyed as 'file_results'.\n"
        "If the report is complete (saved in a disk path that is keyed under 'final_report_path'), ensure all three formats are saved to disk.\n"
        "Memories that might help:\n{memories}\n"
        "Here is the current plan as it stands:"
        "{plan_summary}"
        "Steps:"
        "\n{plan_steps}\n"

        "Already marked complete (steps):"
        "\n{completed_steps}\n"

        "Already marked complete (tasks):"
        "\n{completed_tasks}\n"

        "The following agent workers have marked their tasks as completed, though of course you should always verify yourself:"
        "\n{completed_agents}\n"

        "The following agent workers have NOT yet marked their tasks complete:"
        "\n{remaining_agents}\n"

        "Remaining To-Do (may include items that are actually done; verify from the work):"
        "\n{to_do_list}\n"

        "Here is the latest progress report:"
        "{latest_progress}"

        "The last message passed into state was:"
        "{last_message}"

        "The last agent to have been invoked was {last_agent_id}, whom you had given the following task as a message: {next_agent_prompt} \n: They left the following message for you, the supervisor:"
        "{reply_msg_to_supervisor}\n"
        "They {finished_this_task} the task you gave them, and they {expect_reply} a reply from you. If you choose to reply to them and you also choose to route back to them, put the message in Router.next_agent_prompt."
        "However if you plan to route to a different agent, hold off on the reply for now, you will be prompted for it after choosing the next worker agent to route to.\n"

        "Perhaps the following memories may be helpful:"
        "\n{memories}\n"

        "You will encode your decisions into the Router class: next to assign the next worker/agent, next_agent_prompt to instruct them (use prompt engineering knowledge to instruct on the goal but leave the details up to the agent)."
        "The process will require constant two-way communication with the workers, including checking their work and tracking progress."
        "To send instructions to the agent you route to, use the next_agent_prompt field in the Router class. If you also need to send data as a payload, use the next_agent_metadata field."

    )),MessagesPlaceholder(variable_name="messages")]
    )
    supervisor_prompt = system_prompt.partial(members=options, user_prompt=user_prompt)

    # Reintroduce a dedicated progress-accounting prompt (fixed)
    PROGRESS_ACCOUNTING_STR = """Since a full turn has passed, review all prior messages and state to mark which plan steps and tasks are complete.

Your main objective from the user:
{user_prompt}

Current plan:
{plan_summary}
Steps:
\n{plan_steps}\n

Already marked complete (steps):
{completed_steps}

Already marked complete (tasks):
\n{completed_tasks}\n

"The following agent workers have marked their tasks as completed, though of course you should always verify yourself:"
"\n{completed_agents}\n"

"The following agent workers have NOT yet marked their tasks complete:"
"\n{remaining_agents}\n"

"Here is the latest progress report:"
"{latest_progress}"

"The last message passed into state was:"
"{last_message}"

"Memories that might help:"
"\n{memories}\n"

Remaining To-Do (may include items that are actually done; verify from the work):
\n{to_do_list}\n

Return only the updated lists of completed steps and completed tasks, based on actual work observed.
"""

    class Router(BaseNoExtrasModel):
        next: AgentId = Field(..., description="Next agent to invoke.")
        next_agent_prompt: str = Field(..., description="Actionable prompt for the selected worker.")
        next_agent_metadata: Optional[NextAgentMetadata]
    def _dedup(seq):
        return list(dict.fromkeys(seq or []))
    def _parse_cst_with_plan(plan: Plan):
        def _inner(raw: dict) -> CompletedStepsAndTasks:
            # If the OpenAI SDK returns a JSON string, load it first; otherwise dict is fine.
            if isinstance(raw, str):
                return CompletedStepsAndTasks.model_validate_json(raw, context={"plan": plan})
            return CompletedStepsAndTasks.model_validate(raw, context={"plan": plan})
        return _inner
    def schema_for_completed_steps(plan: Plan) -> dict:
        # Base schema from Pydantic (includes top-level fields & ProgressReport, etc.)
        base = CompletedStepsAndTasks.model_json_schema()

        # Allowed item shapes for completed_steps (one per plan step)
        allowed_anyof = []
        for ps in plan.plan_steps:
            allowed_anyof.append({
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    # BaseNoExtrasModel fields:
                    "reply_msg_to_supervisor": {"type": "string"},
                    "finished_this_task": {"type": "boolean"},
                    "expect_reply": {"type": "boolean"},
                    # The identity triplet is locked to this exact plan step:
                    "step_number": {"type":"number","const": ps.step_number},
                    "step_name": {"type": "string","const": ps.step_name},
                    "step_description": {"type": "string","const": ps.step_description},
                    # Must be completed:
                    "is_step_complete": {"type": "boolean","const": True},
                },
                "required": [
                    "reply_msg_to_supervisor", "finished_this_task", "expect_reply",
                    "step_number", "step_name", "step_description", "is_step_complete",
                ],
            })

        # Replace the completed_steps field to allow only those shapes
        base["properties"]["completed_steps"] = {
            "type": "array",
            "items": {"anyOf": allowed_anyof},
            # NOTE: JSON Schema's uniqueItems checks whole-object equality;
            # base fields differing would defeat dedup. We enforce dedup by triplet in Pydantic validator above.
            # "uniqueItems": True,  # optional; harmless but not sufficient for triplet-uniqueness
        }
        return base

    def supervisor_node(state: State, config: RunnableConfig):
        _count = int(state.get("_count_", 0)) + 1
        last_count = int(state["_count_"]) - 1
        last_agent_id = state.get("last_agent_id", state.get("next", None))
        last_agent_prompt = state.get("next_agent_prompt", None)
        assert last_agent_id, "No last agent ID"
        supervisor_msgs = []
        if last_count == 0:
            progress_report: ProgressReport = ProgressReport(latest_progress="This is the first turn. and no progress has been made yet.", finished_this_task=False, reply_msg_to_supervisor="This progress report needs filled out", expect_reply=True)
        else:
            progress_str = state.get("latest_progress", "No progress has been made yet.")
            if not progress_str or not isinstance(progress_str, str):
                progress_report: ProgressReport = ProgressReport(latest_progress="No progress has been made yet.",finished_this_task=False, reply_msg_to_supervisor="This progress report needs filled out after progress has been made", expect_reply=True)
            else:
                progress_report = ProgressReport(latest_progress=progress_str,finished_this_task=False, reply_msg_to_supervisor="This progress report needs filled out after progress has been made", expect_reply=True)

        user_prompt = state["user_prompt"]
        # Completion flags → for routing context (not used to infer step/task completion)
        complete_map = {
            "initial_analysis": bool(state.get("initial_analysis_complete")),
            "data_cleaner": bool(state.get("data_cleaning_complete")),
            "analyst": bool(state.get("analyst_complete")),
            "file_writer": bool(state.get("file_writer_complete")),
            "visualization": bool(state.get("visualization_complete")),
            "report_orchestrator": bool(state.get("report_generator_complete")),
        }
        task_fin_str_map = {True: "are currently awaiting", False: "are not expecting or waiting for","True": "are currently awaiting", "False": "are not expecting or waiting for","true": "are currently awaiting", "false": "are not expecting or waiting for"}
        completed_agents = [k for k, v in complete_map.items() if v]
        remaining_agents = [k for k, v in complete_map.items() if not v]
        reply_str_map = {True: "are currently awaiting", False: "are not expecting or waiting for","True": "are currently awaiting", "False": "are not expecting or waiting for","true": "are currently awaiting", "false": "are not expecting or waiting for"}



        agent_output_map = {"initial_analysis": {"class":InitialDescription, "class_name": "InitialDescription", "schema": InitialDescription.model_json_schema(), "state_obj_key": "initial_description", "task_description": "generate an initial analysis of the data"},
                                                 "data_cleaner": {"class":CleaningMetadata, "class_name": "CleaningMetadata", "schema": CleaningMetadata.model_json_schema(), "state_obj_key": "cleaning_metadata", "task_description": "clean the data"},
                            "analyst": {"class":AnalysisInsights, "class_name": "AnalysisInsights", "schema": AnalysisInsights.model_json_schema(), "state_obj_key": "analysis_insights", "task_description": "generate insights from the data"},
                            "file_writer": {"class":FileResults, "class_name": "FileResults", "schema": FileResults.model_json_schema(), "state_obj_key": "file_results", "task_description": "write data to disk"},
                            "visualization": {"class":VisualizationResults, "class_name": "VisualizationResults", "schema": VisualizationResults.model_json_schema(), "state_obj_key": "visualization_results", "task_description": "generate visualizations from the data"},
                            "report_orchestrator": {"class":ReportOutline, "class_name": "ReportOutline", "schema": ReportOutline.model_json_schema(), "state_obj_key": "report_outline", "task_description": "generate a report outline"},
                            "report_section_worker": {"class":Section, "class_name": "Section", "schema": Section.model_json_schema(), "state_obj_key_and_idx": ("sections", -1), "task_description": "generate a section of the report"},
                            "report_packager": {"class":ReportResults, "class_name": "ReportResults", "schema": ReportResults.model_json_schema(), "state_obj_key": "report_results", "task_description": "generate a full report in PDF, Markdown, and HTML"},
                            "viz_evaluator": {"class":VizFeedback, "class_name": "VizFeedback", "schema": VizFeedback.model_json_schema(), "state_obj_key": "viz_eval_results", "task_description": "evaluate the visualizations"},
                            "viz_worker": {"class": DataVisualization, "class_name": "DataVisualization", "schema": DataVisualization.model_json_schema(), "state_obj_key_and_idx": ("visualization_results", -1), "task_description": "generate a visualization from the data"},
                            "routing": {"class":Router, "class_name": "Router", "schema": Router.model_json_schema(), "state_obj_key": "router", "task_description": "route to another agent"},
                            "progress": {"class":CompletedStepsAndTasks, "class_name": "CompletedStepsAndTasks", "schema": CompletedStepsAndTasks.model_json_schema(), "state_obj_key": "completed_plan_steps", "task_description": "progress accounting"},
                            "plan": {"class":Plan, "class_name": "Plan", "schema": Plan.model_json_schema(), "state_obj_key": "plan", "task_description": "plan generation"},
                            "todo":  {"class":ToDoList, "class_name": "ToDoList", "schema": ToDoList.model_json_schema(), "state_obj_key": "todo_list", "task_description": "to-do list generation"},
                            }
        # State hydration
        curr_plan: Plan = state.get("current_plan") or Plan(plan_summary="", plan_steps=[], plan_title="", finished_this_task=False, reply_msg_to_supervisor="This plan still needs thought out", expect_reply=True)
        done_steps: List[str] = _dedup(state.get("completed_plan_steps", []))
        done_tasks: List[str] = _dedup(state.get("completed_tasks", []))
        todo_list: List[str] = _dedup(state.get("to_do_list", []))
        latest_message = state.get("last_agent_message",None)
        last_message_text = None
        if not latest_message:
            lm_name= last_agent_id
            if lm_name == "":
                lm_name = "user"
                latest_message = HumanMessage(content="No message", name=lm_name)
                last_message_text = latest_message.text()
            else:
                # iterate in reverse from last message to first until find one with lm_name as .name attr
                for msg in reversed(state.get("messages", [])):
                    if msg.name == lm_name and msg.text():
                        latest_message = msg
                        last_message_text = latest_message.text() if isinstance(latest_message, AIMessage) else "No message"
                        break

        elif isinstance(latest_message, (HumanMessage, AIMessage)):
            last_message_text = latest_message.text()
        else:
            try:
                if getattr(latest_message, "text"):
                    last_message_text = str(getattr(latest_message, "text"))
            except:
                last_message_text = "No message"
        if not last_message_text:
            last_message_text = "No message"

        final_turn_msgs_list = state.get("final_turn_msgs_list", [latest_message])
        if not final_turn_msgs_list:
            final_turn_msgs_list = [latest_message]

        # --- Phase 1: Progress Accounting (only if we have any prior messages) ---
        progress_supervisor_expects_reply = False
        if state.get("_count_", 0) > 0 and state.get("messages", False) and state.get("_count_", 0) > last_count:
            cst_schema = schema_for_completed_steps(curr_plan)

            progress_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=PROGRESS_ACCOUNTING_STR),
                MessagesPlaceholder(variable_name="messages"),
            ])
            progress_vars = {
                "messages":final_turn_msgs_list,
                "user_prompt":user_prompt,
                "plan_summary":curr_plan.plan_summary,
                "plan_steps":curr_plan.plan_steps,
                "completed_steps":done_steps,
                "completed_tasks":done_tasks,
                "to_do_list":todo_list,
                "latest_progress":progress_report.latest_progress,
                "completed_agents":completed_agents,
                "remaining_agents":remaining_agents,
                "last_message":last_message_text,
                "memories":_mem_text(last_message_text),
                "cleaning_metadata":state.get("cleaning_metadata",None),
                "output_schema_name" : "CompletedStepsAndTasks",
                "initial_description":state.get("initial_description",None),
                "cleaned_dataset_description":state.get("cleaned_dataset_description",None),
                "analysis_insights":state.get("analysis_insights",None),
                "visualization_results":state.get("visualization_results",None),
                }
            updated_progress_prompt = progress_prompt.partial(**progress_vars)
            rendered_progress_prompt = progress_prompt.format_messages(**progress_vars)
            # cst_llm = supervisor_llm.bind(response_format={"type": "json_schema","json_schema": {"name": "CompletedStepsAndTasks", "schema": cst_schema, "strict": True},})

            cst_llm = supervisor_llms[5].with_structured_output(cst_schema, strict=True)
            progress_llm = updated_progress_prompt | cst_llm | RunnableLambda(_parse_cst_with_plan(curr_plan))
            progress_result: CompletedStepsAndTasks = progress_llm.invoke(progress_vars, config=state["_config"], prompt_cache_key = "progress_prompt")

            progress = None

            if isinstance(progress_result, CompletedStepsAndTasks):
                progress_supervisor_expects_reply = progress_result.expect_reply
                progress = progress_result
                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
            elif isinstance(progress_result, dict):
                if "structured_response" in progress_result:
                    progress = progress_result["structured_response"]
                    supervisor_msgs = supervisor_msgs + progress_result["messages"]
                else:
                    progress = CompletedStepsAndTasks.model_validate(progress_result)
                    supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
            elif isinstance(progress_result, str):
                progress = CompletedStepsAndTasks.model_validate_json(progress_result)
                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
            assert progress, "Failed to parse progress result"
            assert isinstance(progress, CompletedStepsAndTasks), "Failed to parse progress result"
            progress_report = progress.progress_report
            # Merge (dedup) newly completed items
            done_steps = _dedup(done_steps + (progress.completed_steps or []))
            done_tasks = _dedup(done_tasks + (progress.finished_tasks or []))

            # Remove completed steps from the current plan (safe filter)
            if isinstance(curr_plan, Plan) and isinstance(curr_plan.plan_steps, list):
                for pstep in curr_plan.plan_steps:
                    if pstep in done_steps and not pstep.is_step_complete:
                        pstep.is_step_complete = True

            # Trim completed tasks from To-Do
            todo_list = [t for t in todo_list if t not in done_tasks]

        #write progress report to a file in state["p
        replan_vars={
                "user_prompt":user_prompt,
                "current_plan":curr_plan,
                "plan_summary":curr_plan.plan_summary,
                "plan_steps":curr_plan.plan_steps,
                "past_steps":done_steps,
                "latest_progress":progress_report.latest_progress,
                "output_schema_name" : "Plan",
                "completed_tasks":done_tasks,
                "completed_agents":completed_agents,
                "remaining_agents":remaining_agents,
                "memories":_mem_text(last_message_text),
                "to_do_list":todo_list,
            }

        # --- Phase 1: Progress Accounting (only if we have any prior messages) ---
        prompt_for_planning = replan_prompt
        planning_llm = supervisor_llms[4]
        plan_prompt_key = "replan_prompt"
        # --- Phase 2: Replan against current reality ---
        if curr_plan.plan_title.strip() == "" or _count == 1 or curr_plan.plan_summary.strip() == "":
            curr_plan.plan_title = "Initial Plan Needed"
            curr_plan.plan_summary = "No plan has been developed yet. Please create one!"
            curr_plan.plan_steps = []
            todo_list = []
            prompt_for_planning = plan_prompt
            replan_vars = {
                "user_prompt":user_prompt,
                "output_schema_name" : "Plan",
                "agents": options,
            }
            planning_llm = supervisor_llms[3]
            plan_prompt_key = "plan_prompt"


        base_replan_prompt = prompt_for_planning
        updated_replan_prompt = base_replan_prompt.partial(**replan_vars)
        rendered_new_plan_prompt = updated_replan_prompt.format_messages(messages=[*final_turn_msgs_list,AIMessage(content="Please reformulate the plan based on current progress.", name="supervisor")],**replan_vars)

        mems = _mem_text(user_prompt)
        planning_supervisor_llm = updated_replan_prompt | planning_llm.with_structured_output(Plan, strict=True)
        replan_vars["messages"] = rendered_new_plan_prompt
        plan_supervisor_expects_reply = False
        new_plan = planning_supervisor_llm.invoke(replan_vars, config=state["_config"], prompt_cache_key = plan_prompt_key)
        if isinstance(new_plan, dict):
            if "structured_response" in new_plan:
                supervisor_msgs = supervisor_msgs + new_plan["messages"]
                new_plan = new_plan["structured_response"]
            else:
                new_plan = Plan.model_validate(new_plan)
                supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
        elif isinstance(new_plan, Plan):
            new_plan = new_plan
            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
        elif isinstance(new_plan, str):
            supervisor_msgs.append(AIMessage(content=new_plan, name="supervisor"))
            new_plan = Plan.model_validate_json(new_plan)

        else:
            new_plan = Plan(plan_title="", plan_summary="", plan_steps=[], finished_this_task=False, reply_msg_to_supervisor="This plan still needs thought out", expect_reply=True)
            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
        assert isinstance(new_plan, Plan), "Failed to parse plan result"
        plan_supervisor_expects_reply = new_plan.expect_reply
        prev_plan = curr_plan
        curr_plan = new_plan

        # --- Phase 3: Refresh To-Do list ---
        base_todo_prompt = todo_prompt
        todo_vars = {
            "user_prompt":user_prompt,
            "plan_summary":new_plan.plan_summary,
            "plan_steps":new_plan.plan_steps,
            "completed_tasks":done_tasks,
            "completed_steps":done_steps,
            "latest_progress":progress_report.latest_progress,
            "last_message":last_message_text,
            "memories":mems,
            "output_schema_name" : "ToDoList",
            "remaining_agents":remaining_agents,
            "completed_agents":completed_agents,
            }
        updated_todo_prompt = base_todo_prompt.partial(**todo_vars)
        rendered_todo_prompt = updated_todo_prompt.format_messages(messages=[*final_turn_msgs_list,AIMessage(content="Please create a fresh To-Do list based on current progress.", name="supervisor")],**todo_vars
        )
        todo_llm = updated_todo_prompt | supervisor_llms[6].with_structured_output(ToDoList, strict=True)
        todo_vars["messages"] = rendered_todo_prompt
        todo_supervisor_expects_reply = False
        todo_results = todo_llm.invoke(
            todo_vars, config=state["_config"], prompt_cache_key = "todo_prompt"
        )
        if isinstance(todo_results, dict):
            if "structured_response" in todo_results:
                supervisor_msgs = supervisor_msgs + todo_results["messages"]
                todo_results = todo_results["structured_response"]
            else:
                msg = getattr(todo_results, "text", getattr(todo_results, "output_text", None))
                todo_results = ToDoList.model_validate(todo_results)
                if msg:
                    supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
                else:
                    supervisor_msgs.append(AIMessage(content=todo_results.model_dump_json(), name="supervisor"))
        elif isinstance(todo_results, ToDoList):
            todo_results = todo_results
            msg = getattr(todo_results, "text", getattr(todo_results, "output_text", None))
            if msg:
                supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
        assert isinstance(todo_results, ToDoList), "Failed to parse todo list result"
        todo_supervisor_expects_reply = todo_results.expect_reply
        todo_list = _dedup([t for t in todo_results.to_do_list if t not in done_tasks])

        # --- Phase 4: Route to next worker (or FINISH) ---
        supervisor_prompt = system_prompt.partial(members=options, user_prompt=user_prompt)
        completion_order = [
            agent_output_map["initial_analysis"]["state_obj_key"],
            agent_output_map["data_cleaner"]["state_obj_key"],
            agent_output_map["analyst"]["state_obj_key"],
            agent_output_map["viz_worker"]["state_obj_key_and_idx"][0],
            agent_output_map["viz_evaluator"]["state_obj_key"],
            agent_output_map["visualization"]["state_obj_key"],
            agent_output_map["report_section_worker"]["state_obj_key_and_idx"][0],
            agent_output_map["report_packager"]["state_obj_key"],
            agent_output_map["report_orchestrator"]["state_obj_key"],
            agent_output_map["file_writer"]["state_obj_key"],

        ]
        secondary_transition_map = {

            "visualization": {"viz_worker": (agent_output_map["viz_worker"]["state_obj_key_and_idx"][0],-1)},
            "report_orchestrator": {"report_section_worker": (agent_output_map["report_section_worker"]["state_obj_key_and_idx"][0],-1)},
        }


        assert curr_plan is not None, "No plan"
        assert isinstance(curr_plan, Plan), "No plan"
        stobj_key:str = agent_output_map.get(last_agent_id,agent_output_map.get("initial_analysis",{"state_obj_key":"initial_description"})).get("state_obj_key",agent_output_map.get(last_agent_id,{"state_obj_key_and_idx":"completed_plan_steps"}).get("state_obj_key_and_idx",("completed_plan_steps",-1))[0])
        last_output_obj: BaseNoExtrasModel = state.get(state.get("last_created_obj")) or state.get(stobj_key) or state.get("initial_description") or curr_plan
        last_agent_finished = last_output_obj.finished_this_task if hasattr(last_output_obj, "finished_this_task") else False
        last_agent_reply_msg = last_output_obj.reply_msg_to_supervisor if hasattr(last_output_obj, "reply_msg_to_supervisor") else ""
        last_agent_expects_reply = last_output_obj.expect_reply if hasattr(last_output_obj, "expect_reply") else False
        if not isinstance(last_agent_finished, bool):
            last_agent_finished = False
        nap = state.get("next_agent_prompt")
        if nap is None:
            out = agent_output_map.get(last_agent_id) or {}
            # If out isn't a dict, this yields {} and .get is safe
            if not isinstance(out, dict):
                out = {}
            nap = out.get("task_description") or "generate an initial analysis of the data"

        map_key = [k for k,cls in agent_output_map.items() if isinstance(last_output_obj, cls["class"])][0] or last_agent_id
        if last_agent_id != map_key:
            print(f"Warning: last_agent_id {last_agent_id} does not match map_key {map_key}")
        routing_state_vars = {
            "memories":mems,
            "user_prompt":user_prompt,
            "plan_summary":new_plan.plan_summary,
            "plan_steps":new_plan.plan_steps,
            "completed_steps":done_steps,
            "completed_tasks":done_tasks,
            "completed_agents":completed_agents,
            "remaining_agents":remaining_agents,
            "to_do_list":todo_list,
            "latest_progress":progress_report.latest_progress,
            "last_message":last_message_text,
            "next":None,
            "next_agent_prompt":nap,
            "next_agent_metadata":None,
            "last_agent_id":last_agent_id,
            "last_agent_message":latest_message,
            "output_schema_name" : "Router",
            "finished_this_task": last_agent_finished,
            "expect_reply": last_agent_expects_reply,
            "reply_msg_to_supervisor": last_agent_reply_msg,
            "initial_analysis_complete":state.get("initial_analysis_complete",False),
            "data_cleaning_complete":state.get("data_cleaning_complete",False),

        }




        rendered_routing_prompt = supervisor_prompt.format_messages(messages=[*final_turn_msgs_list,HumanMessage(content="Please route to the next worker agent. Carefully consider what has been done already and what needs done next.", name="user")],**routing_state_vars)

        routing_state_vars["messages"]=rendered_routing_prompt

        routing_llm = supervisor_prompt | supervisor_llms[1].with_structured_output(Router, strict=True)
        routing_supervisor_expects_reply = False
        routing = routing_llm.invoke(routing_state_vars, config=state["_config"], prompt_cache_key = "routing_prompt")
        if isinstance(routing, dict):
            if "structured_response" in routing:
                supervisor_msgs = supervisor_msgs + routing["messages"]
                routing = routing["structured_response"]

            else:
                msg = getattr(todo_results, "text", getattr(todo_results, "output_text", None))
                routing = Router.model_validate(**routing)
                if msg:
                    supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
                else:
                    supervisor_msgs.append(AIMessage(content=routing.model_dump_json(), name="supervisor"))
        elif isinstance(routing, Router):
            routing = routing
            msg = getattr(routing, "text", getattr(routing, "output_text", None))
            if msg:
                supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
            else:
                supervisor_msgs.append(AIMessage(content=routing.model_dump_json(), name="supervisor"))

        assert isinstance(routing, Router), "Failed to parse routing result"
        routing_supervisor_expects_reply = routing.expect_reply
        goto = routing.next


        if last_agent_expects_reply and goto != last_agent_id or progress_supervisor_expects_reply or plan_supervisor_expects_reply or todo_supervisor_expects_reply or routing_supervisor_expects_reply:
            replies_map_bools = {last_agent_expects_reply: "last_agent", progress_supervisor_expects_reply: "progress", plan_supervisor_expects_reply: "plan", todo_supervisor_expects_reply: "todo", routing_supervisor_expects_reply: "routing"}
            replies_order = ["last_agent", "progress", "plan", "todo", "routing"]
            needs_replies = [v for k,v in replies_map_bools.items() if k]
            needs_replies.sort(key=lambda x: replies_order.index(x))
            this_last_agent_reply_msg = last_agent_reply_msg
            this_last_agent_finished = last_agent_finished
            this_last_agent_id = last_agent_id
            this_nap = nap
            reply_objs = []
            for reply_key in needs_replies:
                if reply_key == "last_agent":
                    this_last_agent_reply_msg = last_agent_reply_msg
                    this_last_agent_finished = last_agent_finished
                    this_last_agent_id = last_agent_id
                    this_nap = nap
                elif reply_key == "progress":
                    this_last_agent_reply_msg = progress_report.reply_msg_to_supervisor
                    this_last_agent_finished = progress_report.finished_this_task
                    this_last_agent_id = "progress"
                    this_nap = "To review progress and update the progress report based on the current state."
                elif reply_key == "plan":
                    this_last_agent_reply_msg = new_plan.reply_msg_to_supervisor
                    this_last_agent_finished = new_plan.finished_this_task
                    this_last_agent_id = "plan"
                    this_nap = "To formulate or reformulate the plan based on current progress and completed steps, based on the current state."
                elif reply_key == "todo":
                    this_last_agent_reply_msg = todo_results.reply_msg_to_supervisor
                    this_last_agent_finished = todo_results.finished_this_task
                    this_last_agent_id = "todo"
                    this_nap = "To create a fresh To-Do list based on current progress and completed steps, based on the current state and the plan and progress."
                elif reply_key == "routing":
                    this_last_agent_reply_msg = routing.reply_msg_to_supervisor
                    this_last_agent_finished = routing.finished_this_task
                    this_last_agent_id = "routing"
                    this_nap = "To route to the next worker agent, based on the current state, also providing an instructional message prompt for the next worker agent."
                reply_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content= ("You are a Supervisor agent assistant managing these workers: \n{members}\n."

                    "Your current task is only to reply to agent workers that have sent you a message. The following context will be used to help you reply: \n"
            "User request: {user_prompt}\n\n"
            "Goal: a thorough EDA + strong visuals + a final report (Markdown, PDF, HTML) saved to disk.\n"
            "The Initial Analysis agent simply produces an initial description of the dataset and a data sample in the form of the 'InititialDescription' class found keyed as 'initial_description'. \n"
            "The Initial Analysis agent MUST be finished before any other agents can begin. \n"
            "The Data Cleaner (aka 'data_cleaner') needs to save the cleaned data and provide a way to access the newly cleaned dataset. The Data Cleaner returns the cleaned data in the form of the 'CleaningMetadata' class found keyed as 'cleaning_metadata'.\n"
                    "The Analyst (aka 'analyst') produces insights from the data. The Analyst returns the insights in the form of the 'AnalysisInsights' class found keyed as 'analysis_insights'.\n"
            "The visualization agent produces images (keyed as 'visualization_results). Files are saved to disk via FileWriter, aka 'file_writer'. FileWriter returns the file metadata in the form of the 'FileResults' class usually found keyed as 'file_results'.\n"
            "The various report agents generate the final report.\n"
            "Memories that might help:\n{memories}\n"
            "Here is the current plan as it stands:"
            "{plan_summary}"
            "Steps:"
            "\n{plan_steps}\n"

            "Already marked complete (steps):"
            "\n{completed_steps}\n"

            "Already marked complete (tasks):"
            "\n{completed_tasks}\n"

            "The following agent workers have marked their tasks as completed, though of course you should always verify yourself:"
            "\n{completed_agents}\n"

            "The following agent workers have NOT yet marked their tasks complete:"
            "\n{remaining_agents}\n"

            "Remaining To-Do (may include items that are actually done; verify from the work):"
            "\n{to_do_list}\n"

            "Here is the latest progress report:"
            "{latest_progress}"

            "The last message passed into state was:"
            "{last_message}"
                    "Please reply to the agent worker specified below using the SendAgentMessageNoRouting class schema. Carefully consider what to say and how it may impact the workflow. Keep it simple.")),
                    AIMessage(content=this_last_agent_reply_msg, name=this_last_agent_id),
                    HumanMessage(content=("The last agent to have been invoked was {last_agent_id}, whom you had given the following task (may be paraphrased): {next_agent_prompt} \n: They left the following message for you, the supervisor:"
                "{reply_msg_to_supervisor}\n"
            "They {finished_this_task} the task you gave them, and they {expect_reply} a reply from you. Please reply to the agent worker agent using the SendAgentMessageNoRouting class schema. Carefully consider what to say and how it may impact the workflow. Keep it simple."),name="user"),
                    MessagesPlaceholder(variable_name="messages"),
                ])
                reply_prompt = reply_prompt.partial(reply_msg_to_supervisor=this_last_agent_reply_msg, finished_this_task=this_last_agent_finished, expect_reply=True, last_agent_id=this_last_agent_id, next_agent_prompt=this_nap)
                routing_state_vars.pop("messages")
                rendered_reply_prompt = reply_prompt.format_messages(messages=[HumanMessage(content="Please formulate a reply to the above message.", name="user")],**routing_state_vars)

                replying_supervisor_llm = reply_prompt | supervisor_llms[2].with_structured_output(SendAgentMessageNoRouting, strict=True)
                routing_state_vars["messages"] = rendered_reply_prompt
                reply_result = replying_supervisor_llm.invoke(routing_state_vars, config=state["_config"], prompt_cache_key = "reply_prompt")
                reply_obj = None
                if isinstance(reply_result, dict):
                    if "structured_response" in reply_result:
                        supervisor_msgs = supervisor_msgs + reply_result["messages"]
                        reply_obj = reply_result["structured_response"]
                else:
                    if isinstance(reply_result, SendAgentMessageNoRouting):
                        reply_obj = reply_result
                        supervisor_msgs.append(AIMessage(content=reply_result.model_dump_json(), name="supervisor"))
                assert reply_obj is not None, "Failed to parse reply result"
                assert isinstance(reply_obj, SendAgentMessageNoRouting), "Failed to parse reply result"
                reply_objs.append((reply_obj,AIMessage(content=this_last_agent_reply_msg, name=this_last_agent_id)))
            reply_msgs = {} # {reply_msg.recipent:{"reply_obj":reply_obj,"reply_msg":AIMessage(...),"critical":reply_msg.is_message_critical,"emergency_reroute":(reply_msg.emergency_reroute,reply_msg.recipent), output_needs_recreated: reply_obj.agent_obj_needs_recreated_bool}}
            for reply_obj in reply_objs:
                assert isinstance(reply_obj[0], SendAgentMessageNoRouting), "Failed to parse reply result"
                if reply_obj[0].recipient in ["supervisor", "progress", "routing", "plan", "todo"]:
                    reply_msgs[reply_obj[0].recipient] = {"reply_obj": reply_obj[0], "reply_msg": HumanMessage(content=reply_obj[0].message, name="user"), "orig_msg": reply_obj[1],"critical": reply_obj[0].is_message_critical, "emergency_reroute": (reply_obj[0].immediate_emergency_reroute_to_recipient, reply_obj[0].recipient), "output_needs_recreated": reply_obj[0].agent_obj_needs_recreated_bool}
                else:
                    reply_msgs[reply_obj[0].recipient] = {"reply_obj": reply_obj[0], "reply_msg": AIMessage(content=reply_obj[0].message, name="supervisor"),"orig_msg": reply_obj[1], "critical": reply_obj[0].is_message_critical, "emergency_reroute": (reply_obj[0].immediate_emergency_reroute_to_recipient, reply_obj[0].recipient), "output_needs_recreated": reply_obj[0].agent_obj_needs_recreated_bool}
            sv_roles = ["supervisor", "progress","plan", "todo","routing"]
            supervisor_replies = {recip:reply_data for recip,reply_data in reply_msgs.items() if recip in sv_roles}

            priority_sorted_reply_keys = []
            for recip,reply_data in supervisor_replies.items():
                if recip == "progress":
                    priority_sorted_reply_keys.insert(0,recip)
                elif recip == "plan":
                    priority_sorted_reply_keys.insert(1,recip)
                elif recip == "todo":
                    priority_sorted_reply_keys.insert(2,recip)
                elif recip == "routing":
                    priority_sorted_reply_keys.insert(3,recip)
                else:
                    priority_sorted_reply_keys.append(recip)
            temp_sorted = {} #{key:score}
            downcount = len(priority_sorted_reply_keys) +1
            for key in priority_sorted_reply_keys:
                downcount -= 1
                if key in supervisor_replies:
                    score_ = 0 + (0.5 * downcount)
                    if supervisor_replies[key]["agent_obj_needs_recreated_bool"]:
                        score_ += 1
                    if supervisor_replies[key]["critical"]:
                        score_ += 2
                    if supervisor_replies[key]["emergency_reroute"][0]:
                        score_ += 2
                    temp_sorted[key] = score_
                else:
                    temp_sorted[key] = 0
            temp_sorted_list = []
            for key,score in temp_sorted.items():
                temp_sorted_list.append((key,score))
            temp_sorted_list.sort(key=lambda x: x[1], reverse=True)
            class ConversationalResponse(BaseModel):
                """Respond in a conversational manner. Be kind and helpful."""
                response: str = Field(description="A conversational response to the user's query")
            for key,score in temp_sorted_list:
                if (supervisor_replies[key]["reply_obj"].is_message_critical or supervisor_replies[key]["reply_obj"].immediate_emergency_reroute_to_recipient):
                    if key == "progress":




                        cst_schema = schema_for_completed_steps(curr_plan)
                        class FinalProgressResponse(BaseModel):
                            final_output: Union[Annotated[CompletedStepsAndTasks,AfterValidator(_assert_sorted_completed_no_dups)], ConversationalResponse]

                        progress_prompt = ChatPromptTemplate.from_messages([
                            SystemMessage(content=PROGRESS_ACCOUNTING_STR),
                            supervisor_replies[key]["orig_msg"],
                            supervisor_replies[key]["reply_msg"],
                            MessagesPlaceholder(variable_name="messages"),
                        ])
                        progress_vars = {
                            "user_prompt":user_prompt,
                            "plan_summary":curr_plan.plan_summary,
                            "plan_steps":curr_plan.plan_steps,
                            "completed_steps":done_steps,
                            "completed_tasks":done_tasks,
                            "to_do_list":todo_list,
                            "latest_progress":progress_report.latest_progress,
                            "completed_agents":completed_agents,
                            "remaining_agents":remaining_agents,
                            "last_message":last_message_text,
                            "memories":_mem_text(last_message_text),
                            "cleaning_metadata":state.get("cleaning_metadata",None),
                            "output_schema_name" : "FinalProgressResponse",
                            "initial_description":state.get("initial_description",None),
                            "cleaned_dataset_description":state.get("cleaned_dataset_description",None),
                            "analysis_insights":state.get("analysis_insights",None),
                            "visualization_results":state.get("visualization_results",None),
                            }
                        if supervisor_replies[key]["reply_obj"].agent_obj_needs_recreated_bool:
                            cst_llm = supervisor_llms[5].with_structured_output(CompletedStepsAndTasks, strict=True)
                            updated_progress_prompt = progress_prompt.partial(**progress_vars)
                            rendered_progress_prompt = progress_prompt.format_messages(**progress_vars)
                            progress_vars["messages"] = rendered_progress_prompt
                            progress_llm = updated_progress_prompt | cst_llm | RunnableLambda(_parse_cst_with_plan(curr_plan))
                            progress_result: CompletedStepsAndTasks = progress_llm.invoke(progress_vars, config=state["_config"], prompt_cache_key = "progress_prompt")
                            progress = None
                            if isinstance(progress_result, CompletedStepsAndTasks):
                                progress_supervisor_expects_reply = progress_result.expect_reply
                                progress = progress_result
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            elif isinstance(progress_result, str):
                                progress = CompletedStepsAndTasks.model_validate_json(progress_result)
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            assert progress, "Failed to parse progress result"
                            assert isinstance(progress, CompletedStepsAndTasks), "Failed to parse progress result"
                            progress_report = progress.progress_report
                            # Merge (dedup) newly completed items
                            done_steps = _dedup(done_steps + (progress.completed_steps or []))
                            done_tasks = _dedup(done_tasks + (progress.finished_tasks or []))
                            # Remove completed steps from the current plan (safe filter)
                            if isinstance(curr_plan, Plan) and isinstance(curr_plan.plan_steps, list):
                                for pstep in curr_plan.plan_steps:
                                    if pstep in done_steps and not pstep.is_step_complete:
                                        pstep.is_step_complete = True
                        else:
                            prog_llm = supervisor_llms[5].with_structured_output(ConversationalResponse, strict=True)
                            updated_progress_prompt = progress_prompt.partial(**progress_vars)
                            rendered_progress_prompt = progress_prompt.format_messages(**progress_vars)
                            progress_vars["messages"] = rendered_progress_prompt
                            progress_llm = updated_progress_prompt | prog_llm
                            progress_result: ConversationalResponse = progress_llm.invoke(progress_vars, config=state["_config"], prompt_cache_key = "progress_prompt")
                            progress = None
                            if isinstance(progress_result, ConversationalResponse):
                                progress_supervisor_expects_reply = progress_result.expect_reply
                                progress = progress_result
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            elif isinstance(progress_result, str):
                                progress = ConversationalResponse.model_validate_json(progress_result)
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            assert progress, "Failed to parse progress result"
                            assert isinstance(progress, ConversationalResponse), "Failed to parse progress result"
                            supervisor_msgs.append(supervisor_replies[key]["reply_msg"])
                            supervisor_msgs.append(AIMessage(content=progress.response, name="supervisor"))
                            progress_report = progress.response

                    elif key == "plan":
                        class FinalPlanResponse(BaseModel):
                            final_output: Union[Plan, ConversationalResponse]
                        replan_vars={
                              "user_prompt":user_prompt,
                              "current_plan":curr_plan,
                              "plan_summary":curr_plan.plan_summary,
                              "plan_steps":curr_plan.plan_steps,
                              "past_steps":done_steps,
                              "latest_progress":progress_report.latest_progress,
                              "output_schema_name" : "FinalPlanResponse",
                              "completed_tasks":done_tasks,
                              "completed_agents":completed_agents,
                              "remaining_agents":remaining_agents,
                              "memories":_mem_text(last_message_text),
                              "to_do_list":todo_list,
                          }
                        prompt_for_planning = replan_prompt
                        planning_llm = supervisor_llms[4]
                        plan_prompt_key = "replan_prompt"
                        # --- Phase 2: Replan against current reality ---
                        if curr_plan.plan_title.strip() == "" or _count == 1 or curr_plan.plan_summary.strip() == "":
                            curr_plan.plan_title = "Initial Plan Needed"
                            curr_plan.plan_summary = "No plan has been developed yet. Please create one!"
                            curr_plan.plan_steps = []
                            todo_list = []
                            prompt_for_planning = plan_prompt
                            replan_vars = {
                                "user_prompt":user_prompt,
                                "output_schema_name" : "Plan",
                                "agents": options,
                            }
                            planning_llm = supervisor_llms[3]
                            plan_prompt_key = "plan_prompt"


                        base_replan_prompt = prompt_for_planning
                        updated_replan_prompt = base_replan_prompt.partial(**replan_vars)
                        rendered_new_plan_prompt = updated_replan_prompt.format_messages(messages=[supervisor_replies[key]["orig_msg"],supervisor_replies[key]["reply_msg"]],**replan_vars)

                        mems = _mem_text(user_prompt)
                        planning_supervisor_llm = updated_replan_prompt | planning_llm.with_structured_output(FinalPlanResponse, strict=True)
                        replan_vars["messages"] = rendered_new_plan_prompt
                        plan_supervisor_expects_reply = False
                        new_plan = planning_supervisor_llm.invoke(replan_vars, config=state["_config"], prompt_cache_key = plan_prompt_key)
                        if isinstance(new_plan, dict):
                            if "structured_response" in new_plan:
                                supervisor_msgs = supervisor_msgs + new_plan["messages"]
                                new_plan = new_plan["structured_response"]
                            else:
                                new_plan = Plan.model_validate(new_plan)
                                supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
                        elif isinstance(new_plan, Plan):
                            new_plan = new_plan
                            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
                        elif isinstance(new_plan, str):
                            supervisor_msgs.append(AIMessage(content=new_plan, name="supervisor"))
                            new_plan = Plan.model_validate_json(new_plan)

                        else:
                            new_plan = Plan(plan_title="", plan_summary="", plan_steps=[], finished_this_task=False, reply_msg_to_supervisor="This plan still needs thought out", expect_reply=True)
                            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
                        assert isinstance(new_plan, Plan), "Failed to parse plan result"
                        plan_supervisor_expects_reply = new_plan.expect_reply
                        prev_plan = curr_plan
                        curr_plan = new_plan

            supervisor_replies = temp_sorted






        next_agent_prompt = routing.next_agent_prompt

        new_messages: List[BaseMessage] = [*supervisor_msgs,AIMessage(content=next_agent_prompt, name="supervisor")]


        return {
                "messages": new_messages,
                "_count_": _count,
                "next_agent_prompt": next_agent_prompt,
                "current_plan": new_plan,
                "to_do_list": todo_list,
                "completed_plan_steps": done_steps,
                "completed_tasks": done_tasks,
                "latest_progress": progress_report.latest_progress,
                "plan_summary": new_plan.plan_summary,
                "plan_steps": new_plan.plan_steps,
                "user_prompt": user_prompt,
                "next_agent_metadata": routing.next_agent_metadata,
                "progress_reports": [progress_report.latest_progress],
                "next": goto,
                "last_agent_id": "supervisor",
                "last_agent_message": new_messages[-1],
            }


    supervisor_node.name = "supervisor"

```

**Purpose**: Converts text into vector embeddings for semantic search
**Integration**: Required by InMemoryStore for similarity-based memory retrieval
**Process Flow**: text → OpenAI embeddings → 1536-dimension vectors → stored with memories

#### 1.2.2 InMemoryStore Configuration (Cell 13: ff7w7v0dtWBy)

```python
in_memory_store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": _embed_docs,
    }
)
```

**Key Features**:
- **Namespace Organization**: Memories stored in dedicated `("memories",)` namespace
- **Vector Search**: 1536-dimensional embeddings for semantic similarity
- **Serialization**: Built-in pickle serialization for persistence
- **Global Access**: Shared across all agents in the system

#### 1.2.3 LangMem Tool Creation (Cell 13: ff7w7v0dtWBy)

```python
mem_manage = create_manage_memory_tool(namespace=("memories",))
mem_search = create_search_memory_tool(namespace=("memories",))
```

**Tool Distribution Strategy**:
- Memory tools are added to ALL agent tool lists
- Deduplication ensures tools aren't added multiple times
- Consistent namespace `("memories",)` across all tools

#### 1.2.4 Memory Text Helper Function (Cell 13: ff7w7v0dtWBy)

```python
def _mem_text(query: str, limit: int = 5) -> str:
    try:
        items = in_memory_store.search(("memories",), query=query, limit=limit)
        if not items:
            return "None."
        # items are dict-like; stringify safely
        return "\n".join(str(it) for it in items)
    except Exception:
        return "None."

# -------------------------
# Agent factories
# -------------------------
def create_data_cleaner_agent(initial_description: InitialDescription, df_ids: List[str] = []):

    checkpointer = InMemorySaver()
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in data_cleaning_tools)
    init_df_id_str = ", /n".join(df_ids)
    init_dc_vars = {"available_df_ids":init_df_id_str,"dataset_description":initial_description.dataset_description,
                    "tool_descriptions":tool_descriptions,"tooling_guidelines" : DEFAULT_TOOLING_GUIDELINES,"output_format" : CleaningMetadata.model_json_schema(),"memories" : "No memories yet",
                    "data_sample":initial_description.data_sample}
    prompt = data_cleaner_prompt_template.partial(**init_dc_vars)
    # NOTE: response_format prefers a Pydantic model class, not a JSON schema string
    return create_react_agent(
        data_cleaner_llm,
        tools=data_cleaning_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=CleaningMetadata,
        prompt=prompt,
        name="data_cleaner",
        version="v2",
    )

def create_initial_analysis_agent(user_prompt: str, df_ids: List[str] = []):
    init_df_id_str = ", /n".join(df_ids)

    checkpointer = InMemorySaver()
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in init_analyst_tools)
    init_ia_vars = {"available_df_ids":init_df_id_str,"dataset_description":initial_description.dataset_description,
                    "tool_descriptions":tool_descriptions,"tooling_guidelines" : DEFAULT_TOOLING_GUIDELINES,"output_format" : InitialDescription.model_json_schema(),"memories" : "No memories yet",
                    "data_sample":initial_description.data_sample,"user_prompt":user_prompt}
    prompt = analyst_prompt_template_initial.partial(**init_ia_vars)

    return create_react_agent(
        initial_analyst_llm,
        tools=init_analyst_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=InitialDescription,
        prompt=prompt,
        name="initial_analysis",
        version="v2",
    )

def create_analyst_agent(initial_description: InitialDescription, df_ids: List[str] = []):
    init_df_id_str = ", /n".join(df_ids)
    checkpointer = InMemorySaver()
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in analyst_tools)
    init_analyst_vars = {"available_df_ids":init_df_id_str,"cleaned_dataset_description":initial_description.dataset_description,
                    "tool_descriptions":tool_descriptions,"output_format" : AnalysisInsights.model_json_schema(),"memories" : "No memories yet",
                    "data_sample":initial_description.data_sample}
    prompt = analyst_prompt_template_main.partial(**init_analyst_vars)
    return create_react_agent(
        analyst_llm,
        tools=analyst_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=AnalysisInsights,
        prompt=prompt,
        name="analyst",
        version="v2",
    )

def create_file_writer_agent(df_ids: List[str] = []):
    init_df_id_str = ", /n".join(df_ids)
    checkpointer = InMemorySaver()
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in file_writer_tools)
    init_fw_vars = {"available_df_ids":init_df_id_str,"tool_descriptions":tool_descriptions,"tooling_guidelines" : DEFAULT_TOOLING_GUIDELINES, "output_format" : FileResults.model_json_schema(),
                    "memories" : "No memories yet", "file_content": "No content yet","file_name": "No file name yet", "file_type": "No file type yet"}
    # NOTE: response_format prefers a Pydantic model class, not a JSON schema string
    prompt = file_writer_prompt_template.partial(**init_fw_vars)
    return create_react_agent(
        file_writer_llm,
        tools=file_writer_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        prompt=prompt,
        response_format=FileResults.model_json_schema(),   # 👈
        name="file_writer",
        version="v2",
    )

def create_visualization_agent(df_ids: List[str] = []):
    init_df_id_str = ", /n".join(df_ids)
    checkpointer = InMemorySaver()
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in visualization_tools)
    init_vis_vars = {"available_df_ids":init_df_id_str,"tool_descriptions":tool_descriptions,"tooling_guidelines" : DEFAULT_TOOLING_GUIDELINES, "output_format" : VisualizationResults.model_json_schema(),
                    "memories" : "No memories yet", "analysis_insights": "No analysis insights yet","cleaned_dataset_description": "No cleaned dataset description yet"}

    prompt = visualization_prompt_template.partial(**init_vis_vars)
    return create_react_agent(
        visualization_orchestrator_llm,
        tools=visualization_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=VisualizationResults,
        prompt=prompt,
        name="visualization",
        version="v2",
    )

def create_viz_evaluator_agent():
    checkpointer = InMemorySaver()

    init_viz_vars = {"output_format" : VizFeedback.model_json_schema(), "memories" : "No memories yet", "analysis_insights": "No analysis insights yet","cleaned_dataset_description": "No cleaned dataset description yet",
                    "visualization_results": "No visualization results yet"}
    prompt = viz_evaluator_prompt_template.partial(**init_viz_vars)
    return create_react_agent(
        viz_evaluator_llm,
        tools=[list_visualizations, get_visualization],
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=VizFeedback,
        prompt=prompt,
        name="viz_evaluator",
        version="v2",
    )

def create_report_generator_agent(df_ids: List[str] = [], rg_agent_task : Literal["outline","section","package"] = "outline"):
    init_df_id_str = ", /n".join(df_ids)
    checkpointer = InMemorySaver()
    output_format_map = {"outline" : {"output_format" : ReportOutline.model_json_schema(), "report_task": "generate a report outline", "name": "report_orchestrator","llm": report_orchestrator_llm},
                    "section" : {"output_format" : Section.model_json_schema(), "report_task": "generate a section of the report", "name": "report_section_worker","llm": report_section_worker_llm},
                    "package" : {"output_format" : ReportResults.model_json_schema(), "report_task": "generate a full report package in PDF, Markdown, and HTML", "name": "report_packager","llm": report_packager_llm}}
    output_format = output_format_map[rg_agent_task]
    report_task = output_format["report_task"]
    tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in report_generator_tools)
    init_rg_vars = {"available_df_ids":init_df_id_str,"tool_descriptions":tool_descriptions,"tooling_guidelines" : DEFAULT_TOOLING_GUIDELINES, "output_format" : output_format,
                    "memories" : "No memories yet", "analysis_insights": "No analysis insights yet", "cleaned_dataset_description": "No cleaned dataset description yet",
                    "visualization_results": "No visualization results yet", "report_task": report_task}

    prompt = report_generator_prompt_template.partial(**init_rg_vars)
    return create_react_agent(
        output_format_map[rg_agent_task]["llm"],
        tools=report_generator_tools,
        state_schema=State,
        checkpointer=checkpointer,
        store=in_memory_store,
        response_format=ReportOutline,
        prompt=prompt,
        name=output_format_map[rg_agent_task]["name"],
        version="v2",
    )

# (optional) simple memory write helper
@lru_cache(maxsize=128)
def update_memory(state: Union[MessagesState, State], config: RunnableConfig, *, memstore: InMemoryStore):
    user_id = str(config.get("configurable", {}).get("user_id", "user"))
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    memstore.put(namespace, memory_id, {"memory": state["messages"][-1].text()})


# -------------------------
# Supervisor
# -------------------------
from pydantic import BaseModel, Field
from typing import Literal

def make_supervisor_node(supervisor_llms: List[BaseChatModel], members: list[str], user_prompt: str):
    #    [big_picture_llm,router_llm, reply_llm, plan_llm, replan_llm, progress_llm, todo_llm],

    options = list(dict.fromkeys(members + ["FINISH"]))  # keep order, dedupe

    system_prompt = ChatPromptTemplate.from_messages(
    [ SystemMessage(content= (
        "You are a supervisor managing these workers: {members}.\n"
        "User request: {user_prompt}\n\n"
        "Goal: a thorough EDA + strong visuals + a final report (Markdown, PDF, HTML) saved to disk.\n"
        "Before each handoff, think step-by-step and maintain (1) the Plan and (2) a To-Do list.\n"
        "Only route to workers that still have work; FINISH when everything is done."
        "The Initial Analysis agent (aka 'initial_analysis') simply produces an initial description of the dataset and a data sample in the form of the 'InititialDescription' class found keyed as 'initial_description'. \n"
        "The Initial Analysis agent should be finished before your first turn, only route back to Initial Analysis agent if it did not finish. The Initial Analysis agent MUST be finished before any other agents can begin. \n"
        "The Data Cleaner (aka 'data_cleaner') needs to save the cleaned data and provide a way to access the newly cleaned dataset. The Data Cleaner returns the cleaned data in the form of the 'CleaningMetadata' class found keyed as 'cleaning_metadata'.\n"
        "The Data Cleaner MUST also be finished before Analyst or other agents (besides Initial Analysis) can begin. \n"
        "The Analyst (aka 'analyst') produces insights from the data. The Analyst returns the insights in the form of the 'AnalysisInsights' class found keyed as 'analysis_insights'.\n"
        "If the visualization agent produces images (keyed as 'visualization_results), ensure they are saved to disk (directly or via FileWriter, aka 'file_writer'). FileWriter returns the file metadata in the form of the 'FileResults' class found keyed as 'file_results'.\n"
        "If the report is complete (saved in a disk path that is keyed under 'final_report_path'), ensure all three formats are saved to disk.\n"
        "Memories that might help:\n{memories}\n"
        "Here is the current plan as it stands:"
        "{plan_summary}"
        "Steps:"
        "\n{plan_steps}\n"

        "Already marked complete (steps):"
        "\n{completed_steps}\n"

        "Already marked complete (tasks):"
        "\n{completed_tasks}\n"

        "The following agent workers have marked their tasks as completed, though of course you should always verify yourself:"
        "\n{completed_agents}\n"

        "The following agent workers have NOT yet marked their tasks complete:"
        "\n{remaining_agents}\n"

        "Remaining To-Do (may include items that are actually done; verify from the work):"
        "\n{to_do_list}\n"

        "Here is the latest progress report:"
        "{latest_progress}"

        "The last message passed into state was:"
        "{last_message}"

        "The last agent to have been invoked was {last_agent_id}, whom you had given the following task as a message: {next_agent_prompt} \n: They left the following message for you, the supervisor:"
        "{reply_msg_to_supervisor}\n"
        "They {finished_this_task} the task you gave them, and they {expect_reply} a reply from you. If you choose to reply to them and you also choose to route back to them, put the message in Router.next_agent_prompt."
        "However if you plan to route to a different agent, hold off on the reply for now, you will be prompted for it after choosing the next worker agent to route to.\n"

        "Perhaps the following memories may be helpful:"
        "\n{memories}\n"

        "You will encode your decisions into the Router class: next to assign the next worker/agent, next_agent_prompt to instruct them (use prompt engineering knowledge to instruct on the goal but leave the details up to the agent)."
        "The process will require constant two-way communication with the workers, including checking their work and tracking progress."
        "To send instructions to the agent you route to, use the next_agent_prompt field in the Router class. If you also need to send data as a payload, use the next_agent_metadata field."

    )),MessagesPlaceholder(variable_name="messages")]
    )
    supervisor_prompt = system_prompt.partial(members=options, user_prompt=user_prompt)

    # Reintroduce a dedicated progress-accounting prompt (fixed)
    PROGRESS_ACCOUNTING_STR = """Since a full turn has passed, review all prior messages and state to mark which plan steps and tasks are complete.

Your main objective from the user:
{user_prompt}

Current plan:
{plan_summary}
Steps:
\n{plan_steps}\n

Already marked complete (steps):
{completed_steps}

Already marked complete (tasks):
\n{completed_tasks}\n

"The following agent workers have marked their tasks as completed, though of course you should always verify yourself:"
"\n{completed_agents}\n"

"The following agent workers have NOT yet marked their tasks complete:"
"\n{remaining_agents}\n"

"Here is the latest progress report:"
"{latest_progress}"

"The last message passed into state was:"
"{last_message}"

"Memories that might help:"
"\n{memories}\n"

Remaining To-Do (may include items that are actually done; verify from the work):
\n{to_do_list}\n

Return only the updated lists of completed steps and completed tasks, based on actual work observed.
"""

    class Router(BaseNoExtrasModel):
        next: AgentId = Field(..., description="Next agent to invoke.")
        next_agent_prompt: str = Field(..., description="Actionable prompt for the selected worker.")
        next_agent_metadata: Optional[NextAgentMetadata]
    def _dedup(seq):
        return list(dict.fromkeys(seq or []))
    def _parse_cst_with_plan(plan: Plan):
        def _inner(raw: dict) -> CompletedStepsAndTasks:
            # If the OpenAI SDK returns a JSON string, load it first; otherwise dict is fine.
            if isinstance(raw, str):
                return CompletedStepsAndTasks.model_validate_json(raw, context={"plan": plan})
            return CompletedStepsAndTasks.model_validate(raw, context={"plan": plan})
        return _inner
    def schema_for_completed_steps(plan: Plan) -> dict:
        # Base schema from Pydantic (includes top-level fields & ProgressReport, etc.)
        base = CompletedStepsAndTasks.model_json_schema()

        # Allowed item shapes for completed_steps (one per plan step)
        allowed_anyof = []
        for ps in plan.plan_steps:
            allowed_anyof.append({
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    # BaseNoExtrasModel fields:
                    "reply_msg_to_supervisor": {"type": "string"},
                    "finished_this_task": {"type": "boolean"},
                    "expect_reply": {"type": "boolean"},
                    # The identity triplet is locked to this exact plan step:
                    "step_number": {"type":"number","const": ps.step_number},
                    "step_name": {"type": "string","const": ps.step_name},
                    "step_description": {"type": "string","const": ps.step_description},
                    # Must be completed:
                    "is_step_complete": {"type": "boolean","const": True},
                },
                "required": [
                    "reply_msg_to_supervisor", "finished_this_task", "expect_reply",
                    "step_number", "step_name", "step_description", "is_step_complete",
                ],
            })

        # Replace the completed_steps field to allow only those shapes
        base["properties"]["completed_steps"] = {
            "type": "array",
            "items": {"anyOf": allowed_anyof},
            # NOTE: JSON Schema's uniqueItems checks whole-object equality;
            # base fields differing would defeat dedup. We enforce dedup by triplet in Pydantic validator above.
            # "uniqueItems": True,  # optional; harmless but not sufficient for triplet-uniqueness
        }
        return base

    def supervisor_node(state: State, config: RunnableConfig):
        _count = int(state.get("_count_", 0)) + 1
        last_count = int(state["_count_"]) - 1
        last_agent_id = state.get("last_agent_id", state.get("next", None))
        last_agent_prompt = state.get("next_agent_prompt", None)
        assert last_agent_id, "No last agent ID"
        supervisor_msgs = []
        if last_count == 0:
            progress_report: ProgressReport = ProgressReport(latest_progress="This is the first turn. and no progress has been made yet.", finished_this_task=False, reply_msg_to_supervisor="This progress report needs filled out", expect_reply=True)
        else:
            progress_str = state.get("latest_progress", "No progress has been made yet.")
            if not progress_str or not isinstance(progress_str, str):
                progress_report: ProgressReport = ProgressReport(latest_progress="No progress has been made yet.",finished_this_task=False, reply_msg_to_supervisor="This progress report needs filled out after progress has been made", expect_reply=True)
            else:
                progress_report = ProgressReport(latest_progress=progress_str,finished_this_task=False, reply_msg_to_supervisor="This progress report needs filled out after progress has been made", expect_reply=True)

        user_prompt = state["user_prompt"]
        # Completion flags → for routing context (not used to infer step/task completion)
        complete_map = {
            "initial_analysis": bool(state.get("initial_analysis_complete")),
            "data_cleaner": bool(state.get("data_cleaning_complete")),
            "analyst": bool(state.get("analyst_complete")),
            "file_writer": bool(state.get("file_writer_complete")),
            "visualization": bool(state.get("visualization_complete")),
            "report_orchestrator": bool(state.get("report_generator_complete")),
        }
        task_fin_str_map = {True: "are currently awaiting", False: "are not expecting or waiting for","True": "are currently awaiting", "False": "are not expecting or waiting for","true": "are currently awaiting", "false": "are not expecting or waiting for"}
        completed_agents = [k for k, v in complete_map.items() if v]
        remaining_agents = [k for k, v in complete_map.items() if not v]
        reply_str_map = {True: "are currently awaiting", False: "are not expecting or waiting for","True": "are currently awaiting", "False": "are not expecting or waiting for","true": "are currently awaiting", "false": "are not expecting or waiting for"}



        agent_output_map = {"initial_analysis": {"class":InitialDescription, "class_name": "InitialDescription", "schema": InitialDescription.model_json_schema(), "state_obj_key": "initial_description", "task_description": "generate an initial analysis of the data"},
                                                 "data_cleaner": {"class":CleaningMetadata, "class_name": "CleaningMetadata", "schema": CleaningMetadata.model_json_schema(), "state_obj_key": "cleaning_metadata", "task_description": "clean the data"},
                            "analyst": {"class":AnalysisInsights, "class_name": "AnalysisInsights", "schema": AnalysisInsights.model_json_schema(), "state_obj_key": "analysis_insights", "task_description": "generate insights from the data"},
                            "file_writer": {"class":FileResults, "class_name": "FileResults", "schema": FileResults.model_json_schema(), "state_obj_key": "file_results", "task_description": "write data to disk"},
                            "visualization": {"class":VisualizationResults, "class_name": "VisualizationResults", "schema": VisualizationResults.model_json_schema(), "state_obj_key": "visualization_results", "task_description": "generate visualizations from the data"},
                            "report_orchestrator": {"class":ReportOutline, "class_name": "ReportOutline", "schema": ReportOutline.model_json_schema(), "state_obj_key": "report_outline", "task_description": "generate a report outline"},
                            "report_section_worker": {"class":Section, "class_name": "Section", "schema": Section.model_json_schema(), "state_obj_key_and_idx": ("sections", -1), "task_description": "generate a section of the report"},
                            "report_packager": {"class":ReportResults, "class_name": "ReportResults", "schema": ReportResults.model_json_schema(), "state_obj_key": "report_results", "task_description": "generate a full report in PDF, Markdown, and HTML"},
                            "viz_evaluator": {"class":VizFeedback, "class_name": "VizFeedback", "schema": VizFeedback.model_json_schema(), "state_obj_key": "viz_eval_results", "task_description": "evaluate the visualizations"},
                            "viz_worker": {"class": DataVisualization, "class_name": "DataVisualization", "schema": DataVisualization.model_json_schema(), "state_obj_key_and_idx": ("visualization_results", -1), "task_description": "generate a visualization from the data"},
                            "routing": {"class":Router, "class_name": "Router", "schema": Router.model_json_schema(), "state_obj_key": "router", "task_description": "route to another agent"},
                            "progress": {"class":CompletedStepsAndTasks, "class_name": "CompletedStepsAndTasks", "schema": CompletedStepsAndTasks.model_json_schema(), "state_obj_key": "completed_plan_steps", "task_description": "progress accounting"},
                            "plan": {"class":Plan, "class_name": "Plan", "schema": Plan.model_json_schema(), "state_obj_key": "plan", "task_description": "plan generation"},
                            "todo":  {"class":ToDoList, "class_name": "ToDoList", "schema": ToDoList.model_json_schema(), "state_obj_key": "todo_list", "task_description": "to-do list generation"},
                            }
        # State hydration
        curr_plan: Plan = state.get("current_plan") or Plan(plan_summary="", plan_steps=[], plan_title="", finished_this_task=False, reply_msg_to_supervisor="This plan still needs thought out", expect_reply=True)
        done_steps: List[str] = _dedup(state.get("completed_plan_steps", []))
        done_tasks: List[str] = _dedup(state.get("completed_tasks", []))
        todo_list: List[str] = _dedup(state.get("to_do_list", []))
        latest_message = state.get("last_agent_message",None)
        last_message_text = None
        if not latest_message:
            lm_name= last_agent_id
            if lm_name == "":
                lm_name = "user"
                latest_message = HumanMessage(content="No message", name=lm_name)
                last_message_text = latest_message.text()
            else:
                # iterate in reverse from last message to first until find one with lm_name as .name attr
                for msg in reversed(state.get("messages", [])):
                    if msg.name == lm_name and msg.text():
                        latest_message = msg
                        last_message_text = latest_message.text() if isinstance(latest_message, AIMessage) else "No message"
                        break

        elif isinstance(latest_message, (HumanMessage, AIMessage)):
            last_message_text = latest_message.text()
        else:
            try:
                if getattr(latest_message, "text"):
                    last_message_text = str(getattr(latest_message, "text"))
            except:
                last_message_text = "No message"
        if not last_message_text:
            last_message_text = "No message"

        final_turn_msgs_list = state.get("final_turn_msgs_list", [latest_message])
        if not final_turn_msgs_list:
            final_turn_msgs_list = [latest_message]

        # --- Phase 1: Progress Accounting (only if we have any prior messages) ---
        progress_supervisor_expects_reply = False
        if state.get("_count_", 0) > 0 and state.get("messages", False) and state.get("_count_", 0) > last_count:
            cst_schema = schema_for_completed_steps(curr_plan)

            progress_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=PROGRESS_ACCOUNTING_STR),
                MessagesPlaceholder(variable_name="messages"),
            ])
            progress_vars = {
                "messages":final_turn_msgs_list,
                "user_prompt":user_prompt,
                "plan_summary":curr_plan.plan_summary,
                "plan_steps":curr_plan.plan_steps,
                "completed_steps":done_steps,
                "completed_tasks":done_tasks,
                "to_do_list":todo_list,
                "latest_progress":progress_report.latest_progress,
                "completed_agents":completed_agents,
                "remaining_agents":remaining_agents,
                "last_message":last_message_text,
                "memories":_mem_text(last_message_text),
                "cleaning_metadata":state.get("cleaning_metadata",None),
                "output_schema_name" : "CompletedStepsAndTasks",
                "initial_description":state.get("initial_description",None),
                "cleaned_dataset_description":state.get("cleaned_dataset_description",None),
                "analysis_insights":state.get("analysis_insights",None),
                "visualization_results":state.get("visualization_results",None),
                }
            updated_progress_prompt = progress_prompt.partial(**progress_vars)
            rendered_progress_prompt = progress_prompt.format_messages(**progress_vars)
            # cst_llm = supervisor_llm.bind(response_format={"type": "json_schema","json_schema": {"name": "CompletedStepsAndTasks", "schema": cst_schema, "strict": True},})

            cst_llm = supervisor_llms[5].with_structured_output(cst_schema, strict=True)
            progress_llm = updated_progress_prompt | cst_llm | RunnableLambda(_parse_cst_with_plan(curr_plan))
            progress_result: CompletedStepsAndTasks = progress_llm.invoke(progress_vars, config=state["_config"], prompt_cache_key = "progress_prompt")

            progress = None

            if isinstance(progress_result, CompletedStepsAndTasks):
                progress_supervisor_expects_reply = progress_result.expect_reply
                progress = progress_result
                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
            elif isinstance(progress_result, dict):
                if "structured_response" in progress_result:
                    progress = progress_result["structured_response"]
                    supervisor_msgs = supervisor_msgs + progress_result["messages"]
                else:
                    progress = CompletedStepsAndTasks.model_validate(progress_result)
                    supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
            elif isinstance(progress_result, str):
                progress = CompletedStepsAndTasks.model_validate_json(progress_result)
                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
            assert progress, "Failed to parse progress result"
            assert isinstance(progress, CompletedStepsAndTasks), "Failed to parse progress result"
            progress_report = progress.progress_report
            # Merge (dedup) newly completed items
            done_steps = _dedup(done_steps + (progress.completed_steps or []))
            done_tasks = _dedup(done_tasks + (progress.finished_tasks or []))

            # Remove completed steps from the current plan (safe filter)
            if isinstance(curr_plan, Plan) and isinstance(curr_plan.plan_steps, list):
                for pstep in curr_plan.plan_steps:
                    if pstep in done_steps and not pstep.is_step_complete:
                        pstep.is_step_complete = True

            # Trim completed tasks from To-Do
            todo_list = [t for t in todo_list if t not in done_tasks]

        #write progress report to a file in state["p
        replan_vars={
                "user_prompt":user_prompt,
                "current_plan":curr_plan,
                "plan_summary":curr_plan.plan_summary,
                "plan_steps":curr_plan.plan_steps,
                "past_steps":done_steps,
                "latest_progress":progress_report.latest_progress,
                "output_schema_name" : "Plan",
                "completed_tasks":done_tasks,
                "completed_agents":completed_agents,
                "remaining_agents":remaining_agents,
                "memories":_mem_text(last_message_text),
                "to_do_list":todo_list,
            }

        # --- Phase 1: Progress Accounting (only if we have any prior messages) ---
        prompt_for_planning = replan_prompt
        planning_llm = supervisor_llms[4]
        plan_prompt_key = "replan_prompt"
        # --- Phase 2: Replan against current reality ---
        if curr_plan.plan_title.strip() == "" or _count == 1 or curr_plan.plan_summary.strip() == "":
            curr_plan.plan_title = "Initial Plan Needed"
            curr_plan.plan_summary = "No plan has been developed yet. Please create one!"
            curr_plan.plan_steps = []
            todo_list = []
            prompt_for_planning = plan_prompt
            replan_vars = {
                "user_prompt":user_prompt,
                "output_schema_name" : "Plan",
                "agents": options,
            }
            planning_llm = supervisor_llms[3]
            plan_prompt_key = "plan_prompt"


        base_replan_prompt = prompt_for_planning
        updated_replan_prompt = base_replan_prompt.partial(**replan_vars)
        rendered_new_plan_prompt = updated_replan_prompt.format_messages(messages=[*final_turn_msgs_list,AIMessage(content="Please reformulate the plan based on current progress.", name="supervisor")],**replan_vars)

        mems = _mem_text(user_prompt)
        planning_supervisor_llm = updated_replan_prompt | planning_llm.with_structured_output(Plan, strict=True)
        replan_vars["messages"] = rendered_new_plan_prompt
        plan_supervisor_expects_reply = False
        new_plan = planning_supervisor_llm.invoke(replan_vars, config=state["_config"], prompt_cache_key = plan_prompt_key)
        if isinstance(new_plan, dict):
            if "structured_response" in new_plan:
                supervisor_msgs = supervisor_msgs + new_plan["messages"]
                new_plan = new_plan["structured_response"]
            else:
                new_plan = Plan.model_validate(new_plan)
                supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
        elif isinstance(new_plan, Plan):
            new_plan = new_plan
            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
        elif isinstance(new_plan, str):
            supervisor_msgs.append(AIMessage(content=new_plan, name="supervisor"))
            new_plan = Plan.model_validate_json(new_plan)

        else:
            new_plan = Plan(plan_title="", plan_summary="", plan_steps=[], finished_this_task=False, reply_msg_to_supervisor="This plan still needs thought out", expect_reply=True)
            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
        assert isinstance(new_plan, Plan), "Failed to parse plan result"
        plan_supervisor_expects_reply = new_plan.expect_reply
        prev_plan = curr_plan
        curr_plan = new_plan

        # --- Phase 3: Refresh To-Do list ---
        base_todo_prompt = todo_prompt
        todo_vars = {
            "user_prompt":user_prompt,
            "plan_summary":new_plan.plan_summary,
            "plan_steps":new_plan.plan_steps,
            "completed_tasks":done_tasks,
            "completed_steps":done_steps,
            "latest_progress":progress_report.latest_progress,
            "last_message":last_message_text,
            "memories":mems,
            "output_schema_name" : "ToDoList",
            "remaining_agents":remaining_agents,
            "completed_agents":completed_agents,
            }
        updated_todo_prompt = base_todo_prompt.partial(**todo_vars)
        rendered_todo_prompt = updated_todo_prompt.format_messages(messages=[*final_turn_msgs_list,AIMessage(content="Please create a fresh To-Do list based on current progress.", name="supervisor")],**todo_vars
        )
        todo_llm = updated_todo_prompt | supervisor_llms[6].with_structured_output(ToDoList, strict=True)
        todo_vars["messages"] = rendered_todo_prompt
        todo_supervisor_expects_reply = False
        todo_results = todo_llm.invoke(
            todo_vars, config=state["_config"], prompt_cache_key = "todo_prompt"
        )
        if isinstance(todo_results, dict):
            if "structured_response" in todo_results:
                supervisor_msgs = supervisor_msgs + todo_results["messages"]
                todo_results = todo_results["structured_response"]
            else:
                msg = getattr(todo_results, "text", getattr(todo_results, "output_text", None))
                todo_results = ToDoList.model_validate(todo_results)
                if msg:
                    supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
                else:
                    supervisor_msgs.append(AIMessage(content=todo_results.model_dump_json(), name="supervisor"))
        elif isinstance(todo_results, ToDoList):
            todo_results = todo_results
            msg = getattr(todo_results, "text", getattr(todo_results, "output_text", None))
            if msg:
                supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
        assert isinstance(todo_results, ToDoList), "Failed to parse todo list result"
        todo_supervisor_expects_reply = todo_results.expect_reply
        todo_list = _dedup([t for t in todo_results.to_do_list if t not in done_tasks])

        # --- Phase 4: Route to next worker (or FINISH) ---
        supervisor_prompt = system_prompt.partial(members=options, user_prompt=user_prompt)
        completion_order = [
            agent_output_map["initial_analysis"]["state_obj_key"],
            agent_output_map["data_cleaner"]["state_obj_key"],
            agent_output_map["analyst"]["state_obj_key"],
            agent_output_map["viz_worker"]["state_obj_key_and_idx"][0],
            agent_output_map["viz_evaluator"]["state_obj_key"],
            agent_output_map["visualization"]["state_obj_key"],
            agent_output_map["report_section_worker"]["state_obj_key_and_idx"][0],
            agent_output_map["report_packager"]["state_obj_key"],
            agent_output_map["report_orchestrator"]["state_obj_key"],
            agent_output_map["file_writer"]["state_obj_key"],

        ]
        secondary_transition_map = {

            "visualization": {"viz_worker": (agent_output_map["viz_worker"]["state_obj_key_and_idx"][0],-1)},
            "report_orchestrator": {"report_section_worker": (agent_output_map["report_section_worker"]["state_obj_key_and_idx"][0],-1)},
        }


        assert curr_plan is not None, "No plan"
        assert isinstance(curr_plan, Plan), "No plan"
        stobj_key:str = agent_output_map.get(last_agent_id,agent_output_map.get("initial_analysis",{"state_obj_key":"initial_description"})).get("state_obj_key",agent_output_map.get(last_agent_id,{"state_obj_key_and_idx":"completed_plan_steps"}).get("state_obj_key_and_idx",("completed_plan_steps",-1))[0])
        last_output_obj: BaseNoExtrasModel = state.get(state.get("last_created_obj")) or state.get(stobj_key) or state.get("initial_description") or curr_plan
        last_agent_finished = last_output_obj.finished_this_task if hasattr(last_output_obj, "finished_this_task") else False
        last_agent_reply_msg = last_output_obj.reply_msg_to_supervisor if hasattr(last_output_obj, "reply_msg_to_supervisor") else ""
        last_agent_expects_reply = last_output_obj.expect_reply if hasattr(last_output_obj, "expect_reply") else False
        if not isinstance(last_agent_finished, bool):
            last_agent_finished = False
        nap = state.get("next_agent_prompt")
        if nap is None:
            out = agent_output_map.get(last_agent_id) or {}
            # If out isn't a dict, this yields {} and .get is safe
            if not isinstance(out, dict):
                out = {}
            nap = out.get("task_description") or "generate an initial analysis of the data"

        map_key = [k for k,cls in agent_output_map.items() if isinstance(last_output_obj, cls["class"])][0] or last_agent_id
        if last_agent_id != map_key:
            print(f"Warning: last_agent_id {last_agent_id} does not match map_key {map_key}")
        routing_state_vars = {
            "memories":mems,
            "user_prompt":user_prompt,
            "plan_summary":new_plan.plan_summary,
            "plan_steps":new_plan.plan_steps,
            "completed_steps":done_steps,
            "completed_tasks":done_tasks,
            "completed_agents":completed_agents,
            "remaining_agents":remaining_agents,
            "to_do_list":todo_list,
            "latest_progress":progress_report.latest_progress,
            "last_message":last_message_text,
            "next":None,
            "next_agent_prompt":nap,
            "next_agent_metadata":None,
            "last_agent_id":last_agent_id,
            "last_agent_message":latest_message,
            "output_schema_name" : "Router",
            "finished_this_task": last_agent_finished,
            "expect_reply": last_agent_expects_reply,
            "reply_msg_to_supervisor": last_agent_reply_msg,
            "initial_analysis_complete":state.get("initial_analysis_complete",False),
            "data_cleaning_complete":state.get("data_cleaning_complete",False),

        }




        rendered_routing_prompt = supervisor_prompt.format_messages(messages=[*final_turn_msgs_list,HumanMessage(content="Please route to the next worker agent. Carefully consider what has been done already and what needs done next.", name="user")],**routing_state_vars)

        routing_state_vars["messages"]=rendered_routing_prompt

        routing_llm = supervisor_prompt | supervisor_llms[1].with_structured_output(Router, strict=True)
        routing_supervisor_expects_reply = False
        routing = routing_llm.invoke(routing_state_vars, config=state["_config"], prompt_cache_key = "routing_prompt")
        if isinstance(routing, dict):
            if "structured_response" in routing:
                supervisor_msgs = supervisor_msgs + routing["messages"]
                routing = routing["structured_response"]

            else:
                msg = getattr(todo_results, "text", getattr(todo_results, "output_text", None))
                routing = Router.model_validate(**routing)
                if msg:
                    supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
                else:
                    supervisor_msgs.append(AIMessage(content=routing.model_dump_json(), name="supervisor"))
        elif isinstance(routing, Router):
            routing = routing
            msg = getattr(routing, "text", getattr(routing, "output_text", None))
            if msg:
                supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
            else:
                supervisor_msgs.append(AIMessage(content=routing.model_dump_json(), name="supervisor"))

        assert isinstance(routing, Router), "Failed to parse routing result"
        routing_supervisor_expects_reply = routing.expect_reply
        goto = routing.next


        if last_agent_expects_reply and goto != last_agent_id or progress_supervisor_expects_reply or plan_supervisor_expects_reply or todo_supervisor_expects_reply or routing_supervisor_expects_reply:
            replies_map_bools = {last_agent_expects_reply: "last_agent", progress_supervisor_expects_reply: "progress", plan_supervisor_expects_reply: "plan", todo_supervisor_expects_reply: "todo", routing_supervisor_expects_reply: "routing"}
            replies_order = ["last_agent", "progress", "plan", "todo", "routing"]
            needs_replies = [v for k,v in replies_map_bools.items() if k]
            needs_replies.sort(key=lambda x: replies_order.index(x))
            this_last_agent_reply_msg = last_agent_reply_msg
            this_last_agent_finished = last_agent_finished
            this_last_agent_id = last_agent_id
            this_nap = nap
            reply_objs = []
            for reply_key in needs_replies:
                if reply_key == "last_agent":
                    this_last_agent_reply_msg = last_agent_reply_msg
                    this_last_agent_finished = last_agent_finished
                    this_last_agent_id = last_agent_id
                    this_nap = nap
                elif reply_key == "progress":
                    this_last_agent_reply_msg = progress_report.reply_msg_to_supervisor
                    this_last_agent_finished = progress_report.finished_this_task
                    this_last_agent_id = "progress"
                    this_nap = "To review progress and update the progress report based on the current state."
                elif reply_key == "plan":
                    this_last_agent_reply_msg = new_plan.reply_msg_to_supervisor
                    this_last_agent_finished = new_plan.finished_this_task
                    this_last_agent_id = "plan"
                    this_nap = "To formulate or reformulate the plan based on current progress and completed steps, based on the current state."
                elif reply_key == "todo":
                    this_last_agent_reply_msg = todo_results.reply_msg_to_supervisor
                    this_last_agent_finished = todo_results.finished_this_task
                    this_last_agent_id = "todo"
                    this_nap = "To create a fresh To-Do list based on current progress and completed steps, based on the current state and the plan and progress."
                elif reply_key == "routing":
                    this_last_agent_reply_msg = routing.reply_msg_to_supervisor
                    this_last_agent_finished = routing.finished_this_task
                    this_last_agent_id = "routing"
                    this_nap = "To route to the next worker agent, based on the current state, also providing an instructional message prompt for the next worker agent."
                reply_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content= ("You are a Supervisor agent assistant managing these workers: \n{members}\n."

                    "Your current task is only to reply to agent workers that have sent you a message. The following context will be used to help you reply: \n"
            "User request: {user_prompt}\n\n"
            "Goal: a thorough EDA + strong visuals + a final report (Markdown, PDF, HTML) saved to disk.\n"
            "The Initial Analysis agent simply produces an initial description of the dataset and a data sample in the form of the 'InititialDescription' class found keyed as 'initial_description'. \n"
            "The Initial Analysis agent MUST be finished before any other agents can begin. \n"
            "The Data Cleaner (aka 'data_cleaner') needs to save the cleaned data and provide a way to access the newly cleaned dataset. The Data Cleaner returns the cleaned data in the form of the 'CleaningMetadata' class found keyed as 'cleaning_metadata'.\n"
                    "The Analyst (aka 'analyst') produces insights from the data. The Analyst returns the insights in the form of the 'AnalysisInsights' class found keyed as 'analysis_insights'.\n"
            "The visualization agent produces images (keyed as 'visualization_results). Files are saved to disk via FileWriter, aka 'file_writer'. FileWriter returns the file metadata in the form of the 'FileResults' class usually found keyed as 'file_results'.\n"
            "The various report agents generate the final report.\n"
            "Memories that might help:\n{memories}\n"
            "Here is the current plan as it stands:"
            "{plan_summary}"
            "Steps:"
            "\n{plan_steps}\n"

            "Already marked complete (steps):"
            "\n{completed_steps}\n"

            "Already marked complete (tasks):"
            "\n{completed_tasks}\n"

            "The following agent workers have marked their tasks as completed, though of course you should always verify yourself:"
            "\n{completed_agents}\n"

            "The following agent workers have NOT yet marked their tasks complete:"
            "\n{remaining_agents}\n"

            "Remaining To-Do (may include items that are actually done; verify from the work):"
            "\n{to_do_list}\n"

            "Here is the latest progress report:"
            "{latest_progress}"

            "The last message passed into state was:"
            "{last_message}"
                    "Please reply to the agent worker specified below using the SendAgentMessageNoRouting class schema. Carefully consider what to say and how it may impact the workflow. Keep it simple.")),
                    AIMessage(content=this_last_agent_reply_msg, name=this_last_agent_id),
                    HumanMessage(content=("The last agent to have been invoked was {last_agent_id}, whom you had given the following task (may be paraphrased): {next_agent_prompt} \n: They left the following message for you, the supervisor:"
                "{reply_msg_to_supervisor}\n"
            "They {finished_this_task} the task you gave them, and they {expect_reply} a reply from you. Please reply to the agent worker agent using the SendAgentMessageNoRouting class schema. Carefully consider what to say and how it may impact the workflow. Keep it simple."),name="user"),
                    MessagesPlaceholder(variable_name="messages"),
                ])
                reply_prompt = reply_prompt.partial(reply_msg_to_supervisor=this_last_agent_reply_msg, finished_this_task=this_last_agent_finished, expect_reply=True, last_agent_id=this_last_agent_id, next_agent_prompt=this_nap)
                routing_state_vars.pop("messages")
                rendered_reply_prompt = reply_prompt.format_messages(messages=[HumanMessage(content="Please formulate a reply to the above message.", name="user")],**routing_state_vars)

                replying_supervisor_llm = reply_prompt | supervisor_llms[2].with_structured_output(SendAgentMessageNoRouting, strict=True)
                routing_state_vars["messages"] = rendered_reply_prompt
                reply_result = replying_supervisor_llm.invoke(routing_state_vars, config=state["_config"], prompt_cache_key = "reply_prompt")
                reply_obj = None
                if isinstance(reply_result, dict):
                    if "structured_response" in reply_result:
                        supervisor_msgs = supervisor_msgs + reply_result["messages"]
                        reply_obj = reply_result["structured_response"]
                else:
                    if isinstance(reply_result, SendAgentMessageNoRouting):
                        reply_obj = reply_result
                        supervisor_msgs.append(AIMessage(content=reply_result.model_dump_json(), name="supervisor"))
                assert reply_obj is not None, "Failed to parse reply result"
                assert isinstance(reply_obj, SendAgentMessageNoRouting), "Failed to parse reply result"
                reply_objs.append((reply_obj,AIMessage(content=this_last_agent_reply_msg, name=this_last_agent_id)))
            reply_msgs = {} # {reply_msg.recipent:{"reply_obj":reply_obj,"reply_msg":AIMessage(...),"critical":reply_msg.is_message_critical,"emergency_reroute":(reply_msg.emergency_reroute,reply_msg.recipent), output_needs_recreated: reply_obj.agent_obj_needs_recreated_bool}}
            for reply_obj in reply_objs:
                assert isinstance(reply_obj[0], SendAgentMessageNoRouting), "Failed to parse reply result"
                if reply_obj[0].recipient in ["supervisor", "progress", "routing", "plan", "todo"]:
                    reply_msgs[reply_obj[0].recipient] = {"reply_obj": reply_obj[0], "reply_msg": HumanMessage(content=reply_obj[0].message, name="user"), "orig_msg": reply_obj[1],"critical": reply_obj[0].is_message_critical, "emergency_reroute": (reply_obj[0].immediate_emergency_reroute_to_recipient, reply_obj[0].recipient), "output_needs_recreated": reply_obj[0].agent_obj_needs_recreated_bool}
                else:
                    reply_msgs[reply_obj[0].recipient] = {"reply_obj": reply_obj[0], "reply_msg": AIMessage(content=reply_obj[0].message, name="supervisor"),"orig_msg": reply_obj[1], "critical": reply_obj[0].is_message_critical, "emergency_reroute": (reply_obj[0].immediate_emergency_reroute_to_recipient, reply_obj[0].recipient), "output_needs_recreated": reply_obj[0].agent_obj_needs_recreated_bool}
            sv_roles = ["supervisor", "progress","plan", "todo","routing"]
            supervisor_replies = {recip:reply_data for recip,reply_data in reply_msgs.items() if recip in sv_roles}

            priority_sorted_reply_keys = []
            for recip,reply_data in supervisor_replies.items():
                if recip == "progress":
                    priority_sorted_reply_keys.insert(0,recip)
                elif recip == "plan":
                    priority_sorted_reply_keys.insert(1,recip)
                elif recip == "todo":
                    priority_sorted_reply_keys.insert(2,recip)
                elif recip == "routing":
                    priority_sorted_reply_keys.insert(3,recip)
                else:
                    priority_sorted_reply_keys.append(recip)
            temp_sorted = {} #{key:score}
            downcount = len(priority_sorted_reply_keys) +1
            for key in priority_sorted_reply_keys:
                downcount -= 1
                if key in supervisor_replies:
                    score_ = 0 + (0.5 * downcount)
                    if supervisor_replies[key]["agent_obj_needs_recreated_bool"]:
                        score_ += 1
                    if supervisor_replies[key]["critical"]:
                        score_ += 2
                    if supervisor_replies[key]["emergency_reroute"][0]:
                        score_ += 2
                    temp_sorted[key] = score_
                else:
                    temp_sorted[key] = 0
            temp_sorted_list = []
            for key,score in temp_sorted.items():
                temp_sorted_list.append((key,score))
            temp_sorted_list.sort(key=lambda x: x[1], reverse=True)
            class ConversationalResponse(BaseModel):
                """Respond in a conversational manner. Be kind and helpful."""
                response: str = Field(description="A conversational response to the user's query")
            for key,score in temp_sorted_list:
                if (supervisor_replies[key]["reply_obj"].is_message_critical or supervisor_replies[key]["reply_obj"].immediate_emergency_reroute_to_recipient):
                    if key == "progress":




                        cst_schema = schema_for_completed_steps(curr_plan)
                        class FinalProgressResponse(BaseModel):
                            final_output: Union[Annotated[CompletedStepsAndTasks,AfterValidator(_assert_sorted_completed_no_dups)], ConversationalResponse]

                        progress_prompt = ChatPromptTemplate.from_messages([
                            SystemMessage(content=PROGRESS_ACCOUNTING_STR),
                            supervisor_replies[key]["orig_msg"],
                            supervisor_replies[key]["reply_msg"],
                            MessagesPlaceholder(variable_name="messages"),
                        ])
                        progress_vars = {
                            "user_prompt":user_prompt,
                            "plan_summary":curr_plan.plan_summary,
                            "plan_steps":curr_plan.plan_steps,
                            "completed_steps":done_steps,
                            "completed_tasks":done_tasks,
                            "to_do_list":todo_list,
                            "latest_progress":progress_report.latest_progress,
                            "completed_agents":completed_agents,
                            "remaining_agents":remaining_agents,
                            "last_message":last_message_text,
                            "memories":_mem_text(last_message_text),
                            "cleaning_metadata":state.get("cleaning_metadata",None),
                            "output_schema_name" : "FinalProgressResponse",
                            "initial_description":state.get("initial_description",None),
                            "cleaned_dataset_description":state.get("cleaned_dataset_description",None),
                            "analysis_insights":state.get("analysis_insights",None),
                            "visualization_results":state.get("visualization_results",None),
                            }
                        if supervisor_replies[key]["reply_obj"].agent_obj_needs_recreated_bool:
                            cst_llm = supervisor_llms[5].with_structured_output(CompletedStepsAndTasks, strict=True)
                            updated_progress_prompt = progress_prompt.partial(**progress_vars)
                            rendered_progress_prompt = progress_prompt.format_messages(**progress_vars)
                            progress_vars["messages"] = rendered_progress_prompt
                            progress_llm = updated_progress_prompt | cst_llm | RunnableLambda(_parse_cst_with_plan(curr_plan))
                            progress_result: CompletedStepsAndTasks = progress_llm.invoke(progress_vars, config=state["_config"], prompt_cache_key = "progress_prompt")
                            progress = None
                            if isinstance(progress_result, CompletedStepsAndTasks):
                                progress_supervisor_expects_reply = progress_result.expect_reply
                                progress = progress_result
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            elif isinstance(progress_result, str):
                                progress = CompletedStepsAndTasks.model_validate_json(progress_result)
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            assert progress, "Failed to parse progress result"
                            assert isinstance(progress, CompletedStepsAndTasks), "Failed to parse progress result"
                            progress_report = progress.progress_report
                            # Merge (dedup) newly completed items
                            done_steps = _dedup(done_steps + (progress.completed_steps or []))
                            done_tasks = _dedup(done_tasks + (progress.finished_tasks or []))
                            # Remove completed steps from the current plan (safe filter)
                            if isinstance(curr_plan, Plan) and isinstance(curr_plan.plan_steps, list):
                                for pstep in curr_plan.plan_steps:
                                    if pstep in done_steps and not pstep.is_step_complete:
                                        pstep.is_step_complete = True
                        else:
                            prog_llm = supervisor_llms[5].with_structured_output(ConversationalResponse, strict=True)
                            updated_progress_prompt = progress_prompt.partial(**progress_vars)
                            rendered_progress_prompt = progress_prompt.format_messages(**progress_vars)
                            progress_vars["messages"] = rendered_progress_prompt
                            progress_llm = updated_progress_prompt | prog_llm
                            progress_result: ConversationalResponse = progress_llm.invoke(progress_vars, config=state["_config"], prompt_cache_key = "progress_prompt")
                            progress = None
                            if isinstance(progress_result, ConversationalResponse):
                                progress_supervisor_expects_reply = progress_result.expect_reply
                                progress = progress_result
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            elif isinstance(progress_result, str):
                                progress = ConversationalResponse.model_validate_json(progress_result)
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            assert progress, "Failed to parse progress result"
                            assert isinstance(progress, ConversationalResponse), "Failed to parse progress result"
                            supervisor_msgs.append(supervisor_replies[key]["reply_msg"])
                            supervisor_msgs.append(AIMessage(content=progress.response, name="supervisor"))
                            progress_report = progress.response

                    elif key == "plan":
                        class FinalPlanResponse(BaseModel):
                            final_output: Union[Plan, ConversationalResponse]
                        replan_vars={
                              "user_prompt":user_prompt,
                              "current_plan":curr_plan,
                              "plan_summary":curr_plan.plan_summary,
                              "plan_steps":curr_plan.plan_steps,
                              "past_steps":done_steps,
                              "latest_progress":progress_report.latest_progress,
                              "output_schema_name" : "FinalPlanResponse",
                              "completed_tasks":done_tasks,
                              "completed_agents":completed_agents,
                              "remaining_agents":remaining_agents,
                              "memories":_mem_text(last_message_text),
                              "to_do_list":todo_list,
                          }
                        prompt_for_planning = replan_prompt
                        planning_llm = supervisor_llms[4]
                        plan_prompt_key = "replan_prompt"
                        # --- Phase 2: Replan against current reality ---
                        if curr_plan.plan_title.strip() == "" or _count == 1 or curr_plan.plan_summary.strip() == "":
                            curr_plan.plan_title = "Initial Plan Needed"
                            curr_plan.plan_summary = "No plan has been developed yet. Please create one!"
                            curr_plan.plan_steps = []
                            todo_list = []
                            prompt_for_planning = plan_prompt
                            replan_vars = {
                                "user_prompt":user_prompt,
                                "output_schema_name" : "Plan",
                                "agents": options,
                            }
                            planning_llm = supervisor_llms[3]
                            plan_prompt_key = "plan_prompt"


                        base_replan_prompt = prompt_for_planning
                        updated_replan_prompt = base_replan_prompt.partial(**replan_vars)
                        rendered_new_plan_prompt = updated_replan_prompt.format_messages(messages=[supervisor_replies[key]["orig_msg"],supervisor_replies[key]["reply_msg"]],**replan_vars)

                        mems = _mem_text(user_prompt)
                        planning_supervisor_llm = updated_replan_prompt | planning_llm.with_structured_output(FinalPlanResponse, strict=True)
                        replan_vars["messages"] = rendered_new_plan_prompt
                        plan_supervisor_expects_reply = False
                        new_plan = planning_supervisor_llm.invoke(replan_vars, config=state["_config"], prompt_cache_key = plan_prompt_key)
                        if isinstance(new_plan, dict):
                            if "structured_response" in new_plan:
                                supervisor_msgs = supervisor_msgs + new_plan["messages"]
                                new_plan = new_plan["structured_response"]
                            else:
                                new_plan = Plan.model_validate(new_plan)
                                supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
                        elif isinstance(new_plan, Plan):
                            new_plan = new_plan
                            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
                        elif isinstance(new_plan, str):
                            supervisor_msgs.append(AIMessage(content=new_plan, name="supervisor"))
                            new_plan = Plan.model_validate_json(new_plan)

                        else:
                            new_plan = Plan(plan_title="", plan_summary="", plan_steps=[], finished_this_task=False, reply_msg_to_supervisor="This plan still needs thought out", expect_reply=True)
                            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
                        assert isinstance(new_plan, Plan), "Failed to parse plan result"
                        plan_supervisor_expects_reply = new_plan.expect_reply
                        prev_plan = curr_plan
                        curr_plan = new_plan

            supervisor_replies = temp_sorted






        next_agent_prompt = routing.next_agent_prompt

        new_messages: List[BaseMessage] = [*supervisor_msgs,AIMessage(content=next_agent_prompt, name="supervisor")]


        return {
                "messages": new_messages,
                "_count_": _count,
                "next_agent_prompt": next_agent_prompt,
                "current_plan": new_plan,
                "to_do_list": todo_list,
                "completed_plan_steps": done_steps,
                "completed_tasks": done_tasks,
                "latest_progress": progress_report.latest_progress,
                "plan_summary": new_plan.plan_summary,
                "plan_steps": new_plan.plan_steps,
                "user_prompt": user_prompt,
                "next_agent_metadata": routing.next_agent_metadata,
                "progress_reports": [progress_report.latest_progress],
                "next": goto,
                "last_agent_id": "supervisor",
                "last_agent_message": new_messages[-1],
            }


    supervisor_node.name = "supervisor"

```

**Purpose**: Provides formatted memory search results for prompt templates
**Error Handling**: Returns "None." if no memories found or on exception
**Usage**: Called in supervisor prompts with `_mem_text(query)` for contextual memory

#### 1.2.5 Memory Update Function (Cell 13: ff7w7v0dtWBy)

```python
def update_memory(state: Union[MessagesState, State], config: RunnableConfig, *, memstore: InMemoryStore):
    user_id = str(config.get("configurable", {}).get("user_id", "user"))
    namespace = (user_id, "memories")
    memory_id = str(uuid.uuid4())
    memstore.put(namespace, memory_id, {"memory": state["messages"][-1].text()})


# -------------------------
# Supervisor
# -------------------------
from pydantic import BaseModel, Field
from typing import Literal

def make_supervisor_node(supervisor_llms: List[BaseChatModel], members: list[str], user_prompt: str):
    #    [big_picture_llm,router_llm, reply_llm, plan_llm, replan_llm, progress_llm, todo_llm],

    options = list(dict.fromkeys(members + ["FINISH"]))  # keep order, dedupe

    system_prompt = ChatPromptTemplate.from_messages(
    [ SystemMessage(content= (
        "You are a supervisor managing these workers: {members}.\n"
        "User request: {user_prompt}\n\n"
        "Goal: a thorough EDA + strong visuals + a final report (Markdown, PDF, HTML) saved to disk.\n"
        "Before each handoff, think step-by-step and maintain (1) the Plan and (2) a To-Do list.\n"
        "Only route to workers that still have work; FINISH when everything is done."
        "The Initial Analysis agent (aka 'initial_analysis') simply produces an initial description of the dataset and a data sample in the form of the 'InititialDescription' class found keyed as 'initial_description'. \n"
        "The Initial Analysis agent should be finished before your first turn, only route back to Initial Analysis agent if it did not finish. The Initial Analysis agent MUST be finished before any other agents can begin. \n"
        "The Data Cleaner (aka 'data_cleaner') needs to save the cleaned data and provide a way to access the newly cleaned dataset. The Data Cleaner returns the cleaned data in the form of the 'CleaningMetadata' class found keyed as 'cleaning_metadata'.\n"
        "The Data Cleaner MUST also be finished before Analyst or other agents (besides Initial Analysis) can begin. \n"
        "The Analyst (aka 'analyst') produces insights from the data. The Analyst returns the insights in the form of the 'AnalysisInsights' class found keyed as 'analysis_insights'.\n"
        "If the visualization agent produces images (keyed as 'visualization_results), ensure they are saved to disk (directly or via FileWriter, aka 'file_writer'). FileWriter returns the file metadata in the form of the 'FileResults' class found keyed as 'file_results'.\n"
        "If the report is complete (saved in a disk path that is keyed under 'final_report_path'), ensure all three formats are saved to disk.\n"
        "Memories that might help:\n{memories}\n"
        "Here is the current plan as it stands:"
        "{plan_summary}"
        "Steps:"
        "\n{plan_steps}\n"

        "Already marked complete (steps):"
        "\n{completed_steps}\n"

        "Already marked complete (tasks):"
        "\n{completed_tasks}\n"

        "The following agent workers have marked their tasks as completed, though of course you should always verify yourself:"
        "\n{completed_agents}\n"

        "The following agent workers have NOT yet marked their tasks complete:"
        "\n{remaining_agents}\n"

        "Remaining To-Do (may include items that are actually done; verify from the work):"
        "\n{to_do_list}\n"

        "Here is the latest progress report:"
        "{latest_progress}"

        "The last message passed into state was:"
        "{last_message}"

        "The last agent to have been invoked was {last_agent_id}, whom you had given the following task as a message: {next_agent_prompt} \n: They left the following message for you, the supervisor:"
        "{reply_msg_to_supervisor}\n"
        "They {finished_this_task} the task you gave them, and they {expect_reply} a reply from you. If you choose to reply to them and you also choose to route back to them, put the message in Router.next_agent_prompt."
        "However if you plan to route to a different agent, hold off on the reply for now, you will be prompted for it after choosing the next worker agent to route to.\n"

        "Perhaps the following memories may be helpful:"
        "\n{memories}\n"

        "You will encode your decisions into the Router class: next to assign the next worker/agent, next_agent_prompt to instruct them (use prompt engineering knowledge to instruct on the goal but leave the details up to the agent)."
        "The process will require constant two-way communication with the workers, including checking their work and tracking progress."
        "To send instructions to the agent you route to, use the next_agent_prompt field in the Router class. If you also need to send data as a payload, use the next_agent_metadata field."

    )),MessagesPlaceholder(variable_name="messages")]
    )
    supervisor_prompt = system_prompt.partial(members=options, user_prompt=user_prompt)

    # Reintroduce a dedicated progress-accounting prompt (fixed)
    PROGRESS_ACCOUNTING_STR = """Since a full turn has passed, review all prior messages and state to mark which plan steps and tasks are complete.

Your main objective from the user:
{user_prompt}

Current plan:
{plan_summary}
Steps:
\n{plan_steps}\n

Already marked complete (steps):
{completed_steps}

Already marked complete (tasks):
\n{completed_tasks}\n

"The following agent workers have marked their tasks as completed, though of course you should always verify yourself:"
"\n{completed_agents}\n"

"The following agent workers have NOT yet marked their tasks complete:"
"\n{remaining_agents}\n"

"Here is the latest progress report:"
"{latest_progress}"

"The last message passed into state was:"
"{last_message}"

"Memories that might help:"
"\n{memories}\n"

Remaining To-Do (may include items that are actually done; verify from the work):
\n{to_do_list}\n

Return only the updated lists of completed steps and completed tasks, based on actual work observed.
"""

    class Router(BaseNoExtrasModel):
        next: AgentId = Field(..., description="Next agent to invoke.")
        next_agent_prompt: str = Field(..., description="Actionable prompt for the selected worker.")
        next_agent_metadata: Optional[NextAgentMetadata]
    def _dedup(seq):
        return list(dict.fromkeys(seq or []))
    def _parse_cst_with_plan(plan: Plan):
        def _inner(raw: dict) -> CompletedStepsAndTasks:
            # If the OpenAI SDK returns a JSON string, load it first; otherwise dict is fine.
            if isinstance(raw, str):
                return CompletedStepsAndTasks.model_validate_json(raw, context={"plan": plan})
            return CompletedStepsAndTasks.model_validate(raw, context={"plan": plan})
        return _inner
    def schema_for_completed_steps(plan: Plan) -> dict:
        # Base schema from Pydantic (includes top-level fields & ProgressReport, etc.)
        base = CompletedStepsAndTasks.model_json_schema()

        # Allowed item shapes for completed_steps (one per plan step)
        allowed_anyof = []
        for ps in plan.plan_steps:
            allowed_anyof.append({
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    # BaseNoExtrasModel fields:
                    "reply_msg_to_supervisor": {"type": "string"},
                    "finished_this_task": {"type": "boolean"},
                    "expect_reply": {"type": "boolean"},
                    # The identity triplet is locked to this exact plan step:
                    "step_number": {"type":"number","const": ps.step_number},
                    "step_name": {"type": "string","const": ps.step_name},
                    "step_description": {"type": "string","const": ps.step_description},
                    # Must be completed:
                    "is_step_complete": {"type": "boolean","const": True},
                },
                "required": [
                    "reply_msg_to_supervisor", "finished_this_task", "expect_reply",
                    "step_number", "step_name", "step_description", "is_step_complete",
                ],
            })

        # Replace the completed_steps field to allow only those shapes
        base["properties"]["completed_steps"] = {
            "type": "array",
            "items": {"anyOf": allowed_anyof},
            # NOTE: JSON Schema's uniqueItems checks whole-object equality;
            # base fields differing would defeat dedup. We enforce dedup by triplet in Pydantic validator above.
            # "uniqueItems": True,  # optional; harmless but not sufficient for triplet-uniqueness
        }
        return base

    def supervisor_node(state: State, config: RunnableConfig):
        _count = int(state.get("_count_", 0)) + 1
        last_count = int(state["_count_"]) - 1
        last_agent_id = state.get("last_agent_id", state.get("next", None))
        last_agent_prompt = state.get("next_agent_prompt", None)
        assert last_agent_id, "No last agent ID"
        supervisor_msgs = []
        if last_count == 0:
            progress_report: ProgressReport = ProgressReport(latest_progress="This is the first turn. and no progress has been made yet.", finished_this_task=False, reply_msg_to_supervisor="This progress report needs filled out", expect_reply=True)
        else:
            progress_str = state.get("latest_progress", "No progress has been made yet.")
            if not progress_str or not isinstance(progress_str, str):
                progress_report: ProgressReport = ProgressReport(latest_progress="No progress has been made yet.",finished_this_task=False, reply_msg_to_supervisor="This progress report needs filled out after progress has been made", expect_reply=True)
            else:
                progress_report = ProgressReport(latest_progress=progress_str,finished_this_task=False, reply_msg_to_supervisor="This progress report needs filled out after progress has been made", expect_reply=True)

        user_prompt = state["user_prompt"]
        # Completion flags → for routing context (not used to infer step/task completion)
        complete_map = {
            "initial_analysis": bool(state.get("initial_analysis_complete")),
            "data_cleaner": bool(state.get("data_cleaning_complete")),
            "analyst": bool(state.get("analyst_complete")),
            "file_writer": bool(state.get("file_writer_complete")),
            "visualization": bool(state.get("visualization_complete")),
            "report_orchestrator": bool(state.get("report_generator_complete")),
        }
        task_fin_str_map = {True: "are currently awaiting", False: "are not expecting or waiting for","True": "are currently awaiting", "False": "are not expecting or waiting for","true": "are currently awaiting", "false": "are not expecting or waiting for"}
        completed_agents = [k for k, v in complete_map.items() if v]
        remaining_agents = [k for k, v in complete_map.items() if not v]
        reply_str_map = {True: "are currently awaiting", False: "are not expecting or waiting for","True": "are currently awaiting", "False": "are not expecting or waiting for","true": "are currently awaiting", "false": "are not expecting or waiting for"}



        agent_output_map = {"initial_analysis": {"class":InitialDescription, "class_name": "InitialDescription", "schema": InitialDescription.model_json_schema(), "state_obj_key": "initial_description", "task_description": "generate an initial analysis of the data"},
                                                 "data_cleaner": {"class":CleaningMetadata, "class_name": "CleaningMetadata", "schema": CleaningMetadata.model_json_schema(), "state_obj_key": "cleaning_metadata", "task_description": "clean the data"},
                            "analyst": {"class":AnalysisInsights, "class_name": "AnalysisInsights", "schema": AnalysisInsights.model_json_schema(), "state_obj_key": "analysis_insights", "task_description": "generate insights from the data"},
                            "file_writer": {"class":FileResults, "class_name": "FileResults", "schema": FileResults.model_json_schema(), "state_obj_key": "file_results", "task_description": "write data to disk"},
                            "visualization": {"class":VisualizationResults, "class_name": "VisualizationResults", "schema": VisualizationResults.model_json_schema(), "state_obj_key": "visualization_results", "task_description": "generate visualizations from the data"},
                            "report_orchestrator": {"class":ReportOutline, "class_name": "ReportOutline", "schema": ReportOutline.model_json_schema(), "state_obj_key": "report_outline", "task_description": "generate a report outline"},
                            "report_section_worker": {"class":Section, "class_name": "Section", "schema": Section.model_json_schema(), "state_obj_key_and_idx": ("sections", -1), "task_description": "generate a section of the report"},
                            "report_packager": {"class":ReportResults, "class_name": "ReportResults", "schema": ReportResults.model_json_schema(), "state_obj_key": "report_results", "task_description": "generate a full report in PDF, Markdown, and HTML"},
                            "viz_evaluator": {"class":VizFeedback, "class_name": "VizFeedback", "schema": VizFeedback.model_json_schema(), "state_obj_key": "viz_eval_results", "task_description": "evaluate the visualizations"},
                            "viz_worker": {"class": DataVisualization, "class_name": "DataVisualization", "schema": DataVisualization.model_json_schema(), "state_obj_key_and_idx": ("visualization_results", -1), "task_description": "generate a visualization from the data"},
                            "routing": {"class":Router, "class_name": "Router", "schema": Router.model_json_schema(), "state_obj_key": "router", "task_description": "route to another agent"},
                            "progress": {"class":CompletedStepsAndTasks, "class_name": "CompletedStepsAndTasks", "schema": CompletedStepsAndTasks.model_json_schema(), "state_obj_key": "completed_plan_steps", "task_description": "progress accounting"},
                            "plan": {"class":Plan, "class_name": "Plan", "schema": Plan.model_json_schema(), "state_obj_key": "plan", "task_description": "plan generation"},
                            "todo":  {"class":ToDoList, "class_name": "ToDoList", "schema": ToDoList.model_json_schema(), "state_obj_key": "todo_list", "task_description": "to-do list generation"},
                            }
        # State hydration
        curr_plan: Plan = state.get("current_plan") or Plan(plan_summary="", plan_steps=[], plan_title="", finished_this_task=False, reply_msg_to_supervisor="This plan still needs thought out", expect_reply=True)
        done_steps: List[str] = _dedup(state.get("completed_plan_steps", []))
        done_tasks: List[str] = _dedup(state.get("completed_tasks", []))
        todo_list: List[str] = _dedup(state.get("to_do_list", []))
        latest_message = state.get("last_agent_message",None)
        last_message_text = None
        if not latest_message:
            lm_name= last_agent_id
            if lm_name == "":
                lm_name = "user"
                latest_message = HumanMessage(content="No message", name=lm_name)
                last_message_text = latest_message.text()
            else:
                # iterate in reverse from last message to first until find one with lm_name as .name attr
                for msg in reversed(state.get("messages", [])):
                    if msg.name == lm_name and msg.text():
                        latest_message = msg
                        last_message_text = latest_message.text() if isinstance(latest_message, AIMessage) else "No message"
                        break

        elif isinstance(latest_message, (HumanMessage, AIMessage)):
            last_message_text = latest_message.text()
        else:
            try:
                if getattr(latest_message, "text"):
                    last_message_text = str(getattr(latest_message, "text"))
            except:
                last_message_text = "No message"
        if not last_message_text:
            last_message_text = "No message"

        final_turn_msgs_list = state.get("final_turn_msgs_list", [latest_message])
        if not final_turn_msgs_list:
            final_turn_msgs_list = [latest_message]

        # --- Phase 1: Progress Accounting (only if we have any prior messages) ---
        progress_supervisor_expects_reply = False
        if state.get("_count_", 0) > 0 and state.get("messages", False) and state.get("_count_", 0) > last_count:
            cst_schema = schema_for_completed_steps(curr_plan)

            progress_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=PROGRESS_ACCOUNTING_STR),
                MessagesPlaceholder(variable_name="messages"),
            ])
            progress_vars = {
                "messages":final_turn_msgs_list,
                "user_prompt":user_prompt,
                "plan_summary":curr_plan.plan_summary,
                "plan_steps":curr_plan.plan_steps,
                "completed_steps":done_steps,
                "completed_tasks":done_tasks,
                "to_do_list":todo_list,
                "latest_progress":progress_report.latest_progress,
                "completed_agents":completed_agents,
                "remaining_agents":remaining_agents,
                "last_message":last_message_text,
                "memories":_mem_text(last_message_text),
                "cleaning_metadata":state.get("cleaning_metadata",None),
                "output_schema_name" : "CompletedStepsAndTasks",
                "initial_description":state.get("initial_description",None),
                "cleaned_dataset_description":state.get("cleaned_dataset_description",None),
                "analysis_insights":state.get("analysis_insights",None),
                "visualization_results":state.get("visualization_results",None),
                }
            updated_progress_prompt = progress_prompt.partial(**progress_vars)
            rendered_progress_prompt = progress_prompt.format_messages(**progress_vars)
            # cst_llm = supervisor_llm.bind(response_format={"type": "json_schema","json_schema": {"name": "CompletedStepsAndTasks", "schema": cst_schema, "strict": True},})

            cst_llm = supervisor_llms[5].with_structured_output(cst_schema, strict=True)
            progress_llm = updated_progress_prompt | cst_llm | RunnableLambda(_parse_cst_with_plan(curr_plan))
            progress_result: CompletedStepsAndTasks = progress_llm.invoke(progress_vars, config=state["_config"], prompt_cache_key = "progress_prompt")

            progress = None

            if isinstance(progress_result, CompletedStepsAndTasks):
                progress_supervisor_expects_reply = progress_result.expect_reply
                progress = progress_result
                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
            elif isinstance(progress_result, dict):
                if "structured_response" in progress_result:
                    progress = progress_result["structured_response"]
                    supervisor_msgs = supervisor_msgs + progress_result["messages"]
                else:
                    progress = CompletedStepsAndTasks.model_validate(progress_result)
                    supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
            elif isinstance(progress_result, str):
                progress = CompletedStepsAndTasks.model_validate_json(progress_result)
                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
            assert progress, "Failed to parse progress result"
            assert isinstance(progress, CompletedStepsAndTasks), "Failed to parse progress result"
            progress_report = progress.progress_report
            # Merge (dedup) newly completed items
            done_steps = _dedup(done_steps + (progress.completed_steps or []))
            done_tasks = _dedup(done_tasks + (progress.finished_tasks or []))

            # Remove completed steps from the current plan (safe filter)
            if isinstance(curr_plan, Plan) and isinstance(curr_plan.plan_steps, list):
                for pstep in curr_plan.plan_steps:
                    if pstep in done_steps and not pstep.is_step_complete:
                        pstep.is_step_complete = True

            # Trim completed tasks from To-Do
            todo_list = [t for t in todo_list if t not in done_tasks]

        #write progress report to a file in state["p
        replan_vars={
                "user_prompt":user_prompt,
                "current_plan":curr_plan,
                "plan_summary":curr_plan.plan_summary,
                "plan_steps":curr_plan.plan_steps,
                "past_steps":done_steps,
                "latest_progress":progress_report.latest_progress,
                "output_schema_name" : "Plan",
                "completed_tasks":done_tasks,
                "completed_agents":completed_agents,
                "remaining_agents":remaining_agents,
                "memories":_mem_text(last_message_text),
                "to_do_list":todo_list,
            }

        # --- Phase 1: Progress Accounting (only if we have any prior messages) ---
        prompt_for_planning = replan_prompt
        planning_llm = supervisor_llms[4]
        plan_prompt_key = "replan_prompt"
        # --- Phase 2: Replan against current reality ---
        if curr_plan.plan_title.strip() == "" or _count == 1 or curr_plan.plan_summary.strip() == "":
            curr_plan.plan_title = "Initial Plan Needed"
            curr_plan.plan_summary = "No plan has been developed yet. Please create one!"
            curr_plan.plan_steps = []
            todo_list = []
            prompt_for_planning = plan_prompt
            replan_vars = {
                "user_prompt":user_prompt,
                "output_schema_name" : "Plan",
                "agents": options,
            }
            planning_llm = supervisor_llms[3]
            plan_prompt_key = "plan_prompt"


        base_replan_prompt = prompt_for_planning
        updated_replan_prompt = base_replan_prompt.partial(**replan_vars)
        rendered_new_plan_prompt = updated_replan_prompt.format_messages(messages=[*final_turn_msgs_list,AIMessage(content="Please reformulate the plan based on current progress.", name="supervisor")],**replan_vars)

        mems = _mem_text(user_prompt)
        planning_supervisor_llm = updated_replan_prompt | planning_llm.with_structured_output(Plan, strict=True)
        replan_vars["messages"] = rendered_new_plan_prompt
        plan_supervisor_expects_reply = False
        new_plan = planning_supervisor_llm.invoke(replan_vars, config=state["_config"], prompt_cache_key = plan_prompt_key)
        if isinstance(new_plan, dict):
            if "structured_response" in new_plan:
                supervisor_msgs = supervisor_msgs + new_plan["messages"]
                new_plan = new_plan["structured_response"]
            else:
                new_plan = Plan.model_validate(new_plan)
                supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
        elif isinstance(new_plan, Plan):
            new_plan = new_plan
            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
        elif isinstance(new_plan, str):
            supervisor_msgs.append(AIMessage(content=new_plan, name="supervisor"))
            new_plan = Plan.model_validate_json(new_plan)

        else:
            new_plan = Plan(plan_title="", plan_summary="", plan_steps=[], finished_this_task=False, reply_msg_to_supervisor="This plan still needs thought out", expect_reply=True)
            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
        assert isinstance(new_plan, Plan), "Failed to parse plan result"
        plan_supervisor_expects_reply = new_plan.expect_reply
        prev_plan = curr_plan
        curr_plan = new_plan

        # --- Phase 3: Refresh To-Do list ---
        base_todo_prompt = todo_prompt
        todo_vars = {
            "user_prompt":user_prompt,
            "plan_summary":new_plan.plan_summary,
            "plan_steps":new_plan.plan_steps,
            "completed_tasks":done_tasks,
            "completed_steps":done_steps,
            "latest_progress":progress_report.latest_progress,
            "last_message":last_message_text,
            "memories":mems,
            "output_schema_name" : "ToDoList",
            "remaining_agents":remaining_agents,
            "completed_agents":completed_agents,
            }
        updated_todo_prompt = base_todo_prompt.partial(**todo_vars)
        rendered_todo_prompt = updated_todo_prompt.format_messages(messages=[*final_turn_msgs_list,AIMessage(content="Please create a fresh To-Do list based on current progress.", name="supervisor")],**todo_vars
        )
        todo_llm = updated_todo_prompt | supervisor_llms[6].with_structured_output(ToDoList, strict=True)
        todo_vars["messages"] = rendered_todo_prompt
        todo_supervisor_expects_reply = False
        todo_results = todo_llm.invoke(
            todo_vars, config=state["_config"], prompt_cache_key = "todo_prompt"
        )
        if isinstance(todo_results, dict):
            if "structured_response" in todo_results:
                supervisor_msgs = supervisor_msgs + todo_results["messages"]
                todo_results = todo_results["structured_response"]
            else:
                msg = getattr(todo_results, "text", getattr(todo_results, "output_text", None))
                todo_results = ToDoList.model_validate(todo_results)
                if msg:
                    supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
                else:
                    supervisor_msgs.append(AIMessage(content=todo_results.model_dump_json(), name="supervisor"))
        elif isinstance(todo_results, ToDoList):
            todo_results = todo_results
            msg = getattr(todo_results, "text", getattr(todo_results, "output_text", None))
            if msg:
                supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
        assert isinstance(todo_results, ToDoList), "Failed to parse todo list result"
        todo_supervisor_expects_reply = todo_results.expect_reply
        todo_list = _dedup([t for t in todo_results.to_do_list if t not in done_tasks])

        # --- Phase 4: Route to next worker (or FINISH) ---
        supervisor_prompt = system_prompt.partial(members=options, user_prompt=user_prompt)
        completion_order = [
            agent_output_map["initial_analysis"]["state_obj_key"],
            agent_output_map["data_cleaner"]["state_obj_key"],
            agent_output_map["analyst"]["state_obj_key"],
            agent_output_map["viz_worker"]["state_obj_key_and_idx"][0],
            agent_output_map["viz_evaluator"]["state_obj_key"],
            agent_output_map["visualization"]["state_obj_key"],
            agent_output_map["report_section_worker"]["state_obj_key_and_idx"][0],
            agent_output_map["report_packager"]["state_obj_key"],
            agent_output_map["report_orchestrator"]["state_obj_key"],
            agent_output_map["file_writer"]["state_obj_key"],

        ]
        secondary_transition_map = {

            "visualization": {"viz_worker": (agent_output_map["viz_worker"]["state_obj_key_and_idx"][0],-1)},
            "report_orchestrator": {"report_section_worker": (agent_output_map["report_section_worker"]["state_obj_key_and_idx"][0],-1)},
        }


        assert curr_plan is not None, "No plan"
        assert isinstance(curr_plan, Plan), "No plan"
        stobj_key:str = agent_output_map.get(last_agent_id,agent_output_map.get("initial_analysis",{"state_obj_key":"initial_description"})).get("state_obj_key",agent_output_map.get(last_agent_id,{"state_obj_key_and_idx":"completed_plan_steps"}).get("state_obj_key_and_idx",("completed_plan_steps",-1))[0])
        last_output_obj: BaseNoExtrasModel = state.get(state.get("last_created_obj")) or state.get(stobj_key) or state.get("initial_description") or curr_plan
        last_agent_finished = last_output_obj.finished_this_task if hasattr(last_output_obj, "finished_this_task") else False
        last_agent_reply_msg = last_output_obj.reply_msg_to_supervisor if hasattr(last_output_obj, "reply_msg_to_supervisor") else ""
        last_agent_expects_reply = last_output_obj.expect_reply if hasattr(last_output_obj, "expect_reply") else False
        if not isinstance(last_agent_finished, bool):
            last_agent_finished = False
        nap = state.get("next_agent_prompt")
        if nap is None:
            out = agent_output_map.get(last_agent_id) or {}
            # If out isn't a dict, this yields {} and .get is safe
            if not isinstance(out, dict):
                out = {}
            nap = out.get("task_description") or "generate an initial analysis of the data"

        map_key = [k for k,cls in agent_output_map.items() if isinstance(last_output_obj, cls["class"])][0] or last_agent_id
        if last_agent_id != map_key:
            print(f"Warning: last_agent_id {last_agent_id} does not match map_key {map_key}")
        routing_state_vars = {
            "memories":mems,
            "user_prompt":user_prompt,
            "plan_summary":new_plan.plan_summary,
            "plan_steps":new_plan.plan_steps,
            "completed_steps":done_steps,
            "completed_tasks":done_tasks,
            "completed_agents":completed_agents,
            "remaining_agents":remaining_agents,
            "to_do_list":todo_list,
            "latest_progress":progress_report.latest_progress,
            "last_message":last_message_text,
            "next":None,
            "next_agent_prompt":nap,
            "next_agent_metadata":None,
            "last_agent_id":last_agent_id,
            "last_agent_message":latest_message,
            "output_schema_name" : "Router",
            "finished_this_task": last_agent_finished,
            "expect_reply": last_agent_expects_reply,
            "reply_msg_to_supervisor": last_agent_reply_msg,
            "initial_analysis_complete":state.get("initial_analysis_complete",False),
            "data_cleaning_complete":state.get("data_cleaning_complete",False),

        }




        rendered_routing_prompt = supervisor_prompt.format_messages(messages=[*final_turn_msgs_list,HumanMessage(content="Please route to the next worker agent. Carefully consider what has been done already and what needs done next.", name="user")],**routing_state_vars)

        routing_state_vars["messages"]=rendered_routing_prompt

        routing_llm = supervisor_prompt | supervisor_llms[1].with_structured_output(Router, strict=True)
        routing_supervisor_expects_reply = False
        routing = routing_llm.invoke(routing_state_vars, config=state["_config"], prompt_cache_key = "routing_prompt")
        if isinstance(routing, dict):
            if "structured_response" in routing:
                supervisor_msgs = supervisor_msgs + routing["messages"]
                routing = routing["structured_response"]

            else:
                msg = getattr(todo_results, "text", getattr(todo_results, "output_text", None))
                routing = Router.model_validate(**routing)
                if msg:
                    supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
                else:
                    supervisor_msgs.append(AIMessage(content=routing.model_dump_json(), name="supervisor"))
        elif isinstance(routing, Router):
            routing = routing
            msg = getattr(routing, "text", getattr(routing, "output_text", None))
            if msg:
                supervisor_msgs.append(AIMessage(content=msg, name="supervisor"))
            else:
                supervisor_msgs.append(AIMessage(content=routing.model_dump_json(), name="supervisor"))

        assert isinstance(routing, Router), "Failed to parse routing result"
        routing_supervisor_expects_reply = routing.expect_reply
        goto = routing.next


        if last_agent_expects_reply and goto != last_agent_id or progress_supervisor_expects_reply or plan_supervisor_expects_reply or todo_supervisor_expects_reply or routing_supervisor_expects_reply:
            replies_map_bools = {last_agent_expects_reply: "last_agent", progress_supervisor_expects_reply: "progress", plan_supervisor_expects_reply: "plan", todo_supervisor_expects_reply: "todo", routing_supervisor_expects_reply: "routing"}
            replies_order = ["last_agent", "progress", "plan", "todo", "routing"]
            needs_replies = [v for k,v in replies_map_bools.items() if k]
            needs_replies.sort(key=lambda x: replies_order.index(x))
            this_last_agent_reply_msg = last_agent_reply_msg
            this_last_agent_finished = last_agent_finished
            this_last_agent_id = last_agent_id
            this_nap = nap
            reply_objs = []
            for reply_key in needs_replies:
                if reply_key == "last_agent":
                    this_last_agent_reply_msg = last_agent_reply_msg
                    this_last_agent_finished = last_agent_finished
                    this_last_agent_id = last_agent_id
                    this_nap = nap
                elif reply_key == "progress":
                    this_last_agent_reply_msg = progress_report.reply_msg_to_supervisor
                    this_last_agent_finished = progress_report.finished_this_task
                    this_last_agent_id = "progress"
                    this_nap = "To review progress and update the progress report based on the current state."
                elif reply_key == "plan":
                    this_last_agent_reply_msg = new_plan.reply_msg_to_supervisor
                    this_last_agent_finished = new_plan.finished_this_task
                    this_last_agent_id = "plan"
                    this_nap = "To formulate or reformulate the plan based on current progress and completed steps, based on the current state."
                elif reply_key == "todo":
                    this_last_agent_reply_msg = todo_results.reply_msg_to_supervisor
                    this_last_agent_finished = todo_results.finished_this_task
                    this_last_agent_id = "todo"
                    this_nap = "To create a fresh To-Do list based on current progress and completed steps, based on the current state and the plan and progress."
                elif reply_key == "routing":
                    this_last_agent_reply_msg = routing.reply_msg_to_supervisor
                    this_last_agent_finished = routing.finished_this_task
                    this_last_agent_id = "routing"
                    this_nap = "To route to the next worker agent, based on the current state, also providing an instructional message prompt for the next worker agent."
                reply_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content= ("You are a Supervisor agent assistant managing these workers: \n{members}\n."

                    "Your current task is only to reply to agent workers that have sent you a message. The following context will be used to help you reply: \n"
            "User request: {user_prompt}\n\n"
            "Goal: a thorough EDA + strong visuals + a final report (Markdown, PDF, HTML) saved to disk.\n"
            "The Initial Analysis agent simply produces an initial description of the dataset and a data sample in the form of the 'InititialDescription' class found keyed as 'initial_description'. \n"
            "The Initial Analysis agent MUST be finished before any other agents can begin. \n"
            "The Data Cleaner (aka 'data_cleaner') needs to save the cleaned data and provide a way to access the newly cleaned dataset. The Data Cleaner returns the cleaned data in the form of the 'CleaningMetadata' class found keyed as 'cleaning_metadata'.\n"
                    "The Analyst (aka 'analyst') produces insights from the data. The Analyst returns the insights in the form of the 'AnalysisInsights' class found keyed as 'analysis_insights'.\n"
            "The visualization agent produces images (keyed as 'visualization_results). Files are saved to disk via FileWriter, aka 'file_writer'. FileWriter returns the file metadata in the form of the 'FileResults' class usually found keyed as 'file_results'.\n"
            "The various report agents generate the final report.\n"
            "Memories that might help:\n{memories}\n"
            "Here is the current plan as it stands:"
            "{plan_summary}"
            "Steps:"
            "\n{plan_steps}\n"

            "Already marked complete (steps):"
            "\n{completed_steps}\n"

            "Already marked complete (tasks):"
            "\n{completed_tasks}\n"

            "The following agent workers have marked their tasks as completed, though of course you should always verify yourself:"
            "\n{completed_agents}\n"

            "The following agent workers have NOT yet marked their tasks complete:"
            "\n{remaining_agents}\n"

            "Remaining To-Do (may include items that are actually done; verify from the work):"
            "\n{to_do_list}\n"

            "Here is the latest progress report:"
            "{latest_progress}"

            "The last message passed into state was:"
            "{last_message}"
                    "Please reply to the agent worker specified below using the SendAgentMessageNoRouting class schema. Carefully consider what to say and how it may impact the workflow. Keep it simple.")),
                    AIMessage(content=this_last_agent_reply_msg, name=this_last_agent_id),
                    HumanMessage(content=("The last agent to have been invoked was {last_agent_id}, whom you had given the following task (may be paraphrased): {next_agent_prompt} \n: They left the following message for you, the supervisor:"
                "{reply_msg_to_supervisor}\n"
            "They {finished_this_task} the task you gave them, and they {expect_reply} a reply from you. Please reply to the agent worker agent using the SendAgentMessageNoRouting class schema. Carefully consider what to say and how it may impact the workflow. Keep it simple."),name="user"),
                    MessagesPlaceholder(variable_name="messages"),
                ])
                reply_prompt = reply_prompt.partial(reply_msg_to_supervisor=this_last_agent_reply_msg, finished_this_task=this_last_agent_finished, expect_reply=True, last_agent_id=this_last_agent_id, next_agent_prompt=this_nap)
                routing_state_vars.pop("messages")
                rendered_reply_prompt = reply_prompt.format_messages(messages=[HumanMessage(content="Please formulate a reply to the above message.", name="user")],**routing_state_vars)

                replying_supervisor_llm = reply_prompt | supervisor_llms[2].with_structured_output(SendAgentMessageNoRouting, strict=True)
                routing_state_vars["messages"] = rendered_reply_prompt
                reply_result = replying_supervisor_llm.invoke(routing_state_vars, config=state["_config"], prompt_cache_key = "reply_prompt")
                reply_obj = None
                if isinstance(reply_result, dict):
                    if "structured_response" in reply_result:
                        supervisor_msgs = supervisor_msgs + reply_result["messages"]
                        reply_obj = reply_result["structured_response"]
                else:
                    if isinstance(reply_result, SendAgentMessageNoRouting):
                        reply_obj = reply_result
                        supervisor_msgs.append(AIMessage(content=reply_result.model_dump_json(), name="supervisor"))
                assert reply_obj is not None, "Failed to parse reply result"
                assert isinstance(reply_obj, SendAgentMessageNoRouting), "Failed to parse reply result"
                reply_objs.append((reply_obj,AIMessage(content=this_last_agent_reply_msg, name=this_last_agent_id)))
            reply_msgs = {} # {reply_msg.recipent:{"reply_obj":reply_obj,"reply_msg":AIMessage(...),"critical":reply_msg.is_message_critical,"emergency_reroute":(reply_msg.emergency_reroute,reply_msg.recipent), output_needs_recreated: reply_obj.agent_obj_needs_recreated_bool}}
            for reply_obj in reply_objs:
                assert isinstance(reply_obj[0], SendAgentMessageNoRouting), "Failed to parse reply result"
                if reply_obj[0].recipient in ["supervisor", "progress", "routing", "plan", "todo"]:
                    reply_msgs[reply_obj[0].recipient] = {"reply_obj": reply_obj[0], "reply_msg": HumanMessage(content=reply_obj[0].message, name="user"), "orig_msg": reply_obj[1],"critical": reply_obj[0].is_message_critical, "emergency_reroute": (reply_obj[0].immediate_emergency_reroute_to_recipient, reply_obj[0].recipient), "output_needs_recreated": reply_obj[0].agent_obj_needs_recreated_bool}
                else:
                    reply_msgs[reply_obj[0].recipient] = {"reply_obj": reply_obj[0], "reply_msg": AIMessage(content=reply_obj[0].message, name="supervisor"),"orig_msg": reply_obj[1], "critical": reply_obj[0].is_message_critical, "emergency_reroute": (reply_obj[0].immediate_emergency_reroute_to_recipient, reply_obj[0].recipient), "output_needs_recreated": reply_obj[0].agent_obj_needs_recreated_bool}
            sv_roles = ["supervisor", "progress","plan", "todo","routing"]
            supervisor_replies = {recip:reply_data for recip,reply_data in reply_msgs.items() if recip in sv_roles}

            priority_sorted_reply_keys = []
            for recip,reply_data in supervisor_replies.items():
                if recip == "progress":
                    priority_sorted_reply_keys.insert(0,recip)
                elif recip == "plan":
                    priority_sorted_reply_keys.insert(1,recip)
                elif recip == "todo":
                    priority_sorted_reply_keys.insert(2,recip)
                elif recip == "routing":
                    priority_sorted_reply_keys.insert(3,recip)
                else:
                    priority_sorted_reply_keys.append(recip)
            temp_sorted = {} #{key:score}
            downcount = len(priority_sorted_reply_keys) +1
            for key in priority_sorted_reply_keys:
                downcount -= 1
                if key in supervisor_replies:
                    score_ = 0 + (0.5 * downcount)
                    if supervisor_replies[key]["agent_obj_needs_recreated_bool"]:
                        score_ += 1
                    if supervisor_replies[key]["critical"]:
                        score_ += 2
                    if supervisor_replies[key]["emergency_reroute"][0]:
                        score_ += 2
                    temp_sorted[key] = score_
                else:
                    temp_sorted[key] = 0
            temp_sorted_list = []
            for key,score in temp_sorted.items():
                temp_sorted_list.append((key,score))
            temp_sorted_list.sort(key=lambda x: x[1], reverse=True)
            class ConversationalResponse(BaseModel):
                """Respond in a conversational manner. Be kind and helpful."""
                response: str = Field(description="A conversational response to the user's query")
            for key,score in temp_sorted_list:
                if (supervisor_replies[key]["reply_obj"].is_message_critical or supervisor_replies[key]["reply_obj"].immediate_emergency_reroute_to_recipient):
                    if key == "progress":




                        cst_schema = schema_for_completed_steps(curr_plan)
                        class FinalProgressResponse(BaseModel):
                            final_output: Union[Annotated[CompletedStepsAndTasks,AfterValidator(_assert_sorted_completed_no_dups)], ConversationalResponse]

                        progress_prompt = ChatPromptTemplate.from_messages([
                            SystemMessage(content=PROGRESS_ACCOUNTING_STR),
                            supervisor_replies[key]["orig_msg"],
                            supervisor_replies[key]["reply_msg"],
                            MessagesPlaceholder(variable_name="messages"),
                        ])
                        progress_vars = {
                            "user_prompt":user_prompt,
                            "plan_summary":curr_plan.plan_summary,
                            "plan_steps":curr_plan.plan_steps,
                            "completed_steps":done_steps,
                            "completed_tasks":done_tasks,
                            "to_do_list":todo_list,
                            "latest_progress":progress_report.latest_progress,
                            "completed_agents":completed_agents,
                            "remaining_agents":remaining_agents,
                            "last_message":last_message_text,
                            "memories":_mem_text(last_message_text),
                            "cleaning_metadata":state.get("cleaning_metadata",None),
                            "output_schema_name" : "FinalProgressResponse",
                            "initial_description":state.get("initial_description",None),
                            "cleaned_dataset_description":state.get("cleaned_dataset_description",None),
                            "analysis_insights":state.get("analysis_insights",None),
                            "visualization_results":state.get("visualization_results",None),
                            }
                        if supervisor_replies[key]["reply_obj"].agent_obj_needs_recreated_bool:
                            cst_llm = supervisor_llms[5].with_structured_output(CompletedStepsAndTasks, strict=True)
                            updated_progress_prompt = progress_prompt.partial(**progress_vars)
                            rendered_progress_prompt = progress_prompt.format_messages(**progress_vars)
                            progress_vars["messages"] = rendered_progress_prompt
                            progress_llm = updated_progress_prompt | cst_llm | RunnableLambda(_parse_cst_with_plan(curr_plan))
                            progress_result: CompletedStepsAndTasks = progress_llm.invoke(progress_vars, config=state["_config"], prompt_cache_key = "progress_prompt")
                            progress = None
                            if isinstance(progress_result, CompletedStepsAndTasks):
                                progress_supervisor_expects_reply = progress_result.expect_reply
                                progress = progress_result
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            elif isinstance(progress_result, str):
                                progress = CompletedStepsAndTasks.model_validate_json(progress_result)
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            assert progress, "Failed to parse progress result"
                            assert isinstance(progress, CompletedStepsAndTasks), "Failed to parse progress result"
                            progress_report = progress.progress_report
                            # Merge (dedup) newly completed items
                            done_steps = _dedup(done_steps + (progress.completed_steps or []))
                            done_tasks = _dedup(done_tasks + (progress.finished_tasks or []))
                            # Remove completed steps from the current plan (safe filter)
                            if isinstance(curr_plan, Plan) and isinstance(curr_plan.plan_steps, list):
                                for pstep in curr_plan.plan_steps:
                                    if pstep in done_steps and not pstep.is_step_complete:
                                        pstep.is_step_complete = True
                        else:
                            prog_llm = supervisor_llms[5].with_structured_output(ConversationalResponse, strict=True)
                            updated_progress_prompt = progress_prompt.partial(**progress_vars)
                            rendered_progress_prompt = progress_prompt.format_messages(**progress_vars)
                            progress_vars["messages"] = rendered_progress_prompt
                            progress_llm = updated_progress_prompt | prog_llm
                            progress_result: ConversationalResponse = progress_llm.invoke(progress_vars, config=state["_config"], prompt_cache_key = "progress_prompt")
                            progress = None
                            if isinstance(progress_result, ConversationalResponse):
                                progress_supervisor_expects_reply = progress_result.expect_reply
                                progress = progress_result
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            elif isinstance(progress_result, str):
                                progress = ConversationalResponse.model_validate_json(progress_result)
                                supervisor_msgs.append(AIMessage(content=progress.model_dump_json(), name="supervisor"))
                            assert progress, "Failed to parse progress result"
                            assert isinstance(progress, ConversationalResponse), "Failed to parse progress result"
                            supervisor_msgs.append(supervisor_replies[key]["reply_msg"])
                            supervisor_msgs.append(AIMessage(content=progress.response, name="supervisor"))
                            progress_report = progress.response

                    elif key == "plan":
                        class FinalPlanResponse(BaseModel):
                            final_output: Union[Plan, ConversationalResponse]
                        replan_vars={
                              "user_prompt":user_prompt,
                              "current_plan":curr_plan,
                              "plan_summary":curr_plan.plan_summary,
                              "plan_steps":curr_plan.plan_steps,
                              "past_steps":done_steps,
                              "latest_progress":progress_report.latest_progress,
                              "output_schema_name" : "FinalPlanResponse",
                              "completed_tasks":done_tasks,
                              "completed_agents":completed_agents,
                              "remaining_agents":remaining_agents,
                              "memories":_mem_text(last_message_text),
                              "to_do_list":todo_list,
                          }
                        prompt_for_planning = replan_prompt
                        planning_llm = supervisor_llms[4]
                        plan_prompt_key = "replan_prompt"
                        # --- Phase 2: Replan against current reality ---
                        if curr_plan.plan_title.strip() == "" or _count == 1 or curr_plan.plan_summary.strip() == "":
                            curr_plan.plan_title = "Initial Plan Needed"
                            curr_plan.plan_summary = "No plan has been developed yet. Please create one!"
                            curr_plan.plan_steps = []
                            todo_list = []
                            prompt_for_planning = plan_prompt
                            replan_vars = {
                                "user_prompt":user_prompt,
                                "output_schema_name" : "Plan",
                                "agents": options,
                            }
                            planning_llm = supervisor_llms[3]
                            plan_prompt_key = "plan_prompt"


                        base_replan_prompt = prompt_for_planning
                        updated_replan_prompt = base_replan_prompt.partial(**replan_vars)
                        rendered_new_plan_prompt = updated_replan_prompt.format_messages(messages=[supervisor_replies[key]["orig_msg"],supervisor_replies[key]["reply_msg"]],**replan_vars)

                        mems = _mem_text(user_prompt)
                        planning_supervisor_llm = updated_replan_prompt | planning_llm.with_structured_output(FinalPlanResponse, strict=True)
                        replan_vars["messages"] = rendered_new_plan_prompt
                        plan_supervisor_expects_reply = False
                        new_plan = planning_supervisor_llm.invoke(replan_vars, config=state["_config"], prompt_cache_key = plan_prompt_key)
                        if isinstance(new_plan, dict):
                            if "structured_response" in new_plan:
                                supervisor_msgs = supervisor_msgs + new_plan["messages"]
                                new_plan = new_plan["structured_response"]
                            else:
                                new_plan = Plan.model_validate(new_plan)
                                supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
                        elif isinstance(new_plan, Plan):
                            new_plan = new_plan
                            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
                        elif isinstance(new_plan, str):
                            supervisor_msgs.append(AIMessage(content=new_plan, name="supervisor"))
                            new_plan = Plan.model_validate_json(new_plan)

                        else:
                            new_plan = Plan(plan_title="", plan_summary="", plan_steps=[], finished_this_task=False, reply_msg_to_supervisor="This plan still needs thought out", expect_reply=True)
                            supervisor_msgs.append(AIMessage(content=new_plan.model_dump_json(), name="supervisor"))
                        assert isinstance(new_plan, Plan), "Failed to parse plan result"
                        plan_supervisor_expects_reply = new_plan.expect_reply
                        prev_plan = curr_plan
                        curr_plan = new_plan

            supervisor_replies = temp_sorted






        next_agent_prompt = routing.next_agent_prompt

        new_messages: List[BaseMessage] = [*supervisor_msgs,AIMessage(content=next_agent_prompt, name="supervisor")]


        return {
                "messages": new_messages,
                "_count_": _count,
                "next_agent_prompt": next_agent_prompt,
                "current_plan": new_plan,
                "to_do_list": todo_list,
                "completed_plan_steps": done_steps,
                "completed_tasks": done_tasks,
                "latest_progress": progress_report.latest_progress,
                "plan_summary": new_plan.plan_summary,
                "plan_steps": new_plan.plan_steps,
                "user_prompt": user_prompt,
                "next_agent_metadata": routing.next_agent_metadata,
                "progress_reports": [progress_report.latest_progress],
                "next": goto,
                "last_agent_id": "supervisor",
                "last_agent_message": new_messages[-1],
            }


    supervisor_node.name = "supervisor"

```

**Namespace Strategy**: 
- Uses `(user_id, "memories")` for user isolation
- Generates unique UUID for each memory entry
- Stores last message text with semantic embeddings
- Enables future retrieval via similarity search

### 1.3 Universal Memory Retrieval Pattern

#### 1.3.1 Node-Level Memory Access (Cell 17: ffsSXHWQt5Yw)

Every agent node implements the same memory retrieval pattern:

```python
def retrieve_mem(state):
    store = get_store()
    return store.search(("memories",), query=state.get("next_agent_prompt") or user_prompt, limit=5)
```

**Standardization Benefits**:
- Consistent memory access across all nodes
- Context-aware queries based on agent prompts
- Configurable result limits (default: 5 items)
- Seamless integration with agent prompt templates

#### 1.3.2 Agent-Specific Memory Integration

Each agent node function follows this pattern:

1. **Memory Retrieval**: `mems = retrieve_mem(state)`
2. **Prompt Integration**: Memory results added to prompt variables as `"memories": mems`
3. **Contextual Usage**: Agent prompts reference relevant memories for decision-making

### 1.4 Checkpointing and State Persistence

#### 1.4.1 Agent-Level Checkpointing

```python
checkpointer = InMemorySaver()
```

Used in every agent factory for individual agent state persistence.

#### 1.4.2 Graph-Level Checkpointing

The compiled graph uses `MemorySaver` for overall workflow state persistence, enabling:
- Resumable conversations
- State recovery after interruptions
- Workflow debugging and replay

### 1.5 Memory Workflow Integration

#### 1.5.1 Memory Flow in Agent Execution

1. **Entry**: Agent node calls `retrieve_mem(state)` on entry
2. **Context**: Retrieved memories integrated into agent prompt
3. **Processing**: Agent uses memory context for informed decision-making
4. **Storage**: Agent responses stored via memory tools or `update_memory()`
5. **Propagation**: New memories available to subsequent agents

#### 1.5.2 Supervisor Memory Management

The supervisor node uses memory in multiple ways:
- **Progress Tracking**: `_mem_text(last_message_text)` for contextual progress updates
- **Decision Making**: Memory context influences routing decisions
- **State Continuity**: Maintains conversation context across agent handoffs

### 1.6 Memory Search and Update Workflow

#### 1.6.1 Search Integration Process

1. Each node calls `retrieve_mem(state)` on entry
2. Uses `get_store()` to access the shared memory store
3. Searches the `("memories",)` namespace with contextual queries
4. Returns up to 5 relevant memory items
5. Memory results integrated into agent prompts

#### 1.6.2 Update Mechanism Process

1. `update_memory()` called with state and configuration
2. Extracts user ID from config for namespace isolation
3. Generates unique memory ID for each entry
4. Stores message content with semantic embeddings
5. Enables future retrieval via similarity search

---

## Part 2: Integration and Implementation Improvements

### 2.1 Current Architecture Analysis

#### 2.1.1 Strengths of Current Implementation

**✅ Comprehensive Integration**:
- LangMem tools fully integrated with InMemoryStore
- Universal memory retrieval pattern across all nodes
- Consistent namespace organization
- Semantic search capabilities enabled

**✅ Scalable Architecture**:
- User isolation through namespace strategy
- Efficient deduplication of memory tools
- Configurable memory limits and search parameters
- Global store accessibility

**✅ Robust Error Handling**:
- Graceful fallbacks when no memories exist
- Exception handling in memory operations
- Safe default values in prompt templates

#### 2.1.2 Identified Issues and Gaps

**❌ Limited Memory Types**:
- Only conversation history stored
- No specialized memory for analysis insights, cleaning patterns, or visualization preferences
- Missing structured memory for different data types

**❌ Memory Lifecycle Management**:
- No memory expiration or cleanup mechanisms
- Potential for memory store growth without bounds
- No memory prioritization or relevance scoring

**❌ Context Window Limitations**:
- Fixed 5-item retrieval limit may be insufficient for complex analyses
- No dynamic context sizing based on task complexity
- Potential loss of important historical context

**❌ Integration Gaps**:
- ChromaDB installed but not integrated into workflow
- No persistent storage beyond session (InMemoryStore)
- Limited cross-session memory continuity

### 2.2 Proposed Improvements

#### 2.2.1 Enhanced Memory Categorization

**Implementation**: Extend namespace strategy to include memory types:

```python
# Current: ("memories",)
# Proposed: ("memories", "conversation") | ("memories", "analysis") | ("memories", "cleaning") | ("memories", "visualization")

def categorized_memory_store():
    return {
        "conversation": ("memories", "conversation"),
        "analysis": ("memories", "analysis"), 
        "cleaning": ("memories", "cleaning"),
        "visualization": ("memories", "visualization"),
        "insights": ("memories", "insights"),
        "errors": ("memories", "errors")
    }
```

**Benefits**:
- Targeted memory retrieval for specific contexts
- Reduced noise in memory search results
- Better organization of historical knowledge

#### 2.2.2 Dynamic Memory Context Sizing

**Implementation**: Adaptive memory retrieval based on task complexity:

```python
def adaptive_retrieve_mem(state, task_type="default"):
    store = get_store()
    
    # Task-specific limits
    limits = {
        "initial_analysis": 3,
        "data_cleaning": 7,
        "analysis": 10,
        "visualization": 5,
        "reporting": 8,
        "default": 5
    }
    
    limit = limits.get(task_type, 5)
    query = state.get("next_agent_prompt") or state.get("user_prompt", "")
    
    return store.search(("memories",), query=query, limit=limit)
```

#### 2.2.3 Persistent Memory Storage Integration

**Implementation**: ChromaDB integration for persistent memory:

```python
import chromadb
from chromadb.config import Settings

def setup_persistent_memory():
    # ChromaDB for persistent storage
    chroma_client = chromadb.PersistentClient(
        path="./idd_memory_store",
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Hybrid approach: InMemoryStore + ChromaDB
    persistent_collection = chroma_client.get_or_create_collection(
        name="idd_memories",
        embedding_function=embeddings
    )
    
    return persistent_collection
```

#### 2.2.4 Memory Quality and Relevance Scoring

**Implementation**: Enhanced memory retrieval with quality metrics:

```python
def enhanced_memory_search(query, limit=5, min_relevance=0.7):
    raw_results = in_memory_store.search(("memories",), query=query, limit=limit*2)
    
    # Score and filter results
    scored_results = []
    for result in raw_results:
        relevance_score = calculate_relevance(query, result)
        if relevance_score >= min_relevance:
            scored_results.append((result, relevance_score))
    
    # Sort by relevance and return top results
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return [result for result, score in scored_results[:limit]]
```

#### 2.2.5 Memory Lifecycle Management

**Implementation**: Automated memory cleanup and archiving:

```python
def memory_lifecycle_manager():
    # Archive old memories
    # Cleanup low-relevance memories  
    # Compress similar memories
    # Maintain memory quality thresholds
    pass
```

### 2.3 Implementation Roadmap

#### 2.3.1 Phase 1: Memory Categorization (Immediate)
- Implement categorized namespaces
- Update all `retrieve_mem` functions to use specific categories
- Modify memory storage to include category metadata

#### 2.3.2 Phase 2: Persistent Storage Integration (Short-term)
- Integrate ChromaDB for session persistence
- Implement hybrid memory approach (InMemory + Persistent)
- Add memory migration utilities

#### 2.3.3 Phase 3: Advanced Memory Features (Medium-term)
- Implement adaptive context sizing
- Add memory quality scoring and filtering
- Develop memory lifecycle management

#### 2.3.4 Phase 4: Optimization and Scaling (Long-term)
- Performance optimization for large memory stores
- Distributed memory for multi-user scenarios
- Advanced memory analytics and insights

### 2.4 Immediate Action Items

1. **Extend Memory Categorization**: Modify existing `retrieve_mem` functions to use task-specific namespaces
2. **Implement ChromaDB Integration**: Add persistent storage alongside InMemoryStore
3. **Enhance Error Handling**: Improve memory operation robustness
4. **Add Memory Metrics**: Implement logging and monitoring for memory operations
5. **Create Memory Utilities**: Develop tools for memory inspection and management

### 2.5 Expected Benefits

- **Improved Context Awareness**: Agents will have access to more relevant historical information
- **Better Persistence**: Memory continuity across sessions and deployments
- **Enhanced Performance**: More efficient memory operations and reduced noise
- **Scalability**: System can handle larger datasets and longer conversations
- **Maintainability**: Better organization and lifecycle management of memory data

---

## Conclusion

The IDD v4 memory system demonstrates a solid foundation with comprehensive integration of LangMem tools and LangGraph's InMemoryStore. The universal memory retrieval pattern ensures consistent access across all agents, while the dual-layer architecture provides both high-level tooling and low-level vector storage capabilities.

The proposed improvements focus on enhancing memory categorization, implementing persistent storage, and adding advanced features like dynamic context sizing and quality scoring. These enhancements will significantly improve the system's ability to maintain context, learn from previous interactions, and provide more informed decision-making capabilities.

Implementation should proceed in phases, starting with memory categorization and ChromaDB integration, followed by advanced features and optimization. This approach ensures backward compatibility while progressively enhancing the memory system's capabilities.
