<!-- repo-agent-bootstrap:file-kind=architecture-doc -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Architecture

## Overview
The Intelligent Data Detective (IDD) is a multi-agent autonomous data analysis system implemented entirely in `IntelligentDataDetective_beta_v5.ipynb` (27 cells). It uses a **supervisor-worker LangGraph** pattern: a central supervisor routes a shared `State` object to specialised worker nodes, then collates results into an HTML/PDF report.

For the full authoritative graph topology, see `idd_v4_state_graph.mmd`.

## Graph topology
```
__start__
    └── initial_analysis
            └── supervisor
                    ├── data_cleaner
                    ├── analyst
                    ├── visualization
                    │       ├── viz_worker (parallel)
                    │       └── viz_join
                    │               └── viz_evaluator
                    ├── report_orchestrator
                    │       ├── report_section_worker (parallel)
                    │       └── report_join
                    │               └── report_packager
                    └── file_writer
                            └── __end__

supervisor → EMERGENCY_MSG → __end__  (fatal error path)
```
Terminal condition: `report_done ∧ report_ready ∧ already_wrote`

## State flow (Cell 7)
`State` is a TypedDict with reducer-annotated fields. Every node receives the full `State` dict and returns a partial update. Reductions happen automatically via LangGraph:

| Annotation | Semantic |
|---|---|
| `Annotated[T, keep_first]` | Immutable after first non-None set |
| `Annotated[list, operator.add]` | Accumulate items (messages, artifact paths) |
| `Annotated[list, _reduce_plan_keep_sorted]` | Merge plan items, deduplicate by key, keep sorted |
| `Annotated[str, ...]` plain | Last-write-wins (status flags, routing hints) |

## Key subsystems

### MyChatOpenai (Cell 5)
Wrapper around `ChatOpenAI` that handles o-series model quirks and the OpenAI Responses API. **Always use `MyChatOpenai`; never `ChatOpenAI` directly.**

### DataFrameRegistry (Cell 8)
Thread-safe LRU cache; single source of truth for all DataFrames. Reference by `df_id` (UUID string). Auto-reloads from CSV on cache miss. Never pass DataFrames between functions without registry.

### Tool registration pattern (Cells 12–13)
```python
@handle_tool_errors
def tool_name(df_id: str, ...) -> tuple[str, dict]:
    validate_dataframe_exists(df_id)
    ...
typed_tool_list.append(tool_name)
```

### Memory system (Cells 14–15)
Namespaced vector memory: `('memories', '<kind>')` where `kind` ∈ `{conversation, analysis, cleaning, visualization, insights, errors}`. TTL and limits from `memory_config.yaml`.

### Report pipeline (Cells 22–24)
`report_orchestrator` decomposes the report into sections. `report_section_worker` nodes execute in parallel. `report_join` collates. `report_packager` renders HTML+PDF. `file_writer` persists via `_resolve_artifact_path()`.

## Notebook cell map
| Cells | Content |
|---|---|
| 1–3 | Imports, environment setup, API keys |
| 4–5 | `MyChatOpenai`, LLM factory helpers |
| 6–7 | `State` TypedDict, reducers, `BaseNoExtrasModel` |
| 8 | `DataFrameRegistry` |
| 9–11 | Supervisor logic and routing |
| 12–13 | Tool definitions and registration |
| 14–15 | Memory subsystem |
| 16–20 | Worker agents (cleaner, analyst, viz, report) |
| 21 | Graph assembly and compilation |
| 22–24 | Report orchestration and packaging |
| 25–27 | File writer, entry point, sample invocation |

## Security / reliability considerations
- All file writes go through `_resolve_artifact_path()` — no raw `open()` calls in tools.
- `@handle_tool_errors` wraps every tool — exceptions surface as supervisor-readable messages, not crashes.
- State reducers prevent accidental field overwrite across parallel branches.
- `DataFrameRegistry` prevents large DataFrames being serialised into State.
<!-- repo-agent-bootstrap:managed:end -->
