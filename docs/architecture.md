<!-- repo-agent-bootstrap:file-kind=architecture-doc -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Architecture

## Overview
The Intelligent Data Detective (IDD) is a multi-agent autonomous data analysis system implemented as a Jupyter-notebook pipeline. Current runnable proof work uses the committed generated notebook `IntelligentDataDetective_beta_v5_patched.ipynb` (99 cells in the W14 completion baseline), regenerated from `IntelligentDataDetective_beta_v5.ipynb` by `_patch_notebook.py`. It uses a **supervisor-worker LangGraph** pattern: a central supervisor routes a shared `State` object to specialised worker nodes, then collates results into HTML/Markdown/PDF final reports.

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

## Current completion baseline
The current completion baseline is `IDD_run_run_default_id-20260504-1338-b3079aea`, produced from `IntelligentDataDetective_beta_v5_patched.ipynb` with `IDD_SAMPLE_DATASET=retail_orders`.

Validation bar:
- `validate_run.py --latest --log-path notebook_run_log.txt --window 180` scores 12/12.
- `validate_artifact_quality.py --latest` scores 9/9.
- The run log has native structured-output markers for initial analysis, data cleaning, analysis, visualization, report orchestration, section writing, report packaging, and file writing.
- The run log has zero recovery, final-hop, native-failure, path-normalization, recursion, or traceback markers.
- Root `final_report.html`, `final_report.md`, and `final_report.pdf` exist; HTML embeds resolving visualizations and PDF is parseable.

## State flow (Cell 7)
`State` is a TypedDict with reducer-annotated fields. Every node receives the full `State` dict and returns a partial update. Reductions happen automatically via LangGraph:

| Annotation | Semantic |
|---|---|
| `Annotated[T, keep_first]` | Immutable after first non-None set |
| `Annotated[list, operator.add]` | Accumulate items (messages, artifact paths) |
| `Annotated[list, _reduce_plan_keep_sorted]` | Merge plan items, deduplicate by key, keep sorted |
| `Annotated[str, ...]` plain | Last-write-wins (status flags, routing hints) |

W14H protects visualization fan-in from last-writer loss by rebuilding `viz_join` from all available visualization channels (`viz_results`, `visualization_results`, `viz_paths`) plus discovered PNG artifacts before evaluation. Preserve this union behavior when editing visualization state.

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

### Report pipeline
`report_orchestrator` decomposes the report into sections. `report_section_worker` nodes execute in parallel. `report_join` collates. `report_packager` returns native `ReportResults`, then the renderer writes canonical `final_report.md`, `final_report.html`, and `final_report.pdf`. `file_writer` produces a no-write final manifest over existing artifacts.

## Notebook source and generation map
| File/area | Content |
|---|---|
| `_patch_notebook.py` | Durable patch source for notebook behavior changes |
| `IntelligentDataDetective_beta_v5.ipynb` | Source notebook input |
| `IntelligentDataDetective_beta_v5_patched.ipynb` | Generated runnable W14 notebook |
| Early notebook cells | Imports, environment setup, API keys, model wrappers, state definitions |
| Tool/agent cells | Tool definitions, prompt construction, agent construction, graph compilation |
| Final notebook cells | Dataset setup, execution entrypoint, artifact/report finalization |

## Security / reliability considerations
- All file writes go through `_resolve_artifact_path()` — no raw `open()` calls in tools.
- `@handle_tool_errors` wraps every tool — exceptions surface as supervisor-readable messages, not crashes.
- State reducers prevent accidental field overwrite across parallel branches.
- `DataFrameRegistry` prevents large DataFrames being serialised into State.
- Final report quality is validated by both production gates and artifact/readability gates, not by boolean completion flags alone.
<!-- repo-agent-bootstrap:managed:end -->
