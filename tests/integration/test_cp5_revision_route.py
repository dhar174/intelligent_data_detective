"""SQLite-backed canonical revision path with controlled, offline model responses."""

from pathlib import Path
import json
import socket
from threading import Lock

import pandas as pd
from PIL import Image, ImageChops
import pytest
from pydantic import Field
from langgraph.checkpoint.sqlite import SqliteSaver

from idd_graph import build_graph
from idd_registry import DataFrameRegistry, override_global_registry
from idd_state import (
    AnalysisInsights,
    DataVisualization,
    VizFeedback,
    VizSpec,
    VisualizationResults,
)
from tests.unit.test_cp5_report_grounding import generated_supervisor
from tests.unit.test_cp5_report_grounding import report_agent, report_state
from tests.unit.test_native_completion import (
    ControlledChatModel,
    FLAGS,
    analysis_insights,
    analyst_state,
    execute_definitions,
    load_analyst,
)
from tests.unit.test_viz_join import figures

_VISUALIZATION_MODEL_LOCK = Lock()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden in revision graph test")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setenv("IDD_ALLOW_RECOVERY", "0")


class SequencedChatModel(ControlledChatModel):
    """Return one deterministic structured result per invocation."""

    payloads: list[dict] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = len(self._requests)
        assert index < len(self.payloads), "Unexpected extra controlled model request"
        self.payload = self.payloads[index]
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class RuntimeTaskChatModel(ControlledChatModel):
    """Select the controlled result for the newest assigned visualization task."""

    payload_by_instruction: dict[str, dict] = Field(default_factory=dict)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        contexts = [
            message for message in messages if message.name == "runtime_context"
        ]
        assert contexts, "Generated visualization worker must receive runtime context"
        runtime = json.loads(str(contexts[-1].content))
        # The rendered assignment is a projection, not the checkpointed model.
        spec = runtime["assigned_spec"]
        assert set(spec) == set(VizSpec.model_fields) - {"finished_this_task"}
        context = runtime["visualization_task"]
        assigned_id = spec["viz_id"]
        assert (
            isinstance(assigned_id, str) and assigned_id
        ), "Generated worker omitted its assigned visualization ID"
        matches = [
            (instruction, payload)
            for instruction, payload in self.payload_by_instruction.items()
            if any(
                item["visualization_id"] == assigned_id
                for item in payload["visualizations"]
            )
        ]
        assert len(matches) == 1, f"No unique controlled result for {assigned_id!r}"
        instruction, payload = matches[0]
        assert instruction in context
        assert all(
            item["visualization_id"] == assigned_id
            for item in payload["visualizations"]
        ), f"Assigned task {instruction!r} omitted its real visualization IDs"
        with _VISUALIZATION_MODEL_LOCK:
            self.payload = payload
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )


def _revision_spec(df_id, *, viz_id, title, task, viz_type, x, y=None, hue=None):
    return VizSpec(
        **FLAGS,
        title=title,
        viz_type=viz_type,
        df_id=df_id,
        viz_instructions=task,
        viz_id=viz_id,
        columns=[column for column in (x, y, hue) if column],
        x=x,
        y=y,
        hue=hue,
        bins=5 if viz_type == "histogram" else None,
        agg=None,
        query=None,
        description=f"Controlled production plot for {title}.",
        limit=None,
        style="whitegrid",
    )


def _revision_round_specs(label, df_id, round_name):
    if label == "thread-a" and round_name == "initial":
        return [
            _revision_spec(
                df_id,
                viz_id="thread-a-initial-revenue-hist",
                title="Thread A initial revenue distribution",
                task="Plot thread A initial revenue distribution.",
                viz_type="histogram",
                x="revenue",
            ),
            _revision_spec(
                df_id,
                viz_id="thread-a-initial-discount-scatter",
                title="Thread A initial discount and revenue",
                task="Plot thread A initial discount against revenue.",
                viz_type="scatter",
                x="discount",
                y="revenue",
            ),
        ]
    if label == "thread-a" and round_name == "revision":
        return [
            _revision_spec(
                df_id,
                viz_id="thread-a-revised-region-box",
                title="Thread A revised revenue range by region",
                task="Plot thread A revised revenue range by region.",
                viz_type="box",
                x="revenue",
                hue="region",
            ),
            _revision_spec(
                df_id,
                viz_id="thread-a-revised-discount-scatter",
                title="Thread A revised discount and revenue",
                task="Plot thread A revised discount against revenue.",
                viz_type="scatter",
                x="discount",
                y="revenue",
            ),
        ]
    return [
        _revision_spec(
            df_id,
            viz_id="thread-b-revenue-hist",
            title="Thread B revenue distribution",
            task="Plot thread B revenue distribution.",
            viz_type="histogram",
            x="revenue",
        ),
        _revision_spec(
            df_id,
            viz_id="thread-b-region-box",
            title="Thread B revenue range by region",
            task="Plot thread B revenue range by region.",
            viz_type="box",
            x="revenue",
            hue="region",
        ),
    ]


def _produce_real_plot(namespace, base, df_id, spec, monkeypatch):
    if spec.viz_type == "histogram":
        plot = namespace["create_histogram"]
        args = {"df_id": df_id, "columns": [spec.x], "bins": 5}
    elif spec.viz_type == "scatter":
        plot = namespace["create_scatter_plot"]
        args = {"df_id": df_id, "x": spec.x, "y": spec.y}
    elif spec.viz_type == "box":
        plot = namespace["create_box_plot"]
        args = {"df_id": df_id, "values": spec.x, "group": spec.hue}
    else:
        raise AssertionError(f"Unsupported controlled plot type: {spec.viz_type}")

    output_root = base / "plot_outputs" / spec.viz_id
    with monkeypatch.context() as plot_env:
        plot_env.setenv("IDD_ARTIFACTS_DIR", str(output_root))
        tool_result = plot.invoke(
            {
                "type": "tool_call",
                "id": spec.viz_id,
                "name": plot.name,
                "args": args,
            }
        )
    artifact = getattr(tool_result, "artifact", None) or {}
    assert (
        "image_path" in artifact
    ), f"{plot.name} did not save a PNG: {getattr(tool_result, 'content', tool_result)}"
    path = Path(artifact["image_path"]).resolve()
    assert path.is_relative_to(base.resolve())
    assert path.suffix.lower() == ".png" and path.stat().st_size > 0
    with Image.open(path) as image:
        assert image.format == "PNG"
        rgb = image.convert("RGB")
        background = Image.new("RGB", rgb.size, rgb.getpixel((0, 0)))
        assert ImageChops.difference(rgb, background).getbbox()

    return DataVisualization(
        **FLAGS,
        path=str(path),
        visualization_id=spec.viz_id,
        visualization_type=artifact.get("plot_type", spec.viz_type),
        visualization_description=spec.description,
        visualization_style=spec.style or "whitegrid",
        visualization_title=spec.title,
    )


def _thread_runtime(base, label, df_id, monkeypatch):
    import matplotlib
    from math import nan

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        has_sns = True
    except Exception:
        sns = None
        has_sns = False

    namespace = load_analyst(base)
    execute_definitions(
        namespace,
        57,
        {
            "MANDATORY_SPEC_KEYS",
            "ALLOWED_SPEC_KEYS",
            "_normalize_meta",
            "_guess_viz_type",
            "_norm_title",
            "_normalize_viz_spec",
            "visualization_orchestrator",
        },
    )
    assert namespace["analyst_node"].__code__.co_filename == "<generated Cell 57>"
    assert (
        namespace["create_analyst_agent"].__code__.co_filename == "<generated Cell 44>"
    )
    assert (
        namespace["visualization_orchestrator"].__code__.co_filename
        == "<generated Cell 57>"
    )
    for name in ("viz_worker", "viz_join", "viz_evaluator_node"):
        assert namespace[name].__code__.co_filename == "<generated Cell 57>"
    registry = DataFrameRegistry(data_dir=base / "registry")
    namespace.update(
        analyst_registry=registry,
        analyst_df_id=df_id,
        get_global_df_registry=lambda: registry,
        global_df_registry=registry,
        plt=plt,
        nan=nan,
        sns=sns,
        _HAS_SNS=has_sns,
    )
    registry.register_dataframe(
        pd.DataFrame(
            {
                "region": ["North", "South", "East", "West"] * 3,
                "revenue": [120, 95, 180, 150, 140, 105, 210, 165, 160, 115, 240, 185],
                "discount": [5, 12, 8, 10, 6, 14, 7, 9, 4, 13, 6, 11],
            }
        ),
        df_id,
    )

    initial_specs = _revision_round_specs(label, df_id, "initial")
    revision_specs = (
        _revision_round_specs(label, df_id, "revision") if label == "thread-a" else []
    )
    all_specs = initial_specs + revision_specs
    plot_results = {}
    task_payloads = {}
    with override_global_registry(registry):
        for spec in all_specs:
            item = _produce_real_plot(namespace, base, df_id, spec, monkeypatch)
            plot_results[spec.viz_id] = item
            task_payloads[spec.viz_instructions] = VisualizationResults(
                **FLAGS, visualizations=[item]
            ).model_dump()

    if label == "thread-a":
        initial_insights = analysis_insights().model_copy(
            update={
                "summary": "Thread A initial analysis.",
                "recommended_visualizations": initial_specs,
            }
        )
        revised_insights = analysis_insights().model_copy(
            update={
                "summary": "Thread A initial analysis.",
                "recommended_visualizations": revision_specs,
            }
        )
        analyst_payloads = [
            initial_insights.model_dump(),
            revised_insights.model_dump(),
        ]
        evaluation_payloads = [
            VizFeedback(
                **FLAGS,
                grade="revise",
                feedback="Thread A evaluator: correct the axis and retain both requested views.",
                redo_list=[initial_specs[0].viz_id],
            ).model_dump(),
            VizFeedback(
                **FLAGS,
                grade="acceptable",
                feedback="Thread A revised figures are acceptable.",
                redo_list=[],
            ).model_dump(),
        ]
    else:
        initial_insights = analysis_insights().model_copy(
            update={
                "summary": "Thread B isolated analysis.",
                "recommended_visualizations": initial_specs,
            }
        )
        analyst_payloads = [initial_insights.model_dump()]
        evaluation_payloads = [
            VizFeedback(
                **FLAGS,
                grade="acceptable",
                feedback="Thread B figures are acceptable.",
                redo_list=[],
            ).model_dump()
        ]

    analyst_model = SequencedChatModel(
        response_name="AnalysisInsights",
        payload={},
        payloads=analyst_payloads,
    )
    namespace["analyst_llm"] = analyst_model
    initial_state = analyst_state(base, supervisor_instruction=f"Start {label}.")
    namespace["analyst_agent"] = namespace["create_analyst_agent"](
        initial_state["initial_description"], [df_id]
    )

    visualization_model = RuntimeTaskChatModel(
        response_name="VisualizationResults",
        payload={},
        payload_by_instruction=task_payloads,
    )
    namespace["visualization_orchestrator_llm"] = visualization_model
    namespace["visualization_agent"] = namespace["create_visualization_agent"]([df_id])

    evaluator_model = SequencedChatModel(
        response_name="VizFeedback",
        payload={},
        payloads=evaluation_payloads,
    )
    namespace["viz_evaluator_llm"] = evaluator_model
    namespace["viz_evaluator_agent"] = namespace["create_viz_evaluator_agent"]()

    return {
        "namespace": namespace,
        "registry": registry,
        "df_id": df_id,
        "initial_specs": initial_specs,
        "revision_specs": revision_specs,
        "plot_results": plot_results,
        "analyst_model": analyst_model,
        "visualization_model": visualization_model,
        "evaluator_model": evaluator_model,
    }


def _thread_config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _thread_id(state, config=None):
    cfg = config or state.get("_config") or {}
    configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
    return configurable.get("thread_id")


def test_sqlite_generated_revision_route_keeps_real_plots_and_isolated_threads(
    tmp_path, monkeypatch
):
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("IDD_ALLOW_RECOVERY", "0")
    monkeypatch.setenv("MPLBACKEND", "Agg")

    threads = {
        "revision-thread-a": _thread_runtime(
            tmp_path / "thread_a", "thread-a", "retail-thread-a", monkeypatch
        ),
        "revision-thread-b": _thread_runtime(
            tmp_path / "thread_b", "thread-b", "retail-thread-b", monkeypatch
        ),
    }
    events = {thread_id: [] for thread_id in threads}
    supervisor_node = generated_supervisor()
    assert supervisor_node.__code__.co_filename == "<generated Cell 46>"

    def logged_supervisor(state, config=None):
        result = supervisor_node(state, config)
        events[_thread_id(state, config)].append(f"supervisor->{result.goto}")
        return result

    def dispatch_node(name):
        def invoke(state, config=None):
            thread_id = _thread_id(state, config)
            events[thread_id].append(name)
            return threads[thread_id]["namespace"][name](state)

        return invoke

    overrides = {
        "supervisor": logged_supervisor,
        "analyst": dispatch_node("analyst_node"),
        "visualization": dispatch_node("visualization_orchestrator"),
        "viz_worker": dispatch_node("viz_worker"),
        "viz_join": dispatch_node("viz_join"),
        "viz_evaluator": dispatch_node("viz_evaluator_node"),
    }

    def seed_thread(thread_id, label, thread):
        state = analyst_state(thread["namespace"]["WORKING_DIRECTORY"])
        state.update(
            _config=_thread_config(thread_id),
            user_prompt=f"Analyze isolated fixture {label}.",
            available_df_ids=[thread["df_id"]],
            last_agent_id="data_cleaner",
            last_created_obj="cleaning_metadata",
            initial_analysis_complete=True,
            data_cleaning_complete=True,
            analyst_complete=False,
            visualization_complete=False,
            report_generator_complete=False,
            analysis_insights=None,
            viz_tasks=[],
            viz_specs=[],
            viz_results=[],
            viz_paths=[],
            visualization_results=None,
            viz_eval_result=None,
            viz_grade=None,
            viz_feedback=None,
            _viz_retry_count=0,
        )
        config = _thread_config(thread_id)
        with override_global_registry(thread["registry"]):
            graph.update_state(config, state, as_node="data_cleaner")
            graph.invoke(None, config, interrupt_before=["report_orchestrator"])
            snapshot = graph.get_state(config)
        assert snapshot.next == ("report_orchestrator",)
        return snapshot.values

    db = tmp_path / "revision.sqlite"
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(node_overrides=overrides, checkpointer=saver)
        initial_values = seed_thread(
            "revision-thread-a", "thread-a", threads["revision-thread-a"]
        )
        isolated_values = seed_thread(
            "revision-thread-b", "thread-b", threads["revision-thread-b"]
        )

        for thread_id, values in (
            ("revision-thread-a", initial_values),
            ("revision-thread-b", isolated_values),
        ):
            thread = threads[thread_id]
            assert values["viz_grade"] == "acceptable"
            assert values["viz_eval_result"].grade == "acceptable"
            assert values["visualization_complete"] is True
            assert len(values["visualization_results"].visualizations) == 2
            for item in values["visualization_results"].visualizations:
                assert Path(item.path).is_file()
                assert Path(item.path).is_relative_to(tmp_path.resolve())
            assert len(values["viz_paths"]) == 2
            assert {Path(path).resolve() for path in values["viz_paths"]} == {
                Path(item.path).resolve()
                for item in values["visualization_results"].visualizations
            }
            expected_ids = {
                spec.viz_id
                for spec in (
                    threads[thread_id]["revision_specs"]
                    if thread_id == "revision-thread-a"
                    else threads[thread_id]["initial_specs"]
                )
            }
            actual_ids = {
                item.visualization_id
                for item in values["visualization_results"].visualizations
            }
            assert actual_ids == expected_ids
            assert {
                DataVisualization.model_validate(item).visualization_id
                for item in values["viz_results"]
            } == expected_ids
            assert {
                Path(item.path).resolve()
                for item in values["visualization_results"].visualizations
            } == {
                Path(thread["plot_results"][viz_id].path).resolve()
                for viz_id in expected_ids
            }
            assert {Path(path).resolve() for path in values["viz_paths"]} == {
                Path(item.path).resolve()
                for item in values["visualization_results"].visualizations
            }

        thread_a = initial_values
        thread_b = isolated_values
        initial_ids = {
            spec.viz_id for spec in threads["revision-thread-a"]["initial_specs"]
        }
        revised_ids = {
            spec.viz_id for spec in threads["revision-thread-a"]["revision_specs"]
        }
        assert initial_ids.isdisjoint(revised_ids)
        assert {
            v.visualization_id for v in thread_a["visualization_results"].visualizations
        } == revised_ids
        assert initial_ids.isdisjoint(
            {
                v.visualization_id
                for v in thread_a["visualization_results"].visualizations
            }
        )
        assert {
            v.visualization_id for v in thread_b["visualization_results"].visualizations
        } == {spec.viz_id for spec in threads["revision-thread-b"]["initial_specs"]}
        initial_paths = {
            Path(threads["revision-thread-a"]["plot_results"][viz_id].path).resolve()
            for viz_id in initial_ids
        }
        revised_paths = {
            Path(threads["revision-thread-a"]["plot_results"][viz_id].path).resolve()
            for viz_id in revised_ids
        }
        assert initial_paths.isdisjoint(revised_paths)
        assert {Path(path).resolve() for path in thread_a["viz_paths"]} == revised_paths
        assert thread_a["analysis_insights"].summary == "Thread A initial analysis."
        assert thread_b["analysis_insights"].summary == "Thread B isolated analysis."
        initial_analysis = AnalysisInsights.model_validate(
            threads["revision-thread-a"]["analyst_model"].payloads[0]
        )
        revised_analysis = AnalysisInsights.model_validate(
            threads["revision-thread-a"]["analyst_model"].payloads[1]
        )
        assert thread_a["analysis_insights"] == revised_analysis
        for field in (
            "correlation_insights",
            "anomaly_insights",
            "recommended_next_steps",
        ):
            assert getattr(revised_analysis, field) == getattr(initial_analysis, field)
        assert thread_b["analysis_insights"] == AnalysisInsights.model_validate(
            threads["revision-thread-b"]["analyst_model"].payloads[0]
        )
        assert thread_a["viz_feedback"] == "Thread A revised figures are acceptable."
        assert thread_b["viz_feedback"] == "Thread B figures are acceptable."
        assert thread_a["viz_eval_result"].feedback == thread_a["viz_feedback"]
        assert thread_b["viz_eval_result"].feedback == thread_b["viz_feedback"]
        assert {
            VizSpec.model_validate(spec).viz_id for spec in thread_a["viz_specs"]
        } == revised_ids
        assert {
            VizSpec.model_validate(spec).viz_id for spec in thread_b["viz_specs"]
        } == {spec.viz_id for spec in threads["revision-thread-b"]["initial_specs"]}

        for thread_id, expected_analyst_calls, expected_evals, expected_workers in (
            ("revision-thread-a", 2, 2, 4),
            ("revision-thread-b", 1, 1, 2),
        ):
            runtime = threads[thread_id]
            assert len(runtime["analyst_model"]._requests) == expected_analyst_calls
            assert len(runtime["evaluator_model"]._requests) == expected_evals
            assert len(runtime["visualization_model"]._requests) == expected_workers
            assert events[thread_id].count("analyst_node") == expected_analyst_calls
            assert (
                events[thread_id].count("visualization_orchestrator")
                == expected_analyst_calls
            )
            assert events[thread_id].count("viz_worker") == expected_workers
            assert events[thread_id].count("viz_join") == expected_evals
            assert events[thread_id].count("viz_evaluator_node") == expected_evals
            assert "supervisor->analyst" in events[thread_id]
            assert "supervisor->visualization" in events[thread_id]
        revision_prompt = "\n".join(
            str(message.content)
            for message in threads["revision-thread-a"]["analyst_model"]._requests[1]
        )
        assert "Thread A evaluator: correct the axis" in revision_prompt
        assert (
            threads["revision-thread-a"]["initial_specs"][0].viz_id in revision_prompt
        )
        assert (
            threads["revision-thread-a"]["evaluator_model"].payloads[0]["grade"]
            == "revise"
        )

    with SqliteSaver.from_conn_string(str(db)) as saver:
        restored_graph = build_graph(node_overrides=overrides, checkpointer=saver)
        restored_a = restored_graph.get_state(_thread_config("revision-thread-a"))
        restored_b = restored_graph.get_state(_thread_config("revision-thread-b"))
        assert restored_a.next == restored_b.next == ("report_orchestrator",)
        assert (
            restored_a.values["viz_grade"]
            == restored_b.values["viz_grade"]
            == "acceptable"
        )
        assert {
            v.visualization_id
            for v in restored_a.values["visualization_results"].visualizations
        } == revised_ids
        assert {
            v.visualization_id
            for v in restored_b.values["visualization_results"].visualizations
        } == {spec.viz_id for spec in threads["revision-thread-b"]["initial_specs"]}
        assert {Path(path).resolve() for path in restored_a.values["viz_paths"]} == {
            Path(path).resolve() for path in initial_values["viz_paths"]
        }
        assert {
            DataVisualization.model_validate(item).visualization_id
            for item in restored_a.values["viz_results"]
        } == revised_ids
        assert (
            restored_b.values["analysis_insights"].summary
            == "Thread B isolated analysis."
        )


def test_exhausted_retries_stop_the_canonical_graph_without_a_report(tmp_path):
    items = figures(tmp_path)
    state = {
        "initial_analysis_complete": True,
        "data_cleaning_complete": True,
        "analyst_complete": True,
        "visualization_complete": False,
        "report_generator_complete": False,
        "last_agent_id": "analyst",
        "viz_grade": None,
        "viz_results": [item.model_dump() for item in items],
        "visualization_results": VisualizationResults(**FLAGS, visualizations=items),
        "viz_paths": [item.path for item in items],
        "_viz_retry_count": 4,
    }
    calls = []

    def unexpected(current):
        calls.append(current)
        return {}

    overrides = {
        "supervisor": generated_supervisor(),
        "visualization": unexpected,
        "viz_evaluator": unexpected,
        "report_orchestrator": unexpected,
    }
    cfg = {"configurable": {"thread_id": "exhausted-retries"}}
    db = tmp_path / "exhausted.sqlite"
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(node_overrides=overrides, checkpointer=saver)
        graph.update_state(cfg, state, as_node="analyst")
        with pytest.raises(RuntimeError, match=r"(?i)visualization.*retr.*exhaust"):
            graph.invoke(None, cfg)
        assert calls == []
        pending = graph.get_state(cfg)
        assert pending.next == ("supervisor",)
        assert pending.values["visualization_complete"] is False
        assert pending.values["viz_grade"] is None
        assert pending.values["_viz_retry_count"] == 4
        assert pending.values["visualization_results"].visualizations == items
        assert pending.values["viz_paths"] == [item.path for item in items]
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(node_overrides=overrides, checkpointer=saver)
        restored = graph.get_state(cfg)
        assert restored.next == ("supervisor",)
        assert restored.values["visualization_complete"] is False
        assert restored.values["viz_grade"] is None


@pytest.mark.parametrize("serialized", [False, True])
def test_envelope_only_report_dispatch_and_checkpoint(
    tmp_path, report_state, serialized
):
    from idd_state import Section

    figures = list(report_state["viz_results"])
    envelope = VisualizationResults(**FLAGS, visualizations=figures)
    report_state["visualization_results"] = (
        envelope.model_dump() if serialized else envelope
    )
    report_state["viz_results"] = []
    outline_ns, outline_model = report_agent(
        tmp_path, "outline", report_state["report_outline"]
    )
    section = report_state["report_outline"].sections[0]
    response = Section(
        **FLAGS,
        name=section.name,
        section_num=section.section_num,
        description=section.description,
        goals=section.goals,
        data_signals=[],
        expected_figures=figures,
        content="Measured channel revenue varies.",
    )
    worker_ns, worker_model = report_agent(tmp_path, "section", response)
    execute_definitions(outline_ns, 57, {"report_join"})
    overrides = {
        "report_orchestrator": outline_ns["report_orchestrator"],
        "report_section_worker": worker_ns["section_worker"],
        "report_join": outline_ns["report_join"],
    }
    cfg = {"configurable": {"thread_id": "envelope-report"}}
    report_state["_config"] = cfg
    report_state["next"] = "report_orchestrator"
    db = tmp_path / "envelope-report.sqlite"
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(node_overrides=overrides, checkpointer=saver)
        graph.update_state(cfg, report_state, as_node="supervisor")
        result = graph.invoke(None, cfg, interrupt_before=["report_packager"])
        assert len(outline_model._requests) == len(worker_model._requests) == 1
        assert result["sections"][0].expected_figures == figures
        assert result["viz_results"] == []
    with SqliteSaver.from_conn_string(str(db)) as saver:
        restored = build_graph(node_overrides=overrides, checkpointer=saver).get_state(
            cfg
        )
        assert "report_packager" in restored.next
        assert restored.values["sections"][0].expected_figures == figures
