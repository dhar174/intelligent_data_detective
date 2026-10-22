"""Selected production graph path, SQLite, and unchanged W14J renderer, no keys.

Start at the worker checkpoint and stop at the supervisor boundary; inject the
controlled supervisor decision as a checkpoint update, then use canonical report
routing/dispatch/reducers. Unselected nodes fail if accidentally executed.
"""

from pathlib import Path
import sqlite3
from types import SimpleNamespace

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import pytest

from idd_graph import build_graph
from idd_state import (
    AnalysisInsights,
    CleaningMetadata,
    DataVisualization,
    ReportOutline,
    SectionOutline,
)
from tests.unit.test_viz_join import (
    FLAGS,
    figures,
    load_functions,
    notebook,
    state_for,
)
from tests.unit.test_viz_join import no_network  # noqa: F401 - autouse fixture
from tests.unit.test_section_worker import load_worker
from tests.unit.test_patcher_integrity import _compile_report_packager_node


def run_selected_graph(base):
    items = figures(base)
    state = state_for(items)
    state["visualization_results"] = state["visualization_results"].model_dump(
        mode="json"
    )
    config = {"configurable": {"thread_id": "viz-report-offline"}}
    state.update(
        _config=config,
        available_df_ids=["retail_orders"],
        user_prompt="Analyze controlled retail observations.",
        messages=[AIMessage(content="Four charts created.")],
        final_turn_msgs_list=[],
        cleaning_metadata=CleaningMetadata(
            **FLAGS,
            steps_taken=["Removed duplicates"],
            data_description_after_cleaning="180 controlled retail rows.",
        ),
        analysis_insights=AnalysisInsights(
            **FLAGS,
            summary="Online revenue leads.",
            correlation_insights="Discount and revenue: r=-0.1607.",
            anomaly_insights="High discounts warrant review.",
            recommended_visualizations=[],
            recommended_next_steps=["Review discounts"],
        ),
    )
    ns = load_functions(
        base,
        (
            "viz_join",
            "viz_evaluator_node",
            "report_orchestrator",
            "_safe_report_orchestrator_invoke",
            "report_join",
        ),
    )
    observed = []

    def outline_invoke(payload, config):
        actual = [DataVisualization.model_validate(v) for v in payload["viz_results"]]
        assert actual == items
        observed.append(actual)
        sections = [
            SectionOutline(
                **FLAGS,
                section_num=i + 1,
                name=f"Retail finding {i+1}",
                description="Explain measured retail performance.",
                goals=["Assess retail drivers"],
                data_signals_needed={"revenue": "Controlled revenue"},
                data_signals_available=["retail_orders"],
                expected_figures=[actual[i % 4]],
                word_target=100,
            )
            for i in range(6)
        ]
        outline = ReportOutline(
            **sections[0].model_dump(), title="OFFLINE TEST FIXTURE", sections=sections
        )
        return {
            "structured_response": outline,
            "messages": [AIMessage(content="Controlled six-section outline.")],
        }

    ns.update(
        ReportOutline=ReportOutline,
        CleaningMetadata=CleaningMetadata,
        ChatPromptTemplate=ChatPromptTemplate,
        report_generator_tools=[],
        DEFAULT_TOOLING_GUIDELINES="Use registered DataFrame IDs.",
        report_generator_prompt_template=ChatPromptTemplate.from_messages(
            [("system", "{report_task}\nFigures: {viz_results}")]
        ),
        enhanced_retrieve_mem=lambda state: "",
        report_generator_agent=SimpleNamespace(invoke=outline_invoke),
    )
    worker, worker_calls = load_worker(notebook(), base)
    overrides = {
        "viz_join": ns["viz_join"],
        "viz_evaluator": ns["viz_evaluator_node"],
        "report_orchestrator": ns["report_orchestrator"],
        "report_section_worker": worker,
        "report_join": ns["report_join"],
    }
    db = base / "checkpoint.sqlite"
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(node_overrides=overrides, checkpointer=saver)
        graph.update_state(config, state, as_node="viz_worker")
        seeded = graph.get_state(config).values
        assert isinstance(seeded["visualization_results"], dict)
        assert isinstance(seeded["visualization_results"]["visualizations"][0], dict)
        result = graph.invoke(None, config, interrupt_before=["supervisor"])
        assert result["visualization_results"].visualizations == items
        assert len(result["viz_results"]) == 4
        assert result["viz_grade"] == "not_evaluated"
        assert graph.get_state(config).next == ("supervisor",)
        # Decode the actual SQLite typed checkpoint, rather than assuming JSON.
        with sqlite3.connect(db) as conn:
            typed = conn.execute(
                "SELECT type, checkpoint FROM checkpoints ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        decoded = saver.serde.loads_typed(typed)
        inventory = decoded["channel_values"]["visualization_results"]
        assert inventory.visualizations == items
    # Reopen the DB and graph: process-boundary restoration must preserve metadata.
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = build_graph(node_overrides=overrides, checkpointer=saver)
        restored = graph.get_state(config).values
        assert restored["visualization_results"].visualizations == items
        # Controlled supervisor routing decision; its LLM/planning loop is outside
        # this selected-cell test. Canonical route_from_supervisor executes next.
        graph.update_state(
            config, {"next": "report_orchestrator"}, as_node="supervisor"
        )
        result = graph.invoke(None, config, interrupt_before=["report_packager"])
        assert set(graph.get_state(config).next) == {"report_join", "report_packager"}
        assert len(observed) == 1
        assert len(worker_calls) == 6
        assert len(result["sections"]) == 6
        assert len(result["written_sections"]) == 6
        assert result["visualization_results"].visualizations == items
        assert [
            DataVisualization.model_validate(v) for v in result["viz_results"]
        ] == items
        assert {
            f.visualization_id for s in result["sections"] for f in s.expected_figures
        } == {v.visualization_id for v in items}
    return result, items


def test_sqlite_graph_inventory_reaches_orchestrator_sections_and_w14j(
    tmp_path, monkeypatch
):
    state, items = run_selected_graph(tmp_path)
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(tmp_path))
    packager = _compile_report_packager_node(notebook(), tmp_path)
    update = packager(state)
    assert update["report_generator_complete"] is True
    for suffix in ("md", "html", "pdf"):
        assert (tmp_path / "reports" / f"final_report.{suffix}").stat().st_size > 0
    md = (tmp_path / "reports" / "final_report.md").read_text(encoding="utf-8")
    html = (tmp_path / "reports" / "final_report.html").read_text(encoding="utf-8")
    for item in items:
        assert item.visualization_title in md
        assert item.visualization_title in html
        assert f"../visualizations/{Path(item.path).name}" in md


@pytest.mark.parametrize(
    "control",
    ["invented_id", "missing_file", "external_file", "stale_pngs", "two_figures"],
)
def test_original_w14j_rejects_unresolved_or_manufactured_inventory(
    tmp_path, monkeypatch, control
):
    from pathlib import Path
    from idd_state import Section

    items = figures(tmp_path / "current")
    root = tmp_path / "current"
    monkeypatch.setenv("IDD_ARTIFACTS_DIR", str(root))
    if control == "invented_id":
        items[0] = items[0].model_copy(
            update={"path": "", "visualization_id": "invented-never-created"}
        )
    elif control == "missing_file":
        Path(items[0].path).unlink()
    elif control == "external_file":
        items[0] = items[0].model_copy(
            update={"path": figures(tmp_path / "unrelated")[0].path}
        )
    elif control == "stale_pngs":
        figures(root / "historical")
        items = []
    elif control == "two_figures":
        figures(root / "historical")
        items = items[:2]
    section = Section(
        **FLAGS,
        name="Offline control",
        section_num=1,
        description="Negative control",
        goals=["Verify integrity"],
        data_signals=[],
        expected_figures=items,
        content="## Offline control\n\nControlled test observations.",
    )
    state = state_for(items)
    state.update(
        sections=[section],
        written_sections=[section.content],
        messages=[AIMessage(content="Package controlled fixture.")],
        final_turn_msgs_list=[],
        available_df_ids=[],
    )
    packager = _compile_report_packager_node(notebook(), root)
    with pytest.raises(
        RuntimeError,
        match="W14J canonical report (has unresolved expected figures|requires at least 3 distinct figures)",
    ):
        packager(state)
    assert not (root / "reports" / "final_report.pdf").exists()
