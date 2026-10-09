"""Probe notebook cells for symbol definitions and locations."""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def inspect_cells():
    nb_path = Path("IntelligentDataDetective_beta_v5_patched.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Total cells in {nb_path.name}: {len(nb['cells'])}")
    for idx, cell in enumerate(nb["cells"]):
        cell_type = cell.get("cell_type", "")
        cell_id = cell.get("id", f"idx_{idx}")
        src = "".join(cell.get("source", []))
        lines = src.splitlines()

        # Check for key definitions
        targets = [
            "class BaseNoExtrasModel",
            "class State",
            "class Plan",
            "class PlanStep",
            "class DataFrameRegistry",
            "def validate_dataframe_exists",
            "def handle_tool_errors",
            "def _tool_error",
            "def _reduce_plan_keep_sorted",
            "def keep_first",
            "def _resolve_artifact_path",
            "data_analysis_team_builder = StateGraph",
            "supervisor_node",
            "initial_analysis_node",
            "data_cleaner_node",
            "analyst_node",
            "viz_worker_node",
            "viz_join_node",
            "viz_evaluator_node",
            "report_orchestrator_node",
            "report_section_worker_node",
            "report_join_node",
            "report_packager_node",
            "file_writer_node",
            "emergency_correspondence_node",
            "delete_rows",
            "_build_query_view",
        ]
        found = [t for t in targets if any(t in line for line in lines)]
        if found:
            print(f"Cell idx={idx} id={cell_id} type={cell_type} lines={len(lines)}:")
            for t in found:
                for line_no, l in enumerate(lines, 1):
                    if t in l:
                        print(f"   L{line_no}: {l.strip()[:100]}")


if __name__ == "__main__":
    inspect_cells()
