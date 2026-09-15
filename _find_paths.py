import json, sys
sys.stdout.reconfigure(encoding="utf-8")

# Search original notebook
nb = json.load(open(r"C:\Users\darf3\Documents\intelligent_data_detective\IntelligentDataDetective_beta_v5.ipynb", encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "viz_paths" in src or "report_paths" in src:
        lines = src.split("\n")
        for j, line in enumerate(lines):
            if "viz_paths" in line or "report_paths" in line:
                print(f"Cell {i}, line {j}: {line[:120]}")
