"""Helper to inspect notebook cells 48 and 81."""
import json

nb = json.load(open("IntelligentDataDetective_beta_v5.ipynb", "r", encoding="utf-8"))
cells = nb["cells"]
print(f"Total cells: {len(cells)}")

for idx, label in [(47, "CELL 48"), (80, "CELL 81")]:
    c = cells[idx]
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    print(f"\n{'='*60}")
    print(f"{label} (cell_type={c['cell_type']}):")
    print(src[:800])
