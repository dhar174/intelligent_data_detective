
import json

path = "IntelligentDataDetective_beta_v5.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    source = cell.get("source", [])
    for j, line in enumerate(source):
        if "AgentId =" in line or "AgentId=" in line:
            print(f"Found in Cell {i}, Line {j}: {line.strip()}")
        if "class Router" in line:
            print(f"Router found in Cell {i}, Line {j}: {line.strip()}")
