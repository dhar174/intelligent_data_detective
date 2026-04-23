import json, sys
sys.stdout.reconfigure(encoding="utf-8")
nb = json.load(open(r"C:\Users\darf3\Documents\intelligent_data_detective\IntelligentDataDetective_beta_v5.ipynb", encoding="utf-8"))
src = "".join(nb["cells"][57].get("source", []))
lines = src.split("\n")
# Show context around line 566
for target in [560, 956]:
    print(f"\n=== Around line {target} ===")
    for j in range(target-5, min(len(lines), target+25)):
        print(f"{j}: {lines[j]}")
