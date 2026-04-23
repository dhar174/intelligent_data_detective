import json, sys
sys.stdout.reconfigure(encoding="utf-8")
nb = json.load(open(r"C:\Users\darf3\Documents\intelligent_data_detective\IntelligentDataDetective_beta_v5.ipynb", encoding="utf-8"))
src = "".join(nb["cells"][57].get("source", []))
lines = src.split("\n")
# Show context around lines 540-575 to understand node name
print("=== Lines 530-580 ===")
for j in range(530, 582):
    print(f"{j}: {lines[j]}")
print("\n=== Lines 940-970 ===")
for j in range(940, 972):
    print(f"{j}: {lines[j]}")
