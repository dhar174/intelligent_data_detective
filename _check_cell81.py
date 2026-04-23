import json
fname = r'IDD_results\executed_20260422_195018.ipynb'
with open(fname,'r',encoding='utf-8') as f:
    nb = json.load(f)
cell = nb['cells'][81]
print("=== Cell 81 source ===")
print(''.join(cell.get('source',[])))
print()
print("=== Outputs ===")
for out in cell.get('outputs', []):
    if out.get('output_type') == 'error':
        import traceback
        print(f"ERROR: {out.get('ename')}: {out.get('evalue')}")
        tb = out.get('traceback', [])
        for line in tb[-10:]:
            # strip ANSI escapes
            import re
            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            print(line)
