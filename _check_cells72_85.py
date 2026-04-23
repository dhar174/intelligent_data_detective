import json, re

fname = r'IDD_results\executed_20260422_195018.ipynb'
with open(fname,'r',encoding='utf-8') as f:
    nb = json.load(f)

def strip_ansi(s):
    return re.sub(r'\x1b\[[0-9;]*m', '', s)

# Look at cells 72-85 for pipeline execution and aftermath
for i in range(72, min(86, len(nb['cells']))):
    cell = nb['cells'][i]
    if cell['cell_type'] != 'code':
        continue
    outputs = cell.get('outputs', [])
    if not outputs:
        print(f"Cell {i}: [no output]")
        continue
    print(f"\n=== Cell {i} ===")
    for out in outputs:
        if out.get('output_type') == 'error':
            print(f"  ERROR: {out.get('ename')}: {out.get('evalue')[:200]}")
            # Last 5 lines of traceback
            tb = out.get('traceback', [])
            for line in tb[-5:]:
                print(f"  {strip_ansi(line)}")
        elif out.get('output_type') == 'stream':
            text = ''.join(out.get('text', []))
            # Only show last 500 chars
            if text.strip():
                print(f"  stdout: {text[-500:]}")
        elif out.get('output_type') in ('display_data', 'execute_result'):
            data = out.get('data', {})
            text = ''.join(data.get('text/plain', []))
            if text.strip():
                print(f"  result: {text[:200]}")
