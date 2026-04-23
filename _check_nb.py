import json
fname = r'IDD_results\executed_20260422_195018.ipynb'
with open(fname,'r',encoding='utf-8') as f:
    nb = json.load(f)
errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'error':
            errors.append((i, out.get('ename',''), out.get('evalue','')[:300]))
print(f'Total cells: {len(nb["cells"])}')
print(f'Errors: {len(errors)}')
for e in errors[:10]:
    print(f'  Cell {e[0]}: {e[1]}: {e[2]}')
