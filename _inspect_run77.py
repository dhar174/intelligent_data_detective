import json, io, sys
nb_path = sys.argv[1] if len(sys.argv) > 1 else 'IDD_results/executed_20260423_022907.ipynb'
nb = json.load(io.open(nb_path, encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
err_cells = []
for i, c in enumerate(nb['cells']):
    if c.get('cell_type') != 'code':
        continue
    for o in c.get('outputs', []):
        if o.get('output_type') == 'error':
            err_cells.append((i, o.get('ename'), '\n'.join(o.get('traceback', []))[-1500:]))
print(f"Cells with errors: {len(err_cells)}")
for i, ename, tb in err_cells[:6]:
    src = ''.join(nb['cells'][i].get('source', []))
    print(f"\n=== Cell #{i}: {ename} ===")
    print(f"source[:300]: {src[:300]}")
    print("--- traceback (last 1500 chars) ---")
    # strip ANSI
    import re
    tb_clean = re.sub(r'\x1b\[[0-9;]*m', '', tb)
    print(tb_clean)
