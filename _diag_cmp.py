import json
for fn in ['executed_20260423_040749.ipynb','executed_20260423_044425.ipynb']:
    nb = json.load(open(f'IDD_results/{fn}','r',encoding='utf-8'))
    cells = nb['cells']
    print(f'{fn}: total_cells={len(cells)}')
    c48 = cells[48]
    print(f'  cell 48 exec_count={c48.get("execution_count")} outputs={len(c48.get("outputs",[]))}')
    # find cell with execution_count == 22
    for i,c in enumerate(cells):
        if c.get('execution_count') == 22:
            print(f'  exec_count=22 -> cell index {i}')
            src = ''.join(c.get("source",[]))
            print(f'    source head: {src[:80]!r}')
            break
