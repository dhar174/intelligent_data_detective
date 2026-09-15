import json, io, sys, re
nb = json.load(io.open('IDD_results/executed_20260423_024830.ipynb', encoding='utf-8'))
# Find the test/run cell — usually has 'graph.invoke' or 'supervisor' streaming output
markers = ['SHORTCUT', 'STAGE', 'FINAL', 'STRUCT', 'RECOVERY', 'W4-SUPLIMIT', 'GraphRecursion',
           'Traceback', 'NameError', 'KeyError', 'AttributeError', 'final_report', 'report_orchestrator',
           'forrtl', 'window-CLOSE']
hits = {m: 0 for m in markers}
last_lines = []
for i, c in enumerate(nb['cells']):
    if c.get('cell_type') != 'code':
        continue
    for o in c.get('outputs', []):
        text = ''
        if o.get('output_type') == 'stream':
            text = o.get('text', '')
            if isinstance(text, list):
                text = ''.join(text)
        elif o.get('output_type') in ('display_data', 'execute_result'):
            d = o.get('data', {})
            text = d.get('text/plain', '')
            if isinstance(text, list):
                text = ''.join(text)
        elif o.get('output_type') == 'error':
            text = '\n'.join(o.get('traceback', []))
        if not text:
            continue
        for m in markers:
            hits[m] += text.count(m)
        # collect lines that look like routing/shortcut for the LAST executed cell
        for line in text.splitlines():
            if any(s in line for s in ['SHORTCUT', 'STAGE', 'FINAL', 'STRUCT', 'W4-SUPLIMIT', 'next=', 'goto', 'FINISH', 'report_', 'viz_evaluator', 'RECOVERY']):
                last_lines.append((i, line.strip()))
print("=== Marker hit counts ===")
for m, n in hits.items():
    if n: print(f"  {m}: {n}")
print(f"\n=== Last 80 routing/shortcut lines ===")
# Strip ANSI
ansi = re.compile(r'\x1b\[[0-9;]*m')
seen = set()
out = []
for i, ln in last_lines[-200:]:
    cln = ansi.sub('', ln)[:240]
    key = (i, cln)
    if key in seen: continue
    seen.add(key)
    out.append(f"  [c{i}] {cln}")
for l in out[-80:]:
    print(l)
