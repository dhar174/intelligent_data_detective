import json, io, re
nb = json.load(io.open('IDD_results/executed_20260423_024830.ipynb', encoding='utf-8'))
ansi = re.compile(r'\x1b\[[0-9;]*m')
# Find traceback blocks
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
        if not text or 'Traceback' not in text:
            continue
        text = ansi.sub('', text)
        # split into traceback chunks
        chunks = []
        cur = []
        for line in text.splitlines():
            if 'Traceback (most recent call last):' in line:
                if cur: chunks.append(cur)
                cur = [line]
            elif cur:
                cur.append(line)
                if re.match(r'^\w+(?:Error|Exception):', line):
                    chunks.append(cur)
                    cur = []
        if cur: chunks.append(cur)
        for ck in chunks:
            print(f"\n--- cell {i} traceback ---")
            for line in ck[-12:]:
                print(f"  {line[:200]}")
