import nbformat

nb = nbformat.read('IDD_results/executed_20260422_020616.ipynb', as_version=4)

# Check what the last few cells output
for i in range(max(0, len(nb.cells)-10), len(nb.cells)):
    cell = nb.cells[i]
    src = cell.get('source', '')
    if isinstance(src, list): src = ''.join(src)
    outputs = cell.get('outputs', [])
    if outputs:
        print(f"=== Cell {i} src: {src[:80].strip()} ===")
        for out in outputs:
            text = ''
            if out.get('output_type') in ('stream', 'execute_result', 'error'):
                if out.get('output_type') == 'error':
                    text = out.get('ename', '') + ': ' + out.get('evalue', '')[:200]
                else:
                    text_data = out.get('text', out.get('data', {}).get('text/plain', ''))
                    if isinstance(text_data, list): text_data = ''.join(text_data)
                    text = text_data
            if text:
                print(text[:500])
        print()
