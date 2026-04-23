import nbformat

nb = nbformat.read('IDD_results/executed_20260422_020616.ipynb', as_version=4)

# Find the final state summary cell output
for i, cell in enumerate(nb.cells):
    outputs = cell.get('outputs', [])
    for out in outputs:
        text = ''
        if out.get('output_type') in ('stream', 'execute_result'):
            text_data = out.get('text', out.get('data', {}).get('text/plain', ''))
            if isinstance(text_data, list):
                text_data = ''.join(text_data)
            text = text_data
        if 'visualization_complete' in text or 'report_generator_complete' in text or 'Final state summary' in text:
            print(f"=== Cell {i} ===")
            print(text[:2000])
            print()
