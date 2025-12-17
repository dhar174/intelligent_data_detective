import json
import os

file_path = "IntelligentDataDetective_beta_v5.ipynb"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"Analysis of {file_path}")
    print(f"Total Cells: {len(nb.get('cells', []))}")
    
    for i, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get('cell_type', 'unknown')
        source_lines = cell.get("source", [])
        source = "".join(source_lines)
        
        print(f"--- Cell {i} [{cell_type}] ---")
        if cell_type == 'code':
            # Check for functions/classes
            lines = source.split('\n')
            for line in lines:
                if line.strip().startswith('class ') or line.strip().startswith('def '):
                    print(f"DEFINITION: {line.strip()}")
        
        print(source)
        print("\n" + "="*40 + "\n")

except Exception as e:
    print(f"Error reading notebook: {e}")
