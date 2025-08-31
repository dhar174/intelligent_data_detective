#!/usr/bin/env python3
"""
Fix double brace issues in ChatPromptTemplate instances.
"""

import json
import re
from pathlib import Path


def fix_double_braces_in_notebook(notebook_path: str) -> None:
    """Fix double brace placeholder issues in the notebook."""
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = False
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source_lines = cell.get('source', [])
            
            # Process each line
            for i, line in enumerate(source_lines):
                original_line = line
                
                # Replace double braces with single braces, but only in template strings
                # Look for lines that contain template variable patterns
                if '{{' in line and '}}' in line:
                    # Check if this is inside a template string context
                    # Exclude f-strings which legitimately use double braces
                    if not line.strip().startswith('f"""') and 'f"' not in line:
                        # Replace double braces with single braces for template variables
                        # Pattern: {{variable_name}} -> {variable_name}
                        line = re.sub(r'\{\{([^}]+)\}\}', r'{\1}', line)
                        
                        if line != original_line:
                            source_lines[i] = line
                            changes_made = True
                            print(f"Fixed line: {original_line.strip()} -> {line.strip()}")
    
    if changes_made:
        # Save the updated notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print(f"Updated notebook saved to {notebook_path}")
    else:
        print("No changes needed")


def main():
    """Main function."""
    notebook_path = "/home/runner/work/intelligent_data_detective/intelligent_data_detective/IntelligentDataDetective_beta_v4.ipynb"
    fix_double_braces_in_notebook(notebook_path)


if __name__ == "__main__":
    main()