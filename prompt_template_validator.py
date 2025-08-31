#!/usr/bin/env python3
"""
Prompt Template Formatter Validation Tool

This script validates ChatPromptTemplate instances in the IntelligentDataDetective notebook
for correct formatting, proper variable substitution, and string construction.
"""

import json
import re
import ast
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path


class PromptTemplateValidator:
    """Validates prompt template formatting and structure."""
    
    def __init__(self, notebook_path: str):
        self.notebook_path = Path(notebook_path)
        self.issues = []
        
    def load_notebook(self) -> Dict[str, Any]:
        """Load the Jupyter notebook JSON."""
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_prompt_templates(self, notebook: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all ChatPromptTemplate instances from the notebook."""
        templates = []
        
        for i, cell in enumerate(notebook.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                
                # Find ChatPromptTemplate.from_messages calls
                template_matches = re.finditer(
                    r'(\w+)\s*=\s*ChatPromptTemplate\.from_messages\(',
                    source
                )
                
                for match in template_matches:
                    template_name = match.group(1)
                    start_pos = match.start()
                    
                    # Extract the full template definition
                    template_def = self._extract_template_definition(source, start_pos)
                    
                    templates.append({
                        'name': template_name,
                        'cell_index': i,
                        'definition': template_def,
                        'start_line': source[:start_pos].count('\n') + 1
                    })
        
        return templates
    
    def _extract_template_definition(self, source: str, start_pos: int) -> str:
        """Extract the complete template definition from source code."""
        # Find the matching closing parenthesis/bracket
        paren_count = 0
        bracket_count = 0
        in_string = False
        string_char = None
        escaped = False
        
        i = start_pos
        while i < len(source):
            char = source[i]
            
            if escaped:
                escaped = False
                i += 1
                continue
                
            if char == '\\':
                escaped = True
                i += 1
                continue
                
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        # Check if this is followed by .partial(
                        remaining = source[i+1:]
                        partial_match = re.match(r'\s*\.partial\s*\(', remaining)
                        if partial_match:
                            # Continue to include the partial call
                            i += len(partial_match.group(0))
                            paren_count = 1
                        else:
                            return source[start_pos:i+1]
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
            else:
                if char == string_char and not escaped:
                    in_string = False
                    string_char = None
                    
            i += 1
        
        return source[start_pos:]
    
    def validate_template(self, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate a single template for formatting issues."""
        issues = []
        definition = template['definition']
        name = template['name']
        
        # Check for basic syntax issues
        issues.extend(self._check_syntax_issues(template))
        
        # Check for unmatched braces
        issues.extend(self._check_unmatched_braces(template))
        
        # Check for quote consistency
        issues.extend(self._check_quote_consistency(template))
        
        # Check placeholder consistency
        issues.extend(self._check_placeholder_consistency(template))
        
        # Check MessagesPlaceholder configuration
        issues.extend(self._check_messages_placeholder(template))
        
        return issues
    
    def _check_syntax_issues(self, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for basic Python syntax issues."""
        issues = []
        
        try:
            # Try to parse as Python code (this won't execute, just parse)
            ast.parse(template['definition'])
        except SyntaxError as e:
            issues.append({
                'type': 'syntax_error',
                'template': template['name'],
                'line': e.lineno,
                'message': f"Syntax error: {e.msg}",
                'severity': 'error'
            })
        
        return issues
    
    def _check_unmatched_braces(self, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for unmatched braces in template strings."""
        issues = []
        definition = template['definition']
        
        # Extract string literals from the template
        string_literals = re.findall(r'["\']([^"\']*(?:\\.[^"\']*)*)["\']', definition)
        
        for i, literal in enumerate(string_literals):
            # Count braces
            open_braces = literal.count('{')
            close_braces = literal.count('}')
            
            # Account for escaped braces
            escaped_open = literal.count('{{')
            escaped_close = literal.count('}}')
            
            effective_open = open_braces - (escaped_open * 2)
            effective_close = close_braces - (escaped_close * 2)
            
            if effective_open != effective_close:
                issues.append({
                    'type': 'unmatched_braces',
                    'template': template['name'],
                    'message': f"Unmatched braces in string literal {i+1}: {open_braces} open, {close_braces} close",
                    'severity': 'error',
                    'literal_preview': literal[:100] + ('...' if len(literal) > 100 else '')
                })
        
        return issues
    
    def _check_quote_consistency(self, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for quote consistency issues."""
        issues = []
        definition = template['definition']
        
        # Look for mixed quotes in f-strings
        f_string_pattern = r'f["\']([^"\']*(?:\\.[^"\']*)*)["\']'
        f_strings = re.findall(f_string_pattern, definition)
        
        for i, f_string in enumerate(f_strings):
            # Check for quote conflicts within f-strings
            if "'" in f_string and '"' in f_string:
                # This could be problematic - check if quotes are properly escaped
                unescaped_quotes = re.findall(r'(?<!\\)["\']', f_string)
                if len(set(unescaped_quotes)) > 1:
                    issues.append({
                        'type': 'quote_conflict',
                        'template': template['name'],
                        'message': f"Mixed unescaped quotes in f-string {i+1}",
                        'severity': 'warning',
                        'f_string_preview': f_string[:100] + ('...' if len(f_string) > 100 else '')
                    })
        
        return issues
    
    def _check_placeholder_consistency(self, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check that placeholders in templates have consistent naming."""
        issues = []
        definition = template['definition']
        
        # Check for double brace issues first
        double_brace_matches = re.findall(r'\{\{([^}]+)\}\}', definition)
        if double_brace_matches:
            issues.append({
                'type': 'double_brace_placeholders',
                'template': template['name'],
                'message': f"Found double braces (should be single): {', '.join(double_brace_matches)}",
                'severity': 'error',
                'variables': double_brace_matches
            })
        
        # Extract placeholders from template strings
        placeholders = set()
        string_literals = re.findall(r'["\']([^"\']*(?:\\.[^"\']*)*)["\']', definition)
        
        for literal in string_literals:
            # Find all {variable} patterns (single braces)
            placeholder_matches = re.findall(r'(?<!\{)\{([^}]+)\}(?!\})', literal)
            for match in placeholder_matches:
                # Skip format specifiers like {variable:.2f}
                variable_name = match.split(':')[0].split('.')[0]
                if variable_name and not variable_name.isdigit():
                    placeholders.add(variable_name)
        
        # Check if .partial() call exists and extract its parameters
        partial_match = re.search(r'\.partial\s*\(([^)]+)\)', definition)
        partial_vars = set()
        
        if partial_match:
            partial_content = partial_match.group(1)
            # Extract variable names from keyword arguments
            param_matches = re.findall(r'(\w+)\s*=', partial_content)
            partial_vars.update(param_matches)
        
        # Find undeclared placeholders (not in partial)
        undeclared = placeholders - partial_vars
        
        # Common template variables that might be passed at runtime
        runtime_vars = {
            'messages', 'user_prompt', 'available_df_ids', 'tool_descriptions',
            'output_format', 'dataset_description', 'data_sample', 'memories',
            'cleaned_dataset_description', 'cleaning_metadata', 'analysis_insights',
            'visualization_results', 'analysis_config', 'tooling_guidelines',
            'file_name', 'file_type', 'content', 'visualization_task', 'report_task'
        }
        
        # Filter out known runtime variables
        truly_undeclared = undeclared - runtime_vars
        
        if truly_undeclared:
            issues.append({
                'type': 'undeclared_placeholders',
                'template': template['name'],
                'message': f"Placeholders without default values: {', '.join(sorted(truly_undeclared))}",
                'severity': 'warning',
                'placeholders': list(truly_undeclared)
            })
        
        return issues
    
    def _check_messages_placeholder(self, template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check MessagesPlaceholder configuration."""
        issues = []
        definition = template['definition']
        
        # Look for MessagesPlaceholder usage
        placeholder_pattern = r'MessagesPlaceholder\s*\(\s*["\']([^"\']+)["\'](?:\s*,\s*optional\s*=\s*(True|False))?\s*\)'
        matches = re.findall(placeholder_pattern, definition)
        
        for match in matches:
            var_name = match[0]
            optional = match[1] if len(match) > 1 else None
            
            # Check naming convention
            if var_name != 'messages':
                issues.append({
                    'type': 'non_standard_messages_placeholder',
                    'template': template['name'],
                    'message': f"Non-standard MessagesPlaceholder variable name: '{var_name}' (standard is 'messages')",
                    'severity': 'info',
                    'variable_name': var_name
                })
        
        return issues
    
    def validate_all_templates(self) -> Dict[str, Any]:
        """Validate all templates and return summary."""
        notebook = self.load_notebook()
        templates = self.extract_prompt_templates(notebook)
        
        all_issues = []
        
        for template in templates:
            template_issues = self.validate_template(template)
            all_issues.extend(template_issues)
        
        # Categorize issues by severity
        errors = [issue for issue in all_issues if issue['severity'] == 'error']
        warnings = [issue for issue in all_issues if issue['severity'] == 'warning']
        info = [issue for issue in all_issues if issue['severity'] == 'info']
        
        return {
            'total_templates': len(templates),
            'template_names': [t['name'] for t in templates],
            'total_issues': len(all_issues),
            'errors': errors,
            'warnings': warnings,
            'info': info,
            'summary': {
                'error_count': len(errors),
                'warning_count': len(warnings),
                'info_count': len(info)
            }
        }
    
    def print_report(self, validation_result: Dict[str, Any]):
        """Print a formatted validation report."""
        print("=" * 80)
        print("PROMPT TEMPLATE VALIDATION REPORT")
        print("=" * 80)
        print(f"Total templates found: {validation_result['total_templates']}")
        print(f"Template names: {', '.join(validation_result['template_names'])}")
        print()
        
        summary = validation_result['summary']
        print(f"Issues found: {validation_result['total_issues']}")
        print(f"  - Errors: {summary['error_count']}")
        print(f"  - Warnings: {summary['warning_count']}")
        print(f"  - Info: {summary['info_count']}")
        print()
        
        # Print detailed issues
        for severity in ['errors', 'warnings', 'info']:
            issues = validation_result[severity]
            if issues:
                print(f"{severity.upper()}:")
                print("-" * 40)
                for issue in issues:
                    print(f"  Template: {issue['template']}")
                    print(f"  Type: {issue['type']}")
                    print(f"  Message: {issue['message']}")
                    if 'line' in issue:
                        print(f"  Line: {issue['line']}")
                    print()


def main():
    """Main validation function."""
    notebook_path = "/home/runner/work/intelligent_data_detective/intelligent_data_detective/IntelligentDataDetective_beta_v4.ipynb"
    
    validator = PromptTemplateValidator(notebook_path)
    result = validator.validate_all_templates()
    validator.print_report(result)
    
    return result


if __name__ == "__main__":
    result = main()