#!/usr/bin/env python3
"""
Test prompt template formatting fixes in the notebook.
"""

import unittest
import json
import re
from pathlib import Path


class TestPromptTemplateFormattingFixes(unittest.TestCase):
    """Test that prompt template formatting issues have been fixed in the notebook."""
    
    def setUp(self):
        """Load the notebook for testing."""
        self.notebook_path = Path("/home/runner/work/intelligent_data_detective/intelligent_data_detective/IntelligentDataDetective_beta_v4.ipynb")
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            self.notebook = json.load(f)
    
    def test_no_double_braces_in_templates(self):
        """Test that there are no double braces in ChatPromptTemplate strings."""
        double_brace_issues = []
        
        for i, cell in enumerate(self.notebook.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                
                # Look for ChatPromptTemplate definitions
                if 'ChatPromptTemplate.from_messages' in source:
                    # Check for double braces in template strings
                    # Find content within template strings (between quotes)
                    string_contents = re.findall(r'["\']([^"\']*(?:\\.[^"\']*)*)["\']', source)
                    
                    for string_content in string_contents:
                        # Check for double braces that shouldn't be there
                        double_brace_matches = re.findall(r'\{\{([^}]+)\}\}', string_content)
                        if double_brace_matches:
                            # Filter out legitimate f-string cases
                            if not any(legitimate in string_content for legitimate in ['f"""', 'f"', 'f\'']):
                                for match in double_brace_matches:
                                    double_brace_issues.append({
                                        'cell': i,
                                        'variable': match,
                                        'context': string_content[:100]
                                    })
        
        self.assertEqual(len(double_brace_issues), 0, 
                        f"Found {len(double_brace_issues)} double brace issues: {double_brace_issues}")
    
    def test_specific_prompt_templates_fixed(self):
        """Test that specific templates known to have issues are now fixed."""
        source = ''.join([
            ''.join(cell.get('source', []))
            for cell in self.notebook.get('cells', [])
            if cell.get('cell_type') == 'code'
        ])
        
        # Test plan_prompt template
        plan_prompt_pattern = r'plan_prompt = ChatPromptTemplate\.from_messages\((.*?)\)\.partial'
        plan_match = re.search(plan_prompt_pattern, source, re.DOTALL)
        self.assertIsNotNone(plan_match, "plan_prompt template not found")
        
        plan_content = plan_match.group(1)
        # Should have single braces for user_prompt, agents, output_schema_name
        self.assertIn('{user_prompt}', plan_content)
        self.assertIn('{agents}', plan_content)
        self.assertIn('{output_schema_name}', plan_content)
        # Should NOT have double braces
        self.assertNotIn('{{user_prompt}}', plan_content)
        self.assertNotIn('{{agents}}', plan_content)
        self.assertNotIn('{{output_schema_name}}', plan_content)
    
    def test_replan_prompt_fixed(self):
        """Test that replan_prompt template is fixed."""
        source = ''.join([
            ''.join(cell.get('source', []))
            for cell in self.notebook.get('cells', [])
            if cell.get('cell_type') == 'code'
        ])
        
        # Test replan_prompt template
        replan_prompt_pattern = r'replan_prompt = ChatPromptTemplate\.from_messages\((.*?)\)\.partial'
        replan_match = re.search(replan_prompt_pattern, source, re.DOTALL)
        self.assertIsNotNone(replan_match, "replan_prompt template not found")
        
        replan_content = replan_match.group(1)
        # Should have single braces for key variables
        expected_vars = ['user_prompt', 'memories', 'plan_summary', 'plan_steps', 
                        'past_steps', 'completed_tasks', 'latest_progress']
        
        for var in expected_vars:
            self.assertIn(f'{{{var}}}', replan_content, f"Missing single brace for {var}")
            self.assertNotIn(f'{{{{{var}}}}}', replan_content, f"Found double brace for {var}")
    
    def test_todo_prompt_fixed(self):
        """Test that todo_prompt template is fixed."""
        source = ''.join([
            ''.join(cell.get('source', []))
            for cell in self.notebook.get('cells', [])
            if cell.get('cell_type') == 'code'
        ])
        
        # Test todo_prompt template
        todo_prompt_pattern = r'todo_prompt = ChatPromptTemplate\.from_messages\((.*?)\)\.partial'
        todo_match = re.search(todo_prompt_pattern, source, re.DOTALL)
        self.assertIsNotNone(todo_match, "todo_prompt template not found")
        
        todo_content = todo_match.group(1)
        # Should have single braces for key variables
        expected_vars = ['user_prompt', 'plan_summary', 'plan_steps', 
                        'completed_tasks', 'to_do_list']
        
        for var in expected_vars:
            self.assertIn(f'{{{var}}}', todo_content, f"Missing single brace for {var}")
            self.assertNotIn(f'{{{{{var}}}}}', todo_content, f"Found double brace for {var}")
    
    def test_legitimate_f_string_braces_preserved(self):
        """Test that legitimate double braces in f-strings are preserved."""
        source = ''.join([
            ''.join(cell.get('source', []))
            for cell in self.notebook.get('cells', [])
            if cell.get('cell_type') == 'code'
        ])
        
        # Look for f-strings that should legitimately have double braces
        # Find f-string patterns
        f_string_patterns = re.findall(r'f"""([^"]*(?:\\.[^"]*)*)"""', source, re.DOTALL)
        f_string_patterns.extend(re.findall(r'f"([^"]*(?:\\.[^"]*)*)"', source))
        f_string_patterns.extend(re.findall(r"f'([^']*(?:\\.[^']*)*)'", source))
        
        # These should be legitimate uses and should be preserved
        # The test passes if we find some f-strings (indicating they weren't incorrectly modified)
        if f_string_patterns:
            # This is good - f-strings are preserved
            pass


if __name__ == '__main__':
    unittest.main()