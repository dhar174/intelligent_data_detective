#!/usr/bin/env python3
"""
Test prompt template formatting issues before and after fixes.
"""

import pytest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/runner/work/intelligent_data_detective/intelligent_data_detective')

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class TestPromptTemplateFormatting:
    """Test prompt template formatting for correct variable substitution."""
    
    def test_double_brace_issue_detection(self):
        """Test that we can detect double brace formatting issues."""
        # This should fail - double braces won't substitute correctly
        with pytest.raises((KeyError, ValueError)):
            template = ChatPromptTemplate.from_messages([
                ("system", "Objective: {{user_prompt}}")
            ])
            # Try to format with actual variables
            formatted = template.format_messages(user_prompt="test")
    
    def test_correct_single_brace_formatting(self):
        """Test that single braces work correctly."""
        template = ChatPromptTemplate.from_messages([
            ("system", "Objective: {user_prompt}")
        ])
        # This should work
        formatted = template.format_messages(user_prompt="test")
        assert len(formatted) == 1
        assert "test" in formatted[0].content
    
    def test_partial_with_double_braces(self):
        """Test that partial() doesn't work with double braces."""
        template = ChatPromptTemplate.from_messages([
            ("system", "Schema: {{output_schema_name}}")
        ]).partial(output_schema_name="Plan")
        
        # The double braces should not be substituted by partial
        formatted = template.format_messages()
        assert "{{output_schema_name}}" in formatted[0].content
        assert "Plan" not in formatted[0].content
    
    def test_partial_with_single_braces(self):
        """Test that partial() works correctly with single braces."""
        template = ChatPromptTemplate.from_messages([
            ("system", "Schema: {output_schema_name}")
        ]).partial(output_schema_name="Plan")
        
        # The single braces should be substituted by partial
        formatted = template.format_messages()
        assert "Plan" in formatted[0].content
        assert "{output_schema_name}" not in formatted[0].content
    
    def test_mixed_placeholders(self):
        """Test template with both partial and runtime variables."""
        template = ChatPromptTemplate.from_messages([
            ("system", "Schema: {output_schema_name}, User: {user_prompt}")
        ]).partial(output_schema_name="Plan")
        
        # Should work with runtime substitution
        formatted = template.format_messages(user_prompt="test query")
        content = formatted[0].content
        assert "Plan" in content
        assert "test query" in content
        assert "{output_schema_name}" not in content
        assert "{user_prompt}" not in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])