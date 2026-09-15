#!/usr/bin/env python3
"""
Integration Test for Enhanced Memory Categorization System

This test validates the complete integration of the enhanced memory system
within the Intelligent Data Detective workflow, including agent specialization,
memory categorization, and backward compatibility.
"""

import unittest
import tempfile
import sys
import os
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced memory functions
from memory_enhancements import (
    put_memory,
    retrieve_memories,
    format_memories_by_kind,
    enhanced_retrieve_mem,
    enhanced_mem_text,
    MEMORY_CONFIG,
    MemoryKind
)


class MockState:
    """Mock state object for testing agent functions."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)


class MockInMemoryStore:
    """Enhanced mock store for integration testing."""
    
    def __init__(self):
        self.data = {}
        self.search_calls = []
        self.put_calls = []
    
    def put(self, namespace: tuple, memory_id: str, item: Dict[str, Any]):
        """Store an item and track the call."""
        if namespace not in self.data:
            self.data[namespace] = {}
        self.data[namespace][memory_id] = item
        self.put_calls.append((namespace, memory_id, item))
    
    def search(self, namespace: tuple, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for items and track the call."""
        self.search_calls.append((namespace, query, limit))
        
        if namespace not in self.data:
            return []
        
        items = []
        for memory_id, item in self.data[namespace].items():
            # Enhanced keyword matching for testing
            if isinstance(item, dict):
                text = item.get("text", item.get("memory", ""))
                meta = item.get("meta", {})
                
                # Check if query terms match text content or metadata
                query_terms = query.lower().split()
                text_content = text.lower()
                
                if any(term in text_content for term in query_terms):
                    items.append(item.copy())
                elif any(term in str(meta).lower() for term in query_terms):
                    items.append(item.copy())
        
        return items[:limit]


class TestMemoryIntegration(unittest.TestCase):
    """Integration tests for the complete memory enhancement system."""
    
    def setUp(self):
        """Set up test fixtures for integration testing."""
        self.mock_store = MockInMemoryStore()
        self.test_user_id = "integration_test_user"
        
        # Simulate a complete workflow scenario
        self.workflow_scenario = {
            "user_query": "Analyze customer sales data and identify trends",
            "phases": [
                {
                    "agent": "supervisor",
                    "memory_kind": "conversation",
                    "content": "User wants to analyze customer sales data for trends"
                },
                {
                    "agent": "initial_analysis",
                    "memory_kind": "analysis", 
                    "content": "Dataset contains 10,000 customer records with sales from 2020-2023"
                },
                {
                    "agent": "data_cleaner",
                    "memory_kind": "cleaning",
                    "content": "Removed 50 duplicate records and filled missing customer_id values"
                },
                {
                    "agent": "analyst",
                    "memory_kind": "analysis",
                    "content": "Found strong seasonal patterns with 40% higher sales in Q4"
                },
                {
                    "agent": "viz_worker",
                    "memory_kind": "visualization",
                    "content": "Created seasonal sales trend chart and customer segment heatmap"
                },
                {
                    "agent": "report_orchestrator",
                    "memory_kind": "insights",
                    "content": "Sales peak during holiday season with premium customers driving growth"
                }
            ]
        }
    
    def test_workflow_memory_categorization(self):
        """Test memory categorization throughout a complete workflow."""
        # Simulate workflow phases
        for phase in self.workflow_scenario["phases"]:
            memory_id = put_memory(
                self.mock_store,
                phase["memory_kind"],
                phase["content"],
                meta={"agent": phase["agent"]},
                user_id=self.test_user_id
            )
            
            self.assertIsInstance(memory_id, str)
        
        # Verify all memory kinds were created
        expected_namespaces = {
            ("memories", "conversation"),
            ("memories", "analysis"), 
            ("memories", "cleaning"),
            ("memories", "visualization"),
            ("memories", "insights")
        }
        
        actual_namespaces = set(self.mock_store.data.keys())
        self.assertTrue(expected_namespaces.issubset(actual_namespaces))
    
    def test_agent_specific_memory_retrieval(self):
        """Test that different agents retrieve appropriate memory categories."""
        # Store workflow memories
        for phase in self.workflow_scenario["phases"]:
            put_memory(
                self.mock_store,
                phase["memory_kind"],
                phase["content"],
                meta={"agent": phase["agent"]},
                user_id=self.test_user_id
            )
        
        # Test agent-specific retrieval patterns
        agent_memory_tests = [
            {
                "agent": "analyst",
                "kinds": ["conversation", "analysis", "insights"],
                "query": "sales trends",
                "expected_categories": {"conversation", "analysis", "insights"}
            },
            {
                "agent": "viz_worker", 
                "kinds": ["conversation", "analysis", "visualization"],
                "query": "seasonal patterns",
                "expected_categories": {"conversation", "analysis", "visualization"}
            },
            {
                "agent": "report_packager",
                "kinds": ["conversation", "analysis", "cleaning", "visualization", "insights"],
                "query": "customer sales",
                "expected_categories": {"conversation", "analysis", "cleaning", "visualization", "insights"}
            }
        ]
        
        for test_case in agent_memory_tests:
            memories = retrieve_memories(
                self.mock_store,
                test_case["query"],
                kinds=test_case["kinds"],
                user_id=self.test_user_id
            )
            
            self.assertGreater(len(memories), 0, 
                f"Agent {test_case['agent']} should retrieve relevant memories")
            
            found_categories = {memory.get("namespace_kind") for memory in memories}
            expected_overlap = found_categories.intersection(test_case["expected_categories"])
            self.assertGreater(len(expected_overlap), 0,
                f"Agent {test_case['agent']} should find memories in expected categories")
    
    def test_memory_context_assembly(self):
        """Test that memory context is properly assembled for prompts."""
        # Store diverse memories
        memory_data = [
            ("conversation", "User asked about quarterly sales analysis"),
            ("analysis", "Q4 sales increased 40% compared to Q3"),
            ("cleaning", "Standardized date formats and removed outliers"),
            ("visualization", "Generated quarterly comparison bar chart"),
            ("insights", "Holiday shopping drives Q4 sales surge")
        ]
        
        for kind, content in memory_data:
            put_memory(self.mock_store, kind, content, user_id=self.test_user_id)
        
        # Retrieve and format for prompt
        memories = retrieve_memories(
            self.mock_store,
            "quarterly sales",
            kinds=["conversation", "analysis", "visualization", "insights"],
            user_id=self.test_user_id
        )
        
        formatted_context = format_memories_by_kind(memories)
        
        # Verify formatted output has proper sections
        self.assertIn("[Conversation Memory]", formatted_context)
        self.assertIn("[Analysis Memory]", formatted_context)
        self.assertIn("[Visualization Memory]", formatted_context)
        self.assertIn("[Insights Memory]", formatted_context)
        
        # Verify content is included
        self.assertIn("quarterly sales analysis", formatted_context)
        self.assertIn("40% compared", formatted_context)
        self.assertIn("comparison bar chart", formatted_context)
        self.assertIn("Holiday shopping", formatted_context)
    
    def test_backward_compatibility_fallback(self):
        """Test fallback to legacy memory namespace when categorized memories don't exist."""
        # Store memory in legacy format
        legacy_namespace = ("memories",)
        legacy_content = "Legacy memory about data analysis workflow"
        self.mock_store.put(legacy_namespace, "legacy_id", {
            "memory": legacy_content
        })
        
        # Try to retrieve with categories that don't exist
        memories = retrieve_memories(
            self.mock_store,
            "data analysis",
            kinds=["nonexistent_category"],
            user_id=self.test_user_id
        )
        
        # Should fall back to legacy namespace
        self.assertGreater(len(memories), 0)
        self.assertEqual(memories[0].get("namespace_kind"), "generic")
        self.assertIn("Legacy memory", str(memories[0]))
    
    def test_memory_limits_enforcement(self):
        """Test that memory retrieval respects configured limits."""
        # Store many memories of the same kind
        for i in range(15):
            put_memory(
                self.mock_store,
                "analysis",
                f"Analysis result {i} about sales data patterns",
                user_id=self.test_user_id
            )
        
        # Test limit enforcement
        limited_memories = retrieve_memories(
            self.mock_store,
            "sales data",
            kinds=["analysis"],
            limit=5,
            user_id=self.test_user_id
        )
        
        self.assertLessEqual(len(limited_memories), 5)
        
        # Test default config limits
        config_limited = retrieve_memories(
            self.mock_store,
            "sales data", 
            kinds=["analysis"],
            user_id=self.test_user_id
        )
        
        expected_limit = MEMORY_CONFIG["kinds"]["analysis"]["limit"]
        self.assertLessEqual(len(config_limited), expected_limit)
    
    def test_enhanced_retrieve_mem_function(self):
        """Test the enhanced_retrieve_mem function used by agent nodes."""
        # Store test memories
        put_memory(self.mock_store, "analysis", "Sales correlation analysis complete", user_id=self.test_user_id)
        put_memory(self.mock_store, "visualization", "Created sales trend visualization", user_id=self.test_user_id)
        
        # Mock state object
        mock_state = MockState(
            next_agent_prompt="sales analysis trends",
            user_prompt="analyze sales data"
        )
        
        # Test enhanced retrieval
        with patch('langgraph.utils.config.get_store', return_value=self.mock_store):
            memories = enhanced_retrieve_mem(
                mock_state,
                kinds=["analysis", "visualization"],
                limit=10
            )
        
        self.assertIsInstance(memories, list)
        if memories:  # Only test if we got results
            found_kinds = {m.get("namespace_kind") for m in memories}
            self.assertTrue(found_kinds.intersection({"analysis", "visualization"}))
    
    def test_enhanced_mem_text_formatting(self):
        """Test the enhanced_mem_text function with grouped output."""
        # Store memories across categories
        test_memories = [
            ("conversation", "User wants quarterly sales report"),
            ("analysis", "Identified seasonal sales patterns"),
            ("visualization", "Created quarterly sales charts")
        ]
        
        for kind, content in test_memories:
            put_memory(self.mock_store, kind, content, user_id=self.test_user_id)
        
        # Test enhanced formatting
        formatted_output = enhanced_mem_text(
            "quarterly sales",
            kinds=["conversation", "analysis", "visualization"],
            store=self.mock_store
        )
        
        # Should not be empty
        self.assertNotEqual(formatted_output, "None.")
        
        # Should contain section headers
        expected_sections = ["[Conversation Memory]", "[Analysis Memory]", "[Visualization Memory]"]
        for section in expected_sections:
            self.assertIn(section, formatted_output)
    
    def test_memory_deduplication_across_kinds(self):
        """Test handling of similar content across different memory categories."""
        # Store similar content in different categories
        similar_content = "Sales analysis shows strong Q4 performance"
        
        put_memory(self.mock_store, "analysis", similar_content, 
                  meta={"source": "analyst"}, user_id=self.test_user_id)
        put_memory(self.mock_store, "insights", similar_content,
                  meta={"source": "report_orchestrator"}, user_id=self.test_user_id)
        
        # Retrieve from both categories
        memories = retrieve_memories(
            self.mock_store,
            "Q4 performance",
            kinds=["analysis", "insights"],
            user_id=self.test_user_id
        )
        
        # Should get results from both categories
        found_kinds = {m.get("namespace_kind") for m in memories}
        self.assertEqual(len(found_kinds), 2)
        self.assertEqual(found_kinds, {"analysis", "insights"})
    
    def test_error_handling_robustness(self):
        """Test error handling in memory operations."""
        # Test with invalid store
        class FailingStore:
            def search(self, namespace, query, limit):
                raise Exception("Store connection failed")
        
        failing_store = FailingStore()
        
        # Should handle gracefully
        result = enhanced_mem_text("test query", store=failing_store)
        self.assertEqual(result, "None.")
        
        # Test with empty/None inputs
        empty_result = format_memories_by_kind([])
        self.assertEqual(empty_result, "None.")
        
        none_result = format_memories_by_kind(None)
        self.assertEqual(none_result, "None.")


class TestMemoryConfigurationCompliance(unittest.TestCase):
    """Test compliance with memory configuration specifications."""
    
    def test_memory_config_structure(self):
        """Test that MEMORY_CONFIG has required structure."""
        self.assertIn("kinds", MEMORY_CONFIG)
        self.assertIn("default_limit", MEMORY_CONFIG)
        self.assertIn("fallback_limit", MEMORY_CONFIG)
        
        # Test required memory kinds
        required_kinds = ["conversation", "analysis", "cleaning", "visualization"]
        for kind in required_kinds:
            self.assertIn(kind, MEMORY_CONFIG["kinds"])
            self.assertIn("limit", MEMORY_CONFIG["kinds"][kind])
    
    def test_memory_kind_limits(self):
        """Test that memory kind limits are reasonable."""
        for kind, config in MEMORY_CONFIG["kinds"].items():
            limit = config["limit"]
            self.assertIsInstance(limit, int)
            self.assertGreater(limit, 0)
            self.assertLessEqual(limit, 20)  # Reasonable upper bound


if __name__ == "__main__":
    # Run integration tests
    unittest.main(verbosity=2)