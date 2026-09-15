#!/usr/bin/env python3
"""
Tests for Enhanced Memory Categorization System

This test suite validates the structured multi-namespace memory categorization
functionality, including kind-based retrieval, fallback mechanisms, and 
backward compatibility.
"""

import unittest
import uuid
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Import the enhanced memory functions
from memory_enhancements import (
    put_memory,
    retrieve_memories,
    format_memories_by_kind,
    enhanced_retrieve_mem,
    enhanced_mem_text,
    update_memory_with_kind,
    MemoryKind,
    MEMORY_CONFIG
)


class MockInMemoryStore:
    """Mock InMemoryStore for testing without actual LangGraph dependencies."""
    
    def __init__(self):
        self.data = {}
    
    def put(self, namespace: tuple, memory_id: str, item: Dict[str, Any]):
        """Store an item in the mock store."""
        if namespace not in self.data:
            self.data[namespace] = {}
        self.data[namespace][memory_id] = item
    
    def search(self, namespace: tuple, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for items in the mock store."""
        if namespace not in self.data:
            return []
        
        items = []
        for memory_id, item in self.data[namespace].items():
            # Simple keyword matching for testing
            if isinstance(item, dict):
                text = item.get("text", item.get("memory", ""))
                if any(word.lower() in text.lower() for word in query.split()):
                    items.append(item.copy())
        
        return items[:limit]


class TestMemoryCategorization(unittest.TestCase):
    """Test suite for memory categorization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_store = MockInMemoryStore()
        self.test_user_id = "test_user"
        
        # Sample memory data
        self.sample_memories = {
            "conversation": [
                "User asked about data analysis",
                "User wants to clean missing values",
                "User requested visualization of correlations"
            ],
            "analysis": [
                "Found strong correlation between age and income",
                "Detected outliers in price column",
                "Missing values concentrated in specific columns"
            ],
            "cleaning": [
                "Removed 150 duplicate rows",
                "Filled missing values with median",
                "Encoded categorical variables"
            ],
            "visualization": [
                "Created scatter plot for age vs income",
                "Generated correlation heatmap",
                "Plotted distribution of target variable"
            ]
        }
    
    def test_put_memory_basic(self):
        """Test basic memory storage functionality."""
        memory_id = put_memory(
            self.mock_store,
            "analysis",
            "Test analysis memory",
            meta={"source": "test"},
            user_id=self.test_user_id
        )
        
        # Verify memory was stored
        self.assertIsInstance(memory_id, str)
        namespace = ("memories", "analysis")
        self.assertIn(namespace, self.mock_store.data)
        self.assertIn(memory_id, self.mock_store.data[namespace])
        
        stored_item = self.mock_store.data[namespace][memory_id]
        self.assertEqual(stored_item["text"], "Test analysis memory")
        self.assertEqual(stored_item["kind"], "analysis")
        self.assertEqual(stored_item["user_id"], self.test_user_id)
        self.assertIn("created_at", stored_item)
    
    def test_put_memory_all_kinds(self):
        """Test storing memories of all supported kinds."""
        for kind, texts in self.sample_memories.items():
            for text in texts:
                memory_id = put_memory(self.mock_store, kind, text, user_id=self.test_user_id)
                self.assertIsInstance(memory_id, str)
        
        # Verify all namespaces were created
        expected_namespaces = {("memories", kind) for kind in self.sample_memories.keys()}
        actual_namespaces = set(self.mock_store.data.keys())
        self.assertTrue(expected_namespaces.issubset(actual_namespaces))
    
    def test_retrieve_memories_single_kind(self):
        """Test retrieving memories from a single kind."""
        # Store test memories
        for kind, texts in self.sample_memories.items():
            for text in texts:
                put_memory(self.mock_store, kind, text, user_id=self.test_user_id)
        
        # Retrieve analysis memories
        results = retrieve_memories(
            self.mock_store,
            "correlation",
            kinds=["analysis"],
            user_id=self.test_user_id
        )
        
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertEqual(result.get("namespace_kind"), "analysis")
            self.assertIn("correlation", result.get("text", "").lower())
    
    def test_retrieve_memories_multiple_kinds(self):
        """Test retrieving memories from multiple kinds."""
        # Store test memories
        for kind, texts in self.sample_memories.items():
            for text in texts:
                put_memory(self.mock_store, kind, text, user_id=self.test_user_id)
        
        # Retrieve from analysis and visualization kinds
        results = retrieve_memories(
            self.mock_store,
            "correlation",
            kinds=["analysis", "visualization"],
            user_id=self.test_user_id
        )
        
        self.assertGreater(len(results), 0)
        found_kinds = {result.get("namespace_kind") for result in results}
        self.assertTrue(found_kinds.issubset({"analysis", "visualization"}))
    
    def test_retrieve_memories_fallback_to_generic(self):
        """Test fallback to generic namespace when no categorized memories found."""
        # Store memory in generic namespace (simulating old format)
        generic_namespace = ("memories",)
        self.mock_store.put(generic_namespace, str(uuid.uuid4()), {
            "memory": "Generic memory about data analysis"
        })
        
        # Try to retrieve specific kind that doesn't exist
        results = retrieve_memories(
            self.mock_store,
            "data analysis",
            kinds=["nonexistent_kind"],
            user_id=self.test_user_id
        )
        
        # Should fallback to generic namespace
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].get("namespace_kind"), "generic")
    
    def test_retrieve_memories_user_specific_fallback(self):
        """Test fallback to user-specific namespace."""
        # Store memory in user-specific namespace
        user_namespace = (self.test_user_id, "memories")
        self.mock_store.put(user_namespace, str(uuid.uuid4()), {
            "memory": "User-specific memory"
        })
        
        # Try to retrieve when no categorized memories exist
        results = retrieve_memories(
            self.mock_store,
            "memory",
            kinds=["analysis"],
            user_id=self.test_user_id
        )
        
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].get("namespace_kind"), "generic")
    
    def test_retrieve_memories_with_limit(self):
        """Test limit enforcement in memory retrieval."""
        # Store many memories of same kind
        for i in range(10):
            put_memory(
                self.mock_store,
                "analysis",
                f"Analysis memory {i} about correlation",
                user_id=self.test_user_id
            )
        
        # Retrieve with limit
        results = retrieve_memories(
            self.mock_store,
            "correlation",
            kinds=["analysis"],
            limit=3,
            user_id=self.test_user_id
        )
        
        self.assertLessEqual(len(results), 3)
    
    def test_format_memories_by_kind(self):
        """Test formatting of memories grouped by kind."""
        # Create test memories with different kinds
        memories = [
            {"text": "Analysis finding", "namespace_kind": "analysis"},
            {"text": "Cleaning step", "namespace_kind": "cleaning"},
            {"text": "Another analysis", "namespace_kind": "analysis"},
            {"text": "Visualization created", "namespace_kind": "visualization"}
        ]
        
        formatted = format_memories_by_kind(memories)
        
        # Should contain section headers
        self.assertIn("[Analysis Memory]", formatted)
        self.assertIn("[Cleaning Memory]", formatted)
        self.assertIn("[Visualization Memory]", formatted)
        
        # Should contain memory content
        self.assertIn("Analysis finding", formatted)
        self.assertIn("Cleaning step", formatted)
        self.assertIn("Visualization created", formatted)
    
    def test_format_memories_empty(self):
        """Test formatting of empty memory list."""
        formatted = format_memories_by_kind([])
        self.assertEqual(formatted, "None.")
    
    def test_format_memories_generic_kind(self):
        """Test formatting of generic/legacy memories."""
        memories = [
            {"memory": "Old format memory", "namespace_kind": "generic"}
        ]
        
        formatted = format_memories_by_kind(memories)
        self.assertIn("[Previous Context]", formatted)
        self.assertIn("Old format memory", formatted)
    
    def test_enhanced_retrieve_mem_with_state(self):
        """Test enhanced_retrieve_mem function with state parameter."""
        # Store test memories
        put_memory(self.mock_store, "analysis", "Test analysis result", user_id=self.test_user_id)
        
        # Mock state
        mock_state = {
            "next_agent_prompt": "analysis",
            "user_prompt": "fallback prompt"
        }
        
        # Test by providing the store directly to the internal function
        with patch('langgraph.utils.config.get_store', return_value=self.mock_store):
            results = enhanced_retrieve_mem(
                mock_state,
                kinds=["analysis"],
                limit=5
            )
        
        self.assertIsInstance(results, list)
    
    def test_enhanced_mem_text_integration(self):
        """Test enhanced_mem_text function."""
        # Store test memories
        for kind, texts in self.sample_memories.items():
            for text in texts[:2]:  # Store first 2 of each kind
                put_memory(self.mock_store, kind, text, user_id=self.test_user_id)
        
        # Test with specific kinds
        result = enhanced_mem_text(
            "correlation",
            kinds=["analysis", "visualization"],
            store=self.mock_store
        )
        
        self.assertNotEqual(result, "None.")
        self.assertIn("Memory]", result)  # Should have section headers
    
    def test_enhanced_mem_text_fallback(self):
        """Test enhanced_mem_text fallback when store not available."""
        result = enhanced_mem_text("test query", store=None)
        self.assertEqual(result, "None.")
    
    def test_update_memory_with_kind(self):
        """Test update_memory_with_kind function."""
        # Mock state with messages
        mock_message = Mock()
        mock_message.text.return_value = "Test message content"
        
        mock_state = {"messages": [mock_message]}
        mock_config = {"configurable": {"user_id": self.test_user_id}}
        
        memory_id = update_memory_with_kind(
            mock_state,
            mock_config,
            "conversation",
            memstore=self.mock_store
        )
        
        self.assertIsInstance(memory_id, str)
        
        # Verify memory was stored with correct kind
        namespace = ("memories", "conversation")
        self.assertIn(namespace, self.mock_store.data)
        self.assertIn(memory_id, self.mock_store.data[namespace])
    
    def test_backward_compatibility_namespace(self):
        """Test that system works with existing generic namespace format."""
        # Store memory in old format
        old_namespace = ("memories",)
        old_memory_id = str(uuid.uuid4())
        self.mock_store.put(old_namespace, old_memory_id, {
            "memory": "Legacy memory format"
        })
        
        # Should be retrievable via fallback
        results = retrieve_memories(
            self.mock_store,
            "legacy",
            kinds=["nonexistent"],
            user_id=self.test_user_id
        )
        
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].get("namespace_kind"), "generic")
    
    def test_memory_config_limits(self):
        """Test that memory configuration limits are respected."""
        # Test that config has expected structure
        self.assertIn("kinds", MEMORY_CONFIG)
        self.assertIn("default_limit", MEMORY_CONFIG)
        
        for kind in ["conversation", "analysis", "cleaning", "visualization"]:
            self.assertIn(kind, MEMORY_CONFIG["kinds"])
            self.assertIn("limit", MEMORY_CONFIG["kinds"][kind])
    
    def test_error_handling_in_retrieve(self):
        """Test error handling in memory retrieval."""
        # Create a store that raises exceptions
        class FailingStore:
            def search(self, namespace, query, limit):
                raise Exception("Store failure")
        
        failing_store = FailingStore()
        
        # Should handle exceptions gracefully
        results = retrieve_memories(
            failing_store,
            "test query",
            kinds=["analysis"],
            user_id=self.test_user_id
        )
        
        self.assertEqual(results, [])
    
    def test_memory_deduplication(self):
        """Test that duplicate memories are handled appropriately."""
        # Store same content in different kinds
        same_text = "Important finding about data"
        
        id1 = put_memory(self.mock_store, "analysis", same_text, user_id=self.test_user_id)
        id2 = put_memory(self.mock_store, "insights", same_text, user_id=self.test_user_id)
        
        # Both should be stored with different IDs
        self.assertNotEqual(id1, id2)
        
        # Should retrieve from both namespaces when searching multiple kinds
        results = retrieve_memories(
            self.mock_store,
            "finding",
            kinds=["analysis", "insights"],
            user_id=self.test_user_id
        )
        
        # Should get results from both kinds
        found_kinds = {result.get("namespace_kind") for result in results}
        self.assertIn("analysis", found_kinds)
        self.assertIn("insights", found_kinds)


class TestMemoryIntegration(unittest.TestCase):
    """Integration tests for memory system."""
    
    def test_complete_workflow_simulation(self):
        """Test a complete workflow using the enhanced memory system."""
        store = MockInMemoryStore()
        user_id = "workflow_user"
        
        # Simulate conversation start
        conv_id = put_memory(store, "conversation", "User wants to analyze sales data", user_id=user_id)
        
        # Simulate cleaning phase
        cleaning_id = put_memory(store, "cleaning", "Removed 50 duplicate records", user_id=user_id)
        
        # Simulate analysis phase
        analysis_id = put_memory(store, "analysis", "Found seasonal sales patterns", user_id=user_id)
        
        # Simulate visualization phase
        viz_id = put_memory(store, "visualization", "Created monthly sales trend chart", user_id=user_id)
        
        # Now retrieve context for report generation (should get all relevant context)
        report_context = retrieve_memories(
            store,
            "sales analysis",
            kinds=["conversation", "analysis", "visualization"],
            user_id=user_id
        )
        
        self.assertGreater(len(report_context), 0)
        
        # Format for prompt
        formatted_context = format_memories_by_kind(report_context)
        
        # Should contain multiple sections
        self.assertIn("[Conversation Memory]", formatted_context)
        self.assertIn("[Analysis Memory]", formatted_context)
        self.assertIn("[Visualization Memory]", formatted_context)
        
        # Should contain relevant content
        self.assertIn("sales data", formatted_context)
        self.assertIn("seasonal sales patterns", formatted_context)
        self.assertIn("trend chart", formatted_context)


if __name__ == "__main__":
    unittest.main()