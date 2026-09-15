#!/usr/bin/env python3
"""
Tests for Memory Lifecycle Management System

This test suite validates the memory lifecycle management functionality,
including TTL expiration, pruning strategies, importance scoring, 
duplicate detection, and retrieval ranking.
"""

import unittest
import time
import os
import tempfile
import yaml
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Import the memory lifecycle functions
from memory_enhancements import (
    MemoryRecord,
    MemoryPolicy,
    RankingWeights,
    PruneReport,
    MemoryPolicyEngine,
    put_memory_with_policy,
    retrieve_memories_with_ranking,
    prune_memories,
    recalculate_importance,
    estimate_importance,
    calculate_similarity,
    get_memory_metrics,
    reset_memory_metrics,
    memory_policy_report,
    load_memory_policy,
    MEMORY_POLICIES,
    RANKING_WEIGHTS
)


class MockInMemoryStore:
    """Enhanced mock store for lifecycle testing."""
    
    def __init__(self):
        self.data = {}
        self.search_calls = []
    
    def put(self, namespace: tuple, memory_id: str, item: Dict[str, Any]):
        """Store an item in the mock store."""
        if namespace not in self.data:
            self.data[namespace] = {}
        self.data[namespace][memory_id] = item.copy()
    
    def search(self, namespace: tuple, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for items in the mock store."""
        self.search_calls.append((namespace, query, limit))
        
        if namespace not in self.data:
            return []
        
        items = []
        for memory_id, item in self.data[namespace].items():
            # Simple keyword matching for testing
            if isinstance(item, dict):
                text = item.get("text", "")
                if not query or any(word.lower() in text.lower() for word in query.split()):
                    item_copy = item.copy()
                    item_copy["id"] = memory_id
                    items.append(item_copy)
        
        return items[:limit]
    
    def clear(self):
        """Clear all data."""
        self.data = {}
        self.search_calls = []


class TestMemoryLifecycle(unittest.TestCase):
    """Test suite for memory lifecycle management."""
    
    def setUp(self):
        """Set up test environment."""
        self.store = MockInMemoryStore()
        reset_memory_metrics()
        
        # Set debug mode for testing
        os.environ["DEBUG_MEMORY"] = "true"
    
    def tearDown(self):
        """Clean up after tests."""
        os.environ.pop("DEBUG_MEMORY", None)
    
    def test_memory_record_creation(self):
        """Test MemoryRecord dataclass functionality."""
        record = MemoryRecord(
            id="test-123",
            kind="analysis",
            text="This is a test analysis result",
            base_importance=0.8
        )
        
        self.assertEqual(record.id, "test-123")
        self.assertEqual(record.kind, "analysis")
        self.assertEqual(record.text, "This is a test analysis result")
        self.assertEqual(record.base_importance, 0.8)
        self.assertEqual(record.dynamic_importance, 0.8)
        self.assertEqual(record.usage_count, 0)
        self.assertIsNone(record.last_used_at)
        self.assertFalse(record.degraded)
    
    def test_memory_policy_creation(self):
        """Test MemoryPolicy dataclass functionality."""
        policy = MemoryPolicy(
            ttl_seconds=86400,
            max_items=100,
            min_importance=0.1
        )
        
        self.assertEqual(policy.ttl_seconds, 86400)
        self.assertEqual(policy.max_items, 100)
        self.assertEqual(policy.min_importance, 0.1)
    
    def test_estimate_importance(self):
        """Test importance estimation heuristics."""
        # Test with analysis content
        analysis_text = "Found significant correlation between variables. Key insight: pattern detected."
        importance = estimate_importance("analysis", analysis_text)
        self.assertGreater(importance, 0.65)  # Should be high due to keywords and type
        
        # Test with conversation content
        conversation_text = "Hello, how are you?"
        importance = estimate_importance("conversation", conversation_text)
        self.assertLess(importance, 0.6)  # Should be lower
        
        # Test with insights content (note: "insights" has high role weight but fewer keywords in this text)
        insight_text = "Critical finding: anomaly in data requires immediate attention insight analysis"
        importance = estimate_importance("insights", insight_text)
        self.assertGreater(importance, 0.7)  # Should be very high with more keywords
    
    def test_calculate_similarity(self):
        """Test similarity calculation."""
        text1 = "machine learning analysis results"
        text2 = "machine learning analysis findings"
        similarity = calculate_similarity(text1, text2)
        self.assertGreaterEqual(similarity, 0.6)  # High similarity
        
        text3 = "completely different topic about cooking"
        similarity_low = calculate_similarity(text1, text3)
        self.assertLess(similarity_low, 0.2)  # Low similarity
        
        # Test edge cases
        self.assertEqual(calculate_similarity("", "test"), 0.0)
        self.assertEqual(calculate_similarity("test", ""), 0.0)
    
    def test_policy_engine_insertion(self):
        """Test memory insertion with policy engine."""
        engine = MemoryPolicyEngine(self.store, debug=True)
        
        record = MemoryRecord(
            id="test-insert",
            kind="analysis",
            text="Important analysis result with correlation insight",
            base_importance=0.8
        )
        
        # Test insertion
        result = engine.insert(record)
        self.assertEqual(result.id, "test-insert")
        
        # Verify storage
        namespace = ("memories", "analysis")
        items = self.store.search(namespace, "", 10)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], record.text)
    
    def test_policy_engine_retrieval(self):
        """Test enhanced retrieval with ranking."""
        engine = MemoryPolicyEngine(self.store, debug=True)
        
        # Insert test records with different importance and ages
        records = [
            MemoryRecord(
                id="recent-important",
                kind="analysis", 
                text="recent critical analysis finding",
                created_at=time.time() - 3600,  # 1 hour ago
                base_importance=0.9,
                dynamic_importance=0.9,
                usage_count=5
            ),
            MemoryRecord(
                id="old-less-important",
                kind="analysis",
                text="old analysis result",
                created_at=time.time() - 86400*7,  # 1 week ago
                base_importance=0.3,
                dynamic_importance=0.3,
                usage_count=1
            ),
            MemoryRecord(
                id="medium-recent",
                kind="analysis",
                text="medium importance analysis",
                created_at=time.time() - 7200,  # 2 hours ago
                base_importance=0.6,
                dynamic_importance=0.6,
                usage_count=2
            )
        ]
        
        for record in records:
            engine.insert(record)
        
        # Test retrieval with ranking
        results = engine.retrieve("analysis", ["analysis"], 3)
        
        self.assertEqual(len(results), 3)
        
        # First result should be the most recent and important
        self.assertEqual(results[0].id, "recent-important")
        
        # Verify usage counts were updated
        self.assertGreater(results[0].usage_count, 5)
    
    def test_duplicate_detection(self):
        """Test duplicate suppression."""
        engine = MemoryPolicyEngine(self.store, debug=True)
        
        # Insert original record
        original = MemoryRecord(
            id="original",
            kind="analysis",
            text="This is a unique analysis result",
            base_importance=0.7
        )
        engine.insert(original)
        
        # Try to insert nearly identical record (should be detected as duplicate)
        duplicate = MemoryRecord(
            id="duplicate",
            kind="analysis",
            text="This is a unique analysis result",  # Exact same text
            base_importance=0.7
        )
        engine.insert(duplicate)
        
        # Should only have one record stored (duplicate should be dropped)
        namespace = ("memories", "analysis")
        items = self.store.search(namespace, "", 10)
        self.assertEqual(len(items), 1)
    
    def test_ttl_pruning(self):
        """Test time-based pruning."""
        engine = MemoryPolicyEngine(self.store, debug=True)
        
        # Insert expired record
        old_record = MemoryRecord(
            id="old-expired",
            kind="conversation",
            text="Old conversation content",
            created_at=time.time() - 86400*30,  # 30 days ago (beyond TTL)
            base_importance=0.5
        )
        engine.insert(old_record)
        
        # Insert fresh record
        fresh_record = MemoryRecord(
            id="fresh",
            kind="conversation", 
            text="Fresh conversation content",
            created_at=time.time() - 3600,  # 1 hour ago
            base_importance=0.5
        )
        engine.insert(fresh_record)
        
        # Run pruning
        report = engine.prune("test_ttl")
        
        # Should have pruned the expired record
        self.assertGreater(report.expired_count, 0)
        self.assertGreater(report.total_pruned, 0)
    
    def test_size_based_pruning(self):
        """Test size-based pruning."""
        # Create a temporary policy with very low limits
        test_policy = MemoryPolicy(
            ttl_seconds=86400*365,  # Very long TTL
            max_items=2,  # Very low limit
            min_importance=0.01
        )
        
        with patch('memory_enhancements.MEMORY_POLICIES', 
                   {"analysis": test_policy, "conversation": test_policy, 
                    "cleaning": test_policy, "visualization": test_policy,
                    "insights": test_policy, "errors": test_policy}):
            
            engine = MemoryPolicyEngine(self.store, debug=True)
            
            # Insert more records than the limit
            for i in range(5):
                record = MemoryRecord(
                    id=f"record-{i}",
                    kind="analysis",
                    text=f"Analysis result {i}",
                    base_importance=0.5 + (i * 0.1),  # Increasing importance
                    dynamic_importance=0.5 + (i * 0.1)
                )
                engine.insert(record)
            
            # Run pruning
            report = engine.prune("test_size")
            
            # Should have pruned excess records
            self.assertGreater(report.size_pruned_count, 0)
    
    def test_importance_recalculation(self):
        """Test dynamic importance updates."""
        engine = MemoryPolicyEngine(self.store, debug=True)
        
        record = MemoryRecord(
            id="test-recalc",
            kind="analysis",
            text="Analysis for importance testing",
            created_at=time.time() - 3600,  # 1 hour ago
            base_importance=0.5,
            usage_count=10  # High usage
        )
        
        engine.insert(record)
        
        # Recalculate importance
        engine.recalc_importance([record])
        
        # Should have higher dynamic importance due to usage
        self.assertGreater(record.dynamic_importance, record.base_importance)
    
    def test_memory_metrics(self):
        """Test metrics collection."""
        reset_memory_metrics()
        
        engine = MemoryPolicyEngine(self.store, debug=True)
        
        # Insert some records
        for i in range(3):
            record = MemoryRecord(
                id=f"metric-test-{i}",
                kind="analysis",
                text=f"Test content {i}",
                base_importance=0.5
            )
            engine.insert(record)
        
        # Check metrics
        metrics = get_memory_metrics()
        self.assertEqual(metrics["memory_put_total"], 3)
        self.assertGreater(metrics["memory_items_total"], 0)
        self.assertIn("analysis", metrics["memory_items_by_kind"])
    
    def test_policy_configuration_loading(self):
        """Test loading policy configuration from YAML."""
        # Create temporary config file
        config_data = {
            "memory_policy": {
                "defaults": {
                    "ttl_seconds": 12345,
                    "max_items": 999,
                    "min_importance": 0.123
                },
                "kinds": {
                    "analysis": {
                        "ttl_seconds": 54321,
                        "max_items": 888
                    }
                }
            },
            "ranking": {
                "weights": {
                    "similarity": 0.6,
                    "importance": 0.3,
                    "recency": 0.08,
                    "usage": 0.02
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            # Load configuration
            policies, weights = load_memory_policy(temp_config_path)
            
            # Verify default policy values
            self.assertEqual(policies["conversation"].ttl_seconds, 12345)
            self.assertEqual(policies["conversation"].max_items, 999)
            self.assertEqual(policies["conversation"].min_importance, 0.123)
            
            # Verify kind-specific overrides
            self.assertEqual(policies["analysis"].ttl_seconds, 54321)
            self.assertEqual(policies["analysis"].max_items, 888)
            
            # Verify ranking weights
            self.assertEqual(weights.similarity, 0.6)
            self.assertEqual(weights.importance, 0.3)
            
        finally:
            os.unlink(temp_config_path)
    
    def test_memory_policy_report(self):
        """Test diagnostic reporting."""
        engine = MemoryPolicyEngine(self.store, debug=True)
        
        # Insert some test data
        record = MemoryRecord(
            id="report-test",
            kind="analysis",
            text="Test for reporting",
            base_importance=0.5
        )
        engine.insert(record)
        
        # Generate report
        report = memory_policy_report(self.store)
        
        self.assertIn("timestamp", report)
        self.assertIn("metrics", report)
        self.assertIn("kind_status", report)
        self.assertIn("analysis", report["kind_status"])
    
    def test_backward_compatibility(self):
        """Test that existing functions still work."""
        # Test original put_memory function
        from memory_enhancements import put_memory
        
        memory_id = put_memory(
            self.store,
            "analysis",
            "Test backward compatibility",
            use_policy_engine=False  # Test original path
        )
        
        self.assertIsInstance(memory_id, str)
        
        # Test original retrieve_memories function
        from memory_enhancements import retrieve_memories
        
        results = retrieve_memories(
            self.store,
            "compatibility",
            kinds=["analysis"],
            use_policy_engine=False  # Test original path
        )
        
        self.assertIsInstance(results, list)
    
    def test_enhanced_wrapper_functions(self):
        """Test enhanced wrapper functions."""
        # Test put_memory_with_policy
        memory_id = put_memory_with_policy(
            self.store,
            "insights",
            "Important insight with keywords: critical finding anomaly",
            meta={"source": "test"}
        )
        
        self.assertIsInstance(memory_id, str)
        
        # Test retrieve_memories_with_ranking
        results = retrieve_memories_with_ranking(
            self.store,
            "insight critical",
            kinds=["insights"],
            limit=5
        )
        
        self.assertIsInstance(results, list)
        if results:
            # Should have enhanced fields
            self.assertIn("usage_count", results[0])
            self.assertIn("dynamic_importance", results[0])
    
    def test_prune_memories_function(self):
        """Test standalone prune function."""
        # Insert test data
        engine = MemoryPolicyEngine(self.store)
        record = MemoryRecord(
            id="prune-test",
            kind="conversation",
            text="Test for pruning function",
            created_at=time.time() - 86400*30,  # Old record
            base_importance=0.1  # Low importance
        )
        engine.insert(record)
        
        # Test pruning
        report = prune_memories(self.store, "test_standalone")
        self.assertIsInstance(report, PruneReport)
    
    def test_recalculate_importance_function(self):
        """Test standalone importance recalculation."""
        # Insert test data
        engine = MemoryPolicyEngine(self.store)
        record = MemoryRecord(
            id="importance-test",
            kind="analysis",
            text="Test for importance recalculation",
            base_importance=0.5,
            usage_count=5
        )
        engine.insert(record)
        
        # Test recalculation
        updated_count = recalculate_importance(self.store, ["analysis"])
        self.assertGreater(updated_count, 0)


class TestMemoryIntegrationLifecycle(unittest.TestCase):
    """Integration tests for complete memory lifecycle workflows."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.store = MockInMemoryStore()
        reset_memory_metrics()
    
    def test_complete_lifecycle_workflow(self):
        """Test complete memory lifecycle from insertion to pruning."""
        engine = MemoryPolicyEngine(self.store, debug=True)
        
        # Step 1: Insert diverse memories
        records = []
        for i in range(10):
            record = MemoryRecord(
                id=f"workflow-{i}",
                kind="analysis" if i % 2 == 0 else "conversation",
                text=f"Test memory {i} with various importance levels",
                created_at=time.time() - (i * 3600),  # Increasing age
                base_importance=0.1 + (i * 0.1),  # Increasing importance
                usage_count=i
            )
            records.append(record)
            engine.insert(record)
        
        # Step 2: Retrieve and verify ranking
        results = engine.retrieve("memory importance", ["analysis", "conversation"], 5)
        self.assertEqual(len(results), 5)
        
        # Should be ranked by combined score
        scores = [engine._calculate_score("memory importance", r) for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Step 3: Recalculate importance
        engine.recalc_importance(records)
        
        # Step 4: Run pruning
        report = engine.prune("integration_test")
        
        # Step 5: Generate diagnostic report
        diagnostic = memory_policy_report(self.store)
        self.assertIn("kind_status", diagnostic)
        
        # Step 6: Verify metrics
        metrics = get_memory_metrics()
        self.assertGreater(metrics["memory_put_total"], 0)
        self.assertGreater(metrics["memory_retrieval_requests_total"], 0)


if __name__ == "__main__":
    unittest.main()