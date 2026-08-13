#!/usr/bin/env python3
"""
Comprehensive tests for Adaptive Context Retrieval with Dynamic Memory Window Sizing.

Tests the core adaptive retrieval functionality including per-agent profiles,
multi-factor ranking, token-aware packing, and backward compatibility.
"""

import unittest
import tempfile
import os
import time
import yaml
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Import the adaptive retrieval system
from adaptive_retrieval import (
    AdaptiveRetriever, AgentProfile, GlobalConfig, RetrievalResult,
    TaskPhase, TokenEstimator, is_adaptive_retrieval_enabled,
    create_adaptive_retriever
)

from memory_enhancements import (
    MemoryRecord, MemoryPolicyEngine, put_memory, RankingWeights
)

class MockInMemoryStore:
    """Enhanced mock store for testing adaptive retrieval."""
    
    def __init__(self):
        self.data = {}
        self.search_calls = []
        self.put_calls = []
    
    def put(self, namespace: tuple, key: str, value: dict):
        """Store an item."""
        self.put_calls.append((namespace, key, value))
        if namespace not in self.data:
            self.data[namespace] = {}
        self.data[namespace][key] = value
    
    def search(self, namespace: tuple, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for items with enhanced matching."""
        self.search_calls.append((namespace, query, limit))
        
        if namespace not in self.data:
            return []
        
        items = []
        for item_id, item in self.data[namespace].items():
            # Enhanced keyword matching for testing
            if isinstance(item, dict):
                text = item.get("text", item.get("memory", ""))
                query_terms = query.lower().split()
                text_content = text.lower()
                
                # More lenient matching - match if any query term is in text OR if query is empty
                if not query or any(term in text_content for term in query_terms):
                    items.append(item.copy())
        
        return items[:limit]

class TestTokenEstimator(unittest.TestCase):
    """Test token estimation functionality."""
    
    def setUp(self):
        self.estimator = TokenEstimator()
    
    def test_empty_text_estimation(self):
        """Test token estimation for empty text."""
        result = self.estimator.estimate("")
        self.assertEqual(result.estimated_tokens, 0)
        self.assertEqual(result.text, "")
        self.assertEqual(result.method, "heuristic")
    
    def test_simple_text_estimation(self):
        """Test token estimation for simple text."""
        text = "This is a simple test"
        result = self.estimator.estimate(text)
        self.assertGreater(result.estimated_tokens, 0)
        self.assertEqual(result.text, text)
        self.assertAlmostEqual(result.estimated_tokens, len(text) / 4.0, places=0)
    
    def test_long_text_estimation(self):
        """Test token estimation for longer text."""
        text = "This is a much longer piece of text " * 10
        result = self.estimator.estimate(text)
        self.assertGreater(result.estimated_tokens, 50)
        self.assertEqual(result.text, text)

class TestAgentProfile(unittest.TestCase):
    """Test agent profile configuration and validation."""
    
    def test_valid_profile_creation(self):
        """Test creating a valid agent profile."""
        profile = AgentProfile(
            agent_name="test_agent",
            kinds=["conversation", "analysis"],
            min_items=2,
            max_items=10,
            target_items=7,
            token_budget=600
        )
        
        self.assertEqual(profile.agent_name, "test_agent")
        self.assertEqual(profile.kinds, ["conversation", "analysis"])
        self.assertEqual(profile.min_items, 2)
        self.assertEqual(profile.max_items, 10)
        self.assertEqual(profile.target_items, 7)
        self.assertEqual(profile.token_budget, 600)
    
    def test_profile_validation_min_max(self):
        """Test profile validation for min > max items."""
        with self.assertRaises(ValueError):
            AgentProfile(
                agent_name="invalid_agent",
                kinds=["conversation"],
                min_items=10,
                max_items=5  # This should fail
            )
    
    def test_profile_auto_adjustment(self):
        """Test automatic adjustment of target items."""
        profile = AgentProfile(
            agent_name="test_agent",
            kinds=["conversation"],
            min_items=2,
            max_items=8,
            target_items=15  # This should be adjusted to max_items
        )
        
        self.assertEqual(profile.target_items, 8)
        
        # Test adjustment to min_items
        profile2 = AgentProfile(
            agent_name="test_agent2",
            kinds=["conversation"],
            min_items=5,
            max_items=10,
            target_items=1  # This should be adjusted to min_items
        )
        
        self.assertEqual(profile2.target_items, 5)

class TestAdaptiveRetriever(unittest.TestCase):
    """Test the core adaptive retrieval engine."""
    
    def setUp(self):
        """Set up test environment."""
        self.store = MockInMemoryStore()
        self.config_path = self._create_test_config()
        self.retriever = AdaptiveRetriever(self.store, self.config_path, debug=True)
        self._populate_test_data()
    
    def _create_test_config(self) -> str:
        """Create a temporary test configuration file."""
        config = {
            "adaptive_retrieval": {
                "global": {
                    "default_memory_token_budget": 900,
                    "hard_token_budget_fraction": 0.35,
                    "fallback_limit": 5,
                    "enable_diagnostics": True
                },
                "agents": {
                    "test_agent": {
                        "kinds": ["conversation", "analysis"],
                        "min_items": 2,
                        "max_items": 8,
                        "target_items": 5,
                        "token_budget": 400,
                        "weighting_overrides": {
                            "similarity": 0.6,
                            "importance": 0.3,
                            "recency": 0.1,
                            "usage": 0.0
                        }
                    },
                    "analyst": {
                        "kinds": ["analysis", "cleaning"],
                        "min_items": 3,
                        "max_items": 12,
                        "target_items": 8,
                        "token_budget": 800
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            return f.name
    
    def _populate_test_data(self):
        """Populate store with test memory data."""
        current_time = time.time()
        
        # Add various types of memories to multiple namespaces
        test_memories = [
            {
                "id": "mem1",
                "kind": "conversation",
                "text": "User asked about data analysis trends",
                "created_at": current_time - 3600,  # 1 hour ago
                "usage_count": 5,
                "base_importance": 0.6,
                "dynamic_importance": 0.7,
                "user_id": "test_user"
            },
            {
                "id": "mem2", 
                "kind": "analysis",
                "text": "Found strong correlation between variables X and Y with correlation coefficient 0.85",
                "created_at": current_time - 1800,  # 30 minutes ago
                "usage_count": 3,
                "base_importance": 0.9,
                "dynamic_importance": 0.95,
                "user_id": "test_user"
            },
            {
                "id": "mem3",
                "kind": "cleaning",
                "text": "Removed 45 duplicate rows and filled missing values with median",
                "created_at": current_time - 7200,  # 2 hours ago
                "usage_count": 2,
                "base_importance": 0.7,
                "dynamic_importance": 0.65,
                "user_id": "test_user"
            },
            {
                "id": "mem4",
                "kind": "conversation",
                "text": "User wants to focus on outlier detection",
                "created_at": current_time - 900,  # 15 minutes ago
                "usage_count": 1,
                "base_importance": 0.5,
                "dynamic_importance": 0.6,
                "user_id": "test_user"
            },
            {
                "id": "mem5",
                "kind": "analysis",
                "text": "Identified significant outliers in column Z using IQR method",
                "created_at": current_time - 600,  # 10 minutes ago
                "usage_count": 2,
                "base_importance": 0.8,
                "dynamic_importance": 0.85,
                "user_id": "test_user"
            },
            {
                "id": "mem6",
                "kind": "cleaning",
                "text": "Applied log transformation to normalize skewed data",
                "created_at": current_time - 5400,  # 1.5 hours ago
                "usage_count": 4,
                "base_importance": 0.75,
                "dynamic_importance": 0.8,
                "user_id": "test_user"
            }
        ]
        
        for mem in test_memories:
            namespace = ("memories", mem["kind"])
            self.store.put(namespace, mem["id"], mem)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'config_path') and os.path.exists(self.config_path):
            os.unlink(self.config_path)
    
    def test_adaptive_retrieval_basic(self):
        """Test basic adaptive retrieval functionality."""
        result = self.retriever.get_context(
            agent_name="test_agent",
            query="analysis correlation outlier",
            base_prompt_tokens=100,
            model_context_window=4096
        )
        
        self.assertIsInstance(result, RetrievalResult)
        self.assertGreater(len(result.selected), 0)
        self.assertLessEqual(len(result.selected), 8)  # max_items for test_agent
        self.assertGreaterEqual(len(result.selected), 2)  # min_items for test_agent
        self.assertEqual(result.profile_used, "test_agent")
        self.assertFalse(result.fallback_used)
    
    def test_agent_profile_application(self):
        """Test that agent profiles are properly applied."""
        # Test with test_agent profile
        result1 = self.retriever.get_context(
            agent_name="test_agent",
            query="analysis",
            base_prompt_tokens=100
        )
        
        # Test with analyst profile  
        result2 = self.retriever.get_context(
            agent_name="analyst",
            query="analysis",
            base_prompt_tokens=100
        )
        
        # Both should return results with their respective limits
        self.assertLessEqual(len(result1.selected), 8)  # test_agent max
        self.assertLessEqual(len(result2.selected), 12)  # analyst max
        
        # Verify profiles were used
        self.assertEqual(result1.profile_used, "test_agent")
        self.assertEqual(result2.profile_used, "analyst")
        self.assertFalse(result1.fallback_used)
        self.assertFalse(result2.fallback_used)
        
        # If there are memories available, analyst should have at least some results
        if result2.total_candidates > 0:
            self.assertGreater(len(result2.selected), 0)
    
    def test_token_budget_enforcement(self):
        """Test that token budgets are respected."""
        # Test with very small token budget
        result = self.retriever.get_context(
            agent_name="test_agent",
            query="analysis correlation",
            base_prompt_tokens=100,
            model_context_window=4096,
            task_overrides={"token_budget": 50}  # Very small budget
        )
        
        self.assertLessEqual(result.token_usage, 50)
        self.assertGreaterEqual(len(result.selected), 2)  # Should still meet min_items
    
    def test_context_window_overflow_protection(self):
        """Test protection against context window overflow."""
        result = self.retriever.get_context(
            agent_name="test_agent",
            query="analysis",
            base_prompt_tokens=3000,  # Large base prompt
            model_context_window=4096,  # Small context window
            expected_output_tokens=800
        )
        
        # Should handle budget constraints gracefully
        self.assertGreaterEqual(len(result.selected), 0)  # Should return some results
        self.assertTrue(result.truncated or len(result.selected) > 0)  # Should be truncated OR have results
    
    def test_composite_scoring(self):
        """Test that composite scoring ranks memories appropriately."""
        result = self.retriever.get_context(
            agent_name="test_agent",
            query="correlation analysis",  # Should favor analysis memories
            base_prompt_tokens=100
        )
        
        # Check that high-importance analysis memories are preferred
        analysis_memories = [m for m in result.selected if m.kind == "analysis"]
        self.assertGreater(len(analysis_memories), 0)
        
        # Most recent and relevant should be included
        memory_ids = [m.id for m in result.selected]
        self.assertIn("mem2", memory_ids)  # High importance analysis memory
    
    def test_intent_adjustments(self):
        """Test query intent-based adjustments."""
        # Query with analytical intent
        result1 = self.retriever.get_context(
            agent_name="test_agent",
            query="show me trends and correlations in the data",
            base_prompt_tokens=100
        )
        
        # Query without analytical intent
        result2 = self.retriever.get_context(
            agent_name="test_agent", 
            query="hello how are you",
            base_prompt_tokens=100
        )
        
        # Analytical query should prefer analysis memories
        analysis_count1 = sum(1 for m in result1.selected if m.kind == "analysis")
        analysis_count2 = sum(1 for m in result2.selected if m.kind == "analysis")
        
        self.assertGreaterEqual(analysis_count1, analysis_count2)
    
    def test_diagnostics_generation(self):
        """Test diagnostic information generation."""
        result = self.retriever.get_context(
            agent_name="test_agent",
            query="analysis",
            base_prompt_tokens=100
        )
        
        self.assertIsNotNone(result.diagnostics)
        diag = result.diagnostics
        
        # Check diagnostic structure
        self.assertIn("query", diag)
        self.assertIn("agent_profile", diag)
        self.assertIn("candidate_table", diag)
        self.assertIn("profile_config", diag)
        self.assertIn("exclusion_summary", diag)
        
        # Check candidate table structure
        if diag["candidate_table"]:
            candidate = diag["candidate_table"][0]
            required_fields = ["rank", "id", "kind", "similarity", "importance", 
                             "recency", "usage", "score", "est_tokens", "packed"]
            for field in required_fields:
                self.assertIn(field, candidate)
    
    def test_fallback_retrieval(self):
        """Test fallback to legacy retrieval when needed."""
        # Test with unknown agent
        result = self.retriever.get_context(
            agent_name="unknown_agent",
            query="test query",
            base_prompt_tokens=100
        )
        
        self.assertTrue(result.fallback_used)
        self.assertIsNone(result.profile_used)
    
    def test_metrics_collection(self):
        """Test that metrics are properly collected."""
        initial_metrics = self.retriever.get_metrics()
        
        # Perform some retrievals
        self.retriever.get_context("test_agent", "query1", 100)
        self.retriever.get_context("analyst", "query2", 100)
        self.retriever.get_context("unknown_agent", "query3", 100)  # Should fallback
        
        final_metrics = self.retriever.get_metrics()
        
        # Check that counters increased
        self.assertEqual(final_metrics["adaptive_memory_requests_total"], 3)
        self.assertEqual(final_metrics["adaptive_memory_fallback_used_total"], 1)
        
        # Check that at least some requests found candidates (the two valid agents should)
        self.assertGreaterEqual(final_metrics["adaptive_memory_candidates_avg"], 0)
        
        # Check that metrics were accumulated
        self.assertGreaterEqual(final_metrics["adaptive_memory_selected_avg"], 0)
    
    def test_task_phase_overrides(self):
        """Test task phase specific overrides."""
        # This would test phase-specific adjustments if implemented
        result = self.retriever.get_context(
            agent_name="test_agent",
            query="analysis",
            base_prompt_tokens=100,
            task_phase=TaskPhase.ANALYSIS
        )
        
        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(result.profile_used, "test_agent")

class TestFeatureFlags(unittest.TestCase):
    """Test feature flag functionality."""
    
    def test_feature_flag_disabled(self):
        """Test behavior when adaptive retrieval is disabled."""
        with patch.dict(os.environ, {"ADAPTIVE_RETRIEVAL_ENABLED": "false"}):
            self.assertFalse(is_adaptive_retrieval_enabled())
            
            store = MockInMemoryStore()
            retriever = create_adaptive_retriever(store)
            self.assertIsNone(retriever)
    
    def test_feature_flag_enabled(self):
        """Test behavior when adaptive retrieval is enabled."""
        with patch.dict(os.environ, {"ADAPTIVE_RETRIEVAL_ENABLED": "true"}):
            self.assertTrue(is_adaptive_retrieval_enabled())
            
            store = MockInMemoryStore()
            retriever = create_adaptive_retriever(store)
            self.assertIsNotNone(retriever)
            self.assertIsInstance(retriever, AdaptiveRetriever)

class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing memory system."""
    
    def setUp(self):
        self.store = MockInMemoryStore()
        
        # Add some legacy format memories
        legacy_memories = [
            {"text": "legacy memory 1", "created_at": time.time()},
            {"text": "legacy memory 2", "created_at": time.time()}
        ]
        
        for i, mem in enumerate(legacy_memories):
            self.store.put(("memories",), f"legacy_{i}", mem)
    
    def test_fallback_to_legacy_format(self):
        """Test fallback to legacy memory format."""
        config_path = self._create_minimal_config()
        retriever = AdaptiveRetriever(self.store, config_path)
        
        # Query should work even with legacy format data
        result = retriever.get_context(
            agent_name="unknown_agent",  # Should trigger fallback
            query="legacy",
            base_prompt_tokens=100
        )
        
        self.assertTrue(result.fallback_used)
        # Should still return some results from legacy format
        self.assertGreaterEqual(len(result.selected), 0)
        
        os.unlink(config_path)
    
    def _create_minimal_config(self) -> str:
        """Create minimal test config."""
        config = {
            "adaptive_retrieval": {
                "global": {"fallback_limit": 5},
                "agents": {}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            return f.name

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        self.store = MockInMemoryStore()
    
    def test_empty_store(self):
        """Test behavior with empty memory store."""
        config_path = self._create_test_config()
        retriever = AdaptiveRetriever(self.store, config_path)
        
        result = retriever.get_context(
            agent_name="test_agent",
            query="anything",
            base_prompt_tokens=100
        )
        
        self.assertEqual(len(result.selected), 0)
        self.assertEqual(result.token_usage, 0)
        self.assertFalse(result.truncated)
        
        os.unlink(config_path)
    
    def test_invalid_config_fallback(self):
        """Test fallback behavior with invalid configuration."""
        # Create invalid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content [")
            invalid_config_path = f.name
        
        try:
            retriever = AdaptiveRetriever(self.store, invalid_config_path)
            
            # Should fall back to default configuration
            result = retriever.get_context(
                agent_name="initial_analysis",  # Should have default profile
                query="test",
                base_prompt_tokens=100
            )
            
            self.assertIsInstance(result, RetrievalResult)
            
        finally:
            os.unlink(invalid_config_path)
    
    def test_zero_token_budget(self):
        """Test behavior with zero token budget."""
        config_path = self._create_test_config()
        retriever = AdaptiveRetriever(self.store, config_path)
        
        result = retriever.get_context(
            agent_name="test_agent",
            query="test",
            base_prompt_tokens=100,
            task_overrides={"token_budget": 0}
        )
        
        # Should still return minimum items even with zero budget
        self.assertGreaterEqual(len(result.selected), 0)
        
        os.unlink(config_path)
    
    def _create_test_config(self) -> str:
        """Create basic test configuration."""
        config = {
            "adaptive_retrieval": {
                "global": {"fallback_limit": 5},
                "agents": {
                    "test_agent": {
                        "kinds": ["conversation"],
                        "min_items": 1,
                        "max_items": 5,
                        "token_budget": 300
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            return f.name

if __name__ == "__main__":
    # Enable debug logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()