#!/usr/bin/env python3
"""
Demonstration of Adaptive Context Retrieval with Dynamic Memory Window Sizing

This script shows how to use the new adaptive retrieval system with different
agent types and demonstrates the key features.
"""

import os
import time
import tempfile
from typing import Dict, Any

# Set up adaptive retrieval
os.environ["ADAPTIVE_RETRIEVAL_ENABLED"] = "true"

from adaptive_retrieval import AdaptiveRetriever, TaskPhase
from adaptive_integration import (
    get_adaptive_retrieval_diagnostics,
    adaptive_retrieve_mem,
    analyst_retrieve_mem,
    visualization_retrieve_mem
)
from test_adaptive_retrieval import MockInMemoryStore

def create_demo_store():
    """Create a store with sample memory data."""
    store = MockInMemoryStore()
    current_time = time.time()
    
    # Sample memories for demonstration
    memories = [
        {
            "id": "conv1",
            "kind": "conversation", 
            "text": "User wants to analyze sales trends and find correlations with marketing spend",
            "created_at": current_time - 1800,
            "usage_count": 3,
            "base_importance": 0.7,
            "dynamic_importance": 0.75
        },
        {
            "id": "analysis1",
            "kind": "analysis",
            "text": "Found strong positive correlation (r=0.83) between marketing spend and sales revenue",
            "created_at": current_time - 1200,
            "usage_count": 5,
            "base_importance": 0.9,
            "dynamic_importance": 0.95
        },
        {
            "id": "cleaning1", 
            "kind": "cleaning",
            "text": "Removed 23 duplicate records and filled missing values in marketing_spend column using median",
            "created_at": current_time - 3600,
            "usage_count": 2,
            "base_importance": 0.6,
            "dynamic_importance": 0.65
        },
        {
            "id": "viz1",
            "kind": "visualization",
            "text": "Created scatter plot showing marketing spend vs sales with trend line (R² = 0.69)",
            "created_at": current_time - 900,
            "usage_count": 4,
            "base_importance": 0.8,
            "dynamic_importance": 0.85
        },
        {
            "id": "insights1",
            "kind": "insights",
            "text": "Key insight: 20% increase in marketing spend correlates with 16% boost in sales revenue",
            "created_at": current_time - 600,
            "usage_count": 7,
            "base_importance": 0.95,
            "dynamic_importance": 0.98
        },
        {
            "id": "analysis2",
            "kind": "analysis", 
            "text": "Identified seasonal patterns in sales data with Q4 showing 35% higher performance",
            "created_at": current_time - 2400,
            "usage_count": 3,
            "base_importance": 0.85,
            "dynamic_importance": 0.88
        }
    ]
    
    for mem in memories:
        namespace = ("memories", mem["kind"])
        store.put(namespace, mem["id"], mem)
    
    return store

def demo_agent_profiles():
    """Demonstrate different agent profiles in action."""
    print("🔍 ADAPTIVE RETRIEVAL DEMO: Agent Profiles")
    print("=" * 50)
    
    store = create_demo_store()
    
    # Create config file
    config_path = create_demo_config()
    
    try:
        retriever = AdaptiveRetriever(store, config_path, debug=True)
        
        # Test different agents with the same query
        query = "analyze correlation trends in marketing data"
        
        agents_to_test = [
            ("analyst", "Analyst Agent - Comprehensive analysis focus"),
            ("visualization", "Visualization Agent - Chart and graph focus"),
            ("data_cleaner", "Data Cleaner - Preprocessing focus")
        ]
        
        for agent_name, description in agents_to_test:
            print(f"\n📊 {description}")
            print("-" * 40)
            
            result = retriever.get_context(
                agent_name=agent_name,
                query=query,
                base_prompt_tokens=150,
                model_context_window=4096
            )
            
            print(f"Selected memories: {len(result.selected)}")
            print(f"Token usage: {result.token_usage}")
            print(f"Total candidates: {result.total_candidates}")
            print(f"Truncated: {result.truncated}")
            
            if result.selected:
                print("Memory kinds retrieved:")
                kinds = {}
                for mem in result.selected:
                    kinds[mem.kind] = kinds.get(mem.kind, 0) + 1
                for kind, count in kinds.items():
                    print(f"  - {kind}: {count}")
            
            if result.diagnostics and result.diagnostics.get("candidate_table"):
                print("\nTop 3 ranked memories:")
                for i, candidate in enumerate(result.diagnostics["candidate_table"][:3]):
                    print(f"  {i+1}. {candidate['kind']} (score: {candidate['score']:.3f}, tokens: {candidate['est_tokens']})")
    
    finally:
        os.unlink(config_path)

def demo_token_budgets():
    """Demonstrate token budget enforcement."""
    print("\n\n💰 ADAPTIVE RETRIEVAL DEMO: Token Budget Management")
    print("=" * 55)
    
    store = create_demo_store()
    config_path = create_demo_config()
    
    try:
        retriever = AdaptiveRetriever(store, config_path, debug=False)
        
        query = "show me all analysis and insights about sales trends"
        
        # Test with different token budgets
        budgets_to_test = [
            (100, "Very tight budget"),
            (300, "Small budget"),
            (600, "Medium budget"),
            (1200, "Large budget")
        ]
        
        for budget, description in budgets_to_test:
            print(f"\n💸 {description} ({budget} tokens)")
            print("-" * 35)
            
            result = retriever.get_context(
                agent_name="analyst",
                query=query,
                base_prompt_tokens=100,
                task_overrides={"token_budget": budget}
            )
            
            print(f"Memories selected: {len(result.selected)}")
            print(f"Actual token usage: {result.token_usage}")
            print(f"Budget utilization: {(result.token_usage/budget)*100:.1f}%")
            print(f"Truncated due to budget: {result.truncated}")
    
    finally:
        os.unlink(config_path)

def demo_intent_adjustments():
    """Demonstrate query intent-based adjustments."""
    print("\n\n🎯 ADAPTIVE RETRIEVAL DEMO: Query Intent Adjustments")
    print("=" * 55)
    
    store = create_demo_store()
    config_path = create_demo_config()
    
    try:
        retriever = AdaptiveRetriever(store, config_path, debug=False)
        
        queries_to_test = [
            ("hello how are you", "General conversation"),
            ("show me the analysis trends and correlations", "Analytical intent"),
            ("create a chart visualization", "Visualization intent"),
            ("what insights did we find", "Insights intent")
        ]
        
        for query, description in queries_to_test:
            print(f"\n🔍 {description}")
            print(f"Query: '{query}'")
            print("-" * 35)
            
            result = retriever.get_context(
                agent_name="analyst",
                query=query,
                base_prompt_tokens=100
            )
            
            if result.selected:
                kinds = {}
                for mem in result.selected:
                    kinds[mem.kind] = kinds.get(mem.kind, 0) + 1
                
                print("Memory distribution:")
                for kind, count in sorted(kinds.items()):
                    print(f"  - {kind}: {count}")
            else:
                print("No memories selected")
    
    finally:
        os.unlink(config_path)

def demo_diagnostics():
    """Demonstrate comprehensive diagnostics."""
    print("\n\n🔬 ADAPTIVE RETRIEVAL DEMO: Comprehensive Diagnostics")
    print("=" * 55)
    
    store = create_demo_store()
    config_path = create_demo_config()
    
    try:
        # Enable diagnostics
        retriever = AdaptiveRetriever(store, config_path, debug=True)
        
        result = retriever.get_context(
            agent_name="analyst",
            query="correlation analysis trends insights",
            base_prompt_tokens=100
        )
        
        if result.diagnostics:
            diag = result.diagnostics
            print(f"Query: {diag['query']}")
            print(f"Agent Profile: {diag['agent_profile']}")
            print(f"Total Candidates: {diag['total_candidates']}")
            print(f"Selected Count: {diag['selected_count']}")
            print(f"Token Usage: {diag['token_usage']}/{diag['token_budget']}")
            print(f"Budget Utilization: {diag['budget_utilization']:.1%}")
            
            print("\nExclusion Summary:")
            for reason, count in diag['exclusion_summary'].items():
                if count > 0:
                    print(f"  - {reason.replace('_', ' ').title()}: {count}")
            
            print("\nTop 5 Candidates (Ranked):")
            for candidate in diag['candidate_table'][:5]:
                status = "✓ SELECTED" if candidate['packed'] else f"✗ {candidate['exclusion_reason']}"
                print(f"  {candidate['rank']}. {candidate['kind']} | Score: {candidate['score']:.3f} | "
                     f"Tokens: {candidate['est_tokens']} | {status}")
        
        # Show global metrics
        print(f"\nGlobal Metrics:")
        metrics = retriever.get_metrics()
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value > 0:
                print(f"  - {key}: {value}")
    
    finally:
        os.unlink(config_path)

def create_demo_config():
    """Create a demo configuration file."""
    config_content = """
adaptive_retrieval:
  global:
    default_memory_token_budget: 900
    hard_token_budget_fraction: 0.35
    fallback_limit: 5
    enable_diagnostics: true
  agents:
    analyst:
      kinds: ["analysis", "insights", "conversation"]
      min_items: 3
      max_items: 12
      target_items: 8
      token_budget: 800
      weighting_overrides:
        similarity: 0.4
        importance: 0.35
        recency: 0.15
        usage: 0.1
    visualization:
      kinds: ["visualization", "analysis"]
      min_items: 2
      max_items: 8
      target_items: 5
      token_budget: 500
      weighting_overrides:
        similarity: 0.5
        importance: 0.3
        recency: 0.15
        usage: 0.05
    data_cleaner:
      kinds: ["cleaning", "analysis"]
      min_items: 2
      max_items: 6
      target_items: 4
      token_budget: 400
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        return f.name

def main():
    """Run all demonstrations."""
    print("🚀 ADAPTIVE CONTEXT RETRIEVAL DEMONSTRATION")
    print("=" * 60)
    print("This demo shows the key features of the new adaptive")
    print("memory retrieval system with dynamic window sizing.")
    print("=" * 60)
    
    try:
        demo_agent_profiles()
        demo_token_budgets()
        demo_intent_adjustments()
        demo_diagnostics()
        
        print("\n\n✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 40)
        print("Key takeaways:")
        print("• Different agents get optimized memory context")
        print("• Token budgets prevent context window overflow")
        print("• Query intent affects memory selection")
        print("• Comprehensive diagnostics aid debugging")
        print("• Backward compatibility is maintained")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()