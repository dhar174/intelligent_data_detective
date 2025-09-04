#!/usr/bin/env python3
"""
Demonstration of Enhanced Memory Categorization System

This script showcases the structured multi-namespace memory categorization
functionality implemented for the Intelligent Data Detective system.
"""

import sys
import os
from typing import List

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_enhancements import (
    put_memory,
    retrieve_memories,
    format_memories_by_kind,
    enhanced_mem_text,
    MEMORY_CONFIG,
    MemoryKind
)

# Mock store for demonstration
class DemoInMemoryStore:
    """Simple in-memory store for demonstration purposes."""
    
    def __init__(self):
        self.data = {}
        self.operation_count = 0
    
    def put(self, namespace: tuple, memory_id: str, item: dict):
        """Store an item in the demo store."""
        if namespace not in self.data:
            self.data[namespace] = {}
        self.data[namespace][memory_id] = item
        self.operation_count += 1
    
    def search(self, namespace: tuple, query: str, limit: int = 5) -> List[dict]:
        """Search for items in the demo store."""
        self.operation_count += 1
        
        if namespace not in self.data:
            return []
        
        items = []
        for memory_id, item in self.data[namespace].items():
            if isinstance(item, dict):
                text = item.get("text", item.get("memory", ""))
                if any(word.lower() in text.lower() for word in query.split()):
                    items.append(item.copy())
        
        return items[:limit]


def demonstrate_memory_categorization():
    """Demonstrate the enhanced memory categorization system."""
    
    print("=" * 70)
    print("INTELLIGENT DATA DETECTIVE - ENHANCED MEMORY CATEGORIZATION DEMO")
    print("=" * 70)
    print()
    
    # Initialize demo store
    store = DemoInMemoryStore()
    user_id = "demo_user"
    
    print("1. MEMORY CONFIGURATION")
    print("-" * 30)
    print("Available memory categories:")
    for kind, config in MEMORY_CONFIG["kinds"].items():
        print(f"  • {kind.upper()}: limit={config['limit']}")
    print()
    
    print("2. WORKFLOW SIMULATION - STORING CATEGORIZED MEMORIES")
    print("-" * 50)
    
    # Simulate a data analysis workflow
    workflow_memories = [
        {
            "kind": "conversation",
            "content": "User requests analysis of e-commerce sales data to identify customer purchasing patterns",
            "phase": "Initial Request"
        },
        {
            "kind": "analysis", 
            "content": "Dataset contains 50,000 transactions from 2022-2024 with customer demographics and purchase history",
            "phase": "Initial Analysis"
        },
        {
            "kind": "cleaning",
            "content": "Removed 127 duplicate transactions and standardized product category names across 15 categories",
            "phase": "Data Cleaning"
        },
        {
            "kind": "analysis",
            "content": "Discovered strong correlation (r=0.73) between customer age and premium product purchases",
            "phase": "Statistical Analysis"
        },
        {
            "kind": "analysis",
            "content": "Identified seasonal patterns: 45% increase in electronics sales during November-December",
            "phase": "Trend Analysis"
        },
        {
            "kind": "visualization",
            "content": "Created customer segmentation heatmap showing age vs. spending patterns across product categories",
            "phase": "Visualization Creation"
        },
        {
            "kind": "visualization",
            "content": "Generated seasonal sales trends line chart highlighting Q4 electronics surge",
            "phase": "Trend Visualization"
        },
        {
            "kind": "insights",
            "content": "Key finding: Customers aged 35-50 drive 60% of premium electronics revenue during holiday season",
            "phase": "Insight Generation"
        },
        {
            "kind": "insights",
            "content": "Recommendation: Target marketing campaigns for electronics to 35-50 age group in October-November",
            "phase": "Strategic Recommendations"
        }
    ]
    
    # Store each memory with categorization
    for i, memory in enumerate(workflow_memories, 1):
        memory_id = put_memory(
            store,
            memory["kind"],
            memory["content"],
            meta={"phase": memory["phase"], "timestamp": f"2024-01-{i:02d}"},
            user_id=user_id
        )
        print(f"  ✓ Stored {memory['kind'].upper()} memory: {memory['phase']}")
    
    print(f"\n  Total memories stored: {len(workflow_memories)}")
    print(f"  Store operations: {store.operation_count}")
    print()
    
    print("3. AGENT-SPECIFIC MEMORY RETRIEVAL")
    print("-" * 40)
    
    # Demonstrate agent-specific memory retrieval
    agent_scenarios = [
        {
            "agent": "Data Analyst",
            "query": "customer purchasing patterns",
            "kinds": ["conversation", "analysis", "insights"],
            "description": "Needs conversation context, analysis results, and insights"
        },
        {
            "agent": "Visualization Specialist", 
            "query": "seasonal sales trends",
            "kinds": ["analysis", "visualization"],
            "description": "Needs analysis data and previous visualization work"
        },
        {
            "agent": "Report Generator",
            "query": "electronics sales recommendations",
            "kinds": ["conversation", "analysis", "visualization", "insights"],
            "description": "Needs comprehensive context for report writing"
        },
        {
            "agent": "Data Cleaner",
            "query": "data quality issues",
            "kinds": ["conversation", "cleaning", "analysis"],
            "description": "Needs cleaning history and data understanding"
        }
    ]
    
    for scenario in agent_scenarios:
        print(f"\n{scenario['agent']} Query: '{scenario['query']}'")
        print(f"Memory kinds: {scenario['kinds']}")
        print(f"Context: {scenario['description']}")
        
        memories = retrieve_memories(
            store,
            scenario["query"],
            kinds=scenario["kinds"],
            user_id=user_id
        )
        
        print(f"Retrieved {len(memories)} relevant memories:")
        for memory in memories:
            kind = memory.get("namespace_kind", memory.get("kind", "unknown"))
            text = memory.get("text", "")[:60] + "..." if len(memory.get("text", "")) > 60 else memory.get("text", "")
            print(f"  • [{kind.upper()}] {text}")
    
    print()
    print("4. FORMATTED MEMORY CONTEXT FOR PROMPTS")
    print("-" * 45)
    
    # Demonstrate formatted memory context
    print("Example: Report Generator retrieving comprehensive context")
    memories = retrieve_memories(
        store,
        "customer analysis recommendations",
        kinds=["conversation", "analysis", "visualization", "insights"],
        limit=10,
        user_id=user_id
    )
    
    formatted_context = format_memories_by_kind(memories)
    print("\nFormatted memory context for prompt injection:")
    print("-" * 60)
    print(formatted_context)
    print("-" * 60)
    
    print()
    print("5. ENHANCED MEMORY TEXT FUNCTION")
    print("-" * 35)
    
    # Demonstrate enhanced memory text function
    enhanced_output = enhanced_mem_text(
        "electronics sales seasonal trends",
        kinds=["analysis", "visualization", "insights"],
        store=store
    )
    
    print("Enhanced memory text output (grouped by category):")
    print("-" * 50)
    print(enhanced_output)
    print("-" * 50)
    
    print()
    print("6. BACKWARD COMPATIBILITY TEST")
    print("-" * 35)
    
    # Store a memory in legacy format
    legacy_namespace = ("memories",)
    store.put(legacy_namespace, "legacy_001", {
        "memory": "Legacy memory: Previous analysis of quarterly sales showed consistent growth"
    })
    
    # Try to retrieve with categories that don't exist, should fall back to legacy
    fallback_memories = retrieve_memories(
        store,
        "quarterly sales growth",
        kinds=["nonexistent_category"],
        user_id=user_id
    )
    
    if fallback_memories:
        print("✓ Backward compatibility confirmed: Successfully retrieved legacy memory")
        print(f"  Legacy memory: {fallback_memories[0].get('memory', 'N/A')}")
    else:
        print("✗ Backward compatibility test failed")
    
    print()
    print("7. SYSTEM STATISTICS")
    print("-" * 25)
    
    # Show system statistics
    total_namespaces = len(store.data)
    total_memories = sum(len(memories) for memories in store.data.values())
    
    print(f"Total memory namespaces: {total_namespaces}")
    print(f"Total memories stored: {total_memories}")
    print(f"Total store operations: {store.operation_count}")
    
    print("\nNamespace distribution:")
    for namespace, memories in store.data.items():
        namespace_name = namespace[-1] if len(namespace) > 1 else "generic"
        print(f"  • {namespace_name}: {len(memories)} memories")
    
    print()
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Key Benefits Demonstrated:")
    print("• Structured memory categorization by agent role and content type")
    print("• Targeted memory retrieval reducing noise and improving relevance")
    print("• Grouped memory formatting for better prompt context")
    print("• Graceful fallback to legacy memory formats")
    print("• Configurable limits per memory category")
    print("• Full backward compatibility with existing system")
    print()


if __name__ == "__main__":
    demonstrate_memory_categorization()