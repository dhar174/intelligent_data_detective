#!/usr/bin/env python3
"""
Memory Lifecycle Management Demo

This script demonstrates the key features of the memory lifecycle management
system including policy-driven storage, enhanced retrieval ranking, duplicate
detection, importance scoring, and pruning operations.
"""

import time
import os
from memory_enhancements import (
    MemoryRecord,
    MemoryPolicyEngine,
    put_memory_with_policy,
    retrieve_memories_with_ranking,
    prune_memories,
    recalculate_importance,
    get_memory_metrics,
    reset_memory_metrics,
    memory_policy_report,
    estimate_importance
)

# Mock store for demonstration
class DemoInMemoryStore:
    def __init__(self):
        self.data = {}
        self.search_calls = []
    
    def put(self, namespace: tuple, memory_id: str, item: dict):
        if namespace not in self.data:
            self.data[namespace] = {}
        self.data[namespace][memory_id] = item.copy()
    
    def search(self, namespace: tuple, query: str, limit: int = 5):
        self.search_calls.append((namespace, query, limit))
        if namespace not in self.data:
            return []
        
        items = []
        for memory_id, item in self.data[namespace].items():
            if isinstance(item, dict):
                text = item.get("text", "")
                if not query or any(word.lower() in text.lower() for word in query.split()):
                    item_copy = item.copy()
                    item_copy["id"] = memory_id
                    items.append(item_copy)
        
        return items[:limit]


def demo_memory_lifecycle():
    """Demonstrate memory lifecycle management features."""
    print("🧠 Memory Lifecycle Management Demo")
    print("=" * 50)
    
    # Setup
    store = DemoInMemoryStore()
    reset_memory_metrics()
    os.environ["DEBUG_MEMORY"] = "true"
    
    engine = MemoryPolicyEngine(store, debug=True)
    
    print("\n1. 📊 Importance Scoring Demo")
    print("-" * 30)
    
    # Test importance estimation for different content types
    test_contents = [
        ("conversation", "Hello, how are you today?"),
        ("analysis", "Found significant correlation coefficient of 0.85 between variables. Key insight: strong positive relationship."),
        ("insights", "Critical finding: anomaly detected in data pattern. Immediate analysis required."),
        ("cleaning", "Removed 150 duplicate rows and filled missing values using median imputation."),
        ("visualization", "Created scatter plot showing relationship between age and income."),
        ("errors", "Error: Division by zero encountered in calculation. Warning: data quality issue.")
    ]
    
    for kind, text in test_contents:
        importance = estimate_importance(kind, text)
        print(f"  {kind:>12}: {importance:.3f} - {text[:60]}...")
    
    print("\n2. 💾 Policy-Driven Storage Demo")
    print("-" * 30)
    
    # Insert various memory types with automatic importance scoring
    memory_contents = [
        ("analysis", "Statistical analysis reveals strong correlation between features A and B. Pearson coefficient: 0.92. Significant at p<0.001."),
        ("insights", "Key insight: Customer behavior patterns show seasonal trends. Peak activity in Q4. Critical for planning."),
        ("conversation", "User asked about data visualization options for time series data."),
        ("cleaning", "Data preprocessing complete: normalized numerical features, encoded categorical variables."),
        ("analysis", "Regression model shows R² = 0.87. Important predictors: age, income, education level."),
        ("visualization", "Generated heatmap showing correlation matrix. Strong patterns visible in feature relationships."),
        ("conversation", "User requested export of analysis results in CSV format."),
        ("errors", "Warning: 15% missing values detected in income column. Recommend imputation strategy."),
        ("insights", "Business recommendation: Focus marketing on demographic segments with highest predicted value."),
        ("analysis", "Time series analysis indicates upward trend with seasonal component. ARIMA(2,1,2) model selected.")
    ]
    
    inserted_ids = []
    for kind, text in memory_contents:
        memory_id = put_memory_with_policy(store, kind, text)
        inserted_ids.append(memory_id)
        print(f"  ✅ Stored {kind}: {memory_id[:8]}")
    
    print(f"\n📈 Storage Metrics:")
    metrics = get_memory_metrics()
    print(f"  Total items stored: {metrics['memory_items_total']}")
    print(f"  Items by kind: {metrics['memory_items_by_kind']}")
    
    print("\n3. 🔍 Enhanced Retrieval Demo")
    print("-" * 30)
    
    # Test enhanced retrieval with ranking
    queries = [
        ("correlation analysis", ["analysis", "insights"]),
        ("user questions", ["conversation"]),
        ("data quality", ["cleaning", "errors"]),
        ("business insights", ["insights", "analysis"])
    ]
    
    for query, kinds in queries:
        print(f"\n  Query: '{query}' in {kinds}")
        results = retrieve_memories_with_ranking(store, query, kinds=kinds, limit=3)
        
        for i, result in enumerate(results, 1):
            importance = result.get('dynamic_importance', 0)
            usage = result.get('usage_count', 0)
            print(f"    {i}. [imp:{importance:.2f}, usage:{usage}] {result['text'][:80]}...")
    
    print("\n4. 🔄 Usage Tracking Demo")
    print("-" * 30)
    
    # Simulate multiple retrievals to show usage tracking
    print("  Simulating repeated access to analysis memories...")
    
    for _ in range(3):
        results = retrieve_memories_with_ranking(store, "correlation", kinds=["analysis"], limit=2)
        time.sleep(0.1)  # Small delay for timestamp differences
    
    # Show updated usage counts
    final_results = retrieve_memories_with_ranking(store, "correlation", kinds=["analysis"], limit=2)
    print("  Updated usage counts:")
    for result in final_results:
        usage = result.get('usage_count', 0)
        last_used = result.get('last_used_at', 0)
        print(f"    Usage: {usage}, Last used: {time.ctime(last_used) if last_used else 'Never'}")
    
    print("\n5. 🧹 Pruning Demo")
    print("-" * 30)
    
    # Insert some old records for TTL demonstration
    old_time = time.time() - 86400 * 8  # 8 days ago
    
    old_record = MemoryRecord(
        id="old-record",
        kind="conversation",
        text="Old conversation from last week",
        created_at=old_time,
        base_importance=0.3
    )
    engine.insert(old_record)
    print("  📅 Inserted old record for TTL demonstration")
    
    # Run pruning
    print("  🧹 Running memory pruning...")
    report = prune_memories(store, reason="demo_cleanup")
    
    print(f"  Pruning Results:")
    print(f"    Expired items: {report.expired_count}")
    print(f"    Size-pruned items: {report.size_pruned_count}")
    print(f"    Low-importance items: {report.low_importance_count}")
    print(f"    Total pruned: {report.total_pruned}")
    print(f"    Remaining items: {report.remaining_count}")
    
    print("\n6. 🔄 Importance Recalculation Demo")
    print("-" * 30)
    
    # Show importance updates
    print("  Recalculating dynamic importance scores...")
    updated_count = recalculate_importance(store, ["analysis", "insights"])
    print(f"  Updated {updated_count} records")
    
    # Show updated scores
    results = retrieve_memories_with_ranking(store, "analysis", kinds=["analysis"], limit=3)
    print("  Updated importance scores:")
    for result in results:
        base_imp = result.get('base_importance', 0)
        dynamic_imp = result.get('dynamic_importance', 0)
        usage = result.get('usage_count', 0)
        print(f"    Base: {base_imp:.3f} → Dynamic: {dynamic_imp:.3f} (usage: {usage})")
    
    print("\n7. 📊 Policy Compliance Report")
    print("-" * 30)
    
    # Generate diagnostic report
    report = memory_policy_report(store)
    
    print("  Memory Status by Kind:")
    for kind, status in report.get('kind_status', {}).items():
        if isinstance(status, dict) and 'current_count' in status:
            count = status['current_count']
            max_allowed = status['max_allowed']
            compliance = "✅" if status.get('compliance', False) else "⚠️"
            print(f"    {kind:>12}: {count:>3}/{max_allowed:<3} items {compliance}")
    
    if report.get('recommendations'):
        print("  Recommendations:")
        for rec in report['recommendations']:
            print(f"    💡 {rec}")
    
    print("\n8. 📊 Final Metrics Summary")
    print("-" * 30)
    
    final_metrics = get_memory_metrics()
    print("  Operation Counters:")
    print(f"    Memory insertions: {final_metrics['memory_put_total']}")
    print(f"    Retrieval requests: {final_metrics['memory_retrieval_requests_total']}")
    print(f"    Prune operations: {final_metrics['memory_prune_runs_total']}")
    print(f"    Items pruned: {final_metrics['memory_pruned_items_total']}")
    print(f"    Duplicates dropped: {final_metrics['memory_duplicate_dropped_total']}")
    print(f"    Degraded memories: {final_metrics['memory_degraded_total']}")
    
    print(f"\n  Current Storage:")
    print(f"    Total items: {final_metrics['memory_items_total']}")
    for kind, count in final_metrics['memory_items_by_kind'].items():
        print(f"    {kind:>12}: {count} items")
    
    print("\n🎉 Demo Complete!")
    print("The memory lifecycle management system provides:")
    print("  ✅ Policy-driven TTL and size management")
    print("  ✅ Intelligent importance scoring and ranking")
    print("  ✅ Usage tracking and dynamic score updates")
    print("  ✅ Duplicate detection and suppression")
    print("  ✅ Multi-strategy pruning (time, size, importance)")
    print("  ✅ Comprehensive metrics and diagnostics")
    print("  ✅ Full backward compatibility")


if __name__ == "__main__":
    demo_memory_lifecycle()