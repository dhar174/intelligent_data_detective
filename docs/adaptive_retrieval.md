# Adaptive Context Retrieval Documentation

## Overview

The Adaptive Context Retrieval system provides dynamic memory window sizing with per-agent profiles, multi-factor ranking, and token-aware packing for optimal context utilization in the Intelligent Data Detective workflow.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Agent Profiles](#agent-profiles)
4. [Configuration](#configuration)
5. [Integration Guide](#integration-guide)
6. [Diagnostics and Debugging](#diagnostics-and-debugging)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)
9. [Migration Guide](#migration-guide)

## Quick Start

### Enable Adaptive Retrieval

```bash
# Set environment variable to enable adaptive retrieval
export ADAPTIVE_RETRIEVAL_ENABLED=true
```

### Basic Usage

```python
from adaptive_integration import adaptive_retrieve_mem

# In your agent node function
def analyst_node(state):
    # Replace the old retrieve_mem function
    memories = adaptive_retrieve_mem(
        state=state,
        agent_name="analyst",
        base_prompt_tokens=150,  # Token count of your base prompt
        model_context_window=4096  # Your model's context window
    )
    
    # Use memories as before
    formatted_memories = format_memories_by_kind(memories)
```

### Agent-Specific Functions

```python
from adaptive_integration import (
    analyst_retrieve_mem,
    visualization_retrieve_mem,
    data_cleaner_retrieve_mem
)

# Drop-in replacements for existing retrieve_mem functions
memories = analyst_retrieve_mem(state)
viz_memories = visualization_retrieve_mem(state)
cleaning_memories = data_cleaner_retrieve_mem(state)
```

## Core Concepts

### 1. Per-Agent Profiles

Each agent type has a customized memory retrieval profile:

- **Memory Kinds**: Specific types of memories relevant to the agent
- **Item Limits**: Minimum, maximum, and target number of memories
- **Token Budget**: Maximum tokens allocated for memory context
- **Ranking Weights**: Custom weights for similarity, importance, recency, and usage

### 2. Multi-Factor Ranking

Memories are ranked using a composite score:

```
score = w_sim * similarity 
      + w_imp * dynamic_importance
      + w_rec * recency_factor
      + w_use * usage_factor
```

Where:
- **Similarity**: Query-memory text similarity (0-1)
- **Dynamic Importance**: Content quality with usage boost (0-1)
- **Recency Factor**: Exponential decay based on age
- **Usage Factor**: Logarithmic scaling of access count

### 3. Token-Aware Packing

The system respects token budgets through:

- **Greedy Packing**: Add memories until budget exhausted
- **Minimum Items**: Always include min_items even if over budget
- **Context Window Protection**: Ensure total prompt fits in model limits
- **Overflow Trimming**: Remove least important memories if needed

### 4. Query Intent Adjustments

Analytical queries automatically boost relevant memory types:

- **Analytical Keywords**: "trend", "correlation", "outlier", "pattern", etc.
- **Intent Boost**: 15% score increase for analysis/insights memories
- **Dynamic Weighting**: Temporary adjustment based on query content

## Agent Profiles

### Initial Analysis Agent
```yaml
initial_analysis:
  kinds: ["conversation", "analysis"]
  min_items: 3
  max_items: 10
  target_items: 7
  token_budget: 600
  weighting_overrides:
    similarity: 0.6
    importance: 0.2
    recency: 0.15
    usage: 0.05
```

**Purpose**: Understands user requirements and leverages previous analysis insights.

### Data Cleaner Agent
```yaml
data_cleaner:
  kinds: ["cleaning", "analysis", "conversation"]
  min_items: 2
  max_items: 8
  target_items: 5
  token_budget: 450
```

**Purpose**: Remembers cleaning requirements and previous preprocessing steps.

### Analyst Agent
```yaml
analyst:
  kinds: ["analysis", "cleaning", "conversation"]
  min_items: 5
  max_items: 18
  target_items: 12
  token_budget: 1000
  weighting_overrides:
    similarity: 0.4
    importance: 0.35
    recency: 0.15
    usage: 0.1
```

**Purpose**: Comprehensive analytical context with emphasis on importance and usage.

### Visualization Agent
```yaml
visualization:
  kinds: ["visualization", "analysis"]
  min_items: 3
  max_items: 12
  target_items: 8
  token_budget: 650
```

**Purpose**: Creates relevant visualizations based on analysis and previous charts.

### Report Orchestrator
```yaml
report_orchestrator:
  kinds: ["analysis", "visualization", "cleaning"]
  min_items: 4
  max_items: 15
  target_items: 10
  token_budget: 800
```

**Purpose**: Plans comprehensive reports using all analytical context.

### File Writer Agent
```yaml
file_writer:
  kinds: ["analysis", "visualization"]
  min_items: 1
  max_items: 6
  target_items: 4
  token_budget: 300
```

**Purpose**: Minimal but targeted context for file generation tasks.

## Configuration

### Global Configuration

```yaml
adaptive_retrieval:
  global:
    default_memory_token_budget: 900        # Default budget if not specified
    hard_token_budget_fraction: 0.35        # Max 35% of context window
    fallback_limit: 5                       # Legacy fallback limit
    intent_boost_factor: 0.15               # Query intent boost percentage
    enable_intent_adjustments: true         # Enable/disable intent detection
    enable_diagnostics: false               # Global diagnostics flag
```

### Custom Agent Profile

```yaml
adaptive_retrieval:
  agents:
    custom_agent:
      kinds: ["conversation", "analysis", "insights"]
      min_items: 2                          # Always include at least 2
      max_items: 10                         # Never exceed 10
      target_items: 6                       # Aim for 6 items
      token_budget: 500                     # Max 500 tokens for memory
      weighting_overrides:                  # Custom ranking weights
        similarity: 0.5
        importance: 0.3
        recency: 0.15
        usage: 0.05
      phase_overrides:                      # Task phase adjustments
        analysis:
          token_budget: 700                 # More budget during analysis
          max_items: 12
```

### Task Phase Overrides

```python
from adaptive_retrieval import TaskPhase

# Use phase-specific settings
result = retriever.get_context(
    agent_name="analyst",
    query="analyze trends",
    task_phase=TaskPhase.ANALYSIS  # Triggers phase overrides
)
```

## Integration Guide

### Method 1: Direct Integration

Replace existing `retrieve_mem` functions:

```python
# Before (legacy)
def retrieve_mem(state):
    store = get_store()
    return store.search(("memories",), query=state.get("user_prompt"), limit=5)

# After (adaptive)
def retrieve_mem(state):
    from adaptive_integration import adaptive_retrieve_mem
    return adaptive_retrieve_mem(state, agent_name="your_agent_name")
```

### Method 2: Agent-Specific Functions

Use pre-configured agent functions:

```python
from adaptive_integration import analyst_retrieve_mem

def analyst_node(state):
    # This automatically uses the analyst profile
    memories = analyst_retrieve_mem(state)
    # Rest of your function remains the same
```

### Method 3: Custom Integration

For advanced use cases:

```python
from adaptive_integration import get_adaptive_retriever

def custom_retrieve(state, agent_name, task_type=None):
    retriever = get_adaptive_retriever()
    if retriever:
        result = retriever.get_context(
            agent_name=agent_name,
            query=state.get("user_prompt"),
            base_prompt_tokens=count_tokens(base_prompt),
            task_overrides={"token_budget": 800} if task_type == "complex" else None
        )
        return convert_to_legacy_format(result.selected)
    else:
        # Fallback to legacy retrieval
        return legacy_retrieve_mem(state)
```

## Diagnostics and Debugging

### Enable Diagnostics

```bash
# Environment variable
export ADAPTIVE_RETRIEVAL_ENABLED=true

# Or in configuration
adaptive_retrieval:
  global:
    enable_diagnostics: true
```

### Diagnostic Information

```python
from adaptive_integration import get_adaptive_retrieval_diagnostics

# Get comprehensive diagnostics
diagnostics = get_adaptive_retrieval_diagnostics()

print(f"Adaptive retrieval enabled: {diagnostics['enabled']}")
print(f"Active retriever: {diagnostics['active']}")
print(f"Total requests: {diagnostics['metrics']['adaptive_memory_requests_total']}")
```

### Detailed Query Diagnostics

```python
retriever = get_adaptive_retriever()
result = retriever.get_context(
    agent_name="analyst",
    query="your query here",
    base_prompt_tokens=100
)

if result.diagnostics:
    diag = result.diagnostics
    print(f"Total candidates: {diag['total_candidates']}")
    print(f"Selected: {diag['selected_count']}")
    print(f"Token usage: {diag['token_usage']}/{diag['token_budget']}")
    
    # Ranked candidate table
    for candidate in diag['candidate_table'][:5]:
        print(f"Rank {candidate['rank']}: {candidate['kind']} "
              f"(score: {candidate['score']:.3f}, "
              f"packed: {candidate['packed']})")
```

### Common Diagnostic Patterns

**Memory Not Found**:
```python
if result.total_candidates == 0:
    print("No memories found - check memory kinds and store content")
```

**Budget Exceeded**:
```python
if result.truncated and result.diagnostics:
    over_budget = result.diagnostics['exclusion_summary']['over_budget']
    print(f"{over_budget} memories excluded due to budget constraints")
```

**Score Threshold Issues**:
```python
if len(result.selected) < profile.min_items:
    print("Consider adjusting similarity threshold or ranking weights")
```

## Performance Tuning

### Token Budget Optimization

1. **Monitor Usage**:
```python
diagnostics = get_adaptive_retrieval_diagnostics()
avg_usage = diagnostics['metrics']['adaptive_memory_token_usage']
avg_requests = diagnostics['metrics']['adaptive_memory_requests_total']
print(f"Average tokens per request: {avg_usage / avg_requests:.1f}")
```

2. **Adjust Budgets**:
```yaml
# Start with observed usage + 20% margin
analyst:
  token_budget: 650  # If average usage is ~540 tokens
```

### Ranking Weight Tuning

Based on your use case:

**High Similarity Priority** (exact matches important):
```yaml
weighting_overrides:
  similarity: 0.7
  importance: 0.15
  recency: 0.1
  usage: 0.05
```

**High Importance Priority** (quality over relevance):
```yaml
weighting_overrides:
  similarity: 0.3
  importance: 0.5
  recency: 0.15
  usage: 0.05
```

**Recency Focus** (fresh information priority):
```yaml
weighting_overrides:
  similarity: 0.4
  importance: 0.25
  recency: 0.3
  usage: 0.05
```

### Candidate Pool Optimization

Adjust the expansion factor in your configuration:

```python
# In adaptive_retrieval.py
expansion_factor = 3  # Get 3x max_items candidates for ranking

# For better quality but slower performance:
expansion_factor = 5

# For faster but potentially lower quality:
expansion_factor = 2
```

## Troubleshooting

### Common Issues

**1. No Memories Retrieved**

```python
# Debug steps:
# 1. Check if adaptive retrieval is enabled
print(f"Enabled: {is_adaptive_retrieval_enabled()}")

# 2. Verify agent profile exists
retriever = get_adaptive_retriever()
if retriever:
    profile = retriever.agent_profiles.get("your_agent")
    print(f"Profile found: {profile is not None}")

# 3. Check memory store content
store = get_store()
items = store.search(("memories", "conversation"), "", limit=10)
print(f"Total conversation memories: {len(items)}")

# 4. Verify memory kinds match profile
profile_kinds = retriever.agent_profiles["your_agent"].kinds
print(f"Agent expects kinds: {profile_kinds}")
```

**2. Token Budget Issues**

```python
# Check budget configuration
diagnostics = get_adaptive_retrieval_diagnostics()
profile_config = diagnostics['config']['agent_profiles']['analyst']
print(f"Token budget: {profile_config['token_budget']}")

# Monitor actual usage
result = retriever.get_context(agent_name="analyst", query="test")
print(f"Actual usage: {result.token_usage}")
print(f"Truncated: {result.truncated}")
```

**3. Performance Issues**

```python
# Check candidate pool size
metrics = retriever.get_metrics()
avg_candidates = metrics['adaptive_memory_candidates_avg']
print(f"Average candidates per request: {avg_candidates}")

# If too high, reduce expansion factor or limit memory store size
```

**4. Ranking Quality Issues**

```python
# Enable diagnostics to see ranking
retriever = AdaptiveRetriever(store, config_path, debug=True)
result = retriever.get_context(agent_name="analyst", query="test")

# Check score distribution
if result.diagnostics:
    scores = [c['score'] for c in result.diagnostics['candidate_table']]
    print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
    
    # Look for score clustering (may indicate poor discrimination)
    unique_scores = len(set(scores))
    print(f"Score diversity: {unique_scores}/{len(scores)}")
```

### Error Recovery

**Configuration Errors**:
```python
# System falls back to default profiles on config errors
# Check logs for configuration warnings
import logging
logging.basicConfig(level=logging.WARNING)
```

**Memory Store Errors**:
```python
# System falls back to legacy retrieval on store errors
# Verify store connectivity and memory content
```

**Token Estimation Errors**:
```python
# System uses fallback estimation on tokenizer errors
# Monitor token usage vs actual consumption
```

## Migration Guide

### Phase 1: Enable and Test

1. Set environment variable:
```bash
export ADAPTIVE_RETRIEVAL_ENABLED=true
```

2. Test with existing code (should work unchanged)

3. Monitor diagnostics:
```python
diagnostics = get_adaptive_retrieval_diagnostics()
print(f"Requests handled: {diagnostics['metrics']['adaptive_memory_requests_total']}")
print(f"Fallbacks used: {diagnostics['metrics']['adaptive_memory_fallback_used_total']}")
```

### Phase 2: Gradual Agent Migration

1. Start with one agent type:
```python
# Replace in analyst_node function
from adaptive_integration import analyst_retrieve_mem
memories = analyst_retrieve_mem(state)
```

2. Test thoroughly with representative workloads

3. Monitor token usage and quality

4. Migrate remaining agents one by one

### Phase 3: Optimization

1. Tune agent profiles based on observed usage
2. Adjust token budgets and ranking weights
3. Enable diagnostics in production for monitoring
4. Create custom profiles for specialized workflows

### Phase 4: Full Adoption

1. Remove legacy retrieve_mem functions
2. Disable fallback mechanisms if desired
3. Create monitoring dashboards for metrics
4. Document custom configurations for your team

## Best Practices

### Configuration Management

1. **Version Control**: Keep `adaptive_memory_config.yaml` in version control
2. **Environment-Specific**: Use different configs for dev/staging/prod
3. **Documentation**: Document any custom profiles or overrides

### Monitoring

1. **Regular Review**: Check diagnostics weekly during initial deployment
2. **Performance Baselines**: Establish token usage and retrieval quality baselines
3. **Alert Thresholds**: Set up alerts for high fallback usage or budget overruns

### Development

1. **Enable Diagnostics**: Always use diagnostics during development
2. **Test Edge Cases**: Test with empty stores, large budgets, and complex queries
3. **Backward Compatibility**: Keep legacy fallback enabled during migration

### Production

1. **Gradual Rollout**: Enable for subset of agents initially
2. **Monitor Quality**: Compare retrieval quality with legacy system
3. **Performance Impact**: Monitor overall system performance impact
4. **User Experience**: Ensure response quality meets user expectations

## API Reference

### Core Classes

**AdaptiveRetriever**
- `get_context(agent_name, query, base_prompt_tokens, ...)` → RetrievalResult
- `get_metrics()` → Dict[str, Any]
- `reset_metrics()` → None

**RetrievalResult**
- `selected: List[MemoryRecord]` - Selected memories
- `total_candidates: int` - Total candidates considered
- `truncated: bool` - Whether results were truncated
- `token_usage: int` - Actual token usage
- `diagnostics: Optional[Dict]` - Diagnostic information
- `profile_used: Optional[str]` - Agent profile used
- `fallback_used: bool` - Whether fallback was used

### Integration Functions

**adaptive_retrieve_mem(state, agent_name, ...)**
Main integration function with legacy compatibility.

**Agent-specific functions:**
- `analyst_retrieve_mem(state, **kwargs)`
- `visualization_retrieve_mem(state, **kwargs)`
- `data_cleaner_retrieve_mem(state, **kwargs)`
- `report_orchestrator_retrieve_mem(state, **kwargs)`
- `file_writer_retrieve_mem(state, **kwargs)`

### Utility Functions

**get_adaptive_retrieval_diagnostics()**
Get comprehensive system diagnostics.

**reset_adaptive_retrieval_metrics()**
Reset global metrics counters.

**create_legacy_retrieve_mem(agent_name, kinds)**
Create legacy-compatible retrieve_mem function.

---

For additional support or questions, refer to the test files (`test_adaptive_retrieval.py`) for usage examples or run the demo script (`demo_adaptive_retrieval.py`) to see the system in action.