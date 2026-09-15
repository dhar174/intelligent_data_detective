# Memory Lifecycle Management Documentation

## Overview

The Memory Lifecycle Management system provides comprehensive policy-driven memory management for the Intelligent Data Detective, including TTL expiration, relevance scoring, duplicate detection, and intelligent pruning strategies.

## Key Features

### Policy-Driven Configuration
- **TTL Management**: Configurable time-to-live per memory kind
- **Size Limits**: Maximum item counts with intelligent pruning
- **Importance Thresholds**: Minimum relevance scores for retention
- **Ranking Weights**: Customizable retrieval ranking factors

### Dynamic Importance Scoring
- **Base Importance**: Content analysis with keyword detection and role weighting
- **Dynamic Updates**: Usage frequency and recency decay factors
- **Automatic Recomputation**: Importance scores update based on access patterns

### Intelligent Pruning
- **Time-based**: Remove expired entries based on TTL policies
- **Size-based**: Keep highest-value items when limits exceeded  
- **Importance-based**: Remove low-relevance entries below thresholds
- **Superseded Cleanup**: Remove outdated duplicates with grace periods

### Enhanced Retrieval
- **Multi-factor Ranking**: Weighted combination of similarity, importance, recency, usage
- **Usage Tracking**: Access counts and timestamps for relevance boosting
- **Degraded Support**: Fallback for memories without embeddings

## Configuration

### Memory Policy (memory_config.yaml)

```yaml
memory_policy:
  defaults:
    ttl_seconds: 604800          # 7 days default TTL
    max_items: 1500              # Global maximum
    max_items_per_kind: 400      # Per-kind maximum
    min_importance: 0.05         # Minimum relevance threshold
    decay:
      half_life_seconds: 259200   # 3 days recency decay
      floor: 0.05                # Minimum decay value

  kinds:
    conversation:
      ttl_seconds: 259200         # 3 days (shorter for chat)
      max_items: 600
    analysis:
      ttl_seconds: 1209600        # 14 days (longer for insights)
      max_items: 500
    insights:
      ttl_seconds: 1814400        # 21 days (longest retention)
      max_items: 600

ranking:
  weights:
    similarity: 0.55    # Content similarity weight
    importance: 0.25    # Dynamic importance weight
    recency: 0.15       # Time-based recency weight
    usage: 0.05         # Access frequency weight
```

### Per-Kind Policies

Different memory types have tailored policies:

| Kind | TTL | Max Items | Description |
|------|-----|-----------|-------------|
| `conversation` | 3 days | 600 | User interactions, shorter-lived |
| `analysis` | 14 days | 500 | Statistical analysis results |
| `cleaning` | 14 days | 400 | Data preprocessing steps |
| `visualization` | 7 days | 300 | Chart and graph metadata |
| `insights` | 21 days | 600 | Key findings, longest retention |
| `errors` | 7 days | 200 | Error patterns and diagnostics |

## API Reference

### Core Classes

#### MemoryRecord
Enhanced memory record with lifecycle metadata:

```python
@dataclass
class MemoryRecord:
    id: str                           # Unique identifier
    kind: str                         # Memory category
    text: str                         # Content text
    vector: Optional[List[float]]     # Embedding vector
    created_at: float                 # Creation timestamp
    last_used_at: Optional[float]     # Last access time
    usage_count: int                  # Access frequency
    base_importance: float            # Initial relevance score
    dynamic_importance: float         # Updated relevance score
    degraded: bool                    # Embedding failure flag
    superseded_by: Optional[str]      # Replacement record ID
    meta: Dict[str, Any]              # Additional metadata
    user_id: str                      # User namespace
```

#### MemoryPolicyEngine
Core engine for lifecycle management:

```python
class MemoryPolicyEngine:
    def insert(self, record: MemoryRecord) -> MemoryRecord
    def retrieve(self, query: str, kinds: List[str], limit: int) -> List[MemoryRecord]
    def prune(self, reason: Optional[str] = None) -> PruneReport
    def recalc_importance(self, records: List[MemoryRecord])
```

### Enhanced Functions

#### Memory Storage
```python
# Policy-enabled storage with importance scoring
memory_id = put_memory_with_policy(
    store=in_memory_store,
    kind="analysis",
    text="Found significant correlation between variables",
    meta={"confidence": 0.95}
)
```

#### Enhanced Retrieval
```python
# Multi-factor ranked retrieval
memories = retrieve_memories_with_ranking(
    store=in_memory_store,
    query="correlation analysis",
    kinds=["analysis", "insights"],
    limit=5
)
```

#### Manual Operations
```python
# Trigger pruning
report = prune_memories(store, reason="maintenance")

# Recalculate importance scores
updated_count = recalculate_importance(store, kinds=["analysis"])

# Get diagnostic report
report = memory_policy_report(store)
```

## Importance Scoring Algorithm

### Base Importance Calculation
1. **Length Score**: `min(1.0, token_count / 100)`
2. **Keyword Score**: Presence of analytical keywords (insight, correlation, etc.)
3. **Role Weight**: Memory kind importance multiplier

```python
# Role weights by kind
role_weights = {
    "analysis": 0.9,      # High value analytical content
    "insights": 0.95,     # Highest value conclusions  
    "cleaning": 0.7,      # Moderate value preprocessing
    "visualization": 0.6,  # Lower value display info
    "conversation": 0.5,   # Baseline user interactions
    "errors": 0.8         # Important diagnostic info
}

base_importance = (length_score * 0.3 + 
                  keyword_score * 0.4 + 
                  role_weight * 0.3)
```

### Dynamic Importance Updates
Scores are updated based on usage patterns:

```python
# Recency decay factor
recency_factor = exp(-age / half_life_seconds)

# Usage boost factor  
usage_factor = 1.0 + log(1 + usage_count) * 0.15

# Updated dynamic score
dynamic_importance = base_importance * recency_factor * usage_factor
```

## Retrieval Ranking

Multi-factor weighted scoring for retrieval results:

```python
final_score = (
    0.55 * content_similarity +      # Text similarity to query
    0.25 * dynamic_importance +      # Computed relevance score
    0.15 * recency_factor +          # Time-based decay
    0.05 * usage_factor              # Access frequency boost
)
```

## Pruning Strategies

### Time-Based Pruning
- Remove memories older than TTL for their kind
- Grace period for superseded items (24 hours)

### Size-Based Pruning  
- Triggered when kind exceeds max_items limit
- Keep highest-scoring items by keep_score formula:
  ```python
  keep_score = importance * recency_factor * (1 + sqrt(usage_count))
  ```

### Importance-Based Pruning
- Remove items below min_importance threshold
- Applied after decay and usage updates

## Metrics and Monitoring

### Available Metrics
- `memory_items_total`: Total stored items
- `memory_items_by_kind`: Per-kind item counts
- `memory_put_total`: Insertion counter
- `memory_prune_runs_total`: Pruning operation counter
- `memory_pruned_items_total`: Items removed counter
- `memory_duplicate_dropped_total`: Duplicates prevented
- `memory_retrieval_requests_total`: Query counter

### Diagnostic Functions
```python
# Get current metrics
metrics = get_memory_metrics()

# Generate policy compliance report
report = memory_policy_report(store)

# Reset counters
reset_memory_metrics()
```

## Debug and Troubleshooting

### Debug Mode
Enable detailed logging:
```bash
export DEBUG_MEMORY=true
```

Debug output includes:
- Top candidate scores for each retrieval
- Pruning operation summaries  
- Duplicate detection decisions
- Importance score calculations

### Common Issues

#### High Memory Usage
- Check metrics for excessive item counts
- Review TTL settings for aggressive retention
- Consider lowering max_items limits
- Run manual pruning: `prune_memories(store)`

#### Poor Retrieval Relevance
- Adjust ranking weights in config
- Recalculate importance scores
- Review importance thresholds
- Check for stale/unused memories

#### Duplicate Content
- Lower similarity threshold (currently 0.96)
- Enable superseded cleanup
- Manual deduplication may be needed

## Migration and Compatibility

### Backward Compatibility
- All existing memory functions preserved
- New features opt-in via parameters
- Graceful fallback on errors
- Legacy namespace support maintained

### Migration Path
1. Update configuration with desired policies
2. Enable policy engine: `use_policy_engine=True`
3. Run importance recalculation
4. Monitor metrics and adjust policies
5. Gradually tighten limits and thresholds

## Future Enhancements

### Planned Features
- ML-based importance learning
- Advanced duplicate clustering
- Persistent backend integration
- Federated multi-user memory
- Real-time policy updates
- Advanced analytics dashboard

### Extension Points
- Custom importance heuristics
- Pluggable similarity metrics
- Alternative pruning strategies
- External embedding services
- Custom ranking algorithms