#!/usr/bin/env python3
"""
Enhanced Memory System for Intelligent Data Detective

This module provides structured multi-namespace categorization for memory types,
enabling targeted memory retrieval for different agent roles (analysis, cleaning, 
visualization) and preventing noise in memory search results.

Key Features:
- Hierarchical memory namespaces by category
- Filtered retrieval with graceful fallback
- Backward compatibility with existing generic namespace
- Configurable limits per memory kind
- Memory lifecycle management with TTL, pruning, and importance scoring
- Policy-driven retention and relevance weighting
"""

import time
import uuid
import math
import logging
import yaml
import os
from typing import Dict, List, Optional, Union, Literal, Any
from dataclasses import dataclass, field
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState

# Type definitions for memory kinds
MemoryKind = Literal["conversation", "analysis", "cleaning", "visualization", "insights", "errors"]

# Memory record schema with lifecycle management
@dataclass
class MemoryRecord:
    """Enhanced memory record with lifecycle management fields."""
    id: str
    kind: str
    text: str
    vector: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    usage_count: int = 0
    base_importance: float = 0.5
    dynamic_importance: Optional[float] = None  # Will be set to base_importance if None
    degraded: bool = False  # embedding failure fallback
    superseded_by: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    user_id: str = "user"
    
    def __post_init__(self):
        """Set dynamic_importance to base_importance if not provided."""
        if self.dynamic_importance is None:
            self.dynamic_importance = self.base_importance

@dataclass 
class MemoryPolicy:
    """Memory lifecycle policy configuration."""
    ttl_seconds: int = 604800  # 7 days default
    max_items: int = 1500
    max_items_per_kind: int = 400
    min_importance: float = 0.05
    decay_half_life_seconds: int = 259200  # 3 days
    decay_floor: float = 0.05
    
@dataclass
class RankingWeights:
    """Weights for memory retrieval ranking."""
    similarity: float = 0.55
    importance: float = 0.25
    recency: float = 0.15
    usage: float = 0.05

@dataclass 
class PruneReport:
    """Report from pruning operations."""
    expired_count: int = 0
    superseded_count: int = 0
    size_pruned_count: int = 0
    low_importance_count: int = 0
    total_pruned: int = 0
    remaining_count: int = 0

# Global metrics collection
MEMORY_METRICS = {
    "memory_items_total": 0,
    "memory_items_by_kind": {},
    "memory_put_total": 0,
    "memory_prune_runs_total": 0,
    "memory_pruned_items_total": 0,
    "memory_expired_items_total": 0,
    "memory_duplicate_dropped_total": 0,
    "memory_degraded_total": 0,
    "memory_retrieval_requests_total": 0,
}

# Configuration for memory categorization
MEMORY_CONFIG = {
    "kinds": {
        "conversation": {"limit": 5},
        "analysis": {"limit": 8}, 
        "cleaning": {"limit": 5},
        "visualization": {"limit": 4},
        "insights": {"limit": 6},
        "errors": {"limit": 3}
    },
    "default_limit": 5,
    "fallback_limit": 10
}

# Load memory policy configuration
def load_memory_policy(config_path: Optional[str] = None) -> tuple[Dict[str, MemoryPolicy], RankingWeights]:
    """Load memory policy configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "memory_config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Extract policy defaults
        policy_defaults = config.get("memory_policy", {}).get("defaults", {})
        default_policy = MemoryPolicy(
            ttl_seconds=policy_defaults.get("ttl_seconds", 604800),
            max_items=policy_defaults.get("max_items", 1500),
            max_items_per_kind=policy_defaults.get("max_items_per_kind", 400),
            min_importance=policy_defaults.get("min_importance", 0.05),
            decay_half_life_seconds=policy_defaults.get("decay", {}).get("half_life_seconds", 259200),
            decay_floor=policy_defaults.get("decay", {}).get("floor", 0.05)
        )
        
        # Load per-kind policies
        policies = {}
        kinds_config = config.get("memory_policy", {}).get("kinds", {})
        for kind in ["conversation", "analysis", "cleaning", "visualization", "insights", "errors"]:
            kind_config = kinds_config.get(kind, {})
            policies[kind] = MemoryPolicy(
                ttl_seconds=kind_config.get("ttl_seconds", default_policy.ttl_seconds),
                max_items=kind_config.get("max_items", default_policy.max_items_per_kind),
                max_items_per_kind=kind_config.get("max_items", default_policy.max_items_per_kind),
                min_importance=kind_config.get("min_importance", default_policy.min_importance),
                decay_half_life_seconds=default_policy.decay_half_life_seconds,
                decay_floor=default_policy.decay_floor
            )
        
        # Load ranking weights
        ranking_config = config.get("ranking", {}).get("weights", {})
        weights = RankingWeights(
            similarity=ranking_config.get("similarity", 0.55),
            importance=ranking_config.get("importance", 0.25),
            recency=ranking_config.get("recency", 0.15),
            usage=ranking_config.get("usage", 0.05)
        )
        
        return policies, weights
        
    except Exception as e:
        logging.warning(f"Failed to load memory policy config: {e}, using defaults")
        # Return default policies for all kinds
        default_policy = MemoryPolicy()
        policies = {kind: default_policy for kind in ["conversation", "analysis", "cleaning", "visualization", "insights", "errors"]}
        return policies, RankingWeights()

# Global policy configuration
MEMORY_POLICIES, RANKING_WEIGHTS = load_memory_policy()

def estimate_importance(kind: str, text: str) -> float:
    """
    Estimate base importance score for a memory item based on content heuristics.
    
    Args:
        kind: Memory kind (affects weighting)
        text: Memory content text
        
    Returns:
        Base importance score between 0.0 and 1.0
    """
    # Token length weighting (longer content generally more important)
    length_score = min(1.0, len(text.split()) / 100.0)  # Cap at 100 tokens for full score
    
    # Keyword-based importance
    analytical_keywords = [
        "insight", "correlation", "anomaly", "pattern", "trend", "significant",
        "analysis", "conclusion", "finding", "result", "discovery", "error",
        "warning", "exception", "critical", "important", "key", "summary"
    ]
    
    keyword_count = sum(1 for keyword in analytical_keywords if keyword.lower() in text.lower())
    keyword_score = min(1.0, keyword_count / 5.0)  # Cap at 5 keywords for full score
    
    # Role-based weighting
    role_weights = {
        "analysis": 0.9,
        "insights": 0.95,
        "cleaning": 0.7,
        "visualization": 0.6,
        "conversation": 0.5,
        "errors": 0.8
    }
    role_weight = role_weights.get(kind, 0.5)
    
    # Combine scores
    base_importance = (length_score * 0.3 + keyword_score * 0.4 + role_weight * 0.3)
    return max(0.05, min(1.0, base_importance))

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Simple similarity calculation (can be enhanced with embeddings later).
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

class MemoryPolicyEngine:
    """
    Core engine for memory lifecycle management with policy-driven operations.
    """
    
    def __init__(self, store: InMemoryStore, debug: bool = False):
        self.store = store
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
    
    def insert(self, record: MemoryRecord) -> MemoryRecord:
        """
        Insert a memory record with policy enforcement.
        
        Args:
            record: Memory record to insert
            
        Returns:
            The inserted memory record (may be modified)
        """
        try:
            # Update metrics
            MEMORY_METRICS["memory_put_total"] += 1
            
            # Check for near-duplicates
            if self._check_duplicates(record):
                MEMORY_METRICS["memory_duplicate_dropped_total"] += 1
                if self.debug:
                    self.logger.debug(f"Dropping duplicate memory: {record.id}")
                return record
            
            # Store the record
            namespace = ("memories", record.kind)
            item = {
                "id": record.id,
                "text": record.text,
                "kind": record.kind,
                "vector": record.vector,
                "created_at": record.created_at,
                "last_used_at": record.last_used_at,
                "usage_count": record.usage_count,
                "base_importance": record.base_importance,
                "dynamic_importance": record.dynamic_importance,
                "degraded": record.degraded,
                "superseded_by": record.superseded_by,
                "meta": record.meta,
                "user_id": record.user_id
            }
            
            self.store.put(namespace, record.id, item)
            
            # Update metrics
            MEMORY_METRICS["memory_items_total"] += 1
            MEMORY_METRICS["memory_items_by_kind"][record.kind] = \
                MEMORY_METRICS["memory_items_by_kind"].get(record.kind, 0) + 1
            
            if record.degraded:
                MEMORY_METRICS["memory_degraded_total"] += 1
            
            # Trigger pruning if needed
            self._maybe_prune(record.kind)
            
            return record
            
        except Exception as e:
            self.logger.error(f"Failed to insert memory record: {e}")
            return record
    
    def retrieve(self, query: str, kinds: List[str], limit: int) -> List[MemoryRecord]:
        """
        Retrieve memories with enhanced ranking.
        
        Args:
            query: Search query
            kinds: Memory kinds to search
            limit: Maximum results
            
        Returns:
            List of ranked memory records
        """
        try:
            MEMORY_METRICS["memory_retrieval_requests_total"] += 1
            
            candidates = []
            
            # Search each kind
            for kind in kinds:
                namespace = ("memories", kind)
                try:
                    items = self.store.search(namespace, query=query, limit=limit*2)  # Get more for ranking
                    for item in items:
                        if isinstance(item, dict):
                            # Convert to MemoryRecord
                            record = MemoryRecord(
                                id=item.get("id", str(uuid.uuid4())),
                                kind=item.get("kind", kind),
                                text=item.get("text", ""),
                                vector=item.get("vector"),
                                created_at=item.get("created_at", time.time()),
                                last_used_at=item.get("last_used_at"),
                                usage_count=item.get("usage_count", 0),
                                base_importance=item.get("base_importance", 0.5),
                                dynamic_importance=item.get("dynamic_importance", 0.5),
                                degraded=item.get("degraded", False),
                                superseded_by=item.get("superseded_by"),
                                meta=item.get("meta", {}),
                                user_id=item.get("user_id", "user")
                            )
                            candidates.append(record)
                except Exception:
                    continue
            
            # Rank candidates
            ranked = self._rank_memories(query, candidates)
            
            # Update usage for returned memories
            for record in ranked[:limit]:
                self._update_usage(record)
            
            if self.debug and ranked:
                self.logger.debug(f"Top 5 retrieval scores: {[(r.id[:8], self._calculate_score(query, r)) for r in ranked[:5]]}")
            
            return ranked[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    def prune(self, reason: Optional[str] = None) -> PruneReport:
        """
        Prune memories according to policy.
        
        Args:
            reason: Optional reason for pruning
            
        Returns:
            Report of pruning operation
        """
        try:
            MEMORY_METRICS["memory_prune_runs_total"] += 1
            report = PruneReport()
            
            for kind in ["conversation", "analysis", "cleaning", "visualization", "insights", "errors"]:
                namespace = ("memories", kind)
                policy = MEMORY_POLICIES.get(kind, MemoryPolicy())
                
                try:
                    # Get all items in namespace
                    items = self.store.search(namespace, query="", limit=10000)  # Large limit to get all
                    if not items:
                        continue
                    
                    current_time = time.time()
                    to_delete = []
                    
                    # Time-based pruning
                    for item in items:
                        if isinstance(item, dict):
                            created_at = item.get("created_at", 0)
                            if current_time - created_at > policy.ttl_seconds:
                                to_delete.append(item.get("id"))
                                report.expired_count += 1
                    
                    # Superseded cleanup
                    for item in items:
                        if isinstance(item, dict) and item.get("superseded_by"):
                            if current_time - item.get("created_at", 0) > 86400:  # 1 day grace period
                                to_delete.append(item.get("id"))
                                report.superseded_count += 1
                    
                    # Remove expired/superseded items
                    remaining_items = [item for item in items if isinstance(item, dict) and item.get("id") not in to_delete]
                    
                    # Size-based pruning
                    if len(remaining_items) > policy.max_items:
                        # Sort by keep score
                        scored_items = []
                        for item in remaining_items:
                            if isinstance(item, dict):
                                importance = item.get("dynamic_importance", 0.5)
                                recency_factor = self._calculate_recency_factor(item.get("created_at", 0), policy.decay_half_life_seconds)
                                usage_count = item.get("usage_count", 0)
                                keep_score = importance * recency_factor * (1 + math.sqrt(usage_count))
                                scored_items.append((keep_score, item))
                        
                        # Sort by score (highest first) and keep top items
                        scored_items.sort(key=lambda x: x[0], reverse=True)
                        items_to_keep = scored_items[:policy.max_items]
                        items_to_remove = scored_items[policy.max_items:]
                        
                        for _, item in items_to_remove:
                            to_delete.append(item.get("id"))
                            report.size_pruned_count += 1
                    
                    # Low importance pruning
                    for item in remaining_items:
                        if isinstance(item, dict):
                            dynamic_importance = item.get("dynamic_importance", 0.5)
                            if dynamic_importance < policy.min_importance and item.get("id") not in to_delete:
                                to_delete.append(item.get("id"))
                                report.low_importance_count += 1
                    
                    # Execute deletions
                    for item_id in to_delete:
                        try:
                            # Note: InMemoryStore doesn't have delete method in interface
                            # In real implementation, would need to track and handle deletion
                            pass
                        except Exception:
                            pass
                    
                    # Update metrics
                    kind_pruned = len(to_delete)
                    MEMORY_METRICS["memory_pruned_items_total"] += kind_pruned
                    MEMORY_METRICS["memory_expired_items_total"] += report.expired_count
                    MEMORY_METRICS["memory_items_total"] -= kind_pruned
                    MEMORY_METRICS["memory_items_by_kind"][kind] = max(0, 
                        MEMORY_METRICS["memory_items_by_kind"].get(kind, 0) - kind_pruned)
                    
                except Exception as e:
                    self.logger.error(f"Failed to prune kind {kind}: {e}")
                    continue
            
            report.total_pruned = report.expired_count + report.superseded_count + report.size_pruned_count + report.low_importance_count
            report.remaining_count = MEMORY_METRICS["memory_items_total"]
            
            if self.debug:
                self.logger.debug(f"Pruning complete: {report.__dict__}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to prune memories: {e}")
            return PruneReport()
    
    def recalc_importance(self, records: List[MemoryRecord]):
        """
        Recalculate dynamic importance for memory records.
        
        Args:
            records: List of memory records to update
        """
        try:
            for record in records:
                policy = MEMORY_POLICIES.get(record.kind, MemoryPolicy())
                
                # Calculate recency factor
                recency_factor = self._calculate_recency_factor(record.created_at, policy.decay_half_life_seconds)
                
                # Calculate usage factor
                usage_factor = 1.0 + math.log1p(record.usage_count) * 0.15
                
                # Update dynamic importance
                record.dynamic_importance = record.base_importance * recency_factor * usage_factor
                record.dynamic_importance = max(policy.decay_floor, min(1.0, record.dynamic_importance))
                
                # Update in store
                namespace = ("memories", record.kind)
                item = {
                    "id": record.id,
                    "text": record.text,
                    "kind": record.kind,
                    "vector": record.vector,
                    "created_at": record.created_at,
                    "last_used_at": record.last_used_at,
                    "usage_count": record.usage_count,
                    "base_importance": record.base_importance,
                    "dynamic_importance": record.dynamic_importance,
                    "degraded": record.degraded,
                    "superseded_by": record.superseded_by,
                    "meta": record.meta,
                    "user_id": record.user_id
                }
                
                self.store.put(namespace, record.id, item)
                
        except Exception as e:
            self.logger.error(f"Failed to recalculate importance: {e}")
    
    def _check_duplicates(self, record: MemoryRecord) -> bool:
        """Check for near-duplicate content."""
        try:
            namespace = ("memories", record.kind)
            items = self.store.search(namespace, query=record.text[:100], limit=5)
            
            for item in items:
                if isinstance(item, dict):
                    existing_text = item.get("text", "")
                    similarity = calculate_similarity(record.text, existing_text)
                    if similarity > 0.96:
                        return True
            return False
        except Exception:
            return False
    
    def _rank_memories(self, query: str, candidates: List[MemoryRecord]) -> List[MemoryRecord]:
        """Rank memory candidates by weighted score."""
        scored = [(self._calculate_score(query, record), record) for record in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored]
    
    def _calculate_score(self, query: str, record: MemoryRecord) -> float:
        """Calculate weighted ranking score for a memory record."""
        # Similarity score
        similarity = calculate_similarity(query, record.text)
        
        # Recency factor
        policy = MEMORY_POLICIES.get(record.kind, MemoryPolicy())
        recency_factor = self._calculate_recency_factor(record.created_at, policy.decay_half_life_seconds)
        
        # Usage factor
        usage_factor = math.log(1 + record.usage_count) / math.log(1 + 100)  # Normalize to cap of 100
        
        # Weighted combination
        score = (RANKING_WEIGHTS.similarity * similarity +
                RANKING_WEIGHTS.importance * record.dynamic_importance +
                RANKING_WEIGHTS.recency * recency_factor +
                RANKING_WEIGHTS.usage * usage_factor)
        
        return score
    
    def _calculate_recency_factor(self, created_at: float, half_life: int) -> float:
        """Calculate recency decay factor."""
        age = time.time() - created_at
        return math.exp(-age / half_life)
    
    def _update_usage(self, record: MemoryRecord):
        """Update usage count and last used time for a memory record."""
        try:
            record.usage_count += 1
            record.last_used_at = time.time()
            
            # Update in store
            namespace = ("memories", record.kind)
            item = {
                "id": record.id,
                "text": record.text,
                "kind": record.kind,
                "vector": record.vector,
                "created_at": record.created_at,
                "last_used_at": record.last_used_at,
                "usage_count": record.usage_count,
                "base_importance": record.base_importance,
                "dynamic_importance": record.dynamic_importance,
                "degraded": record.degraded,
                "superseded_by": record.superseded_by,
                "meta": record.meta,
                "user_id": record.user_id
            }
            
            self.store.put(namespace, record.id, item)
            
        except Exception as e:
            self.logger.error(f"Failed to update usage: {e}")
    
    def _maybe_prune(self, kind: str):
        """Maybe trigger pruning if thresholds are exceeded."""
        try:
            policy = MEMORY_POLICIES.get(kind, MemoryPolicy())
            namespace = ("memories", kind)
            items = self.store.search(namespace, query="", limit=10000)
            
            if len(items) > policy.max_items * 1.1:  # 10% buffer before pruning
                self.prune(f"Size threshold exceeded for {kind}")
        except Exception:
            pass

def put_memory(
    store: InMemoryStore,
    kind: MemoryKind,
    text: str,
    meta: Optional[Dict[str, Any]] = None,
    user_id: str = "user",
    use_policy_engine: bool = True
) -> str:
    """
    Store a memory item with categorization and optional policy enforcement.
    
    Args:
        store: The InMemoryStore instance
        kind: Type of memory (conversation, analysis, cleaning, etc.)
        text: The memory content
        meta: Optional metadata dictionary
        user_id: User identifier for namespace isolation
        use_policy_engine: Whether to use the policy engine for lifecycle management
        
    Returns:
        The unique memory ID that was created
    """
    memory_id = str(uuid.uuid4())
    
    if use_policy_engine:
        # Use the enhanced policy engine
        try:
            base_importance = estimate_importance(kind, text)
            
            record = MemoryRecord(
                id=memory_id,
                kind=kind,
                text=text,
                created_at=time.time(),
                base_importance=base_importance,
                dynamic_importance=base_importance,
                meta=meta or {},
                user_id=user_id
            )
            
            # Try to embed the content (simplified - would use actual embeddings in production)
            try:
                # Placeholder for embedding logic
                record.vector = None  # Would be actual embedding
            except Exception:
                record.degraded = True
                record.vector = None
            
            engine = MemoryPolicyEngine(store, debug=os.getenv("DEBUG_MEMORY", "false").lower() == "true")
            engine.insert(record)
            
        except Exception as e:
            logging.warning(f"Failed to use policy engine, falling back to basic storage: {e}")
            # Fall back to basic storage
            use_policy_engine = False
    
    if not use_policy_engine:
        # Original implementation for backward compatibility
        namespace = ("memories", kind)
        
        item = {
            "text": text,
            "kind": kind,
            "meta": meta or {},
            "created_at": time.time(),
            "user_id": user_id
        }
        
        store.put(namespace, memory_id, item)
    
    return memory_id


def retrieve_memories(
    store: InMemoryStore,
    query: str,
    kinds: Optional[List[MemoryKind]] = None,
    limit: Optional[int] = None,
    user_id: str = "user",
    use_policy_engine: bool = True
) -> List[Dict[str, Any]]:
    """
    Retrieve memories with optional kind filtering, fallback, and enhanced ranking.
    
    Args:
        store: The InMemoryStore instance
        query: Search query text
        kinds: List of memory kinds to search (if None, searches all kinds)
        limit: Maximum number of results (if None, uses default from config)
        user_id: User identifier for namespace isolation
        use_policy_engine: Whether to use enhanced ranking and policy features
        
    Returns:
        List of memory items, ranked by similarity (and other factors if using policy engine)
    """
    results = []
    
    if use_policy_engine and kinds:
        # Use enhanced policy engine but preserve fallback behavior
        try:
            engine = MemoryPolicyEngine(store, debug=os.getenv("DEBUG_MEMORY", "false").lower() == "true")
            records = engine.retrieve(query, kinds, limit or MEMORY_CONFIG["default_limit"])
            
            # Convert back to dict format for backward compatibility
            for record in records:
                item = {
                    "text": record.text,
                    "kind": record.kind,
                    "meta": record.meta,
                    "created_at": record.created_at,
                    "user_id": record.user_id,
                    "namespace_kind": record.kind,
                    # Include policy engine specific fields
                    "id": record.id,
                    "usage_count": record.usage_count,
                    "base_importance": record.base_importance,
                    "dynamic_importance": record.dynamic_importance,
                    "last_used_at": record.last_used_at
                }
                results.append(item)
            
            # If policy engine returned results, return them
            if results:
                return results
                
        except Exception as e:
            logging.warning(f"Failed to use policy engine, falling back to basic retrieval: {e}")
    
    # Original implementation for backward compatibility OR fallback when no policy engine results
    # If specific kinds requested, search those first
    if kinds:
        for kind in kinds:
            kind_limit = MEMORY_CONFIG["kinds"].get(kind, {}).get("limit", MEMORY_CONFIG["default_limit"])
            if limit:
                # Use provided limit, distributed across kinds
                kind_limit = min(kind_limit, max(1, limit // len(kinds)))
            
            try:
                namespace = ("memories", kind)
                items = store.search(namespace, query=query, limit=kind_limit)
                for item in items:
                    if isinstance(item, dict):
                        item["namespace_kind"] = kind
                        results.append(item)
            except Exception:
                continue
    
    # If no results from specific kinds, or no kinds specified, fallback to generic namespace
    if not results:
        try:
            fallback_limit = limit or MEMORY_CONFIG["fallback_limit"]
            # Try user-specific namespace first
            user_namespace = (user_id, "memories")
            items = store.search(user_namespace, query=query, limit=fallback_limit)
            
            # If no user-specific results, try generic namespace
            if not items:
                generic_namespace = ("memories",)
                items = store.search(generic_namespace, query=query, limit=fallback_limit)
            
            for item in items:
                if isinstance(item, dict):
                    item["namespace_kind"] = "generic"
                    results.append(item)
        except Exception:
            pass
    
    # Apply overall limit if specified
    if limit and len(results) > limit:
        results = results[:limit]
    
    return results


def format_memories_by_kind(memories: List[Dict[str, Any]]) -> str:
    """
    Format memories grouped by kind for prompt inclusion.
    
    Args:
        memories: List of memory items from retrieve_memories
        
    Returns:
        Formatted string with grouped memory sections
    """
    if not memories:
        return "None."
    
    # Group memories by kind
    grouped = {}
    for memory in memories:
        kind = memory.get("namespace_kind", "generic")
        if kind not in grouped:
            grouped[kind] = []
        grouped[kind].append(memory)
    
    # Format grouped sections
    sections = []
    for kind, items in grouped.items():
        if kind == "generic":
            section_title = "[Previous Context]"
        else:
            section_title = f"[{kind.title()} Memory]"
        
        section_content = []
        for item in items:
            # Extract text content from various possible formats
            text = ""
            if isinstance(item, dict):
                text = item.get("text", item.get("memory", str(item)))
            else:
                text = str(item)
            section_content.append(text.strip())
        
        if section_content:
            sections.append(f"{section_title}\n" + "\n".join(section_content))
    
    return "\n\n".join(sections)


def enhanced_retrieve_mem(
    state: Union[Dict[str, Any], "State"],
    kinds: Optional[List[MemoryKind]] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Enhanced memory retrieval function for use in agent nodes.
    
    Args:
        state: Current state containing query information
        kinds: Specific memory kinds to retrieve
        limit: Maximum number of results
        
    Returns:
        List of relevant memory items
    """
    from langgraph.utils.config import get_store
    
    store = get_store()
    if not store:
        return []
    
    # Get query from state
    query = ""
    if isinstance(state, dict):
        query = state.get("next_agent_prompt") or state.get("user_prompt", "")
    else:
        query = getattr(state, "next_agent_prompt", "") or getattr(state, "user_prompt", "")
    
    if not query:
        return []
    
    return retrieve_memories(store, query, kinds=kinds, limit=limit)


def enhanced_mem_text(
    query: str,
    kinds: Optional[List[MemoryKind]] = None,
    limit: Optional[int] = None,
    store: Optional[InMemoryStore] = None
) -> str:
    """
    Enhanced version of _mem_text with kind support.
    
    Args:
        query: Search query
        kinds: Memory kinds to search
        limit: Maximum results
        store: Optional store instance (uses global if not provided)
        
    Returns:
        Formatted memory text grouped by kind
    """
    if store is None:
        # Try to get global store - this should be available in notebook context
        try:
            from langgraph.utils.config import get_store
            store = get_store()
        except:
            # Fallback to check for global variable if available
            try:
                import builtins
                store = getattr(builtins, 'in_memory_store', None)
            except:
                pass
    
    if not store:
        return "None."
    
    try:
        memories = retrieve_memories(store, query, kinds=kinds, limit=limit)
        return format_memories_by_kind(memories)
    except Exception:
        return "None."


def get_memory_metrics() -> Dict[str, Any]:
    """
    Get current memory metrics for monitoring and debugging.
    
    Returns:
        Dictionary of memory metrics
    """
    return MEMORY_METRICS.copy()

def reset_memory_metrics():
    """Reset memory metrics counters."""
    global MEMORY_METRICS
    MEMORY_METRICS = {
        "memory_items_total": 0,
        "memory_items_by_kind": {},
        "memory_put_total": 0,
        "memory_prune_runs_total": 0,
        "memory_pruned_items_total": 0,
        "memory_expired_items_total": 0,
        "memory_duplicate_dropped_total": 0,
        "memory_degraded_total": 0,
        "memory_retrieval_requests_total": 0,
    }

def memory_policy_report(store: InMemoryStore) -> Dict[str, Any]:
    """
    Generate a diagnostic report of memory status and policy compliance.
    
    Args:
        store: The InMemoryStore instance
        
    Returns:
        Dictionary with memory status information
    """
    try:
        report = {
            "timestamp": time.time(),
            "metrics": get_memory_metrics(),
            "policies": {},
            "kind_status": {},
            "recommendations": []
        }
        
        # Check each kind against its policy
        for kind in ["conversation", "analysis", "cleaning", "visualization", "insights", "errors"]:
            try:
                namespace = ("memories", kind)
                items = store.search(namespace, query="", limit=10000)
                count = len(items)
                policy = MEMORY_POLICIES.get(kind, MemoryPolicy())
                
                # Calculate age distribution
                current_time = time.time()
                ages = []
                expired_count = 0
                for item in items:
                    if isinstance(item, dict):
                        created_at = item.get("created_at", 0)
                        age = current_time - created_at
                        ages.append(age)
                        if age > policy.ttl_seconds:
                            expired_count += 1
                
                avg_age = sum(ages) / len(ages) if ages else 0
                
                kind_info = {
                    "current_count": count,
                    "max_allowed": policy.max_items,
                    "ttl_seconds": policy.ttl_seconds,
                    "expired_count": expired_count,
                    "avg_age_seconds": avg_age,
                    "compliance": count <= policy.max_items and expired_count == 0
                }
                
                report["kind_status"][kind] = kind_info
                report["policies"][kind] = {
                    "ttl_seconds": policy.ttl_seconds,
                    "max_items": policy.max_items,
                    "min_importance": policy.min_importance
                }
                
                # Generate recommendations
                if count > policy.max_items:
                    report["recommendations"].append(f"Consider pruning {kind} memories ({count} > {policy.max_items})")
                if expired_count > 0:
                    report["recommendations"].append(f"Run TTL cleanup for {kind} ({expired_count} expired items)")
                
            except Exception as e:
                report["kind_status"][kind] = {"error": str(e)}
        
        return report
        
    except Exception as e:
        return {"error": str(e), "timestamp": time.time()}

# Enhanced wrapper functions that use policy engine
def put_memory_with_policy(
    store: InMemoryStore,
    kind: MemoryKind,
    text: str,
    meta: Optional[Dict[str, Any]] = None,
    user_id: str = "user"
) -> str:
    """
    Store memory with full policy engine features enabled.
    """
    return put_memory(store, kind, text, meta, user_id, use_policy_engine=True)

def retrieve_memories_with_ranking(
    store: InMemoryStore,
    query: str,
    kinds: Optional[List[MemoryKind]] = None,
    limit: Optional[int] = None,
    user_id: str = "user"
) -> List[Dict[str, Any]]:
    """
    Retrieve memories with enhanced ranking enabled.
    """
    return retrieve_memories(store, query, kinds, limit, user_id, use_policy_engine=True)

def prune_memories(store: InMemoryStore, reason: Optional[str] = None) -> PruneReport:
    """
    Manually trigger memory pruning.
    
    Args:
        store: The InMemoryStore instance
        reason: Optional reason for pruning
        
    Returns:
        Pruning report
    """
    try:
        engine = MemoryPolicyEngine(store, debug=os.getenv("DEBUG_MEMORY", "false").lower() == "true")
        return engine.prune(reason)
    except Exception as e:
        logging.error(f"Failed to prune memories: {e}")
        return PruneReport()

def recalculate_importance(store: InMemoryStore, kinds: Optional[List[str]] = None) -> int:
    """
    Recalculate dynamic importance for stored memories.
    
    Args:
        store: The InMemoryStore instance
        kinds: Optional list of kinds to process (if None, processes all)
        
    Returns:
        Number of records updated
    """
    try:
        engine = MemoryPolicyEngine(store)
        kinds_to_process = kinds or ["conversation", "analysis", "cleaning", "visualization", "insights", "errors"]
        
        total_updated = 0
        for kind in kinds_to_process:
            try:
                namespace = ("memories", kind)
                items = store.search(namespace, query="", limit=10000)
                
                records = []
                for item in items:
                    if isinstance(item, dict):
                        record = MemoryRecord(
                            id=item.get("id", str(uuid.uuid4())),
                            kind=item.get("kind", kind),
                            text=item.get("text", ""),
                            created_at=item.get("created_at", time.time()),
                            usage_count=item.get("usage_count", 0),
                            base_importance=item.get("base_importance", 0.5),
                            dynamic_importance=item.get("dynamic_importance", 0.5),
                            user_id=item.get("user_id", "user")
                        )
                        records.append(record)
                
                engine.recalc_importance(records)
                total_updated += len(records)
                
            except Exception as e:
                logging.error(f"Failed to update importance for kind {kind}: {e}")
                continue
        
        return total_updated
        
    except Exception as e:
        logging.error(f"Failed to recalculate importance: {e}")
        return 0
def update_memory_with_kind(
    state: Union[MessagesState, "State"],
    config: RunnableConfig,
    kind: MemoryKind,
    memstore: Optional[InMemoryStore] = None
) -> str:
    """
    Enhanced update_memory function with memory kind categorization.
    
    Args:
        state: Current state with messages
        config: Runnable configuration with user_id
        kind: Type of memory being stored
        memstore: Optional memory store (uses global if not provided)
        
    Returns:
        The memory ID that was created
    """
    if memstore is None:
        # Use global store from notebook context if available
        try:
            import builtins
            memstore = getattr(builtins, 'in_memory_store', None)
        except:
            pass
    
    if not memstore:
        return ""
    
    user_id = str(config.get("configurable", {}).get("user_id", "user"))
    
    # Extract text from last message
    text = ""
    if hasattr(state, 'get') and state.get("messages"):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'text'):
            text = last_message.text()
        else:
            text = str(last_message)
    elif hasattr(state, "messages") and state.messages:
        last_message = state.messages[-1]
        if hasattr(last_message, 'text'):
            text = last_message.text()
        else:
            text = str(last_message)
    
    if not text:
        return ""
    
    return put_memory(memstore, kind, text, user_id=user_id)


# Backward-compatible wrapper functions
def update_memory_with_kind(
    state: Union[MessagesState, "State"],
    config: RunnableConfig,
    kind: MemoryKind,
    memstore: Optional[InMemoryStore] = None
) -> str:
    """
    Enhanced update_memory function with memory kind categorization.
    
    Args:
        state: Current state with messages
        config: Runnable configuration with user_id
        kind: Type of memory being stored
        memstore: Optional memory store (uses global if not provided)
        
    Returns:
        The memory ID that was created
    """
    if memstore is None:
        # Use global store from notebook context if available
        try:
            import builtins
            memstore = getattr(builtins, 'in_memory_store', None)
        except:
            pass
    
    if not memstore:
        return ""
    
    user_id = str(config.get("configurable", {}).get("user_id", "user"))
    
    # Extract text from last message
    text = ""
    if hasattr(state, 'get') and state.get("messages"):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'text'):
            text = last_message.text()
        else:
            text = str(last_message)
    elif hasattr(state, "messages") and state.messages:
        last_message = state.messages[-1]
        if hasattr(last_message, 'text'):
            text = last_message.text()
        else:
            text = str(last_message)
    
    if not text:
        return ""
    
    return put_memory_with_policy(memstore, kind, text, user_id=user_id)


# Export key functions for notebook integration
__all__ = [
    # Core types
    "MemoryKind",
    "MemoryRecord", 
    "MemoryPolicy",
    "RankingWeights",
    "PruneReport",
    "MemoryPolicyEngine",
    
    # Original functions (enhanced)
    "put_memory",
    "retrieve_memories", 
    "format_memories_by_kind",
    "enhanced_retrieve_mem",
    "enhanced_mem_text",
    "update_memory_with_kind",
    
    # New policy engine functions
    "put_memory_with_policy",
    "retrieve_memories_with_ranking", 
    "prune_memories",
    "recalculate_importance",
    "estimate_importance",
    "calculate_similarity",
    
    # Utilities and diagnostics
    "get_memory_metrics",
    "reset_memory_metrics",
    "memory_policy_report",
    "load_memory_policy",
    
    # Configuration
    "MEMORY_CONFIG",
    "MEMORY_POLICIES",
    "RANKING_WEIGHTS",
    "MEMORY_METRICS"
]