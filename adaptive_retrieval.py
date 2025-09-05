#!/usr/bin/env python3
"""
Adaptive Context Retrieval with Dynamic Memory Window Sizing

This module implements a sophisticated memory retrieval system that adapts
context size and composition based on agent type, task complexity, and token
budgets. It provides per-agent profiles, multi-factor ranking, and token-aware
packing for optimal context utilization.

Key Features:
- Per-agent + per-phase retrieval profiles with configurable limits
- Multi-factor ranking (similarity + importance + recency + usage)
- Token-aware greedy packing respecting budget constraints
- Task-specific overrides and query intent adjustments
- Comprehensive diagnostics and observability metrics
- Backward compatibility with legacy fixed-limit retrieval
"""

import os
import time
import math
import logging
import yaml
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum

# LangGraph and memory imports
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables.config import RunnableConfig

# Import existing memory enhancement components
from memory_enhancements import (
    MemoryKind, MemoryRecord, MemoryPolicyEngine, RankingWeights,
    retrieve_memories, MEMORY_POLICIES, RANKING_WEIGHTS,
    calculate_similarity, estimate_importance
)

# Type definitions
AgentName = Literal[
    "initial_analysis", "data_cleaner", "analyst", "visualization", 
    "report_orchestrator", "file_writer", "supervisor"
]

class TaskPhase(Enum):
    """Task phases that may need different memory profiles."""
    EXPLORATION = "exploration"
    ANALYSIS = "analysis"
    CLEANING = "cleaning"
    VISUALIZATION = "visualization"
    REPORTING = "reporting"
    PACKAGING = "packaging"
    REFINEMENT = "refinement"

@dataclass
class AgentProfile:
    """Per-agent memory retrieval profile configuration."""
    agent_name: str
    kinds: List[MemoryKind]
    min_items: int = 2
    max_items: int = 10
    target_items: int = 7
    token_budget: int = 600
    weighting_overrides: Optional[Dict[str, float]] = None
    phase_overrides: Optional[Dict[TaskPhase, Dict[str, Any]]] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.min_items > self.max_items:
            raise ValueError(f"min_items ({self.min_items}) cannot exceed max_items ({self.max_items})")
        if self.target_items > self.max_items:
            self.target_items = self.max_items
        if self.target_items < self.min_items:
            self.target_items = self.min_items

@dataclass
class GlobalConfig:
    """Global adaptive retrieval configuration."""
    default_memory_token_budget: int = 900
    hard_token_budget_fraction: float = 0.35
    fallback_limit: int = 5
    intent_boost_factor: float = 0.15
    enable_intent_adjustments: bool = True
    enable_diagnostics: bool = False

@dataclass
class RetrievalResult:
    """Result of adaptive memory retrieval."""
    selected: List[MemoryRecord]
    total_candidates: int
    truncated: bool
    token_usage: int
    diagnostics: Optional[Dict[str, Any]] = None
    profile_used: Optional[str] = None
    fallback_used: bool = False

@dataclass
class TokenEstimate:
    """Token estimation for a piece of text."""
    text: str
    estimated_tokens: int
    method: str = "heuristic"

class TokenEstimator:
    """Estimates token count for text using various methods."""
    
    def __init__(self, method: str = "heuristic"):
        self.method = method
        self.avg_chars_per_token = 4.0  # Conservative estimate
        
    def estimate(self, text: str) -> TokenEstimate:
        """Estimate token count for given text."""
        if not text:
            return TokenEstimate(text="", estimated_tokens=0, method=self.method)
        
        if self.method == "heuristic":
            # Simple heuristic: average 4 characters per token
            estimated = max(1, int(len(text) / self.avg_chars_per_token))
        else:
            # Future: could use tiktoken or other tokenizers
            estimated = max(1, len(text.split()) // 2)  # Rough approximation
            
        return TokenEstimate(
            text=text,
            estimated_tokens=estimated,
            method=self.method
        )

class AdaptiveRetriever:
    """
    Core adaptive memory retrieval engine.
    
    Provides context-aware memory retrieval with per-agent profiles,
    multi-factor ranking, and token budget management.
    """
    
    def __init__(
        self,
        store: InMemoryStore,
        config_path: Optional[str] = None,
        debug: bool = None
    ):
        self.store = store
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.global_config, self.agent_profiles = self._load_config(config_path)
        
        # Setup debug mode
        self.debug = debug if debug is not None else self.global_config.enable_diagnostics
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
            
        # Initialize components
        self.token_estimator = TokenEstimator()
        self.policy_engine = MemoryPolicyEngine(store, debug=self.debug)
        
        # Metrics
        self.metrics = {
            "adaptive_memory_requests_total": 0,
            "adaptive_memory_candidates_avg": 0,
            "adaptive_memory_selected_avg": 0,
            "adaptive_memory_truncated_total": 0,
            "adaptive_memory_token_usage": 0,
            "adaptive_memory_overflow_avoided_total": 0,
            "adaptive_memory_kind_distribution": {},
            "adaptive_memory_fallback_used_total": 0
        }
    
    def get_context(
        self,
        agent_name: str,
        query: str,
        base_prompt_tokens: int = 0,
        model_context_window: int = 8192,
        expected_output_tokens: int = 512,
        task_phase: Optional[TaskPhase] = None,
        task_overrides: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Get adaptive memory context for an agent.
        
        Args:
            agent_name: Name of the requesting agent
            query: Search query for memory retrieval
            base_prompt_tokens: Token count of base prompt (without memory)
            model_context_window: Maximum context window for the model
            expected_output_tokens: Expected tokens needed for response
            task_phase: Optional task phase for phase-specific adjustments
            task_overrides: Optional per-task configuration overrides
            
        Returns:
            RetrievalResult with selected memories and metadata
        """
        try:
            self.metrics["adaptive_memory_requests_total"] += 1
            
            # Get agent profile
            profile = self._get_agent_profile(agent_name, task_phase, task_overrides)
            if not profile:
                return self._fallback_retrieval(query, agent_name)
            
            # Step 1: Retrieve candidate pool
            candidates = self._get_candidate_pool(profile, query)
            self.metrics["adaptive_memory_candidates_avg"] = len(candidates)
            
            if not candidates:
                # If no candidates found, try fallback approach
                fallback_candidates = self._get_fallback_candidates(profile, query)
                candidates = fallback_candidates
                
            if not candidates:
                return RetrievalResult(
                    selected=[],
                    total_candidates=0,
                    truncated=False,
                    token_usage=0,
                    profile_used=agent_name,
                    fallback_used=False
                )
            
            # Step 2: Apply query intent adjustments
            if self.global_config.enable_intent_adjustments:
                candidates = self._apply_intent_adjustments(query, candidates)
            
            # Step 3: Rank with composite score
            ranked_candidates = self._rank_candidates(query, candidates, profile)
            
            # Step 4: Greedy token-aware packing
            selected, token_usage, truncated = self._greedy_pack(
                ranked_candidates, profile, base_prompt_tokens, 
                model_context_window, expected_output_tokens
            )
            
            # Step 5: Update metrics
            self.metrics["adaptive_memory_selected_avg"] += len(selected)
            self.metrics["adaptive_memory_token_usage"] += token_usage
            self.metrics["adaptive_memory_candidates_avg"] = (
                (self.metrics["adaptive_memory_candidates_avg"] * (self.metrics["adaptive_memory_requests_total"] - 1) + len(candidates))
                / self.metrics["adaptive_memory_requests_total"]
            )
            if truncated:
                self.metrics["adaptive_memory_truncated_total"] += 1
            
            # Step 6: Update usage statistics for selected memories
            for record in selected:
                self.policy_engine._update_usage(record)
            
            # Step 7: Generate diagnostics if enabled
            diagnostics = None
            if self.debug:
                diagnostics = self._generate_diagnostics(
                    query, ranked_candidates, selected, profile, token_usage
                )
            
            return RetrievalResult(
                selected=selected,
                total_candidates=len(candidates),
                truncated=truncated,
                token_usage=token_usage,
                diagnostics=diagnostics,
                profile_used=agent_name,
                fallback_used=False
            )
            
        except Exception as e:
            self.logger.error(f"Adaptive retrieval failed for agent {agent_name}: {e}")
            return self._fallback_retrieval(query, agent_name)
    
    def _load_config(self, config_path: Optional[str]) -> Tuple[GlobalConfig, Dict[str, AgentProfile]]:
        """Load adaptive retrieval configuration from YAML."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "adaptive_memory_config.yaml")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load global config
            global_cfg = config.get("adaptive_retrieval", {}).get("global", {})
            global_config = GlobalConfig(
                default_memory_token_budget=global_cfg.get("default_memory_token_budget", 900),
                hard_token_budget_fraction=global_cfg.get("hard_token_budget_fraction", 0.35),
                fallback_limit=global_cfg.get("fallback_limit", 5),
                intent_boost_factor=global_cfg.get("intent_boost_factor", 0.15),
                enable_intent_adjustments=global_cfg.get("enable_intent_adjustments", True),
                enable_diagnostics=global_cfg.get("enable_diagnostics", False)
            )
            
            # Load agent profiles
            agent_profiles = {}
            agents_cfg = config.get("adaptive_retrieval", {}).get("agents", {})
            
            for agent_name, agent_cfg in agents_cfg.items():
                try:
                    profile = AgentProfile(
                        agent_name=agent_name,
                        kinds=agent_cfg.get("kinds", ["conversation"]),
                        min_items=agent_cfg.get("min_items", 2),
                        max_items=agent_cfg.get("max_items", 10),
                        target_items=agent_cfg.get("target_items", 7),
                        token_budget=agent_cfg.get("token_budget", 600),
                        weighting_overrides=agent_cfg.get("weighting_overrides"),
                        phase_overrides=agent_cfg.get("phase_overrides")
                    )
                    agent_profiles[agent_name] = profile
                except Exception as e:
                    self.logger.warning(f"Failed to load profile for agent {agent_name}: {e}")
                    continue
            
            return global_config, agent_profiles
            
        except Exception as e:
            self.logger.warning(f"Failed to load adaptive config: {e}, using defaults")
            # Return default configuration
            return GlobalConfig(), self._get_default_profiles()
    
    def _get_default_profiles(self) -> Dict[str, AgentProfile]:
        """Get default agent profiles when config loading fails."""
        return {
            "initial_analysis": AgentProfile(
                agent_name="initial_analysis",
                kinds=["conversation", "analysis"],
                min_items=3, max_items=10, target_items=7, token_budget=600,
                weighting_overrides={"similarity": 0.6, "importance": 0.2, "recency": 0.15, "usage": 0.05}
            ),
            "data_cleaner": AgentProfile(
                agent_name="data_cleaner",
                kinds=["cleaning", "analysis", "conversation"],
                min_items=2, max_items=8, target_items=5, token_budget=450
            ),
            "analyst": AgentProfile(
                agent_name="analyst",
                kinds=["analysis", "cleaning", "conversation"],
                min_items=5, max_items=18, target_items=12, token_budget=1000
            ),
            "visualization": AgentProfile(
                agent_name="visualization",
                kinds=["visualization", "analysis"],
                min_items=3, max_items=12, target_items=8, token_budget=650
            ),
            "report_orchestrator": AgentProfile(
                agent_name="report_orchestrator",
                kinds=["analysis", "visualization", "cleaning"],
                min_items=4, max_items=15, target_items=10, token_budget=800
            ),
            "file_writer": AgentProfile(
                agent_name="file_writer",
                kinds=["analysis", "visualization"],
                min_items=1, max_items=6, target_items=4, token_budget=300
            )
        }
    
    def _get_agent_profile(
        self, 
        agent_name: str, 
        task_phase: Optional[TaskPhase] = None,
        task_overrides: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentProfile]:
        """Get effective agent profile with phase and task overrides applied."""
        base_profile = self.agent_profiles.get(agent_name)
        if not base_profile:
            return None
        
        # Create a copy for modification
        profile = AgentProfile(
            agent_name=base_profile.agent_name,
            kinds=base_profile.kinds.copy(),
            min_items=base_profile.min_items,
            max_items=base_profile.max_items,
            target_items=base_profile.target_items,
            token_budget=base_profile.token_budget,
            weighting_overrides=base_profile.weighting_overrides.copy() if base_profile.weighting_overrides else None,
            phase_overrides=base_profile.phase_overrides
        )
        
        # Apply phase overrides
        if task_phase and profile.phase_overrides and task_phase in profile.phase_overrides:
            phase_cfg = profile.phase_overrides[task_phase]
            for key, value in phase_cfg.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
        
        # Apply task-specific overrides
        if task_overrides:
            for key, value in task_overrides.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
        
        return profile
    
    def _get_candidate_pool(self, profile: AgentProfile, query: str) -> List[MemoryRecord]:
        """Retrieve candidate pool from memory store."""
        candidates = []
        expansion_factor = 3  # Get more candidates than max_items for better ranking
        
        for kind in profile.kinds:
            try:
                # Try policy engine first, fallback to direct store search
                try:
                    kind_candidates = self.policy_engine.retrieve(
                        query=query,
                        kinds=[kind],
                        limit=profile.max_items * expansion_factor
                    )
                except Exception:
                    # Fallback to direct store search
                    namespace = ("memories", kind)
                    items = self.store.search(namespace, query=query, limit=profile.max_items * expansion_factor)
                    kind_candidates = []
                    
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
                            kind_candidates.append(record)
                
                candidates.extend(kind_candidates)
                
                # Update kind distribution metrics
                kind_key = f"kind_{kind}"
                self.metrics["adaptive_memory_kind_distribution"][kind_key] = (
                    self.metrics["adaptive_memory_kind_distribution"].get(kind_key, 0) + len(kind_candidates)
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to retrieve candidates for kind {kind}: {e}")
                continue
        
        return candidates
    
    def _get_fallback_candidates(self, profile: AgentProfile, query: str) -> List[MemoryRecord]:
        """Fallback candidate retrieval when primary method fails."""
        candidates = []
        
        for kind in profile.kinds:
            try:
                # Direct store search as fallback
                namespace = ("memories", kind)
                items = self.store.search(namespace, query=query, limit=profile.max_items * 2)
                
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
                        candidates.append(record)
                        
            except Exception as e:
                self.logger.warning(f"Fallback retrieval failed for kind {kind}: {e}")
                continue
        
        return candidates
    
    def _apply_intent_adjustments(self, query: str, candidates: List[MemoryRecord]) -> List[MemoryRecord]:
        """Apply query intent-based score adjustments."""
        # Analytical cue words that suggest the need for analytical context
        analytical_cues = [
            "trend", "correlation", "outlier", "pattern", "analysis", "insight",
            "statistic", "distribution", "anomaly", "relationship", "significant"
        ]
        
        query_lower = query.lower()
        has_analytical_intent = any(cue in query_lower for cue in analytical_cues)
        
        if has_analytical_intent:
            boost_factor = 1.0 + self.global_config.intent_boost_factor
            for candidate in candidates:
                if candidate.kind in ["analysis", "insights"]:
                    # Boost dynamic importance for analytical memories
                    candidate.dynamic_importance = min(1.0, candidate.dynamic_importance * boost_factor)
        
        return candidates
    
    def _rank_candidates(
        self, 
        query: str, 
        candidates: List[MemoryRecord], 
        profile: AgentProfile
    ) -> List[MemoryRecord]:
        """Rank candidates using composite scoring."""
        # Get effective weights
        if profile.weighting_overrides:
            weights = RankingWeights(**profile.weighting_overrides)
        else:
            weights = RANKING_WEIGHTS
        
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_composite_score(query, candidate, weights)
            scored_candidates.append((score, candidate))
        
        # Sort by score (descending) with tie-breakers
        scored_candidates.sort(key=lambda x: (
            x[0],  # Primary: composite score
            x[1].dynamic_importance,  # Tie-breaker 1: importance
            -x[1].created_at,  # Tie-breaker 2: recency (newer first)
            -len(x[1].text)  # Tie-breaker 3: shorter text preferred if over budget
        ), reverse=True)
        
        return [candidate for _, candidate in scored_candidates]
    
    def _calculate_composite_score(self, query: str, record: MemoryRecord, weights: RankingWeights) -> float:
        """Calculate weighted composite score for a memory record."""
        # Similarity score
        similarity = calculate_similarity(query, record.text)
        
        # Recency factor (exponential decay)
        age_seconds = time.time() - record.created_at
        half_life = getattr(MEMORY_POLICIES.get(record.kind), 'decay_half_life_seconds', 259200)
        recency_factor = math.exp(-age_seconds / half_life)
        
        # Usage factor (logarithmic scaling)
        usage_factor = math.log(1 + record.usage_count) / math.log(1 + 100)  # Normalize to cap of 100
        
        # Weighted combination
        score = (
            weights.similarity * similarity +
            weights.importance * (record.dynamic_importance or record.base_importance) +
            weights.recency * recency_factor +
            weights.usage * usage_factor
        )
        
        return score
    
    def _greedy_pack(
        self,
        ranked_candidates: List[MemoryRecord],
        profile: AgentProfile,
        base_prompt_tokens: int,
        model_context_window: int,
        expected_output_tokens: int
    ) -> Tuple[List[MemoryRecord], int, bool]:
        """Pack memories greedily respecting token budgets."""
        selected = []
        total_tokens = 0
        truncated = False
        
        # Calculate available token budget
        hard_limit_tokens = int(model_context_window * self.global_config.hard_token_budget_fraction)
        available_tokens = hard_limit_tokens - base_prompt_tokens - expected_output_tokens
        
        # Use profile token budget if smaller than calculated available
        available_tokens = min(profile.token_budget, max(0, available_tokens))
        
        if available_tokens <= 0:
            self.logger.warning(f"No token budget available for memory (base={base_prompt_tokens}, expected_output={expected_output_tokens}, hard_limit={hard_limit_tokens})")
            # Still try to include minimum items even with no budget
            for candidate in ranked_candidates[:profile.min_items]:
                selected.append(candidate)
                token_estimate = self.token_estimator.estimate(candidate.text)
                total_tokens += token_estimate.estimated_tokens
            return selected, total_tokens, True
        
        # Greedy packing
        for candidate in ranked_candidates:
            # Estimate tokens for this candidate
            token_estimate = self.token_estimator.estimate(candidate.text)
            candidate_tokens = token_estimate.estimated_tokens
            
            # Check if we can add this candidate
            if len(selected) < profile.min_items:
                # Always include minimum items, even if over budget
                selected.append(candidate)
                total_tokens += candidate_tokens
            elif (len(selected) < profile.max_items and 
                  total_tokens + candidate_tokens <= available_tokens):
                # Add if within limits and budget
                selected.append(candidate)
                total_tokens += candidate_tokens
            else:
                # Stop packing
                truncated = len(ranked_candidates) > len(selected)
                break
        
        # Additional context window overflow protection
        projected_total = base_prompt_tokens + total_tokens + expected_output_tokens
        hard_limit = int(model_context_window * self.global_config.hard_token_budget_fraction)
        
        if projected_total > hard_limit:
            self.metrics["adaptive_memory_overflow_avoided_total"] += 1
            selected, total_tokens = self._trim_for_context_window(
                selected, hard_limit - base_prompt_tokens - expected_output_tokens
            )
            truncated = True
        
        return selected, total_tokens, truncated
    
    def _trim_for_context_window(
        self, 
        selected: List[MemoryRecord], 
        max_memory_tokens: int
    ) -> Tuple[List[MemoryRecord], int]:
        """Trim selected memories to fit within context window."""
        if not selected:
            return [], 0
        
        # Sort by composite score (lower scored items removed first)
        # Use a simple heuristic: longer texts are more likely to be trimmed
        trimmed = []
        total_tokens = 0
        
        # Sort by preference: shorter texts first, then by score
        selected_with_tokens = []
        for record in selected:
            tokens = self.token_estimator.estimate(record.text).estimated_tokens
            selected_with_tokens.append((record, tokens))
        
        # Sort by tokens (ascending) to prefer shorter memories when trimming
        selected_with_tokens.sort(key=lambda x: x[1])
        
        for record, tokens in selected_with_tokens:
            if total_tokens + tokens <= max_memory_tokens:
                trimmed.append(record)
                total_tokens += tokens
            else:
                break
        
        return trimmed, total_tokens
    
    def _generate_diagnostics(
        self,
        query: str,
        ranked_candidates: List[MemoryRecord],
        selected: List[MemoryRecord],
        profile: AgentProfile,
        token_usage: int
    ) -> Dict[str, Any]:
        """Generate comprehensive diagnostics information."""
        selected_ids = {record.id for record in selected}
        
        # Generate candidate table (top 25)
        candidate_table = []
        for i, candidate in enumerate(ranked_candidates[:25]):
            score = self._calculate_composite_score(
                query, candidate, 
                RankingWeights(**profile.weighting_overrides) if profile.weighting_overrides else RANKING_WEIGHTS
            )
            
            similarity = calculate_similarity(query, candidate.text)
            importance = candidate.dynamic_importance or candidate.base_importance
            age_seconds = time.time() - candidate.created_at
            half_life = MEMORY_POLICIES.get(candidate.kind, type('obj', (object,), {"decay_half_life_seconds": 259200})()).decay_half_life_seconds
            recency = math.exp(-age_seconds / half_life)
            usage = math.log(1 + candidate.usage_count) / math.log(1 + 100)
            est_tokens = self.token_estimator.estimate(candidate.text).estimated_tokens
            
            packed = candidate.id in selected_ids
            
            # Determine exclusion reason
            exclusion_reason = ""
            if not packed:
                if i < profile.min_items:
                    exclusion_reason = "should_be_included"  # This shouldn't happen
                elif token_usage + est_tokens > profile.token_budget:
                    exclusion_reason = "over_budget"
                elif len(selected) >= profile.max_items:
                    exclusion_reason = "max_items_reached"
                else:
                    exclusion_reason = "score_threshold"
            
            candidate_table.append({
                "rank": i + 1,
                "id": candidate.id[:8],
                "kind": candidate.kind,
                "similarity": round(similarity, 3),
                "importance": round(importance, 3),
                "recency": round(recency, 3),
                "usage": round(usage, 3),
                "score": round(score, 3),
                "est_tokens": est_tokens,
                "packed": packed,
                "exclusion_reason": exclusion_reason
            })
        
        # Generate summary statistics
        diagnostics = {
            "query": query,
            "agent_profile": profile.agent_name,
            "total_candidates": len(ranked_candidates),
            "selected_count": len(selected),
            "token_usage": token_usage,
            "token_budget": profile.token_budget,
            "budget_utilization": round(token_usage / profile.token_budget, 3) if profile.token_budget > 0 else 0,
            "candidate_table": candidate_table,
            "profile_config": {
                "kinds": profile.kinds,
                "min_items": profile.min_items,
                "max_items": profile.max_items,
                "target_items": profile.target_items,
                "token_budget": profile.token_budget,
                "weighting_overrides": profile.weighting_overrides
            },
            "exclusion_summary": {
                "over_budget": sum(1 for c in candidate_table if c["exclusion_reason"] == "over_budget"),
                "max_items_reached": sum(1 for c in candidate_table if c["exclusion_reason"] == "max_items_reached"),
                "score_threshold": sum(1 for c in candidate_table if c["exclusion_reason"] == "score_threshold")
            }
        }
        
        return diagnostics
    
    def _fallback_retrieval(self, query: str, agent_name: str) -> RetrievalResult:
        """Fallback to legacy fixed-limit retrieval."""
        self.metrics["adaptive_memory_fallback_used_total"] += 1
        self.logger.debug(f"Using fallback retrieval for agent {agent_name}")
        
        try:
            # Use simple store search with fallback limit
            namespace = ("memories",)
            items = self.store.search(namespace, query=query, limit=self.global_config.fallback_limit)
            
            # Convert to MemoryRecord format
            selected = []
            for item in items:
                if isinstance(item, dict):
                    record = MemoryRecord(
                        id=item.get("id", str(uuid.uuid4())),
                        kind=item.get("kind", "conversation"),
                        text=item.get("text", ""),
                        created_at=item.get("created_at", time.time()),
                        usage_count=item.get("usage_count", 0),
                        base_importance=item.get("base_importance", 0.5),
                        dynamic_importance=item.get("dynamic_importance", 0.5),
                        user_id=item.get("user_id", "user")
                    )
                    selected.append(record)
            
            # Estimate token usage
            total_tokens = sum(
                self.token_estimator.estimate(record.text).estimated_tokens 
                for record in selected
            )
            
            return RetrievalResult(
                selected=selected,
                total_candidates=len(items),
                truncated=False,
                token_usage=total_tokens,
                profile_used=None,
                fallback_used=True
            )
            
        except Exception as e:
            self.logger.error(f"Fallback retrieval failed: {e}")
            return RetrievalResult(
                selected=[],
                total_candidates=0,
                truncated=False,
                token_usage=0,
                fallback_used=True
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current adaptive retrieval metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset metrics counters."""
        for key in self.metrics:
            if isinstance(self.metrics[key], dict):
                self.metrics[key] = {}
            else:
                self.metrics[key] = 0


# Feature flag control
def is_adaptive_retrieval_enabled() -> bool:
    """Check if adaptive retrieval is enabled via environment variable."""
    return os.getenv("ADAPTIVE_RETRIEVAL_ENABLED", "false").lower() == "true"


def create_adaptive_retriever(store: InMemoryStore, config_path: Optional[str] = None) -> Optional[AdaptiveRetriever]:
    """Factory function to create AdaptiveRetriever if enabled."""
    if not is_adaptive_retrieval_enabled():
        return None
    
    try:
        return AdaptiveRetriever(store, config_path)
    except Exception as e:
        logging.error(f"Failed to create AdaptiveRetriever: {e}")
        return None


# Export key components
__all__ = [
    "AdaptiveRetriever",
    "AgentProfile", 
    "GlobalConfig",
    "RetrievalResult",
    "TaskPhase",
    "TokenEstimator",
    "is_adaptive_retrieval_enabled",
    "create_adaptive_retriever"
]