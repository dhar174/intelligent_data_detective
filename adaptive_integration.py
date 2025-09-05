#!/usr/bin/env python3
"""
Integration wrapper for Adaptive Context Retrieval in Intelligent Data Detective.

This module provides seamless integration of the adaptive retrieval system
into the existing notebook workflow with backward compatibility.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union

# Import the adaptive retrieval system
from adaptive_retrieval import (
    AdaptiveRetriever, create_adaptive_retriever, is_adaptive_retrieval_enabled,
    TaskPhase, RetrievalResult
)

# Import existing memory system components
from memory_enhancements import (
    retrieve_memories, format_memories_by_kind, enhanced_retrieve_mem,
    MemoryKind
)

# Global adaptive retriever instance
_global_adaptive_retriever: Optional[AdaptiveRetriever] = None

def get_adaptive_retriever():
    """Get or create the global adaptive retriever instance."""
    global _global_adaptive_retriever
    
    if _global_adaptive_retriever is None and is_adaptive_retrieval_enabled():
        try:
            from langgraph.utils.config import get_store
            store = get_store()
            if store:
                _global_adaptive_retriever = create_adaptive_retriever(store)
        except Exception as e:
            logging.warning(f"Failed to create adaptive retriever: {e}")
    
    return _global_adaptive_retriever

def adaptive_retrieve_mem(
    state: Union[Dict[str, Any], "State"],
    agent_name: str,
    kinds: Optional[List[MemoryKind]] = None,
    limit: Optional[int] = None,
    base_prompt_tokens: int = 0,
    model_context_window: int = 8192,
    expected_output_tokens: int = 512,
    task_phase: Optional[TaskPhase] = None,
    task_overrides: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Adaptive memory retrieval with fallback to legacy system.
    
    This function provides the enhanced adaptive retrieval when enabled,
    or falls back to the existing memory system for backward compatibility.
    
    Args:
        state: Current state containing query information
        agent_name: Name of the requesting agent
        kinds: Memory kinds to retrieve (for legacy fallback)
        limit: Maximum results (for legacy fallback)
        base_prompt_tokens: Token count of base prompt (for adaptive)
        model_context_window: Model context window size (for adaptive)
        expected_output_tokens: Expected output tokens (for adaptive)
        task_phase: Optional task phase for adaptive retrieval
        task_overrides: Optional task-specific overrides
        
    Returns:
        List of memory items in legacy format for compatibility
    """
    # Get query from state
    query = ""
    if isinstance(state, dict):
        query = state.get("next_agent_prompt") or state.get("user_prompt", "")
    else:
        query = getattr(state, "next_agent_prompt", "") or getattr(state, "user_prompt", "")
    
    # Try adaptive retrieval first
    retriever = get_adaptive_retriever()
    if retriever and query:
        try:
            result = retriever.get_context(
                agent_name=agent_name,
                query=query,
                base_prompt_tokens=base_prompt_tokens,
                model_context_window=model_context_window,
                expected_output_tokens=expected_output_tokens,
                task_phase=task_phase,
                task_overrides=task_overrides
            )
            
            # Convert back to legacy format for compatibility
            legacy_memories = []
            for record in result.selected:
                memory_item = {
                    "text": record.text,
                    "kind": record.kind,
                    "meta": record.meta,
                    "created_at": record.created_at,
                    "user_id": record.user_id,
                    "namespace_kind": record.kind,
                    # Include adaptive-specific fields for debugging
                    "id": record.id,
                    "usage_count": record.usage_count,
                    "base_importance": record.base_importance,
                    "dynamic_importance": record.dynamic_importance,
                    "last_used_at": record.last_used_at,
                    # Adaptive metadata
                    "_adaptive_result": {
                        "profile_used": result.profile_used,
                        "total_candidates": result.total_candidates,
                        "truncated": result.truncated,
                        "token_usage": result.token_usage,
                        "fallback_used": result.fallback_used
                    }
                }
                legacy_memories.append(memory_item)
            
            return legacy_memories
            
        except Exception as e:
            logging.warning(f"Adaptive retrieval failed, falling back to legacy: {e}")
    
    # Fallback to legacy enhanced retrieval
    return enhanced_retrieve_mem(state, kinds=kinds, limit=limit)

def adaptive_mem_text(
    query: str,
    agent_name: str,
    kinds: Optional[List[MemoryKind]] = None,
    limit: Optional[int] = None,
    base_prompt_tokens: int = 0,
    model_context_window: int = 8192,
    expected_output_tokens: int = 512,
    task_phase: Optional[TaskPhase] = None
) -> str:
    """
    Get formatted memory text using adaptive retrieval.
    
    Args:
        query: Search query
        agent_name: Name of the requesting agent
        kinds: Memory kinds for legacy fallback
        limit: Maximum results for legacy fallback
        base_prompt_tokens: Token count of base prompt
        model_context_window: Model context window size
        expected_output_tokens: Expected output tokens
        task_phase: Optional task phase
        
    Returns:
        Formatted memory text grouped by kind
    """
    # Try adaptive retrieval first
    retriever = get_adaptive_retriever()
    if retriever and query:
        try:
            result = retriever.get_context(
                agent_name=agent_name,
                query=query,
                base_prompt_tokens=base_prompt_tokens,
                model_context_window=model_context_window,
                expected_output_tokens=expected_output_tokens,
                task_phase=task_phase
            )
            
            if result.selected:
                # Convert to legacy format and use existing formatter
                legacy_memories = []
                for record in result.selected:
                    memory_item = {
                        "text": record.text,
                        "namespace_kind": record.kind
                    }
                    legacy_memories.append(memory_item)
                
                return format_memories_by_kind(legacy_memories)
                
        except Exception as e:
            logging.warning(f"Adaptive memory text failed, falling back to legacy: {e}")
    
    # Fallback to legacy enhanced memory text
    from memory_enhancements import enhanced_mem_text
    return enhanced_mem_text(query, kinds=kinds, limit=limit)

# Agent-specific convenience functions for easy migration
def initial_analysis_retrieve_mem(state, **kwargs):
    """Retrieve memory for initial analysis agent."""
    return adaptive_retrieve_mem(
        state, 
        agent_name="initial_analysis",
        kinds=["conversation", "analysis"],
        **kwargs
    )

def data_cleaner_retrieve_mem(state, **kwargs):
    """Retrieve memory for data cleaner agent."""
    return adaptive_retrieve_mem(
        state,
        agent_name="data_cleaner", 
        kinds=["cleaning", "analysis", "conversation"],
        **kwargs
    )

def analyst_retrieve_mem(state, **kwargs):
    """Retrieve memory for analyst agent."""
    return adaptive_retrieve_mem(
        state,
        agent_name="analyst",
        kinds=["analysis", "cleaning", "conversation"],
        **kwargs
    )

def visualization_retrieve_mem(state, **kwargs):
    """Retrieve memory for visualization agent."""
    return adaptive_retrieve_mem(
        state,
        agent_name="visualization",
        kinds=["visualization", "analysis"],
        **kwargs
    )

def report_orchestrator_retrieve_mem(state, **kwargs):
    """Retrieve memory for report orchestrator agent."""
    return adaptive_retrieve_mem(
        state,
        agent_name="report_orchestrator",
        kinds=["analysis", "visualization", "cleaning"],
        **kwargs
    )

def file_writer_retrieve_mem(state, **kwargs):
    """Retrieve memory for file writer agent."""
    return adaptive_retrieve_mem(
        state,
        agent_name="file_writer",
        kinds=["analysis", "visualization"],
        **kwargs
    )

def supervisor_retrieve_mem(state, **kwargs):
    """Retrieve memory for supervisor agent."""
    return adaptive_retrieve_mem(
        state,
        agent_name="supervisor",
        kinds=["conversation", "analysis", "cleaning", "visualization", "insights"],
        **kwargs
    )

# Diagnostics and monitoring
def get_adaptive_retrieval_diagnostics() -> Dict[str, Any]:
    """Get diagnostic information about adaptive retrieval usage."""
    retriever = get_adaptive_retriever()
    if not retriever:
        return {
            "enabled": is_adaptive_retrieval_enabled(),
            "active": False,
            "reason": "Retriever not initialized or disabled"
        }
    
    metrics = retriever.get_metrics()
    return {
        "enabled": True,
        "active": True,
        "metrics": metrics,
        "config": {
            "global_config": retriever.global_config.__dict__,
            "agent_profiles": {
                name: profile.__dict__ 
                for name, profile in retriever.agent_profiles.items()
            }
        }
    }

def reset_adaptive_retrieval_metrics():
    """Reset adaptive retrieval metrics."""
    retriever = get_adaptive_retriever()
    if retriever:
        retriever.reset_metrics()

# Legacy compatibility function
def create_legacy_retrieve_mem(agent_name: str, kinds: List[MemoryKind]):
    """
    Create a legacy-style retrieve_mem function for a specific agent.
    
    This allows gradual migration by creating drop-in replacements for
    the existing retrieve_mem functions in each node.
    """
    def retrieve_mem(state):
        return adaptive_retrieve_mem(state, agent_name=agent_name, kinds=kinds)
    
    return retrieve_mem

# Export key functions
__all__ = [
    # Main adaptive functions
    "adaptive_retrieve_mem",
    "adaptive_mem_text",
    
    # Agent-specific functions
    "initial_analysis_retrieve_mem",
    "data_cleaner_retrieve_mem", 
    "analyst_retrieve_mem",
    "visualization_retrieve_mem",
    "report_orchestrator_retrieve_mem",
    "file_writer_retrieve_mem",
    "supervisor_retrieve_mem",
    
    # Utilities
    "get_adaptive_retrieval_diagnostics",
    "reset_adaptive_retrieval_metrics",
    "create_legacy_retrieve_mem",
    "get_adaptive_retriever"
]