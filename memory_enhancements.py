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
"""

import time
import uuid
from typing import Dict, List, Optional, Union, Literal, Any
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState

# Type definitions for memory kinds
MemoryKind = Literal["conversation", "analysis", "cleaning", "visualization", "insights", "errors"]

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

def put_memory(
    store: InMemoryStore,
    kind: MemoryKind,
    text: str,
    meta: Optional[Dict[str, Any]] = None,
    user_id: str = "user"
) -> str:
    """
    Store a memory item with categorization.
    
    Args:
        store: The InMemoryStore instance
        kind: Type of memory (conversation, analysis, cleaning, etc.)
        text: The memory content
        meta: Optional metadata dictionary
        user_id: User identifier for namespace isolation
        
    Returns:
        The unique memory ID that was created
    """
    namespace = ("memories", kind)
    memory_id = str(uuid.uuid4())
    
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
    user_id: str = "user"
) -> List[Dict[str, Any]]:
    """
    Retrieve memories with optional kind filtering and fallback.
    
    Args:
        store: The InMemoryStore instance
        query: Search query text
        kinds: List of memory kinds to search (if None, searches all kinds)
        limit: Maximum number of results (if None, uses default from config)
        user_id: User identifier for namespace isolation
        
    Returns:
        List of memory items, ranked by similarity
    """
    results = []
    
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
    
    return put_memory(memstore, kind, text, user_id=user_id)


# Export key functions for notebook integration
__all__ = [
    "put_memory",
    "retrieve_memories", 
    "format_memories_by_kind",
    "enhanced_retrieve_mem",
    "enhanced_mem_text",
    "update_memory_with_kind",
    "MemoryKind",
    "MEMORY_CONFIG"
]