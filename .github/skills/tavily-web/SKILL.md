---
name: tavily-web
description: "Web search, content extraction, crawling, and research capabilities using Tavily API. Use when you need to search the web for current information, extracting content from URLs, or crawling websites."
risk: critical
source: community
date_added: "2026-02-27"
---

# tavily-web

## Overview
Web search, content extraction, crawling, and research capabilities using Tavily API

## When to Use
- When you need to search the web for current information
- When extracting content from URLs
- When crawling websites

## Setup and Configuration

This repository includes `tavily-python` in its dependencies. Ensure your environment variable is set:

```bash
export TAVILY_API_KEY="tvly-your-api-key"
```

If installing into a new Python environment:
```bash
pip install tavily-python
```

## Python Usage Examples

### 1. Basic Web Search
```python
import os
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# Execute targeted search with AI-curated answer
response = tavily.search(
    query="Current state of LangGraph multi-agent systems 2026",
    search_depth="advanced",
    max_results=5,
    include_answer=True,
)

print("Answer:", response.get("answer"))
for result in response.get("results", []):
    print(f"- {result['title']}: {result['url']}")
```

### 2. Context Extraction for RAG
```python
# Extract clean, concise context directly suitable for LLM prompts
context = tavily.get_search_context(
    query="LangGraph StateGraph reducer patterns",
    search_depth="advanced",
    max_tokens=2000,
)
```

### 3. Direct Q&A Search
```python
# Direct question answering for quick fact checks
answer = tavily.qna_search(query="What is the latest stable release of LangGraph?")
print(answer)
```

## Best Practices
- Always pass API keys via the `TAVILY_API_KEY` environment variable; never hardcode credentials.
- Use `search_depth="advanced"` for high-relevance technical queries, or `"basic"` for rapid latency-sensitive queries.
- Scope search domains with `include_domains` when verifying specific libraries or documentation sites.

## Related Skills
- context7-auto-research, exa-search, firecrawl-scraper, codex-review

## Limitations
- Use this skill only when the task clearly matches the scope described above.
- Do not treat the output as a substitute for environment-specific validation, testing, or expert review.
- Stop and ask for clarification if required inputs, permissions, safety boundaries, or success criteria are missing.
