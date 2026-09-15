<!-- repo-agent-bootstrap:file-kind=memory-bank -->
<!-- repo-agent-bootstrap:provenance=repo-agent-bootstrap@2026-04-20 -->
<!-- repo-agent-bootstrap:managed:start -->
# Product Context

## Problem
Data analysts spend substantial time on manual data cleaning, exploratory analysis, and report authoring. IDD eliminates this overhead by driving all stages autonomously: the analyst describes what they want to learn, and the system produces a complete, verified report.

## Target users
- Data scientists and analysts who work with CSV or tabular datasets.
- Contributors extending or debugging the IDD agent pipeline.
- AI coding agents that need accurate repo context to make safe changes.

## Core user goals
1. Upload a dataset, describe an analysis objective, receive a complete multi-section report.
2. Have the system detect data quality issues and resolve them automatically before analysis.
3. Get visualisations and statistical findings without writing any code.

## Input / output flow
```
Input:  CSV file + natural-language question
         ↓  (supervisor routes)
Process: Initial Analysis → Data Cleaner → Analyst → Visualization (parallel workers)
         → Viz Evaluator → Report Orchestrator → Section Workers → Report Packager
         ↓  (file writer)
Output: HTML/PDF report + chart image files in artifacts directory
```

## Non-goals
- No real-time streaming dashboard or web UI.
- No REST API or microservice deployment.
- No multi-user concurrency (single-user notebook execution).
<!-- repo-agent-bootstrap:managed:end -->
