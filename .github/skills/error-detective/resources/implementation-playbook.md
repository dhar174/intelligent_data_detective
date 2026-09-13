# Error Detective Implementation Playbook

This playbook provides actionable, step-by-step procedures, checklists, and code snippets for conducting rigorous error investigations across logs, traces, and codebases.

---

## 1. Investigation Scoping & Evidence Preservation

Before making any modifications or triggering restarts, secure the existing evidence to prevent log rotation or volatile state loss.

### Scoping Checklist
- [ ] Identify affected components, services, and environments (production, staging, CI).
- [ ] Determine the start and end boundaries of the failure window (UTC timestamps).
- [ ] Record incident impact (error rates, latency spikes, affected user identifiers).
- [ ] Verify permissions and adhere to redaction policies before reading or copying logs.

### Evidence Preservation Commands
```bash
# Capture raw log slice for the incident window (preventing loss to log rotation)
journalctl -u service-name --since "2026-09-12 18:00:00" --until "2026-09-12 20:00:00" > incident_raw.log

# For containerized environments:
docker logs --since "2026-09-12T18:00:00Z" --until "2026-09-12T20:00:00Z" container-id > incident_docker.log
```

---

## 2. Timeline Construction & Error Normalization

Correlate logs chronologically and normalize disparate log formats into a single unified timeline.

### Normalization Pattern
Extract timestamp, service, severity level, message signature, and correlation IDs (e.g., `trace_id`, `request_id`).

```python
import re
from datetime import datetime

LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+"
    r"\[(?P<level>[A-Z]+)\]\s+"
    r"\[(?P<service>[\w\-]+)\]\s+"
    r"(?:\[trace_id=(?P<trace_id>[a-f0-9]+)\]\s+)?"
    r"(?P<message>.*)$"
)

def normalize_log_line(line: str) -> dict | None:
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None
    return match.groupdict()
```

### Signature Clustering
Group recurring messages by stripping dynamic variables (UUIDs, IP addresses, numbers):
```python
def extract_error_signature(msg: str) -> str:
    # Replace UUIDs
    sig = re.sub(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", "<UUID>", msg)
    # Replace IP addresses
    sig = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IP>", sig)
    # Replace hexadecimal memory addresses or hashes
    sig = re.sub(r"0x[a-f0-9]+", "<HEX>", sig)
    # Replace numeric IDs
    sig = re.sub(r"\b\d+\b", "<NUM>", sig)
    return sig
```

---

## 3. Stack Trace Analysis Across Runtimes

Deconstruct stack traces from the innermost point of failure back through application and framework code.

### Python Traceback Deconstruction
1. **Identify the root cause exception**: Check the final line first (`ValueError`, `AttributeError`, `TypeError`, `KeyError`).
2. **Trace chained exceptions**: Note `The above exception was the direct cause of the following exception:` (`raise ... from exc`) vs `During handling of the above exception, another exception occurred:` (unhandled secondary error).
3. **Filter out library frames**: Distinguish project code (`/app/...`, `intelligent_data_detective/...`) from third-party vendor frames (`site-packages/...`).
4. **Inspect frame variables**: If available via Sentry, local post-mortem debugging (`pdb.post_mortem()`), or pytest traceback dumps.

```bash
# Run failing pytest test with interactive post-mortem debugging
pytest tests/ -k "test_name" --pdb --tb=short
```

---

## 4. Hypothesis Generation & Testing

Use the scientific method to isolate root causes without trial-and-error changes:

1. **Formulate Hypotheses**:
   - *H1 (Code Change)*: Did recent commits introduce unexpected NoneType or schema change?
   - *H2 (State/Data Drift)*: Did specific input data trigger an unhandled edge case?
   - *H3 (Dependency/Environment)*: Did an upstream API or underlying dependency update break behavior?
2. **Design Falsification Tests**:
   - Construct a minimal reproducible test case (MRE) that fails on the buggy version and passes when the hypothesized root cause is addressed.
   - Test against isolated unit tests rather than full multi-minute pipeline runs.

---

## 5. Cross-Component Correlation & Distributed Traces

In distributed multi-agent systems (e.g., Supervisor -> Worker -> Tool), single errors often manifest as cascaded failures.

### Correlation Protocol
1. Follow the `run_id`, `thread_id`, or `task_id` across state transitions.
2. In LangGraph/IDD architectures, examine agent node outputs in graph state:
   - Check if an earlier agent passed a partial, truncated, or invalid payload that broke a subsequent agent.
   - Inspect tool input validation errors before inspecting LLM generation errors.

---

## 6. Fix Verification & Non-Regression Testing

Ensure any candidate fix completely resolves the root issue without introducing secondary defects:

- [ ] Run the minimal reproducible test case: verify it passes.
- [ ] Run the entire test suite:
  ```bash
  pytest test_intelligent_data_detective.py test_error_handling_framework.py test_memory_categorization.py test_memory_integration.py test_memory_lifecycle.py -v
  ```
- [ ] Verify error logging preserves clean exceptions and doesn't swallow unexpected failure types.

---

## 7. Privacy-Conscious Reporting & Completion Checklist

When generating incident reports or root-cause analyses:

### Privacy Rules
- Redact customer data, email addresses, tokens, and PII.
- Only report schema shapes, error classifications, and sanitized sample values.
- Never output full raw database rows or API tokens in post-incident documentation.

### Investigation Completion Checklist
- [ ] Exact root cause identified with primary evidence (log lines, stack trace, commit diff).
- [ ] Scope and timeframe of impact documented.
- [ ] Minimal reproducible test committed to prevent regression.
- [ ] Remediation applied and validated against all test gates.
- [ ] Preventive measures (defensive guards, schema validation, lint rules) documented.
