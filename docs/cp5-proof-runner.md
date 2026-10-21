# CP5-A strict runner and evidence contract

Parent: [#165](https://github.com/dhar174/intelligent_data_detective/issues/165).
Baseline: `761534391585556d94ae9b7bd465d3e93c2e7702` (merged #169).
Independent review: Antigravity, followed by Charles's acceptance.

**Authorized live spend remains $0.** This change implements execution controls,
not authorization to execute a live CP5 campaign. The production notebook,
patcher, canonical pipeline, models and validators are unchanged. Existing
non-proof behavior remains available; it does not acquire proof guarantees.

## Preparation (no keys, no production cells)

Use a clean, approved commit containing this implementation, a provisioned
Python 3.12 environment and an existing parent directory for attempts. The
source checkout is read, never cleaned or executed by the proof path. A fresh
attempt receives only the explicitly inventoried committed runtime files;
historical outputs, `.env` files and user configuration are not copied.

Provision a kernel with an absolute interpreter in its kernelspec:

```bash
python -m ipykernel install --user --name cp5-python312
python -m jupyter kernelspec list
```

In a credential-free process, after substituting approved values:

```bash
python run_notebook_live.py --proof --prepare-only \
  --attempt-dir /absolute/existing-parent/cp5-preparation-unique \
  --source-sha APPROVED_40_CHARACTER_HEAD_SHA \
  --kernel-name cp5-python312 \
  --expected-python /absolute/path/to/python \
  --expected-python-version 3.12.14 \
  --scenario retail_orders
```

Use the actual full Python version. The selected kernel must exist, launch that
absolute interpreter directly with `-m ipykernel_launcher`, have no environment
overrides, and use the local provisioner without custom configuration. No
fallback is permitted. Preparation verifies a real child interpreter, cwd,
dependency versions, installation flag and disabled tracing through a separate
observer request. **Zero production cells execute.** No credential loader is
called and no model/embedding request is sent.

The attempt directory must not exist, even if empty. A preparation attempt
cannot later be reused for execution. Any stale checkpoint, sidecar, resume
flag, thread file, prior outputs or log in the designated runtime workspace
causes refusal. No cleanup of previous attempts or the source checkout occurs.

## Future authorized execution

First obtain Charles's explicit approval naming the exact revision, environment,
scenarios, model configuration, operator, independent auditor and spend-control
policy. Complete the accepted #164 prerequisite and current no-key gates. Pin
and hash the resulting implementation revision, not the planning baseline above.

Only then, in a process receiving an explicitly authorized `OPENAI_API_KEY`
through secure process configuration, run a new attempt:

```bash
python run_notebook_live.py --proof \
  --attempt-dir /absolute/existing-parent/cp5-retail-unique \
  --source-sha APPROVED_40_CHARACTER_HEAD_SHA \
  --kernel-name cp5-python312 \
  --expected-python /absolute/path/to/python \
  --expected-python-version 3.12.14 \
  --scenario retail_orders --credential-source process
```

Repeat with `sample_dirty` and a new directory/kernel for the distinct second
scenario. Only the notebook's existing two fixture choices are exposed. Proof
mode fixes the notebook to the committed patched target; `IDD_NOTEBOOK` does not
select an alternative. The effective Cell 51 request replaces the earlier Cell
48 template: both are hashed separately. No prompt/model/retry tuning occurs.

Proof mode rejects `--resume`, legacy preflight and LangSmith checks. It requires
installation suppression, uses `allow_errors=False`, `force_raise_errors=True`
and an empty error-name allowlist. Even `raises-exception` tags cannot authorize
continuation. A swallowed streaming traceback stops execution before the next
cell. Timeouts, kernel death, unavailable observations and runner errors fail.

Credentials come **only** from the explicitly selected process source. No
dotenv, Windows credential scope or tracing loader runs. Ambient API endpoints,
proxies, Tavily keys, LangSmith keys and client configurations are not forwarded.
Tracing is explicitly disabled. This is a source/runner isolation boundary, not
an operating-system sandbox against arbitrary malicious Python code. Runner
files, kernel connection files and configured caches live inside the attempt;
the unchanged notebook's canonical artifact-path protections still apply.

## Evidence layout and schema v1

```text
attempt/
  workspace/                 verified committed runtime snapshot; kernel cwd
    checkpoints.sqlite*      active database and optional WAL/SHM
    current_run_thread_id.txt
    notebook_run_log.txt     exclusively initialized; never append fallback
    IDD_results/IDD_run_<runtime_run_id>/...
    tmp/...                  child TemporaryDirectory and isolated caches
  evidence/                  outside the analytical run directory
    evidence.json            atomic, incrementally replaced allowlisted index
    executed.ipynb           partial or completed, original cell sources/order
    console.log              notebook stdout/stderr, including raw errors
    partial-work/...         available owned temporary artifacts after failure
    validate_run.log
    validate_artifact_quality.log
```

`evidence.json` is an index, not a certificate. Its top-level fields include:

| Field | Meaning |
|---|---|
| `schema_version`, `attempt_id`, UTC timestamps | Schema and unique attempt identity |
| `execution_kind` | Preparation, synthetic no-key test, or provider execution requiring owner authorization |
| `source_sha`, `sources` | Approved SHA; raw and CRLF-to-LF SHA-256 for each inventoried file |
| `scenario`, `credential_provenance`, `external_services` | Explicit non-secret choices; never credential values |
| `kernelspec`, `nbclient` | Verified launch specification and supported strict controls/version |
| `observations` | Per-stage request/result status and actual child-kernel facts |
| `execution` | Start/end, last cell index and completion/failure status |
| `attempt_status`, `evidence_completeness` | Overall operation result and whether evidence was captured, partial, or could not be saved |
| `provenance`, `association` | Separate integrity and run/log/output association statuses |
| `files`, `artifacts`, `output_directory` | Exact locations, file hashes and sizes where reached |
| `validators`, `run_directory_mtime_ns` | Commands, selected log/window, statuses, raw output hashes and original timestamp |
| `complete_log_scan` | Separate complete console/pipeline marker scan |
| `first_failure`, `secondary_failures` | Error class, controlled code, stage, cell index, UTC; raw exception text stays restricted |
| `kernel_cleanup` | Whether owned kernel context cleanup completed |
| `artifact_inspection`, `cp5_pass` | Independent review pending; `cp5_pass` always false |

Hash conventions are explicit: `raw_sha256` hashes bytes as stored;
`lf_sha256` hashes bytes after replacing CRLF with LF. The runtime snapshot is
written from committed Git blobs. Source hashes are frozen before execution
and checked during attestation and again after execution. The observer's file
path and hash identify its version. Python object IDs are compared only within
one kernel lifetime, never across attempts.

Observer stages (zero-based original notebook indices):

| Stage | When | Required facts |
|---|---|---|
| `identity` | Before any production cell | Absolute interpreter, full version, cwd, dependencies, safe flags and UTC offset |
| `bootstrap` | After 4 | Verified bootstrap module origin/hash |
| `canonical` | After 32 | Canonical module origins/hashes, available object identities, 70 State fields |
| `input` | After 48 | Selected input path/ID, actual file checksum, dimensions, initial template hash |
| `request` | After 51 | Effective final prompt/config hashes and allowlisted actual model settings |
| `graph` | After 60 | Kernel-local graph/checkpointer IDs, Router identity, 15/33/21 topology and resolved SQLite location |
| `runtime` | After 72 | Runtime run ID, configurable thread ID, displayed stream identifier, actual initial request/config hashes |
| `stream` | After 75 | Preserved graph/checkpointer identity |
| `migration`, `late` | After 94/96 | Preserved production graph, isolated demo path, history counts and closed restore connection, or unavailable SQLite-demo facts |

Observations use Jupyter `execute_request` with empty code, `silent=True`,
`store_history=False` and a `user_expressions` call into the versioned observer.
They do not add cells or change source. The observer never invokes the graph,
resumes it, restores state, writes configuration or seeds checkpoints. SQLite
location inspection uses read-only `PRAGMA database_list`. Missing stages remain
`NOT_REACHED`; failed/unavailable observations cannot become `VERIFIED`.

Example **synthetic** record excerpt (no live result):

```json
{
  "schema_version": 1,
  "execution_kind": "synthetic-no-key",
  "execution": {"status": "FAILED", "last_cell_index": 0},
  "observations": {"graph": {"status": "NOT_REACHED"}},
  "first_failure": {"type": "CellExecutionError", "stage": "cell", "cell_index": 0},
  "automatic_validation": "NOT_REACHED",
  "artifact_inspection": "PENDING_INDEPENDENT_REVIEW",
  "cp5_pass": false
}
```

## Association, validation and failure handling

One new directory and one kernel bind an attempt to its snapshot and log
generation. Exactly one run directory and one canonical nested reports directory
must agree with the observed runtime ID. The saved thread ID and console stream
identifier must match. Foreign run identifiers, log timestamps outside the
observed execution interval, escaping image references and symlinks are refused.
These controls do not authenticate arbitrary counterfeit lines written by an
external process with access to the same workspace; retain exclusive access.

`associate_outputs(..., execution=record["execution"])` requires the successful
runner execution record, including timezone-aware `started_utc` and `ended_utc`.
The runner records the start before kernel setup and the end after all notebook
cells, kernel context cleanup and final notebook saving. Association uses this
actual interval, rather than the Cell 75 stream observation: Cell 81 persists
artifacts and Cell 83 writes the legitimate `FINAL` pipeline record afterward.
No additional production cells or observer requests are added.

Every verified observation now retains the runner's `requested_utc` and
`received_utc` alongside the child's `utc`. The child timestamp must fall within
that request/reply bracket with at most five seconds of clock tolerance. At
association, required identity/runtime/stream observations and all other present
stages must be verified, ordered, and enclosed by the successful execution record;
missing, naive, reversed or uncorrelated timing fails closed. The explicitly
verified local provisioner shares the host clock. Clock jumps or larger skew
require investigation, not timestamp correction or a wider window.

Pipeline prefixes contain local `HH:MM:SS` at one-second precision. The interval
uses the child identity's UTC offset, with five seconds of boundary tolerance;
its lower bound is rounded down to include the intersecting first second.
Adjacent calendar dates preserve midnight rollover. Intervals spanning a full
24 hours after tolerance/rounding are refused because time-only prefixes cannot
disambiguate them. Historical lines with the same time of day cannot be dated
from their prefix alone; fresh exclusive logs, workspace ownership and identifier
bindings remain mandatory. The fixed observed UTC offset assumes no local
timezone/DST change during execution. No log timestamps are rewritten.

The existing validators run with their positional directory arguments:

```bash
python validate_run.py EXACT_RUN_DIR --log-path EXACT_PIPELINE_LOG --window 180 --json
python validate_artifact_quality.py EXACT_RUN_DIR
```

No `--latest`, pooled artifact scan, timestamp rewriting, or redundant selection
interface is used. Both results, including nonzero exits, timeouts and launch
failures, are retained. The run
directory's original mtime is recorded and checked for change. Complete logs are
scanned separately from the validator's time window.

Execution status, provenance/association, automatic checks and human inspection
remain separate. Exit zero means the selected runner operation completed; it
**never means CP5 PASS**. Each live run still needs 12/12, 9/9 and independent
inspection of HTML, Markdown, PDF and at least three genuinely distinct figures.
Two consecutive fresh successes on different inputs and the same approved SHA
are required. Failures remain visible; repairs require a newly approved revision.

Partial notebook output and available owned temporary artifacts are retained
without executing later production cells or cleanup cells. Atomic index updates
make the last saved stage recoverable. If evidence storage itself fails, the
original execution exception remains primary; a last `RUNNING` record after
interruption/save failure is **not success**. A full disk or process kill cannot
guarantee complete evidence. Check the nonzero process result and raw partials.

Raw logs, notebooks and temporary artifacts may contain source literals, model
outputs or sensitive error messages. They are **restricted, never automatically
published**. The index uses allowlisted settings and hashes prompts/configs,
never environment dumps or client representations. Review every file before
publishing sanitized copies; preserve original and sanitized hashes separately.
Do not commit raw credentials, connection files, personal inputs or bulky outputs.

## Spend-control decision and stop gate

The proposed **$5 campaign / $2.50 per attempt is not approved**. Two attempts is
a campaign policy, not a monetary guarantee. Account for all retries, embeddings,
reasoning and image-input usage, including failed attempts. Provider alerts only
notify; they do not enforce limits. Provider-enforced limits can be delayed and
must not be represented as an exact hard cap without supporting evidence.

Available approaches include operator monitoring with conservative headroom
(delayed visibility, no exact guarantee), provider project restrictions/limits
(verify actual enforcement and timing), or a separately reviewed request-level
reservation mechanism (more engineering and assumptions about in-flight usage).
This task implements none of them and changes no provider settings. Select and
approve a policy before live work; do not infer permission from this CLI.

Stop for Antigravity's independent review and Charles's acceptance. Do not merge,
close roadmap issues, release, or run CP5 under this implementation task.

## Implementation verification and limitations

The implementation was tested without provider credentials on Linux, Python
3.12.14, nbclient 0.11.0, nbformat 5.11.1, jupyter-client 8.10.0 and ipykernel
7.4.0. Synthetic proof kernels deny non-loopback socket connections while
retaining local Jupyter communication. They exercise failure paths and actual
kernel observations, not a live CP5 result. The native Windows workflow runs
the same new regressions alongside the established real-kernel harness; its
hosted result must be checked for the exact PR revision.

The broad suite deliberately excludes `tests/trajectory`: its existing smoke
module discovers Windows user-scope credentials during import. No credential
discovery was authorized. A missing historical RUN_88 fixture also skips one
root validator test.

An existing diagnostic test fails independently on untouched baseline
`761534391585556d94ae9b7bd465d3e93c2e7702`:

```bash
python -m pytest tools/diagnostics/test_cp0_diagnostics.py::test_claim_12_production_tool_cases_pass -q
```

Its extracted `delete_rows` namespace lacks `get_global_df_registry`, producing
`NameError` and diagnostic `REPRODUCED` instead of `OBSERVED INVARIANT`. A separate
correction should bind the canonical registry accessor in that diagnostic
harness and rerun its twelve tool cases. This task changes neither the
diagnostic nor production tools.

Run checkout-guarded kernel tests sequentially with graph validation. The latter
creates `checkpoints.sqlite` in its working directory; concurrent execution
correctly triggers the existing checkout mutation guard. Keep this graph-only
validation out of proof attempt workspaces.
