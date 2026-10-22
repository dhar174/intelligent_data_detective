# CP5 #177: native completion and strict recovery boundary

This owner-authorized follow-up extends PR #176 directly on
`codex/cp5-viz-join`, starting from `622287204010cadef02190a592b139e2fe5eec6e`.
Its accepted main base is `b8da639b63a3aa604630b4cc1700dae2a88440ef`.
Antigravity's earlier approval applies to `6222872`; this extension needs a new
independent review. Neither issue #177 nor parent #165 is closed automatically.

## Source-grounded findings

1. All seven generated Cell 57 safe wrappers attempted the legacy structured-output
   final hop before checking `IDD_ALLOW_RECOVERY`. Visualization, evaluation and
   report-outline wrappers also retried transient-looking errors before checking
   that flag. Thus strict proof execution could issue another provider request
   and return a recovery result after native failure.
2. Both the cleaner node and its rendered prompt template instructed the model to
   call `respond`, but its actual `ToolStrategy` response tool is `CleaningMetadata`.
   The unknown-tool middleware does not recognize `respond`. A controlled model
   following that instruction reaches the actual 160-step graph limit; a valid
   `CleaningMetadata` call terminates the real inner agent in one model step.
3. The cleaner and visualization wrappers discarded the fully rendered runtime
   SystemMessage. The cleaner's stable factory prompt nevertheless referred to
   that context. The visualization factory supplied its raw template, leaving
   placeholders such as `{visualization_task}` and `{output_format}` unrendered.
4. The visualization factory returns `VisualizationResults`, while `viz_worker`
   requested and handled only `DataVisualization`. The native envelope therefore
   remained in the inner agent's `structured_response`, a channel intentionally
   absent from the outer canonical State, instead of contributing figure records.

These are independently reproduced source defects. They do **not** establish the
exact message/tool sequence responsible for the preserved live run's loops.

## Correction and invariants

The active `CP5-NATIVE-COMPLETION` patch runs after the legacy transformations.
Every wrapper immediately re-raises its original native exception unless
`IDD_ALLOW_RECOVERY` is exactly `1`. This happens before wrapper retry sleeps,
final-hop calls, recovery scanning or fallback-result construction. Exception
identity and traceback, the existing 160-step cap and configurable thread ID are
preserved. KeyboardInterrupt and SystemExit continue to propagate.

The evaluator node's outer exception handler has the same strict guard. Two
additional failing regressions showed that it otherwise caught the wrapper's
re-raised error and force-approved the figures. Strict failures now reach the
runner without fabricating an evaluation grade; explicit legacy opt-in retains
the original detectable recovery branch.

Explicitly enabled legacy recovery retains its original final-hop calls and
prohibited markers. It remains unsuitable for strict CP5 acceptance. The marker
scan and both production validators are unchanged.

The cleaner's node and template name `CleaningMetadata`. Cleaner and visualization
runtime context is carried as a HumanMessage named `runtime_context`, preserving
content and message IDs without adding a second system role. Visualization uses
a stable factory instruction and requests its existing `VisualizationResults`
schema, with real `DataVisualization` children.
The rendered task includes the actual assigned VizSpec ID, so the model can
return it directly; the worker no longer rewrites a returned figure's ID.

The worker validates canonical envelopes/dictionaries and accepts legacy single
figures. Empty, malformed, missing or unfinished required responses fail. It
publishes only its new flat figure records and paths through the existing
reducers, preserving source IDs and metadata. It neither replays inherited
aggregation history nor copies/scans files to manufacture a result. The existing
join retains ownership checks and publishes the aggregate inventory.

Notebook changes are confined to Cells **25, 44 and 57**. Cell 25 was additionally
required because the real node test exposed the contradictory rendered cleaner
template after the first correction. Other cells retain their order and content.
Canonical modules, schemas, reducers, graph, provider configurations, bootstrap,
manifest, proof runner and validators are unchanged.

## Offline verification design

`tests/unit/test_native_completion.py` executes actual generated functions,
actual decorated tool catalogs from Cells 31/32/42, real factory wiring,
ToolStrategy, unknown-tool guards and pre-model summarization middleware. Only
chat/summary model responses and embedding-backed memory are controlled; the
store has no embedding index. Outbound sockets are denied. Controlled completion
cases do not invoke business tools or claim to simulate provider behavior.

The wrapper matrix covers all seven roles, unset/disabled/non-opt-in recovery
flags, recursion/ordinary/transient-looking errors, explicit recovery opt-in,
native success and interrupts. Native agent tests cover correct response-tool
termination, an instruction-following cleaner and a deliberately nonterminating
unknown-tool loop. Worker cases cover models, dictionaries, legacy single figures,
delta aggregation and invalid completion.

The SQLite integration additionally starts with an empty figure inventory and
uses canonical visualization dispatch to execute four real native worker agents.
Each controlled model verifies that it receives its actual task and assigned
figure ID, then returns one distinct owned test figure. The actual join,
task evaluator node and wrapper, SQLite typed decoding/reopening, report
orchestrator, section dispatch and section workers run to the packaging boundary.
The existing W14J renderer resolves four owned PNGs and writes temporary test-only
Markdown, HTML and PDF. Original invented-ID, missing-file, external-file,
stale-PNG and fewer-than-three controls remain rejecting.

The attested child-kernel test reruns the focused cases. Three additional kernel
cases raise native recursion errors (including through the evaluator node) and
verify strict first-error stopping before
both a recovery-boundary sentinel and a subsequent notebook cell. Synthetic
execution remains `cp5_pass=False`.

## Verified results

The final regression harness against immutable generated cells from `6222872`
produces **79 failures / 28 passes**. The first capture had 77 failures; two
additional cases established the evaluator outer-handler escape before its
correction. The assigned-ID assertion also failed against the otherwise repaired
native graph until the actual VizSpec ID was included in the rendered request.
These failures were observed before the corresponding production corrections.

Reproduce the original source with the current test harness without checking out
or modifying historical files:

```python
import json, subprocess, pytest
import tests.unit.test_native_completion as suite
baseline = json.loads(subprocess.check_output([
    "git", "show",
    "622287204010cadef02190a592b139e2fe5eec6e:IntelligentDataDetective_beta_v5_patched.ipynb",
]))
suite.notebook = lambda: baseline
raise SystemExit(pytest.main(["tests/unit/test_native_completion.py", "-q"]))
```

Final local Linux / Python 3.12.14 checks:

```bash
python -m pytest tests test_*.py tools/diagnostics/test_cp0_diagnostics.py \
  --ignore=tests/trajectory --ignore=tests/smoke -q -ra
# 1,148 passed, 1 skipped (absent historical Run 88 fixture)
IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight \
  python -m pytest tests/smoke -q -ra
# 51 passed, no skips
python validate_notebook_integrity.py IntelligentDataDetective_beta_v5_patched.ipynb
python validate_graph.py --notebook IntelligentDataDetective_beta_v5_patched.ipynb
# 99 cells; 15 nodes, 33 edges / 21 conditional, 70 State fields
```

The broad total includes 107 native-completion cases, 26 join cases, 10 reporting
integration cases (143 focused cases), all 38 parity, 16 mandatory error-handling,
22 core, 118 patcher-integrity, 76 proof-runner unit, 54 CP0 and existing report,
state, routing and integrity regressions. These selections overlap and must not
be added to the broad count. The child-kernel test's nested 143 cases are likewise
not another set of independent cases.

One intermediate broad run failed because the scripted model searched all retained
inner-agent history for its assigned task. It now follows the newest explicitly
named runtime context; the subsequent full suite and real-kernel suite passed.
No production routing, checkpoint contract or validation gate was changed to
accommodate that harness correction. Four broad warnings concern upstream
LangChain/experimental-community, LangGraph `Send`, and `fitz` deprecations.

Two regenerations in fresh owned directories are byte-identical to the candidate:
notebook SHA-256
`0d430c58e1ab0a83dcf430ac9e612869057b4602ff54b98953eef0e1eff18077`.
All 99 cell positions, IDs and other metadata are retained; only Cells 25, 44 and
57 differ. The approved `viz_join` body is unchanged from `6222872`. All seven
manifest digests and module origins verify; canonical modules, source notebook,
manifest, proof runner and validators are byte-identical to accepted main.
Black/Flake8 checks on the touched tests and `git diff --check` pass.
Validation-created SQLite files were preserved outside the isolated checkout
before kernel checks; no historical checkpoint was used or modified.

Published commit and Linux/native Windows CI URLs and conclusions are recorded
in PR #176 so their status can be updated without changing the tested source.

## Historical marker disposition and limitations

Antigravity's [independent review](https://github.com/dhar174/intelligent_data_detective/pull/176#pullrequestreview-5616346002)
reported prohibited `W2-BA-finalhop[DC]` at console line 906 and
`W2-BA-finalhop[VZ]` at lines 1930 and 2138 in preserved
`attempt_live_retail_orders_20261022_run1`, after the inner 160-step limit.
Those findings independently invalidate that historical attempt. The restricted
Windows raw evidence is unavailable on this Linux host: these historical claims
are attributed to Antigravity, not represented as personally re-read evidence.

No historical attempt, artifact, checkpoint, ledger or memory-bank content is
modified. No live inference, embeddings, credential discovery, paid requests,
campaign resume, merge or release is authorized by this repair. Provider spend
is **$0**. Offline controlled responses cannot establish clean native behavior
for an actual provider or certify report substance. Hosted ToolStrategy paths
are exercised; local-model configuration remains unchanged.

Live CP5 proof stays blocked pending independent review and Charles's separate
authorization for a newly approved SHA and fresh attempt. First-error stopping,
child interpreter/source attestation, run/thread/log association, isolation,
full-log scans, 12/12 production and 9/9 artifact gates, at least three real
distinct figures, complete HTML/Markdown/PDF integrity and human substance review
all remain required.
