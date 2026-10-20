# Issue #164 Evidence Report: Real-Kernel Runtime Verification Matrix

- **Issue:** [#164 — Real-Kernel Runtime Verification](https://github.com/dhar174/intelligent_data_detective/issues/164) · **PR:** [#169](https://github.com/dhar174/intelligent_data_detective/pull/169) (Draft)
- **Branch:** `antigravity/issue-164-real-kernel-validation`
- **Starting baseline (`main` post-CP4):** `08bdc2b9e61fe67423dcc4528895bedd25c6d209`
- **Tested implementation SHA:** `b7687ab1817120b1798f6fbaa9a870028a689952` (all code, notebook, patcher, workflow and harness changes). The commit that updates this report changes no runtime code.
- **Revision history:** first submitted at `de0b22c`; revised after Codex review 1 (P1-1, P1-2, P2-3, P2-4) at `490bc3d`; revised after Codex review 2 (P2: lossless pending writes in Cell 94 migration with task_path recovery, delta replay assertion, deterministic ordering regression test, and replace_put_writes_with_pass negative control) at `b7687ab`.

> [!IMPORTANT]
> **Digest convention.** All digests below are SHA-256 of the **committed git blob** (LF line endings), i.e. `git show <sha>:<path> | sha256sum`. Windows working copies created with `core.autocrlf=true` contain CRLF and hash differently unless you normalize CRLF→LF first. Cell 4 / `idd_bootstrap` already hash modules this way.

## 0. Machine-readable evidence (verified by `tests/unit/test_issue164_evidence_report.py`)

The unit test checks this block against immutable git objects at `tested_implementation_sha`. It verifies the digests, derives the test inventory from the AST and checks it, confirms that every negative-control mutation id appears in the tested smoke file, and checks that every cited run's checkout is the tested SHA (or a PR merge whose head parent is the tested SHA).

```json issue164-evidence
{
  "tested_implementation_sha": "b7687ab1817120b1798f6fbaa9a870028a689952",
  "digest_convention": "sha256 of committed git blob (LF line endings); Windows CRLF working copies must be normalized CRLF->LF before hashing",
  "artifact_sha256": {
    "IntelligentDataDetective_beta_v5_patched.ipynb": "e624dcd23d6558498af4dc87da47955bee36e8a494524cc43fed62d67567b3e7",
    "idd_manifest.json": "022b5b85cf0cad984c7f5e8fd56802e183a33af4dfaed59df5f737578631faa1",
    "idd_bootstrap.py": "7e7054b1d7c2d31aaac0352627620ba7c086887f9576a45544429be6f83e1602",
    "_patch_notebook.py": "184e055870bbab66ebb009759fa3f1873353385c9b70cbeb5f1f6386a81a2e61",
    "tests/smoke/test_real_kernel_notebook.py": "a041b6cecedb1ed27fd5d76d752a9e90770a43336e1cc04e14d756e91fc064a4"
  },
  "kernel_test_inventory": {
    "tests/smoke/test_real_kernel_notebook.py": [
      "TestRealKernelNotebookExecution::test_cell46_real_supervisor_runtime_routing_in_kernel",
      "TestRealKernelNotebookExecution::test_cell94_lossless_pending_writes_deterministic_order_and_delta_replay",
      "TestRealKernelNotebookExecution::test_cells_4_through_57_canonical_provenance_tools_supervisor_and_emergency_in_kernel",
      "TestRealKernelNotebookExecution::test_cells_60_94_96_graph_lifecycle_checkpointer_and_transitions_in_kernel",
      "TestRealKernelNotebookExecution::test_committed_notebook_static_parity_precheck",
      "TestRealKernelSecurityAndNegativeControls::test_harness_refuses_cell60_without_isolated_work_dir",
      "TestRealKernelSecurityAndNegativeControls::test_negative_controls_checkpoint_migration_and_restore_mutations_fail",
      "TestRealKernelSecurityAndNegativeControls::test_negative_controls_deliberately_broken_notebook_cells_fail_kernel_verification",
      "TestRealKernelSecurityAndNegativeControls::test_negative_controls_real_supervisor_mutations_fail",
      "TestRealKernelSecurityAndNegativeControls::test_stage4_1_missing_canonical_file_fails_closed_before_later_cells",
      "TestRealKernelSecurityAndNegativeControls::test_stage4_2_tampered_manifest_digest_rejects_before_module_import",
      "TestRealKernelSecurityAndNegativeControls::test_stage4_3_modified_idd_bootstrap_rejected_before_import",
      "TestRealKernelSecurityAndNegativeControls::test_stage4_4_already_imported_canonical_module_from_unrelated_checkout_rejected",
      "TestRealKernelSecurityAndNegativeControls::test_stage4_5_competing_module_locations_on_sys_path",
      "TestRealKernelSecurityAndNegativeControls::test_stage4_6_missing_cloned_repository_in_colab_style_environment_fails_closed",
      "TestRealKernelSecurityAndNegativeControls::test_stage4_7_corrupted_or_unexpected_patcher_source_anchors_fail_closed_in_kernel"
    ],
    "tests/smoke/test_scientific_kernel.py": [
      "test_selected_kernel_scientific_imports_and_install_flag"
    ]
  },
  "negative_controls": [
    {"test": "TestRealKernelSecurityAndNegativeControls::test_negative_controls_deliberately_broken_notebook_cells_fail_kernel_verification", "mutation_id": "Negative control caught shadowed Router in Cell 46!"},
    {"test": "TestRealKernelSecurityAndNegativeControls::test_negative_controls_deliberately_broken_notebook_cells_fail_kernel_verification", "mutation_id": "Negative control caught Cell 96 overwriting data_detective_graph!"},
    {"test": "TestRealKernelSecurityAndNegativeControls::test_negative_controls_real_supervisor_mutations_fail", "mutation_id": "raise_at_supervisor_entry"},
    {"test": "TestRealKernelSecurityAndNegativeControls::test_negative_controls_real_supervisor_mutations_fail", "mutation_id": "router_binding_replaced_with_plan"},
    {"test": "TestRealKernelSecurityAndNegativeControls::test_negative_controls_checkpoint_migration_and_restore_mutations_fail", "mutation_id": "migration_raises_and_is_swallowed"},
    {"test": "TestRealKernelSecurityAndNegativeControls::test_negative_controls_checkpoint_migration_and_restore_mutations_fail", "mutation_id": "noop_migrate_thread"},
    {"test": "TestRealKernelSecurityAndNegativeControls::test_negative_controls_checkpoint_migration_and_restore_mutations_fail", "mutation_id": "replace_put_writes_with_pass"},
    {"test": "TestRealKernelSecurityAndNegativeControls::test_negative_controls_checkpoint_migration_and_restore_mutations_fail", "mutation_id": "restore_compiled_without_checkpointer"}
  ],
  "ci_runs": [
    {
      "platform": "ubuntu-latest",
      "workflow": "copilot-setup-steps",
      "event": "workflow_dispatch",
      "run_id": 39512986907,
      "job_url": "https://github.com/dhar174/intelligent_data_detective/actions/runs/39512986907/job/118938303109",
      "head_sha": "b7687ab1817120b1798f6fbaa9a870028a689952",
      "checkout_sha": "b7687ab1817120b1798f6fbaa9a870028a689952",
      "conclusion": "success"
    },
    {
      "platform": "windows-latest",
      "workflow": "real-kernel-windows",
      "event": "pull_request",
      "run_id": 39512987322,
      "job_url": "https://github.com/dhar174/intelligent_data_detective/actions/runs/39512987322/job/118938299563",
      "head_sha": "b7687ab1817120b1798f6fbaa9a870028a689952",
      "checkout_sha": "bd0b1cce483f389885e19ec45ea2b8659ac8ac18",
      "checkout_parents": ["b54cbd1e66daa4f944cbf09a0005223bdce02dca", "b7687ab1817120b1798f6fbaa9a870028a689952"],
      "conclusion": "success"
    }
  ]
}
```

## 1. Execution matrix

| Scenario | Environment | Real kernel | SQLite mode | Result | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Linux, SQLite present | GitHub `ubuntu-latest` (image `ubuntu-24.04` 20261018.339.1, kernel `7.0.0-1017-azure`), CPython **3.12.15**, checkout = `b7687ab` | `idd-preflight` (`ipykernel 7.4.0`, `nbclient 0.11.0`, `jupyter-client 8.10.0`, `nbformat 5.11.1`) | `langgraph-checkpoint-sqlite 3.1.2` installed; sqlite_absent case blocks the import in-kernel | **PASS**: 20 passed (notebook smoke) | [run 39512986907](https://github.com/dhar174/intelligent_data_detective/actions/runs/39512986907/job/118938303109) |
| Linux, SQLite genuinely absent | same | same | `pip uninstall -y langgraph-checkpoint-sqlite` + `IDD_EXPECT_SQLITE_UNINSTALLED=1` (kernel asserts `find_spec(...) is None`) | **PASS**: 1 passed, 19 deselected (intentional `-k sqlite_absent` selection) | same run |
| Windows, SQLite present | GitHub `windows-latest` = **Windows Server 2025** (`Windows-2025Server-10.0.26100-SP0`, image `windows-2025-vs2026`), CPython **3.12.10** (MSC v.1943 AMD64). **PR merge checkout** `bd0b1cc`, parents `b54cbd1` (main) + `b7687ab` (head) | `idd-preflight` (same pinned Jupyter stack) | installed / in-kernel import block | **PASS**: 20 passed | [run 39512987322](https://github.com/dhar174/intelligent_data_detective/actions/runs/39512987322/job/118938299563) |
| Windows, SQLite genuinely absent | same | same | package uninstalled + `IDD_EXPECT_SQLITE_UNINSTALLED=1` | **PASS**: 1 passed, 19 deselected | same run |
| Windows local (author) | `Windows-11-10.0.26340-SP0`, CPython 3.12.2 | `py312-codex` (`ipykernel 7.2.0`, `nbclient 0.10.4`, `jupyter_client 8.8.0`, `nbformat 5.10.4`) | installed / in-kernel import block | **PASS** (unversioned local log, not independently verifiable): 23 passed (both smoke files, including deterministic task_path order test). All 6 mutation controls passed. | local only |
| Google Colab clone | — | — | — | **NOT EXECUTED**: no live Colab runtime was used. The simulated missing-clone test (Stage 4 #6) is **not** Colab evidence. | runbook §7 |

**Merge base note.** Since CP4, `main` has advanced to `b54cbd1` through docs-only PR #168 (`FEATURES.md`, `README.md`, and Linux workflow triggers commented out). That is why the Linux workflow runs by `workflow_dispatch` on the branch head. The Windows PR job ran on the merge with that docs change. #168 touches no runtime code.

**Key package versions (both hosted jobs):** `langgraph 1.2.16`, `langgraph-checkpoint 4.3.0`, `langgraph-checkpoint-sqlite 3.1.2`, `langchain-core 1.6.12`, `pandas 3.0.6`, `numpy 2.5.4`, `scipy 1.18.1`, `matplotlib 3.11.2`, `pytest 9.1.1`. **Author local:** `langgraph 1.2.16`, `langgraph-checkpoint 4.3.0`, `langgraph-checkpoint-sqlite 3.1.2`, `langchain-core 1.6.12`, `pandas 2.2.1`, `numpy 1.26.4`, `pytest 8.4.2`.

## 2. Other suites on the tested SHA

| Suite | Linux run 39512986907 | Windows run 39512987322 | Author local (Windows 11) |
| :--- | :--- | :--- | :--- |
| `validate_notebook_integrity.py` (99 cells) | PASS | PASS | PASS |
| `validate_graph.py` (15 nodes / 33 edges / 21 conditional / 70 fields) | PASS | PASS | PASS |
| `tests/unit/test_core_parity_guard.py` | in unit suite below | 38 passed | 38 passed |
| `test_validate_run.py tests/unit tests/integration` | 695 passed, 3 skipped | not run by this job | 689 passed, 9 skipped |
| `tests/smoke/test_scientific_kernel.py` | 3 passed | 3 passed | 3 passed |
| root regression + memory | 110 passed | not run | 127 passed (root + memory + prompt + integrity) |
| prompt templates | 17 passed | not run | included above |
| `test_error_handling_framework.py` (mandatory) | 16 passed | 16 passed | 16 passed |
| `black --check` / `flake8` (CI file set) | green | — | green on changed test files |

**Skips.** On Linux the 3 skips are inherited: the archival Run 88 fixture is missing (`test_validate_run.py`), and 2 optional `pyarrow` cases. Locally there are 6 more POSIX-permission cases (skipped on NTFS). The hosted Windows job runs only the narrower suites listed above. The smoke suites have 0 skips. The 19 deselections in the uninstall steps come from intentional `-k` selection.

## 3. What executes vs. what is mocked, per scenario

All selected cells are read from the committed notebook (`load_patched_notebook`) and executed with their full source by `nbclient.NotebookClient` in **one real `ipykernel` per scenario**. The kernel starts in the checkout so Cell 4 can locate the verified repository. In every scenario that reaches later cells, the harness inserts an `os.chdir(<tmp work_dir>)` cell immediately after Cell 4. These are notebook-list indices: intermediate and live cells (11, 37, 49–55, 62+) are **not** executed. Preserved state is the selected-cell namespace plus injected setup, not a full sequential notebook run.

| Scenario (test) | Real notebook cells | Real behaviour exercised | Mocked / replaced |
| :--- | :--- | :--- | :--- |
| `test_cells_4_through_57_…` | 1, 4, 7, 10, 16, 19, 22, 28, 31, 32, 46, 57 | Cell 4 bootstrap/manifest/origins, canonical identity (`is`) for Cells 7/10/16/19/22, 70-field State, real decorated `drop_column` and `get_dataframe_schema` under `override_global_registry`, structured missing-DataFrame error, `_resolve_artifact_path` traversal rejection before mkdir, real `emergency_correspondence_node` → `analyst` | Cell 46 is checked by **closure inspection only** (`Router is idd_graph.Router`). Its runtime is covered in the next row. The factory gets `MagicMock` LLMs. |
| `test_cell46_real_supervisor_runtime_routing_in_kernel` | 1, 4, 7, 10, 16, 19, 22, 28, 31, 32, 46 | The **real callable returned by `make_supervisor_node`**, including the W4 wrapper, is invoked inside a real LangGraph `StateGraph(State)` runtime. **Model-driven path:** real planning → to-do → routing. The test asserts that `with_structured_output` received `idd_graph.Router` (identity) with `strict=True, method="json_schema"`, that the `Plan`, `ToDoList` and `Router` invocations happened in that order, and that the result is `Command(goto="analyst")` with no `ROUTING FALLBACK`/`WARNING`. **Shortcut path:** `data_cleaner`, with zero model calls. **W4 cap:** `FINISH`. | The 8 chat-model slots are a structured-output boundary stub that returns schema-valid `Plan`/`ToDoList`/`Router` objects and raises on unknown schemas. Cell 37's memory/embedding helpers (`enhanced_mem_text`, `update_memory_with_kind`, `query_embed_func`) are stubbed because they need OpenAI embeddings. |
| `test_cells_60_94_96_…[sqlite_present\|sqlite_absent]` | 1, 4, 7, 10, 16, 19, 22, 28, 31, 32, 46, 57, 60, 94, 96 | Real `build_graph` (15/33/21) with `SqliteSaver` or `MemorySaver`. **Before Cell 94**, the production graph is invoked under the exact `run_config` thread, seeding ≥10 checkpoints (18 observed) including mid-run pending routing. **Cell 94** copies every seeded checkpoint: the count equals the seeded history, and a fresh `SqliteSaver` read of `checkpoints_demo.sqlite` matches the source on ids, parents, `next`, task names, step/source, all latest values, and ordered pending writes. **Cell 96** reads the restored state and history while its connection is open (ids and `next` match the source), and its connection is then closed. Production history is unchanged and the production connection is usable. A second production run after 94/96 completes. In MemorySaver mode, Cells 94/96 print their skip notices, no demo DB is created, and history is preserved. Real Cell 57 `viz_join`, `report_join` and `emergency_correspondence_node` run, along with the real `idd_graph` routers. | The **supervisor node is scripted** in this scenario: the real Cell 46 factory is called, but its returned callable is not used for routing. The 10 LLM-calling worker bodies (`initial_analysis_node`, `data_cleaner_node`, `analyst_node`, `visualization_orchestrator`, `viz_worker`, `viz_evaluator_node`, `report_orchestrator`, `section_worker`, `report_packager_node`, `file_writer_node`) are deterministic stubs. `update_memory_with_kind` is stubbed, `RUNTIME` is a `MagicMock`, and the 8 LLM globals are `MagicMock`. Report paths returned by stubs do **not** establish artifact generation. |
| Stage 4 #1–#7 | 1, 4 (+ a sentinel cell) | Fail-closed startup. Rejected code is never imported or executed (canary files are asserted absent). | Colab is simulated by a `google.colab` entry in `sys.modules` (#6). |

## 4. Negative controls (in-memory notebook mutations; committed notebook untouched)

| Control | Mutation | Required failure (asserted) |
| :--- | :--- | :--- |
| A | Cell 46: local `class Router` shadows `idd_graph.Router` | closure identity assertion |
| B | Cell 96: append `data_detective_graph = object()` | graph-identity assertion (runs under the checkout canary) |
| `raise_at_supervisor_entry` | first statement of real `supervisor_node` raises | `REVIEW_MUTATION_SUPERVISOR_BODY_NEVER_WORKS` |
| `router_binding_replaced_with_plan` | routing `with_structured_output(Router,…)` → `Plan` | `ROUTING FALLBACK` (the masked fallback routes to `initial_analysis`, not `analyst`) |
| `migration_raises_and_is_swallowed` | Cell 94 migration call → raise inside try/except that prints WARNING | `AssertionError: (0, N)` (migrated ≠ seeded) |
| `noop_migrate_thread` | `migrate_thread` returns 0 immediately | `AssertionError: (0, N)` |
| `replace_put_writes_with_pass` | Cell 94 migration drops pending writes (`put_writes` -> `pass`) | `Migrated history differs from source history` (pending_writes signature mismatch) |
| `restore_compiled_without_checkpointer` | Cell 96 `checkpointer=_restore_cp` → `None` | `No checkpointer set` |
| harness | Cell 60 requested without `work_dir` | `ValueError` before any kernel starts |

## 5. Production changes in this revision (via `_patch_notebook.py`; notebook regenerated, 99 cells)

1. **Cell 94 (`ISSUE164-CELL94-LOSSLESS-MIGRATE`).** The inherited CP4 migration replayed snapshots through `update_state(as_node=metadata["writes"][-1])` and skipped any snapshot without `writes`. Current LangGraph no longer records `writes` (observed metadata: `{'source': 'loop', 'step': N, 'parents': {}}`), so **every snapshot was skipped and zero were migrated, silently**. Replaying full values through `update_state` would also re-apply reducers (`operator.add` channels would duplicate). The cell now copies `CheckpointTuple`s at the checkpointer level (`put` with parent linkage, plus `put_writes` for pending writes) and reports `_migrated_checkpoint_count`. In addition, `migrate_thread` preserves execution-path order for pending writes by recovering the source checkpointer's `task_path` (via `src_cp.cursor(transaction=False)` or `src_cp.writes`), with a deterministic monotonic fallback `f"mig_{task_seq:08d}"`, preventing arbitrary task-UUID reordering across checkpoints. The demo DB stays isolated in `checkpoints_demo.sqlite`.
2. **Cell 96 (`ISSUE164-CELL96-VERIFIED-RESTORE`).** It now reads `get_state` and `get_state_history` from the restored graph **before** closing its connection, and records `_restored_snapshot`, `_restored_history_ids` and `_restored_next`. Graph non-replacement and connection closure are unchanged.
3. **`tests/unit/test_core_parity_guard.py`.** The CP4 static check that pinned the legacy `_FIX_U_SKIP_NONE_WRITER` marker is replaced by assertions for the lossless copy (no `update_state(`) and for the restore read happening before `close()`.

The canonical modules, manifest, bootstrap, prompts, model selection and graph topology are unchanged.

## 6. Filesystem isolation (P2-3)

- `execute_cells_in_real_kernel` refuses to run Cell 60 without a `work_dir` and chdirs into it after Cell 4.
- Connection and tempdir cleanup runs in a harness-level `finally`, even after an expected cell failure.
- Every kernel run snapshots the checkout's top-level files and the `checkpoints*.sqlite*` digests, and fails if anything is created, removed or modified.
- The `checkout_checkpoint_canary` fixture seeds (or reuses) a root `checkpoints.sqlite` with a `reviewer_canary` table. It applies to the positive lifecycle test, negative controls A/B and the migration/restore mutation controls, so it covers both success and failure paths. It asserts unchanged bytes and schema and no new checkpoint/WAL/SHM files.

## 7. Google Colab clone procedure (**NOT EXECUTED**)

Pinned to the tested commit, not a moving branch:

```bash
!git clone https://github.com/dhar174/intelligent_data_detective.git /content/intelligent_data_detective
%cd /content/intelligent_data_detective
!git checkout b7687ab1817120b1798f6fbaa9a870028a689952
!python -m pip install -q langchain langchain-core langchain-community langchain-openai langchain_experimental \
  langgraph langgraph-checkpoint-sqlite langmem kagglehub pandas numpy scipy scikit-learn matplotlib seaborn \
  pydantic python-dotenv pyyaml tiktoken openpyxl tavily-python chromadb joblib typing_extensions \
  xhtml2pdf pypdf pymupdf pillow markdown bleach pytest \
  nbclient==0.11.0 nbformat==5.11.1 jupyter-client==8.10.0 ipykernel==7.4.0
!python -m ipykernel install --user --name idd-preflight --display-name "IDD scientific smoke"
!IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight python -m pytest tests/smoke/ -v
!python -m pip uninstall -y langgraph-checkpoint-sqlite
!IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight IDD_EXPECT_SQLITE_UNINSTALLED=1 \
  python -m pytest tests/smoke/test_real_kernel_notebook.py -k sqlite_absent -v
```

Standalone single-file Colab acquisition (no clone) is out of scope here and belongs to #166.

## 8. Reproduce locally

```bash
python _patch_notebook.py && git diff --exit-code IntelligentDataDetective_beta_v5_patched.ipynb
python -m ipykernel install --user --name idd-preflight
IDD_RUN_KERNEL_SMOKE=1 IDD_KERNEL_NAME=idd-preflight python -m pytest tests/smoke/ -v
python -m pytest tests/unit/test_core_parity_guard.py tests/unit/test_issue164_evidence_report.py test_error_handling_framework.py -v
```

## 9. Remaining limitations

- Colab: **NOT EXECUTED**.
- No paid inference was run. Cells 11, 37, 49–55 and 62+ are not executed, and CP5 end-to-end report readiness and `validate_run.py`/`validate_artifact_quality.py` gates are **not** established by these no-key tests.
- Supervisor coverage is representative: one model-driven Router path plus the shortcut and turn-cap paths. The reply/emergency-reroute branches of `supervisor_node` (which use `MessagesToAgentsList`/`ConversationalResponse`) are not invoked. The full-graph transition scenario uses a scripted supervisor.
- Barrier coverage uses single-worker fan-out (one viz worker, one report section). There is no multi-worker barrier guarantee.
- Author-local results are not versioned logs. Hosted runs are the authoritative evidence.
