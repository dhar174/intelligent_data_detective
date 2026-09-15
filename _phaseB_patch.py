"""Phase B telemetry patch — adds _pl_logger bootstrap + 10 STATE log sites.
Idempotent: re-running won't double-insert (anchors check for sentinel)."""
import json, sys, shutil, datetime as dt

NB = "IntelligentDataDetective_beta_v5.ipynb"
BACKUP = f"_phaseB_backup_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.ipynb"
shutil.copyfile(NB, BACKUP)
print(f"[backup] {BACKUP}")

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)
assert len(nb["cells"]) == 98, f"expected 98 cells got {len(nb['cells'])}"

# ---------- cell 4: logger bootstrap ----------
BOOT_TAG = "_pl_logger = logging.getLogger(\"idd.pipeline\")"
BOOT = (
    "\n# --- Phase B logger bootstrap (idd.pipeline) ---\n"
    "import logging\n"
    "_pl_logger = logging.getLogger(\"idd.pipeline\")\n"
    "if not _pl_logger.handlers:\n"
    "    _pl_logger.setLevel(logging.INFO)\n"
    "    _h = logging.StreamHandler()\n"
    "    _h.setLevel(logging.INFO)\n"
    "    _h.setFormatter(logging.Formatter(\"%(asctime)s %(levelname)s %(name)s: %(message)s\"))\n"
    "    _pl_logger.addHandler(_h)\n"
    "    _pl_logger.propagate = False\n"
    "    try:\n"
    "        _fh = logging.FileHandler(\"notebook_run_log.txt\", mode=\"a\", encoding=\"utf-8\")\n"
    "        _fh.setLevel(logging.INFO)\n"
    "        _fh.setFormatter(logging.Formatter(\"%(asctime)s %(levelname)s %(name)s: %(message)s\"))\n"
    "        _pl_logger.addHandler(_fh)\n"
    "    except Exception:\n"
    "        pass\n"
    "# --- end Phase B logger bootstrap ---\n"
)
c4 = "".join(nb["cells"][4]["source"])
if BOOT_TAG not in c4:
    c4 = c4.rstrip() + "\n" + BOOT
    nb["cells"][4]["source"] = c4.splitlines(keepends=True)
    print("[cell 4] inserted logger bootstrap")
else:
    print("[cell 4] bootstrap already present, skipped")

def wrap(body_lines, indent="    "):
    """Wrap a list of code lines (no trailing \\n) in try/except _pl_logger.info(...) block."""
    out = [f"{indent}try:\n"]
    for ln in body_lines:
        out.append(f"{indent}    {ln}\n")
    out.append(f"{indent}except Exception:\n")
    out.append(f"{indent}    pass\n")
    return out

def patch_cell(cell_idx, edits):
    """edits: list of (anchor_line_text, position, telemetry_lines, sentinel)
       position: 'after' or 'before'  (line containing anchor)
       Anchor matching: substring match by default; if anchor starts with '==' the
       remainder is matched as full-line equality (whitespace-sensitive)."""
    src = "".join(nb["cells"][cell_idx]["source"])
    lines = src.split("\n")
    edits_sorted = []
    for anchor, pos, tele, sentinel in edits:
        exact = anchor.startswith("==")
        needle = anchor[2:] if exact else anchor
        idx = None
        for i, l in enumerate(lines):
            if sentinel in src:
                break
            if exact:
                if l == needle:
                    idx = i; break
            else:
                if needle in l:
                    idx = i; break
        if idx is None:
            if sentinel in src:
                print(f"[cell {cell_idx}] sentinel {sentinel!r} already present, skipping")
            else:
                print(f"[cell {cell_idx}] ANCHOR NOT FOUND: {anchor!r}")
            continue
        edits_sorted.append((idx, pos, tele, sentinel, anchor))
    edits_sorted.sort(key=lambda x: -x[0])
    for idx, pos, tele, sentinel, anchor in edits_sorted:
        insert_at = idx + 1 if pos == "after" else idx
        for j, t in enumerate(tele):
            lines.insert(insert_at + j, t.rstrip("\n"))
        print(f"[cell {cell_idx}] inserted {sentinel} {pos} line~{idx+1} ({anchor[:60]!r})")
    new_src = "\n".join(lines)
    nb["cells"][cell_idx]["source"] = new_src.splitlines(keepends=True)
    # ensure last line keeps original newline state
    if not src.endswith("\n") and nb["cells"][cell_idx]["source"] and nb["cells"][cell_idx]["source"][-1].endswith("\n"):
        nb["cells"][cell_idx]["source"][-1] = nb["cells"][cell_idx]["source"][-1].rstrip("\n")

# ============== cell 57 edits ==============
# B7 file_writer.pre_invoke — anchor: 'result = file_writer_agent.invoke('
b7 = wrap([
    '_pl_logger.info(',
    '    "STATE file_writer.pre_invoke file_type=%s file_name=%s content_len=%d report_text_len=%d written_sections_count=%d",',
    '    file_type, file_name, len(content or ""),',
    '    len(getattr(state.get("report_results"), "report_text", "") or ""),',
    '    len(state.get("written_sections", []) or []),',
    ')',
], indent="    ")

# B1 viz_worker.start — anchor: line ending '"individual_viz_task",{state.get("viz_spec", None)})'
# Use the assignment line for task. We'll insert AFTER line 858 (end of default_instruction conversion).
b1_start = wrap([
    '_pl_logger.info(',
    '    "STATE viz_worker.start task_vizid=%s task_len=%d available_df_ids=%s",',
    '    task_vizid, len(task or ""), state.get("available_df_ids", []),',
    ')',
], indent="    ")
# B1 viz_worker.end — anchor: 'update_memory_with_kind(state, state["_config"], "visualization"' (the LAST one in viz_worker, line 999)
b1_end = wrap([
    '_pl_logger.info(',
    '    "STATE viz_worker.end viz_id=%s viz_type=%s style=%s artifact_path=%s sr_present=%s",',
    '    sr.visualization_id, getattr(sr, "visualization_type", None),',
    '    getattr(sr, "visualization_style", None), getattr(sr, "path", None), sr is not None,',
    ')',
], indent="        ")

# B1-pre assign_viz_workers — anchor: 'tasks = state.get("viz_tasks", []) or []' inside that fn
b1_pre = wrap([
    '_pl_logger.info(',
    '    "STATE assign_viz_workers.send_dispatch tasks_count=%d viz_specs_count=%d",',
    '    len(state.get("viz_tasks", []) or []), len(state.get("viz_specs", []) or []),',
    ')',
], indent="    ")

# B2 viz_join — anchor: 'update_memory_with_kind(state, state["_config"], "visualization", in_memory_store or get_store(), text=memory_text)'
# Need the one inside viz_join (line 1058). Use unique pattern: 'text=memory_text)\n\n    return {' nearby. Use 'last_agent_id": "viz_join"' return as anchor (insert BEFORE return).
b2 = wrap([
    '_pl_logger.info(',
    '    "STATE viz_join sent_count=%d received_count=%d unique_viz_ids=%s",',
    '    len(state.get("viz_tasks", []) or []),',
    '    len(getattr(all_viz, "visualizations", []) or []),',
    '    sorted({getattr(v, "visualization_id", None) for v in (getattr(all_viz, "visualizations", []) or []) if getattr(v, "visualization_id", None)}),',
    ')',
], indent="    ")

# B3 viz_evaluator.start — anchor: first 'tasks = state.get("viz_tasks", []) or []' in viz_evaluator (line 1078)
b3_start = wrap([
    '_pl_logger.info(',
    '    "STATE viz_evaluator.start viz_tasks_count=%d viz_results_count=%d",',
    '    len(state.get("viz_tasks", []) or []),',
    '    len(state.get("viz_results", []) or []),',
    ')',
], indent="    ")
# B3 viz_evaluator.early_exit — anchor: '"messages": [AIMessage(content="No viz tasks assigned. If this doesn\'t sound right, inform Supervisor agent or visualization agent")],'
b3_early = wrap([
    '_pl_logger.warning(',
    '    "STATE viz_evaluator.early_exit reason=empty_viz_tasks viz_results_count=%d",',
    '    len(state.get("viz_results", []) or []),',
    ')',
], indent="        ")
# B3 viz_evaluator.end — anchor: the outer return (line 1247). Will detect by 'return {"viz_grade": final_grade.grade'
b3_end = wrap([
    '_pl_logger.info(',
    '    "STATE viz_evaluator.end grade=%s redo_list=%s feedback_len=%d",',
    '    final_grade.grade, getattr(final_grade, "redo_list", None),',
    '    len(getattr(final_grade, "feedback", "") or ""),',
    ')',
], indent="    ")

# B4 report_orchestrator — anchor: 'return {"report_outline": outline_response["structured_response"]'
b4 = wrap([
    '_outline = outline_response["structured_response"]',
    '_pl_logger.info(',
    '    "STATE report_orchestrator section_count=%d section_titles=%s",',
    '    len(getattr(_outline, "sections", []) or []),',
    '    [s.name for s in (getattr(_outline, "sections", []) or [])],',
    ')',
], indent="    ")

# B5 section_worker — anchor: 'return {' followed by '"written_sections": [f"## {section.name}\\n\\n{content}".strip()],'
b5 = wrap([
    '_pl_logger.info(',
    '    "STATE report_section_worker section_name=%s body_len=%d body_word_count=%d expects_reply=%s",',
    '    section.name, len(content or ""), len((content or "").split()),',
    '    getattr(section_text, "expect_reply", None),',
    ')',
], indent="    ")

# B5-pre assign_section_workers — anchor: 'return [Send("report_section_worker"'
b5_pre = wrap([
    '_pl_logger.info(',
    '    "STATE assign_section_workers.send_dispatch outline_section_count=%d section_names=%s",',
    '    len(getattr(outline, "sections", []) or []),',
    '    [s.name for s in (getattr(outline, "sections", []) or [])],',
    ')',
], indent="    ")

# B6 report_packager pre_draft — anchor: 'draft = f"# {title}\\n\\n" + "\\n\\n".join(written_sections)'
b6 = wrap([
    '_pl_logger.info(',
    '    "STATE report_packager.pre_draft written_sections_count=%d total_chars=%d outline_sections=%d draft_chars=%d",',
    '    len(written_sections),',
    '    sum(len(s) for s in written_sections),',
    '    len(getattr(state.get("report_outline"), "sections", []) or []),',
    '    len(draft),',
    ')',
], indent="    ")

cell57_edits = [
    # (anchor_substr, pos, telemetry_lines, sentinel_for_idempotency)
    ('result = file_writer_agent.invoke(',                                   'before', b7,       'STATE file_writer.pre_invoke'),
    ('task = state.get("individual_viz_task",{state.get("viz_spec", None)})', 'after',  b1_start, 'STATE viz_worker.start'),
    ('memory_text += f"The visualization with id {sr.visualization_id} was {fin_str[0]}', 'after', b1_end, 'STATE viz_worker.end'),
    ('def assign_viz_workers(state: State):',                                'after',  b1_pre,   'STATE assign_viz_workers.send_dispatch'),
    ('==    update_memory_with_kind(state, state["_config"], "visualization", in_memory_store or get_store(), text=memory_text)', 'after', b2, 'STATE viz_join sent_count'),
    ('def viz_evaluator_node(state: State):',                                'after',  b3_start, 'STATE viz_evaluator.start'),
    ('"messages": [AIMessage(content="No viz tasks assigned. If this doesn\'t sound right, inform Supervisor agent or visualization agent")],', 'before', b3_early, 'STATE viz_evaluator.early_exit'),
    ('return {"viz_grade": final_grade.grade, "viz_feedback": final_grade.feedback, "viz_results": results, "viz_specs": specs,  "last_agent_message": fb["messages"][-1], "last_agent_expects_reply": expect_reply, "last_agent_reply_msg": reply_msg_to_supervisor, "last_agent_finished_this_task": finished_this_task, "last_created_obj": "viz_feedback" if fb["structured_response"] else None, "last_agent_id": "viz_evaluator", "current_turn_agent_id": "supervisor"}', 'before', b3_end, 'STATE viz_evaluator.end'),
    ('return {"report_outline": outline_response["structured_response"]',    'before', b4,       'STATE report_orchestrator section_count'),
    ('"written_sections": [f"## {section.name}',                             'before', b5,       'STATE report_section_worker section_name'),
    ('return [Send("report_section_worker", {"section": s}) for s in secs]', 'before', b5_pre,   'STATE assign_section_workers.send_dispatch'),
    ('draft = f"# {title}\\n\\n" + "\\n\\n".join(written_sections)',          'after',  b6,       'STATE report_packager.pre_draft'),
]
patch_cell(57, cell57_edits)

# ============== cell 46 edits ==============
b8 = wrap([
    '_pl_logger.info(',
    '    "STATE supervisor.gate goto=%s ia=%s dc=%s ac=%s vc=%s rg=%s fw=%s "',
    '    "report_text_len=%d written_sections=%d viz_count=%d sections=%d",',
    '    goto,',
    '    state.get("initial_analysis_complete"),',
    '    state.get("data_cleaning_complete"),',
    '    state.get("analyst_complete"),',
    '    state.get("visualization_complete"),',
    '    state.get("report_generator_complete"),',
    '    state.get("file_writer_complete"),',
    '    len(getattr(state.get("report_results"), "report_text", "") or ""),',
    '    len(state.get("written_sections", []) or []),',
    '    len(state.get("viz_results", []) or []),',
    '    len(state.get("sections", []) or []),',
    ')',
], indent="        ")
patch_cell(46, [
    ('return Command(goto=goto or "supervisor", update=updates)', 'before', b8, 'STATE supervisor.gate'),
])

# ---------- write back ----------
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"[done] cells={len(nb['cells'])}")
