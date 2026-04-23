"""Fix AQ-1/2/3: Increase recursion limits for viz_worker, viz_evaluator, report_orchestrator, report_packager."""
with open('_patch_notebook.py', 'r', encoding='utf-8') as f:
    content = f.read()

changes = [
    # viz_worker: RL 40 -> 80 (identified by _VAIM marker after it)
    (
        "'recursion_limit': 80}  # cap=80 viz_worker (Fix AQ-1)\\n\"\n        \"    from langchain_core.messages import AIMessage as _VAIM",
        "'recursion_limit': 40}  # cap=40 viz_worker (reverted)\\n\"\n        \"    from langchain_core.messages import AIMessage as _VAIM",
    ),
    # viz_evaluator: RL 40 -> 80 (identified by _veretries marker after it)
    (
        "'recursion_limit': 80}  # cap=80 viz_evaluator (Fix AQ-1)\\n\"\n        \"    _veretries = 0",
        "'recursion_limit': 40}  # cap=40 viz_evaluator (reverted)\\n\"\n        \"    _veretries = 0",
    ),
    # report_orchestrator: RL 80 -> 120 (identified by _roretries marker after it)
    (
        "'recursion_limit': 80}  # cap=80 report_orchestrator (Fix AP-1)\\n\"\n        \"    _roretries = 0",
        "'recursion_limit': 120}  # cap=120 report_orchestrator (Fix AQ-2)\\n\"\n        \"    _roretries = 0",
    ),
    # report_packager: RL 80 -> 120 (identified by _RAIM marker after it)
    (
        "'recursion_limit': 80}  # cap=80 report_packager (Fix AP-2)\\n\"\n        \"    from langchain_core.messages import AIMessage as _RAIM",
        "'recursion_limit': 120}  # cap=120 report_packager (Fix AQ-3)\\n\"\n        \"    from langchain_core.messages import AIMessage as _RAIM",
    ),
]

for old, new in changes:
    if old in content:
        content = content.replace(old, new, 1)
        print(f"Replaced: {old[:70]}...")
    else:
        print(f"NOT FOUND: {old[:70]}...")

with open('_patch_notebook.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done writing _patch_notebook.py")
