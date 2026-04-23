import json
nb=json.load(open('IntelligentDataDetective_beta_v5.ipynb',encoding='utf-8'))
old = ('for src in [\n'
       '    "initial_analysis", "data_cleaner", "analyst",\n'
       '    "viz_worker", "viz_join", "viz_evaluator",\n'
       '    "report_orchestrator", "report_section_worker", "report_join",\n'
       '\n'
       ']:\n'
       '    data_analysis_team_builder.add_edge(src, "supervisor")')
src=''.join(nb['cells'][60]['source'])
print('FOUND:', old in src)
i = src.find('for src in [')
print('SRC SLICE:')
print(repr(src[i:i+len(old)+10]))
print('OLD:')
print(repr(old))
