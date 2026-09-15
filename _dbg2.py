import json
nb=json.load(open('IntelligentDataDetective_beta_v5_patched.ipynb',encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    src=''.join(c['source'])
    if 'add_edge(src,' in src and 'report_join' in src:
        idx = src.find('for src in [')
        print('CELL', i)
        print(repr(src[idx:idx+450]))
