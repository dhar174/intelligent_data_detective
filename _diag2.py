import json, sys
fn = sys.argv[1]
nb = json.load(open(fn,'r',encoding='utf-8'))
c = nb['cells'][48]
print(f'cell 48 outputs ({len(c.get("outputs",[]))}):')
for i,o in enumerate(c.get('outputs',[])):
    if o.get('output_type')=='stream':
        text = ''.join(o.get('text','')) if isinstance(o.get('text'),list) else str(o.get('text',''))
        print(f'  stream[{i}]: {text[:600]}')
    elif o.get('output_type')=='error':
        print(f'  error: {o["ename"]}: {o["evalue"]}')
        for line in o.get('traceback',[])[-20:]:
            # strip ANSI
            import re
            s = re.sub(r'\x1b\[[0-9;]*m', '', line)
            print(f'    TB: {s[:240]}')
