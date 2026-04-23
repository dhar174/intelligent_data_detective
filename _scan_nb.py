import json
nb=json.load(open(r"IDD_results/executed_20260423_010132.ipynb","r",encoding="utf-8"))
cells=nb["cells"]
print("total cells:", len(cells))
for i,c in enumerate(cells):
    outs=c.get("outputs",[]) or []
    errs=[o for o in outs if o.get("output_type")=="error"]
    if errs:
        e=errs[0]
        print(i,"ERR",e.get("ename"),"::",(e.get("evalue") or "")[:300])
        tb=e.get("traceback") or []
        for line in tb[-6:]:
            print("   |", line[:200])
print("--- last executed cell ---")
last=None
for i,c in enumerate(cells):
    if c.get("cell_type")=="code" and c.get("execution_count"):
        last=(i,c.get("execution_count"))
print("last code cell with exec_count:", last)
