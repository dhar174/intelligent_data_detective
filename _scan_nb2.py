import json
nb=json.load(open(r"IDD_results/executed_20260423_010132.ipynb","r",encoding="utf-8"))
cells=nb["cells"]
for i in range(70,90):
    c=cells[i]
    ec=c.get("execution_count")
    src="".join(c.get("source",[]))[:90].replace("\n"," ")
    nouts=len(c.get("outputs",[]) or [])
    print(f"{i:3d} ec={ec} outs={nouts} src={src!r}")
