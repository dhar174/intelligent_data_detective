import json
nb=json.load(open(r"IDD_results/executed_20260423_010132.ipynb","r",encoding="utf-8"))
c=nb["cells"][75]
outs=c.get("outputs") or []
print("num outputs:", len(outs))
print("--- LAST 25 OUTPUTS (text only) ---")
def gettxt(o):
    if o.get("output_type")=="stream":
        return "".join(o.get("text",[]))
    if o.get("output_type") in ("execute_result","display_data"):
        d=o.get("data",{})
        if "text/plain" in d: return "".join(d["text/plain"])
    if o.get("output_type")=="error":
        return f"!!! ERROR: {o.get('ename')} :: {o.get('evalue')}\n" + "\n".join(o.get("traceback",[]))
    return ""
acc=[]
for o in outs[-50:]:
    t=gettxt(o)
    if t: acc.append(t)
joined="".join(acc)
# print last 6000 chars
print(joined[-6000:])
