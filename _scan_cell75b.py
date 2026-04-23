import json,re
nb=json.load(open(r"IDD_results/executed_20260423_010132.ipynb","r",encoding="utf-8"))
c=nb["cells"][75]
outs=c.get("outputs") or []
def gettxt(o):
    if o.get("output_type")=="stream":
        return "".join(o.get("text",[]))
    if o.get("output_type") in ("execute_result","display_data"):
        d=o.get("data",{})
        return "".join(d.get("text/plain",[]))
    if o.get("output_type")=="error":
        return f"\n!!! ERROR: {o.get('ename')} :: {o.get('evalue')}\n" + "\n".join(o.get("traceback",[]))
    return ""
allt="".join(gettxt(o) for o in outs)
print("total chars:", len(allt))
patterns=["FINAL viz","RECOVERY","GraphRecursionError","InvalidUpdateError","Invalid managed channels","_rip_n >= 3","with_structured_output final-hop","report_generator","viz_grader","report_grader","Traceback","ERROR","KeyboardInterrupt","CancelledError","TimeoutError"]
for p in patterns:
    n=len(re.findall(re.escape(p),allt))
    print(f"  {p}: {n}")
print("--- last 3000 chars ---")
print(allt[-3000:])
