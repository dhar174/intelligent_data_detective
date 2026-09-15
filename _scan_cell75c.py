import json,re
nb=json.load(open(r"IDD_results/executed_20260423_010132.ipynb","r",encoding="utf-8"))
c=nb["cells"][75]
outs=c.get("outputs") or []
def gettxt(o):
    if o.get("output_type")=="stream":
        return "".join(o.get("text",[]))
    if o.get("output_type") in ("execute_result","display_data"):
        return "".join(o.get("data",{}).get("text/plain",[]))
    return ""
allt="".join(gettxt(o) for o in outs)
# find context around final-hop
for m in re.finditer("with_structured_output final-hop", allt):
    s=max(0,m.start()-150); e=min(len(allt), m.end()+50)
    snip=allt[s:e].replace("\n"," | ")
    print(snip[-260:])
    print("---")
print("\n=== distinct tool_call names observed ===")
names=set(re.findall(r"tool_call[s]?\s*[:=]\s*\[?\{?\s*['\"]?name['\"]?\s*[:=]\s*['\"]([A-Za-z_]+)", allt))
print(sorted(names))
names2=set(re.findall(r"\bAIMessage\(.*?tool_calls=\[.*?'name':\s*'([A-Za-z_]+)'", allt[:200000]))
print("AI msg names sample:", sorted(names2))
# look for stage stamps like supervisor decisions
print("\n=== supervisor route lines (sample) ===")
sup=re.findall(r"\[supervisor\][^\n]{0,200}|next agent[^\n]{0,200}|route.*?\b(viz|report|analyst|cleaner|finalize)\w*", allt, flags=re.IGNORECASE)
for x in sup[:20]:
    print(x if isinstance(x,str) else x[0])
