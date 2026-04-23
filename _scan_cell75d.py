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
# look for supervisor next-agent decisions / classifier outputs
patterns=[
  ("STAGE supervisor", r"STAGE supervisor [A-Z]+"),
  ("STAGE viz_worker", r"STAGE viz_worker"),
  ("STAGE visualization", r"STAGE visualization (START|DONE)"),
  ("STAGE viz_evaluator", r"STAGE viz_evaluator"),
  ("STAGE viz_grader", r"STAGE viz_grader"),
  ("STAGE report_generator", r"STAGE report_generator"),
  ("STAGE supervisor START/DONE counts", r"STAGE supervisor (START|DONE)"),
  ("classifier viz", r"classifier.{0,40}viz"),
  ("classifier report", r"classifier.{0,40}report"),
  ("AgentClass=", r"AgentClass=\w+"),
  ("next=", r"next=\w+"),
  ("route_to=", r"route_to=\w+"),
  ("viz_grade", r"viz_grade=\w+"),
  ("viz_feedback", r"viz_feedback"),
  ("structured_response=", r"structured_response="),
  ("VizFeedback", r"VizFeedback"),
  ("ReportOutline", r"ReportOutline"),
  ("ReportResults", r"ReportResults"),
  ("CleaningMetadata", r"CleaningMetadata"),
  ("AnalysisInsights", r"AnalysisInsights"),
  ("VisualizationResults", r"VisualizationResults"),
  ("InitialDescription", r"InitialDescription"),
]
for label,p in patterns:
    n=len(re.findall(p,allt))
    print(f"  {label}: {n}")
print()
# find the agent class hits to confirm visualization loop
for m in re.finditer(r"AgentClass=(\w+)",allt):
    print(m.group(1))
