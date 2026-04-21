"""
Patch IntelligentDataDetective_beta_v5.ipynb:
  1. Replace cell idx 48 (the kagglehub download) with fixture CSV injection
  2. Fix cell idx 81: state_vals.get("final_report") → state_vals.get("report_results")
  3. Replace input() pause in any code cell with pass (headless-safe)
Saves patched notebook as IntelligentDataDetective_beta_v5_patched.ipynb
"""
import json
import copy

INPUT_NB = "IntelligentDataDetective_beta_v5.ipynb"
OUTPUT_NB = "IntelligentDataDetective_beta_v5_patched.ipynb"

CELL48_REPLACEMENT = '''\
import os, glob
# Use local fixture CSV instead of Kaggle download
raw_path_str = os.path.abspath(os.path.join(os.getcwd(), "tests", "trajectory", "fixtures", "sample_dirty.csv"))
print(f"Using fixture dataset: {raw_path_str}")

df = pd.read_csv(raw_path_str)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Register DF in the global registry
df_name = "sample_dirty"
df_id = global_df_registry.register_dataframe(df, df_name, raw_path_str)
print(f"Registered df_id: {df_id}, df_name: {df_name}")

# Compose the sample prompt
sample_prompt_text = (
    f"Please analyze the dataset named {df_name}. You have tools for accessing the data "
    f"using the following df_id: `{df_id}`. A full analysis will be performed, followed by "
    f"meaningful visualizations (histograms of \'value\', bar chart of \'category\' counts, scatter of \'score\' vs \'value\'), "
    f"then a final report in PDF, Markdown, and HTML."
)
sample_prompt_tuple = ("user", sample_prompt_text)
print("Prompt:", sample_prompt_text[:120])

initial_description = InitialDescription(
    dataset_description="No description yet",
    data_sample=df.head(5).to_string()[:100],
    notes="Dataset has missing values and duplicate rows \\u2014 needs cleaning",
    finished_this_task=False,
    reply_msg_to_supervisor="This is a blank InitialDescription",
    expect_reply=True
)

data_cleaner_agent = create_data_cleaner_agent(initial_description=initial_description, df_ids=[df_id])
initial_analysis_agent = create_initial_analysis_agent(user_prompt=sample_prompt_text, df_ids=[df_id])
analyst_agent = create_analyst_agent(initial_description=initial_description, df_ids=[df_id])
file_writer_agent = create_file_writer_agent(df_ids=[df_id])
visualization_agent = create_visualization_agent(df_ids=[df_id])
report_generator_agent = create_report_generator_agent(df_ids=[df_id], rg_agent_task="outline")
report_section_agent = create_report_generator_agent(df_ids=[df_id], rg_agent_task="section")
report_packager_agent = create_report_generator_agent(df_ids=[df_id], rg_agent_task="package")
viz_evaluator_agent = create_viz_evaluator_agent()

print("Agents created:", type(data_cleaner_agent).__name__, type(initial_analysis_agent).__name__)
'''

CELL81_OLD = (
    'state_vals.get("final_report") is not None and isinstance(state_vals.get("final_report"), ReportResults):\n'
    '        assert state_vals.get("final_report") is not None\n'
    '        report_results: ReportResults = state_vals.get("final_report")\n'
    '        assert report_results is not None and isinstance(report_results, ReportResults)'
)

CELL81_NEW = (
    'state_vals.get("report_results") is not None and isinstance(state_vals.get("report_results"), ReportResults):\n'
    '        report_results: ReportResults = state_vals.get("report_results")\n'
    '        assert report_results is not None and isinstance(report_results, ReportResults)'
)


def join_source(src):
    return "".join(src) if isinstance(src, list) else src


def main():
    with open(INPUT_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    print(f"Loaded notebook with {len(cells)} cells")

    # --- Patch cell idx 48 (the kagglehub download code cell) ---
    c48 = cells[48]
    src48 = join_source(c48["source"])
    if "kagglehub" in src48 or "KaggleHub" in src48 or "kaggle" in src48.lower():
        c48["source"] = CELL48_REPLACEMENT
        # Clear any existing outputs
        if c48["cell_type"] == "code":
            c48["outputs"] = []
            c48["execution_count"] = None
        print("✅ Cell idx 48: replaced Kaggle download with fixture CSV injection")
    else:
        print(f"⚠️  Cell idx 48 does not look like Kaggle cell. First 100 chars: {src48[:100]}")
        print("   Attempting patch anyway...")
        # Find the actual Kaggle cell
        for i, c in enumerate(cells):
            s = join_source(c["source"])
            if "kagglehub" in s and c["cell_type"] == "code":
                c["source"] = CELL48_REPLACEMENT
                if "outputs" in c:
                    c["outputs"] = []
                c["execution_count"] = None
                print(f"✅ Found and patched Kaggle cell at index {i}")
                break

    # --- Patch cell idx 81: fix final_report → report_results ---
    c81 = cells[81]
    src81 = join_source(c81["source"])
    if 'state_vals.get("final_report")' in src81:
        new_src = src81.replace(
            'if state_vals.get("final_report") is not None and isinstance(state_vals.get("final_report"), ReportResults):\n'
            '        assert state_vals.get("final_report") is not None\n'
            '        report_results: ReportResults = state_vals.get("final_report")\n'
            '        assert report_results is not None and isinstance(report_results, ReportResults)',
            'if state_vals.get("report_results") is not None and isinstance(state_vals.get("report_results"), ReportResults):\n'
            '        report_results: ReportResults = state_vals.get("report_results")\n'
            '        assert report_results is not None and isinstance(report_results, ReportResults)',
        )
        if new_src == src81:
            # Try a broader replacement
            new_src = src81.replace(
                'state_vals.get("final_report")',
                'state_vals.get("report_results")',
            )
            # But only in the persistence block (not in assert messages etc.)
            print("⚠️  Used broad replacement for cell 81")
        c81["source"] = new_src
        if c81["cell_type"] == "code":
            c81["outputs"] = []
            c81["execution_count"] = None
        print("✅ Cell idx 81: fixed final_report → report_results")
    else:
        print("⚠️  Cell idx 81: 'final_report' key not found - skipping")

    # --- Patch cell idx 7: fix _is_colab() false-positive on Windows ---
    # On this machine C:\content exists (old Colab artifact), causing _is_colab() to return True.
    # Fix: replace the fallback `return os.path.isdir("/content")` with `return False`.
    c7 = cells[7]
    src7 = join_source(c7["source"])
    OLD_IS_COLAB = (
        "def _is_colab() -> bool:\n"
        "    try:\n"
        "        import google.colab  # type: ignore\n"
        "        return True\n"
        "    except Exception:\n"
        "        return os.path.isdir(\"/content\")"
    )
    NEW_IS_COLAB = (
        "def _is_colab() -> bool:\n"
        "    try:\n"
        "        import google.colab  # type: ignore\n"
        "        import google.colab.output  # type: ignore  # extra guard: present only in real Colab\n"
        "        return True\n"
        "    except Exception:\n"
        "        return False"
    )
    if OLD_IS_COLAB in src7:
        c7["source"] = src7.replace(OLD_IS_COLAB, NEW_IS_COLAB)
        if c7.get("cell_type") == "code":
            c7["outputs"] = []
            c7["execution_count"] = None
        print("✅ Cell idx 7: fixed _is_colab() false-positive on Windows")
    else:
        print("⚠️  Cell idx 7: expected _is_colab() pattern not found — skipping")

    # --- Patch all cells: replace input() calls that block headless execution ---
    import re as _re
    input_pattern = _re.compile(r'\binput\s*\([^)]*\)', _re.DOTALL)
    patched_input_cells = 0
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = join_source(cell["source"])
        if "input(" in src:
            new_src = input_pattern.sub('None  # input() removed for headless execution', src)
            if new_src != src:
                cell["source"] = new_src
                if "outputs" in cell:
                    cell["outputs"] = []
                cell["execution_count"] = None
                patched_input_cells += 1
                print(f"✅ Cell idx {idx}: replaced input() call(s) for headless execution")
    if patched_input_cells == 0:
        print("ℹ️  No input() calls found in code cells")

    with open(OUTPUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✅ Patched notebook saved to: {OUTPUT_NB}")


if __name__ == "__main__":
    main()
