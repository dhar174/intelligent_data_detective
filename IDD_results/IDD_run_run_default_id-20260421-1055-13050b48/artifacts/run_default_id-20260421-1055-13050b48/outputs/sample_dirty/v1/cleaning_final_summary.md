Final summary:
- Data cleaned: 50 rows, 7 columns (id, name, value, category, score, value_outlier, score_outlier)
- Duplicates removed: 3
- Missing values imputed: value median (6), score median (3)
- Outlier flags added: value_outlier, score_outlier
- Score range: 0.1 - 0.93 (no scaling needed)
- In-memory cleaned_df available for downstream use.
- Logs and summaries saved; see files:
  - cleaning_log_v1.json: outputs/sample_dirty/v1/cleaning_log_v1.json
  - cleaning_summary_v1.md: outputs/sample_dirty/v1/cleaning_summary_v1.md
  - cleaning_report_v1.md: outputs/sample_dirty/v1/cleaning_report_v1.md

Next actions:
- Re-export cleaned_df to sample_dirty_cleaned_v1.csv when registry access is available.
- Generate PDF/HTML/Markdown reports from the cleaning log and cleaned data.
