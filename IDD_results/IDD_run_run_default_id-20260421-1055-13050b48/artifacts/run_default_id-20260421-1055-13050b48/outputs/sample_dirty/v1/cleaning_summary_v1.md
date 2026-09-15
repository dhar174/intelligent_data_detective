Final cleaning summary attached. Key results are:
- Raw: 53x5; Cleaned: 50x7
- Unique ids after cleaning: 50
- Duplicates removed: 3
- Missing value counts pre-imputation: 6; pre-score: 3
- Imputations: value median (6); score median (3)
- Outlier flags added: value_outlier, score_outlier
- Score range after cleaning: min 0.1, max 0.93 (no scaling)
- In-memory cleaned DataFrame: cleaned_df (50x7)

Deliverables:
- cleaning_log_v1.json: outputs/sample_dirty/v1/cleaning_log_v1.json
- cleaning_summary_v1.md: outputs/sample_dirty/v1/cleaning_summary_v1.md
- cleaned dataset: in memory as cleaned_df; CSV export pending due to registry access for cleaned_df

Would you like me to export the cleaned CSV now (sample_dirty_cleaned_v1.csv) once the cleaned_df is registered, and then generate PDF/HTML reports from the log and data?