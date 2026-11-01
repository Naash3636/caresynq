
import sqlite3
import pandas as pd
from pathlib import Path

# =========================================
# CARESYNQ FEATURE ENGINEERING PREPARATION
# =========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "caresynq.db"
OUTPUT_DIR = BASE_DIR / "dataset" / "feature_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query(
    "SELECT * FROM survey_responses",
    conn
)

conn.close()

print("\n========== CARESYNQ FEATURE INSPECTION ==========")

print("\nDataset shape:", df.shape)

# Identify timestamp
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    print("\nTimestamp range:")
    print("First:", df["Timestamp"].min())
    print("Last:", df["Timestamp"].max())

# Generate category report for every column
reports = []

for column in df.columns:

    if column == "source_row_id":
        continue

    print(f"\n{'=' * 55}")
    print("COLUMN:", column)

    counts = df[column].value_counts(
        dropna=False
    )

    print(counts.to_string())

    for value, count in counts.items():

        reports.append({
            "Column": column,
            "Response": str(value),
            "Count": int(count),
            "Percentage": round(
                count / len(df) * 100, 2
            )
        })

# Save category report
report_df = pd.DataFrame(reports)

report_df.to_csv(
    OUTPUT_DIR / "response_categories.csv",
    index=False
)

# Save descriptive statistics
df.describe(include="all").to_csv(
    OUTPUT_DIR / "dataset_summary.csv"
)

# Missing value report
missing = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage": (
        df.isnull().mean().values * 100
    ).round(2)
})

missing.to_csv(
    OUTPUT_DIR / "missing_values.csv",
    index=False
)

print("\nReports saved in:", OUTPUT_DIR)
print("\nFEATURE INSPECTION COMPLETED")
