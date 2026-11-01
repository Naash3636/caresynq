
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "validated_dataset.csv"
OUTPUT_DIR = BASE_DIR / "dataset" / "pattern_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n========== CARESYNQ PATTERN ANALYSIS ==========")

# Convert timestamps
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], errors="coerce"
)

# Columns containing symptom responses
symptom_columns = [
    col for col in df.columns
    if col not in ["Timestamp", "Age"]
]

# 1. Response distribution
distribution = []

for col in symptom_columns:
    counts = df[col].fillna("Missing").astype(str).value_counts()

    for response, count in counts.items():
        distribution.append({
            "Symptom": col,
            "Response": response,
            "Count": int(count),
            "Percentage": round(count / len(df) * 100, 2)
        })

distribution_df = pd.DataFrame(distribution)

distribution_df.to_csv(
    OUTPUT_DIR / "symptom_distribution.csv",
    index=False
)

# 2. Age group distribution
age_distribution = (
    df["Age"].fillna("Missing")
    .value_counts()
    .reset_index()
)

age_distribution.columns = ["Age_Group", "Count"]

age_distribution.to_csv(
    OUTPUT_DIR / "age_distribution.csv",
    index=False
)

# 3. Timestamp-based record volume
df["Date"] = df["Timestamp"].dt.date

daily_volume = (
    df.groupby("Date", dropna=True)
    .size()
    .reset_index(name="Record_Count")
)

daily_volume.to_csv(
    OUTPUT_DIR / "daily_record_volume.csv",
    index=False
)

# 4. Missing data report
missing = df.isnull().sum().reset_index()

missing.columns = ["Column", "Missing_Count"]

missing.to_csv(
    OUTPUT_DIR / "missing_summary.csv",
    index=False
)

print("\nDataset records:", len(df))
print("Symptom columns:", len(symptom_columns))
print("Date range:", df["Timestamp"].min(), "to",
      df["Timestamp"].max())

print("\nMost frequent responses:")
print(distribution_df.sort_values(
    "Count", ascending=False
).head(15).to_string(index=False))

print("\nReports saved to:", OUTPUT_DIR)

print("\n========== ANALYSIS COMPLETED ==========\n")
