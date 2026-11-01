import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "prepared_dataset.csv"
OUTPUT_PATH = BASE_DIR / "dataset" / "validated_dataset.csv"
REPORT_PATH = BASE_DIR / "dataset" / "category_report.txt"

df = pd.read_csv(DATA_PATH)

# Clean column names and text values
df.columns = df.columns.str.strip()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.strip()

# Convert timestamp
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

print("\n========== CARESYNQ VALIDATION ==========")

print("Records:", len(df))
print("Columns:", len(df.columns))

print("\nTIMESTAMP VALIDATION")
print("Invalid timestamps:", df["Timestamp"].isna().sum())
print("Date range:", df["Timestamp"].min(), "to", df["Timestamp"].max())

print("\nMISSING VALUES")
print(df.isnull().sum())

# Generate category report
with open(REPORT_PATH, "w", encoding="utf-8") as f:

    for col in df.columns:

        f.write("\n" + "=" * 55 + "\n")
        f.write(f"COLUMN: {col}\n")
        f.write("=" * 55 + "\n")

        f.write(
            df[col].value_counts(dropna=False).to_string()
        )

        f.write("\n")

# Save validated dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nGenerated:")
print("1. validated_dataset.csv")
print("2. category_report.txt")

print("\nVALIDATION COMPLETED")