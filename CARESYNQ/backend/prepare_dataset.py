import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "post_natal_data_10000_records.csv"
OUTPUT_DIR = BASE_DIR / "dataset"

print("\n========== CARESYNQ DATA PREPARATION ==========\n")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip()

print("Total Records:", len(df))
print("Total Columns:", len(df.columns))

# 1. Missing values
missing = df.isnull().sum()

# 2. Duplicate records
duplicates = df.duplicated().sum()

# 3. Data types
data_types = df.dtypes.astype(str)

# 4. Unique values
unique_values = df.nunique()

# Create summary
summary = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": data_types.values,
    "Missing_Values": missing.values,
    "Unique_Values": unique_values.values
})

# Save summary
summary.to_csv(OUTPUT_DIR / "column_summary.csv", index=False)

# Save response distributions
with open(OUTPUT_DIR / "response_analysis.txt", "w", encoding="utf-8") as file:

    file.write("CARESYNQ RESPONSE ANALYSIS\n")
    file.write("=" * 50 + "\n")

    for col in df.columns:

        file.write(f"\nCOLUMN: {col}\n")
        file.write("-" * 40 + "\n")

        counts = df[col].value_counts(dropna=False)

        file.write(counts.to_string())
        file.write("\n")

# Save cleaned copy without dropping any records
df.to_csv(OUTPUT_DIR / "prepared_dataset.csv", index=False)

print("\nDuplicate Records:", duplicates)
print("Missing Values:", missing.sum())

print("\nFiles generated:")
print("1. column_summary.csv")
print("2. response_analysis.txt")
print("3. prepared_dataset.csv")

print("\nDATA PREPARATION COMPLETED")