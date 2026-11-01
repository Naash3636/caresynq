import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CARESYNQ - POSTNATAL DATASET ANALYSIS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = (
    PROJECT_DIR
    / "dataset"
    / "post_natal_data_10000_records.csv"
)

print("\n========== CARESYNQ DATA ANALYSIS ==========\n")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic information
print("1. DATASET OVERVIEW")
print("-------------------")
print("Total records:", df.shape[0])
print("Total columns:", df.shape[1])

print("\n2. COLUMN NAMES")
print("-------------------")
for column in df.columns:
    print("-", column)

# Missing values
print("\n3. MISSING VALUES")
print("-------------------")
print(df.isnull().sum())

# Duplicate records
print("\n4. DUPLICATE RECORDS")
print("-------------------")
print("Duplicates:", df.duplicated().sum())

# Data types
print("\n5. DATA TYPES")
print("-------------------")
print(df.dtypes)

# Categorical response patterns
print("\n6. RESPONSE DISTRIBUTIONS")
print("-------------------")

for column in df.select_dtypes(include="object").columns:
    if column != "Timestamp":
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).head(10))

# Timestamp analysis
if "Timestamp" in df.columns:
    print("\n7. TIMESTAMP ANALYSIS")
    print("-------------------")
    timestamps = pd.to_datetime(
        df["Timestamp"], errors="coerce"
    )
    print("First timestamp:", timestamps.min())
    print("Last timestamp:", timestamps.max())
    print("Invalid timestamps:", timestamps.isna().sum())

# Save a basic report
report = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique(dropna=True).values,
    "Data Type": df.dtypes.astype(str).values
})

report.to_csv(BASE_DIR / "dataset_report.csv", index=False)

print("\nAnalysis report saved as dataset_report.csv")
print("\n========== ANALYSIS COMPLETED ==========\n")
