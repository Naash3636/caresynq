import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILE_PATH = (
    BASE_DIR
    / "dataset"
    / "clinical_review"
    / "clinical_review_results.csv"
)

df = pd.read_csv(FILE_PATH)

print("\n========== CARESYNQ CATEGORY AUDIT ==========")

print("\n1. Available columns:")
print(df.columns.tolist())

print("\n2. Review category distribution:")
print(df["Review_Category"].value_counts(dropna=False))

print("\n3. Suicide-related response distribution:")
print(df["Suicide attempt"].value_counts(dropna=False))

print("\n4. Suicide response vs review category:")

print(pd.crosstab(
    df["Suicide attempt"],
    df["Review_Category"],
    margins=True
))

print("\n5. Duplicate source row IDs:")

if "source_row_id" in df.columns:
    print(df["source_row_id"].duplicated().sum())
else:
    print("source_row_id not found.")

print("\n6. Missing values:")
print(df.isnull().sum())

print("\n========== AUDIT COMPLETED ==========")
