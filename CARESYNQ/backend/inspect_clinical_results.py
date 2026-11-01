import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULT_PATH = (
    BASE_DIR
    / "dataset"
    / "clinical_review"
    / "clinical_review_results.csv"
)

df = pd.read_csv(RESULT_PATH)

print("\n========== CARESYNQ REVIEW RESULTS AUDIT ==========")

print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. AVAILABLE COLUMNS")
for col in df.columns:
    print("-", col)

print("\n3. MISSING VALUES")
print(df.isnull().sum())

print("\n4. REVIEW CATEGORY DISTRIBUTION")

category_col = "Review_Category"

if category_col in df.columns:
    print(df[category_col].value_counts(dropna=False))
else:
    print("Review_Category not found.")
    print("Please check the column names above.")

print("\n5. FIRST 10 RECORDS")
print(df.head(10).to_string(index=False))

print("\n6. SUICIDE-RELATED RESPONSE DISTRIBUTION")

suicide_col = "Suicide attempt"

if suicide_col in df.columns:
    print(df[suicide_col].value_counts(dropna=False))
else:
    print("Suicide attempt column not found in results.")

print("\n========== AUDIT COMPLETED ==========")
