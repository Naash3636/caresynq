import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILE_PATH = (
    BASE_DIR
    / "dataset"
    / "clinical_review"
    / "clinical_review_results.csv"
)

OUTPUT_DIR = BASE_DIR / "dataset" / "clinical_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FILE_PATH)

print("\n========== CLINICAL REVIEW VALIDATION ==========")

print("\n1. Dataset shape:")
print(df.shape)

print("\n2. Column names:")
print(df.columns.tolist())

print("\n3. Missing values:")
print(df.isnull().sum())

print("\n4. Review category distribution:")
if "Review_Category" in df.columns:
    print(df["Review_Category"].value_counts(dropna=False))
else:
    print("Review_Category column not found.")

print("\n5. Suicide-related responses:")
suicide_col = "Suicide attempt"

if suicide_col in df.columns:
    print(df[suicide_col].value_counts(dropna=False))
else:
    print("Suicide attempt column not found.")

print("\n6. Duplicate records:")
print(df.duplicated().sum())

print("\n7. Sample records:")
print(df.head(10).to_string(index=False))

print("\n8. Validation report saved.")

df.describe(include="all").transpose().to_csv(
    OUTPUT_DIR / "validation_summary.csv"
)

print("\n========== VALIDATION COMPLETED ==========")
