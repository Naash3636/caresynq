
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "validated_dataset.csv"
OUTPUT_DIR = BASE_DIR / "dataset" / "model_readiness"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("\n========== CARESYNQ MODEL READINESS ==========\n")

df = pd.read_csv(DATA_PATH)

# 1. Basic dataset information
print("Dataset shape:", df.shape)
print("\nColumns:")
for col in df.columns:
    print("-", col)

# 2. Missing values
missing = df.isnull().sum().reset_index()
missing.columns = ["Column", "Missing_Values"]
missing["Missing_Percentage"] = (
    missing["Missing_Values"] / len(df) * 100
)
missing.to_csv(OUTPUT_DIR / "missing_report.csv", index=False)

# 3. Unique values in categorical columns
categorical_report = []

for col in df.select_dtypes(include="object").columns:
    categorical_report.append({
        "Column": col,
        "Unique_Values": df[col].nunique(),
        "Most_Common_Value": (
            df[col].mode().iloc[0]
            if not df[col].mode().empty else None
        )
    })

pd.DataFrame(categorical_report).to_csv(
    OUTPUT_DIR / "categorical_report.csv",
    index=False
)

# 4. Timestamp inspection
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce"
    )

    print("\nTimestamp range:")
    print("Start:", df["Timestamp"].min())
    print("End:", df["Timestamp"].max())

    print(
        "Unique timestamps:",
        df["Timestamp"].nunique()
    )

# 5. Check whether a mother identifier exists
possible_ids = [
    col for col in df.columns
    if any(word in col.lower()
           for word in ["mother_id", "patient_id", "user_id"])
]

print("\nMother identifier columns:", possible_ids)

# 6. Check whether a verified outcome label exists
possible_targets = [
    col for col in df.columns
    if any(word in col.lower()
           for word in ["diagnosis", "clinical_outcome",
                        "validated_label"])
]

print("Potential verified outcome columns:", possible_targets)

# 7. Save readiness summary
summary = {
    "Total_Records": len(df),
    "Total_Columns": len(df.columns),
    "Duplicate_Rows": int(df.duplicated().sum()),
    "Missing_Cells": int(df.isnull().sum().sum()),
    "Mother_ID_Available": bool(possible_ids),
    "Verified_Outcome_Column_Available": bool(possible_targets)
}

pd.DataFrame([summary]).to_csv(
    OUTPUT_DIR / "readiness_summary.csv",
    index=False
)

print("\nReadiness summary:")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nReports saved in:", OUTPUT_DIR)
print("\n========== AUDIT COMPLETED ==========\n")
