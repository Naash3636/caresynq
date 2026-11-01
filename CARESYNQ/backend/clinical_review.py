
import sqlite3
import pandas as pd
from pathlib import Path

# =====================================================
# CARESYNQ - CLINICAL REVIEW SUPPORT ENGINE
# This is NOT a diagnostic system.
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "caresynq.db"
OUTPUT_DIR = BASE_DIR / "dataset" / "clinical_review"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)

# Read imported screening responses
# Read the actual imported questionnaire responses
df = pd.read_sql_query(
    "SELECT * FROM survey_responses",
    conn
)

print("Questionnaire records loaded:", len(df))
print("Available columns:", df.columns.tolist())

conn.close()

print("\n========== CARESYNQ CLINICAL REVIEW ==========")

print("Total records:", len(df))

# Normalise response text
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()

# Fields that require human review when concerning responses occur
symptom_columns = [
    "Feeling sad or Tearful",
    "Irritable towards baby & partner",
    "Trouble sleeping at night",
    "Problems concentrating or making decision",
    "Overeating or loss of appetite",
    "Feeling anxious",
    "Feeling of guilt",
    "Problems of bonding with baby"
]

# Check that expected columns exist
available_columns = [
    col for col in symptom_columns if col in df.columns
]

# Count concerning responses for review.
# These counts are NOT a clinical diagnosis or validated score.
def concerning_response(value):
    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return value in ["yes", "often", "sometimes"]

df["Concerning_Response_Count"] = df[
    available_columns
].apply(
    lambda row: sum(concerning_response(v) for v in row),
    axis=1
)

# Separate safety-related responses.
suicide_col = "Suicide attempt"

if suicide_col in df.columns:
    df["Safety_Review_Required"] = (
        df[suicide_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("yes")
    )
else:
    df["Safety_Review_Required"] = False

# Assign administrative review categories only.
def review_category(row):
    if row["Safety_Review_Required"]:
        return "URGENT HUMAN SAFETY REVIEW"

    if row["Concerning_Response_Count"] >= 4:
        return "PRIORITY CLINICIAN REVIEW"

    if row["Concerning_Response_Count"] >= 2:
        return "FOLLOW-UP REVIEW"

    return "ROUTINE REVIEW"

df["Review_Category"] = df.apply(
    review_category,
    axis=1
)

# Save results
output_file = OUTPUT_DIR / "clinical_review_results.csv"

df.to_csv(output_file, index=False)

# Summary
print("\nReview category distribution:")
print(df["Review_Category"].value_counts())

print("\nSafety review records:",
      int(df["Safety_Review_Required"].sum()))

print("\nResults saved:", output_file)

print("\n========== REVIEW ENGINE COMPLETED ==========")
