
import sqlite3
import pandas as pd
import numpy as np
import re
import json

from pathlib import Path

# ============================================================
# CARESYNQ - PHASE 2
# FEATURE ENGINEERING & ML DATASET PREPARATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "caresynq.db"

OUTPUT_DIR = BASE_DIR / "dataset" / "ml_preparation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== CARESYNQ PHASE 2 ==========")
print("FEATURE ENGINEERING AND DATA PREPARATION")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query(
    "SELECT * FROM survey_responses",
    conn
)

conn.close()

if df.empty:
    raise ValueError("No questionnaire records found.")

print("\n1. DATA LOADED")
print("Records:", len(df))
print("Columns:", len(df.columns))

# ------------------------------------------------------------
# 2. IDENTIFY QUESTIONNAIRE FEATURES
# ------------------------------------------------------------

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

available_features = [
    col for col in symptom_columns
    if col in df.columns
]

if not available_features:
    raise ValueError("Expected questionnaire feature columns not found.")

safety_column = "Suicide attempt"

print("\n2. QUESTIONNAIRE FEATURES")
for col in available_features:
    print("-", col)

# ------------------------------------------------------------
# 3. CLEAN TEXT RESPONSES
# ------------------------------------------------------------

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA
        })
    )

print("\n3. TEXT CLEANING COMPLETED")

# ------------------------------------------------------------
# 4. STANDARDIZE RESPONSE VALUES
# ------------------------------------------------------------

def normalize_response(value):

    if pd.isna(value):
        return pd.NA

    value = str(value).strip().lower()

    response_map = {
        "yes": "Yes",
        "no": "No",
        "sometimes": "Sometimes",
        "often": "Often",
        "not at all": "Not at all",
        "maybe": "Maybe",
        "not interested to say": "Not interested to say"
    }

    return response_map.get(value, value.title())


for col in available_features:
    df[col] = df[col].apply(normalize_response)

print("4. RESPONSE STANDARDIZATION COMPLETED")

# ------------------------------------------------------------
# 5. AGE FEATURE ENGINEERING
# ------------------------------------------------------------

def extract_age(value):

    if pd.isna(value):
        return np.nan

    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))

    if not numbers:
        return np.nan

    values = [float(number) for number in numbers]

    # Convert age ranges such as 25-30 to midpoint 27.5
    if len(values) >= 2:
        return sum(values[:2]) / 2

    return values[0]


if "Age" in df.columns:
    df["Age_Numeric"] = df["Age"].apply(extract_age)

print("\n5. AGE FEATURE ENGINEERING COMPLETED")

# ------------------------------------------------------------
# 6. CREATE CLEAN QUESTIONNAIRE DATASET
# ------------------------------------------------------------

clean_columns = []

if "source_row_id" in df.columns:
    clean_columns.append("source_row_id")

if "Age_Numeric" in df.columns:
    clean_columns.append("Age_Numeric")

clean_columns.extend(available_features)

clean_df = df[clean_columns].copy()

clean_file = OUTPUT_DIR / "clean_questionnaire_features.csv"
clean_df.to_csv(clean_file, index=False)

print("\n6. CLEAN DATASET SAVED")
print(clean_file)

# ------------------------------------------------------------
# 7. SEPARATE SAFETY-RELATED RESPONSES
# ------------------------------------------------------------

if safety_column in df.columns:

    safety_columns = []

    if "source_row_id" in df.columns:
        safety_columns.append("source_row_id")

    safety_columns.append(safety_column)

    safety_df = df[safety_columns].copy()

    safety_df["Safety_Response_Requires_Review"] = (
        safety_df[safety_column]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("yes")
    )

    safety_file = OUTPUT_DIR / "separate_safety_review.csv"

    safety_df.to_csv(safety_file, index=False)

    print("\n7. SAFETY RESPONSES SEPARATED")
    print("Saved:", safety_file)

# ------------------------------------------------------------
# 8. HANDLE MISSING VALUES
# ------------------------------------------------------------

missing_report = clean_df.isnull().sum().reset_index()

missing_report.columns = [
    "Feature",
    "Missing_Count"
]

missing_report["Missing_Percentage"] = (
    missing_report["Missing_Count"] / len(clean_df) * 100
).round(2)

missing_file = OUTPUT_DIR / "feature_missing_report.csv"
missing_report.to_csv(missing_file, index=False)

print("\n8. MISSING VALUE REPORT SAVED")

# ------------------------------------------------------------
# 9. ONE-HOT ENCODING
# ------------------------------------------------------------

# Questionnaire answers are categorical.
# One-hot encoding avoids assuming that categories
# have a meaningful numerical order.

encoding_features = available_features.copy()

if "Age_Numeric" in clean_df.columns:
    encoding_features.insert(0, "Age_Numeric")

feature_input = clean_df[encoding_features].copy()

encoded_df = pd.get_dummies(
    feature_input,
    columns=available_features,
    dummy_na=True,
    dtype=int
)

encoded_df = encoded_df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Preserve missing values rather than silently
# treating missing answers as negative answers.

encoded_file = OUTPUT_DIR / "questionnaire_features_encoded.csv"

encoded_df.to_csv(encoded_file, index=False)

print("\n9. ENCODING COMPLETED")
print("Original feature count:", len(encoding_features))
print("Encoded feature count:", len(encoded_df.columns))
print("Saved:", encoded_file)

# ------------------------------------------------------------
# 10. FEATURE DOCUMENTATION
# ------------------------------------------------------------

feature_info = []

for feature in encoded_df.columns:

    feature_info.append({
        "Feature": feature,
        "Data_Type": str(encoded_df[feature].dtype),
        "Missing_Values": int(encoded_df[feature].isna().sum()),
        "Unique_Values": int(encoded_df[feature].nunique())
    })

feature_report = pd.DataFrame(feature_info)

feature_report.to_csv(
    OUTPUT_DIR / "feature_documentation.csv",
    index=False
)

# ------------------------------------------------------------
# 11. DATASET SUMMARY
# ------------------------------------------------------------

summary = {
    "total_records": int(len(df)),
    "original_columns": int(len(df.columns)),
    "questionnaire_features": available_features,
    "encoded_feature_count": int(len(encoded_df.columns)),
    "missing_values_total": int(clean_df.isna().sum().sum()),
    "safety_column_separated": safety_column in df.columns,
    "target_variable_created": False,
    "clinical_prediction_ready": False
}

with open(
    OUTPUT_DIR / "preparation_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(summary, file, indent=4)

# ------------------------------------------------------------
# 12. FINAL REPORT
# ------------------------------------------------------------

print("\n========== PHASE 2 SUMMARY ==========")

print("Total records:", len(df))
print("Questionnaire features:", len(available_features))
print("Encoded features:", len(encoded_df.columns))

print(
    "Missing values:",
    int(clean_df.isna().sum().sum())
)

print("\nGENERATED FILES:")

for file in sorted(OUTPUT_DIR.glob("*")):
    if file.is_file():
        print("-", file.name)

print("\nIMPORTANT:")
print("No clinical target labels were invented.")
print("No depression diagnosis was generated.")
print("Safety-related responses were separated.")
print("Questionnaire data alone does not establish")
print("app-usage-based prediction or clinical validity.")

print("\n========== PHASE 2 COMPLETED ==========")
