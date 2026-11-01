
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CARESYNQ - PHASE 2: DETAILED DATA ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "post_natal_data_10000_records.csv"

OUTPUT_DIR = BASE_DIR / "dataset" / "detailed_analysis"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== CARESYNQ DATA ANALYSIS ==========")

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\n1. DATASET INFORMATION")
print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# 2. DATA QUALITY ANALYSIS
# ============================================================

print("\n2. DATA QUALITY ANALYSIS")

quality = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": df.dtypes.astype(str).values,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage": (
        df.isnull().mean().values * 100
    ).round(2),
    "Unique_Values": df.nunique().values
})

quality.to_csv(
    OUTPUT_DIR / "data_quality_report.csv",
    index=False
)

print(quality.to_string(index=False))

print("\nDuplicate records:", df.duplicated().sum())

# ============================================================
# 3. IDENTIFY QUESTIONNAIRE COLUMNS
# ============================================================

question_columns = [
    "Feeling sad or Tearful",
    "Irritable towards baby & partner",
    "Trouble sleeping at night",
    "Problems concentrating or making decision",
    "Overeating or loss of appetite",
    "Feeling anxious",
    "Feeling of guilt",
    "Problems of bonding with baby",
    "Suicide attempt"
]

question_columns = [
    col for col in question_columns if col in df.columns
]

# ============================================================
# 4. RESPONSE DISTRIBUTION ANALYSIS
# ============================================================

print("\n3. QUESTIONNAIRE RESPONSE DISTRIBUTIONS")

response_results = []

for col in question_columns:

    counts = df[col].fillna("Missing").value_counts()

    for response, count in counts.items():

        response_results.append({
            "Question": col,
            "Response": response,
            "Count": count,
            "Percentage": round(
                count / len(df) * 100, 2
            )
        })

    print("\nQuestion:", col)
    print(counts.to_string())

response_df = pd.DataFrame(response_results)

response_df.to_csv(
    OUTPUT_DIR / "response_distribution.csv",
    index=False
)

# ============================================================
# 5. AGE GROUP ANALYSIS
# ============================================================

print("\n4. AGE GROUP ANALYSIS")

if "Age" in df.columns:

    age_distribution = (
        df["Age"]
        .fillna("Missing")
        .value_counts()
        .reset_index()
    )

    age_distribution.columns = ["Age_Group", "Count"]

    age_distribution["Percentage"] = (
        age_distribution["Count"] / len(df) * 100
    ).round(2)

    age_distribution.to_csv(
        OUTPUT_DIR / "age_distribution.csv",
        index=False
    )

    print(age_distribution.to_string(index=False))

    for col in question_columns:

        age_table = pd.crosstab(
            df["Age"].fillna("Missing"),
            df[col].fillna("Missing"),
            normalize="index"
        ) * 100

        age_table.round(2).to_csv(
            OUTPUT_DIR / f"age_pattern_{question_columns.index(col)+1}.csv"
        )

# ============================================================
# 6. TIMESTAMP ANALYSIS
# ============================================================

print("\n5. TIMESTAMP ANALYSIS")

if "Timestamp" in df.columns:

    df["Parsed_Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    print(
        "Valid timestamps:",
        df["Parsed_Timestamp"].notna().sum()
    )

    timestamp_summary = (
        df["Parsed_Timestamp"]
        .dt.date
        .value_counts()
        .sort_index()
    )

    timestamp_summary.to_csv(
        OUTPUT_DIR / "timestamp_distribution.csv"
    )

# ============================================================
# 7. RESPONSE COMBINATION ANALYSIS
# ============================================================

print("\n6. RESPONSE COMBINATION ANALYSIS")

if len(question_columns) >= 2:

    response_encoded = df[question_columns].copy()

    # Count exact repeated response combinations.
    combinations = (
        response_encoded
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .head(30)
        .reset_index(name="Frequency")
    )

    combinations.to_csv(
        OUTPUT_DIR / "common_response_combinations.csv",
        index=False
    )

    print("Top response combinations:")
    print(combinations.head(10).to_string(index=False))

# ============================================================
# 8. RESPONSE CORRELATION / ASSOCIATION
# ============================================================

print("\n7. CATEGORICAL ASSOCIATION ANALYSIS")

# These are descriptive cross-tabulations.
# We do not assign numerical severity scores to responses.

for i, col1 in enumerate(question_columns):

    for j, col2 in enumerate(question_columns):

        if j <= i:
            continue

        table = pd.crosstab(
            df[col1].fillna("Missing"),
            df[col2].fillna("Missing")
        )

        table.to_csv(
            OUTPUT_DIR /
            f"association_{i+1}_{j+1}.csv"
        )

print("Pairwise categorical cross-tabulations saved.")

# ============================================================
# 9. VISUALIZATION - AGE DISTRIBUTION
# ============================================================

print("\n8. GENERATING VISUALIZATIONS")

if "Age" in df.columns:

    plt.figure(figsize=(10, 6))

    df["Age"].fillna("Missing").value_counts().plot(
        kind="bar"
    )

    plt.title("CARESYNQ - Age Group Distribution")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "age_distribution.png",
        dpi=300
    )

    plt.close()

# ============================================================
# 10. VISUALIZATION - QUESTION RESPONSE DISTRIBUTIONS
# ============================================================

for index, col in enumerate(question_columns):

    counts = df[col].fillna("Missing").value_counts()

    plt.figure(figsize=(9, 5))

    counts.plot(kind="bar")

    plt.title(f"Response Distribution: {col}")
    plt.xlabel("Response")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=35)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / f"question_{index+1}.png",
        dpi=300
    )

    plt.close()

# ============================================================
# 11. VISUALIZATION - MISSING VALUES
# ============================================================

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:

    plt.figure(figsize=(11, 6))

    missing.sort_values().plot(kind="bar")

    plt.title("CARESYNQ - Missing Values by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing Records")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "missing_values.png",
        dpi=300
    )

    plt.close()

# ============================================================
# 12. FINAL ANALYSIS SUMMARY
# ============================================================

summary = []

summary.append("CARESYNQ - DETAILED DATA ANALYSIS REPORT")
summary.append("=" * 55)

summary.append(f"Total records: {len(df)}")
summary.append(f"Total columns: {len(df.columns)}")
summary.append(f"Duplicate records: {df.duplicated().sum()}")
summary.append(f"Total missing cells: {df.isnull().sum().sum()}")

summary.append("\nQUESTIONNAIRE COLUMNS")

for col in question_columns:

    summary.append(f"\n{col}")

    counts = df[col].fillna("Missing").value_counts()

    for response, count in counts.items():

        percentage = count / len(df) * 100

        summary.append(
            f"  {response}: {count} ({percentage:.2f}%)"
        )

summary.append("\nANALYSIS LIMITATIONS")
summary.append(
    "This report describes the supplied questionnaire dataset."
)
summary.append(
    "Response frequencies are not clinical diagnoses."
)
summary.append(
    "No clinical prediction accuracy is established by this analysis."
)
summary.append(
    "Missing responses must not be interpreted as negative responses."
)

with open(
    OUTPUT_DIR / "detailed_analysis_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(summary))

print("\n========== ANALYSIS COMPLETED ==========")

print("\nResults saved in:")
print(OUTPUT_DIR)

print("\nCharts saved in:")
print(CHART_DIR)

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.rglob("*")):
    if file.is_file():
        print(file.relative_to(OUTPUT_DIR))
