
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# ============================================================
# CARESYNQ - PHASE 3
# EXPLORATORY MACHINE LEARNING & PATTERN DISCOVERY
#
# This is exploratory clustering, NOT clinical prediction.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset" / "ml_preparation"

INPUT_FILE = DATA_DIR / "questionnaire_features_encoded.csv"
CLEAN_FILE = DATA_DIR / "clean_questionnaire_features.csv"

OUTPUT_DIR = BASE_DIR / "dataset" / "pattern_discovery"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

print("\n========== CARESYNQ PHASE 3 ==========")
print("EXPLORATORY PATTERN DISCOVERY")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Encoded dataset not found: {INPUT_FILE}"
    )

if not CLEAN_FILE.exists():
    raise FileNotFoundError(
        f"Clean dataset not found: {CLEAN_FILE}"
    )

encoded_df = pd.read_csv(INPUT_FILE)
clean_df = pd.read_csv(CLEAN_FILE)

if len(encoded_df) != len(clean_df):
    raise ValueError(
        "Encoded and clean datasets have different record counts."
    )

print("\n1. DATA LOADED")
print("Records:", len(encoded_df))
print("Features:", len(encoded_df.columns))

# ------------------------------------------------------------
# 2. PREPARE FEATURE MATRIX
# ------------------------------------------------------------

X = encoded_df.select_dtypes(include=[np.number]).copy()

# Remove columns that contain no information.
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.empty:
    raise ValueError("No usable numerical features found.")

print("\n2. FEATURE MATRIX")
print("Usable features:", len(X.columns))

# ------------------------------------------------------------
# 3. HANDLE MISSING VALUES
# ------------------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_imputed = imputer.fit_transform(X)

print("\n3. MISSING VALUE HANDLING COMPLETED")

# ------------------------------------------------------------
# 4. STANDARDIZE FEATURES
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print("\n4. FEATURE SCALING COMPLETED")

# ------------------------------------------------------------
# 5. TEST DIFFERENT CLUSTER COUNTS
# ------------------------------------------------------------

scores = []

minimum_k = 2
maximum_k = min(6, len(X_scaled) - 1)

for k in range(minimum_k, maximum_k + 1):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    if len(np.unique(labels)) < 2:
        continue

    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=min(2000, len(X_scaled)),
        random_state=42
    )

    scores.append({
        "Clusters": k,
        "Silhouette_Score": score,
        "Inertia": model.inertia_
    })

    print(
        f"Clusters: {k} | "
        f"Silhouette Score: {score:.4f}"
    )

if not scores:
    raise ValueError("Unable to calculate clustering scores.")

# ------------------------------------------------------------
# 6. SELECT BEST EXPLORATORY GROUPING
# ------------------------------------------------------------

score_df = pd.DataFrame(scores)

score_df.to_csv(
    OUTPUT_DIR / "cluster_evaluation.csv",
    index=False
)

best_k = int(
    score_df.loc[
        score_df["Silhouette_Score"].idxmax(),
        "Clusters"
    ]
)

print("\n6. BEST EXPLORATORY CLUSTER COUNT:", best_k)

# ------------------------------------------------------------
# 7. TRAIN FINAL K-MEANS MODEL
# ------------------------------------------------------------

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

cluster_labels = final_model.fit_predict(X_scaled)

results = clean_df.copy()

results["Exploratory_Cluster"] = cluster_labels

results.to_csv(
    OUTPUT_DIR / "exploratory_cluster_assignments.csv",
    index=False
)

print("\n7. CLUSTER ASSIGNMENTS GENERATED")

# ------------------------------------------------------------
# 8. CLUSTER DISTRIBUTION
# ------------------------------------------------------------

cluster_counts = (
    results["Exploratory_Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_counts.columns = [
    "Cluster",
    "Record_Count"
]

cluster_counts["Percentage"] = (
    cluster_counts["Record_Count"] / len(results) * 100
).round(2)

cluster_counts.to_csv(
    OUTPUT_DIR / "cluster_distribution.csv",
    index=False
)

print("\n8. CLUSTER DISTRIBUTION")
print(cluster_counts.to_string(index=False))

# ------------------------------------------------------------
# 9. CLUSTER FEATURE PROFILES
# ------------------------------------------------------------

profile_data = pd.DataFrame(
    X_imputed,
    columns=X.columns
)

profile_data["Exploratory_Cluster"] = cluster_labels

cluster_profiles = (
    profile_data
    .groupby("Exploratory_Cluster")
    .mean()
    .round(3)
)

cluster_profiles.to_csv(
    OUTPUT_DIR / "cluster_feature_profiles.csv"
)

print("\n9. CLUSTER FEATURE PROFILES SAVED")

# ------------------------------------------------------------
# 10. PCA VISUALIZATION
# ------------------------------------------------------------

if X_scaled.shape[1] >= 2:

    pca = PCA(
        n_components=2,
        random_state=42
    )

    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "Component_1": components[:, 0],
        "Component_2": components[:, 1],
        "Cluster": cluster_labels
    })

    pca_df.to_csv(
        OUTPUT_DIR / "pca_coordinates.csv",
        index=False
    )

    plt.figure(figsize=(10, 7))

    for cluster in sorted(np.unique(cluster_labels)):

        subset = pca_df[
            pca_df["Cluster"] == cluster
        ]

        plt.scatter(
            subset["Component_1"],
            subset["Component_2"],
            label=f"Cluster {cluster}",
            alpha=0.6,
            s=18
        )

    plt.title(
        "CARESYNQ - Exploratory Questionnaire Patterns"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "exploratory_clusters.png",
        dpi=200
    )

    plt.close()

    print("\n10. PCA CHART GENERATED")

# ------------------------------------------------------------
# 11. SAVE FINAL SUMMARY
# ------------------------------------------------------------

summary = {
    "total_records": int(len(results)),
    "features_used": int(X.shape[1]),
    "tested_cluster_counts": [
        int(x) for x in score_df["Clusters"].tolist()
    ],
    "selected_exploratory_clusters": best_k,
    "best_silhouette_score": round(
        float(score_df["Silhouette_Score"].max()), 4
    ),
    "model": "KMeans",
    "purpose": "Exploratory questionnaire pattern discovery",
    "clinical_prediction": False
}

import json

with open(
    OUTPUT_DIR / "pattern_discovery_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(summary, file, indent=4)

# ------------------------------------------------------------
# 12. FINAL OUTPUT
# ------------------------------------------------------------

print("\n========== GENERATED FILES ==========")

for file in sorted(OUTPUT_DIR.rglob("*")):

    if file.is_file():
        print("-", file.relative_to(OUTPUT_DIR))

print("\nIMPORTANT:")
print("Clusters represent statistical response patterns only.")
print("They are NOT depression diagnoses or clinical risk groups.")
print("The model does NOT predict postpartum depression.")

print("\n========== PHASE 3 COMPLETED ==========")
