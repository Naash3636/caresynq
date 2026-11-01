
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ==========================================================
# CARESYNQ - MODEL DEPLOYMENT PREPARATION
# Questionnaire pattern analysis only.
# NOT a clinical diagnostic or depression prediction model.
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "trained_models"
OUTPUT_DIR = DATA_DIR / "model_deployment"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "prepared_dataset.csv"

MODEL_FILE = MODEL_DIR / "caresynq_pattern_pipeline.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
N_CLUSTERS = 4

# Columns that must not be used as model features
EXCLUDED_COLUMNS = [
    "source_row_id",
    "id",
    "user_id",
    "mother_id",
    "Timestamp",
    "timestamp",
    "Review_Category",
    "Safety_Review_Required",
    "Concerning_Response_Count",
    "Suicide attempt"
]


def load_dataset():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Check that dataset/prepared_dataset.csv exists."
        )

    df = pd.read_csv(DATA_FILE)

    if df.empty:
        raise ValueError("The prepared dataset is empty.")

    print("Dataset loaded:", DATA_FILE)
    print("Total records:", len(df))

    return df



def prepare_features(df):

    print("\n========== FEATURE PREPARATION ==========")

    # Columns that must never become model features
    excluded_columns = [
        "source_row_id",
        "id",
        "user_id",
        "mother_id",
        "Timestamp",
        "timestamp",
        "Review_Category",
        "Safety_Review_Required",
        "Concerning_Response_Count",
        "Suicide attempt"
    ]

    # Remove excluded columns
    feature_df = df.drop(
        columns=[
            col for col in excluded_columns
            if col in df.columns
        ],
        errors="ignore"
    ).copy()

    # Remove identifier-like columns
    identifier_columns = [
        col for col in feature_df.columns
        if col.lower().endswith("_id")
    ]

    feature_df.drop(
        columns=identifier_columns,
        inplace=True,
        errors="ignore"
    )

    # Remove completely empty columns
    feature_df.dropna(
        axis=1,
        how="all",
        inplace=True
    )

    # Replace infinite numeric values
    feature_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Convert categorical questionnaire responses
    # into separate binary columns.
    categorical_columns = feature_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    for col in categorical_columns:
        feature_df[col] = (
            feature_df[col]
            .astype("string")
            .str.strip()
            .fillna("MISSING")
        )

    feature_df = pd.get_dummies(
        feature_df,
        columns=categorical_columns,
        dtype=int
    )

    # Remove columns with no variation
    feature_df = feature_df.loc[
        :, feature_df.nunique(dropna=True) > 1
    ]

    if feature_df.shape[1] == 0:
        raise ValueError(
            "No usable features found after encoding. "
            "Inspect the prepared dataset."
        )

    if len(feature_df) < N_CLUSTERS:
        raise ValueError(
            "Not enough records for clustering."
        )

    print("\nEncoded features:")
    for col in feature_df.columns:
        print("-", col)

    print("\nTotal records:", len(feature_df))
    print("Total usable features:", feature_df.shape[1])

    return feature_df


def train_pipeline(X):

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "cluster_model",
            KMeans(
                n_clusters=N_CLUSTERS,
                random_state=RANDOM_STATE,
                n_init=20
            )
        )
    ])

    pipeline.fit(X)

    cluster_labels = pipeline.predict(X)

    return pipeline, cluster_labels


def evaluate_model(X, pipeline, labels):

    unique_clusters = np.unique(labels)

    distribution = pd.Series(labels).value_counts().sort_index()

    print("\n========== MODEL EVALUATION ==========")

    print("Number of clusters:", len(unique_clusters))

    print("\nCluster distribution:")
    print(distribution)

    if 1 < len(unique_clusters) < len(X):
        transformed = pipeline.named_steps["scaler"].transform(
            pipeline.named_steps["imputer"].transform(X)
        )

        score = silhouette_score(
            transformed,
            labels
        )

        print("\nSilhouette score:", round(score, 4))
    else:
        score = None
        print("\nSilhouette score unavailable.")

    return score, distribution


def save_outputs(
    df,
    X,
    pipeline,
    labels,
    silhouette
):

    # Save model pipeline
    joblib.dump(pipeline, MODEL_FILE)

    # Save cluster assignments
    assignments = pd.DataFrame({
        "record_number": np.arange(1, len(X) + 1),
        "pattern_cluster": labels
    })

    assignments.to_csv(
        OUTPUT_DIR / "deployment_cluster_assignments.csv",
        index=False
    )

    # Save cluster profiles using original numeric features
    profile_data = X.copy()
    profile_data["pattern_cluster"] = labels

    profiles = profile_data.groupby(
        "pattern_cluster"
    ).mean(numeric_only=True)

    profiles.to_csv(
        OUTPUT_DIR / "deployment_cluster_profiles.csv"
    )

    # Save model metadata
    metadata = {
        "project": "CARESYNQ",
        "model_name": "Questionnaire Pattern Clustering",
        "model_type": "KMeans",
        "model_version": "1.0",
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "training_records": int(len(X)),
        "feature_count": int(X.shape[1]),
        "features": list(X.columns),
        "number_of_clusters": N_CLUSTERS,
        "silhouette_score": (
            float(silhouette)
            if silhouette is not None else None
        ),
        "random_state": RANDOM_STATE,
        "purpose": (
            "Exploratory questionnaire-response pattern analysis"
        ),
        "clinical_status": (
            "Not clinically validated. Not a diagnostic system."
        )
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\n========== DEPLOYMENT FILES SAVED ==========")

    print("Model:", MODEL_FILE)
    print("Metadata:", METADATA_FILE)

    print(
        "Assignments:",
        OUTPUT_DIR / "deployment_cluster_assignments.csv"
    )

    print(
        "Profiles:",
        OUTPUT_DIR / "deployment_cluster_profiles.csv"
    )


# ==========================================================
# REUSABLE PREDICTION FUNCTION
# Import this function into the Flask application later.
# ==========================================================

def predict_pattern(input_data):

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found. Run this script first."
        )

    bundle = joblib.load(MODEL_FILE)

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    features = metadata["features"]

    input_df = pd.DataFrame(
        [input_data]
    )

    missing = [
        feature for feature in features
        if feature not in input_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    input_df = input_df[features]

    result = bundle.predict(input_df)

    return {
        "pattern_cluster": int(result[0]),
        "interpretation": (
            "Statistical questionnaire-response pattern "
            "for exploratory analysis only."
        ),
        "clinical_diagnosis": False
    }


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main():

    print("\n========== CARESYNQ MODEL DEPLOYMENT ==========")

    df = load_dataset()

    X = prepare_features(df)

    pipeline, labels = train_pipeline(X)

    silhouette, distribution = evaluate_model(
        X,
        pipeline,
        labels
    )

    save_outputs(
        df,
        X,
        pipeline,
        labels,
        silhouette
    )

    print("\nIMPORTANT:")
    print("This model identifies statistical patterns only.")
    print("It does not diagnose postpartum depression.")
    print("It is not clinically validated.")

    print("\n========== MODEL DEPLOYMENT PREPARATION COMPLETED ==========")


if __name__ == "__main__":
    main()
