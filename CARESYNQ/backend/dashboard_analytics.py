"""
CARESYNQ dashboard analytics.
Reads the supplied anonymous CSV and existing analysis outputs.
All dataset-derived results are aggregate/exploratory, never patient-level alerts.
"""
from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np

SYMPTOMS = [
    "Feeling sad or Tearful",
    "Irritable towards baby & partner",
    "Trouble sleeping at night",
    "Problems concentrating or making decision",
    "Overeating or loss of appetite",
    "Feeling anxious",
    "Feeling of guilt",
    "Problems of bonding with baby",
]
SHORT = [
    "Sadness", "Irritability", "Sleep difficulty", "Concentration",
    "Appetite change", "Anxiety", "Guilt", "Bonding difficulty"
]

def _cramers_v(a, b):
    """Bias-unadjusted Cramer's V for descriptive categorical association."""
    tab = pd.crosstab(a, b)
    n = int(tab.to_numpy().sum())
    if n == 0 or min(tab.shape) < 2:
        return 0.0
    obs = tab.to_numpy(dtype=float)
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    expected = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    return float(np.sqrt((chi2 / n) / max(1, min(tab.shape) - 1)))

def build_dashboard_context(dataset_dir):
    dataset_dir = Path(dataset_dir)
    csv_path = dataset_dir / "post_natal_data_10000_records.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Put post_natal_data_10000_records.csv in the project dataset folder."
        )

    df = pd.read_csv(csv_path)
    required = ["Age", "Suicide attempt"] + SYMPTOMS
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing_columns))

    total = int(len(df))
    symptom_missing = int(df[SYMPTOMS].isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    age_counts = df["Age"].fillna("Unknown").astype(str).value_counts().sort_index()
    age_labels = age_counts.index.tolist()
    age_values = [int(v) for v in age_counts.tolist()]

    # Response distribution: preserve every original response category.
    response_rows = []
    response_labels = []
    response_values = []
    response_names = []
    for col, short in zip(SYMPTOMS, SHORT):
        vc = df[col].fillna("Missing").astype(str).value_counts()
        for response, count in vc.items():
            response_rows.append({
                "question": short, "response": response,
                "count": int(count), "percent": round(float(count) * 100 / total, 2)
            })
            response_labels.append(f"{short}: {response}")
            response_values.append(int(count))
            response_names.append(short)

    # Age-wise proportion for key questionnaire items.
    age_key_labels = []
    age_key_values = []
    age_key_series = [
        ("Sadness — Yes", "Feeling sad or Tearful", "Yes"),
        ("Irritability — Yes", "Irritable towards baby & partner", "Yes"),
        ("Sleep — 2+ days/week", "Trouble sleeping at night", "Two or more days a week"),
        ("Concentration — Often", "Problems concentrating or making decision", "Often"),
        ("Anxiety — Yes", "Feeling anxious", "Yes"),
        ("Bonding — Yes", "Problems of bonding with baby", "Yes"),
    ]
    for label, col, response in age_key_series:
        age_key_labels.append(label)
        vals = []
        for age in age_labels:
            subset = df[df["Age"].fillna("Unknown").astype(str) == age]
            denom = int(subset[col].notna().sum())
            num = int((subset[col].astype(str) == response).sum())
            vals.append(round(num * 100 / denom, 2) if denom else 0)
        age_key_values.append(vals)

    # Cramer's V association matrix for eight symptom response variables.
    assoc = []
    for i, a in enumerate(SYMPTOMS):
        row = []
        for j, b in enumerate(SYMPTOMS):
            row.append(1.0 if i == j else round(_cramers_v(df[a], df[b]), 3))
        assoc.append(row)

    # Most common complete questionnaire response combinations.
    combo_cols = SYMPTOMS + ["Suicide attempt"]
    combos = (
        df[combo_cols].fillna("Missing").astype(str)
        .value_counts().head(10).reset_index(name="frequency")
    )
    combo_rows = []
    for _, r in combos.iterrows():
        combo_rows.append({
            "frequency": int(r["frequency"]),
            "responses": [str(r[c]) for c in combo_cols]
        })

    # Missing data by column.
    missing_labels = ["Age"] + SHORT + ["Suicide attempt"]
    missing_cols = ["Age"] + SYMPTOMS + ["Suicide attempt"]
    missing_values = [int(df[c].isna().sum()) for c in missing_cols]

    # Safety response distribution is anonymous aggregate data only.
    safety = df["Suicide attempt"].fillna("Missing").astype(str).value_counts()
    safety_order = ["Yes", "No", "Not interested to say", "Missing"]
    safety_labels = [x for x in safety_order if x in safety.index]
    safety_values = [int(safety[x]) for x in safety_labels]

    # Existing heuristic review-category file is explicitly not clinical triage.
    clinical_path = dataset_dir / "clinical_review" / "clinical_review_results.csv"
    review_labels, review_values = [], []
    if clinical_path.exists():
        cr = pd.read_csv(clinical_path)
        if "Review_Category" in cr.columns:
            vc = cr["Review_Category"].fillna("Unspecified").value_counts()
            review_labels = vc.index.astype(str).tolist()
            review_values = [int(x) for x in vc.tolist()]

    # Existing exploratory cluster outputs, not necessarily the deployed model.
    cluster_path = dataset_dir / "pattern_discovery" / "cluster_distribution.csv"
    cluster_labels, cluster_values = [], []
    if cluster_path.exists():
        cl = pd.read_csv(cluster_path)
        if {"Cluster", "Record_Count"}.issubset(cl.columns):
            cluster_labels = cl["Cluster"].astype(str).tolist()
            cluster_values = [int(x) for x in cl["Record_Count"].tolist()]

    eval_path = dataset_dir / "pattern_discovery" / "cluster_evaluation.csv"
    silhouette_rows = []
    if eval_path.exists():
        ev = pd.read_csv(eval_path)
        if {"Clusters", "Silhouette_Score"}.issubset(ev.columns):
            silhouette_rows = [
                {"clusters": int(r["Clusters"]), "score": round(float(r["Silhouette_Score"]), 4)}
                for _, r in ev.iterrows()
            ]

    # Timestamp counts are included only as dataset coverage, not a real-world trend.
    time_labels, time_values = [], []
    if "Timestamp" in df.columns:
        parsed = pd.to_datetime(df["Timestamp"], errors="coerce")
        daily = parsed.dt.date.value_counts().sort_index()
        time_labels = [str(x) for x in daily.index]
        time_values = [int(x) for x in daily.values]

    # Build a plain-language insight list from observed frequencies.
    anxiety_yes = int((df["Feeling anxious"].astype(str) == "Yes").sum())
    sleep_two = int((df["Trouble sleeping at night"].astype(str) == "Two or more days a week").sum())
    irritability_yes = int((df["Irritable towards baby & partner"].astype(str) == "Yes").sum())
    sadness_yes = int((df["Feeling sad or Tearful"].astype(str) == "Yes").sum())
    suicide_yes = int((df["Suicide attempt"].astype(str) == "Yes").sum())

    return {
        "dataset_summary": {
            "total": total,
            "columns": int(len(df.columns)),
            "missing": symptom_missing,
            "duplicates": duplicates,
            "age_groups": int(df["Age"].nunique(dropna=True)),
            "suicide_yes": suicide_yes,
            "suicide_undisclosed": int((df["Suicide attempt"].astype(str) == "Not interested to say").sum()),
        },
        "age_labels": age_labels,
        "age_values": age_values,
        "response_rows": response_rows,
        "response_labels": response_labels,
        "response_values": response_values,
        "age_key_labels": age_key_labels,
        "age_key_values": age_key_values,
        "association_labels": SHORT,
        "association_matrix": assoc,
        "combo_rows": combo_rows,
        "missing_labels": missing_labels,
        "missing_values": missing_values,
        "safety_labels": safety_labels,
        "safety_values": safety_values,
        "review_labels": review_labels,
        "review_values": review_values,
        "cluster_labels": cluster_labels,
        "cluster_values": cluster_values,
        "silhouette_rows": silhouette_rows,
        "time_labels": time_labels,
        "time_values": time_values,
        "insights": [
            f"{anxiety_yes:,} of {total:,} records ({anxiety_yes/total*100:.1f}%) answered Yes to feeling anxious.",
            f"{sleep_two:,} records ({sleep_two/total*100:.1f}%) selected sleep difficulty two or more days a week.",
            f"{irritability_yes:,} records ({irritability_yes/total*100:.1f}%) answered Yes to irritability.",
            f"{sadness_yes:,} records ({sadness_yes/total*100:.1f}%) answered Yes to sadness or tearfulness.",
            f"{suicide_yes:,} records ({suicide_yes/total*100:.1f}%) answered Yes to the anonymous suicide-attempt question; these are not identifiable current cases."
        ],
    }
