import joblib
import pandas as pd
from pathlib import Path
"""Transparent prototype logic; not a clinical prediction model."""

def review_prompt(activity_rows):
    """
    Returns a simple data-quality/activity-change prompt for demonstration.
    It does not estimate postpartum depression risk.
    """
    rows = list(activity_rows)
    if len(rows) < 4:
        return {
            "flag": False,
            "message": "Not enough entries to compare a personal pattern yet.",
            "reasons": []
        }

    recent = rows[-3:]
    previous = rows[-7:-3]
    if not previous:
        return {
            "flag": False,
            "message": "Continue adding optional entries to establish a baseline.",
            "reasons": []
        }

    old_avg = sum(int(r["sessions"] or 0) for r in previous) / len(previous)
    new_avg = sum(int(r["sessions"] or 0) for r in recent) / len(recent)
    reasons = []
    if old_avg >= 1 and new_avg < old_avg * 0.5:
        reasons.append("App sessions are lower than the earlier recorded period.")
    if all(int(r["sessions"] or 0) == 0 for r in recent):
        reasons.append("No app sessions were recorded on the latest three entered days.")

    if reasons:
        return {
            "flag": True,
            "message": "A change in recorded activity may be worth a gentle human check-in.",
            "reasons": reasons
        }
    return {
        "flag": False,
        "message": "No review prompt from this simple demo rule.",
        "reasons": []
    }
# Activity pattern analysis
def review_prompt(activity_rows):
    # Keep your existing function here
    pass


# Questionnaire clustering
def predict_questionnaire_pattern(responses):
    # We will add the KMeans inference code here
    pass
# ==========================================================
# CARESYNQ QUESTIONNAIRE KMEANS MODEL
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "trained_models"
    / "caresynq_pattern_pipeline.joblib"
)

QUESTIONNAIRE_FEATURES = [
    "Age_25-30",
    "Age_30-35",
    "Age_35-40",
    "Age_40-45",
    "Age_45-50",

    "Feeling sad or Tearful_No",
    "Feeling sad or Tearful_Sometimes",
    "Feeling sad or Tearful_Yes",

    "Irritable towards baby & partner_MISSING",
    "Irritable towards baby & partner_No",
    "Irritable towards baby & partner_Sometimes",
    "Irritable towards baby & partner_Yes",

    "Trouble sleeping at night_No",
    "Trouble sleeping at night_Two or more days a week",
    "Trouble sleeping at night_Yes",

    "Problems concentrating or making decision_MISSING",
    "Problems concentrating or making decision_No",
    "Problems concentrating or making decision_Often",
    "Problems concentrating or making decision_Yes",

    "Overeating or loss of appetite_No",
    "Overeating or loss of appetite_Not at all",
    "Overeating or loss of appetite_Yes",

    "Feeling anxious_No",
    "Feeling anxious_Yes",

    "Feeling of guilt_MISSING",
    "Feeling of guilt_Maybe",
    "Feeling of guilt_No",
    "Feeling of guilt_Yes",

    "Problems of bonding with baby_No",
    "Problems of bonding with baby_Sometimes",
    "Problems of bonding with baby_Yes"
]


def predict_questionnaire_pattern(age_group, answers):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    row = {feature: 0 for feature in QUESTIONNAIRE_FEATURES}

    # Age
    age_feature = f"Age_{age_group}"

    if age_feature not in row:
        raise ValueError("Invalid age group.")

    row[age_feature] = 1

    # Exact response categories from model metadata
    allowed_answers = {
        "Feeling sad or Tearful": [
            "No", "Sometimes", "Yes"
        ],
        "Irritable towards baby & partner": [
            "MISSING", "No", "Sometimes", "Yes"
        ],
        "Trouble sleeping at night": [
            "No", "Two or more days a week", "Yes"
        ],
        "Problems concentrating or making decision": [
            "MISSING", "No", "Often", "Yes"
        ],
        "Overeating or loss of appetite": [
            "No", "Not at all", "Yes"
        ],
        "Feeling anxious": [
            "No", "Yes"
        ],
        "Feeling of guilt": [
            "MISSING", "Maybe", "No", "Yes"
        ],
        "Problems of bonding with baby": [
            "No", "Sometimes", "Yes"
        ]
    }

    for question, options in allowed_answers.items():

        answer = answers.get(question)

        if answer not in options:
            raise ValueError(
                f"Invalid or missing response for: {question}"
            )

        feature = f"{question}_{answer}"

        if feature not in row:
            raise ValueError(
                f"Feature not found in model: {feature}"
            )

        row[feature] = 1

    # Preserve the exact training feature order
    input_data = pd.DataFrame(
        [[row[f] for f in QUESTIONNAIRE_FEATURES]],
        columns=QUESTIONNAIRE_FEATURES
    )

    if getattr(model, "n_features_in_", 31) != 31:
        raise ValueError(
            "The saved model does not accept 31 input features. "
            "Verify the original training pipeline."
        )

    cluster = int(model.predict(input_data)[0])

    return cluster