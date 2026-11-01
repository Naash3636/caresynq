
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd

# ==========================================================
# CARESYNQ - ONE SHOT PROJECT KNOWLEDGE BUILDER
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "knowledge_base"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "caresynq_complete_knowledge.json"

FRONTEND_DIR = BASE_DIR / "frontend"
BACKEND_DIR = BASE_DIR / "backend"
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "trained_models"

DB_PATH = BASE_DIR / "caresynq.db"

knowledge = {
    "project": "CARESYNQ",
    "description": (
        "CARESYNQ is a postpartum maternal wellbeing "
        "support and monitoring prototype."
    ),
    "generated_at": datetime.now().isoformat(),
    "sections": []
}


def add_section(title, content):
    knowledge["sections"].append({
        "title": title,
        "content": content
    })


# ==========================================================
# 1. PROJECT OVERVIEW
# ==========================================================

add_section(
    "Project Overview",
    """
CARESYNQ is a maternal postpartum wellbeing support platform.

The application provides:
- Mother registration and login.
- Doctor login.
- Mother dashboard.
- Doctor dashboard.
- Activity tracking.
- Wellbeing check-ins.
- Optional wellbeing questionnaire.
- Exploratory machine learning pattern analysis.
- Questionnaire review by doctors.
- Gemini AI chatbot.

The application is a supportive prototype.
It does not diagnose medical conditions.
"""
)


# ==========================================================
# 2. TECHNICAL ARCHITECTURE
# ==========================================================

add_section(
    "Technical Architecture",
    """
Frontend:
- HTML
- CSS
- JavaScript

Backend:
- Python
- Flask

Database:
- SQLite
- caresynq.db

Data processing:
- Pandas
- Joblib

Machine learning:
- KMeans clustering pipeline

AI:
- Google Gemini API

Application flow:

User
  |
  v
HTML/CSS/JavaScript frontend
  |
  v
Flask backend
  |
  +---- SQLite database
  |
  +---- ML model
  |
  +---- Gemini AI assistant
"""
)


# ==========================================================
# 3. AUTOMATIC FRONTEND KNOWLEDGE EXTRACTION
# ==========================================================

def read_frontend_files():

    if not FRONTEND_DIR.exists():
        return

    supported = [".html", ".css", ".js"]

    for file in FRONTEND_DIR.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() not in supported:
            continue

        try:
            content = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            add_section(
                f"Frontend File: {file.relative_to(BASE_DIR)}",
                content[:30000]
            )

            print("Frontend indexed:", file.name)

        except Exception as error:
            print("Frontend error:", file, error)


# ==========================================================
# 4. AUTOMATIC BACKEND KNOWLEDGE EXTRACTION
# ==========================================================

def read_backend_files():

    if not BACKEND_DIR.exists():
        return

    for file in BACKEND_DIR.rglob("*.py"):

        if file.name == Path(__file__).name:
            continue

        try:
            content = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            add_section(
                f"Backend File: {file.relative_to(BASE_DIR)}",
                content[:40000]
            )

            print("Backend indexed:", file.name)

        except Exception as error:
            print("Backend error:", file, error)


# ==========================================================
# 5. DATABASE SCHEMA EXTRACTION
# ==========================================================

def extract_database_schema():

    if not DB_PATH.exists():
        add_section(
            "Database",
            "Database file was not found during knowledge generation."
        )
        return

    try:

        connection = sqlite3.connect(DB_PATH)

        tables = connection.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name NOT LIKE 'sqlite_%'
        """).fetchall()

        database_info = []

        for (table,) in tables:

            columns = connection.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()

            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]

            column_names = [
                {
                    "name": col[1],
                    "type": col[2],
                    "primary_key": bool(col[5])
                }
                for col in columns
            ]

            database_info.append({
                "table": table,
                "columns": column_names,
                "record_count": count
            })

        connection.close()

        add_section(
            "Live Database Schema",
            json.dumps(database_info, indent=2)
        )

        print("Database schema extracted.")

    except Exception as error:
        print("Database error:", error)


# ==========================================================
# 6. DATASET ANALYSIS
# ==========================================================

def inspect_datasets():

    if not DATASET_DIR.exists():
        return

    for file in DATASET_DIR.glob("*.csv"):

        try:

            df = pd.read_csv(file)

            summary = {
                "file": file.name,
                "rows": len(df),
                "columns_count": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": {
                    str(k): str(v)
                    for k, v in df.dtypes.items()
                },
                "missing_values": {
                    str(k): int(v)
                    for k, v in df.isnull().sum().items()
                },
                "numeric_summary": (
                    df.describe().to_dict()
                ),
                "categorical_summary": {}
            }

            for column in df.select_dtypes(
                include=["object", "category"]
            ).columns:

                summary["categorical_summary"][column] = (
                    df[column]
                    .value_counts(dropna=False)
                    .head(20)
                    .to_dict()
                )

            add_section(
                f"Dataset Analysis: {file.name}",
                json.dumps(summary, indent=2, default=str)
            )

            print(
                "Dataset indexed:",
                file.name,
                len(df),
                "records"
            )

        except Exception as error:
            print("Dataset error:", file, error)


# ==========================================================
# 7. ML MODEL KNOWLEDGE
# ==========================================================

def inspect_models():

    if not MODEL_DIR.exists():
        add_section(
            "Machine Learning Models",
            "Trained model directory was not found."
        )
        return

    files = []

    for file in MODEL_DIR.iterdir():

        if file.is_file():

            files.append({
                "name": file.name,
                "size_bytes": file.stat().st_size
            })

    add_section(
        "Trained Model Files",
        json.dumps(files, indent=2)
    )


add_section(
    "Questionnaire KMeans Workflow",
    """
The questionnaire clustering pipeline uses 31 input features.

One age group is encoded using one-hot encoding.

Eight questionnaire responses are encoded into
30 additional one-hot features.

The saved pipeline is:
trained_models/caresynq_pattern_pipeline.joblib

The model receives the exact feature order defined in
model.py and returns a cluster ID.

The cluster ID identifies an exploratory response pattern.
It is not a diagnosis or a clinical risk score.
"""
)


add_section(
    "Activity Pattern Analysis",
    """
The review_prompt function compares recent activity
with earlier activity.

It checks whether recent app sessions have dropped
substantially compared with the earlier period.

It also checks whether the latest three entered days
have zero recorded sessions.

It can generate a gentle human-review prompt.

This is a demonstration rule, not a medical prediction model.
"""
)


# ==========================================================
# 8. LOGIN AND REGISTRATION WORKFLOW
# ==========================================================

add_section(
    "Registration and Login Workflow",
    """
Registration route: /register

The registration process collects:
- Name
- Email
- Password
- Consent confirmation
- Optional postpartum day

Password must contain at least 8 characters.

New registrations receive the mother role.

Passwords are stored as password hashes.

Login route: /login

The login process checks the email and password hash.

Successful login redirects:
- Mother -> /mother
- Doctor -> /doctor

Logout route: /logout

Logout clears the Flask session.
"""
)


# ==========================================================
# 9. DATABASE TABLE EXPLANATION
# ==========================================================

add_section(
    "Database Tables",
    """
users:
Stores account name, email, password hash, role,
postpartum day and account creation time.

activity:
Stores user activity date, sessions, minutes,
feeding logs, sleep logs and notes.

wellbeing:
Stores mood, contact request, notes and submission time.

questionnaire_responses:
Stores user ID, age group, questionnaire answers,
cluster ID, model version, doctor review status,
reviewing doctor, review notes and review date.

All user-related records are associated through user_id.
"""
)


# ==========================================================
# 10. PROJECT KNOWLEDGE SAFETY INSTRUCTIONS
# ==========================================================

add_section(
    "AI Assistant Instructions",
    """
You are the CARESYNQ Project AI Assistant.

You must answer questions about:
- Project functionality.
- Registration and login.
- Mother dashboard.
- Doctor dashboard.
- Database.
- Dataset.
- Machine learning.
- KMeans clustering.
- Application navigation.
- Technical architecture.
- Deployment and troubleshooting.

Use the indexed project documentation as your source.

IMPORTANT:
1. Never invent features that are not implemented.
2. If documentation does not contain an answer,
   clearly say that the detail is not available.
3. Explain technical concepts in simple language.
4. Give step-by-step instructions when asked.
5. Never reveal passwords, API keys or secrets.
6. Never expose another user's personal records.
7. Never claim that clustering diagnoses depression.
8. Never prescribe medicines or treatments.
9. For immediate danger, recommend emergency help.
10. Distinguish implemented features from planned features.

For India:
Emergency: 112
Tele-MANAS: 14416
"""
)


# ==========================================================
# 11. EXTRACT ALL AVAILABLE PROJECT FILES
# ==========================================================

print("\nStarting CARESYNQ knowledge extraction...\n")

read_frontend_files()
read_backend_files()
extract_database_schema()
inspect_datasets()
inspect_models()


# ==========================================================
# 12. SAVE COMPLETE KNOWLEDGE BASE
# ==========================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        knowledge,
        file,
        indent=2,
        ensure_ascii=False,
        default=str
    )


print("\n==========================================")
print("CARESYNQ KNOWLEDGE BUILD COMPLETED")
print("==========================================")

print("Knowledge sections:", len(knowledge["sections"]))
print("Output file:", OUTPUT_FILE)
print("==========================================")
