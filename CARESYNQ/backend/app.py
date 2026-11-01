import os
from flask import jsonify
from google import genai
from google.genai import types
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import check_password_hash, generate_password_hash
from pathlib import Path
from .database import get_db, init_db
from .model import review_prompt
import json
import pandas as pd
from .dotenv import load_dotenv

from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for
)
ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DATASET_PATH = ROOT / "dataset" / "post_natal_data_10000_records.csv"

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
init_db()
app = Flask(
    __name__,
    template_folder=str(FRONTEND),
    static_folder=str(FRONTEND),
    static_url_path="/assets"
)

app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY",
    "caresynq-development-secret-change-before-deployment"
)
from model import (
    review_prompt,
    predict_questionnaire_pattern
)
def current_user():
    if not session.get("user_id"):
        return None
    with get_db() as db:
        return db.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()
    
def get_mother_personal_context(user_id):
    """
    Retrieve only the logged-in mother's own records.
    Excludes free-text personal notes.
    """

    with get_db() as db:

        # 1. Recent activity tracking
        activity_rows = db.execute("""
            SELECT entry_date, sessions, minutes,
                   feeding_logs, sleep_logs
            FROM activity
            WHERE user_id = ?
            ORDER BY entry_date DESC
            LIMIT 14
        """, (user_id,)).fetchall()

        # 2. Recent wellbeing check-ins
        wellbeing_rows = db.execute("""
            SELECT mood, contact_requested, created_at
            FROM wellbeing
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 10
        """, (user_id,)).fetchall()

        # 3. Recent questionnaire submissions
        questionnaire_rows = db.execute("""
            SELECT age_group, answers_json, cluster_id,
                   model_version, created_at
            FROM questionnaire_responses
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 3
        """, (user_id,)).fetchall()

    # Convert database rows into safe JSON-compatible data
    activity = [dict(row) for row in activity_rows]

    wellbeing = [dict(row) for row in wellbeing_rows]

    questionnaires = []

    for row in questionnaire_rows:
        item = dict(row)

        try:
            item["answers"] = json.loads(item.pop("answers_json"))
        except (ValueError, TypeError):
            item["answers"] = {}

        questionnaires.append(item)

    return {
        "activity_history": activity,
        "wellbeing_checkins": wellbeing,
        "questionnaire_history": questionnaires
    }


def login_required(role=None):
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            user = current_user()
            if not user:
                return redirect(url_for("login"))
            if role and user["role"] != role:
                return redirect(url_for("mother_dashboard" if user["role"] == "mother" else "doctor_dashboard"))
            return fn(*args, **kwargs)
        return wrapped
    return decorator
# ============================================================
# CARESYNQ AI KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_FILE = ROOT / "knowledge_base" / "caresynq_complete_knowledge.json"


def load_caresynq_knowledge():
    if not KNOWLEDGE_FILE.exists():
        print("WARNING: CARESYNQ knowledge file not found.")
        return []

    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:
            knowledge = json.load(file)

        sections = knowledge.get("sections", [])

        if isinstance(sections, dict):
            sections = [
                {"title": key, "content": value}
                for key, value in sections.items()
            ]

        return sections if isinstance(sections, list) else []

    except Exception as error:
        print("Knowledge loading error:", error)
        return []


CARESYNQ_KNOWLEDGE = load_caresynq_knowledge()


def get_relevant_knowledge(question, limit=6):
    question_words = set(question.lower().split())
    scored_sections = []

    for section in CARESYNQ_KNOWLEDGE:
        if isinstance(section, dict):
            section_text = json.dumps(section, ensure_ascii=False)
        else:
            section_text = str(section)

        searchable_text = section_text.lower()

        score = sum(
            1 for word in question_words
            if len(word) > 2 and word in searchable_text
        )

        if score > 0:
            scored_sections.append((score, section_text))

    scored_sections.sort(key=lambda item: item[0], reverse=True)

    return "\n\n".join(
        content[:12000]
        for score, content in scored_sections[:limit]
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        consent = request.form.get("consent")
        day = request.form.get("postpartum_day") or None
        if not consent:
            error = "Please confirm the prototype notice to continue."
        elif not name or not email or len(password) < 8:
            error = "Enter your name, a valid email and a password of at least 8 characters."
        else:
            try:
                with get_db() as db:
                    cur = db.execute(
                        "INSERT INTO users(name,email,password_hash,role,postpartum_day) VALUES(?,?,?,?,?)",
                        (name, email, generate_password_hash(password), "mother", int(day) if day else None)
                    )
                    session["user_id"] = cur.lastrowid
                return redirect(url_for("mother_dashboard"))
            except Exception:
                error = "That email may already be registered. Try logging in."
    return render_template("register.html", error=error)

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as db:
            user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            return redirect(url_for("doctor_dashboard" if user["role"] == "doctor" else "mother_dashboard"))
        error = "Email or password is incorrect."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/mother")
@login_required("mother")
def mother_dashboard():

    user = current_user()

    with get_db() as db:

        # 1. Total activity entries
        total_entries = db.execute(
            "SELECT COUNT(*) FROM activity WHERE user_id=?",
            (user["id"],)
        ).fetchone()[0]

        # 2. Latest activity
        latest_activity = db.execute("""
            SELECT *
            FROM activity
            WHERE user_id=?
            ORDER BY entry_date DESC
            LIMIT 1
        """, (user["id"],)).fetchone()

        # 3. Latest wellbeing check-in
        latest_wellbeing = db.execute("""
            SELECT *
            FROM wellbeing
            WHERE user_id=?
            ORDER BY created_at DESC
            LIMIT 1
        """, (user["id"],)).fetchone()

        # 4. Total wellbeing check-ins
        wellbeing_count = db.execute(
            "SELECT COUNT(*) FROM wellbeing WHERE user_id=?",
            (user["id"],)
        ).fetchone()[0]

        # 5. Recent activity history
        recent_activity = db.execute("""
            SELECT *
            FROM activity
            WHERE user_id=?
            ORDER BY entry_date DESC
            LIMIT 5
        """, (user["id"],)).fetchall()

        # 6. Last 14 activity records for charts
        activity_trends = db.execute("""
            SELECT entry_date, sessions, minutes,
                   feeding_logs, sleep_logs
            FROM activity
            WHERE user_id=?
            ORDER BY entry_date DESC
            LIMIT 14
        """, (user["id"],)).fetchall()

        # Reverse to show oldest to newest
        activity_trends = list(reversed(activity_trends))

        # 7. Prepare chart data
        trend_labels = [
            row["entry_date"] for row in activity_trends
        ]

        trend_sessions = [
            row["sessions"] for row in activity_trends
        ]

        trend_minutes = [
            row["minutes"] for row in activity_trends
        ]

        trend_feeding = [
            row["feeding_logs"] for row in activity_trends
        ]

        trend_sleep = [
            row["sleep_logs"] for row in activity_trends
        ]

    # 8. Send everything to HTML
    return render_template(
        "mother-dashboard.html",
        name=user["name"].split()[0],
        total_entries=total_entries,
        latest_activity=latest_activity,
        latest_wellbeing=latest_wellbeing,
        wellbeing_count=wellbeing_count,
        recent_activity=recent_activity,
        postpartum_day=user["postpartum_day"],
        trend_labels=trend_labels,
        trend_sessions=trend_sessions,
        trend_minutes=trend_minutes,
        trend_feeding=trend_feeding,
        trend_sleep=trend_sleep
    )

@app.route("/activity", methods=["GET", "POST"])
@login_required("mother")
def activity_page():
    message = None
    if request.method == "POST":
        try:
            values = (
                session["user_id"], request.form["entry_date"],
                max(0, int(request.form.get("sessions") or 0)),
                max(0, int(request.form.get("minutes") or 0)),
                max(0, int(request.form.get("feeding_logs") or 0)),
                max(0, int(request.form.get("sleep_logs") or 0)),
                request.form.get("note", "").strip()[:500]
            )
            with get_db() as db:
                db.execute("""INSERT INTO activity(user_id,entry_date,sessions,minutes,feeding_logs,sleep_logs,note)
                    VALUES(?,?,?,?,?,?,?) ON CONFLICT(user_id,entry_date) DO UPDATE SET
                    sessions=excluded.sessions, minutes=excluded.minutes,
                    feeding_logs=excluded.feeding_logs, sleep_logs=excluded.sleep_logs, note=excluded.note""", values)
                
            message = "Your activity note has been saved."
        except (ValueError, KeyError):
            message = "Please check the date and number fields."
    return render_template("activity.html", message=message)

@app.route("/patterns")
@login_required("mother")
def patterns():

    user_id = session["user_id"]

    with get_db() as db:

        activity_rows = db.execute(
            "SELECT * FROM activity WHERE user_id = ?",
            (user_id,)
        ).fetchall()

        wellbeing_rows = db.execute(
            "SELECT * FROM wellbeing WHERE user_id = ?",
            (user_id,)
        ).fetchall()

    return render_template(
        "patterns.html",
        activity_data=activity_rows,
        wellbeing_data=wellbeing_rows,
        total_activity=len(activity_rows),
        total_wellbeing=len(wellbeing_rows)
    )
@app.route("/wellbeing", methods=["GET", "POST"])
@login_required("mother")
def wellbeing_page():
    message = None
    if request.method == "POST":
        mood = request.form.get("mood", "")
        if mood not in ["Good", "Okay", "Mixed", "Difficult", "Very difficult"]:
            message = "Please select how today felt."
        else:
            with get_db() as db:
                db.execute("INSERT INTO wellbeing(user_id,mood,contact_requested,note) VALUES(?,?,?,?)",
                    (session["user_id"], mood, request.form.get("contact", "No"),
                     request.form.get("note", "").strip()[:1000]))
            message = "Thank you for checking in. Your response has been saved."
    return render_template("wellbeing.html", message=message)



@app.route("/questionnaire", methods=["GET", "POST"])
@login_required("mother")
def questionnaire_page():

    error = None

    if request.method == "POST":

        age_group = request.form.get("age_group", "").strip()

        if not age_group:
            error = "Please select your age group."

        else:
            # Collect every answer actually submitted by the form.
            # Unselected/disabled questions are not required.
            answers = {
                key: value.strip()
                for key, value in request.form.items()
                if key != "age_group" and value.strip()
            }

            if not answers:
                error = "Please answer at least one question."

            else:
                support_tips = []

                # General supportive suggestions based on submitted answers.
                supportive_questions = {
                    "coping_difficulty": (
                        "If daily responsibilities feel overwhelming, "
                        "consider asking someone you trust for practical help."
                    ),
                    "exhausted_after_sleep": (
                        "Try arranging a protected rest period with support "
                        "from your partner or family."
                    ),
                    "unsupported_parenting": (
                        "You deserve support. Consider sharing your needs "
                        "with someone you trust."
                    ),
                    "social_isolation": (
                        "Staying connected with a trusted friend or relative "
                        "may help you feel less alone."
                    ),
                    "medical_appointments": (
                        "If attending appointments is difficult, ask someone "
                        "you trust to help arrange transport or childcare."
                    ),
                    "family_support": (
                        "Consider discussing the practical support you need "
                        "with your family."
                    ),
                    "low_energy": (
                        "Allow yourself time to rest and ask for help with "
                        "everyday responsibilities."
                    )
                }

                for key, tip in supportive_questions.items():
                    if answers.get(key) in ["Sometimes", "Yes"]:
                        support_tips.append(tip)

                if not support_tips:
                    support_tips.append(
                        "Continue checking in with yourself. You deserve "
                        "care, rest and support throughout your recovery."
                    )

                # Check responses to the three safety questions.
                safety_keys = [
                    "self_harm_thoughts",
                    "baby_harm_thoughts",
                    "safety_concerns"
                ]

                safety_concern = any(
                    answers.get(key) in ["Sometimes", "Yes"]
                    for key in safety_keys
                )

                if safety_concern:
                    support_tips.insert(
                        0,
                        "Your safety and your baby's safety matter. "
                        "Please reach out to a trusted person and seek "
                        "urgent support from a qualified healthcare "
                        "professional. If there is immediate danger "
                        "in India, call 112. Tele-MANAS is available "
                        "at 14416."
                    )

                # Do not generate a cluster using incomplete or
                # incompatible inputs from the older ML model.
                # All submitted answers are still saved.
                cluster_id = None

                try:
                    with get_db() as db:
                        db.execute("""
                            INSERT INTO questionnaire_responses
                            (user_id, age_group, answers_json,
                             cluster_id, model_version)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            session["user_id"],
                            age_group,
                            json.dumps(answers, ensure_ascii=False),
                            cluster_id,
                            "questionnaire-v2"
                        ))

                    return render_template(
                        "questionnaire-result.html",
                        cluster_id=cluster_id,
                        support_tips=support_tips
                    )

                except Exception:
                    app.logger.exception(
                        "Questionnaire response could not be saved"
                    )
                    error = (
                        "We could not save your responses. "
                        "Please try again."
                    )

    return render_template(
        "questionnaire.html",
        error=error
    )


@app.route("/doctor")
@login_required("doctor")
def doctor_dashboard():

    # Live application data
    with get_db() as db:

        mothers = db.execute("""
            SELECT u.id, u.name, u.email, u.postpartum_day,
                   COUNT(a.id) AS entries
            FROM users u
            LEFT JOIN activity a ON a.user_id = u.id
            WHERE u.role = 'mother'
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """).fetchall()

        total_entries = db.execute(
            "SELECT COUNT(*) FROM activity"
        ).fetchone()[0]

        questionnaires = db.execute("""
            SELECT q.*, u.name AS mother_name,
                   u.email AS mother_email
            FROM questionnaire_responses q
            JOIN users u ON q.user_id = u.id
            ORDER BY q.created_at DESC
        """).fetchall()

        pending_reviews = db.execute("""
            SELECT COUNT(*) FROM questionnaire_responses
            WHERE review_status = 'Pending'
        """).fetchone()[0]

    # Read CSV
    df = pd.read_csv(DATASET_PATH)

    symptoms = [
        "Feeling sad or Tearful",
        "Irritable towards baby & partner",
        "Trouble sleeping at night",
        "Problems concentrating or making decision",
        "Overeating or loss of appetite",
        "Feeling anxious",
        "Feeling of guilt",
        "Problems of bonding with baby"
    ]

    labels = [
        "Sadness", "Irritability", "Sleep",
        "Concentration", "Appetite", "Anxiety",
        "Guilt", "Bonding"
    ]

    symptom_yes = [
        int((df[col].astype(str).str.strip().str.lower() == "yes").sum())
        for col in symptoms
    ]

    age_data = df["Age"].fillna("Unknown").astype(str).value_counts().sort_index()

    suicide_data = (
        df["Suicide attempt"]
        .fillna("Not interested to say")
        .astype(str).str.strip()
        .value_counts()
    )

    suicide_labels = [
        x for x in ["Yes", "No", "Not interested to say"]
        if x in suicide_data.index
    ]

    suicide_counts = [
        int(suicide_data[x]) for x in suicide_labels
    ]

    missing_counts = [
        int(df[col].isna().sum()) for col in symptoms
    ]

    dataset_summary = {
        "total": len(df),
        "missing": int(df[symptoms].isna().sum().sum()),
        "age_groups": int(df["Age"].nunique()),
        "suicide_yes": int(
            (df["Suicide attempt"].astype(str).str.strip().str.lower() == "yes").sum()
        )
    }

    return render_template(
        "doctor-dashboard.html",

        mothers=mothers,
        total_mothers=len(mothers),
        total_entries=total_entries,

        questionnaires=questionnaires,
        pending_reviews=pending_reviews,

        dataset_summary=dataset_summary,

        symptom_labels=labels,
        symptom_yes=symptom_yes,

        age_labels=age_data.index.tolist(),
        age_counts=age_data.tolist(),

        suicide_labels=suicide_labels,
        suicide_counts=suicide_counts,

        missing_labels=labels,
        missing_counts=missing_counts
    )
@app.route(
    "/doctor/questionnaire/<int:response_id>/review",
    methods=["POST"]
)
@login_required("doctor")
def review_questionnaire(response_id):

    notes = request.form.get("review_notes", "").strip()[:2000]

    with get_db() as db:

        response = db.execute("""
            SELECT id
            FROM questionnaire_responses
            WHERE id = ?
        """, (response_id,)).fetchone()

        if not response:
            return "Questionnaire not found", 404

        db.execute("""
            UPDATE questionnaire_responses
            SET review_status = 'Reviewed',
                reviewed_by = ?,
                review_notes = ?,
                reviewed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            session["user_id"],
            notes,
            response_id
        ))

    return redirect(url_for("doctor_dashboard"))


@app.route("/api/chatbot", methods=["POST"])
def caresynq_chatbot():
    try:
        # 1. Verify logged-in user
        user = current_user()

        if not user:
            return jsonify({"error": "Please log in first."}), 401

        if user["role"] not in ["mother", "doctor"]:
            return jsonify({"error": "Access denied."}), 403

        # 2. Read request
        data = request.get_json(silent=True) or {}

        message = str(data.get("message", "")).strip()

        if not message:
            return jsonify({"error": "Please enter a message."}), 400

        if len(message) > 2000:
            return jsonify({
                "error": "Message is too long. Maximum 2000 characters."
            }), 400

        # 3. Require explicit consent for personalisation
        if data.get("personalization_consent") is not True:
            return jsonify({
                "error": (
                    "Please provide consent before using your "
                    "personal records for personalized AI responses."
                )
            }), 403

        # 4. Get personal records only for the logged-in mother
        personal_context = {}

        if user["role"] == "mother":
            personal_context = get_mother_personal_context(user["id"])

        # 5. Get CARESYNQ project knowledge
        project_context = ""

        if "get_relevant_knowledge" in globals():
            project_context = get_relevant_knowledge(message)

        if not project_context:
            project_context = (
                "Use only verified CARESYNQ project information. "
                "If information is unavailable, say so."
            )

        # 6. Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return jsonify({
                "error": "Gemini API key is not configured."
            }), 500

        client = genai.Client(api_key=api_key)

        
        system_prompt = f"""
        language_instruction = 
        You are CARESYNQ AI, a supportive maternal wellbeing assistant.

        MULTILINGUAL RESPONSE RULES:

        1. Detect the language used by the user.
        2. Respond in the same language as the user's message.
        3. Support English, Telugu, Tamil, Kannada, Hindi and other languages supported by Gemini.
        4. If the user explicitly requests a language, always use that language.
        5. Maintain natural, simple and compassionate language.
        6. Do not switch back to English unless requested.
        7. Keep medical and wellbeing explanations clear and easy to understand.
        8. Never provide a medical diagnosis. Encourage professional medical help when appropriate.
        
        IDENTITY:
        You are CARESYNQ AI, the official intelligent assistant of CARESYNQ+.

        You have two equally important responsibilities:
        1. Be a warm, sweet, caring and supportive postpartum wellbeing companion.
        2. Be a knowledgeable guide who understands the entire CARESYNQ website,
        its features, functionality, technology and project architecture.

        You are not a generic chatbot. You are the intelligent assistant
        built specifically for the CARESYNQ platform.

        ==================================================
        1. PERSONALITY AND SWEET COMMUNICATION
        ==================================================

        Your personality must be:
        - Extremely warm, gentle, lovely and approachable.
        - Friendly, patient, empathetic and encouraging.
        - Natural and conversational, never robotic.
        - Respectful of every user's feelings and circumstances.

        Speak like a caring companion who genuinely wants to help.

        Use natural expressions occasionally:
        - "Of course, gurll! "
        - "I'm right here to help you."
        - "Let's take this one step at a time."
        - "Please remember to take care of yourself too. 💜"
        - "That's wonderful! I'm happy to help."

        Communication rules:
        - Make users feel welcomed and comfortable.
        - Do not sound like a medical textbook.
        - Do not repeat the same greeting in every response.
        - Avoid unnecessarily long explanations.
        - Use emojis occasionally, not in every sentence.
        - Never judge, blame, frighten or pressure a mother.
        - Adapt your tone to the user's mood and question.
        - If someone is upset, acknowledge their feelings before
        offering practical suggestions.

        ==================================================
        2. COMPLETE CARESYNQ WEBSITE KNOWLEDGE
        ==================================================

        You are also the official CARESYNQ website assistant.

        Understand and explain the following functionality using
        the supplied project knowledge:

        A. ACCOUNT AND NAVIGATION
        - User registration and login.
        - Account authentication and logout.
        - Mother Dashboard navigation.
        - Doctor Dashboard navigation.
        - Available website pages and their functionality.

        B. MOTHER DASHBOARD
        - Daily overview.
        - Activity tracking.
        - Activity history.
        - Wellbeing check-ins.
        - Questionnaire access.
        - Personal pattern exploration.
        - Postpartum journey information.
        - Available support resources.

        C. ACTIVITY TRACKING
        - Explain how activity entries are recorded.
        - Explain sessions, minutes, feeding logs and sleep logs.
        - Help users understand their saved activity history.
        - Explain how activity records can be used for exploratory
        pattern analysis.

        D. WELLBEING CHECK-INS
        - Explain how to submit a check-in.
        - Explain available wellbeing information.
        - Help users understand their saved check-in history.
        - Respond empathetically when someone shares their feelings.

        E. QUESTIONNAIRE AND MACHINE LEARNING
        - Explain questionnaire completion and submission.
        - Explain the purpose of questionnaire responses.
        - Explain data preprocessing and feature preparation.
        - Explain the KMeans clustering workflow.
        - Explain exploratory cluster patterns.
        - Explain how questionnaire submissions are reviewed
        when the relevant feature is implemented.

        F. DOCTOR DASHBOARD
        - Explain doctor dashboard functionality.
        - Explain questionnaire submission review.
        - Explain available review statuses and workflow.
        - Explain how doctors can inspect submitted information.
        - Never reveal private information without authorized access.

        G. PROJECT ARCHITECTURE
        Explain the implemented technologies, including:
        - Python and Flask backend.
        - SQLite database.
        - HTML, CSS and JavaScript frontend.
        - Pandas and data preprocessing.
        - Scikit-learn and KMeans.
        - Joblib model serialization.
        - Gemini API integration.
        - Dataset structure and processing.
        - Authentication and database relationships.

        IMPORTANT:
        Use PROJECT KNOWLEDGE below as the source of truth.
        Never invent features, pages, buttons, technologies or
        functionalities that are not supported by the project.

        When a feature is not implemented, clearly say that it is
        not currently available in the prototype.

        When someone asks how to use the website, provide simple,
        numbered, step-by-step instructions.

        ==================================================
        3. PERSONALIZED MOTHER WELLBEING ASSISTANCE
        ==================================================

        Use the supplied logged-in mother's records whenever relevant.

        Available personal information may include:
        - Recent activity history.
        - Sessions and minutes.
        - Feeding and sleep log counts.
        - Wellbeing check-ins.
        - Selected moods.
        - Questionnaire responses.
        - Exploratory KMeans cluster information.

        Personalization rules:
        - Refer to actual saved records when answering relevant questions.
        - Consider recent records before older records.
        - Explain observations in simple, gentle language.
        - Do not assume that missing records mean something is wrong.
        - Never invent activities, feelings, symptoms or responses.
        - Clearly distinguish recorded information from suggestions.
        - Do not expose one mother's information to another user.

        When personal records are unavailable, explain that naturally
        and still provide useful general assistance.

        ==================================================
        4. FOOD AND DAILY ROUTINE SUGGESTIONS
        ==================================================

        When a mother asks what to eat, provide actual food suggestions.

        Do not simply tell her to consult a doctor for ordinary
        general food questions.

        Offer practical Indian food options when appropriate.

        Examples include:
        - Breakfast: Idli, dosa, vegetable upma, oats or poha.
        - Mid-morning: Seasonal fruits, banana or nuts.
        - Lunch: Rice, dal, vegetables, chapati and curd.
        - Evening: Roasted chana, fruit or homemade snacks.
        - Dinner: Chapati, vegetable curry, dal or khichdi.

        These are examples, not mandatory meals.

        When asked for a meal plan:
        - Organize suggestions into breakfast, lunch, snacks and dinner.
        - Give a variety of practical food choices.
        - Explain the suggestions briefly.
        - Consider dietary preferences and restrictions if provided.
        - Ask about allergies or dietary restrictions when relevant.

        When asked what to do today:
        - Consider the mother's recent activity and sleep logs.
        - Suggest gentle, achievable activities.
        - Encourage hydration, rest and regular meals.
        - Suggest asking family members for help when appropriate.
        - Never make a mother feel guilty about incomplete activity logs.

        Do not prescribe supplements, medication or restrictive diets.
        For medical conditions, complications or specific nutritional
        requirements, recommend a qualified healthcare professional.

        ==================================================
        5. EMOTIONAL SUPPORT AND CONVERSATION
        ==================================================

        If a mother says:
        - "I am tired" — respond with empathy and gentle suggestions.
        - "I feel sad" — listen, acknowledge and offer support.
        - "I cannot sleep" — provide general sleep-support suggestions.
        - "I am overwhelmed" — reassure her and suggest manageable steps.
        - "I need help" — understand what kind of help she needs.

        Do not immediately redirect every emotional conversation
        to emergency services.

        However, take statements about self-harm, harm to the baby
        or immediate danger seriously.

        For immediate danger in India:
        - Emergency services: 112.
        - Tele-MANAS: 14416.

        Encourage contacting a trusted person and qualified support.

        ==================================================
        6. HEALTH SAFETY AND RESPONSIBLE AI
        ==================================================

        - You are not a doctor or a medical professional.
        - Never diagnose medical or mental health conditions.
        - Never prescribe medicines or supplements.
        - Never claim that KMeans clusters are clinical diagnoses.
        - Never describe exploratory patterns as validated risk scores.
        - Never make unsupported clinical predictions.
        - Do not interpret missing records as evidence of illness.
        - Give general educational wellbeing and food suggestions.
        - Recommend professional care when medical concerns arise.

        ==================================================
        7. RESPONSE STYLE
        ==================================================

        Follow these response rules:

        1. Understand exactly what the user is asking.
        2. Answer the actual question directly.
        3. Use a warm and natural opening where appropriate.
        4. Use short headings and bullet points for longer answers.
        5. Give practical examples instead of vague advice.
        6. Keep simple questions short.
        7. Explain technical topics clearly without unnecessary jargon.
        8. Avoid repeating the same disclaimer in every response.
        9. Do not provide unrelated information.
        10. End naturally and supportively when appropriate.

        If a user asks for a list, provide only the requested list.
        If a user asks a simple question, do not give a lengthy lecture.
        If a user asks for detailed guidance, explain step by step.

        ==================================================
        8. PROJECT KNOWLEDGE
        ==================================================

        Use the following information as your primary reference
        for all CARESYNQ website and technical questions:

        {project_context}

        ==================================================
        9. LOGGED-IN MOTHER'S PERSONAL RECORDS
        ==================================================

        The following information belongs only to the authenticated
        mother whose session is currently active:

        {json.dumps(personal_context, ensure_ascii=False, default=str)}

        Use these records only when relevant to the user's question.
        Never fabricate information or expose another user's records.

        ==================================================
        10. FINAL IDENTITY AND BEHAVIOUR
        ==================================================

        You are CARESYNQ AI.

        You must be equally capable of:
        - Welcoming and comforting mothers.
        - Suggesting practical daily food and wellbeing ideas.
        - Understanding personal activity and wellbeing records.
        - Explaining every implemented website feature.
        - Guiding users through website navigation.
        - Explaining the project's machine learning and architecture.
        - Answering technical questions about CARESYNQ.

        Be sweet in your communication.
        Be accurate in your knowledge.
        Be practical in your suggestions.
        Be responsible with personal information.

        Your goal is to make every user feel:
        "I am understood, I am supported, and I know how to use CARESYNQ."
        

        Answer clearly, naturally, warmly and accurately.
        """


        # 7. Generate personalized response
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=1000
            )
        )

        reply = response.text or (
            "I'm sorry, I couldn't generate a response. Please try again."
        )

        return jsonify({"reply": reply})

    except Exception as e:
        app.logger.exception("CARESYNQ chatbot error")

        return jsonify({
            "error": "The AI assistant is temporarily unavailable. Please try again."
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
