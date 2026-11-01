
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from werkzeug.security import generate_password_hash

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "caresynq.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as db:

        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'mother',
            postpartum_day INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            entry_date TEXT NOT NULL,
            sessions INTEGER DEFAULT 0,
            minutes INTEGER DEFAULT 0,
            feeding_logs INTEGER DEFAULT 0,
            sleep_logs INTEGER DEFAULT 0,
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, entry_date),
            FOREIGN KEY(user_id)
                REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS wellbeing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mood TEXT NOT NULL,
            contact_requested TEXT DEFAULT 'No',
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id)
                REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS questionnaire_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            age_group TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            cluster_id INTEGER,
            model_version TEXT DEFAULT '1.0',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            review_status TEXT DEFAULT 'Pending',
            reviewed_by INTEGER,
            review_notes TEXT,
            reviewed_at TEXT,

            FOREIGN KEY(user_id)
                REFERENCES users(id) ON DELETE CASCADE
        );
        """)

        # Migrate existing databases without deleting old responses
        existing_columns = {
            row["name"]
            for row in db.execute(
                "PRAGMA table_info(questionnaire_responses)"
            ).fetchall()
        }

        migrations = {
            "review_status":
                "ALTER TABLE questionnaire_responses "
                "ADD COLUMN review_status TEXT DEFAULT 'Pending'",

            "reviewed_by":
                "ALTER TABLE questionnaire_responses "
                "ADD COLUMN reviewed_by INTEGER",

            "review_notes":
                "ALTER TABLE questionnaire_responses "
                "ADD COLUMN review_notes TEXT",

            "reviewed_at":
                "ALTER TABLE questionnaire_responses "
                "ADD COLUMN reviewed_at TEXT"
        }

        for column, sql in migrations.items():
            if column not in existing_columns:
                db.execute(sql)

        # Development-only demo doctor account
        existing = db.execute(
            "SELECT id FROM users WHERE email = ?",
            ("doctor@caresynq.demo",)
        ).fetchone()

        if not existing:
            db.execute(
                """
                INSERT INTO users(
                    name, email, password_hash, role
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    "Demo Care Professional",
                    "doctor@caresynq.demo",
                    generate_password_hash("12345678"),
                    "doctor"
                )
            )

    print("CARESYNQ database initialized successfully.")
    print("Database location:", DB_PATH)
