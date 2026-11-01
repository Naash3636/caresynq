
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "caresynq.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_live_database():

    with get_connection() as conn:

        # Store daily app activity submitted by mothers
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_activity_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mother_id TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                app_sessions INTEGER,
                app_minutes REAL,
                feeding_logs INTEGER,
                sleep_notes INTEGER,
                optional_note TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(mother_id, entry_date)
            )
        """)

        # Store optional wellbeing check-ins
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_wellbeing_checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mother_id TEXT NOT NULL,
                checkin_date TEXT NOT NULL,
                overall_feeling TEXT,
                contact_requested TEXT,
                optional_note TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Record consent for live tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_tracking_consent (
                mother_id TEXT PRIMARY KEY,
                consent_given INTEGER NOT NULL DEFAULT 0,
                consent_date TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

    print("CARESYNQ live database initialized successfully!")


if __name__ == "__main__":
    initialize_live_database()
