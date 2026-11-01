import sqlite3

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "caresynq.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("\n========== CARESYNQ SCREENING TABLE INSPECTION ==========\n")

cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""")

tables = [row[0] for row in cursor.fetchall()]

for table in tables:
    cursor.execute(f'PRAGMA table_info("{table}")')
    columns = cursor.fetchall()

    print(f"\nTABLE: {table}")
    print("COLUMNS:")

    for column in columns:
        print(f"  - {column[1]} ({column[2]})")

    cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
    print("RECORDS:", cursor.fetchone()[0])

conn.close()

print("\n========== INSPECTION COMPLETED ==========")