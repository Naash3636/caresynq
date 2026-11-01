
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "caresynq.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("\n========== CARESYNQ DATABASE TABLES ==========\n")

cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""")

tables = cursor.fetchall()

for (table_name,) in tables:
    print(f"\nTABLE: {table_name}")

    cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    count = cursor.fetchone()[0]
    print("RECORDS:", count)

    cursor.execute(f'PRAGMA table_info("{table_name}")')
    columns = cursor.fetchall()

    print("COLUMNS:")
    for column in columns:
        print(f"  - {column[1]} ({column[2]})")

conn.close()

print("\n========== INSPECTION COMPLETED ==========\n")
