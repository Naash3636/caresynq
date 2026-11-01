
import sqlite3
from pathlib import Path

# CARESYNQ project root
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "caresynq.db"

print("CARESYNQ DATABASE CHECK")
print("Database:", DB_PATH)

if not DB_PATH.exists():
    print("ERROR: Database file not found!")
    raise SystemExit

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""")

tables = cursor.fetchall()

print("\nDATABASE TABLES:")

if not tables:
    print("No tables found.")
else:
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")

        cursor.execute(f'PRAGMA table_info("{table_name}")')
        for column in cursor.fetchall():
            print(f"  {column[1]} - {column[2]}")

        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        print("  Records:", cursor.fetchone()[0])

conn.close()

print("\nDATABASE CHECK COMPLETED")

