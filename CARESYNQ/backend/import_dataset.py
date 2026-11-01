
import pandas as pd
import sqlite3
from pathlib import Path

# ==========================================
# CARESYNQ DATASET DATABASE INTEGRATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "validated_dataset.csv"
DB_PATH = BASE_DIR / "caresynq.db"

TABLE_NAME = "survey_responses"

print("\n========== CARESYNQ DATA IMPORT ==========\n")

# Check dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Total records:", len(df))
print("Total columns:", len(df.columns))

# Preserve original responses.
# Do not fill missing answers with assumed responses.
df.insert(0, "source_row_id", range(1, len(df) + 1))

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)

try:
    # Import into a separate table.
    # This replaces only survey_responses on a rerun.
    df.to_sql(
        TABLE_NAME,
        conn,
        if_exists="replace",
        index=False
    )

    # Verify import
    cursor = conn.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    imported_records = cursor.fetchone()[0]

    cursor.execute(f'PRAGMA table_info("{TABLE_NAME}")')
    columns = [row[1] for row in cursor.fetchall()]

    print("\nDATABASE IMPORT SUMMARY")
    print("-----------------------")
    print("Imported records:", imported_records)
    print("Imported columns:", len(columns))

    print("\nTABLE COLUMNS:")
    for column in columns:
        print("-", column)

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print("\nSample records:")
    cursor.execute(
        f'SELECT * FROM "{TABLE_NAME}" LIMIT 5'
    )

    for row in cursor.fetchall():
        print(row)

    if imported_records == len(df):
        print("\nSUCCESS: All records imported and verified.")
    else:
        print("\nWARNING: Imported record count does not match.")

finally:
    conn.close()

print("\n========== IMPORT COMPLETED ==========\n")
