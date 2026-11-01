import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "post_natal_data_10000_records.csv"
)

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("Shape:", df.shape)

print("\nCOLUMN NAMES:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nSAMPLE RECORDS:")
print(df.head(5).to_string())

print("\nUNIQUE VALUES IN EACH COLUMN:")
for col in df.columns:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False).head(10))
print("\n========== IMPORTANT HEALTH VARIABLES ==========")

important_columns = [
    col for col in df.columns
    if any(word in col.lower() for word in [
        "sad", "tearful", "irritable",
        "anxious", "sleep", "appetite",
        "bonding", "suicide", "interest",
        "support", "feeling"
    ])
]

for col in important_columns:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False))