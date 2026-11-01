import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "validated_dataset.csv"
OUTPUT_DIR = BASE_DIR / "dataset" / "eda_results"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], errors="coerce"
)

print("\n========== CARESYNQ EDA ANALYSIS ==========")

print("\nDataset shape:", df.shape)

# Exclude timestamp from categorical analysis
columns = [
    col for col in df.columns
    if col != "Timestamp"
]

# 1. Missing value analysis
missing = df.isnull().sum()

missing.to_csv(
    OUTPUT_DIR / "missing_values.csv"
)

# 2. Response distribution analysis
with open(
    OUTPUT_DIR / "eda_report.txt",
    "w",
    encoding="utf-8"
) as report:

    report.write("CARESYNQ EXPLORATORY DATA ANALYSIS\n")
    report.write("=" * 55 + "\n")

    report.write(f"\nTotal records: {len(df)}\n")
    report.write(f"Total columns: {len(df.columns)}\n")

    for col in columns:

        report.write("\n" + "=" * 45 + "\n")
        report.write(f"\nCOLUMN: {col}\n")

        counts = df[col].value_counts(
            dropna=False
        )

        report.write(counts.to_string())
        report.write("\n")

        # Generate individual charts
        plt.figure(figsize=(9, 5))

        counts.plot(
            kind="bar",
            edgecolor="black"
        )

        plt.title(f"Response Distribution - {col}")
        plt.xlabel("Response")
        plt.ylabel("Number of Records")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        filename = (
            str(len(list(OUTPUT_DIR.glob("*.png"))))
            + "_"
            + str(abs(hash(col)))
            + ".png"
        )

        plt.savefig(OUTPUT_DIR / filename)
        plt.close()

# 3. Timestamp analysis
daily_records = (
    df.dropna(subset=["Timestamp"])
      .set_index("Timestamp")
      .resample("D")
      .size()
)

plt.figure(figsize=(11, 5))
daily_records.plot(marker="o")

plt.title("Records Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "daily_records.png")
plt.close()

daily_records.to_csv(
    OUTPUT_DIR / "daily_records.csv"
)

print("\nEDA completed successfully!")

print("Results saved in:")
print(OUTPUT_DIR)

print("\nGenerated:")
print("- eda_report.txt")
print("- missing_values.csv")
print("- daily_records.csv")
print("- Response distribution charts")