import pandas as pd

FILE = r"datasets/processed/nlp/master_feedback_nlp.csv"

df = pd.read_csv(FILE, nrows=5)

print("=" * 70)
print("CIVICVOICE MASTER DATASET INSPECTION")
print("=" * 70)

print("\nColumns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\nNumber of columns:", len(df.columns))

print("\nFirst 5 rows:")
print(df.to_string(index=False))

print("\nData types:")
print(df.dtypes)
