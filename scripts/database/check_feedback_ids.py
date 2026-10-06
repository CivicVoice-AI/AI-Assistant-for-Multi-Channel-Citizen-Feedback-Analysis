import pandas as pd

CSV_FILE = "datasets/processed/nlp/master_feedback_nlp.csv"

df = pd.read_csv(
    CSV_FILE,
    usecols=["feedback_id"],
    low_memory=False
)

ids = df["feedback_id"].dropna().astype(str).str.strip()

print("=" * 70)
print("CIVICVOICE - FEEDBACK ID VALIDATION")
print("=" * 70)

print(f"\nTotal IDs: {len(ids):,}")
print(f"Unique IDs: {ids.nunique():,}")
print(f"Duplicate IDs: {ids.duplicated().sum():,}")

print(f"\nShortest ID length: {ids.str.len().min()}")
print(f"Longest ID length: {ids.str.len().max()}")

print("\nSample IDs:")
for value in ids.head(10):
    print(f"  {value}")

print("\nSample alphanumeric/non-numeric IDs:")

non_numeric = ids[
    ~ids.str.fullmatch(r"\d+")
]

for value in non_numeric.head(10):
    print(f"  {value}")

print(f"\nTotal non-numeric IDs: {len(non_numeric):,}")

print("\n" + "=" * 70)
