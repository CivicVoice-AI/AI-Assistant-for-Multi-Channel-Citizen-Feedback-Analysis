import pandas as pd

FILE = "datasets/processed/nlp/master_feedback_nlp.csv"

df = pd.read_csv(
    FILE,
    usecols=["source", "loc", "urgency"]
)

print("=" * 70)
print("SOURCE VALUES")
print("=" * 70)
print(df["source"].value_counts(dropna=False))

print("\n" + "=" * 70)
print("URGENCY VALUES")
print("=" * 70)
print(df["urgency"].value_counts(dropna=False))

print("\n" + "=" * 70)
print("LOCATION VALUES")
print("=" * 70)
print(df["loc"].value_counts(dropna=False).head(30))
