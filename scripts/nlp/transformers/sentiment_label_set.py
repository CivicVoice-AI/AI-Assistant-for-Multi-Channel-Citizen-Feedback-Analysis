import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "datasets/processed/nlp/master_feedback_nlp.csv"
)

OUTPUT_DIR = Path(
    "datasets/processed/nlp/transformer_evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "sentiment_human_label_set.csv"
)

SAMPLE_SIZE = 200
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CIVICVOICE - HUMAN SENTIMENT LABEL SET")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    f"\nDataset rows: {len(df):,}"
)


# ============================================================
# VALIDATE
# ============================================================

required_columns = [
    "feedback_id",
    "source",
    "text",
    "text_clean"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# REMOVE EMPTY TEXT
# ============================================================

df["text_clean"] = (
    df["text_clean"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["text_clean"] != ""
].copy()


# ============================================================
# SAMPLE
# ============================================================

sample_size = min(
    SAMPLE_SIZE,
    len(df)
)

sample_df = df.sample(
    n=sample_size,
    random_state=RANDOM_STATE
).copy()


# ============================================================
# CREATE LABELING FILE
# ============================================================

label_df = sample_df[
    [
        "feedback_id",
        "source",
        "text",
        "text_clean"
    ]
].copy()

# Human annotator fills this column
label_df["human_sentiment"] = ""


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

label_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nCreated {len(label_df)} records."
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("\n" + "=" * 70)
print("LABEL SET CREATED")
print("=" * 70)