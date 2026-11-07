
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = Path("datasets/processed")
MASTER_DIR = Path("datasets/masterd")

OUTPUT_FILE = MASTER_DIR / "master_feedback.csv"


# ============================================================
# STANDARD MASTER DATASET SCHEMA
# ============================================================

MASTER_COLUMNS = [
    "feedback_id",
    "source",
    "timestamp",
    "text",
    "organization",
    "loc",
    "urgency",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(series):
    """Normalize whitespace without removing useful text."""

    return (
        series
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def clean_column(series):
    """Convert missing values to empty strings and trim whitespace."""

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def validate_file(file_path):
    """Ensure an input dataset exists before loading it."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {file_path}"
        )


# ============================================================
# TWITTER DATASET
# ============================================================

def normalize_twitter():
    file = PROCESSED_DIR / "twitter_clean.csv"
    validate_file(file)

    df = pd.read_csv(file, dtype=str)

    required = [
        "feedback_id",
        "source",
        "timestamp",
        "text",
        "organization",
        "state",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Twitter dataset is missing columns: {missing}")

    clean = pd.DataFrame({
        "feedback_id": df["feedback_id"],
        "source": df["source"],
        "timestamp": df["timestamp"],
        "text": df["text"],
        "organization": df["organization"],
        "loc": df["state"],
        "urgency": "",
    })

    return clean


# ============================================================
# EMAIL DATASET
# ============================================================

def normalize_email():
    file = PROCESSED_DIR / "email_clean.csv"
    validate_file(file)

    df = pd.read_csv(file, dtype=str)

    missing = [column for column in MASTER_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(f"Email dataset is missing columns: {missing}")

    return df[MASTER_COLUMNS].copy()


# ============================================================
# SURVEY DATASET
# ============================================================

def normalize_survey():
    file = PROCESSED_DIR / "survey_clean.csv"
    validate_file(file)

    df = pd.read_csv(file, dtype=str)

    required = [
        "feedback_id",
        "source",
        "timestamp",
        "text",
        "service",
        "district",
        "urgency",
    ]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Survey dataset is missing columns: {missing}")

    clean = pd.DataFrame({
        "feedback_id": df["feedback_id"],
        "source": df["source"],
        "timestamp": df["timestamp"],
        "text": df["text"],
        "organization": df["service"],
        "loc": df["district"],
        "urgency": df["urgency"],
    })

    return clean


# ============================================================
# MAIN MERGE
# ============================================================

def main():

    print("=" * 65)
    print("CIVICVOICE - MASTER FEEDBACK DATASET CREATION")
    print("=" * 65)

    print("\nLoading and normalizing available datasets...\n")

    # Load available datasets
    twitter = normalize_twitter()
    print(f"Twitter: {len(twitter):,} rows")

    email = normalize_email()
    print(f"Email:   {len(email):,} rows")

    survey = normalize_survey()
    print(f"Survey:  {len(survey):,} rows")

    # --------------------------------------------------------
    # COMBINE DATASETS
    # --------------------------------------------------------

    master = pd.concat(
        [twitter, email, survey],
        ignore_index=True
    )

    print(f"\nTotal rows before cleaning: {len(master):,}")

    # --------------------------------------------------------
    # STANDARDIZE TEXT AND COLUMNS
    # --------------------------------------------------------

    for column in [
        "feedback_id",
        "source",
        "organization",
        "loc",
        "urgency",
    ]:
        master[column] = clean_column(master[column])

    master["text"] = clean_text(master["text"])

    # --------------------------------------------------------
    # NORMALIZE TIMESTAMPS
    # --------------------------------------------------------

    master["timestamp"] = pd.to_datetime(
        master["timestamp"],
        errors="coerce",
        utc=True,
        format="mixed"
    )

    invalid_timestamps = master["timestamp"].isna().sum()

    # --------------------------------------------------------
    # REMOVE EMPTY FEEDBACK
    # --------------------------------------------------------

    rows_before = len(master)

    master = master[
        master["text"].str.strip() != ""
    ].copy()

    removed_empty = rows_before - len(master)

    # --------------------------------------------------------
    # CHECK DUPLICATE FEEDBACK IDS
    # --------------------------------------------------------

    duplicate_count = master["feedback_id"].duplicated().sum()

    # --------------------------------------------------------
    # STANDARDIZE COLUMN ORDER
    # --------------------------------------------------------

    master = master[MASTER_COLUMNS]

    # --------------------------------------------------------
    # SAVE MASTER DATASET
    # --------------------------------------------------------

    MASTER_DIR.mkdir(parents=True, exist_ok=True)

    master.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("MASTER DATASET CREATION COMPLETED")
    print("=" * 65)

    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"Final rows:  {len(master):,}")
    print(f"Columns:     {len(master.columns)}")

    print(f"\nEmpty-text rows removed: {removed_empty:,}")
    print(f"Invalid timestamps:     {invalid_timestamps:,}")
    print(f"Duplicate feedback IDs:  {duplicate_count:,}")

    print("\nFinal columns:")
    print(master.columns.tolist())

    print("\nRecords by source:")
    print(master["source"].value_counts().to_string())

    print("\nUrgency distribution:")
    print(
        master["urgency"]
        .replace("", "Not Available")
        .value_counts()
        .to_string()
    )

    print("\nFirst five records:")
    print(master.head().to_string(index=False))

    print("\n" + "=" * 65)
    print("CIVICVOICE MASTER DATASET READY")
    print("=" * 65)


if __name__ == "__main__":
    main()