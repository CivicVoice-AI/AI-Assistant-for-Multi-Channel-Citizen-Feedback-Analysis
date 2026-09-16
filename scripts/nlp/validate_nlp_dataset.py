import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"datasets\processed\nlp\master_feedback_nlp.csv"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CIVIVOICE - NLP DATASET VALIDATION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print("\nERROR: NLP dataset not found:")
        print(INPUT_FILE)
        return

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading NLP dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "feedback_id",
        "source",
        "timestamp",
        "text",
        "organization",
        "loc",
        "urgency",
        "text_clean"
    ]

    print("\n" + "=" * 70)
    print("COLUMN VALIDATION")
    print("=" * 70)

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("\nMissing columns:")
        for col in missing_columns:
            print(f"  - {col}")
    else:
        print("\nAll required columns present: PASS")

    # --------------------------------------------------------
    # Row-level validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ROW VALIDATION")
    print("=" * 70)

    duplicate_ids = df["feedback_id"].duplicated().sum()

    print(f"\nDuplicate feedback IDs : {duplicate_ids:,}")

    if duplicate_ids == 0:
        print("Duplicate ID check: PASS")

    # --------------------------------------------------------
    # Original text validation
    # --------------------------------------------------------

    original_empty = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    clean_empty = (
        df["text_clean"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"\nOriginal empty text : {original_empty:,}")
    print(f"Clean empty text    : {clean_empty:,}")

    # --------------------------------------------------------
    # Text length comparison
    # --------------------------------------------------------

    df["original_chars"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["clean_chars"] = (
        df["text_clean"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["original_words"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    df["clean_words"] = (
        df["text_clean"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print("\n" + "=" * 70)
    print("TEXT LENGTH COMPARISON")
    print("=" * 70)

    print("\nOriginal text characters:")
    print(df["original_chars"].describe().round(2))

    print("\nClean text characters:")
    print(df["clean_chars"].describe().round(2))

    print("\nOriginal word counts:")
    print(df["original_words"].describe().round(2))

    print("\nClean word counts:")
    print(df["clean_words"].describe().round(2))

    # --------------------------------------------------------
    # Source distribution
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SOURCE DISTRIBUTION")
    print("=" * 70)

    source_counts = df["source"].value_counts(dropna=False)

    print("\n" + source_counts.to_string())

    # --------------------------------------------------------
    # Records where cleaned text became much shorter
    # --------------------------------------------------------

    df["char_reduction_pct"] = (
        (df["original_chars"] - df["clean_chars"])
        / df["original_chars"].replace(0, pd.NA)
    ) * 100

    print("\n" + "=" * 70)
    print("LARGE TEXT REDUCTIONS")
    print("=" * 70)

    large_reduction = df[
        df["char_reduction_pct"].fillna(0) >= 50
    ]

    print(
        f"\nRecords with >=50% character reduction: "
        f"{len(large_reduction):,}"
    )

    if len(large_reduction) > 0:

        print("\nSample records:")

        sample = large_reduction.head(10)

        for _, row in sample.iterrows():

            print("\n---")
            print(f"feedback_id : {row['feedback_id']}")
            print(f"source      : {row['source']}")
            print(f"original    : {str(row['text'])[:300]}")
            print(f"clean       : {str(row['text_clean'])[:300]}")

    # --------------------------------------------------------
    # Cleanup temporary analysis columns
    # --------------------------------------------------------

    df.drop(
        columns=[
            "original_chars",
            "clean_chars",
            "original_words",
            "clean_words",
            "char_reduction_pct"
        ],
        inplace=True
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()