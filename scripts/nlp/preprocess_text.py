import pandas as pd
import re
import html
import unicodedata
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"datasets\masterd\master_feedback.csv"
OUTPUT_DIR = r"datasets\processed\nlp"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "master_feedback_nlp.csv")


# ============================================================
# TEXT PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):
    """
    Conservative NLP preprocessing.

    Important:
    - Original text is NOT modified.
    - Useful multilingual/code-mixed content is preserved.
    - Only obvious formatting/noise is normalized.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # 1. Unicode normalization
    # --------------------------------------------------------
    text = unicodedata.normalize("NFKC", text)

    # --------------------------------------------------------
    # 2. Decode HTML entities
    # Example:
    # &amp; -> &
    # &#39; -> '
    # &quot; -> "
    # --------------------------------------------------------
    text = html.unescape(text)

    # --------------------------------------------------------
    # 3. Normalize URLs
    # Preserve the fact that a URL existed without keeping
    # the exact URL string.
    # --------------------------------------------------------
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " <URL> ",
        text,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # 4. Normalize email addresses
    # --------------------------------------------------------
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " <EMAIL> ",
        text
    )

    # --------------------------------------------------------
    # 5. Normalize @mentions
    # Example:
    # @username -> <MENTION>
    # --------------------------------------------------------
    text = re.sub(
        r"(?<!\w)@[A-Za-z0-9_]+",
        " <MENTION> ",
        text
    )

    # --------------------------------------------------------
    # 6. Hashtags
    #
    # Keep the actual hashtag word because it may contain
    # useful topic information.
    #
    # #WaterProblem -> WaterProblem
    # --------------------------------------------------------
    text = re.sub(
        r"#([^\s#]+)",
        r"\1",
        text
    )

    # --------------------------------------------------------
    # 7. Remove long structural separator lines
    #
    # Example:
    # -------------------------
    # _________________________
    # --------------------------------------------------------
    text = re.sub(
        r"[-_=]{5,}",
        " ",
        text
    )

    # --------------------------------------------------------
    # 8. CPGRAMS hierarchy markers
    #
    # Keep the words around them, but remove the >> symbols.
    #
    # Example:
    # Ministry >> Department >> Issue
    # becomes:
    # Ministry Department Issue
    # --------------------------------------------------------
    text = re.sub(
        r"\s*>>\s*",
        " ",
        text
    )

    # --------------------------------------------------------
    # 9. Normalize repeated punctuation
    #
    # We do NOT remove punctuation completely because
    # punctuation can be useful for sentiment/emotion.
    #
    # !!!!! -> !!
    # ????? -> ??
    # ....... -> ..
    # --------------------------------------------------------
    text = re.sub(
        r"([!?.,])\1{2,}",
        r"\1\1",
        text
    )

    # --------------------------------------------------------
    # 10. Normalize repeated Latin characters
    #
    # Only very long repetitions are reduced.
    #
    # goooooood -> good
    # pleaaaaase -> please
    #
    # This is deliberately restricted to A-Z characters so
    # Indian-language Unicode text is not damaged.
    # --------------------------------------------------------
    text = re.sub(
        r"([A-Za-z])\1{3,}",
        r"\1\1",
        text
    )

    # --------------------------------------------------------
    # 11. Normalize whitespace
    # --------------------------------------------------------
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # --------------------------------------------------------
    # 12. Case normalization
    #
    # Keep original text untouched.
    # The NLP version uses casefold() for more consistent
    # downstream text processing.
    # --------------------------------------------------------
    text = text.casefold()

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CIVIVOICE - NLP TEXT PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found:")
        print(INPUT_FILE)
        return

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load master dataset
    # --------------------------------------------------------

    print("\nLoading master dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "feedback_id",
        "source",
        "timestamp",
        "text",
        "organization",
        "loc",
        "urgency"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Missing required columns:")
        print(missing_columns)
        return

    # --------------------------------------------------------
    # Preserve original text
    # --------------------------------------------------------

    original_text = df["text"].copy()

    # --------------------------------------------------------
    # Create cleaned text
    # --------------------------------------------------------

    print("\nPreprocessing text...")

    df["text_clean"] = df["text"].apply(preprocess_text)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Row count
    print(f"\nOriginal rows : {len(original_text):,}")
    print(f"Output rows   : {len(df):,}")

    if len(original_text) == len(df):
        print("Row count check: PASS")
    else:
        print("Row count check: FAIL")

    # Original text empty
    original_empty = original_text.fillna("").str.strip().eq("").sum()

    # Clean text empty
    clean_empty = df["text_clean"].fillna("").str.strip().eq("").sum()

    print(f"\nOriginal empty text : {original_empty:,}")
    print(f"Clean empty text    : {clean_empty:,}")

    # Changed records
    changed = (
        original_text.fillna("").astype(str)
        != df["text_clean"].fillna("").astype(str)
    ).sum()

    unchanged = len(df) - changed

    print(f"\nChanged text records   : {changed:,}")
    print(f"Unchanged text records : {unchanged:,}")

    # Check duplicate feedback IDs
    duplicate_ids = df["feedback_id"].duplicated().sum()

    print(f"\nDuplicate feedback IDs: {duplicate_ids:,}")

    # Check required columns
    print("\nFinal columns:")
    for column in df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BEFORE / AFTER EXAMPLES")
    print("=" * 70)

    changed_indices = df.index[
        original_text.fillna("").astype(str)
        != df["text_clean"].fillna("").astype(str)
    ]

    example_indices = changed_indices[:10]

    for idx in example_indices:

        print("\n--- Example ---")

        print("BEFORE:")
        print(str(original_text.loc[idx])[:500])

        print("\nAFTER:")
        print(str(df.loc[idx, "text_clean"])[:500])

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING NLP DATASET")
    print("=" * 70)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved successfully:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("NLP PREPROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()