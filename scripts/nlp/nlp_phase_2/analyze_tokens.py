import pandas as pd
import regex
from collections import Counter
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"datasets\processed\nlp\master_feedback_nlp.csv"


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(text):
    """
    Unicode-aware tokenization.

    Keeps:
    - English words
    - Indian-language words
    - Numbers
    - Unicode text

    Removes standalone punctuation from the token list.
    """

    if pd.isna(text):
        return []

    text = str(text)

    # --------------------------------------------------------
    # Unicode-aware token pattern
    #
    # \p{L} = any Unicode letter
    # \p{M} = Unicode combining marks
    # \p{N} = any Unicode number
    #
    # This allows words from multiple Indian scripts to remain
    # intact instead of splitting them into individual letters.
    # --------------------------------------------------------

    tokens = regex.findall(
        r"\p{L}[\p{L}\p{M}\p{N}_]*|\p{N}+",
        text
    )

    return [token.casefold() for token in tokens]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CIVIVOICE - MULTILINGUAL TOKEN & VOCABULARY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        print("\nERROR: Input file not found:")
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

    print(f"Rows loaded: {len(df):,}")

    # --------------------------------------------------------
    # Validate text_clean
    # --------------------------------------------------------

    if "text_clean" not in df.columns:
        print("\nERROR: text_clean column not found.")
        return

    # --------------------------------------------------------
    # Tokenize all text
    # --------------------------------------------------------

    print("\nTokenizing text...")

    df["tokens"] = df["text_clean"].apply(tokenize_text)

    # --------------------------------------------------------
    # Calculate token statistics
    # --------------------------------------------------------

    total_tokens = sum(
        len(tokens)
        for tokens in df["tokens"]
    )

    vocabulary = Counter()

    for tokens in df["tokens"]:
        vocabulary.update(tokens)

    vocabulary_size = len(vocabulary)

    print("\n" + "=" * 70)
    print("OVERALL TOKEN STATISTICS")
    print("=" * 70)

    print(f"\nTotal tokens       : {total_tokens:,}")
    print(f"Unique vocabulary  : {vocabulary_size:,}")

    if total_tokens > 0:

        lexical_diversity = (
            vocabulary_size / total_tokens
        )

        print(
            f"Vocabulary / token ratio: "
            f"{lexical_diversity:.4f}"
        )

    # --------------------------------------------------------
    # Most frequent tokens
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP 50 MOST FREQUENT TOKENS")
    print("=" * 70)

    for rank, (token, count) in enumerate(
        vocabulary.most_common(50),
        start=1
    ):
        print(
            f"{rank:>2}. {token:<30} {count:>10,}"
        )

    # --------------------------------------------------------
    # Rare tokens
    # --------------------------------------------------------

    rare_tokens = sum(
        1
        for token, count in vocabulary.items()
        if count == 1
    )

    low_frequency_tokens = sum(
        1
        for token, count in vocabulary.items()
        if count <= 5
    )

    print("\n" + "=" * 70)
    print("VOCABULARY FREQUENCY")
    print("=" * 70)

    print(
        f"\nTokens appearing exactly once : "
        f"{rare_tokens:,}"
    )

    print(
        f"Tokens appearing <=5 times    : "
        f"{low_frequency_tokens:,}"
    )

    # --------------------------------------------------------
    # Source-wise vocabulary analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SOURCE-WISE VOCABULARY")
    print("=" * 70)

    for source in sorted(df["source"].dropna().unique()):

        source_df = df[df["source"] == source]

        source_counter = Counter()

        for tokens in source_df["tokens"]:
            source_counter.update(tokens)

        source_total = sum(source_counter.values())
        source_vocab = len(source_counter)

        print("\n" + "-" * 70)
        print(f"SOURCE: {source}")

        print(
            f"Records           : {len(source_df):,}"
        )

        print(
            f"Total tokens      : {source_total:,}"
        )

        print(
            f"Unique vocabulary : {source_vocab:,}"
        )

        print("\nTop 20 tokens:")

        for rank, (token, count) in enumerate(
            source_counter.most_common(20),
            start=1
        ):
            print(
                f"{rank:>2}. {token:<30} {count:>10,}"
            )

    # --------------------------------------------------------
    # Multilingual token inspection
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MULTILINGUAL TOKEN INSPECTION")
    print("=" * 70)

    script_patterns = {
        "Latin": r"\p{Latin}",
        "Devanagari": r"\p{Devanagari}",
        "Kannada": r"\p{Kannada}",
        "Tamil": r"\p{Tamil}",
        "Telugu": r"\p{Telugu}",
        "Malayalam": r"\p{Malayalam}",
        "Gujarati": r"\p{Gujarati}",
        "Bengali": r"\p{Bengali}",
        "Gurmukhi": r"\p{Gurmukhi}",
        "Oriya": r"\p{Oriya}"
    }

    for script_name, pattern in script_patterns.items():

        script_tokens = [
            (token, count)
            for token, count in vocabulary.items()
            if regex.search(pattern, token)
        ]

        script_tokens.sort(
            key=lambda x: x[1],
            reverse=True
        )

        print("\n" + "-" * 70)
        print(f"{script_name} tokens")

        print(
            f"Unique tokens containing "
            f"{script_name} characters: "
            f"{len(script_tokens):,}"
        )

        print("Top 10:")

        for rank, (token, count) in enumerate(
            script_tokens[:10],
            start=1
        ):
            print(
                f"{rank:>2}. {token:<30} {count:>10,}"
            )

    # --------------------------------------------------------
    # Sample tokenization examples
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOKENIZATION EXAMPLES")
    print("=" * 70)

    examples = df[
        df["text_clean"]
        .fillna("")
        .str.strip()
        .ne("")
    ].head(10)

    for _, row in examples.iterrows():

        print("\n---")

        print(
            "TEXT:"
        )

        print(
            str(row["text_clean"])[:300]
        )

        print(
            "\nTOKENS:"
        )

        print(
            row["tokens"][:30]
        )

    # --------------------------------------------------------
    # Final note
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOKEN ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nNote:"
        "\nThis analysis is exploratory."
        "\nNo rows were deleted."
        "\nThe master dataset and NLP dataset were not modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()