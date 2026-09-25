import pandas as pd
import regex
from collections import Counter
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"datasets\processed\nlp\master_feedback_nlp.csv"

TOP_N = 50


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_text(text):
    """
    Unicode-aware tokenization.

    Keeps:
    - multilingual words
    - numbers
    - Indian-language scripts

    Removes standalone punctuation from the token sequence.
    """

    if pd.isna(text):
        return []

    text = str(text)

    tokens = regex.findall(
        r"\p{L}[\p{L}\p{M}\p{N}_]*|\p{N}+",
        text
    )

    return [token.casefold() for token in tokens]


# ============================================================
# N-GRAM GENERATION
# ============================================================

def generate_ngrams(tokens, n):
    """
    Generate n-grams from a token list.
    """

    if len(tokens) < n:
        return []

    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


# ============================================================
# CHECK WHETHER N-GRAM IS INFORMATIVE
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "as",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "my",
    "me",
    "we",
    "our",
    "you",
    "your"
}


def is_informative(ngram):
    """
    Keep an n-gram if it contains at least one token
    that is not a common English stopword.

    This is only for exploratory ranking.
    It does NOT modify the dataset.
    """

    return any(
        token not in STOPWORDS
        for token in ngram
    )


# ============================================================
# FORMAT N-GRAM
# ============================================================

def format_ngram(ngram):
    return " ".join(ngram)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CIVIVOICE - N-GRAM / PHRASE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
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
    # Validate column
    # --------------------------------------------------------

    if "text_clean" not in df.columns:
        print("\nERROR: text_clean column not found.")
        return

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    print("\nTokenizing text...")

    df["tokens"] = df["text_clean"].apply(tokenize_text)

    # ========================================================
    # OVERALL BIGRAMS
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 50 RAW BIGRAMS")
    print("=" * 70)

    bigram_counter = Counter()

    for tokens in df["tokens"]:
        bigram_counter.update(
            generate_ngrams(tokens, 2)
        )

    for rank, (ngram, count) in enumerate(
        bigram_counter.most_common(TOP_N),
        start=1
    ):
        print(
            f"{rank:>2}. {format_ngram(ngram):<40} "
            f"{count:>10,}"
        )

    # ========================================================
    # INFORMATIVE BIGRAMS
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 50 INFORMATIVE BIGRAMS")
    print("=" * 70)

    informative_bigrams = Counter()

    for ngram, count in bigram_counter.items():

        if is_informative(ngram):
            informative_bigrams[ngram] = count

    for rank, (ngram, count) in enumerate(
        informative_bigrams.most_common(TOP_N),
        start=1
    ):
        print(
            f"{rank:>2}. {format_ngram(ngram):<40} "
            f"{count:>10,}"
        )

    # ========================================================
    # OVERALL TRIGRAMS
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 50 RAW TRIGRAMS")
    print("=" * 70)

    trigram_counter = Counter()

    for tokens in df["tokens"]:
        trigram_counter.update(
            generate_ngrams(tokens, 3)
        )

    for rank, (ngram, count) in enumerate(
        trigram_counter.most_common(TOP_N),
        start=1
    ):
        print(
            f"{rank:>2}. {format_ngram(ngram):<45} "
            f"{count:>10,}"
        )

    # ========================================================
    # INFORMATIVE TRIGRAMS
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 50 INFORMATIVE TRIGRAMS")
    print("=" * 70)

    informative_trigrams = Counter()

    for ngram, count in trigram_counter.items():

        if is_informative(ngram):
            informative_trigrams[ngram] = count

    for rank, (ngram, count) in enumerate(
        informative_trigrams.most_common(TOP_N),
        start=1
    ):
        print(
            f"{rank:>2}. {format_ngram(ngram):<45} "
            f"{count:>10,}"
        )

    # ========================================================
    # SOURCE-WISE BIGRAMS
    # ========================================================

    print("\n" + "=" * 70)
    print("SOURCE-WISE INFORMATIVE BIGRAMS")
    print("=" * 70)

    for source in sorted(
        df["source"].dropna().unique()
    ):

        source_df = df[
            df["source"] == source
        ]

        source_counter = Counter()

        for tokens in source_df["tokens"]:

            source_counter.update(
                generate_ngrams(tokens, 2)
            )

        source_informative = Counter()

        for ngram, count in source_counter.items():

            if is_informative(ngram):
                source_informative[ngram] = count

        print("\n" + "-" * 70)
        print(f"SOURCE: {source}")

        for rank, (ngram, count) in enumerate(
            source_informative.most_common(20),
            start=1
        ):

            print(
                f"{rank:>2}. {format_ngram(ngram):<40} "
                f"{count:>10,}"
            )

    # ========================================================
    # SAMPLE PHRASE EXAMPLES
    # ========================================================

    print("\n" + "=" * 70)
    print("SAMPLE TEXT + PHRASES")
    print("=" * 70)

    sample = df[
        df["text_clean"]
        .fillna("")
        .str.strip()
        .ne("")
    ].head(5)

    for _, row in sample.iterrows():

        tokens = row["tokens"]

        print("\n---")

        print("TEXT:")
        print(
            str(row["text_clean"])[:300]
        )

        print("\nBIGRAMS:")
        print([
            format_ngram(x)
            for x in generate_ngrams(tokens, 2)[:15]
        ])

        print("\nTRIGRAMS:")
        print([
            format_ngram(x)
            for x in generate_ngrams(tokens, 3)[:15]
        ])

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("N-GRAM ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nNote:"
        "\nThis analysis is exploratory."
        "\nNo rows were deleted."
        "\nThe dataset was not modified."
    )


if __name__ == "__main__":
    main()