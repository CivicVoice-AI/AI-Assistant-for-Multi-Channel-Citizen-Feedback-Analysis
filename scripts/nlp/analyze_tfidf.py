import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"datasets\processed\nlp\master_feedback_nlp.csv"

TOP_N = 30

# Limit vocabulary so the analysis remains practical
MAX_FEATURES = 30000

# Ignore extremely rare terms
MIN_DF = 5

# Ignore terms appearing in more than 95% of documents
MAX_DF = 0.95


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TF-IDF ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = ["feedback_id", "source", "text_clean"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns found.")


# ============================================================
# PREPARE TEXT
# ============================================================

df["text_clean"] = df["text_clean"].fillna("").astype(str)

documents = df["text_clean"].tolist()

print(f"Documents prepared: {len(documents):,}")


# ============================================================
# TF-IDF VECTORIZER
# ============================================================

print("\nCreating TF-IDF matrix...")

vectorizer = TfidfVectorizer(
    lowercase=False,
    token_pattern=r"(?u)\b\w+\b",
    max_features=MAX_FEATURES,
    min_df=MIN_DF,
    max_df=MAX_DF,
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

print("\nTF-IDF matrix created.")

print(f"Matrix shape: {tfidf_matrix.shape}")
print(f"Number of features: {len(feature_names):,}")
print(f"Non-zero values: {tfidf_matrix.nnz:,}")


# ============================================================
# OVERALL TERM IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("TOP TERMS BY TOTAL TF-IDF WEIGHT")
print("=" * 70)

term_scores = tfidf_matrix.sum(axis=0).A1

term_scores_df = pd.DataFrame({
    "term": feature_names,
    "tfidf_score": term_scores
})

term_scores_df = term_scores_df.sort_values(
    "tfidf_score",
    ascending=False
)

print(
    term_scores_df.head(TOP_N).to_string(
        index=False
    )
)


# ============================================================
# SOURCE-WISE TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("SOURCE-WISE TOP TF-IDF TERMS")
print("=" * 70)

sources = df["source"].dropna().unique()

for source in sorted(sources):

    print("\n" + "-" * 70)
    print(f"SOURCE: {source}")
    print("-" * 70)

    source_indices = df.index[
        df["source"] == source
    ].tolist()

    if not source_indices:
        continue

    source_matrix = tfidf_matrix[source_indices]

    source_scores = source_matrix.sum(axis=0).A1

    source_df = pd.DataFrame({
        "term": feature_names,
        "tfidf_score": source_scores
    })

    source_df = source_df[
        source_df["tfidf_score"] > 0
    ].sort_values(
        "tfidf_score",
        ascending=False
    )

    print(
        source_df.head(TOP_N).to_string(
            index=False
        )
    )


# ============================================================
# DOCUMENT-LEVEL EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE DOCUMENT TF-IDF TERMS")
print("=" * 70)

sample_count = min(5, len(df))

for row_number in range(sample_count):

    row = tfidf_matrix[row_number]

    if row.nnz == 0:
        print(
            f"\nDocument {row_number + 1}: "
            "No TF-IDF terms available."
        )
        continue

    scores = row.toarray().flatten()

    top_indices = scores.argsort()[-10:][::-1]

    print(f"\nDocument {row_number + 1}")
    print(f"Feedback ID: {df.iloc[row_number]['feedback_id']}")
    print(f"Source: {df.iloc[row_number]['source']}")

    print("Top terms:")

    for index in top_indices:

        if scores[index] > 0:
            print(
                f"  {feature_names[index]} "
                f"-> {scores[index]:.4f}"
            )


# ============================================================
# VOCABULARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF VOCABULARY SUMMARY")
print("=" * 70)

print(f"Original documents: {len(df):,}")
print(f"TF-IDF vocabulary: {len(feature_names):,}")
print(f"Maximum features allowed: {MAX_FEATURES:,}")
print(f"Minimum document frequency: {MIN_DF}")
print(f"Maximum document frequency: {MAX_DF}")

print("\nAnalysis completed successfully.")
print("No rows were deleted or modified.")
print("=" * 70)