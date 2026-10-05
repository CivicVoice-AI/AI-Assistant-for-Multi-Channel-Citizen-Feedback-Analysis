import pandas as pd
import re

from pathlib import Path

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    ENGLISH_STOP_WORDS
)

from sklearn.decomposition import NMF


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "datasets/processed/nlp/master_feedback_nlp.csv"
)

OUTPUT_DIR = Path(
    "datasets/processed/nlp/topic_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

N_TOPICS = 10
TOP_WORDS = 15
SAMPLE_SIZE = 50000
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CIVICVOICE - IMPROVED TOPIC / ISSUE DISCOVERY")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# CHECK TEXT COLUMN
# ============================================================

if "text_clean" not in df.columns:

    print(
        "\nERROR: text_clean column not found."
    )

    raise SystemExit(1)


# ============================================================
# CLEAN EMPTY TEXT
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

print(
    f"\nRecords with non-empty text: "
    f"{len(df):,}"
)


# ============================================================
# SAMPLE DATA
# ============================================================

print("\n" + "=" * 70)
print("1. SELECTING DATA")
print("=" * 70)

if len(df) > SAMPLE_SIZE:

    topic_df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).copy()

else:

    topic_df = df.copy()

print(
    f"\nRecords used: "
    f"{len(topic_df):,}"
)


# ============================================================
# CUSTOM TOKEN FILTER
# ============================================================

print("\n" + "=" * 70)
print("2. TEXT FILTERING")
print("=" * 70)


# Convert stopword set to a normal Python set
stop_words = set(ENGLISH_STOP_WORDS)


def clean_for_topics(text):

    # Convert to lowercase
    text = text.lower()

    # Keep English/Hindi Unicode letters and numbers
    tokens = re.findall(
        r"[^\W\d_]+",
        text,
        flags=re.UNICODE
    )

    cleaned_tokens = []

    for token in tokens:

        # Remove very short tokens
        if len(token) < 3:
            continue

        # Remove English stopwords
        if token in stop_words:
            continue

        cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


topic_df["topic_text"] = (
    topic_df["text_clean"]
    .apply(clean_for_topics)
)


# Remove empty results
topic_df = topic_df[
    topic_df["topic_text"].str.strip() != ""
].copy()


print(
    f"\nRecords after filtering: "
    f"{len(topic_df):,}"
)


# ============================================================
# TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("3. TF-IDF FEATURE EXTRACTION")
print("=" * 70)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90,
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(
    topic_df["topic_text"]
)

print(
    f"\nTF-IDF matrix shape: "
    f"{tfidf_matrix.shape}"
)

print(
    f"Number of non-zero values: "
    f"{tfidf_matrix.nnz:,}"
)


# ============================================================
# NMF
# ============================================================

print("\n" + "=" * 70)
print("4. NMF TOPIC MODELING")
print("=" * 70)

print(
    f"\nNumber of topics: "
    f"{N_TOPICS}"
)

nmf_model = NMF(
    n_components=N_TOPICS,
    init="nndsvda",
    random_state=RANDOM_STATE,
    max_iter=300
)

topic_matrix = nmf_model.fit_transform(
    tfidf_matrix
)

print(
    "\nNMF topic modeling completed."
)

print(
    f"Topic matrix shape: "
    f"{topic_matrix.shape}"
)


# ============================================================
# TOP WORDS
# ============================================================

print("\n" + "=" * 70)
print("5. TOP WORDS FOR EACH TOPIC")
print("=" * 70)

feature_names = (
    vectorizer.get_feature_names_out()
)

topic_results = []

for topic_index, topic in enumerate(
    nmf_model.components_
):

    top_indices = topic.argsort()[
        -TOP_WORDS:
    ][::-1]

    top_words = [
        feature_names[i]
        for i in top_indices
    ]

    topic_results.append({
        "Topic": f"Topic_{topic_index + 1}",
        "Top_Words": ", ".join(top_words)
    })

    print(
        f"\nTopic {topic_index + 1}:"
    )

    print(
        ", ".join(top_words)
    )


# ============================================================
# SAVE TOPIC WORDS
# ============================================================

topic_words_df = pd.DataFrame(
    topic_results
)

topic_words_file = (
    OUTPUT_DIR /
    "topic_words_v2.csv"
)

topic_words_df.to_csv(
    topic_words_file,
    index=False
)

print(
    f"\nTopic words saved to:"
)

print(topic_words_file)


# ============================================================
# DOMINANT TOPIC
# ============================================================

print("\n" + "=" * 70)
print("6. DOMINANT TOPIC DISTRIBUTION")
print("=" * 70)

dominant_topics = (
    topic_matrix.argmax(axis=1)
)

topic_df["dominant_topic"] = (
    dominant_topics + 1
)

topic_distribution = (
    topic_df["dominant_topic"]
    .value_counts()
    .sort_index()
)

topic_distribution_df = pd.DataFrame({
    "Topic": topic_distribution.index,
    "Records": topic_distribution.values,
    "Percentage": (
        topic_distribution.values /
        len(topic_df) * 100
    ).round(2)
})

print("\n")

print(
    topic_distribution_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE DISTRIBUTION
# ============================================================

distribution_file = (
    OUTPUT_DIR /
    "topic_distribution_v2.csv"
)

topic_distribution_df.to_csv(
    distribution_file,
    index=False
)

print(
    f"\nTopic distribution saved to:"
)

print(distribution_file)


# ============================================================
# SAVE TOPIC ASSIGNMENTS
# ============================================================

assignment_df = topic_df[
    [
        "source",
        "text_clean",
        "dominant_topic"
    ]
].copy()

assignment_file = (
    OUTPUT_DIR /
    "topic_assignments_v2.csv"
)

assignment_df.to_csv(
    assignment_file,
    index=False
)

print(
    f"\nTopic assignments saved to:"
)

print(assignment_file)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("7. VALIDATION")
print("=" * 70)

print(
    f"\nOriginal dataset rows: "
    f"{len(df):,}"
)

print(
    f"Records used for topic modeling: "
    f"{len(topic_df):,}"
)

print(
    f"Number of topics: "
    f"{N_TOPICS}"
)

print(
    f"Top words per topic: "
    f"{TOP_WORDS}"
)

print(
    "\nOriginal master dataset was NOT modified."
)

print("\n" + "=" * 70)
print("IMPROVED TOPIC DISCOVERY COMPLETE")
print("=" * 70)