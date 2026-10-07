import os
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF


# ============================================================
# CiviVoice — NLP Phase 3: Improved Topic Analysis
# ============================================================

INPUT_FILE = r"datasets\processed\nlp\master_feedback_nlp.csv"
OUTPUT_DIR = r"datasets\processed\nlp\topic_analysis"

N_TOPICS = 10
TOP_WORDS = 15
SAMPLE_SIZE = 50000
RANDOM_STATE = 42


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("CiviVoice — NLP Phase 3: Improved Topic Analysis")
print("=" * 70)

print("\nLoading master dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")

df = df[df["text"].notna()].copy()
df["text"] = df["text"].astype(str)

print(f"Records with non-empty text: {len(df):,}")


# ============================================================
# 2. TOPIC-SPECIFIC CLEANING
# ============================================================

# These are structural / administrative words identified from
# the Phase 1 boilerplate analysis.
BOILERPLATE_TERMS = {
    "provided",
    "office",
    "account",
    "number",
    "certificate",
    "scheme",
    "department",
    "ministry",
    "reference",
    "subject",
    "application",
    "name",
    "address",
    "bank",
    "branch",
    "ppo",
    "uan",
    "pf",
    "gist",
    "pmo",
    "copy",
    "sent",
    "kindly",
    "government",
    "state",
    "central",
    "related",
    "others",
    "other",
    "details",
    "mentioned",
    "received",
    "attached",
    "attachment",
    "view",
    "scanned",
    "sub",
    "sir",
    "madam",
    "dear",
    "please",
    "regarding",
    "matter",
    "action",
}

# Common structural CPGRAMS phrases.
BOILERPLATE_PHRASES = [
    r"government\s+kindly",
    r"state\s+government",
    r"central\s+government",
    r"government\s+related",
    r"department\s+related",
    r"office\s+related",
    r"pmo\s+follows",
    r"send\s+pmo",
    r"copy\s+sent",
    r"view\s+scanned",
    r"gist\s+of",
    r"gist",
    r"as\s+per\s+attachment",
    r"as\s+per\s+attatchment",
    r"grievance\s+attached",
    r"call\s+disconnected",
    r"disconnect\s+call",
]

# English stopwords + topic-specific administrative words.
STOP_WORDS = set(ENGLISH_STOP_WORDS)
STOP_WORDS.update(BOILERPLATE_TERMS)


def clean_for_topics(text):
    """
    Topic-specific cleaning.

    IMPORTANT:
    This does NOT modify the original master dataset.
    It only creates cleaned text for topic modelling.
    """

    text = str(text)

    # --------------------------------------------------------
    # Remove HTML entities
    # --------------------------------------------------------
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"&#x[0-9a-fA-F]+;", " ", text)

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # --------------------------------------------------------
    # Remove email addresses
    # --------------------------------------------------------
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text,
    )

    # --------------------------------------------------------
    # Remove @mentions
    # --------------------------------------------------------
    text = re.sub(r"@\w+", " ", text)

    # --------------------------------------------------------
    # Remove hashtags symbol but KEEP the actual word
    # --------------------------------------------------------
    text = re.sub(r"#(\w+)", r"\1", text)

    # --------------------------------------------------------
    # Remove long separators / CPGRAMS formatting
    # Examples:
    # -----------------------
    # X-X-X-X-X
    # >>>>
    # --------------------------------------------------------
    text = re.sub(r"[-_=]{3,}", " ", text)
    text = re.sub(r"(?:X[-X]*){3,}", " ", text)
    text = re.sub(r"[>|]{2,}", " ", text)

    # --------------------------------------------------------
    # Remove known structural phrases
    # --------------------------------------------------------
    text_lower = text.lower()

    for pattern in BOILERPLATE_PHRASES:
        text_lower = re.sub(pattern, " ", text_lower)

    text = text_lower

    # --------------------------------------------------------
    # Remove CPGRAMS-style short codes
    # Examples:
    # PG/SAT
    # PG/BA
    # PG/MD
    # PG/RJ
    # STS
    # FRANK
    # TV
    # SUB
    # --------------------------------------------------------
    text = re.sub(r"\bpg/[a-z]{2,10}\b", " ", text)
    text = re.sub(r"\b(?:sts|frank|tv|sub)\b", " ", text)

    # --------------------------------------------------------
    # Remove standalone field labels
    # --------------------------------------------------------
    field_patterns = [
        r"\bname\s+and\s+address\b",
        r"\bdate\s+of\s+application\b",
        r"\bname\s+of\s+department\b",
        r"\bname\s+of\s+bank\b",
        r"\bpf\s+office\b",
        r"\bscheme\s+certificate\s+number\b",
        r"\bpension\s+payment\s+order\b",
        r"\baccount\s+number\b",
        r"\buan\s+no\b",
        r"\bppo\s+no\b",
    ]

    for pattern in field_patterns:
        text = re.sub(pattern, " ", text)

    # --------------------------------------------------------
    # Remove excessive punctuation
    # --------------------------------------------------------
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # --------------------------------------------------------
    # Tokenize while preserving Unicode characters
    # This allows Hindi and other Indian-language text to remain.
    # --------------------------------------------------------
    tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)

    cleaned_tokens = []

    for token in tokens:

        token_lower = token.lower()

        # Remove English stopwords / administrative boilerplate
        if token_lower in STOP_WORDS:
            continue

        # Remove very short tokens
        if len(token_lower) < 3:
            continue

        # Remove tokens consisting only of digits
        if token_lower.isdigit():
            continue

        cleaned_tokens.append(token_lower)

    return " ".join(cleaned_tokens)


print("\nCleaning text for topic modelling...")

df["topic_text"] = df["text"].apply(clean_for_topics)

# Remove rows that became empty
df = df[df["topic_text"].str.strip().ne("")].copy()

print(f"Records remaining after topic cleaning: {len(df):,}")


# ============================================================
# 3. SAMPLE DATA
# ============================================================

if len(df) > SAMPLE_SIZE:
    topic_df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).copy()
else:
    topic_df = df.copy()

print(f"Records used for topic modelling: {len(topic_df):,}")


# ============================================================
# 4. TF-IDF
# ============================================================

print("\nCreating TF-IDF matrix...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.85,
    sublinear_tf=True,
)

tfidf_matrix = vectorizer.fit_transform(topic_df["topic_text"])

print(f"TF-IDF shape: {tfidf_matrix.shape}")


# ============================================================
# 5. NMF TOPIC MODEL
# ============================================================

print("\nRunning NMF topic model...")

nmf = NMF(
    n_components=N_TOPICS,
    random_state=RANDOM_STATE,
    init="nndsvda",
    max_iter=400,
)

topic_matrix = nmf.fit_transform(tfidf_matrix)

print(f"Topic matrix shape: {topic_matrix.shape}")


# ============================================================
# 6. TOP WORDS PER TOPIC
# ============================================================

feature_names = vectorizer.get_feature_names_out()

topic_rows = []

for topic_idx, topic in enumerate(nmf.components_):

    top_indices = topic.argsort()[-TOP_WORDS:][::-1]

    words = [
        feature_names[i]
        for i in top_indices
    ]

    scores = [
        float(topic[i])
        for i in top_indices
    ]

    topic_rows.append({
        "topic": topic_idx + 1,
        "top_words": ", ".join(words),
        "top_word_scores": ", ".join(
            f"{score:.4f}" for score in scores
        ),
    })

topic_words_df = pd.DataFrame(topic_rows)


# ============================================================
# 7. ASSIGN TOPIC TO EACH RECORD
# ============================================================

topic_assignments = topic_matrix.argmax(axis=1) + 1
topic_scores = topic_matrix.max(axis=1)

topic_df["topic"] = topic_assignments
topic_df["topic_score"] = topic_scores


# ============================================================
# 8. TOPIC DISTRIBUTION
# ============================================================

topic_counts = (
    topic_df["topic"]
    .value_counts()
    .sort_index()
)

topic_distribution_df = pd.DataFrame({
    "topic": topic_counts.index,
    "record_count": topic_counts.values,
})

topic_distribution_df["percentage"] = (
    topic_distribution_df["record_count"]
    / len(topic_df)
    * 100
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

topic_words_path = os.path.join(
    OUTPUT_DIR,
    "topic_words_v3.csv"
)

topic_distribution_path = os.path.join(
    OUTPUT_DIR,
    "topic_distribution_v3.csv"
)

topic_assignments_path = os.path.join(
    OUTPUT_DIR,
    "topic_assignments_v3.csv"
)

topic_words_df.to_csv(
    topic_words_path,
    index=False,
    encoding="utf-8-sig"
)

topic_distribution_df.to_csv(
    topic_distribution_path,
    index=False,
    encoding="utf-8-sig"
)

topic_df[
    [
        "feedback_id",
        "source",
        "timestamp",
        "text",
        "topic_text",
        "topic",
        "topic_score",
    ]
].to_csv(
    topic_assignments_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TOP WORDS PER TOPIC")
print("=" * 70)

for _, row in topic_words_df.iterrows():

    print(f"\nTopic {int(row['topic'])}:")
    print(row["top_words"])


print("\n" + "=" * 70)
print("TOPIC DISTRIBUTION")
print("=" * 70)

for _, row in topic_distribution_df.iterrows():

    print(
        f"Topic {int(row['topic'])}: "
        f"{int(row['record_count']):,} "
        f"({row['percentage']:.2f}%)"
    )


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(topic_words_path)
print(topic_distribution_path)
print(topic_assignments_path)

print("\nTopic analysis complete.")
print("Original master dataset was NOT modified.")