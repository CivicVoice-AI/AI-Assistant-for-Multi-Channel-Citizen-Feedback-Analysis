import pandas as pd
import re
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

MASTER_FILE = r"datasets\masterd\master_feedback.csv"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CiviVoice — NLP Phase 1: Text Exploration")
print("=" * 70)

print("\nLoading master dataset...")

df = pd.read_csv(
    MASTER_FILE,
    low_memory=False
)

print("Dataset loaded successfully.")

# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATASET INFORMATION")
print("=" * 70)

print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")

# ============================================================
# SOURCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("2. SOURCE DISTRIBUTION")
print("=" * 70)

source_counts = df["source"].value_counts(dropna=False)

print(source_counts)

# ============================================================
# TEXT QUALITY
# ============================================================

print("\n" + "=" * 70)
print("3. TEXT QUALITY")
print("=" * 70)

text = df["text"].fillna("").astype(str)

empty_text = (text.str.strip() == "").sum()

print(f"Empty texts: {empty_text:,}")

# ============================================================
# TEXT LENGTH
# ============================================================

print("\n" + "=" * 70)
print("4. TEXT LENGTH STATISTICS")
print("=" * 70)

char_lengths = text.str.len()
word_lengths = text.str.split().str.len()

print("\nCharacter length:")
print(char_lengths.describe())

print("\nWord count:")
print(word_lengths.describe())

# ============================================================
# VERY SHORT TEXT
# ============================================================

print("\n" + "=" * 70)
print("5. VERY SHORT TEXT")
print("=" * 70)

short_mask = word_lengths < 5

print(f"Texts with fewer than 5 words: {short_mask.sum():,}")

print("\nExamples:")

short_examples = text[short_mask].drop_duplicates().head(30)

for value in short_examples:
    print(f"  {repr(value)}")

# ============================================================
# LONGEST TEXT
# ============================================================

print("\n" + "=" * 70)
print("6. LONGEST TEXTS")
print("=" * 70)

longest_indices = char_lengths.nlargest(10).index

for i, index in enumerate(longest_indices, start=1):
    print(f"\n--- Example {i} ({char_lengths[index]:,} characters) ---")
    print(text[index][:1000])

# ============================================================
# WORD FREQUENCY
# ============================================================

print("\n" + "=" * 70)
print("7. COMMON WORDS")
print("=" * 70)

# Basic tokenization for EXPLORATION ONLY.
# We are NOT deciding preprocessing rules yet.

all_words = []

for value in text:
    words = re.findall(r"\b\w+\b", value.lower())
    all_words.extend(words)

word_counter = Counter(all_words)

print(f"Total word tokens: {len(all_words):,}")
print(f"Unique words: {len(word_counter):,}")

print("\nTop 30 words:")

for word, count in word_counter.most_common(30):
    print(f"  {word}: {count:,}")

# ============================================================
# SOURCE-WISE TEXT LENGTH
# ============================================================

print("\n" + "=" * 70)
print("8. SOURCE-WISE TEXT LENGTH")
print("=" * 70)

source_stats = (
    df.assign(
        character_length=char_lengths,
        word_count=word_lengths
    )
    .groupby("source", dropna=False)
    .agg(
        records=("text", "size"),
        avg_characters=("character_length", "mean"),
        median_characters=("character_length", "median"),
        avg_words=("word_count", "mean"),
        median_words=("word_count", "median")
    )
    .sort_values("records", ascending=False)
)

print(source_stats.to_string())

# ============================================================
# SOURCE-WISE SHORT TEXT
# ============================================================

print("\n" + "=" * 70)
print("9. SOURCE-WISE VERY SHORT TEXT")
print("=" * 70)

short_by_source = (
    df.assign(
        word_count=word_lengths,
        is_short=short_mask
    )
    .groupby("source", dropna=False)
    .agg(
        total_records=("text", "size"),
        short_texts=("is_short", "sum")
    )
)

short_by_source["short_text_percentage"] = (
    short_by_source["short_texts"]
    / short_by_source["total_records"]
    * 100
)

print(short_by_source.to_string())

# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("TEXT EXPLORATION COMPLETE")
print("=" * 70)

print("\nNo files were modified.")
print("No preprocessing was performed.")