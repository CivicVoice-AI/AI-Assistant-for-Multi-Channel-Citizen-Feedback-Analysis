import os
import regex as re
from collections import Counter, defaultdict

import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"datasets\processed\nlp\master_feedback_nlp.csv"

# Unicode-aware tokenizer:
# Words can contain letters, marks, numbers and underscore.
TOKEN_PATTERN = re.compile(
    r"\p{L}[\p{L}\p{M}\p{N}_]*|\p{N}+",
    re.UNICODE
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def tokenize(text):
    """
    Convert text into lowercase/casefolded tokens.

    Example:
        "Water Problem in Area"
        -> ["water", "problem", "in", "area"]
    """

    if pd.isna(text):
        return []

    text = str(text).casefold()

    return TOKEN_PATTERN.findall(text)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("TERM FREQUENCY AND DOCUMENT FREQUENCY ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
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
# TERM FREQUENCY + DOCUMENT FREQUENCY
# ============================================================

print("\nAnalyzing terms...")

total_term_frequency = Counter()
document_frequency = Counter()

# Store source-wise document frequency
source_document_frequency = defaultdict(Counter)

# Store number of documents per source
source_document_counts = Counter()

total_documents = len(df)

for _, row in df.iterrows():

    source = str(row["source"])
    tokens = tokenize(row["text_clean"])

    source_document_counts[source] += 1

    # Total frequency:
    # Every occurrence of a token is counted.
    total_term_frequency.update(tokens)

    # Document frequency:
    # A token is counted only once per document.
    unique_tokens = set(tokens)

    document_frequency.update(unique_tokens)

    # Source-wise document frequency
    source_document_frequency[source].update(unique_tokens)


# ============================================================
# BASIC SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total documents: {total_documents:,}")
print(f"Unique terms: {len(total_term_frequency):,}")
print(f"Total tokens: {sum(total_term_frequency.values()):,}")


# ============================================================
# TOP TERMS BY TOTAL FREQUENCY
# ============================================================

print("\n" + "=" * 70)
print("TOP TERMS BY TOTAL TERM FREQUENCY")
print("=" * 70)

top_frequency = total_term_frequency.most_common(30)

print(f"{'term':<25} {'frequency':>12}")

for term, count in top_frequency:
    print(f"{term:<25} {count:>12,}")


# ============================================================
# TOP TERMS BY DOCUMENT FREQUENCY
# ============================================================

print("\n" + "=" * 70)
print("TOP TERMS BY DOCUMENT FREQUENCY")
print("=" * 70)

top_document_frequency = document_frequency.most_common(30)

print(
    f"{'term':<25} "
    f"{'documents':>12} "
    f"{'coverage':>12}"
)

for term, count in top_document_frequency:

    coverage = (count / total_documents) * 100

    print(
        f"{term:<25} "
        f"{count:>12,} "
        f"{coverage:>11.2f}%"
    )


# ============================================================
# VERY RARE TERMS
# ============================================================

print("\n" + "=" * 70)
print("RARE TERMS")
print("=" * 70)

rare_terms = [
    term
    for term, count in document_frequency.items()
    if count == 1
]

rare_terms_sorted = sorted(
    rare_terms,
    key=lambda term: (-total_term_frequency[term], term)
)

print(
    f"Terms appearing in exactly 1 document: "
    f"{len(rare_terms_sorted):,}"
)

print("\nSample rare terms:")

for term in rare_terms_sorted[:50]:
    print(
        f"{term:<30} "
        f"frequency={total_term_frequency[term]}"
    )


# ============================================================
# TERMS APPEARING IN <= 5 DOCUMENTS
# ============================================================

print("\n" + "=" * 70)
print("LOW DOCUMENT FREQUENCY TERMS")
print("=" * 70)

low_df_counts = Counter()

for term, count in document_frequency.items():

    if count <= 5:
        low_df_counts[count] += 1

for df_count in sorted(low_df_counts):

    print(
        f"Terms appearing in {df_count} document(s): "
        f"{low_df_counts[df_count]:,}"
    )


# ============================================================
# TERMS WITH HIGH DOCUMENT COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("HIGH DOCUMENT COVERAGE TERMS")
print("=" * 70)

coverage_threshold = 50

high_coverage_terms = []

for term, count in document_frequency.items():

    coverage = (count / total_documents) * 100

    if coverage >= coverage_threshold:
        high_coverage_terms.append(
            (term, count, coverage)
        )

high_coverage_terms.sort(
    key=lambda x: (-x[1], x[0])
)

print(
    f"Terms appearing in at least "
    f"{coverage_threshold}% of documents: "
    f"{len(high_coverage_terms):,}"
)

print(
    f"\n{'term':<25} "
    f"{'documents':>12} "
    f"{'coverage':>12}"
)

for term, count, coverage in high_coverage_terms[:50]:

    print(
        f"{term:<25} "
        f"{count:>12,} "
        f"{coverage:>11.2f}%"
    )


# ============================================================
# SOURCE-WISE DOCUMENT FREQUENCY
# ============================================================

print("\n" + "=" * 70)
print("SOURCE-WISE TERM COVERAGE")
print("=" * 70)

for source in sorted(source_document_counts):

    print("\n" + "-" * 70)
    print(f"SOURCE: {source}")
    print("-" * 70)

    source_total = source_document_counts[source]

    print(f"Documents: {source_total:,}")

    print(
        f"\n{'term':<25} "
        f"{'documents':>12} "
        f"{'coverage':>12}"
    )

    source_terms = (
        source_document_frequency[source]
        .most_common(20)
    )

    for term, count in source_terms:

        coverage = (count / source_total) * 100

        print(
            f"{term:<25} "
            f"{count:>12,} "
            f"{coverage:>11.2f}%"
        )


# ============================================================
# SPECIFIC TERMS OF INTEREST
# ============================================================

print("\n" + "=" * 70)
print("SELECTED TERM CHECK")
print("=" * 70)

terms_to_check = [
    "x",
    "pg",
    "the",
    "of",
    "to",
    "and",
    "problem",
    "complaint",
    "issue",
    "water",
    "traffic",
    "help"
]

print(
    f"{'term':<20} "
    f"{'frequency':>15} "
    f"{'documents':>15} "
    f"{'coverage':>12}"
)

for term in terms_to_check:

    frequency = total_term_frequency.get(term, 0)
    documents = document_frequency.get(term, 0)

    coverage = (
        documents / total_documents * 100
        if total_documents > 0
        else 0
    )

    print(
        f"{term:<20} "
        f"{frequency:>15,} "
        f"{documents:>15,} "
        f"{coverage:>11.2f}%"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("No rows were deleted.")
print("No dataset values were modified.")
print("Analysis was exploratory only.")