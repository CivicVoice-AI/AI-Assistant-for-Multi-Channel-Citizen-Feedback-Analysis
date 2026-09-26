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
print("CiviVoice — NLP Phase 1: Boilerplate Analysis")
print("=" * 70)

print("\nLoading master dataset...")

df = pd.read_csv(
    MASTER_FILE,
    low_memory=False
)

print("Dataset loaded successfully.")

text = df["text"].fillna("").astype(str)

# ============================================================
# 1. COMMON EXACT TEXTS
# ============================================================

print("\n" + "=" * 70)
print("1. MOST COMMON EXACT TEXTS")
print("=" * 70)

exact_counts = text.value_counts()

print("\nTop 20 repeated complete texts:")

for value, count in exact_counts.head(20).items():
    print(f"\nCount: {count:,}")
    print(value[:500])

# ============================================================
# 2. COMMON LINES / PHRASES
# ============================================================

print("\n" + "=" * 70)
print("2. COMMON STRUCTURAL PHRASES")
print("=" * 70)

patterns = {
    "separator": r"-{5,}",
    "CPGRAMS arrows": r">>",
    "Subject marker": r"\bSubject\s*[:=-]",
    "Name and Address": r"\bName and Address\b",
    "Reference": r"\bReference\s*[:=-]",
    "Date of Application": r"\bDate of Application\b",
    "Bank": r"\bBank\s*:",
    "PF Office": r"\bPF Office\s*:",
    "UAN": r"\bUAN\b",
    "PPO": r"\bPPO\b",
    "Name of": r"\bName of\b",
    "Department": r"\bDepartment\b",
    "Ministry": r"\bMinistry\b",
}

for name, pattern in patterns.items():

    mask = text.str.contains(
        pattern,
        regex=True,
        case=False,
        na=False
    )

    count = mask.sum()

    print(
        f"{name:<25} "
        f"{count:>10,} "
        f"({count / len(df) * 100:>6.2f}%)"
    )

# ============================================================
# 3. SEPARATOR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("3. SEPARATOR ANALYSIS")
print("=" * 70)

separator_pattern = r"-{5,}"

separator_counts = text.str.count(separator_pattern)

print(
    f"Records containing long separators: "
    f"{(separator_counts > 0).sum():,}"
)

print(
    f"Total separator occurrences: "
    f"{separator_counts.sum():,}"
)

# ============================================================
# 4. COMMON CPGRAMS STRUCTURAL PREFIXES
# ============================================================

print("\n" + "=" * 70)
print("4. COMMON CPGRAMS STRUCTURAL PREFIXES")
print("=" * 70)

cpgrams = df[df["source"] == "CPGRAMS"]["text"]

prefix_counter = Counter()

for value in cpgrams:

    # Take text before the first separator.
    prefix = re.split(r"-{5,}", value, maxsplit=1)[0].strip()

    if prefix:
        prefix_counter[prefix] += 1

print("\nTop 30 prefixes before separator:")

for prefix, count in prefix_counter.most_common(30):

    print(f"\nCount: {count:,}")
    print(prefix[:500])

# ============================================================
# 5. COMMON FIRST LINES
# ============================================================

print("\n" + "=" * 70)
print("5. COMMON FIRST LINES")
print("=" * 70)

first_line_counter = Counter()

for value in text:

    lines = value.splitlines()

    if lines:
        first_line = lines[0].strip()

        if first_line:
            first_line_counter[first_line] += 1

print("\nTop 30 first lines:")

for line, count in first_line_counter.most_common(30):

    print(f"\nCount: {count:,}")
    print(line[:500])

# ============================================================
# 6. SOURCE-WISE EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("6. SOURCE-WISE EXACT TEXT DUPLICATES")
print("=" * 70)

for source, group in df.groupby("source"):

    counts = group["text"].value_counts()

    duplicate_records = (counts > 1).sum()
    duplicate_rows = counts[counts > 1].sum()

    print(f"\n{source}")
    print(f"  Unique texts repeated: {duplicate_records:,}")
    print(f"  Records belonging to repeated texts: {duplicate_rows:,}")

# ============================================================
# 7. SOURCE-WISE BOILERPLATE PATTERNS
# ============================================================

print("\n" + "=" * 70)
print("7. BOILERPLATE PATTERNS BY SOURCE")
print("=" * 70)

for source, group in df.groupby("source"):

    source_text = group["text"]

    print(f"\n--- {source} ---")

    for name, pattern in patterns.items():

        count = source_text.str.contains(
            pattern,
            regex=True,
            case=False,
            na=False
        ).sum()

        if count > 0:
            print(
                f"  {name:<23} "
                f"{count:>8,} "
                f"({count / len(group) * 100:>6.2f}%)"
            )

# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("BOILERPLATE ANALYSIS COMPLETE")
print("=" * 70)

print("\nNo files were modified.")
print("No preprocessing was performed.")