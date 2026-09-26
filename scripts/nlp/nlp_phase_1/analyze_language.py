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
print("CiviVoice — NLP Phase 1: Language Analysis")
print("=" * 70)

print("\nLoading master dataset...")

df = pd.read_csv(
    MASTER_FILE,
    low_memory=False
)

print("Dataset loaded successfully.")

text = df["text"].fillna("").astype(str)

# ============================================================
# UNICODE SCRIPT DETECTION
# ============================================================

def detect_scripts(value):
    """
    Detect major writing systems present in the text.
    This is script detection, NOT language identification.
    """

    scripts = set()

    if re.search(r"[\u0900-\u097F]", value):
        scripts.add("Devanagari")

    if re.search(r"[\u0C80-\u0CFF]", value):
        scripts.add("Kannada")

    if re.search(r"[\u0B80-\u0BFF]", value):
        scripts.add("Tamil")

    if re.search(r"[\u0C00-\u0C7F]", value):
        scripts.add("Telugu")

    if re.search(r"[\u0D00-\u0D7F]", value):
        scripts.add("Malayalam")

    if re.search(r"[\u0A80-\u0AFF]", value):
        scripts.add("Gujarati")

    if re.search(r"[\u0A00-\u0A7F]", value):
        scripts.add("Gurmukhi")

    if re.search(r"[\u0980-\u09FF]", value):
        scripts.add("Bengali")

    if re.search(r"[\u0600-\u06FF]", value):
        scripts.add("Arabic")

    if re.search(r"[A-Za-z]", value):
        scripts.add("Latin")

    return ", ".join(sorted(scripts)) if scripts else "Other/Unknown"


print("\nDetecting writing systems...")

script_labels = text.apply(detect_scripts)

# ============================================================
# SCRIPT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("1. WRITING SYSTEM DISTRIBUTION")
print("=" * 70)

script_counts = script_labels.value_counts()

print(script_counts)

print("\nPercentages:")

script_percentages = (
    script_labels.value_counts(normalize=True) * 100
)

for script, percentage in script_percentages.items():
    print(f"  {script}: {percentage:.2f}%")

# ============================================================
# PURE VS MIXED SCRIPT
# ============================================================

print("\n" + "=" * 70)
print("2. PURE VS MIXED SCRIPT TEXT")
print("=" * 70)

def script_type(label):
    if "," not in label:
        return "Single script"
    return "Mixed scripts"


script_types = script_labels.apply(script_type)

print(script_types.value_counts())

# ============================================================
# LATIN-ONLY TEXT
# ============================================================

print("\n" + "=" * 70)
print("3. LATIN-ONLY TEXT")
print("=" * 70)

latin_only_mask = text.apply(
    lambda x: bool(x.strip()) and bool(re.search(r"[A-Za-z]", x))
    and not bool(re.search(r"[\u0900-\u097F\u0C80-\u0CFF\u0B80-\u0BFF\u0C00-\u0C7F\u0D00-\u0D7F\u0A80-\u0AFF\u0A00-\u0A7F\u0980-\u09FF\u0600-\u06FF]", x))
)

print(f"Latin-only records: {latin_only_mask.sum():,}")
print(
    f"Percentage: {latin_only_mask.mean() * 100:.2f}%"
)

# ============================================================
# MIXED LATIN + INDIAN SCRIPT
# ============================================================

print("\n" + "=" * 70)
print("4. MIXED LATIN + INDIAN SCRIPT")
print("=" * 70)

mixed_indian_mask = text.apply(
    lambda x: bool(re.search(r"[A-Za-z]", x))
    and bool(
        re.search(
            r"[\u0900-\u097F\u0C80-\u0CFF\u0B80-\u0BFF\u0C00-\u0C7F\u0D00-\u0D7F\u0A80-\u0AFF\u0A00-\u0A7F\u0980-\u09FF]",
            x
        )
    )
)

print(f"Mixed Latin + Indian-script records: {mixed_indian_mask.sum():,}")
print(
    f"Percentage: {mixed_indian_mask.mean() * 100:.2f}%"
)

# ============================================================
# SOURCE-WISE SCRIPT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("5. SOURCE-WISE WRITING SYSTEM DISTRIBUTION")
print("=" * 70)

source_script_table = pd.crosstab(
    df["source"],
    script_labels
)

print(source_script_table.to_string())

# ============================================================
# SAMPLE NON-LATIN TEXT
# ============================================================

print("\n" + "=" * 70)
print("6. SAMPLE NON-LATIN TEXT")
print("=" * 70)

non_latin_mask = text.apply(
    lambda x: bool(
        re.search(
            r"[\u0900-\u097F\u0C80-\u0CFF\u0B80-\u0BFF\u0C00-\u0C7F\u0D00-\u0D7F\u0A80-\u0AFF\u0A00-\u0A7F\u0980-\u09FF]",
            x
        )
    )
)

non_latin_examples = text[non_latin_mask].drop_duplicates().head(20)

for i, value in enumerate(non_latin_examples, start=1):
    print(f"\n--- Example {i} ---")
    print(value[:500])

# ============================================================
# SAMPLE MIXED TEXT
# ============================================================

print("\n" + "=" * 70)
print("7. SAMPLE MIXED-SCRIPT TEXT")
print("=" * 70)

mixed_examples = text[mixed_indian_mask].drop_duplicates().head(20)

if len(mixed_examples) == 0:
    print("No mixed Latin + Indian-script examples found.")
else:
    for i, value in enumerate(mixed_examples, start=1):
        print(f"\n--- Example {i} ---")
        print(value[:500])

# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("LANGUAGE / SCRIPT ANALYSIS COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print("These results identify writing systems, not exact languages.")
print("No files were modified.")