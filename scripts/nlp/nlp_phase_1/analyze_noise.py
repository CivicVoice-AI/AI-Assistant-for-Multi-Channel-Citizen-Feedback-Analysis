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
print("CiviVoice — NLP Phase 1: Noise Analysis")
print("=" * 70)

print("\nLoading master dataset...")

df = pd.read_csv(
    MASTER_FILE,
    low_memory=False
)

print("Dataset loaded successfully.")

text = df["text"].fillna("").astype(str)

# ============================================================
# HELPER
# ============================================================

def count_matches(pattern):
    return text.str.contains(
        pattern,
        regex=True,
        na=False
    ).sum()

# ============================================================
# 1. HTML ENTITIES
# ============================================================

print("\n" + "=" * 70)
print("1. HTML ENTITY ANALYSIS")
print("=" * 70)

html_entity_pattern = r"&(?:[A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9A-Fa-f]+);"

html_entity_count = count_matches(html_entity_pattern)

print(f"Records containing HTML entities: {html_entity_count:,}")
print(
    f"Percentage: {html_entity_count / len(df) * 100:.2f}%"
)

# Most common HTML entities

entity_counter = Counter()

for value in text:
    matches = re.findall(html_entity_pattern, value)
    entity_counter.update(matches)

print("\nTop HTML entities:")

for entity, count in entity_counter.most_common(20):
    print(f"  {entity}: {count:,}")

# ============================================================
# 2. URLS
# ============================================================

print("\n" + "=" * 70)
print("2. URL ANALYSIS")
print("=" * 70)

url_pattern = r"(?:https?://|www\.)\S+"

url_count = count_matches(url_pattern)

print(f"Records containing URLs: {url_count:,}")
print(
    f"Percentage: {url_count / len(df) * 100:.2f}%"
)

# ============================================================
# 3. EMAIL ADDRESSES
# ============================================================

print("\n" + "=" * 70)
print("3. EMAIL ADDRESS ANALYSIS")
print("=" * 70)

email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

email_count = count_matches(email_pattern)

print(f"Records containing email addresses: {email_count:,}")
print(
    f"Percentage: {email_count / len(df) * 100:.2f}%"
)

# ============================================================
# 4. @MENTIONS
# ============================================================

print("\n" + "=" * 70)
print("4. @MENTION ANALYSIS")
print("=" * 70)

mention_pattern = r"(?<!\w)@[A-Za-z0-9_]+"

mention_count = count_matches(mention_pattern)

print(f"Records containing @mentions: {mention_count:,}")
print(
    f"Percentage: {mention_count / len(df) * 100:.2f}%"
)

# ============================================================
# 5. HASHTAGS
# ============================================================

print("\n" + "=" * 70)
print("5. HASHTAG ANALYSIS")
print("=" * 70)

hashtag_pattern = r"(?<!\w)#[^\s#]+"

hashtag_count = count_matches(hashtag_pattern)

print(f"Records containing hashtags: {hashtag_count:,}")
print(
    f"Percentage: {hashtag_count / len(df) * 100:.2f}%"
)

# ============================================================
# 6. HTML TAGS
# ============================================================

print("\n" + "=" * 70)
print("6. HTML TAG ANALYSIS")
print("=" * 70)

html_tag_pattern = r"<[^>]+>"

html_tag_count = count_matches(html_tag_pattern)

print(f"Records containing HTML tags: {html_tag_count:,}")
print(
    f"Percentage: {html_tag_count / len(df) * 100:.2f}%"
)

# ============================================================
# 7. REPEATED WHITESPACE
# ============================================================

print("\n" + "=" * 70)
print("7. WHITESPACE ANALYSIS")
print("=" * 70)

multiple_space_pattern = r"\s{2,}"

multiple_space_count = count_matches(multiple_space_pattern)

print(
    f"Records containing repeated whitespace: "
    f"{multiple_space_count:,}"
)

print(
    f"Percentage: "
    f"{multiple_space_count / len(df) * 100:.2f}%"
)

# ============================================================
# 8. REPEATED PUNCTUATION
# ============================================================

print("\n" + "=" * 70)
print("8. REPEATED PUNCTUATION ANALYSIS")
print("=" * 70)

repeated_punctuation_pattern = r"([!?.,])\1{2,}"

repeated_punctuation_count = count_matches(
    repeated_punctuation_pattern
)

print(
    f"Records containing repeated punctuation: "
    f"{repeated_punctuation_count:,}"
)

print(
    f"Percentage: "
    f"{repeated_punctuation_count / len(df) * 100:.2f}%"
)

# ============================================================
# 9. ALL-CAPS TEXT
# ============================================================

print("\n" + "=" * 70)
print("9. ALL-CAPS ANALYSIS")
print("=" * 70)

def contains_all_caps(value):
    words = re.findall(r"\b[A-Za-z]{3,}\b", value)

    if not words:
        return False

    return any(word.isupper() for word in words)


all_caps_mask = text.apply(contains_all_caps)

print(
    f"Records containing all-caps words: "
    f"{all_caps_mask.sum():,}"
)

print(
    f"Percentage: "
    f"{all_caps_mask.mean() * 100:.2f}%"
)

# ============================================================
# 10. REPEATED CHARACTERS
# ============================================================

print("\n" + "=" * 70)
print("10. REPEATED CHARACTER ANALYSIS")
print("=" * 70)

repeated_character_pattern = r"(.)\1{3,}"

repeated_character_count = count_matches(
    repeated_character_pattern
)

print(
    f"Records containing 4+ repeated characters: "
    f"{repeated_character_count:,}"
)

print(
    f"Percentage: "
    f"{repeated_character_count / len(df) * 100:.2f}%"
)

# ============================================================
# 11. NON-ASCII / UNICODE
# ============================================================

print("\n" + "=" * 70)
print("11. UNICODE / NON-ASCII ANALYSIS")
print("=" * 70)

non_ascii_mask = text.apply(
    lambda x: any(ord(char) > 127 for char in x)
)

print(
    f"Records containing non-ASCII characters: "
    f"{non_ascii_mask.sum():,}"
)

print(
    f"Percentage: "
    f"{non_ascii_mask.mean() * 100:.2f}%"
)

# ============================================================
# 12. SOURCE-WISE NOISE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("12. SOURCE-WISE NOISE SUMMARY")
print("=" * 70)

noise_masks = {
    "html_entities": text.str.contains(
        html_entity_pattern,
        regex=True,
        na=False
    ),
    "urls": text.str.contains(
        url_pattern,
        regex=True,
        na=False
    ),
    "email_addresses": text.str.contains(
        email_pattern,
        regex=True,
        na=False
    ),
    "mentions": text.str.contains(
        mention_pattern,
        regex=True,
        na=False
    ),
    "hashtags": text.str.contains(
        hashtag_pattern,
        regex=True,
        na=False
    ),
    "html_tags": text.str.contains(
        html_tag_pattern,
        regex=True,
        na=False
    ),
    "repeated_whitespace": text.str.contains(
        multiple_space_pattern,
        regex=True,
        na=False
    ),
    "repeated_punctuation": text.str.contains(
        repeated_punctuation_pattern,
        regex=True,
        na=False
    ),
    "all_caps": all_caps_mask,
    "repeated_characters": text.str.contains(
        repeated_character_pattern,
        regex=True,
        na=False
    ),
    "non_ascii": non_ascii_mask
}

source_noise = pd.DataFrame(
    {
        name: mask.groupby(df["source"]).sum()
        for name, mask in noise_masks.items()
    }
)

source_totals = df["source"].value_counts()

print("\nRecord counts:")
print(source_noise.to_string())

print("\nPercentages within each source:")

source_noise_percent = source_noise.div(
    source_totals,
    axis=0
) * 100

print(source_noise_percent.round(2).to_string())

# ============================================================
# 13. EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("13. NOISE EXAMPLES")
print("=" * 70)

example_patterns = {
    "HTML entities": html_entity_pattern,
    "URLs": url_pattern,
    "Mentions": mention_pattern,
    "Hashtags": hashtag_pattern,
    "HTML tags": html_tag_pattern,
    "Repeated punctuation": repeated_punctuation_pattern,
    "Repeated characters": repeated_character_pattern
}

for label, pattern in example_patterns.items():

    mask = text.str.contains(
        pattern,
        regex=True,
        na=False
    )

    examples = text[mask].drop_duplicates().head(5)

    print(f"\n--- {label} ---")

    if len(examples) == 0:
        print("No examples found.")
    else:
        for value in examples:
            print(f"  {value[:300]}")

# ============================================================
# END
# ============================================================

print("\n" + "=" * 70)
print("NOISE ANALYSIS COMPLETE")
print("=" * 70)

print("\nNo files were modified.")
print("No preprocessing was performed.")