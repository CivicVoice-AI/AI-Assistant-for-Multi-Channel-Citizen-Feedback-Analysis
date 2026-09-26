import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "datasets/processed/nlp/master_feedback_nlp.csv"
)

OUTPUT_DIR = Path(
    "datasets/processed/nlp/urgency_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CIVICVOICE - URGENCY ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")
df = pd.read_csv(INPUT_FILE)

print(f"Dataset loaded successfully.")
print(f"Shape: {df.shape}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["urgency", "source"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Required columns are missing:")
    print(missing_columns)
    raise SystemExit(1)


# ============================================================
# BASIC URGENCY INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. URGENCY COLUMN INFORMATION")
print("=" * 70)

print("\nUrgency data type:")
print(df["urgency"].dtype)

print("\nRaw urgency value counts:")
print(df["urgency"].value_counts(dropna=False))


# ============================================================
# NORMALIZE URGENCY VALUES FOR ANALYSIS ONLY
# ============================================================

# IMPORTANT:
# We are NOT modifying df["urgency"].
# A separate Series is created only for analysis.

urgency_analysis = (
    df["urgency"]
    .fillna("Not Available")
    .astype(str)
    .str.strip()
)

# Treat common missing-value strings as unavailable
missing_strings = [
    "",
    "nan",
    "NaN",
    "None",
    "none",
    "N/A",
    "NA",
    "Not Available"
]

urgency_analysis = urgency_analysis.replace(
    missing_strings,
    "Not Available"
)


# ============================================================
# OVERALL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("2. OVERALL URGENCY DISTRIBUTION")
print("=" * 70)

urgency_counts = urgency_analysis.value_counts()

urgency_percentages = (
    urgency_counts / len(df) * 100
).round(2)

overall_urgency = pd.DataFrame({
    "Urgency": urgency_counts.index,
    "Records": urgency_counts.values,
    "Percentage": urgency_percentages.values
})

print("\n")
print(overall_urgency.to_string(index=False))


# ============================================================
# LABELLED VS UNLABELLED
# ============================================================

print("\n" + "=" * 70)
print("3. LABELLED VS UNLABELLED URGENCY")
print("=" * 70)

labelled_mask = urgency_analysis != "Not Available"

labelled_count = labelled_mask.sum()
unlabelled_count = (~labelled_mask).sum()
total_count = len(df)

labelled_percentage = labelled_count / total_count * 100
unlabelled_percentage = unlabelled_count / total_count * 100

print(f"\nTotal records:       {total_count:,}")
print(
    f"Records with label:  {labelled_count:,} "
    f"({labelled_percentage:.2f}%)"
)
print(
    f"Records without label: {unlabelled_count:,} "
    f"({unlabelled_percentage:.2f}%)"
)


# ============================================================
# LABELED DATA DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. URGENCY DISTRIBUTION AMONG LABELLED RECORDS")
print("=" * 70)

labelled_urgency = urgency_analysis[labelled_mask]

labelled_counts = labelled_urgency.value_counts()

labelled_percentages = (
    labelled_counts / labelled_count * 100
).round(2)

labelled_distribution = pd.DataFrame({
    "Urgency": labelled_counts.index,
    "Records": labelled_counts.values,
    "Percentage_of_Labelled_Data": labelled_percentages.values
})

print("\n")
print(labelled_distribution.to_string(index=False))


# ============================================================
# SOURCE-WISE URGENCY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("5. SOURCE-WISE URGENCY DISTRIBUTION")
print("=" * 70)

source_urgency = pd.crosstab(
    df["source"],
    urgency_analysis
)

print("\n")
print(source_urgency.to_string())


# ============================================================
# SOURCE-WISE LABEL AVAILABILITY
# ============================================================

print("\n" + "=" * 70)
print("6. SOURCE-WISE URGENCY LABEL AVAILABILITY")
print("=" * 70)

source_label_stats = []

for source, group in df.groupby("source", dropna=False):

    source_urgency_values = urgency_analysis.loc[group.index]

    labelled = (
        source_urgency_values != "Not Available"
    ).sum()

    total = len(group)

    percentage = labelled / total * 100 if total > 0 else 0

    source_label_stats.append({
        "Source": source,
        "Total_Records": total,
        "Labelled_Records": labelled,
        "Unlabelled_Records": total - labelled,
        "Label_Availability_Percentage": round(
            percentage, 2
        )
    })

source_label_stats = pd.DataFrame(source_label_stats)

print("\n")
print(
    source_label_stats.to_string(index=False)
)


# ============================================================
# CLASS IMBALANCE
# ============================================================

print("\n" + "=" * 70)
print("7. CLASS IMBALANCE CHECK")
print("=" * 70)

if len(labelled_counts) > 0:

    largest_class = labelled_counts.max()
    smallest_class = labelled_counts.min()

    largest_class_name = labelled_counts.idxmax()
    smallest_class_name = labelled_counts.idxmin()

    imbalance_ratio = (
        largest_class / smallest_class
        if smallest_class > 0
        else float("inf")
    )

    print(
        f"\nLargest class: "
        f"{largest_class_name} ({largest_class:,})"
    )

    print(
        f"Smallest class: "
        f"{smallest_class_name} ({smallest_class:,})"
    )

    print(
        f"Class imbalance ratio: "
        f"{imbalance_ratio:.2f}:1"
    )


# ============================================================
# SAVE ANALYSIS RESULTS
# ============================================================

print("\n" + "=" * 70)
print("8. SAVING ANALYSIS RESULTS")
print("=" * 70)

overall_file = OUTPUT_DIR / "overall_urgency_distribution.csv"

labelled_file = OUTPUT_DIR / "labelled_urgency_distribution.csv"

source_file = OUTPUT_DIR / "source_urgency_distribution.csv"

availability_file = OUTPUT_DIR / "source_label_availability.csv"


overall_urgency.to_csv(
    overall_file,
    index=False
)

labelled_distribution.to_csv(
    labelled_file,
    index=False
)

source_urgency.to_csv(
    source_file
)

source_label_stats.to_csv(
    availability_file,
    index=False
)


print(f"\nSaved:")
print(f"- {overall_file}")
print(f"- {labelled_file}")
print(f"- {source_file}")
print(f"- {availability_file}")


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("9. VALIDATION")
print("=" * 70)

print(
    f"\nOriginal dataset rows: {len(df):,}"
)

print(
    f"Labelled urgency rows: {labelled_count:,}"
)

print(
    f"Unlabelled urgency rows: {unlabelled_count:,}"
)

print(
    f"Labelled + unlabelled: "
    f"{labelled_count + unlabelled_count:,}"
)

if labelled_count + unlabelled_count == total_count:
    print(
        "\nPASS: All records accounted for."
    )
else:
    print(
        "\nWARNING: Record count mismatch."
    )

print("\nOriginal dataset was NOT modified.")

print("\n" + "=" * 70)
print("URGENCY ANALYSIS COMPLETE")
print("=" * 70)