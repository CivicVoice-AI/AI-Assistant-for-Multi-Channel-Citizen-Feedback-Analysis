import getpass
from pathlib import Path

import pandas as pd
import mysql.connector


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = Path(
    "datasets/processed/nlp/master_feedback_nlp.csv"
)

DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "civicvoice_db"
DB_USER = "civicvoice_user"

BATCH_SIZE = 1000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """Convert pandas missing values to None."""
    if pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    return value


def clean_text(value):
    """Convert text fields safely to strings/None."""
    if pd.isna(value):
        return None

    return str(value)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CIVICVOICE - LOAD FEEDBACK DATA")
print("=" * 70)

print(f"\nReading dataset:")
print(CSV_FILE)

df = pd.read_csv(
    CSV_FILE,
    low_memory=False
)

print(f"\nTotal records read: {len(df):,}")

# ------------------------------------------------------------
# Validate expected columns
# ------------------------------------------------------------

required_columns = [
    "feedback_id",
    "source",
    "timestamp",
    "text",
    "organization",
    "loc",
    "urgency",
    "text_clean"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns validated.")


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

missing_feedback_ids = df["feedback_id"].isna().sum()
duplicate_feedback_ids = df["feedback_id"].duplicated().sum()
missing_sources = df["source"].isna().sum()
missing_text = df["text"].isna().sum()

print("\nDATA VALIDATION")
print("-" * 70)
print(f"Missing feedback IDs:      {missing_feedback_ids:,}")
print(f"Duplicate feedback IDs:    {duplicate_feedback_ids:,}")
print(f"Missing sources:            {missing_sources:,}")
print(f"Missing feedback text:      {missing_text:,}")

if missing_feedback_ids > 0:
    raise ValueError("Feedback IDs cannot be missing.")

if duplicate_feedback_ids > 0:
    raise ValueError("Duplicate feedback IDs found.")

if missing_sources > 0:
    raise ValueError("Source values cannot be missing.")

if missing_text > 0:
    raise ValueError("Feedback text cannot be missing.")


# ============================================================
# TIMESTAMP PROCESSING
# ============================================================

print("\nProcessing timestamps...")

df["_timestamp_parsed"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    format="mixed"
)

invalid_timestamps = df["_timestamp_parsed"].isna().sum()

print(
    f"Invalid/unparseable timestamps: "
    f"{invalid_timestamps:,}"
)

if invalid_timestamps > 0:
    print(
        "These timestamps will be stored as NULL."
    )


# ============================================================
# CONNECT TO MYSQL
# ============================================================

print("\nConnecting to MySQL...")

password = getpass.getpass(
    "Enter CIVICVOICE database password: "
)

connection = mysql.connector.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=password,
    database=DB_NAME
)

cursor = connection.cursor()

print("MySQL connection successful.")


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

print("\nLoading source mappings...")

cursor.execute(
    "SELECT source_id, source_name FROM sources"
)

source_map = {
    name: source_id
    for source_id, name in cursor.fetchall()
}

print(
    f"Source mappings loaded: "
    f"{len(source_map)}"
)

print("\nLoading location mappings...")

cursor.execute(
    "SELECT location_id, location_name FROM locations"
)

location_map = {
    name: location_id
    for location_id, name in cursor.fetchall()
}

print(
    f"Location mappings loaded: "
    f"{len(location_map)}"
)


# ============================================================
# VALIDATE SOURCE VALUES
# ============================================================

dataset_sources = (
    df["source"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

unknown_sources = [
    source
    for source in dataset_sources
    if source not in source_map
]

if unknown_sources:
    raise ValueError(
        f"Unknown source values found: {unknown_sources}"
    )

print("\nAll dataset sources found in database.")


# ============================================================
# VALIDATE LOCATION VALUES
# ============================================================

dataset_locations = (
    df["loc"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

unknown_locations = [
    location
    for location in dataset_locations
    if location not in location_map
]

if unknown_locations:
    print(
        "\nWARNING: Unknown location values found:"
    )

    for location in unknown_locations[:20]:
        print(f"  - {location}")

    if len(unknown_locations) > 20:
        print(
            f"  ... and "
            f"{len(unknown_locations) - 20} more"
        )

    raise ValueError(
        "Dataset contains locations not present "
        "in the locations table."
    )

print("All dataset locations found in database.")


# ============================================================
# PREPARE FEEDBACK INSERT
# ============================================================

feedback_sql = """
INSERT INTO feedback (
    feedback_id,
    source_id,
    location_id,
    feedback_timestamp,
    feedback_text,
    organization,
    text_clean
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    source_id = VALUES(source_id),
    location_id = VALUES(location_id),
    feedback_timestamp = VALUES(feedback_timestamp),
    feedback_text = VALUES(feedback_text),
    organization = VALUES(organization),
    text_clean = VALUES(text_clean)
"""


# ============================================================
# LOAD FEEDBACK
# ============================================================

print("\n" + "=" * 70)
print("LOADING FEEDBACK")
print("=" * 70)

feedback_rows = []

for index, row in df.iterrows():

    feedback_id = str(row["feedback_id"]).strip()

    source_name = str(row["source"]).strip()
    source_id = source_map[source_name]

    location_value = clean_value(row["loc"])

    if location_value is None:
        location_id = None
    else:
        location_id = location_map[str(location_value).strip()]

    timestamp_value = row["_timestamp_parsed"]

    if pd.isna(timestamp_value):
        feedback_timestamp = None
    else:
        # MySQL DATETIME does not store timezone information.
        # Convert timezone-aware timestamps to UTC first.
        if timestamp_value.tzinfo is not None:
            timestamp_value = timestamp_value.tz_convert("UTC")
            timestamp_value = timestamp_value.tz_localize(None)

        feedback_timestamp = timestamp_value.to_pydatetime()

    feedback_text = clean_text(row["text"])

    organization = clean_value(row["organization"])

    text_clean = clean_text(row["text_clean"])

    feedback_rows.append(
        (
            feedback_id,
            source_id,
            location_id,
            feedback_timestamp,
            feedback_text,
            organization,
            text_clean
        )
    )

    if len(feedback_rows) >= BATCH_SIZE:

        cursor.executemany(
            feedback_sql,
            feedback_rows
        )

        connection.commit()

        processed = index + 1

        print(
            f"Feedback processed: "
            f"{processed:,} / {len(df):,}"
        )

        feedback_rows = []


# Insert remaining rows

if feedback_rows:

    cursor.executemany(
        feedback_sql,
        feedback_rows
    )

    connection.commit()

    print(
        f"Feedback processed: "
        f"{len(df):,} / {len(df):,}"
    )


# ============================================================
# LOAD URGENCY DATA
# ============================================================

urgency_sql = """
INSERT INTO feedback_urgency (
    feedback_id,
    urgency_label
)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE
    urgency_label = VALUES(urgency_label)
"""

print("\n" + "=" * 70)
print("LOADING URGENCY LABELS")
print("=" * 70)

urgency_rows = []

urgency_count = 0

for _, row in df.iterrows():

    urgency = clean_value(row["urgency"])

    if urgency is None:
        continue

    urgency_rows.append(
        (
            str(row["feedback_id"]).strip(),
            str(urgency).strip()
        )
    )

    urgency_count += 1

    if len(urgency_rows) >= BATCH_SIZE:

        cursor.executemany(
            urgency_sql,
            urgency_rows
        )

        connection.commit()

        urgency_rows = []


if urgency_rows:

    cursor.executemany(
        urgency_sql,
        urgency_rows
    )

    connection.commit()


# ============================================================
# DATABASE VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATABASE VALIDATION")
print("=" * 70)

cursor.execute(
    "SELECT COUNT(*) FROM feedback"
)

feedback_count = cursor.fetchone()[0]

cursor.execute(
    "SELECT COUNT(*) FROM feedback_urgency"
)

urgency_db_count = cursor.fetchone()[0]

cursor.execute(
    "SELECT COUNT(*) "
    "FROM feedback "
    "WHERE location_id IS NULL"
)

missing_location_count = cursor.fetchone()[0]

print(
    f"Feedback records in database: "
    f"{feedback_count:,}"
)

print(
    f"Urgency records in database: "
    f"{urgency_db_count:,}"
)

print(
    f"Feedback records with NULL location_id: "
    f"{missing_location_count:,}"
)

print(
    f"Urgency records expected from CSV: "
    f"{urgency_count:,}"
)


# ============================================================
# FINAL CHECKS
# ============================================================

if feedback_count != len(df):
    raise ValueError(
        "Feedback record count does not match CSV."
    )

if urgency_db_count != urgency_count:
    raise ValueError(
        "Urgency record count does not match CSV."
    )

print("\nAll database validation checks passed.")


# ============================================================
# CLOSE CONNECTION
# ============================================================

cursor.close()
connection.close()

print("\nMySQL connection closed.")

print("\n" + "=" * 70)
print("FEEDBACK DATA LOADING COMPLETED SUCCESSFULLY")
print("=" * 70)
