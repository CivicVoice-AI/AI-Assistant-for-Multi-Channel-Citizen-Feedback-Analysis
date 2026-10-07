import getpass
import pandas as pd
import mysql.connector


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "datasets/processed/nlp/master_feedback_nlp.csv"

DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "civicvoice_db"
DB_USER = "civicvoice_user"


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

print("=" * 70)
print("CIVICVOICE - LOAD REFERENCE DATA")
print("=" * 70)

print("\nReading source and location data...")

df = pd.read_csv(
    CSV_FILE,
    usecols=["source", "loc"],
    low_memory=False
)

print(f"Total CSV records inspected: {len(df):,}")


# ============================================================
# PREPARE SOURCES
# ============================================================

sources = (
    df["source"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

sources = sorted(sources.tolist())


# ============================================================
# PREPARE LOCATIONS
# ============================================================

locations = (
    df["loc"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

locations = sorted(locations.tolist())


print(f"\nUnique sources found: {len(sources)}")

for source in sources:
    print(f"  - {source}")


print(f"\nUnique locations found: {len(locations)}")


# ============================================================
# CONNECT TO MYSQL
# ============================================================

print("\nConnecting to MySQL...")

password = getpass.getpass("Enter CIVICVOICE database password: ")

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
# INSERT SOURCES
# ============================================================

print("\nLoading sources...")

source_sql = """
INSERT INTO sources (source_name)
VALUES (%s)
ON DUPLICATE KEY UPDATE source_name = VALUES(source_name)
"""

for source in sources:
    cursor.execute(source_sql, (source,))

connection.commit()

print(f"Sources processed: {len(sources)}")


# ============================================================
# INSERT LOCATIONS
# ============================================================

print("\nLoading locations...")

location_sql = """
INSERT INTO locations (location_name)
VALUES (%s)
ON DUPLICATE KEY UPDATE location_name = VALUES(location_name)
"""

for location in locations:
    cursor.execute(location_sql, (location,))

connection.commit()

print(f"Locations processed: {len(locations)}")


# ============================================================
# VALIDATION
# ============================================================

cursor.execute("SELECT COUNT(*) FROM sources")
source_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM locations")
location_count = cursor.fetchone()[0]


print("\n" + "=" * 70)
print("REFERENCE DATA VALIDATION")
print("=" * 70)

print(f"Sources in database:   {source_count}")
print(f"Locations in database: {location_count}")


# ============================================================
# CLOSE CONNECTION
# ============================================================

cursor.close()
connection.close()

print("\nMySQL connection closed.")

print("\nReference data loading completed successfully.")
