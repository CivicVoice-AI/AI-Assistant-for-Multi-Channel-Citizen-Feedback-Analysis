import os
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error


# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "civicvoice_user"),
    "database": os.getenv("DB_NAME", "civicvoice_db"),
    "connection_timeout": 10,
}


def get_connection():
    """Connect using credentials loaded from the local .env file."""
    password = os.getenv("DB_PASSWORD")

    if not password:
        raise RuntimeError(
            f"DB_PASSWORD is missing. Check your local .env file: {ENV_FILE}"
        )

    try:
        return mysql.connector.connect(
            **DB_CONFIG,
            password=password,
        )
    except Error as exc:
        raise RuntimeError(
            f"Could not connect to the CIVICVOICE database: {exc}"
        ) from exc


def test_connection():
    """Check database connectivity and record counts."""
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM feedback")
        feedback_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM feedback_urgency")
        urgency_count = cursor.fetchone()[0]

        return {
            "status": "connected",
            "database": DB_CONFIG["database"],
            "feedback_records": feedback_count,
            "urgency_records": urgency_count,
        }

    except Error as exc:
        raise RuntimeError(
            f"Database query failed: {exc}"
        ) from exc

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()