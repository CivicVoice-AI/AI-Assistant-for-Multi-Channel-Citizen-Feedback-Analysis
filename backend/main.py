from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.database.db_service import get_connection
from backend.services.nlp_service import analyze_text

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CIVICVOICE API",
    description="Multi-Channel Citizen Feedback Analysis",
    version="1.0.0",
)

# Enable CORS for the local frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://civicvoicee.netlify.app",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    source: str = Field(default="API", max_length=50)
    location: Optional[str] = Field(default=None, max_length=150)
    organization: Optional[str] = Field(default=None, max_length=255)


@app.get("/")
def root():
    return {
        "message": "CIVICVOICE API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()

        return {
            "status": "healthy",
            "database": "connected",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {exc}",
        ) from exc

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()


@app.post("/analyze")
def analyze_feedback(request: FeedbackRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Feedback text cannot be empty.",
        )

    try:
        return analyze_text(text)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feedback analysis failed: {exc}",
        ) from exc


@app.get("/feedback")
def get_feedback(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    source: Optional[str] = Query(default=None),
    location: Optional[str] = Query(default=None),
    urgency: Optional[str] = Query(default=None),
):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
            SELECT
                f.feedback_id,
                s.source_name AS source,
                l.location_name AS location,
                f.feedback_timestamp,
                f.feedback_text,
                f.organization,
                u.urgency_label AS urgency
            FROM feedback AS f
            JOIN sources AS s
                ON s.source_id = f.source_id
            LEFT JOIN locations AS l
                ON l.location_id = f.location_id
            LEFT JOIN feedback_urgency AS u
                ON u.feedback_id = f.feedback_id
            WHERE 1 = 1
        """

        params = []

        # Filter by source, e.g. BBMP, Survey, Twitter, or Email.
        if source and source.strip() and source.strip().lower() != "all sources":
            query += " AND LOWER(s.source_name) LIKE LOWER(%s)"
            params.append(f"%{source.strip()}%")

        # Filter by location, e.g. Bangalore or Bengaluru.
        if location and location.strip() and location.strip().lower() not in (
            "all locations", "all"
        ):
            query += " AND LOWER(COALESCE(l.location_name, '')) LIKE LOWER(%s)"
            params.append(f"%{location.strip()}%")

        # Filter by urgency.
        if urgency and urgency.strip() and urgency.strip().lower() not in (
            "all levels", "all"
        ):
            query += " AND LOWER(COALESCE(u.urgency_label, '')) = LOWER(%s)"
            params.append(urgency.strip())

        # Count all matching records before applying pagination.
        count_query = f"SELECT COUNT(*) AS total FROM ({query}) AS filtered_feedback"
        cursor.execute(count_query, tuple(params))
        total = cursor.fetchone()["total"]

        query += """
            ORDER BY f.feedback_timestamp DESC, f.feedback_id
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        for row in rows:
            timestamp = row.get("feedback_timestamp")
            if isinstance(timestamp, datetime):
                row["feedback_timestamp"] = timestamp.isoformat()

        return {
            "count": len(rows),
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": rows,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve feedback: {exc}",
        ) from exc

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()



@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Feedback text cannot be empty.",
        )

    connection = None
    cursor = None

    try:
        analysis = analyze_text(text)
        connection = get_connection()
        cursor = connection.cursor()

        source_name = request.source.strip()

        cursor.execute(
            "SELECT source_id FROM sources WHERE source_name = %s",
            (source_name,),
        )
        source_row = cursor.fetchone()

        if source_row is None:
            cursor.execute(
                "INSERT INTO sources (source_name) VALUES (%s)",
                (source_name,),
            )
            source_id = cursor.lastrowid
        else:
            source_id = source_row[0]

        location_id = None

        if request.location and request.location.strip():
            location_name = request.location.strip()

            cursor.execute(
                "SELECT location_id FROM locations WHERE location_name = %s",
                (location_name,),
            )
            location_row = cursor.fetchone()

            if location_row is None:
                cursor.execute(
                    "INSERT INTO locations (location_name) VALUES (%s)",
                    (location_name,),
                )
                location_id = cursor.lastrowid
            else:
                location_id = location_row[0]

        feedback_id = (
            f"API-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        )

        cursor.execute(
            """
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
            """,
            (
                feedback_id,
                source_id,
                location_id,
                datetime.now(),
                text,
                request.organization,
                analysis["text_clean"],
            ),
        )

        cursor.execute(
            """
            INSERT INTO feedback_urgency (feedback_id, urgency_label)
            VALUES (%s, %s)
            """,
            (feedback_id, analysis["urgency"]),
        )

        connection.commit()

        return {
            "message": "Feedback analyzed and saved successfully",
            "feedback_id": feedback_id,
            "analysis": analysis,
        }

    except Exception as exc:
        if connection is not None and connection.is_connected():
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Could not save feedback: {exc}",
        ) from exc

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()

# ==========================================
# CIVICVOICE DASHBOARD STATISTICS API
# ==========================================

@app.get("/dashboard/stats")
def get_dashboard_stats():
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        # Total feedback records
        cursor.execute("""
            SELECT COUNT(*) AS total_feedback
            FROM feedback
        """)
        total_feedback = cursor.fetchone()["total_feedback"]

        # Number of distinct feedback sources
        cursor.execute("""
            SELECT COUNT(DISTINCT source_id) AS total_sources
            FROM feedback
        """)
        total_sources = cursor.fetchone()["total_sources"]

        # Number of high and critical urgency records
        cursor.execute("""
            SELECT COUNT(*) AS high_priority
            FROM feedback_urgency
            WHERE LOWER(urgency_label) IN ('high', 'critical')
        """)
        high_priority = cursor.fetchone()["high_priority"]

        # Number of distinct locations with feedback
        cursor.execute("""
            SELECT COUNT(DISTINCT location_id) AS total_locations
            FROM feedback
            WHERE location_id IS NOT NULL
        """)
        total_locations = cursor.fetchone()["total_locations"]

        # Feedback grouped by source
        cursor.execute("""
            SELECT
                s.source_name AS source,
                COUNT(f.feedback_id) AS total
            FROM sources s
            LEFT JOIN feedback f
                ON s.source_id = f.source_id
            GROUP BY s.source_id, s.source_name
            ORDER BY total DESC
        """)
        source_stats = cursor.fetchall()

        # Feedback grouped by urgency
        cursor.execute("""
            SELECT
                urgency_label AS urgency,
                COUNT(*) AS total
            FROM feedback_urgency
            GROUP BY urgency_label
        """)
        urgency_stats = cursor.fetchall()

        # Feedback grouped by date
        cursor.execute("""
            SELECT
                DATE(feedback_timestamp) AS feedback_date,
                COUNT(*) AS total
            FROM feedback
            WHERE feedback_timestamp IS NOT NULL
            GROUP BY DATE(feedback_timestamp)
            ORDER BY feedback_date DESC
            LIMIT 7
        """)
        trend_stats = cursor.fetchall()

        # Convert dates to strings for JSON serialization.
        for item in trend_stats:
            item["feedback_date"] = item["feedback_date"].isoformat()

        return {
            "total_feedback": total_feedback,
            "total_sources": total_sources,
            "high_priority": high_priority,
            "total_locations": total_locations,
            "source_stats": source_stats,
            "urgency_stats": urgency_stats,
            "trend_stats": list(reversed(trend_stats))
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load dashboard statistics: {exc}"
        ) from exc

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()