from pathlib import Path

import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

URGENCY_MODEL_FILE = MODEL_DIR / "urgency_model.pkl"
URGENCY_VECTORIZER_FILE = MODEL_DIR / "urgency_vectorizer.pkl"


# ============================================================
# LOAD MODELS
# ============================================================

if not URGENCY_MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Urgency model not found:\n{URGENCY_MODEL_FILE}"
    )

if not URGENCY_VECTORIZER_FILE.exists():
    raise FileNotFoundError(
        f"Urgency vectorizer not found:\n{URGENCY_VECTORIZER_FILE}"
    )


urgency_model = joblib.load(
    URGENCY_MODEL_FILE
)

urgency_vectorizer = joblib.load(
    URGENCY_VECTORIZER_FILE
)


# ============================================================
# SENTIMENT ANALYZER
# ============================================================

sentiment_analyzer = SentimentIntensityAnalyzer()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Basic text cleaning for application-time NLP.

    This does not replace the complete dataset
    preprocessing pipeline used during NLP analysis.
    """

    if text is None:
        return ""

    return str(text).strip().lower()


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment using the same VADER
    approach used in Phase 3.
    """

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return {
            "sentiment_positive": 0.0,
            "sentiment_negative": 0.0,
            "sentiment_neutral": 0.0,
            "sentiment_compound": 0.0,
            "sentiment": "neutral"
        }

    scores = sentiment_analyzer.polarity_scores(
        cleaned_text
    )

    compound = scores["compound"]

    # VADER standard thresholds
    if compound >= 0.05:
        sentiment = "positive"

    elif compound <= -0.05:
        sentiment = "negative"

    else:
        sentiment = "neutral"

    return {
        "sentiment_positive": scores["pos"],
        "sentiment_negative": scores["neg"],
        "sentiment_neutral": scores["neu"],
        "sentiment_compound": compound,
        "sentiment": sentiment
    }


# ============================================================
# URGENCY PREDICTION
# ============================================================

def predict_urgency(text: str) -> str:
    """
    Predict urgency using the trained
    TF-IDF + Logistic Regression model.
    """

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return "Low"

    text_vector = urgency_vectorizer.transform(
        [cleaned_text]
    )

    prediction = urgency_model.predict(
        text_vector
    )

    return str(prediction[0])


# ============================================================
# COMPLETE NLP ANALYSIS
# ============================================================

def analyze_text(text: str) -> dict:
    """
    Run the available CivicVoice NLP analysis
    on a single citizen complaint.
    """

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError(
            "Complaint text cannot be empty."
        )

    sentiment_result = analyze_sentiment(
        cleaned_text
    )

    urgency_result = predict_urgency(
        cleaned_text
    )

    return {
        "text": text,
        "text_clean": cleaned_text,
        **sentiment_result,
        "urgency": urgency_result
    }