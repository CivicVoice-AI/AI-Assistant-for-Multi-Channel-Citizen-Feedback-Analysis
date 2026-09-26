import os
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"datasets\processed\nlp\master_feedback_nlp.csv"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("BASELINE SENTIMENT ANALYSIS")
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
# VALIDATE COLUMNS
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
# INITIALIZE VADER
# ============================================================

print("\nInitializing VADER sentiment analyzer...")

analyzer = SentimentIntensityAnalyzer()

print("VADER initialized successfully.")


# ============================================================
# SENTIMENT FUNCTION
# ============================================================

def analyze_sentiment(text):

    if pd.isna(text) or str(text).strip() == "":
        return 0.0, 0.0, 0.0, 0.0, "neutral"

    text = str(text)

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    # VADER standard thresholds
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return (
        scores["pos"],
        scores["neg"],
        scores["neu"],
        compound,
        sentiment
    )


# ============================================================
# RUN SENTIMENT ANALYSIS
# ============================================================

print("\nRunning sentiment analysis...")

results = df["text_clean"].apply(analyze_sentiment)

df["sentiment_positive"] = results.apply(lambda x: x[0])
df["sentiment_negative"] = results.apply(lambda x: x[1])
df["sentiment_neutral"] = results.apply(lambda x: x[2])
df["sentiment_compound"] = results.apply(lambda x: x[3])
df["sentiment"] = results.apply(lambda x: x[4])


# ============================================================
# OVERALL SENTIMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("OVERALL SENTIMENT DISTRIBUTION")
print("=" * 70)

sentiment_counts = df["sentiment"].value_counts()

sentiment_percentages = (
    df["sentiment"]
    .value_counts(normalize=True)
    .mul(100)
)

for sentiment in ["positive", "neutral", "negative"]:

    count = sentiment_counts.get(sentiment, 0)
    percentage = sentiment_percentages.get(sentiment, 0)

    print(
        f"{sentiment:<12} "
        f"{count:>10,} "
        f"({percentage:>6.2f}%)"
    )


# ============================================================
# SENTIMENT SCORE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT SCORE SUMMARY")
print("=" * 70)

print(
    df["sentiment_compound"]
    .describe()
    .to_string()
)


# ============================================================
# SOURCE-WISE SENTIMENT
# ============================================================

print("\n" + "=" * 70)
print("SOURCE-WISE SENTIMENT DISTRIBUTION")
print("=" * 70)

source_sentiment = pd.crosstab(
    df["source"],
    df["sentiment"]
)

# Ensure consistent column order
for col in ["positive", "neutral", "negative"]:
    if col not in source_sentiment.columns:
        source_sentiment[col] = 0

source_sentiment = source_sentiment[
    ["positive", "neutral", "negative"]
]

print("\nCounts:")
print(source_sentiment.to_string())


print("\nPercentages:")

source_percentages = (
    source_sentiment
    .div(source_sentiment.sum(axis=1), axis=0)
    .mul(100)
    .round(2)
)

print(source_percentages.to_string())


# ============================================================
# AVERAGE COMPOUND SCORE BY SOURCE
# ============================================================

print("\n" + "=" * 70)
print("AVERAGE SENTIMENT SCORE BY SOURCE")
print("=" * 70)

average_source_sentiment = (
    df.groupby("source")["sentiment_compound"]
    .agg(["mean", "median", "min", "max"])
    .round(4)
)

print(average_source_sentiment.to_string())


# ============================================================
# MOST POSITIVE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("MOST POSITIVE SAMPLE RECORDS")
print("=" * 70)

positive_samples = (
    df.nlargest(5, "sentiment_compound")
    [["feedback_id", "source", "sentiment_compound", "sentiment", "text_clean"]]
)

for _, row in positive_samples.iterrows():

    print("\nFeedback ID:", row["feedback_id"])
    print("Source:", row["source"])
    print("Score:", round(row["sentiment_compound"], 4))
    print("Sentiment:", row["sentiment"])
    print("Text:", str(row["text_clean"])[:500])


# ============================================================
# MOST NEGATIVE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("MOST NEGATIVE SAMPLE RECORDS")
print("=" * 70)

negative_samples = (
    df.nsmallest(5, "sentiment_compound")
    [["feedback_id", "source", "sentiment_compound", "sentiment", "text_clean"]]
)

for _, row in negative_samples.iterrows():

    print("\nFeedback ID:", row["feedback_id"])
    print("Source:", row["source"])
    print("Score:", round(row["sentiment_compound"], 4))
    print("Sentiment:", row["sentiment"])
    print("Text:", str(row["text_clean"])[:500])


# ============================================================
# CHECK EMPTY SENTIMENT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    "Sentiment values missing:",
    df["sentiment"].isna().sum()
)

print(
    "Compound scores missing:",
    df["sentiment_compound"].isna().sum()
)

print(
    "Original row count:",
    len(df)
)

print(
    "Sentiment row count:",
    len(df)
)


# ============================================================
# IMPORTANT NOTE
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT NOTE")
print("=" * 70)

print(
    "VADER is an English-oriented rule-based baseline."
)

print(
    "Because this dataset contains multilingual Indian-language "
    "feedback, these results should be treated as exploratory "
    "and not as final multilingual sentiment labels."
)

print(
    "A multilingual transformer model can be evaluated later "
    "for the final sentiment classification stage."
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("No rows were deleted.")
print("Original dataset was not overwritten.")
print("Sentiment analysis was exploratory only.")