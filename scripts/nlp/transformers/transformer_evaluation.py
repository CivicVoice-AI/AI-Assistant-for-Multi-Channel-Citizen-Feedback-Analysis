import pandas as pd
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

# Human-labeled evaluation dataset
INPUT_FILE = Path(
    "datasets/processed/nlp/transformer_evaluation/"
    "sentiment_human_label_set_labeled.csv"
)

OUTPUT_DIR = Path(
    "datasets/processed/nlp/transformer_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Multilingual sentiment model
MODEL_NAME = (
    "cardiffnlp/twitter-xlm-roberta-base-sentiment"
)


# ============================================================
# LOAD HUMAN-LABELED DATASET
# ============================================================

print("=" * 70)
print("CIVICVOICE - TRANSFORMER SENTIMENT EVALUATION")
print("=" * 70)

print("\nLoading human-labeled evaluation dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Human-labeled evaluation file not found:\n"
        f"{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(
    f"Dataset rows: {len(df):,}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "feedback_id",
    "source",
    "text_clean",
    "human_sentiment"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# CLEAN TEXT
# ============================================================

df["text_clean"] = (
    df["text_clean"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    df["text_clean"] != ""
].copy()


# ============================================================
# CLEAN HUMAN LABELS
# ============================================================

df["human_sentiment"] = (
    df["human_sentiment"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

valid_labels = {
    "positive",
    "negative",
    "neutral"
}

invalid_labels = set(
    df["human_sentiment"]
) - valid_labels

if invalid_labels:
    raise ValueError(
        "Invalid human sentiment labels found: "
        + ", ".join(sorted(invalid_labels))
    )

df = df[
    df["human_sentiment"].isin(valid_labels)
].copy()

print(
    f"Records with valid text and labels: "
    f"{len(df):,}"
)


# ============================================================
# HUMAN LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("HUMAN-LABELED SENTIMENT DISTRIBUTION")
print("=" * 70)

print(
    df["human_sentiment"].value_counts()
)


# ============================================================
# INITIALIZE VADER
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING VADER")
print("=" * 70)

vader = SentimentIntensityAnalyzer()

print(
    "VADER initialized successfully."
)


# ============================================================
# INITIALIZE TRANSFORMER
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING MULTILINGUAL TRANSFORMER")
print("=" * 70)

print(
    f"\nModel: {MODEL_NAME}"
)

print(
    "\nThe model may need to be downloaded "
    "from Hugging Face the first time."
)

transformer = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME
)

print(
    "\nTransformer initialized successfully."
)


# ============================================================
# VADER FUNCTION
# ============================================================

def get_vader_sentiment(text):

    scores = vader.polarity_scores(
        text
    )

    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "positive"

    elif compound <= -0.05:
        sentiment = "negative"

    else:
        sentiment = "neutral"

    return sentiment, compound


# ============================================================
# TRANSFORMER FUNCTION
# ============================================================

def get_transformer_sentiment(text):

    result = transformer(
        text,
        truncation=True,
        max_length=512
    )[0]

    label = result["label"]
    score = result["score"]

    # CardiffNLP Twitter-XLM-R labels:
    # LABEL_0 = Negative
    # LABEL_1 = Neutral
    # LABEL_2 = Positive

    label_mapping = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }

    sentiment = label_mapping.get(
        label,
        label.lower()
    )

    return sentiment, score


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING HUMAN vs TRANSFORMER EVALUATION")
print("=" * 70)

results = []

for index, row in df.iterrows():

    text = row["text_clean"]

    try:

        # VADER
        vader_sentiment, vader_score = (
            get_vader_sentiment(text)
        )

        # Transformer
        transformer_sentiment, transformer_score = (
            get_transformer_sentiment(text)
        )

        results.append({

            "feedback_id": row["feedback_id"],

            "source": row["source"],

            "text": text,

            "human_sentiment": (
                row["human_sentiment"]
            ),

            "vader_sentiment": (
                vader_sentiment
            ),

            "vader_compound": (
                vader_score
            ),

            "transformer_sentiment": (
                transformer_sentiment
            ),

            "transformer_confidence": (
                transformer_score
            )
        })

    except Exception as e:

        print(
            f"\nError processing record "
            f"{index}: {e}"
        )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

print(
    f"\nSuccessfully processed: "
    f"{len(results_df):,} records"
)


if len(results_df) == 0:

    raise RuntimeError(
        "No records were successfully processed."
    )


# ============================================================
# HUMAN vs TRANSFORMER METRICS
# ============================================================

y_true = results_df[
    "human_sentiment"
]

y_pred = results_df[
    "transformer_sentiment"
]


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    labels=[
        "negative",
        "neutral",
        "positive"
    ],
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    labels=[
        "negative",
        "neutral",
        "positive"
    ],
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    labels=[
        "negative",
        "neutral",
        "positive"
    ],
    average="weighted",
    zero_division=0
)


# ============================================================
# PRINT METRICS
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMER EVALUATION METRICS")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        labels=[
            "negative",
            "neutral",
            "positive"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    "negative",
    "neutral",
    "positive"
]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

confusion_df = pd.DataFrame(
    cm,
    index=[
        "Actual Negative",
        "Actual Neutral",
        "Actual Positive"
    ],
    columns=[
        "Predicted Negative",
        "Predicted Neutral",
        "Predicted Positive"
    ]
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    confusion_df
)


# ============================================================
# TRANSFORMER DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMER SENTIMENT DISTRIBUTION")
print("=" * 70)

print(
    results_df[
        "transformer_sentiment"
    ].value_counts()
)


# ============================================================
# HUMAN vs TRANSFORMER AGREEMENT
# ============================================================

results_df["human_transformer_agreement"] = (
    results_df["human_sentiment"]
    ==
    results_df["transformer_sentiment"]
)

agreement_rate = (
    results_df[
        "human_transformer_agreement"
    ].mean()
    * 100
)

print("\n" + "=" * 70)
print("HUMAN / TRANSFORMER AGREEMENT")
print("=" * 70)

print(
    f"\nAgreement: "
    f"{agreement_rate:.2f}%"
)


# ============================================================
# DISAGREEMENT EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("HUMAN / TRANSFORMER DISAGREEMENT EXAMPLES")
print("=" * 70)

disagreements = results_df[
    results_df[
        "human_transformer_agreement"
    ] == False
].head(10)

for _, row in disagreements.iterrows():

    print("\n" + "-" * 70)

    print(
        "Source:",
        row["source"]
    )

    print(
        "Human:",
        row["human_sentiment"]
    )

    print(
        "Transformer:",
        row["transformer_sentiment"],
        "| Confidence:",
        round(
            row["transformer_confidence"],
            4
        )
    )

    print(
        "Text:",
        row["text"][:500]
    )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

results_file = (
    OUTPUT_DIR /
    "human_vs_transformer.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print(
    f"\nDetailed evaluation results saved to:\n"
    f"{results_file}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "metric": [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "agreement"
    ],

    "value": [
        accuracy,
        precision,
        recall,
        f1,
        agreement_rate / 100
    ]
})

metrics_file = (
    OUTPUT_DIR /
    "transformer_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)

print(
    f"Metrics saved to:\n"
    f"{metrics_file}"
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_file = (
    OUTPUT_DIR /
    "transformer_confusion_matrix.csv"
)

confusion_df.to_csv(
    confusion_file
)

print(
    f"Confusion matrix saved to:\n"
    f"{confusion_file}"
)

# ============================================================
# VADER EVALUATION METRICS
# ============================================================

vader_true = results_df[
    "human_sentiment"
]

vader_pred = results_df[
    "vader_sentiment"
]

vader_accuracy = accuracy_score(
    vader_true,
    vader_pred
)

vader_precision = precision_score(
    vader_true,
    vader_pred,
    labels=[
        "negative",
        "neutral",
        "positive"
    ],
    average="weighted",
    zero_division=0
)

vader_recall = recall_score(
    vader_true,
    vader_pred,
    labels=[
        "negative",
        "neutral",
        "positive"
    ],
    average="weighted",
    zero_division=0
)

vader_f1 = f1_score(
    vader_true,
    vader_pred,
    labels=[
        "negative",
        "neutral",
        "positive"
    ],
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("VADER EVALUATION METRICS")
print("=" * 70)

print(
    f"\nAccuracy : {vader_accuracy:.4f}"
)

print(
    f"Precision: {vader_precision:.4f}"
)

print(
    f"Recall   : {vader_recall:.4f}"
)

print(
    f"F1-score : {vader_f1:.4f}"
)


# ============================================================
# VADER CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("VADER CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        vader_true,
        vader_pred,
        labels=[
            "negative",
            "neutral",
            "positive"
        ],
        zero_division=0
    )
)


# ============================================================
# VADER CONFUSION MATRIX
# ============================================================

vader_cm = confusion_matrix(
    vader_true,
    vader_pred,
    labels=[
        "negative",
        "neutral",
        "positive"
    ]
)

vader_confusion_df = pd.DataFrame(
    vader_cm,
    index=[
        "Actual Negative",
        "Actual Neutral",
        "Actual Positive"
    ],
    columns=[
        "Predicted Negative",
        "Predicted Neutral",
        "Predicted Positive"
    ]
)

print("\n" + "=" * 70)
print("VADER CONFUSION MATRIX")
print("=" * 70)

print(
    vader_confusion_df
)

# ============================================================
# VADER vs TRANSFORMER AGREEMENT
# ============================================================

results_df["vader_transformer_agreement"] = (
    results_df["vader_sentiment"]
    ==
    results_df["transformer_sentiment"]
)

vader_transformer_agreement = (
    results_df[
        "vader_transformer_agreement"
    ].mean()
    * 100
)

print("\n" + "=" * 70)
print("VADER / TRANSFORMER AGREEMENT")
print("=" * 70)

print(
    f"\nAgreement: "
    f"{vader_transformer_agreement:.2f}%"
)
# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    f"\nHuman-labeled records: "
    f"{len(df):,}"
)

print(
    f"Successfully evaluated: "
    f"{len(results_df):,}"
)

print(
    "\nOriginal master dataset was NOT modified."
)

print(
    "\nEvaluation was performed against "
    "the supplied human_sentiment labels."
)

print("\n" + "=" * 70)
print("TRANSFORMER EVALUATION COMPLETE")
print("=" * 70)