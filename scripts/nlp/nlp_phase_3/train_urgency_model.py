import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "datasets/processed/nlp/master_feedback_nlp.csv"
)

OUTPUT_DIR = Path(
    "datasets/processed/nlp/urgency_analysis"
)

MODEL_DIR = Path(
    "backend/models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CIVICVOICE - URGENCY CLASSIFICATION BASELINE")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Dataset shape: {df.shape}")


# ============================================================
# SELECT LABELLED DATA
# ============================================================

print("\n" + "=" * 70)
print("1. SELECTING LABELLED RECORDS")
print("=" * 70)

labelled_df = df[
    df["urgency"].notna()
].copy()

labelled_df["text_clean"] = (
    labelled_df["text_clean"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove empty text
labelled_df = labelled_df[
    labelled_df["text_clean"] != ""
].copy()

print(
    f"\nLabelled records available: "
    f"{len(labelled_df):,}"
)

print("\nUrgency distribution:")

print(
    labelled_df["urgency"].value_counts()
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = labelled_df["text_clean"]
y = labelled_df["urgency"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("2. TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"\nTraining records: {len(X_train):,}"
)

print(
    f"Testing records: {len(X_test):,}"
)


# ============================================================
# TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("3. TF-IDF FEATURE EXTRACTION")
print("=" * 70)

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print(
    f"\nTraining TF-IDF shape: "
    f"{X_train_tfidf.shape}"
)

print(
    f"Testing TF-IDF shape: "
    f"{X_test_tfidf.shape}"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("4. TRAINING LOGISTIC REGRESSION")
print("=" * 70)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print("\nModel training completed.")


# ============================================================
# SAVE MODEL + VECTORIZER
# ============================================================

print("\n" + "=" * 70)
print("5. SAVING MODEL")
print("=" * 70)

model_file = MODEL_DIR / "urgency_model.pkl"
vectorizer_file = MODEL_DIR / "urgency_vectorizer.pkl"

joblib.dump(
    model,
    model_file
)

joblib.dump(
    vectorizer,
    vectorizer_file
)

print(
    f"\nUrgency model saved to:\n{model_file}"
)

print(
    f"\nTF-IDF vectorizer saved to:\n{vectorizer_file}"
)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("6. MAKING PREDICTIONS")
print("=" * 70)

y_pred = model.predict(
    X_test_tfidf
)

print("\nPredictions completed.")


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("7. MODEL EVALUATION")
print("=" * 70)

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print(
    f"\nAccuracy:    {accuracy:.4f}"
)

print(
    f"Macro F1:    {macro_f1:.4f}"
)

print(
    f"Weighted F1: {weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("8. CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("9. CONFUSION MATRIX")
print("=" * 70)

classes = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{c}"
        for c in classes
    ],
    columns=[
        f"Predicted_{c}"
        for c in classes
    ]
)

print("\n")

print(
    cm_df.to_string()
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_file = (
    OUTPUT_DIR /
    "urgency_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_file
)

print(
    f"\nConfusion matrix saved to:\n{cm_file}"
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "text": X_test.values,
    "actual_urgency": y_test.values,
    "predicted_urgency": y_pred
})

prediction_file = (
    OUTPUT_DIR /
    "urgency_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print(
    f"\nTest predictions saved to:\n{prediction_file}"
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("10. FINAL VALIDATION")
print("=" * 70)

print(
    f"\nOriginal dataset rows: {len(df):,}"
)

print(
    f"Labelled records used: {len(labelled_df):,}"
)

print(
    f"Training records: {len(X_train):,}"
)

print(
    f"Testing records: {len(X_test):,}"
)

print(
    "\nOriginal master dataset was NOT modified."
)

print(
    "\nSaved reusable model files:"
)

print(
    f"✓ {model_file}"
)

print(
    f"✓ {vectorizer_file}"
)

print("\n" + "=" * 70)
print("URGENCY CLASSIFICATION COMPLETE")
print("=" * 70)