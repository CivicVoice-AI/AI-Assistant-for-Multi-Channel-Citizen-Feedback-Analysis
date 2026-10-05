# NLP Phase 3 — Urgency Classification Baseline

## Objective

To build a baseline model that predicts citizen-feedback urgency using the existing labelled records.

## Script

```text
scripts/nlp/nlp_phase_3/train_urgency_model.py
```

## Method

```text
Labelled Feedback
      ↓
Text Cleaning
      ↓
TF-IDF
      ↓
Logistic Regression
      ↓
Urgency Prediction
```

- Labelled records: **24,517**
- Training records: **19,613**
- Testing records: **4,904**
- TF-IDF features: **12,330**

## Model Results

| Metric | Score |
|---|---:|
| Accuracy | **54.65%** |
| Macro F1 | **49.07%** |
| Weighted F1 | **55.98%** |

### Class-wise F1

| Urgency | F1-score |
|---|---:|
| Critical | 0.41 |
| High | 0.36 |
| Low | **0.76** |
| Medium | 0.43 |

## Key Findings

- The model performs best on **Low urgency** feedback.
- **High and Medium** urgency are harder to distinguish.
- Macro F1 of **49.07%** shows that performance across all classes is moderate.
- This model is treated as a **baseline**, not the final urgency model.
- The original master dataset was **not modified**.

## Output Files

```text
datasets/processed/nlp/urgency_analysis/
├── urgency_confusion_matrix.csv
└── urgency_test_predictions.csv
```

## Conclusion

TF-IDF + Logistic Regression provides a basic urgency-classification baseline, but the moderate performance indicates that a more advanced or multilingual model may be required for reliable urgency prediction.