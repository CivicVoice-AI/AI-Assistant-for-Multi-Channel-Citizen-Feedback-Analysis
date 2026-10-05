# NLP Phase 3 — Urgency Analysis

## Objective
To analyze the existing urgency labels in the CIVICVOICE NLP dataset before building an urgency classification model.

## Script
```text
scripts/nlp/nlp_phase_3/analyze_urgency.py
```

## Input
```text
datasets/processed/nlp/master_feedback_nlp.csv
```

## Dataset
- Total records: **205,168**
- Records with urgency labels: **24,517 (11.95%)**
- Records without urgency labels: **180,651 (88.05%)**

## Urgency Distribution

| Urgency | Records | Percentage |
|---|---:|---:|
| Low | 10,757 | 43.88%* |
| Medium | 6,950 | 28.35%* |
| High | 4,895 | 19.97%* |
| Critical | 1,915 | 7.81%* |

\*Percentage is among labelled records.

## Source-wise Label Availability

| Source | Label Availability |
|---|---:|
| BBMP_Bengaluru | 100% |
| Survey | 100% |
| CPGRAMS | 0% |
| Email | 0% |
| Twitter | 0% |

## Key Findings
- Only **11.95%** of the dataset has existing urgency labels.
- Urgency labels are available only from **BBMP Bengaluru and Survey**.
- **Low** is the largest class, while **Critical** is the smallest.
- Class imbalance ratio: **5.62:1**.
- Missing urgency values were **not fabricated or assigned**.
- The original dataset was **not modified**.

## Output Files
```text
urgency_analysis/overall_urgency_distribution.csv
urgency_analysis/labelled_urgency_distribution.csv
urgency_analysis/source_urgency_distribution.csv
urgency_analysis/source_label_availability.csv
```

## Next Step
Use the **24,517 labelled records** to build a baseline urgency classifier using TF-IDF + Logistic Regression and evaluate it using F1-score, Macro F1, Weighted F1 and a confusion matrix.