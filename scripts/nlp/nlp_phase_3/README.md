# NLP Phase 3 — Exploratory NLP & Citizen Feedback Intelligence

## Objective

To extract useful NLP signals from the cleaned CIVICVOICE feedback dataset, including **sentiment, urgency, and major topics/issues**.

---

## 1. Sentiment Analysis

**Method:** VADER

**Dataset:** 205,168 records

### Results

| Sentiment | Records | Percentage |
|---|---:|---:|
| Positive | 75,413 | 36.76% |
| Neutral | 71,420 | 34.81% |
| Negative | 58,335 | 28.43% |

### Key Finding

VADER was used as a **baseline exploratory sentiment model**. Since the CIVICVOICE dataset is multilingual and VADER is primarily English-oriented, a multilingual transformer model can be evaluated in future work.

---

## 2. Urgency Analysis

The existing `urgency` field was analyzed before building a classification model.

- **Total records:** 205,168
- **Labelled records:** 24,517 (11.95%)
- **Unlabelled records:** 180,651 (88.05%)

Urgency labels were available only for:

- BBMP Bengaluru
- Survey

### Urgency Distribution

| Urgency | Records |
|---|---:|
| Low | 10,757 |
| Medium | 6,950 |
| High | 4,895 |
| Critical | 1,915 |

Missing urgency values were **not fabricated or assigned**.

---

## 3. Urgency Classification

### Method

```text
TF-IDF → Logistic Regression → Urgency Prediction
```

Only the **24,517 labelled records** were used.

### Dataset Split

- **Training:** 19,613 records
- **Testing:** 4,904 records
- **TF-IDF features:** 12,330

### Results

| Metric | Score |
|---|---:|
| Accuracy | 54.65% |
| Macro F1 | 49.07% |
| Weighted F1 | 55.98% |

### Class-wise F1

| Class | F1-score |
|---|---:|
| Critical | 0.41 |
| High | 0.36 |
| Low | 0.76 |
| Medium | 0.43 |

### Key Finding

The model performed best for **Low urgency**, while **High and Medium** urgency were more difficult to distinguish.

This model is treated as a **baseline**, not the final urgency model.

---

## 4. Topic / Issue Discovery

### Method

```text
TF-IDF → NMF Topic Modeling
```

A sample of the dataset was used for practical processing.

- **Records sampled:** 50,000
- **Records after filtering:** 47,339
- **TF-IDF features:** 20,000
- **Topics generated:** 10

### Identified Themes

Some recurring themes included:

- PF / UAN / Employment
- Complaints / Representations
- Income Tax / Refund / PAN
- Travel / Road-related terms

Some topics contained administrative terminology and abbreviations because the dataset contains multilingual government feedback from different sources.

Therefore, NMF is treated as an **exploratory topic-discovery baseline**.

---

## 5. Phase 3 Summary

```text
Cleaned Citizen Feedback