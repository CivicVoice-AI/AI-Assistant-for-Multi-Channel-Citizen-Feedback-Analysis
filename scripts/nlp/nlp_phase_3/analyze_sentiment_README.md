# NLP Phase 3 — Baseline Sentiment Analysis

## Objective

The objective of this step was to perform an initial sentiment analysis of citizen feedback and identify whether the feedback was **positive, neutral, or negative**.

A lightweight **VADER sentiment analyzer** was used as a baseline.

> Note: VADER is primarily English-oriented. Since our dataset contains multilingual Indian-language feedback, these results are treated as **exploratory baseline results**, not final sentiment labels.

---

## Script

`scripts/nlp/analyze_sentiment.py`

### Input

`datasets/processed/nlp/master_feedback_nlp.csv`

### Dataset

- Total feedback records: **205,168**
- Sentiment labels generated: **205,168**
- Missing sentiment values: **0**

---

## Overall Sentiment Results

| Sentiment | Records | Percentage |
|---|---:|---:|
| Positive | 75,413 | 36.76% |
| Neutral | 71,420 | 34.81% |
| Negative | 58,335 | 28.43% |

The dataset contains all three sentiment categories, with positive feedback being the largest category in this baseline analysis.

---

## Source-Wise Sentiment

| Source | Positive | Neutral | Negative |
|---|---:|---:|---:|
| BBMP_Bengaluru | 9.49% | 8.41% | 82.10% |
| CPGRAMS | 37.58% | 38.31% | 24.10% |
| Email | 64.58% | 3.92% | 31.50% |
| Survey | 59.03% | 18.63% | 22.34% |
| Twitter | 35.38% | 25.60% | 39.03% |

### Observation

The sentiment distribution differs considerably across sources.

For example:

- **BBMP_Bengaluru** has a high proportion of negative feedback.
- **Email** has a high proportion of positive classifications.
- **Survey** also contains a larger proportion of positive classifications.
- **Twitter** has a relatively mixed distribution.
- **CPGRAMS** contains a large proportion of both positive and neutral classifications.

These differences show that sentiment patterns can vary depending on the source of citizen feedback.

---

## Average Sentiment Score

| Source | Mean Compound Score |
|---|---:|
| BBMP_Bengaluru | -0.4582 |
| CPGRAMS | 0.0895 |
| Email | 0.2356 |
| Survey | 0.1861 |
| Twitter | -0.0188 |

The overall compound score had:

- Mean: **0.0544**
- Median: **0.0000**
- Minimum: **-0.9998**
- Maximum: **0.9997**

---

## Important Observation

The analysis also demonstrated a limitation of using an English-oriented sentiment model on multilingual citizen feedback.

Some records received very strong positive or negative scores even when the complete context was more complex.

Therefore, these results should be treated as a **baseline exploratory analysis**.

For the final system, a multilingual transformer-based sentiment model can be evaluated and compared with this baseline.

---

## Validation

- Rows processed: **205,168**
- Missing sentiment values: **0**
- Missing compound scores: **0**
- Rows deleted: **0**
- Original dataset overwritten: **No**

The sentiment analysis was exploratory and did not modify the original NLP dataset.

---

## Why This Step Is Useful

Sentiment analysis can help the final CiviVoice system understand the general emotional tone of citizen feedback.

For example:

```text
Citizen Feedback
       ↓
Text Preprocessing
       ↓
Sentiment Analysis
       ↓
Positive / Neutral / Negative
       ↓
Combine with Topic + Urgency
       ↓
Citizen Feedback Insights