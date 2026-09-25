# NLP Phase 2 — Term Frequency & Document Frequency Analysis

## 1. Objective

The objective of this step was to understand how frequently words occur in the citizen feedback dataset and how widely each word is distributed across different feedback records.

Two measures were analyzed:

- **Term Frequency (TF):** How many times a term occurs in the complete dataset.
- **Document Frequency (DF):** How many different feedback documents contain the term.

This helps distinguish between very common terms, source-specific terms, and rare terms before moving to further NLP modeling.

---

## 2. Script Used

`scripts/nlp/analyze_term_frequency.py`

### Input

`datasets/processed/nlp/master_feedback_nlp.csv`

---

## 3. Dataset Summary

- Total documents: **205,168**
- Unique terms: **373,254**
- Total tokens: **26,012,616**

The analysis completed successfully without deleting or modifying any dataset records.

---

## 4. Top Terms by Frequency

The most frequent terms included:

| Term | Frequency |
|---|---:|
| the | 528,037 |
| of | 466,791 |
| to | 421,312 |
| and | 365,928 |
| x | 258,769 |
| in | 239,423 |
| i | 224,134 |
| है | 221,909 |
| के | 208,365 |
| is | 203,489 |
| my | 183,223 |
| for | 180,850 |
| no | 165,416 |
| में | 157,233 |
| की | 150,463 |

This confirms that the dataset contains both English and multilingual vocabulary.

The term `x` also occurs very frequently, which is consistent with the masked/redacted content observed during earlier NLP analysis.

---

## 5. Document Frequency

Document frequency measures how many separate feedback records contain a term.

Some of the highest-coverage terms were:

| Term | Documents | Coverage |
|---|---:|---:|
| of | 91,833 | 44.76% |
| and | 90,990 | 44.35% |
| to | 87,302 | 42.55% |
| the | 85,951 | 41.89% |
| in | 75,551 | 36.82% |
| is | 72,874 | 35.52% |
| no | 67,477 | 32.89% |
| i | 67,062 | 32.69% |
| for | 65,917 | 32.13% |
| not | 63,434 | 30.92% |

No term appeared in at least 50% of all documents.

---

## 6. Rare Terms

The analysis found:

**212,952 terms appearing in exactly one document.**

Other low-frequency terms were:

- 50,787 terms appeared in exactly 2 documents
- 22,458 terms appeared in exactly 3 documents
- 13,118 terms appeared in exactly 4 documents
- 8,827 terms appeared in exactly 5 documents

This demonstrates a large **long-tail vocabulary**, where many terms occur only in a small number of feedback records.

Rare terms can represent names, locations, identifiers, abbreviations, spelling variations, or genuinely uncommon topics.

Therefore, rare terms should not automatically be treated as noise.

---

## 7. Source-Wise Analysis

### BBMP_Bengaluru

Important terms included:

- `issue`
- `ward`
- `area`
- `complaint`
- `problem`
- `there`
- `nagar`
- `action`
- `residents`

For example:

- `issue` → 25.96% document coverage
- `ward` → 25.82%
- `area` → 24.32%
- `complaint` → 24.31%
- `problem` → 23.24%

This indicates vocabulary related to local areas, complaints, residents, problems, and required action.

---

### CPGRAMS

The most widely distributed terms were mostly common English and administrative-language terms such as:

- `of`
- `and`
- `to`
- `the`
- `in`
- `is`
- `no`
- `not`
- `i`
- `my`

For example:

- `of` → 49.86%
- `and` → 48.13%
- `to` → 44.61%
- `the` → 42.30%

This reflects the large size and diverse nature of the CPGRAMS dataset.

---

### Email

Several terms appeared in all 1,200 email records:

- `madam`
- `in`
- `location`
- `i`
- `citizen`
- `regarding`
- `sir`
- `dear`
- `issue`
- `service`
- `public`
- `regards`
- `concerned`

This indicates a highly repetitive and formal communication structure in the email dataset.

---

### Survey

Frequently distributed terms included:

- `the` → 84.85%
- `near` → 60.89%
- `issue` → 47.93%
- `needs` → 38.50%
- `concerned` → 31.09%
- `please` → 27.98%

Hindi terms such as `पर` and `कृपया` were also present.

This confirms that the survey dataset contains both English and Indian-language content.

---

### Twitter

Twitter showed a more informal and topic-specific vocabulary.

Important terms included:

- `train`
- `traffic`
- `delhi`
- `sir`
- `station`
- `help`

For example:

- `train` → 10.43%
- `traffic` → 9.07%
- `delhi` → 7.95%

This is consistent with short, informal public feedback and transportation-related complaints.

---

## 8. Selected Term Analysis

Specific terms were checked to compare their total frequency with their document coverage.

| Term | Frequency | Documents | Coverage |
|---|---:|---:|---:|
| x | 258,769 | 40,850 | 19.91% |
| pg | 19,536 | 18,274 | 8.91% |
| the | 528,037 | 85,951 | 41.89% |
| of | 466,791 | 91,833 | 44.76% |
| to | 421,312 | 87,302 | 42.55% |
| and | 365,928 | 90,990 | 44.35% |
| problem | 12,144 | 9,686 | 4.72% |
| complaint | 25,296 | 17,329 | 8.45% |
| issue | 33,715 | 26,277 | 12.81% |
| water | 8,664 | 4,930 | 2.40% |
| traffic | 3,190 | 2,212 | 1.08% |
| help | 17,773 | 13,692 | 6.67% |

This comparison is useful because a term can occur many times but still be concentrated in a smaller number of documents.

---

## 9. Important Findings

The analysis showed three major patterns:

### 1. Common vocabulary

Words such as `the`, `of`, `to`, and `and` have very high frequency and high document coverage.

These are common language terms and are less useful for identifying specific citizen issues.

### 2. Source-specific vocabulary

Different sources contain different vocabulary patterns.

For example:

- BBMP → `ward`, `area`, `complaint`, `problem`, `residents`
- CPGRAMS → administrative and multilingual vocabulary
- Email → formal communication terms
- Survey → request and service-related terms
- Twitter → transportation and informal complaint terms

### 3. Long-tail vocabulary

A very large number of terms occur in only a few documents.

This includes names, locations, abbreviations, identifiers, spelling variations, and uncommon words.

Therefore, low document frequency does not automatically mean that a term is useless.

---

## 10. Why This Analysis Is Important

This analysis helps us understand the vocabulary distribution before applying machine-learning models.

It also helps explain why simply removing all rare or frequent words would not be appropriate.

For example:

```text
Very common terms
        ↓
Less useful for identifying specific topics

Source-specific terms
        ↓
Potentially useful for understanding source characteristics

Rare terms
        ↓
May contain names, locations, identifiers or uncommon issues