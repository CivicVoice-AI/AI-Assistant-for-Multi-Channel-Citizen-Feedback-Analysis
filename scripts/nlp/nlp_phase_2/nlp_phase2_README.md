# NLP Phase 2 — Exploratory NLP & Feature Analysis

## Objective

The objective of Phase 2 was to understand the vocabulary, phrases and statistical importance of words in the cleaned citizen-feedback dataset.

No records were deleted or modified during these analyses.

## Scripts

- `analyze_tokens.py`
- `analyze_ngrams.py`
- `analyze_tfidf.py`
- `analyze_term_frequency.py`

## Input Dataset

`datasets/processed/nlp/master_feedback_nlp.csv`

Dataset size:

- **205,168 documents**
- **26,012,616 total tokens**
- **373,254 unique terms**

## 1. Token Analysis

Script:

`analyze_tokens.py`

A Unicode-aware tokenizer was used to support multilingual text.

### Key Findings

- Large vocabulary with **373,254 unique terms**
- English and Indian-language tokens were present
- Many rare and noisy tokens existed
- Source-specific vocabulary was observed
- CPGRAMS contained the largest and most diverse vocabulary

The analysis confirmed that simple English-only tokenization would not be sufficient.

## 2. N-Gram / Phrase Analysis

Script:

`analyze_ngrams.py`

We analyzed:

- Bigrams
- Trigrams
- Raw frequent phrases
- Filtered/informative phrases
- Source-wise phrases

### Examples

Common phrases included:

- `of india`
- `i have`
- `i am`
- `as per`
- `not provided`
- `regional office`
- `traffic jam`
- `please help`
- `immediate action`

### Observation

Some frequent phrases were caused by administrative templates or masked values.

Therefore, raw frequency alone cannot determine semantic importance.

## 3. TF-IDF Analysis

Script:

`analyze_tfidf.py`

TF-IDF was used to identify terms that are relatively informative for individual documents.

### Configuration

- Documents: **205,168**
- Maximum features: **30,000**
- Minimum document frequency: **5**
- Maximum document frequency: **95%**
- Non-zero values: **13,915,485**

The resulting sparse matrix had a shape of:

`205,168 × 30,000`

### Observation

High-weight terms included common English terms, Indian-language characters and masked values.

This demonstrated that TF-IDF is useful as a baseline feature representation, but it does not automatically understand semantic meaning.

## 4. Term Frequency & Document Frequency

Script:

`analyze_term_frequency.py`

We measured:

- Total term frequency
- Document frequency
- Document coverage
- Rare terms
- Source-wise term coverage

### Key Results

- Total terms: **373,254**
- Total tokens: **26,012,616**
- Terms appearing in exactly one document: **212,952**
- No term appeared in at least 50% of all documents

### Selected Terms

| Term | Frequency | Documents | Coverage |
|---|---:|---:|---:|
| `the` | 528,037 | 85,951 | 41.89% |
| `of` | 466,791 | 91,833 | 44.76% |
| `issue` | 33,715 | 26,277 | 12.81% |
| `complaint` | 25,296 | 17,329 | 8.45% |
| `water` | 8,664 | 4,930 | 2.40% |
| `traffic` | 3,190 | 2,212 | 1.08% |

## 5. Source-Level Findings

### BBMP_Bengaluru

Important vocabulary included:

- `issue`
- `ward`
- `area`
- `complaint`
- `problem`
- `residents`
- `action`

### CPGRAMS

Vocabulary was highly multilingual and administrative.

### Email

Vocabulary was formal and service-oriented, including:

- `citizen`
- `service`
- `issue`
- `regarding`
- `concerned`

### Survey

Common terms included:

- `issue`
- `needs`
- `concerned`
- `please`
- `authority`

### Twitter

More informal and transportation-related terms appeared, including:

- `train`
- `traffic`
- `delhi`
- `help`
- `station`

## 6. Important Findings

Phase 2 showed that:

1. The dataset is highly multilingual.
2. The vocabulary has a large long tail.
3. Different sources have different language patterns.
4. Frequency alone is not enough to determine importance.
5. TF-IDF provides a useful baseline for future machine-learning tasks.

## 7. Why Phase 2 Is Important

The Phase 2 analysis provides the vocabulary and statistical foundation for later NLP models.

The progression is:

```text
Clean Text
    ↓
Token Analysis
    ↓
N-Gram Analysis
    ↓
TF-IDF
    ↓
Term / Document Frequency
    ↓
NLP Features
    ↓
Classification / Topic Analysis