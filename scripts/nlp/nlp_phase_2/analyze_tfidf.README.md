# NLP Phase 2 — TF-IDF Analysis

## 1. Objective

The objective of this step was to identify which words are more informative in citizen feedback rather than only looking at words that occur frequently.

For this, **TF-IDF (Term Frequency–Inverse Document Frequency)** was applied to the preprocessed `text_clean` column.

TF-IDF gives a higher weight to a word when:
- It occurs frequently in a particular document.
- It does not occur in too many other documents.

Therefore, TF-IDF helps identify terms that are more useful for later NLP tasks such as topic discovery, classification, and semantic analysis.

---

## 2. Script Used

`scripts/nlp/analyze_tfidf.py`

### Input Dataset

`datasets/processed/nlp/master_feedback_nlp.csv`

The dataset contains:

- **205,168 documents**
- Original master feedback fields
- `text_clean` generated during NLP Phase 1

---

## 3. TF-IDF Configuration

The analysis used a sparse TF-IDF matrix to handle the large dataset efficiently.

Configuration:

- Maximum features: **30,000**
- Minimum document frequency: **5**
- Maximum document frequency: **95%**
- Documents: **205,168**
- TF-IDF matrix: **205,168 × 30,000**
- Non-zero values: **13,915,485**

No rows were deleted or modified.

---

## 4. What Was Analyzed

The analysis included:

1. Overall TF-IDF terms
2. Source-wise important terms
3. Individual document TF-IDF terms
4. TF-IDF vocabulary size
5. Comparison of vocabulary patterns across sources

Sources analyzed:

- CPGRAMS
- BBMP_Bengaluru
- Email
- Survey
- Twitter

---

## 5. Overall Results

The highest total TF-IDF weights included terms such as:

- `क`
- `र`
- `pg`
- `the`
- `न`
- `स`
- `म`
- `of`
- `to`
- `and`
- `i`
- `x`
- `is`
- `in`

### Observation

Some high-weight terms are individual characters from Indian scripts and masked values such as `x`.

This shows that TF-IDF by itself does not automatically understand whether a term is semantically meaningful.

It also shows that the dataset contains:

- Multilingual text
- Administrative/structured text
- Masked or redacted information
- Common English words
- Source-specific vocabulary

Therefore, the TF-IDF results need to be interpreted along with the earlier token, noise, and boilerplate analysis.

---

## 6. Source-Wise Results

### BBMP_Bengaluru

Important terms included:

- `ward`
- `nagar`
- `area`
- `problem`
- `residents`
- `complaint`
- `issue`
- `water`
- `action`
- `location`
- `immediate`

### Observation

BBMP feedback contains vocabulary related to:

- Local areas
- Wards
- Resident complaints
- Problems
- Water and local services
- Required actions

---

### CPGRAMS

Important terms included:

- `pg`
- `of`
- `the`
- `x`
- `my`
- `no`
- `not`
- Hindi-script terms
- Administrative terms

### Observation

CPGRAMS has a large amount of multilingual and administrative content.

The presence of `x` and individual Indian-script characters indicates that some high-frequency patterns are related to masking, formatting, or tokenization rather than meaningful topics.

---

### Email

Important terms included:

- `public`
- `location`
- `service`
- `citizen`
- `issue`
- `concerned`
- `regarding`
- `place`
- `madam`
- `regards`
- `update`

### Observation

Email feedback has a more formal communication style and contains service-related and request-related vocabulary.

---

### Survey

Important terms included:

- `near`
- `needs`
- `issue`
- `concerned`
- `better`
- `attention`
- `residents`
- `please`
- `update`
- `resolved`
- `public`
- `authority`
- `water`

### Observation

Survey feedback contains repeated citizen request and service-related terminology.

---

### Twitter

Important terms included:

- `train`
- `traffic`
- `delhi`
- `pnr`
- `help`
- `plz`
- `jam`
- `coach`
- `station`
- `police`
- `road`

### Observation

Twitter contains shorter and more informal feedback, with terms related to transportation, traffic, help requests, stations, and roads.

---

## 7. Document-Level Analysis

TF-IDF was also calculated for individual feedback records.

Example:

A Twitter document produced important terms such as:

`prevent`, `minor`, `driving`, `crimes`, `strong`, `future`

Another document produced:

`learn`, `cars`, `handle`, `kids`, `jail`, `should`

Another produced:

`hatred`, `incidence`, `useless`, `spreading`, `arrest`

### Why This Is Useful

Instead of only asking:

> "What words occur most often in the entire dataset?"

we can ask:

> "Which words are particularly informative in this specific feedback?"

This can be useful for later:

- Topic discovery
- Complaint categorization
- Text classification
- Similarity analysis
- Search and retrieval

---

## 8. Important Findings

The TF-IDF analysis showed that different sources have different vocabulary patterns.

| Source | Main Vocabulary Pattern |
|---|---|
| CPGRAMS | Multilingual and administrative |
| BBMP | Local problems, areas, residents, services |
| Email | Formal citizen/service communication |
| Survey | Requests, issues, authorities, services |
| Twitter | Short, informal, transportation and public complaints |

The analysis also showed that **high TF-IDF weight does not always mean high semantic importance**.

Some terms are influenced by:

- Masked values
- Administrative templates
- Multilingual tokenization
- Common words
- Structured information

Therefore, TF-IDF will be treated as an **exploratory feature extraction method**, not as the final understanding of citizen issues.

---

## 9. Why TF-IDF Is Important for the Project

TF-IDF gives us a numerical representation of text that can later be used as input for machine learning models.

For example:

```text
Citizen Feedback
       ↓
Text Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Numerical Features
       ↓
Classification / Topic Analysis