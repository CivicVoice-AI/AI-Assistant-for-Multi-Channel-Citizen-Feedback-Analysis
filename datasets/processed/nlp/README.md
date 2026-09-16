# NLP Processed Dataset

## Overview

This directory contains the NLP-preprocessed version of the CiviVoice master feedback dataset.

The preprocessing was performed on:

`datasets/masterd/master_feedback.csv`

The resulting dataset is:

`datasets/processed/nlp/master_feedback_nlp.csv`

The original master dataset was **not modified or overwritten**.

---

## Input Dataset

**File:**

`datasets/masterd/master_feedback.csv`

**Records:**

205,168

**Columns:**

* `feedback_id`
* `source`
* `timestamp`
* `text`
* `organization`
* `loc`
* `urgency`

The original `text` column is preserved exactly in the processed dataset.

---

## Output Dataset

**File:**

`master_feedback_nlp.csv`

The processed dataset contains the original 7 columns plus:

* `text_clean`

Therefore, the processed dataset contains **8 columns**.

### `text_clean`

`text_clean` is a conservatively normalized version of the original `text` field intended for downstream NLP analysis.

---

## Preprocessing Performed

The preprocessing script is:

`scripts/nlp/preprocess_text.py`

The following transformations were applied to create `text_clean`.

### 1. Unicode normalization

Unicode text was normalized using NFKC normalization.

This helps standardize equivalent Unicode representations while preserving multilingual text.

### 2. HTML entity decoding

HTML entities were decoded.

Examples:

```text
&amp;  → &
&#39;  → '
&quot; → "
```

This was particularly relevant because HTML entities were present in a portion of the dataset.

### 3. URL normalization

URLs were replaced with:

```text
<URL>
```

The presence of a URL is retained without keeping the exact URL string.

### 4. Email normalization

Email addresses were replaced with:

```text
<EMAIL>
```

This retains the information that an email address was present while preventing the exact address from becoming part of the NLP text.

### 5. Mention normalization

Social-media style mentions were replaced with:

```text
<MENTION>
```

### 6. Hashtag normalization

The `#` symbol was removed while retaining the hashtag content.

Example:

```text
#WaterProblem
```

becomes:

```text
WaterProblem
```

This preserves potentially useful topic information.

### 7. Structural separator removal

Long separator sequences such as:

```text
----------------
________________
================
```

were removed.

These were treated as formatting noise rather than meaningful language.

### 8. CPGRAMS hierarchy marker normalization

The `>>` hierarchy marker was replaced with whitespace.

For example:

```text
Ministry >> Department >> Issue
```

becomes approximately:

```text
Ministry Department Issue
```

The words themselves are preserved because CPGRAMS structural/category information may be useful for later NLP tasks.

### 9. Repeated punctuation normalization

Excessive repeated punctuation was reduced conservatively.

For example:

```text
!!!!!!
```

becomes:

```text
!!
```

Punctuation was **not removed completely**, because punctuation can contain useful sentiment or emotion signals.

### 10. Repeated Latin-character normalization

Very long repetitions of Latin characters were reduced.

For example:

```text
goooooood
```

may become:

```text
good
```

This transformation was deliberately restricted to Latin characters so that Indian-language Unicode text would not be damaged.

### 11. Whitespace normalization

Multiple whitespace characters were normalized to a single space and leading/trailing whitespace was removed.

### 12. Case normalization

The cleaned text was converted using `casefold()`.

For example:

```text
THIS IS A COMPLAINT
```

becomes:

```text
this is a complaint
```

The original capitalization remains available in the original `text` column.

---

## What Was Deliberately NOT Removed

The preprocessing was intentionally conservative because CiviVoice contains multilingual, code-mixed and source-specific feedback.

The following were **not automatically removed**:

* Stopwords
* Numbers
* All punctuation
* Non-English scripts
* Hindi text
* Kannada text
* Tamil text
* Telugu text
* Malayalam text
* Gujarati text
* Other Unicode text
* Short feedback
* CPGRAMS category information
* Department/ministry information
* Source-specific structural information that may be useful for later analysis

This allows later NLP phases to decide what information is useful for each task rather than permanently deleting potentially valuable information during preprocessing.

---

## Dataset Validation

After preprocessing:

* Original rows: **205,168**
* Processed rows: **205,168**
* Row-count check: **PASS**
* Original empty text records: **0**
* Duplicate `feedback_id` values: **0**
* `text_clean` empty records: **1**

The single empty `text_clean` record represents approximately **0.0005%** of the dataset and is being left as an edge case rather than modifying the preprocessing pipeline solely to handle one record.

---

## Data Preservation Principle

The CiviVoice NLP pipeline follows this principle:

> **Never destroy the original feedback text when creating NLP-ready data.**

The original:

```text
text
```

field is preserved, while:

```text
text_clean
```

provides a normalized version for NLP experiments.

This makes it possible to compare model behavior against the original feedback and revise preprocessing decisions later without needing to reconstruct the raw text.

---

## Current NLP Pipeline Position

This dataset represents the **NLP Phase 1 — Text Exploration & Basic Preprocessing** stage.

Completed activities include:

1. Text length exploration
2. Word-count exploration
3. Short-text analysis
4. Writing-system/script analysis
5. Noise analysis
6. Boilerplate and structural-pattern analysis
7. Conservative text preprocessing
8. Creation of the NLP-ready dataset
9. Validation of the processed dataset

Further NLP processing will be performed in subsequent phases rather than aggressively modifying this dataset at this stage.

---

## Reproducibility

The processed dataset was generated using:

`scripts/nlp/preprocess_text.py`

To regenerate the dataset from the master dataset:

```cmd
python scripts\nlp\preprocess_text.py
```

The master dataset should remain unchanged.

---

## Files

```text
datasets/
└── processed/
    └── nlp/
        ├── README.md
        └── master_feedback_nlp.csv
```

Related preprocessing script:

```text
scripts/
└── nlp/
    └── preprocess_text.py
```
