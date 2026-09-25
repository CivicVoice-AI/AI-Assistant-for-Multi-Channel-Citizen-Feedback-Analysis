# NLP Phase 2 — N-Gram & Phrase Analysis

## Script
`scripts/nlp/analyze_ngrams.py`

## Input
`datasets/processed/nlp/master_feedback_nlp.csv`

## Purpose
This step analyzes frequently occurring word combinations (phrases) in the citizen feedback dataset without modifying or deleting any records.

## What Was Analyzed

- **Bigrams** — 2-word phrases, e.g. `traffic jam`
- **Trigrams** — 3-word phrases, e.g. `government of india`
- Raw and filtered frequent phrases
- Source-wise phrases for:
  - CPGRAMS
  - BBMP_Bengaluru
  - Email
  - Survey
  - Twitter

## Key Findings

- Raw bigrams were heavily influenced by masked/redacted values such as `x x`.
- Common phrases included `of india`, `i have`, `i am`, `as per`, `not provided`, and `regional office`.
- Frequent trigrams included `के संबंध में`, `निवेदन है कि`, `labour and employment`, `pf account no`, and `government of india`.
- **BBMP:** phrases were mainly related to local problems, areas, residents, and actions such as `problem is`, `same area`, and `immediate action`.
- **CPGRAMS:** phrases were dominated by administrative language, Hindi phrases, and masked values.
- **Email:** phrases were mainly formal and service-related, such as `issue in`, `service issue`, and `public service`.
- **Survey:** repeated request/complaint phrases appeared, such as `needs attention`, `concerned authority`, and `take action`.
- **Twitter:** short and informal phrases appeared, such as `train no`, `traffic jam`, `please help`, and `red light`.

## Important Observation

Raw phrase frequency alone does not indicate that a phrase is important. Some frequent phrases come from administrative templates, masked information, or repeated structures. Therefore, future NLP analysis should consider source, context, frequency, and semantic importance together.

