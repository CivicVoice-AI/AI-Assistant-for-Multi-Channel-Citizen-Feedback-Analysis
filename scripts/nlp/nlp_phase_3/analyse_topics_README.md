# NLP Phase 3 — Topic / Issue Discovery

## Objective

To discover major themes present in citizen feedback automatically using **TF-IDF and NMF topic modelling**.

## Script

```text
scripts/nlp/nlp_phase_3/analyze_topics_v2.py
```

## Method

```text
50,000-record sample
        ↓
Text filtering
        ↓
TF-IDF
        ↓
NMF
        ↓
10 Topics
        ↓
Top-word analysis
```

## Results

- Records used: **50,000**
- Records after text filtering: **47,339**
- TF-IDF features: **20,000**
- Topics generated: **10**

Some interpretable themes included:

| Topic | Identified Theme |
|---|---|
| Topic 2 | PF / UAN / Employment |
| Topic 3 | Complaints / Representations |
| Topic 8 | Income Tax / Refund / PAN |
| Topic 1 | Travel / Road-related terms |

Several topics contained government/administrative terminology and abbreviations, particularly from CPGRAMS data.

## Topic Distribution

| Topic | Records | % |
|---|---:|---:|
| 1 | 5,325 | 11.25% |
| 2 | 22,783 | 48.13% |
| 3 | 2,225 | 4.70% |
| 4 | 491 | 1.04% |
| 5 | 8,049 | 17.00% |
| 6 | 427 | 0.90% |
| 7 | 436 | 0.92% |
| 8 | 6,166 | 13.03% |
| 9 | 1,077 | 2.28% |
| 10 | 360 | 0.76% |

## Key Findings

- NMF successfully identified recurring themes from the feedback.
- TF-IDF + NMF works as an **exploratory baseline**.
- Some topics were dominated by administrative terms and abbreviations.
- This is expected because the dataset is large, multilingual and contains different types of government records.
- More advanced multilingual semantic topic modelling can be considered in future work.

## Output Files

```text
datasets/processed/nlp/topic_analysis/
├── topic_words_v2.csv
├── topic_distribution_v2.csv
└── topic_assignments_v2.csv
```

## Conclusion

Topic modelling provided an initial view of recurring issues in CIVICVOICE feedback. The results can be used for exploratory analysis, while advanced multilingual models can provide more accurate semantic topic identification in future work.