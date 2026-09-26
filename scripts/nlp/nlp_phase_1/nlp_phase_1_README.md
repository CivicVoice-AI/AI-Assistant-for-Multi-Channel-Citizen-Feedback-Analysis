NLP Phase 1 --- Text Exploration, Language Analysis & Preprocessing

Objective

The objective of Phase 1 was to understand the raw citizen-feedback
text, identify its major characteristics and noise patterns, and create
a cleaned NLP-ready dataset without changing the original master
dataset.

Scripts

explore_text.py

analyze_language.py

analyze_noise.py

analyze_boilerplate.py

preprocess_text.py

validate_nlp_dataset.py

Input

datasets/processed/nlp/master_feedback_nlp.csv was created as the
NLP-ready dataset after preprocessing.

The original master dataset contains 205,168 feedback records.

1. Text Exploration

We analyzed:

Number of records

Text length

Word count

Short texts

Long texts

Source-wise text characteristics

Key Results

Records: 205,168

Empty original texts: 0

Mean original text length: 729.10 characters

Median original text length: 415 characters

Mean word count: 123.08

Median word count: 68

Maximum word count: 1,039

The analysis showed that text length and writing style vary considerably
across sources.

2. Language / Script Analysis

The dataset contains multilingual and mixed-script feedback.

Major writing-system patterns included:

Latin only: 149,008 records

Devanagari + Latin: 32,966

Devanagari only: 16,794

Other Indian scripts and mixed-script records were also present.

This showed that the final NLP pipeline needs to preserve Unicode and
multilingual text rather than assuming English-only data.

3. Noise Analysis

We identified several types of text noise:

HTML entities

URLs

Email addresses

Mentions

Hashtags

Repeated punctuation

Repeated characters

All-caps text

Non-ASCII characters

Examples of observed patterns included HTML entities such as &amp;,
URLs, email addresses, hashtags and masked/redacted values.

4. Boilerplate Analysis

Repeated and structural text patterns were identified.

Examples included:

CPGRAMS administrative structures

Repeated separators

>> hierarchy markers

Subject

Name and Address

Department

Ministry

PF Office

UAN

PPO

Repeated text was not automatically deleted because some repetition
represents genuine source structure.

5. Text Preprocessing

Script:

preprocess_text.py

The following transformations were applied:

Unicode NFKC normalization

HTML entity decoding

URL replacement with <URL>

Email replacement with <EMAIL>

Mention replacement with <MENTION>

Hashtag symbol removal while retaining hashtag text

Removal of long structural separator lines

Normalization of CPGRAMS >>

Repeated punctuation normalization

Repeated Latin-character normalization

Whitespace normalization

Case folding

Deliberately Preserved

We did not remove:

Stopwords

Numbers

Indian-language scripts

Short feedback

CPGRAMS category information

Department / ministry information

This was done to avoid losing information before later NLP analysis.

6. Validation

Output dataset:

datasets/processed/nlp/master_feedback_nlp.csv

Final schema contains the original fields plus:

text_clean

Results:

Input rows: 205,168

Output rows: 205,168

Duplicate feedback IDs: 0

Original empty text: 0

Clean empty text: 1

Changed records: 184,072

Unchanged records: 21,096

The single empty cleaned record was retained and documented rather than
deleting the source record.

Phase 1 Conclusion

Phase 1 converted the raw citizen-feedback text into a validated
NLP-ready dataset while preserving the original information and
multilingual content.

Review Explanation

"In Phase 1, I first explored the text characteristics and identified
multilingual content, noise and repeated administrative structures. I
then applied conservative preprocessing such as Unicode normalization,
HTML decoding, URL and email masking, punctuation normalization and
whitespace cleanup. Finally, I validated the NLP dataset to make sure
that the row count and feedback IDs were preserved."