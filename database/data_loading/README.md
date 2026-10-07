# CIVICVOICE Database Data Loading

## Main Feedback Data

The master NLP dataset was loaded into the MySQL database:

`civicvoice_db`

Source:

`datasets/processed/nlp/master_feedback_nlp.csv`

### Loaded Data

| Table | Records |
|---|---:|
| sources | 5 |
| locations | 969 |
| feedback | 205,168 |
| feedback_urgency | 24,517 |

### Feedback ID Validation

- Total IDs: 205,168
- Unique IDs: 205,168
- Duplicate IDs: 0
- ID type: VARCHAR(50)

Feedback IDs contain both numeric and alphanumeric values.

Example:

`MORLY/E/2023/0000001`

### Location Handling

Records with missing `loc` values were stored with:

`location_id = NULL`

Missing locations: 56,646

The value:

`No District/Not Known`

was preserved as an actual location.

### Urgency

Urgency-labelled records:

24,517

Unlabelled records remain without a row in `feedback_urgency`.

### Validation

Database validation confirmed:

- All 205,168 feedback records loaded.
- All 24,517 urgency records loaded.
- No orphan source references.
- No orphan location references.
- No orphan urgency references.
- All foreign-key relationships are valid.

## Scripts

- `scripts/database/load_reference_data.py`
- `scripts/database/load_feedback.py`
- `scripts/database/check_feedback_ids.py`

The master NLP CSV was not modified during database loading.
