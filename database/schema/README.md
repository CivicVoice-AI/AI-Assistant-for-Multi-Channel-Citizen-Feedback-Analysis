# CIVICVOICE Database Schema

## Step 1: Database Schema Design

Created the MySQL database:

`civicvoice_db`

### Tables

- `sources` - stores feedback sources.
- `locations` - stores location/district names.
- `feedback` - stores the main citizen feedback records.
- `feedback_urgency` - stores urgency labels for labelled feedback.

### Dataset

The processed master dataset contains:

- Total records: 205,168
- Unique feedback IDs: 205,168
- Duplicate IDs: 0
- Missing IDs: 0

### Sources

- CPGRAMS: 175,779
- BBMP_Bengaluru: 14,517
- Survey: 10,000
- Twitter: 3,672
- Email: 1,200

### Urgency

- Low: 10,757
- Medium: 6,950
- High: 4,895
- Critical: 1,915
- Unlabelled: 180,651

### Database Validation

The following tables were successfully created:

- `feedback`
- `feedback_urgency`
- `locations`
- `sources`
