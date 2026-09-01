# Automated Meeting Decision Extraction System

An NLP-powered web application that finds the sentences in a meeting transcript that express final decisions. A user pastes a transcript into the browser, and the system returns a numbered list of likely decision statements with confidence scores.

## Problem addressed

Important decisions are often hidden among introductions, status updates, questions, and informal conversation. Reviewing long meeting transcripts manually is slow and can cause action items or commitments to be missed. This project automatically identifies decision-bearing sentences so users can review the outcomes of a meeting faster.

**Input:** a free-text meeting transcript.  
**Output:** ordered decision sentences and a confidence value for each one.

## Core algorithm

The system uses a **hybrid NLP classifier**: a machine-learning probability score is combined with lightweight decision-language rules.

1. **Sentence segmentation** - NLTK splits the submitted transcript into individual sentences.
2. **Text normalization** - Each sentence is lowercased, stripped of punctuation, tokenized, and cleaned of English stopwords. Decision-bearing modal words such as `will`, `shall`, `should`, and `must` are intentionally retained.
3. **Feature extraction** - `TfidfVectorizer` converts cleaned sentences into unigram and bigram TF-IDF features. This represents words and short phrases that distinguish decisions from non-decisions.
4. **Model training** - A balanced `LogisticRegression` classifier is trained on labeled sentences from `data/meeting_transcripts.csv`. The data split is stratified (80% training, 20% testing) and fixed with `random_state=42` for repeatable results.
5. **Inference and hybrid decision** - For each new sentence, the trained pipeline produces the probability of the decision class.
   - A sentence is accepted when confidence is **0.45 or higher**.
   - A sentence with confidence from **0.25 to 0.449** is also accepted if it contains a strong decision cue such as “we will,” “we decided,” “we agreed,” “finalized,” or “moving forward with.”
   - Sentences matching common filler/small-talk patterns (for example, “good morning” or “let’s start”) are excluded.
6. **Presentation** - Accepted sentences retain their original text and are returned as numbered decisions with confidence rounded to three decimal places.

This combination makes the behavior more robust than rules alone while still recognizing short, explicit commitments that a small training set may score conservatively.

## Workflow

```text
User pastes transcript in browser
            |
            v
POST /api/extract (Flask)
            |
            v
NLTK sentence segmentation
            |
            v
Cleaning + TF-IDF transformation
            |
            v
Logistic-regression probability
            |
            +--> Filler phrase? ---- yes --> discard
            |
            +--> Confidence >= 0.45? - yes --> keep
            |
            +--> Strong decision phrase and confidence >= 0.25? -- yes --> keep
            |
            v
JSON response: decisions, count, confidence
            |
            v
Browser renders the numbered results
```

## Architecture and project structure

```text
meeting-decision-extractor/
├── backend/
│   ├── app.py                  # Flask web server and API routes
│   ├── extractor.py            # Loads model, scores sentences, applies hybrid rules
│   ├── preprocessing.py        # Shared cleaning, segmentation, and pattern helpers
│   ├── train_model.py          # Trains and saves the TF-IDF + classifier pipeline
│   └── models/                 # Runtime-generated model artifacts (not committed)
├── data/
│   ├── meeting_transcripts.csv # Labeled training sentences
│   └── TEST CASES.docx         # Manual test cases
├── frontend/
│   └── index.html              # Single-page interface, styles, and client-side API calls
├── report/
│   └── Meeting_Decision_Extractor_Report.docx
├── Problem Statement and Topic.docx
├── requirements.txt
└── README.md
```

### Main components

- **Frontend:** A responsive HTML interface provides transcript entry, a sample transcript, extraction controls, and readable decision cards.
- **Flask API:** `POST /api/extract` validates the input, triggers extraction, and returns JSON. `GET /api/health` reports whether the trained model is available.
- **Preprocessing module:** Shared by training and live inference so that the classifier sees the same text representation in both stages.
- **Training module:** Builds a scikit-learn `Pipeline` containing TF-IDF vectorization and logistic regression, then writes it as a Joblib model.
- **Extraction module:** Preserves the original sentence for display while using the cleaned form only for scoring.

## API reference

### `GET /api/health`

Returns service status and whether the model file exists.

```json
{"status": "ok", "model_ready": true}
```

### `POST /api/extract`

Request body:

```json
{"transcript": "We agreed to move the deadline to Friday. Thanks everyone."}
```

Successful response:

```json
{
  "decisions": [
    {
      "index": 1,
      "text": "We agreed to move the deadline to Friday.",
      "confidence": 0.873
    }
  ],
  "count": 1
}
```

## Setup and run

### Prerequisites

- Python 3.10 or later recommended
- Internet access on the first run, if NLTK needs to download tokenizer and stopword resources

### Install and start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python backend/train_model.py
python backend/app.py
```

Open `http://127.0.0.1:5000` in a browser. The app will also train the model automatically if it is missing when an extraction request arrives.

## Technology stack

| Area | Technology |
| --- | --- |
| Web application | Flask, Flask-CORS, HTML, CSS, JavaScript |
| NLP | NLTK |
| Machine learning | scikit-learn, TF-IDF, Logistic Regression |
| Data/model handling | pandas, Joblib |

`spaCy` is listed as an available dependency for future pipeline extensions; the current implementation uses NLTK and scikit-learn directly.

## Important implementation details

- **Reproducibility:** The training/test split uses `random_state=42`; model behavior will still depend on the labeled data and installed package versions.
- **Class imbalance:** Logistic regression uses `class_weight="balanced"` to reduce bias toward the more common class if the training labels are uneven.
- **Model lifecycle:** `backend/models/decision_classifier.joblib` is generated locally and ignored by Git. Run the training command to recreate it after cloning or after changing the dataset/code.
- **Error handling:** Empty API input returns HTTP 400; a missing model returns a clear HTTP 500 error if automatic training cannot create it.
- **Privacy:** The application processes only the transcript submitted to its local Flask server. This repository does not include credentials or cloud services.

## Current limitations and practical next steps

- The approach classifies sentences independently, so it may miss decisions that depend on preceding context or are phrased indirectly.
- Rule patterns are English-specific and should be expanded or localized for other languages.
- The model should be evaluated on a held-out, representative dataset before production use; accuracy is sensitive to the amount and quality of labeled meeting data.
- Potential improvements include speaker-aware segmentation, named-entity extraction for owners/dates, transformer-based sentence classification, configurable thresholds, and an audit/export view of extracted decisions.

## License

No license has been specified. Add a license file before distributing or reusing this project publicly.
