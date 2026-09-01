# Automated Meeting Decision Extractor

A lightweight Python application that parses meeting transcripts, applies NLP preprocessing, and uses a trained decision-classification model to automatically extract key decisions and action items.

## Algorithm Overview
1. **Load Transcript** – Read CSV/TSV of meeting dialogue.
2. **Preprocess** – Tokenise, filter stop-words, lemmatise, and generate TF-IDF features.
3. **Classification** – A scikit-learn DecisionTreeClassifier (or joblib-saved model) predicts whether a sentence contains a decision.
4. **Post-process** – Group consecutive decision sentences, assign timestamps, and output a concise summary.

## Workflow Diagram
`mermaid
flowchart TD
    A[Read transcript CSV] --> B[Preprocess text]
    B --> C[Extract TF-IDF features]
    C --> D[Load trained model]
    D --> E[Predict decision sentences]
    E --> F[Aggregate & format output]
    F --> G[Export JSON/HTML report]
`

## Project Structure
`
Automated Meeting Decision Meeting/
+- backend/
¦   +- app.py            # FastAPI entry point
¦   +- extractor.py      # Core extraction logic
¦   +- preprocessing.py   # Text cleaning and feature extraction
¦   +- train_model.py     # Model training script
¦   +- models/
¦       +- decision_classifier.joblib
+- data/
¦   +- meeting_transcripts.csv
+- frontend/
¦   +- index.html        # Simple UI for uploading transcripts
+- requirements.txt
+- README.md
`

## Usage
`ash
# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn backend.app:app --reload
`

Upload a transcript via the web UI or call the /extract endpoint to receive extracted decisions.

## License
MIT License – see LICENSE file.
