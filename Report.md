# Meeting Decision Extractor Report

**Meeting Decision Extractor** is a lightweight Python application that processes meeting transcripts, applies NLP preprocessing and a trained decision‑classification model, and automatically extracts key decisions and action items. It helps turn raw meeting discussions into concise, actionable minutes.

---

## Algorithm Overview
1. **Preprocessing** – Clean transcript text, split into sentences, and normalize tokens.
2. **Feature Extraction** – Convert sentences into TF‑IDF vectors.
3. **Classification** – A `sklearn` `LinearSVC` model (`decision_classifier.joblib`) predicts whether a sentence contains a decision.
4. **Post‑processing** – Aggregate predicted decision sentences and output them in a structured JSON/CSV format.

---

## Workflow Diagram
```mermaid
flowchart TD
    A[Raw meeting transcript (CSV)] --> B[Preprocessing (preprocessing.py)]
    B --> C[Feature extraction (TF‑IDF)]
    C --> D[Decision classification (extractor.py)]
    D --> E[Extracted decisions (JSON/CSV)]
    style A fill:#f9f,stroke:#333,stroke-width:2px;
    style E fill:#bbf,stroke:#333,stroke-width:2px;
```

---

## Project Structure
```
meeting decidion extractor/
├─ backend/                     # Python backend
│   ├─ app.py                  # FastAPI entry point
│   ├─ extractor.py            # Core extraction logic
│   ├─ preprocessing.py        # Text cleaning utilities
│   ├─ train_model.py          # Model training script
│   └─ models/decision_classifier.joblib
├─ data/                         # Sample data
│   ├─ meeting_transcripts.csv
│   └─ TEST CASES.docx
├─ frontend/                     # Simple UI
│   └─ index.html
├─ report/                       # Original Word report (optional)
├─ requirements.txt
├─ README.md                     # Project overview
└─ .gitignore                    # Git ignore rules
```

---

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn backend.app:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive API docs.

---

## License
This project is released under the MIT License.
