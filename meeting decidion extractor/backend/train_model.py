"""
Train a TF-IDF + Logistic Regression classifier on labeled meeting sentences.
Run once before starting the web app:

    python backend/train_model.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import clean_sentence

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "meeting_transcripts.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "decision_classifier.joblib"


def load_training_data() -> tuple[list[str], list[int]]:
    df = pd.read_csv(DATA_PATH)
    texts = [clean_sentence(s) for s in df["sentence"].astype(str)]
    labels = df["label"].astype(int).tolist()
    return texts, labels


def train_and_save() -> Pipeline:
    texts, labels = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["non-decision", "decision"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return pipeline


if __name__ == "__main__":
    train_and_save()
