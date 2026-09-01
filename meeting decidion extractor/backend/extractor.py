"""
Decision extraction logic shared by the CLI and Flask API.
"""

import joblib

from preprocessing import (
    clean_sentence,
    has_decision_pattern,
    has_filler_pattern,
    segment_into_sentences,
)
from train_model import MODEL_PATH

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Run: python backend/train_model.py"
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def score_sentence(sentence: str) -> float:
    """Return probability that a sentence is a decision (0-1)."""
    pipeline = get_pipeline()
    cleaned = clean_sentence(sentence)
    if not cleaned.strip():
        return 0.0
    return float(pipeline.predict_proba([cleaned])[0][1])


def is_decision_sentence(sentence: str, confidence: float, threshold: float = 0.45) -> bool:
    """
    Classify a single sentence given its precomputed confidence score.
    Rule-based patterns can boost borderline cases.
    """
    stripped = sentence.strip()
    if not stripped or has_filler_pattern(stripped):
        return False

    if confidence >= threshold:
        return True

    # Boost strong lexical decision cues the model may miss on short text.
    if has_decision_pattern(stripped) and confidence >= 0.25:
        return True

    return False


def extract_decisions(raw_transcript: str) -> list[dict]:
    """
    Segment transcript, classify each sentence, return decision list.

    Each item: {"index": 1, "text": "...", "confidence": 0.87}
    """
    sentences = segment_into_sentences(raw_transcript)
    decisions = []

    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue

        confidence = score_sentence(stripped)
        if is_decision_sentence(stripped, confidence):
            decisions.append(
                {
                    "index": len(decisions) + 1,
                    "text": stripped,
                    "confidence": round(confidence, 3),
                }
            )

    return decisions