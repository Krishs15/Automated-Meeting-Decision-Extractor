"""
preprocessing.py
-----------------
Reusable text preprocessing functions for the Meeting Decision Extraction project.

This module is imported by BOTH the training script and the live Flask API,
so the exact same cleaning logic is applied at training time and at prediction
time. This consistency is essential -- if training and inference preprocess
text differently, the trained model's learned patterns won't transfer
correctly to new input.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# ---------------------------------------------------------------------------
# One-time NLTK resource downloads.
# These download to a local NLTK data folder the first time this runs.
# Wrapped in try/except so re-running the script doesn't re-download.
# ---------------------------------------------------------------------------
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

STOPWORDS = set(stopwords.words("english"))

# Keep decision-relevant modal verbs even though they're technically
# "stopwords" in NLTK's list -- removing them would destroy the exact
# signal we're trying to detect (e.g. "will", "shall").
DECISION_SIGNAL_WORDS = {"will", "shall", "should", "must"}
STOPWORDS = STOPWORDS - DECISION_SIGNAL_WORDS

# Phrases that strongly indicate a sentence is a filler / small-talk opener.
# Used as a feature later, not for deletion -- we don't want to throw away
# the sentence, just flag it.
FILLER_PATTERNS = [
    r"\bgood morning\b", r"\bgood afternoon\b", r"\bhi all\b", r"\bhey everyone\b",
    r"\bhello team\b", r"\bthanks everyone\b", r"\blet'?s (start|begin|dive|wrap|go over)\b",
    r"\bhope (you|everyone)\b",
]

# Phrases that strongly indicate a decision was made.
# This list is used for rule-based features (Step 4), defined here since
# it's tightly coupled to the text-cleaning vocabulary.
DECISION_PATTERNS = [
    r"\bwe (will|shall)\b", r"\bwe (have )?decided\b", r"\bwe agreed\b",
    r"\bfinalized\b", r"\bfinalize\b", r"\bagreed to\b", r"\bmoving forward with\b",
]


def clean_sentence(sentence: str) -> str:
    """
    Lowercase, strip punctuation, and remove stopwords (excluding decision
    signal words like 'will'/'shall') from a single sentence.

    Returns a cleaned string of space-separated tokens, suitable for
    TF-IDF vectorization.
    """
    text = sentence.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def segment_into_sentences(raw_text: str) -> list[str]:
    """
    Split a raw transcript block of text into individual sentences.
    Used at inference time when a user pastes a full transcript into the
    web app -- our training CSV is already pre-segmented.
    """
    return sent_tokenize(raw_text)


def has_filler_pattern(sentence: str) -> bool:
    """Return True if the sentence matches a common small-talk/filler pattern."""
    lowered = sentence.lower()
    return any(re.search(pat, lowered) for pat in FILLER_PATTERNS)


def has_decision_pattern(sentence: str) -> bool:
    """Return True if the sentence matches a common decision-language pattern."""
    lowered = sentence.lower()
    return any(re.search(pat, lowered) for pat in DECISION_PATTERNS)


if __name__ == "__main__":
    # Quick manual test -- run `python preprocessing.py` to sanity-check.
    samples = [
        "We will finalize the database schema by Thursday.",
        "Good morning everyone thanks for joining on time.",
        "We have decided to use PostgreSQL instead of MySQL for this project.",
    ]
    for s in samples:
        print(f"Original : {s}")
        print(f"Cleaned  : {clean_sentence(s)}")
        print(f"Filler?  : {has_filler_pattern(s)}")
        print(f"Decision?: {has_decision_pattern(s)}")
        print()
