"""Shared text preprocessing for the sentiment analysis project.

The trained model expects CLEAN text (exactly as produced by notebook 01),
so every review must pass through preprocess_text() before prediction.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_data():
    """Download required NLTK resources only if they are missing."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/stopwords", "stopwords"),
    ]
    for find_path, download_name in resources:
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


_ensure_nltk_data()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"no", "not", "nor", "never"}

contractions = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "isn't": "is not",
    "wasn't": "was not", "aren't": "are not", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "it's": "it is", "i'm": "i am", "i've": "i have", "i'd": "i would",
    "that's": "that is", "there's": "there is", "what's": "what is",
    "let's": "let us", "you're": "you are", "they're": "they are",
}


def preprocess_text(text):
    text = str(text).lower()
    for short, full in contractions.items():
        text = text.replace(short, full)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)