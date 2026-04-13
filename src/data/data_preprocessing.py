import os
import re

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

from src.data.data_ingestion import load_data, save_data, get_logger


logger = get_logger(__name__)


def _ensure_nltk() -> None:
    """Download required NLTK resources only if missing."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        logger.info("Downloading NLTK resource: wordnet")
        nltk.download("wordnet", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        logger.info("Downloading NLTK resource: stopwords")
        nltk.download("stopwords", quiet=True)

_ensure_nltk()

# Cached NLP objects
_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))

# Normalization helpers
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]')

def removing_urls(text: str) -> str:
    if not text:
        return ""
    return _URL_RE.sub(r'', str(text))

def lemmatization(text: str) -> str:
    if not text:
        return ""
    
    text = str(text).split()
    text = [_LEMMATIZER.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text: str) -> str:
    if not text:
        return ""
    
    return " ".join([word for word in str(text).split() if word not in _STOPWORDS])

def removing_numbers(text: str) -> str:
    if not text:
        return ""
    return "".join([char for char in str(text) if not char.isdigit()])

def lower_case(text: str) -> str:
    if text is None:
        return ""
    return str(text).lower()

def removing_punctuations(text: str) -> str:
    if not text:
        return ""
    # Remove punctuation and normalize whitespace
    text = _PUNCT_RE.sub(" ", str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_text(df: pd.DataFrame, text_col: str = "content") -> pd.DataFrame:
    """
    Normalize text in df[text_col] with the following pipeline:
      1) remove_urls
      2) to_lower
      3) remove_numbers
      4) remove_punct
      5) remove_stop_words
      6) lemmatize_text

    Returns a new DataFrame with the normalized text in the same column.
    """
    
    if text_col not in df.columns:
        logger.error("Required text column '%s' not found.", text_col)
        raise KeyError(f"Required text column '{text_col}' not found.")
    
    logger.info("Starting text normalization on column '%s', rows=%d", text_col, len(df))
    out = df.copy()

    # Coerce to string 
    out[text_col] = out[text_col].fillna("").astype(str)

    # Apply pipeline — prefer .map or .apply with clear order
    out[text_col] = out[text_col].map(removing_urls)
    out[text_col] = out[text_col].map(lower_case)
    out[text_col] = out[text_col].map(removing_numbers)
    out[text_col] = out[text_col].map(removing_punctuations)
    out[text_col] = out[text_col].map(remove_stop_words)
    out[text_col] = out[text_col].map(lemmatization)

    logger.info("Completed text normalization. Example preview: %r", out[text_col].head(3).tolist())
    return out


def main() -> None:
    try:
        # Fetch the data from data/raw
        train_data = load_data(os.path.join("data", "raw", "train.csv"))
        test_data = load_data(os.path.join("data", "raw", "test.csv"))

        # Normalize
        train_processed = normalize_text(train_data, text_col="content")
        test_processed = normalize_text(test_data, text_col="content")

        # Save
        data_path = os.path.join("data", "interim")
        save_data(
            data_path=data_path,
            train_data=train_processed,
            test_data=test_processed,
            train_filename="train_processed",
            test_filename="test_processed",
        )

        logger.info("Normalization pipeline completed successfully.")
    except Exception:
        logger.exception("Normalization pipeline failed.")
        raise

if __name__ == "__main__":
    main()

