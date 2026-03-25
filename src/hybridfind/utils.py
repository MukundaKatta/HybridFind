"""Text preprocessing, tokenization, and TF-IDF utilities."""

from __future__ import annotations

import math
import re
import string
from collections import Counter

# Common English stop words
STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "not", "no", "so", "if", "then", "than",
    "as", "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "further",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "just", "also", "very", "too", "quite",
}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize text into lowercase terms, optionally removing stop words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.split(r"\s+", text.strip())
    tokens = [t for t in tokens if t]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute raw term-frequency for a token list."""
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {term: count / total for term, count in counts.items()}


def compute_idf(corpus_tokens: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency across a corpus.

    Uses the smoothed IDF formula: log((N + 1) / (df + 1)) + 1
    """
    n = len(corpus_tokens)
    df: dict[str, int] = {}
    for doc_tokens in corpus_tokens:
        seen: set[str] = set()
        for token in doc_tokens:
            if token not in seen:
                df[token] = df.get(token, 0) + 1
                seen.add(token)
    return {term: math.log((n + 1) / (freq + 1)) + 1 for term, freq in df.items()}


def tfidf_vector(
    tokens: list[str], idf: dict[str, float]
) -> dict[str, float]:
    """Build a TF-IDF vector (sparse dict) for a single document."""
    tf = compute_tf(tokens)
    return {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0
    common_keys = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
