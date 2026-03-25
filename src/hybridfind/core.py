"""Core HybridFind engine — BM25 keyword search, TF-IDF vector similarity, and RRF fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from hybridfind.config import SearchConfig
from hybridfind.utils import (
    compute_idf,
    cosine_similarity,
    tfidf_vector,
    tokenize,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A document stored in the search index."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list, repr=False)


@dataclass
class SearchResult:
    """A single search result."""

    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BM25 Searcher (from-scratch implementation)
# ---------------------------------------------------------------------------

class BM25Searcher:
    """Classic Okapi BM25 scoring implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[Document] = []
        self.avgdl: float = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.doc_len: list[int] = []
        self.n_docs: int = 0
        self.term_freqs: list[dict[str, int]] = []

    def index(self, docs: list[Document]) -> None:
        """Build the BM25 index from a list of documents."""
        self.docs = docs
        self.n_docs = len(docs)
        self.doc_freqs = {}
        self.term_freqs = []
        self.doc_len = []

        total_len = 0
        for doc in docs:
            tf: dict[str, int] = {}
            for token in doc.tokens:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            dl = len(doc.tokens)
            self.doc_len.append(dl)
            total_len += dl
            for term in set(doc.tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avgdl = total_len / self.n_docs if self.n_docs else 0.0

    def _idf(self, term: str) -> float:
        """Compute IDF for a term using the standard BM25 IDF formula."""
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs sorted by descending BM25 score."""
        scores: list[float] = [0.0] * self.n_docs
        for token in query_tokens:
            idf = self._idf(token)
            for idx in range(self.n_docs):
                tf = self.term_freqs[idx].get(token, 0)
                dl = self.doc_len[idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl else tf + self.k1
                numerator = tf * (self.k1 + 1)
                scores[idx] += idf * (numerator / denom) if denom else 0.0

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, sc) for idx, sc in ranked[:top_k] if sc > 0]


# ---------------------------------------------------------------------------
# Vector (TF-IDF + cosine) Searcher
# ---------------------------------------------------------------------------

class VectorSearcher:
    """TF-IDF based semantic search with cosine similarity."""

    def __init__(self) -> None:
        self.docs: list[Document] = []
        self.idf: dict[str, float] = {}
        self.doc_vectors: list[dict[str, float]] = []

    def index(self, docs: list[Document]) -> None:
        """Build TF-IDF vectors for all documents."""
        self.docs = docs
        corpus_tokens = [doc.tokens for doc in docs]
        self.idf = compute_idf(corpus_tokens)
        self.doc_vectors = [tfidf_vector(doc.tokens, self.idf) for doc in docs]

    def search(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_index, cosine_score) pairs sorted descending."""
        query_vec = tfidf_vector(query_tokens, self.idf)
        scored = [
            (idx, cosine_similarity(query_vec, dvec))
            for idx, dvec in enumerate(self.doc_vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(idx, sc) for idx, sc in scored[:top_k] if sc > 0]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using weighted Reciprocal Rank Fusion.

    Args:
        rankings: List of ranked result lists, each containing (doc_index, score).
        weights: Optional per-ranking weight (defaults to uniform).
        k: RRF smoothing constant.

    Returns:
        Fused ranking as (doc_index, rrf_score) sorted descending.
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (doc_idx, _score) in enumerate(ranking, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Main Hybrid Search engine
# ---------------------------------------------------------------------------

class HybridSearch:
    """Combines BM25 keyword search with TF-IDF vector similarity via RRF."""

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self.bm25 = BM25Searcher(k1=self.config.bm25_k1, b=self.config.bm25_b)
        self.vector = VectorSearcher()
        self.documents: list[Document] = []

    def add_documents(
        self,
        texts: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Ingest documents into both search indices."""
        for i, text in enumerate(texts):
            doc_id = ids[i] if ids else str(len(self.documents) + i)
            meta = metadatas[i] if metadatas else {}
            tokens = tokenize(text)
            self.documents.append(Document(doc_id=doc_id, text=text, metadata=meta, tokens=tokens))
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild BM25 and vector indices from current documents."""
        self.bm25.index(self.documents)
        self.vector.index(self.documents)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search combining BM25 and vector results via RRF.

        Args:
            query: The search query string.
            top_k: Number of results to return (overrides config).
            metadata_filter: Optional dict of metadata key-value pairs to filter on.

        Returns:
            Ranked list of SearchResult objects.
        """
        k = top_k or self.config.top_k
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Determine candidate doc indices (apply metadata filter)
        candidate_indices: set[int] | None = None
        if metadata_filter:
            candidate_indices = set()
            for idx, doc in enumerate(self.documents):
                if all(doc.metadata.get(key) == val for key, val in metadata_filter.items()):
                    candidate_indices.add(idx)

        # Run both searchers
        bm25_results = self.bm25.search(query_tokens, top_k=len(self.documents))
        vec_results = self.vector.search(query_tokens, top_k=len(self.documents))

        # Filter by metadata if needed
        if candidate_indices is not None:
            bm25_results = [(idx, sc) for idx, sc in bm25_results if idx in candidate_indices]
            vec_results = [(idx, sc) for idx, sc in vec_results if idx in candidate_indices]

        # Fuse with RRF
        w_bm25, w_vec = self.config.effective_weights()
        fused = reciprocal_rank_fusion(
            [bm25_results, vec_results],
            weights=[w_bm25, w_vec],
            k=self.config.rrf_k,
        )

        # Build results
        results: list[SearchResult] = []
        for doc_idx, score in fused[:k]:
            doc = self.documents[doc_idx]
            results.append(
                SearchResult(doc_id=doc.doc_id, score=score, text=doc.text, metadata=doc.metadata)
            )
        return results
