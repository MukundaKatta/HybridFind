"""Tests for HybridFind core engine."""

from __future__ import annotations

import pytest

from hybridfind.config import SearchConfig
from hybridfind.core import (
    BM25Searcher,
    Document,
    HybridSearch,
    VectorSearcher,
    reciprocal_rank_fusion,
)
from hybridfind.utils import cosine_similarity, tokenize


# -- Fixtures ----------------------------------------------------------------

SAMPLE_TEXTS = [
    "Python is a great programming language for data science",
    "Machine learning models require large datasets for training",
    "Natural language processing enables computers to understand text",
    "Deep learning uses neural networks with many layers",
    "Information retrieval systems help users find relevant documents",
    "Search engines combine keyword matching with semantic understanding",
    "Vector databases store embeddings for similarity search",
    "BM25 is a classic algorithm used in full-text search",
]

SAMPLE_IDS = [f"doc-{i}" for i in range(len(SAMPLE_TEXTS))]
SAMPLE_META = [{"category": "tech" if i % 2 == 0 else "ml"} for i in range(len(SAMPLE_TEXTS))]


@pytest.fixture
def engine() -> HybridSearch:
    hs = HybridSearch()
    hs.add_documents(texts=SAMPLE_TEXTS, ids=SAMPLE_IDS, metadatas=SAMPLE_META)
    return hs


# -- Test cases ---------------------------------------------------------------


class TestBM25Searcher:
    def test_basic_search(self) -> None:
        docs = [
            Document(doc_id="1", text="hello world", tokens=tokenize("hello world")),
            Document(doc_id="2", text="goodbye world", tokens=tokenize("goodbye world")),
        ]
        bm25 = BM25Searcher()
        bm25.index(docs)
        results = bm25.search(tokenize("hello"), top_k=2)
        assert len(results) >= 1
        # The first result should be doc index 0
        assert results[0][0] == 0
        assert results[0][1] > 0

    def test_no_match_returns_empty(self) -> None:
        docs = [Document(doc_id="1", text="alpha beta", tokens=tokenize("alpha beta"))]
        bm25 = BM25Searcher()
        bm25.index(docs)
        results = bm25.search(tokenize("zzz_nonexistent_term"), top_k=5)
        assert results == []


class TestVectorSearcher:
    def test_cosine_relevance(self) -> None:
        docs = [
            Document(doc_id="1", text="machine learning algorithms", tokens=tokenize("machine learning algorithms")),
            Document(doc_id="2", text="cooking recipes bread", tokens=tokenize("cooking recipes bread")),
        ]
        vs = VectorSearcher()
        vs.index(docs)
        results = vs.search(tokenize("learning algorithms"), top_k=2)
        assert len(results) >= 1
        assert results[0][0] == 0  # ML doc should rank first


class TestReciprocalRankFusion:
    def test_rrf_merges_rankings(self) -> None:
        ranking_a = [(0, 5.0), (1, 3.0), (2, 1.0)]
        ranking_b = [(2, 9.0), (0, 4.0), (1, 1.0)]
        fused = reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
        ids = [idx for idx, _ in fused]
        # Doc 0 appears at rank 1 in both → should be top
        assert ids[0] == 0

    def test_rrf_respects_weights(self) -> None:
        ranking_a = [(0, 10.0)]
        ranking_b = [(1, 10.0)]
        fused = reciprocal_rank_fusion([ranking_a, ranking_b], weights=[1.0, 0.0], k=60)
        assert fused[0][0] == 0


class TestHybridSearch:
    def test_search_returns_results(self, engine: HybridSearch) -> None:
        results = engine.search("search algorithms text retrieval")
        assert len(results) > 0
        assert all(r.score > 0 for r in results)

    def test_metadata_filter(self, engine: HybridSearch) -> None:
        results = engine.search("programming language", metadata_filter={"category": "tech"})
        for r in results:
            assert r.metadata["category"] == "tech"

    def test_empty_query(self, engine: HybridSearch) -> None:
        results = engine.search("   ")
        assert results == []

    def test_custom_weights(self) -> None:
        cfg = SearchConfig(bm25_weight=1.0, vector_weight=0.0)
        hs = HybridSearch(config=cfg)
        hs.add_documents(texts=SAMPLE_TEXTS, ids=SAMPLE_IDS)
        results = hs.search("BM25 full text search")
        assert len(results) > 0

    def test_top_k_limit(self, engine: HybridSearch) -> None:
        results = engine.search("learning", top_k=2)
        assert len(results) <= 2


class TestUtils:
    def test_tokenize_removes_stopwords(self) -> None:
        tokens = tokenize("the quick brown fox is very fast")
        assert "the" not in tokens
        assert "quick" in tokens

    def test_cosine_identical_vectors(self) -> None:
        vec = {"a": 1.0, "b": 2.0}
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self) -> None:
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert cosine_similarity(a, b) == pytest.approx(0.0)
