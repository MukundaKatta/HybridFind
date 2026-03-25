"""Search configuration for HybridFind."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchConfig(BaseModel):
    """Configuration for the hybrid search engine."""

    bm25_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for BM25 keyword results"
    )
    vector_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight for vector similarity results"
    )
    rrf_k: int = Field(
        default=60, gt=0, description="RRF constant k (controls rank smoothing)"
    )
    bm25_k1: float = Field(
        default=1.5, gt=0.0, description="BM25 term-frequency saturation parameter"
    )
    bm25_b: float = Field(
        default=0.75, ge=0.0, le=1.0, description="BM25 length-normalization parameter"
    )
    top_k: int = Field(default=10, gt=0, description="Number of results to return")

    def effective_weights(self) -> tuple[float, float]:
        """Return normalized weights that sum to 1."""
        total = self.bm25_weight + self.vector_weight
        if total == 0:
            return 0.5, 0.5
        return self.bm25_weight / total, self.vector_weight / total
