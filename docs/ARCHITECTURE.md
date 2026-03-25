# Architecture

## Overview

HybridFind implements a dual-path search architecture that combines lexical (BM25) and semantic (TF-IDF vector) retrieval, fusing results with Reciprocal Rank Fusion (RRF).

## Components

### BM25Searcher

Implements the Okapi BM25 scoring algorithm from scratch. Key parameters:

- **k1** — controls term-frequency saturation (default 1.5)
- **b** — controls document-length normalization (default 0.75)

The IDF component uses the standard formula: `log(1 + (N - df + 0.5) / (df + 0.5))`.

### VectorSearcher

Builds TF-IDF vectors for every document at index time, then scores queries via cosine similarity. Uses smoothed IDF: `log((N+1)/(df+1)) + 1`.

### Reciprocal Rank Fusion (RRF)

Merges ranked lists from both searchers using the formula:

```
RRF(d) = Σ  w_i / (k + rank_i(d))
```

where `k` is a smoothing constant (default 60) and `w_i` are per-ranking weights.

### HybridSearch

The top-level orchestrator that:

1. Tokenizes and indexes documents into both BM25 and vector stores.
2. At query time, runs both searchers in parallel.
3. Optionally filters by metadata fields.
4. Fuses results with weighted RRF.
5. Returns a ranked list of `SearchResult` objects.

## Data Flow

```
Documents ──► Tokenizer ──┬──► BM25 Index
                          └──► TF-IDF Vectors

Query ──► Tokenizer ──┬──► BM25 Search ──────┐
                      └──► Vector Search ─────┤
                                              ▼
                                     RRF Fusion ──► Ranked Results
```

## Configuration

All tunable parameters live in `SearchConfig` (Pydantic model), making it easy to experiment with different weight combinations.
