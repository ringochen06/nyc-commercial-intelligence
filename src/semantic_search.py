"""
Embedding-based retrieval: encode neighborhood text with sentence-transformers, cosine similarity.

Model default: all-MiniLM-L6-v2 (set in implementation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_or_fit_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """Load SentenceTransformer; cache path can live under outputs/embeddings/."""
    raise NotImplementedError


def encode_texts(texts: list[str], model: Any) -> np.ndarray:
    """Return (n, d) float32 embeddings."""
    raise NotImplementedError


def cosine_similarity_matrix(query_vec: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query row and corpus rows; returns shape (n_corpus,)."""
    raise NotImplementedError


def retrieve_top_k(
    query: str,
    neighborhood_texts: list[str],
    neighborhood_ids: list[str],
    *,
    model: Any | None = None,
    k: int = 10,
    embeddings_cache: Path | None = None,
) -> list[tuple[str, float]]:
    """Return (neighborhood_id, similarity) pairs sorted descending."""
    raise NotImplementedError
