"""
Combine semantic similarity and predicted persistence into a final neighborhood score.

score = α · similarity_norm + β · persistence_pred_norm, with α + β = 1 (configurable).
"""

from __future__ import annotations

import numpy as np


def normalize_minmax(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale to [0, 1] per vector."""
    raise NotImplementedError


def combine_scores(
    similarity: np.ndarray,
    predicted_persistence: np.ndarray,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> np.ndarray:
    """Weighted combination after normalization; alpha + beta should be 1."""
    raise NotImplementedError


def rank_neighborhoods(
    neighborhood_ids: list[str],
    similarity: np.ndarray,
    predicted_persistence: np.ndarray,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> list[tuple[str, float]]:
    """Return (id, final_score) sorted descending."""
    raise NotImplementedError
