"""
K-means clustering implemented from scratch in NumPy (course requirement).

Use Euclidean distance, iterative centroid updates, and convergence via max centroid move or iteration cap.
Do not delegate clustering to scikit-learn here.
"""

from __future__ import annotations

import numpy as np


def pairwise_squared_euclidean(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of X (n, d) and rows of C (k, d) -> (n, k)."""
    raise NotImplementedError


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Run K-means on rows of X.

    Returns
    -------
    labels : (n,) int
    centroids : (k, d)
    n_iter : number of iterations executed
    """
    raise NotImplementedError


def assign_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each row of X to nearest centroid index."""
    raise NotImplementedError


def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Recompute centroids as mean of assigned points; empty clusters need a policy (e.g. reinit)."""
    raise NotImplementedError
