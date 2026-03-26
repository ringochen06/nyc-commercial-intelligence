"""Tests for from-scratch K-means in ``src.kmeans_numpy``."""

import numpy as np
import pytest

from src import kmeans_numpy


@pytest.mark.skip(reason="Implement kmeans_numpy.kmeans first")
def test_kmeans_simple_clusters():
    """Toy 2D data with two well-separated blobs; expect stable labels."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30, 2)) + np.array([0.0, 0.0])
    b = rng.standard_normal((30, 2)) + np.array([5.0, 5.0])
    X = np.vstack([a, b])
    labels, centroids, n_iter = kmeans_numpy.kmeans(X, k=2, random_state=0)
    assert labels.shape == (60,)
    assert centroids.shape == (2, 2)
    assert n_iter >= 1
    # Both clusters non-empty
    assert len(np.unique(labels)) == 2
