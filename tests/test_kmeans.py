"""Tests for from-scratch K-means in ``src.kmeans_numpy``."""

import numpy as np
import pytest

from src.kmeans_numpy import (
    assign_labels,
    compute_inertia,
    kmeans,
    minibatch_kmeans,
    silhouette_score,
)


def test_kmeans_simple_clusters():
    """Toy 2D data with two well-separated blobs; expect stable labels."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30, 2)) + np.array([0.0, 0.0])
    b = rng.standard_normal((30, 2)) + np.array([5.0, 5.0])
    X = np.vstack([a, b])
    labels, centroids, n_iter = kmeans(X, k=2, random_state=0)
    assert labels.shape == (60,)
    assert centroids.shape == (2, 2)
    assert n_iter >= 1
    assert len(np.unique(labels)) == 2


def test_assign_labels_nearest_centroid():
    X = np.array([[0.0, 0.0], [10.0, 0.0]])
    centroids = np.array([[0.0, 0.0], [9.0, 0.0]])
    labels = assign_labels(X, centroids)
    assert labels[0] == 0 and labels[1] == 1


def test_compute_inertia_matches_manual():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    labels = np.array([0, 0, 1])
    centroids = np.array([[1.0, 0.0], [10.0, 0.0]])
    inertia = compute_inertia(X, labels, centroids)
    # Point 0: dist to c0 = 1, point 1: dist to c0 = 1 -> 1+1=2 for dim contribution squared: (0-1)^2+(2-1)^2 = 1+1=2 on x only -> 2
    # Actually sum of squared distances: (0-1)^2 + (2-1)^2 + (10-10)^2 = 1+1+0 = 2
    assert inertia == pytest.approx(2.0)


def test_silhouette_well_separated_positive():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((25, 2)) + np.array([0.0, 0.0])
    b = rng.standard_normal((25, 2)) + np.array([8.0, 0.0])
    X = np.vstack([a, b])
    labels, _, _ = kmeans(X, k=2, random_state=0)
    s = silhouette_score(X, labels)
    assert s > 0.3


def test_minibatch_kmeans_not_implemented():
    X = np.zeros((5, 2))
    with pytest.raises(NotImplementedError):
        minibatch_kmeans(X, k=2)

