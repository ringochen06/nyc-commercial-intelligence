"""Tests for from-scratch K-means in ``src.kmeans_numpy``."""

import numpy as np
import pytest

from src.kmeans_numpy import (
    assign_labels,
    compute_inertia,
    kmeans,
    kmeans_plus_plus_with_caching,
    kmeans_plus_plus,
    minibatch_kmeans,
    silhouette_score,
)


def test_kmeans_simple_clusters():
    """Toy 2D data with two well-separated blobs; expect stable labels."""
    print("\n[TEST] test_kmeans_simple_clusters: Testing K-means clustering on well-separated 2D blobs")
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30, 2)) + np.array([0.0, 0.0])
    b = rng.standard_normal((30, 2)) + np.array([5.0, 5.0])
    X = np.vstack([a, b])
    labels, centroids, n_iter = kmeans_plus_plus(X, k=2, random_state=0)
    assert labels.shape == (60,)
    assert centroids.shape == (2, 2)
    assert n_iter >= 1
    assert len(np.unique(labels)) == 2


def test_kmeans_cache():
    """Test that K-means caching mechanism saves and loads results correctly."""
    print("\n[TEST] test_kmeans_cache: Testing K-means caching mechanism")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    features = [f"feat_{i}" for i in range(X.shape[1])]
    
    # First run should compute and cache results
    labels1, centroids1, n_iter1 = kmeans_plus_plus_with_caching(features, X, k=2, random_state=0)
    
    # Second run should load from cache (simulate by calling again)
    labels2, centroids2, n_iter2 = kmeans_plus_plus_with_caching(features, X, k=2, random_state=0)
    
    assert np.array_equal(labels1, labels2)
    assert np.allclose(centroids1, centroids2)
    assert n_iter2 == 0  # Indicates loaded from cache


def test_assign_labels_nearest_centroid():
    print("\n[TEST] test_assign_labels_nearest_centroid: Testing label assignment to nearest centroid")
    X = np.array([[0.0, 0.0], [10.0, 0.0]])
    centroids = np.array([[0.0, 0.0], [9.0, 0.0]])
    labels = assign_labels(X, centroids)
    assert labels[0] == 0 and labels[1] == 1


def test_compute_inertia_matches_manual():
    print("\n[TEST] test_compute_inertia_matches_manual: Testing inertia calculation against manual computation")
    X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    labels = np.array([0, 0, 1])
    centroids = np.array([[1.0, 0.0], [10.0, 0.0]])
    inertia = compute_inertia(X, labels, centroids)
    # Point 0: dist to c0 = 1, point 1: dist to c0 = 1 -> 1+1=2 for dim contribution squared: (0-1)^2+(2-1)^2 = 1+1=2 on x only -> 2
    # Actually sum of squared distances: (0-1)^2 + (2-1)^2 + (10-10)^2 = 1+1+0 = 2
    assert inertia == pytest.approx(2.0)


def test_silhouette_well_separated_positive():
    print("\n[TEST] test_silhouette_well_separated_positive: Testing silhouette score is positive for well-separated clusters")
    rng = np.random.default_rng(1)
    a = rng.standard_normal((25, 2)) + np.array([0.0, 0.0])
    b = rng.standard_normal((25, 2)) + np.array([8.0, 0.0])
    X = np.vstack([a, b])
    labels, _, _ = kmeans_plus_plus(X, k=2, random_state=0)
    s = silhouette_score(X, labels)
    assert s > 0.3


def test_minibatch_kmeans_not_implemented():
    print("\n[TEST] test_minibatch_kmeans_not_implemented: Testing that minibatch_kmeans raises NotImplementedError")
    X = np.zeros((5, 2))
    with pytest.raises(NotImplementedError):
        minibatch_kmeans(X, k=2)

