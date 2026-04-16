"""Tests for from-scratch K-means in ``src.kmeans_numpy``."""

import numpy as np
import pytest


from src.kmeans_numpy import kmeans


#@pytest.mark.skip(reason="Implement kmeans_numpy.kmeans first")
def test_kmeans_simple_clusters():
    """Toy 2D data with two well-separated blobs; expect stable labels."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((30, 2)) + np.array([0.0, 0.0])
    b = rng.standard_normal((30, 2)) + np.array([5.0, 5.0])
    X = np.vstack([a, b])
    labels, centroids, n_iter = kmeans(X, k=2, random_state=0)
    # Print and assert label shape
    print(f"✓ Testing label shape: {labels.shape}")
    assert labels.shape == (60,)
    
    # Print and assert centroid shape
    print(f"✓ Testing centroid shape: {centroids.shape}")
    assert centroids.shape == (2, 2)
    
    # Print and assert iterations
    print(f"✓ Testing iterations: {n_iter}")
    assert n_iter >= 1
    
    # Print and assert unique clusters
    unique_labels = np.unique(labels)
    print(f"✓ Testing unique clusters: {len(unique_labels)} clusters found")
    assert len(unique_labels) == 2
    print(f"Centroids:\n{centroids}\nLabels:\n{labels}\nIterations: {n_iter}\n")

