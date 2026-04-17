"""
K-means clustering implemented from scratch in NumPy (course requirement).

Use Euclidean distance, iterative centroid updates, and convergence via max centroid move or iteration cap.
Do not delegate clustering to scikit-learn here.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import hashlib

try:
    from . import serialization
except ImportError:
    # Loaded as top-level `kmeans_numpy` when `src/` is on sys.path (e.g. Streamlit `app.py`).
    import serialization

CLUSTER_PATH = Path("outputs/clusters/")

def _features_hash(features: list[str]) -> str:
    """Generate a short hash from a sorted list of feature names."""
    sorted_features = sorted(features)
    combined = "|".join(sorted_features)
    return hashlib.md5(combined.encode()).hexdigest()[:8]

def pairwise_squared_euclidean(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of X (n, d) and rows of C (k, d) -> (n, k)."""
    return np.linalg.norm(X[:, np.newaxis] - C, axis=2) ** 2

def pairwise_absolute_distance(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Absolute distances (L1) between rows of X and C, same shape as above."""
    return np.sum(np.abs(X[:, np.newaxis] - C), axis=2)

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
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    centroids = X[rng.choice(n, size=k, replace=False)]

    for i in range(max_iter):
        z = assign_labels(X, centroids)
        new_centroids = update_centroids(X, z, k)
        if np.linalg.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return z, centroids, i + 1

def kmeans_with_caching(
    features: list[str],
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run kmeans with caching to avoid recomputation during development.
    
    Caches results in HDF5 file per k, with datasets keyed by feature hash.
    Allows appending results for different feature selections within the same k.
    
    Parameters
    ----------
    features : list[str]
        Feature names used for clustering (used to generate cache key).
    X : np.ndarray
        Feature matrix (n, d).
    k : int
        Number of clusters.
    max_iter : int
        Maximum iterations for K-means.
    tol : float
        Convergence tolerance.
    random_state : int | None
        Random seed for reproducibility.
    
    Returns
    -------
    labels : np.ndarray
        Cluster labels (n,).
    centroids : np.ndarray
        Cluster centroids (k, d).
    n_iter : int
        Number of iterations run (0 if loaded from cache).
    """
    CLUSTER_PATH.mkdir(parents=True, exist_ok=True)
    cache_file = CLUSTER_PATH / f"kmeans_k{k}.h5"
    
    # Try to load from cache
    try:
        labels, centroids = load_kmeans_results(k, features, cache_file)
        return labels, centroids, 0  # 0 iterations indicates loaded from cache
    except (FileNotFoundError, KeyError):
        pass  # Cache miss, proceed with computation
    
    # Compute new results
    labels, centroids, n_iter = kmeans(X, k, max_iter=max_iter, tol=tol, random_state=random_state)
    save_kmeans_results(k, features, labels, centroids, cache_file)
    return labels, centroids, n_iter

def assign_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each row of X to nearest centroid index (0 to k-1) based on squared Euclidean distance. Labels should be shape (n,)."""
    distances = pairwise_squared_euclidean(X, centroids)
    return np.argmin(distances, axis=1)


def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Recompute centroids as mean of assigned points; empty clusters need a policy (e.g. reinit)."""
    indicator = np.array([labels == j for j in range(k)], dtype=float)  # (k, n)
    counts = np.sum(indicator, axis=1)  # (k,)
      # (k, d)
    empty = np.where(counts == 0)[0]
    if len(empty) > 0:
        rng = np.random.default_rng()
        reinit_indices = rng.choice(X.shape[0], size=len(empty), replace=False)
        #indicator[empty] = 0.0
        indicator[empty, reinit_indices] = 1.0
        counts[empty] = 1.0
    centroid_sums = indicator @ X
    return centroid_sums / counts[:, np.newaxis]  # (k, n) @ (n, d) -> (k, d)



def minibatch_kmeans(
    X: np.ndarray,
    k: int,
    tol: float = 1e-4,
    random_state: int | None = None,
    batch_size: int = 100,
    learning_rate: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mini-batch K-means (optional extension). Not used by ``app.py``; full batch ``kmeans`` is used instead."""
    raise NotImplementedError(
        "minibatch_kmeans is not implemented; use kmeans() or implement this for large-scale data."
    )


def compute_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Within-cluster sum of squared distances (WCSS / inertia)."""
    return float(np.sum((X - centroids[labels]) ** 2))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient over all samples (pure NumPy).

    s(i) = (b(i) - a(i)) / max(a(i), b(i))

    where a(i) is the mean intra-cluster distance and b(i) is the mean
    distance to the nearest other cluster.  Returns 0.0 for singleton
    clusters and when there is only one cluster.
    """
    # Extract total number of data points
    n = X.shape[0]
    # Get list of unique cluster labels present in the labels array
    unique_labels = np.unique(labels)
    # Guard: silhouette is undefined for single-cluster or empty clustering
    if len(unique_labels) < 2:
        return 0.0

    # Compute pairwise Euclidean distances: diff shape is (n, n, d)
    # where diff[i, j, :] is the vector difference X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]# (n, n, d)
    # Take L2 norm along feature dimension to get (n, n) distance matrix
    # D[i, j] = Euclidean distance from sample i to sample j
    D = np.sqrt(np.sum(diff ** 2, axis=2))             # (n, n)

    # Initialize silhouette coefficient array for each sample
    s = np.zeros(n)
    # Loop through each sample to compute its silhouette coefficient
    for i in range(n):
        # Retrieve the cluster label for the current sample i
        c = labels[i]
        # Create boolean mask: True where labels match sample i's cluster
        same_mask = labels == c
        # Exclude sample i itself (we don't measure distance to self)
        same_mask[i] = False
        # Count how many other samples share the same cluster as i
        cluster_size = np.sum(same_mask)

        # Compute a(i): mean distance to samples in the same cluster
        # Use D[i, same_mask] to index row i at columns where same_mask is True
        # .mean() computes the average; if cluster_size == 0 (sample is alone), set a(i) = 0
        a_i = D[i, same_mask].mean() if cluster_size > 0 else 0.0

        # Compute b(i): minimum mean distance to samples in any other cluster
        # Initialize to infinity (any real value will be smaller)
        b_i = np.inf
        # Iterate through all unique clusters (to consider each as a "nearest neighbor cluster")
        for c_other in unique_labels:
            # Skip the sample's own cluster (we already computed a(i) for that)
            if c_other == c:
                continue
            # Create boolean mask: True where labels belong to cluster c_other
            other_mask = labels == c_other
            # Compute mean distance from sample i to all samples in cluster c_other
            mean_dist = D[i, other_mask].mean()
            # Track the minimum mean distance across all other clusters;
            # this is the distance to the "nearest neighboring cluster"
            if mean_dist < b_i:
                b_i = mean_dist

        # Compute denominator for silhouette formula: max(a(i), b(i))
        # This ensures the coefficient is always in [-1, 1]
        denom = max(a_i, b_i)
        # Compute silhouette coefficient: s(i) = (b(i) - a(i)) / max(a(i), b(i))
        # Numerator (b(i) - a(i)) is positive when sample is closer to its own cluster
        # (i.e., a(i) < b(i), which means tight clusters and good separation)
        # If denom is 0 (sample is isolated), set s(i) = 0 to avoid division by zero
        s[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    # Return the mean silhouette coefficient across all samples
    return float(np.mean(s))



def save_kmeans_results(k: int, features: list[str], labels: np.ndarray, centroids: np.ndarray, path: Path | str) -> None:
    """Persist clustering results to HDF5 file with features-based key.
    
    Each feature selection is stored as a separate dataset group in the same k file,
    allowing multiple feature sets to coexist for the same k value.
    """
    path = Path(path) if isinstance(path, str) else path
    feat_key = _features_hash(features)
    
    data = {
        f"features_{feat_key}": np.array(features, dtype=object),
        f"labels_{feat_key}": labels,
        f"centroids_{feat_key}": centroids,
    }
    serialization.save_hdf5_dict(data, path)

def load_kmeans_results(k: int, features: list[str], path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load persisted clustering results from HDF5 file by k and features.
    
    Returns labels and centroids matching the exact feature set.
    Raises KeyError if the feature combination is not found in the file.
    """
    path = Path(path) if isinstance(path, str) else path
    feat_key = _features_hash(features)
    
    if not path.exists():
        raise FileNotFoundError(f"Cluster cache file not found: {path}")
    
    data = serialization.load_hdf5_dict(path)
    
    labels_key = f"labels_{feat_key}"
    centroids_key = f"centroids_{feat_key}"
    
    if labels_key not in data or centroids_key not in data:
        raise KeyError(
            f"No cached results for k={k} with features {features}. "
            f"Available keys in {path}: {list(data.keys())}"
        )
    
    return data[labels_key], data[centroids_key]
