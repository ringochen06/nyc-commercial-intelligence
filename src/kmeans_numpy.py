"""
K-means clustering implemented from scratch in NumPy (course requirement).

Use Euclidean distance, iterative centroid updates, and convergence via max centroid move or iteration cap.
Do not delegate clustering to scikit-learn here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

CLUSTER_PATH = Path("outputs/clusters/")


def _save_hdf5_dict(data: dict, path: Path) -> None:
    """Save a dictionary of arrays and scalars to an HDF5 file, overwriting existing keys.

    Opens the file in append mode (``'a'``), so existing datasets for other
    keys are preserved. If a key already exists it is deleted before writing,
    ensuring the stored value is always the most-recent one.  The parent
    directory is created automatically if it does not exist.

    Parameters
    ----------
    data : dict
        Mapping of string keys to values.  Supported value types:
        ``np.ndarray``, ``list``, ``tuple`` (converted to ``np.ndarray``),
        and Python scalars (``str``, ``int``, ``float``, ``bool``).  Any
        other type is coerced via ``np.array()``.
    path : Path
        Filesystem path to the target HDF5 file.  Created if absent.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    """
    if h5py is None:
        raise ImportError("h5py is required for cluster caching. pip install h5py")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        for key, value in data.items():
            if key in f:
                del f[key]
            if isinstance(value, np.ndarray):
                f.create_dataset(key, data=value)
            elif isinstance(value, (list, tuple)):
                f.create_dataset(key, data=np.array(value))
            elif isinstance(value, (str, int, float, bool)):
                f.create_dataset(key, data=value)
            else:
                f.create_dataset(key, data=np.array(value))


def _load_hdf5_dict(path: Path) -> dict:
    """Load all datasets from an HDF5 file into a plain Python dictionary.

    Each top-level key in the HDF5 file becomes a dictionary entry.  Values
    are read back as NumPy arrays (or scalars for zero-dimensional datasets)
    via ``dataset[()]``.

    Parameters
    ----------
    path : Path
        Filesystem path to an existing HDF5 file.

    Returns
    -------
    dict
        Mapping of HDF5 dataset names to their NumPy-array contents.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    FileNotFoundError
        If the file at ``path`` does not exist.
    """
    if h5py is None:
        raise ImportError("h5py is required for cluster caching. pip install h5py")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    data: dict = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data[key] = f[key][()]
    return data

def _features_hash(features: list[str]) -> str:
    """Generate a short, order-independent hash from a list of feature names.

    Sorts the feature names before hashing so that identical sets provided
    in different orders produce the same key.  Used as the suffix that
    distinguishes HDF5 datasets written for different feature selections
    within the same ``k`` cache file.

    Parameters
    ----------
    features : list[str]
        Feature column names used for a clustering run.

    Returns
    -------
    str
        An 8-character lowercase hexadecimal MD5 digest of the
        pipe-joined, sorted feature names.
    """
    sorted_features = sorted(features)
    combined = "|".join(sorted_features)
    return hashlib.md5(combined.encode()).hexdigest()[:8]

def pairwise_squared_euclidean(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances between two sets of points.

    Uses NumPy broadcasting: expands ``X`` to shape ``(n, 1, d)`` and
    ``C`` to ``(1, k, d)`` so that the element-wise difference is
    ``(n, k, d)``; squaring the L2 norm along the feature axis gives
    the final ``(n, k)`` distance matrix without an explicit Python loop.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Query points (data matrix).
    C : np.ndarray, shape (k, d)
        Reference points (e.g. cluster centroids).

    Returns
    -------
    np.ndarray, shape (n, k)
        ``D[i, j]`` is the squared Euclidean distance from ``X[i]`` to ``C[j]``.
    """
    return np.linalg.norm(X[:, np.newaxis] - C, axis=2) ** 2


def kmeans_plus_plus(
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run K-means with K-means++ centroid initialisation.

    K-means++ selects initial centroids with probability proportional to
    their squared distance to the nearest already-chosen centroid, reducing
    the chance of poor convergence compared to random initialisation.
    After seeding, the algorithm iterates standard assign → recompute steps
    until either ``max_iter`` is reached or the maximum centroid displacement
    falls below ``tol``.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix of ``n`` data points in ``d``-dimensional space.
        Should be z-score normalised before passing in.
    k : int
        Number of clusters to form.
    max_iter : int, optional
        Maximum number of assign–recompute iterations.  Default ``100``.
    tol : float, optional
        Convergence threshold on the L2 norm of the centroid shift between
        successive iterations.  Default ``1e-4``.
    random_state : int | None, optional
        Seed for ``np.random.default_rng``.  Pass an integer for
        reproducible results; ``None`` uses a random seed.

    Returns
    -------
    labels : np.ndarray, shape (n,)
        Integer cluster index (0 to k-1) for each data point.
    centroids : np.ndarray, shape (k, d)
        Final centroid coordinates in the z-scored feature space.
    n_iter : int
        Number of iterations actually performed (1-indexed; equals
        ``max_iter`` if convergence was not reached).
    """
    
    rng = np.random.default_rng(random_state)
    initial_centroid = X[rng.choice(X.shape[0])]
    centroids = [initial_centroid]

    for c in range(1, k):
        dist = pairwise_squared_euclidean(X, np.array(centroids))  # (n, c)
        min_dist_to_centroid = np.min(dist, axis=1)  # (n,)
        probs = min_dist_to_centroid / np.sum(min_dist_to_centroid)
        next_centroid = X[rng.choice(X.shape[0], p=probs)]
        centroids.append(next_centroid)
    centroids = np.array(centroids)  # (k, d)

    i = 0
    for i in range(max_iter):
        z = assign_labels(X, centroids)
        new_centroids = update_centroids(X, z, k, rng)
        if np.linalg.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return z, centroids, i + 1
    
def kmeans_plus_plus_with_caching(
    features: list[str],
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run K-means with K-means++ initialisation, caching results to avoid recomputation.

    Caches results in an HDF5 file per ``k`` (``outputs/clusters/kmeans_k{k}.h5``),
    with datasets keyed by a hash of the feature names.  On a cache hit the
    stored labels, centroids, and iteration count are returned immediately
    without touching ``X``.  On a cache miss, ``kmeans_plus_plus`` is called
    and the results are persisted before returning.  Multiple feature
    selections for the same ``k`` coexist within a single HDF5 file.

    Parameters
    ----------
    features : list[str]
        Feature column names used to generate the cache key.  Must match
        the columns whose values appear (in the same column order) in ``X``.
    X : np.ndarray, shape (n, d)
        Feature matrix used for clustering.  Ignored on a cache hit.
    k : int
        Number of clusters to form.
    max_iter : int, optional
        Maximum assign–recompute iterations passed to ``kmeans_plus_plus``.
        Default ``100``.
    tol : float, optional
        Convergence tolerance passed to ``kmeans_plus_plus``.  Default ``1e-4``.
    random_state : int | None, optional
        Random seed passed to ``kmeans_plus_plus``.  Default ``None``.

    Returns
    -------
    labels : np.ndarray, shape (n,)
        Cluster index (0 to k-1) for each data point.
    centroids : np.ndarray, shape (k, d)
        Final centroid coordinates.
    n_iter : int
        Iterations run by the K-means algorithm, or ``0`` if the result
        was loaded from cache.
    """
    CLUSTER_PATH.mkdir(parents=True, exist_ok=True)
    cache_file = CLUSTER_PATH / f"kmeans_k{k}.h5"
    
    # Try to load from cache
    try:
        labels, centroids, iterations = load_kmeans_results(k, features, cache_file)
        return labels, centroids, 0  # 0 iterations indicates loaded from cache
    except (FileNotFoundError, KeyError):
        pass  # Cache miss, proceed with computation
    
    # Compute new results
    labels, centroids, n_iter = kmeans_plus_plus(X, k, max_iter=max_iter, tol=tol, random_state=random_state)
    save_kmeans_results(k, features, labels, centroids, n_iter, cache_file)
    return labels, centroids, n_iter


def assign_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each data point to its nearest centroid.

    Computes the full ``(n, k)`` pairwise squared-Euclidean distance matrix
    via ``pairwise_squared_euclidean``, then takes the column-wise argmin to
    produce a hard cluster assignment for every point.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data points to assign.
    centroids : np.ndarray, shape (k, d)
        Current centroid positions.

    Returns
    -------
    np.ndarray, shape (n,), dtype int
        Cluster index in ``[0, k)`` for each row of ``X``.
    """
    distances = pairwise_squared_euclidean(X, centroids)
    return np.argmin(distances, axis=1)


def update_centroids(X: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Recompute cluster centroids as the mean of their assigned points.

    Uses a binary indicator matrix of shape ``(k, n)`` for a vectorised
    sum-then-divide rather than a Python loop over clusters.  Empty clusters
    (which can arise in degenerate configurations) are re-initialised to a
    randomly selected data point so that ``k`` centroids are always returned
    and downstream steps never encounter NaN means.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Full data matrix.
    labels : np.ndarray, shape (n,)
        Cluster assignments produced by ``assign_labels``.
    k : int
        Total number of clusters, including any that may be empty.
    rng : np.random.Generator | None, optional
        NumPy random generator used to sample replacement points for empty
        clusters.  A new generator with an arbitrary seed is created
        internally when ``None``.

    Returns
    -------
    np.ndarray, shape (k, d)
        Updated centroid coordinates.  Each row is either the mean of all
        points assigned to that cluster or a randomly chosen data point if
        the cluster was empty.
    """
    indicator = np.array([labels == j for j in range(k)], dtype=float)  # (k, n)
    counts = np.sum(indicator, axis=1)  # (k,)
    empty = np.where(counts == 0)[0]
    if len(empty) > 0:
        _rng = rng if rng is not None else np.random.default_rng()
        reinit_indices = _rng.choice(X.shape[0], size=len(empty), replace=False)
        #indicator[empty] = 0.0
        indicator[empty, reinit_indices] = 1.0
        counts[empty] = 1.0
    centroid_sums = indicator @ X
    return centroid_sums / counts[:, np.newaxis]  # (k, n) @ (n, d) -> (k, d)





def compute_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Compute within-cluster sum of squared distances (WCSS / inertia).

    Uses advanced indexing (``centroids[labels]``) to gather each point's
    assigned centroid in a single vectorised operation, then sums the
    element-wise squared differences across all points and features.
    Lower inertia indicates tighter, more compact clusters.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data matrix.
    labels : np.ndarray, shape (n,)
        Cluster assignment for each point (output of ``assign_labels``).
    centroids : np.ndarray, shape (k, d)
        Current centroid positions.

    Returns
    -------
    float
        Total within-cluster sum of squared Euclidean distances.
    """
    return float(np.sum((X - centroids[labels]) ** 2))


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute the mean silhouette coefficient over all samples (pure NumPy).

    The silhouette coefficient for sample ``i`` is defined as::

        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    where ``a(i)`` is the mean Euclidean distance from ``i`` to every other
    point in the same cluster (intra-cluster cohesion) and ``b(i)`` is the
    mean distance from ``i`` to every point in the nearest neighbouring
    cluster (inter-cluster separation).  A score near ``+1`` means the
    sample is well inside its own cluster; a score near ``-1`` means it
    would fit better in another cluster; ``0`` indicates overlap.

    The full ``(n, n)`` pairwise distance matrix is precomputed once via
    broadcasting and reused for all per-sample lookups.  Singleton clusters
    (only one point) receive ``a(i) = 0``; if the denominator is zero the
    coefficient is set to ``0.0`` to avoid division by zero.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data matrix (same space used for clustering).
    labels : np.ndarray, shape (n,)
        Cluster assignment for each point.

    Returns
    -------
    float
        Mean silhouette coefficient across all ``n`` samples, in ``[-1, 1]``.
        Returns ``0.0`` when fewer than two distinct clusters are present.
    """
    n = X.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=2))

    s = np.zeros(n)
    for i in range(n):
        c = labels[i]
        same_mask = labels == c
        same_mask[i] = False
        cluster_size = np.sum(same_mask)

        a_i = D[i, same_mask].mean() if cluster_size > 0 else 0.0

        b_i = np.inf
        for c_other in unique_labels:
            if c_other == c:
                continue
            other_mask = labels == c_other
            mean_dist = D[i, other_mask].mean()
            if mean_dist < b_i:
                b_i = mean_dist

        denom = max(a_i, b_i)
        s[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(np.mean(s))



def save_kmeans_results(k: int, features: list[str], labels: np.ndarray, centroids: np.ndarray, n_iterations: int, path: Path | str) -> None:
    """Persist clustering results to an HDF5 file keyed by feature selection.

    Writes four datasets to the file, each suffixed with an 8-character hash
    of the sorted ``features`` list so that results for different feature
    combinations can coexist inside the same ``k``-specific file without
    collision.  Existing entries for the same feature hash are overwritten.

    Dataset names written (where ``<hash>`` = ``_features_hash(features)``):

    * ``features_<hash>``  — object array of feature name strings
    * ``labels_<hash>``    — integer cluster-label array
    * ``centroids_<hash>`` — float centroid matrix
    * ``n_iterations_<hash>`` — single-element int array holding iteration count

    Parameters
    ----------
    k : int
        Number of clusters (used only to compose the default filename in
        callers; not written into the file itself).
    features : list[str]
        Feature column names that produced these results.  Determines the
        cache key; order does not matter (names are sorted before hashing).
    labels : np.ndarray, shape (n,)
        Cluster assignment for each data point.
    centroids : np.ndarray, shape (k, d)
        Final centroid coordinates.
    n_iterations : int
        Number of K-means iterations performed.
    path : Path | str
        Destination HDF5 file path.  Created if absent; parent directories
        are created automatically.

    Returns
    -------
    None
    """
    path = Path(path) if isinstance(path, str) else path
    feat_key = _features_hash(features)
    
    data = {
        f"features_{feat_key}": np.array(features, dtype=object),
        f"labels_{feat_key}": labels,
        f"centroids_{feat_key}": centroids,
        f"n_iterations_{feat_key}": np.array([n_iterations]),
    }
    _save_hdf5_dict(data, path)

def load_kmeans_results(k: int, features: list[str], path: Path | str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load persisted clustering results from an HDF5 file by ``k`` and feature set.

    Derives the dataset key from ``_features_hash(features)`` and looks up
    the corresponding ``labels_*``, ``centroids_*``, and ``n_iterations_*``
    datasets in the file.  The function is the inverse of ``save_kmeans_results``.

    Parameters
    ----------
    k : int
        Number of clusters (informational; used in the error message to aid
        debugging; not read from the file).
    features : list[str]
        Feature column names whose hash identifies the desired result set.
        Must match the list passed to ``save_kmeans_results`` (order-independent).
    path : Path | str
        Path to the HDF5 cache file produced by ``save_kmeans_results``.

    Returns
    -------
    labels : np.ndarray, shape (n,)
        Cluster assignment for each data point.
    centroids : np.ndarray, shape (k, d)
        Centroid coordinates.
    n_iter : int
        Number of K-means iterations that were performed when the result
        was originally computed.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist at ``path``.
    KeyError
        If the file exists but contains no entry for the given feature hash
        (i.e. this feature combination has not been cached yet).
    """
    path = Path(path) if isinstance(path, str) else path
    feat_key = _features_hash(features)
    
    if not path.exists():
        raise FileNotFoundError(f"Cluster cache file not found: {path}")
    
    data = _load_hdf5_dict(path)
    
    labels_key = f"labels_{feat_key}"
    centroids_key = f"centroids_{feat_key}"
    iterations_key = f"n_iterations_{feat_key}"
    
    if labels_key not in data or centroids_key not in data:
        raise KeyError(
            f"No cached results for k={k} with features {features}. "
            f"Available keys in {path}: {list(data.keys())}"
        )
    
    return data[labels_key], data[centroids_key], data[iterations_key][0]

