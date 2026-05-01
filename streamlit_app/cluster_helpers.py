"""Cluster calculations and embedding summaries for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.cluster_helpers import (
    _cluster_brief_description,
    _cluster_rich_description,
    _find_elbow,
    _find_elbow_curvature_knee,
)
from src.embeddings import cosine_similarity, load_embeddings


_EXCLUDED_OPTIONAL_DENSITY_FEATURES = {
    "act_NO_BUSINESS_ACTIVITY_IDENTIFIED_density",
    "act_UNKNOWN_density",
}


def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return (arr - mean) / (std + 1e-8)


def find_elbow(k_range: list[int], inertias: list[float]) -> int:
    return _find_elbow(k_range, inertias)


def find_elbow_curvature_knee(k_range: list[int], inertias: list[float]) -> int:
    return _find_elbow_curvature_knee(k_range, inertias)


def clustering_density_feature_options(df: pd.DataFrame) -> list[str]:
    return sorted(
        col
        for col in df.columns
        if col.startswith("act_")
        and col.endswith("_density")
        and col not in _EXCLUDED_OPTIONAL_DENSITY_FEATURES
    )


def cluster_brief_description(
    centroid: np.ndarray,
    features: list[str],
    *,
    hi_thr: float = 0.5,
    lo_thr: float = -0.5,
) -> str:
    return _cluster_brief_description(
        centroid,
        features,
        hi_thr=hi_thr,
        lo_thr=lo_thr,
    )


def cluster_semantics_from_embeddings(
    df_master: pd.DataFrame,
    df_clustered: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    centroids: np.ndarray | None = None,
    features: list[str] | None = None,
    *,
    top_n: int = 3,
    text_max_len: int = 420,
) -> list[dict[str, object]] | None:
    """Per-cluster representatives using cached neighborhood embeddings."""
    loaded = load_embeddings()
    if loaded is None:
        return None
    emb_all, texts_all = loaded
    n_master = len(df_master)
    if emb_all.shape[0] != n_master or len(texts_all) != n_master:
        return None

    name_to_row = {str(n): i for i, n in enumerate(df_master["neighborhood"].tolist())}
    names_rows = df_clustered["neighborhood"].astype(str).tolist()
    lab = labels.astype(int, copy=False)
    rows_out: list[dict[str, object]] = []

    for c in range(k):
        pairs: list[tuple[str, int]] = []
        for i in range(len(lab)):
            if int(lab[i]) != c:
                continue
            nm = names_rows[i]
            if nm not in name_to_row:
                continue
            pairs.append((nm, name_to_row[nm]))

        if not pairs:
            rows_out.append({"cluster": c, "n": 0, "reps": [], "description": ""})
            continue

        row_idx = np.array([p[1] for p in pairs], dtype=int)
        Xc = emb_all[row_idx].astype(np.float32, copy=False)
        mean_v = Xc.mean(axis=0).astype(np.float32, copy=False)
        sims = cosine_similarity(mean_v, Xc)
        order = np.argsort(-sims)
        take = min(top_n, len(order))
        reps: list[dict[str, object]] = []

        for j in range(take):
            li = int(order[j])
            r = int(row_idx[li])
            txt = str(texts_all[r])
            if len(txt) > text_max_len:
                txt = txt[: text_max_len - 1] + "..."
            reps.append(
                {
                    "neighborhood": pairs[li][0],
                    "cosine_to_mean": float(sims[li]),
                    "profile_excerpt": txt,
                }
            )

        description = ""
        if centroids is not None and features is not None:
            member_mask = lab == c
            description = _cluster_rich_description(
                c,
                centroids[c],
                features,
                df_master,
                df_clustered.loc[member_mask],
                reps,
            )
        rows_out.append(
            {
                "cluster": c,
                "n": len(pairs),
                "reps": reps,
                "description": description,
            }
        )

    return rows_out
