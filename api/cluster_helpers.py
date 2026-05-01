"""Cluster analysis helpers: elbow detection, z-scoring, and rich cluster descriptions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from api.formatting import (
    _activity_label_from_col,
    _display_borough,
    _fmt_list,
    _level_from_percentile,
    _percentile_rank,
    _pretty_feature,
    _series_max,
    _series_sum,
)
from src.embeddings import cosine_similarity, load_embeddings


_EXCLUDED_REQUIRED_DENSITY_FEATURES = {
    "act_NO_BUSINESS_ACTIVITY_IDENTIFIED_density",
    "act_UNKNOWN_density",
}


def _zscore(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return (arr - mean) / (std + 1e-8), mean, std


def _find_elbow(k_range: list[int], inertias: list[float]) -> int:
    ks = np.array(k_range, dtype=float)
    ys = np.array(inertias, dtype=float)
    ks_n = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
    ys_n = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    dx = ks_n[-1] - ks_n[0]
    dy = ys_n[-1] - ys_n[0]
    norm = np.sqrt(dx**2 + dy**2) + 1e-12
    distances = np.abs(dy * ks_n - dx * ys_n + ks_n[-1] * ys_n[0] - ys_n[-1] * ks_n[0]) / norm
    return k_range[int(np.argmax(distances))]


def _find_elbow_curvature_knee(k_range: list[int], inertias: list[float]) -> int:
    """Alternative elbow: k at largest |Δ²(inertia)| on the inertia curve (normalized)."""
    ys = np.asarray(inertias, dtype=float)
    ks = np.asarray(k_range, dtype=float)
    if ks.size < 3:
        return int(k_range[0])
    yn = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    d2 = np.diff(yn, n=2)
    if d2.size == 0:
        return int(k_range[len(k_range) // 2])
    j = int(np.argmax(np.abs(d2)))
    mid = min(max(j + 1, 0), len(k_range) - 1)
    return int(k_range[mid])


def _get_required_features(df: pd.DataFrame) -> list[str]:
    """Get activity density features that are required for clustering."""
    return sorted(
        col
        for col in df.columns
        if col.startswith("act_")
        and col.endswith("_density")
        and col not in _EXCLUDED_REQUIRED_DENSITY_FEATURES
    )


def _has_no_commercial_activity(member_df: pd.DataFrame) -> bool:
    return (
        _series_sum(member_df, "storefront_filing_count") <= 0
        and _series_max(member_df, "commercial_activity_score") <= 0
        and _series_max(member_df, "competitive_score") <= 0
    )


def _no_commercial_activity_mask(member_df: pd.DataFrame) -> pd.Series:
    idx = member_df.index
    filings = pd.to_numeric(
        member_df.get("storefront_filing_count", pd.Series(0, index=idx)),
        errors="coerce",
    ).fillna(0)
    commercial = pd.to_numeric(
        member_df.get("commercial_activity_score", pd.Series(0, index=idx)),
        errors="coerce",
    ).fillna(0)
    competitive = pd.to_numeric(
        member_df.get("competitive_score", pd.Series(0, index=idx)),
        errors="coerce",
    ).fillna(0)
    return (filings <= 0) & (commercial <= 0) & (competitive <= 0)


def _activity_category_profile(
    member_df: pd.DataFrame,
    df_master: pd.DataFrame,
    centroid: np.ndarray,
    features: list[str],
) -> str:
    count_cols = [
        col
        for col in member_df.columns
        if str(col).startswith("act_") and str(col).endswith("_storefront")
    ]
    if not count_cols:
        return ""

    z = {features[i]: float(centroid[i]) for i in range(min(len(features), len(centroid)))}
    density_cols = [
        col for col in df_master.columns if str(col).startswith("act_") and str(col).endswith("_density")
    ]

    totals = member_df[count_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
    total_filings = _series_sum(member_df, "storefront_filing_count")
    denom = total_filings if total_filings > 0 else float(totals.sum())

    # (col, mean_density, pct, share) — sorted by actual density value, not percentile rank
    density_ranked: list[tuple[str, float, int, int]] = []
    for col in count_cols:
        count_total = float(pd.to_numeric(member_df[col], errors="coerce").fillna(0).sum())
        if count_total <= 0:
            continue
        density_col = str(col).removesuffix("_storefront") + "_density"
        if density_col not in member_df.columns or density_col not in df_master.columns:
            continue
        mean_density = float(pd.to_numeric(member_df[density_col], errors="coerce").mean())
        pct = _percentile_rank(df_master[density_col], mean_density)
        if pct is not None:
            share = int(round(100.0 * count_total / max(denom, 1.0)))
            density_ranked.append((str(col), mean_density, pct, share))

    if not density_ranked:
        return ""

    density_ranked.sort(key=lambda item: item[1], reverse=True)
    top_category_bits = [
        f"{_activity_label_from_col(col)}—{share}% of filings, {pct}th pct density"
        for col, _, pct, share in density_ranked[:3]
    ]

    elevated = sorted(
        [
            (_activity_label_from_col(col), val)
            for col, val in z.items()
            if col in density_cols and val >= 0.45
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:4]
    low = sorted(
        [
            (_activity_label_from_col(col), val)
            for col, val in z.items()
            if col in density_cols and val <= -0.45
        ],
        key=lambda item: item[1],
    )[:3]

    bits = [
        "Granular storefront mix: "
        + f"top categories by density are {_fmt_list(top_category_bits, limit=3)}."
    ]
    if elevated:
        bits.append(
            "Unusually elevated category densities include "
            + _fmt_list([f"{label} ({val:.2f})" for label, val in elevated], limit=4)
            + "."
        )
    if low:
        bits.append(
            "Comparatively sparse categories include "
            + _fmt_list([f"{label} ({val:.2f})" for label, val in low], limit=3)
            + "."
        )
    return " ".join(bits)


def _cluster_title(centroid: np.ndarray, features: list[str]) -> str:
    z = {features[i]: float(centroid[i]) for i in range(min(len(features), len(centroid)))}
    dense = max(
        z.get("storefront_density_per_km2", -9.0),
        z.get("storefront_filing_count", -9.0),
        z.get("commercial_activity_score", -9.0),
    )
    transit = max(z.get("transit_activity_score", -9.0), z.get("subway_station_count", -9.0))
    foot = z.get("avg_pedestrian", -9.0)
    jobs = z.get("total_jobs", -9.0)
    entropy = max(z.get("category_entropy", -9.0), z.get("category_diversity", -9.0))
    nfh = max(z.get("nfh_overall_score", -9.0), z.get("nfh_goal4_fin_shocks_score", -9.0))

    
    if dense >= 0.8 and foot >= 0.4:
        return "Dense Mixed-Use Commercial Core" if entropy >= 0 else "High-Traffic Commercial Core"
    if transit >= 0.8 and dense >= 0:
        return "Transit-Oriented Commercial District"
    if jobs >= 0.8 and dense < 0.5:
        return "Employment-Heavy Business District"
    if nfh >= 0.7 and dense <= 0:
        return "Stable Lower-Density Neighborhood Market"
    if dense <= -0.5 and transit <= 0:
        return "Lower-Density Local Commercial Area"
    return "Balanced Neighborhood Market"


def _cluster_brief_description(
    centroid: np.ndarray,
    features: list[str],
    *,
    hi_thr: float = 0.5,
    lo_thr: float = -0.5,
) -> str:
    vals = np.asarray(centroid, dtype=float)
    if vals.size == 0:
        return "Balanced profile (no clear dominant signals)."
    order_hi = np.argsort(-vals)
    order_lo = np.argsort(vals)
    hi = [features[i] for i in order_hi if vals[i] >= hi_thr][:3]
    lo = [features[i] for i in order_lo if vals[i] <= lo_thr][:2]
    hi_txt = ", ".join(_pretty_feature(x) for x in hi)
    lo_txt = ", ".join(_pretty_feature(x) for x in lo)
    if hi and lo:
        return f"Above average on {hi_txt}; relatively lower on {lo_txt}."
    if hi:
        return f"Above average on {hi_txt}."
    if lo:
        return f"No feature is strongly above the filtered-set average (z < {hi_thr:g}); relatively lower on {lo_txt}."
    return f"Mid-range on all selected features for this filter (no z ≥ {hi_thr:g} or z ≤ {lo_thr:g} at the cluster centroid)."


def _find_cluster_reps(
    df_master: pd.DataFrame,
    df_clustered: pd.DataFrame,
    labels: np.ndarray,
    c: int,
    *,
    top_n: int = 3,
    text_max_len: int = 420,
) -> list[dict[str, object]]:
    """Return representative neighborhoods for cluster c using cached embeddings."""
    loaded = load_embeddings()
    if loaded is None:
        return []
    emb_all, texts_all = loaded
    if emb_all.shape[0] != len(df_master) or len(texts_all) != len(df_master):
        return []
    name_to_row = {str(n): i for i, n in enumerate(df_master["neighborhood"].tolist())}
    names_rows = df_clustered["neighborhood"].astype(str).tolist()
    lab = labels.astype(int, copy=False)
    pairs: list[tuple[str, int]] = []
    for i in range(len(lab)):
        if int(lab[i]) != c:
            continue
        nm = names_rows[i]
        if nm not in name_to_row:
            continue
        pairs.append((nm, name_to_row[nm]))
    if not pairs:
        return []
    row_idx = np.array([p[1] for p in pairs], dtype=int)
    Xc = emb_all[row_idx].astype(np.float32, copy=False)
    mean_v = Xc.mean(axis=0).astype(np.float32, copy=False)
    sims = cosine_similarity(mean_v, Xc)
    order = np.argsort(-sims)
    reps: list[dict[str, object]] = []
    for j in range(min(top_n, len(order))):
        li = int(order[j])
        r = int(row_idx[li])
        txt = str(texts_all[r])
        if len(txt) > text_max_len:
            txt = txt[: text_max_len - 1] + "…"
        reps.append({
            "neighborhood": pairs[li][0],
            "cosine_to_mean": float(sims[li]),
            "profile_excerpt": txt,
        })
    return reps


def _cluster_rich_description(
    c: int,
    centroid: np.ndarray,
    features: list[str],
    df_master: pd.DataFrame,
    member_df: pd.DataFrame,
    reps: list[dict[str, object]],
) -> str:
    if member_df.empty:
        return _cluster_brief_description(centroid, features)

    no_commercial_activity = _has_no_commercial_activity(member_df)
    no_activity_mask = _no_commercial_activity_mask(member_df)
    no_activity_count = int(no_activity_mask.sum())
    no_activity_share = no_activity_count / max(len(member_df), 1)
    no_activity_names = set(
        member_df.loc[no_activity_mask, "neighborhood"].astype(str).str.lower()
        if "neighborhood" in member_df.columns
        else []
    )
    rep_no_activity_count = sum(
        1 for rep in reps if str(rep.get("neighborhood", "")).lower() in no_activity_names
    )
    prominent_no_activity = (
        no_commercial_activity
        or no_activity_share >= 0.5
        or (bool(reps) and rep_no_activity_count >= min(2, len(reps)))
    )
    title = (
        "No Recorded Commercial Activity Zone"
        if no_commercial_activity
        else "Mostly No-Activity / Special-Use Geography"
        if prominent_no_activity
        else _cluster_title(centroid, features)
    )

    def _fmt_feature_name(name: str) -> str:
        if name.startswith("act_"):
            return _activity_label_from_col(name)
        return _pretty_feature(name)

    vals = np.asarray(centroid, dtype=float)
    order_hi = np.argsort(-vals)
    order_lo = np.argsort(vals)
    hi = [_fmt_feature_name(features[i]) for i in order_hi if vals[i] >= 0.45][:4]
    lo = [_fmt_feature_name(features[i]) for i in order_lo if vals[i] <= -0.45][:3]

    parts: list[str] = [f"Cluster {c} - {title}."]
    if no_commercial_activity:
        parts.append(
            "No member neighborhoods have recorded non-vacant storefront filings, and both commercial activity and competitive scores are 0; treat this as a non-commercial or special-use geography rather than a conventional retail market."
        )
    elif prominent_no_activity:
        parts.append(
            f"{no_activity_count} of {len(member_df)} member neighborhoods have 0 non-vacant storefront filings plus commercial activity and competitive scores of 0; representative matches may therefore describe parks, open space, waterfront, or other special-use geographies rather than retail markets."
        )
    if hi:
        parts.append(f"Characterized by elevated {_fmt_list(hi, limit=4)}.")
    if lo:
        parts.append(f"Relatively lower on {_fmt_list(lo, limit=3)}.")

    metric_bits: list[str] = []
    for col, label in [
        ("storefront_density_per_km2", "business density"),
        ("avg_pedestrian", "foot traffic"),
        ("category_entropy", "activity-mix diversity"),
        ("transit_activity_score", "transit activity"),
    ]:
        if col not in member_df.columns or col not in df_master.columns:
            continue
        mean_val = float(pd.to_numeric(member_df[col], errors="coerce").mean())
        pct = _percentile_rank(df_master[col], mean_val)
        if pct is not None:
            metric_bits.append(
                f"{_level_from_percentile(pct)} {label} (around the {pct}th percentile citywide)"
            )
    if metric_bits:
        parts.append("Profile signals include " + "; ".join(metric_bits[:3]) + ".")

    if not prominent_no_activity:
        category_profile = _activity_category_profile(member_df, df_master, centroid, features)
        if category_profile:
            parts.append(category_profile)

    boroughs = [
        _display_borough(name)
        for name in member_df["borough"].astype(str).value_counts().head(3).index.tolist()
    ]
    rep_names = [str(rep.get("neighborhood", "")) for rep in reps]
    place_bits = []
    if boroughs:
        place_bits.append(f"Concentrated in {_fmt_list(boroughs, limit=3)}")
    if rep_names:
        place_bits.append(f"nearest text-profile matches include {_fmt_list(rep_names, limit=3)}")
    if place_bits:
        parts.append("; ".join(place_bits) + ".")

    return " ".join(parts)
