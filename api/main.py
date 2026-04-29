"""
FastAPI app exposing the NYC commercial intelligence pipeline.

Run locally:
    uvicorn api.main:app --reload --port 8000

Deploy on Railway:
    web: uvicorn api.main:app --host 0.0.0.0 --port $PORT

CORS origins are taken from the env var FRONTEND_ORIGINS (comma-separated).
Defaults to localhost dev + a "*" fallback only when no env var is set.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import silhouette_score as sklearn_silhouette_score
from sklearn.preprocessing import MinMaxScaler

# Make src/ importable so wrapped modules find their relative imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from api.loaders import (
    CDTA_GEO_JSON,
    load_cdta_bounds,
    load_cdta_geojson,
    load_features,
)
from api.schemas import (
    ClusterPoint,
    ClusterRequest,
    ClusterResponse,
    ClusterSummary,
    FeatureRange,
    FeatureRangesResponse,
    FilterRequest,
    FilterResponse,
    RankRequest,
    RankResponse,
    RankRow,
    Vintage,
)

# Import wrapped modules.
from src.embeddings import (  # noqa: E402
    cosine_similarity,
    embed_neighborhood_features,
    embed_texts,
)
from src.kmeans_numpy import (  # noqa: E402
    compute_inertia,
    kmeans_plus_plus,
    silhouette_score,
)

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)


# ── App + CORS ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NYC Commercial Intelligence API",
    version="0.1.0",
    description="Backend for the NYC Commercial Intelligence dashboard (clustering + ranking).",
)

_origins_env = os.getenv("FRONTEND_ORIGINS", "").strip()
if _origins_env:
    _origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    _origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────


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


def _cluster_brief(centroid: np.ndarray, features: list[str], hi_thr: float = 0.5, lo_thr: float = -0.5) -> str:
    vals = np.asarray(centroid, dtype=float)
    if vals.size == 0:
        return "Balanced profile (no clear dominant signals)."
    order_hi = np.argsort(-vals)
    order_lo = np.argsort(vals)
    hi = [features[i] for i in order_hi if vals[i] >= hi_thr][:3]
    lo = [features[i] for i in order_lo if vals[i] <= lo_thr][:2]
    fmt = lambda n: n.replace("_", " ")
    hi_txt, lo_txt = ", ".join(fmt(x) for x in hi), ", ".join(fmt(x) for x in lo)
    if hi and lo:
        return f"Above average on {hi_txt}; relatively lower on {lo_txt}."
    if hi:
        return f"Above average on {hi_txt}."
    if lo:
        return f"No feature is strongly above the filtered-set average; relatively lower on {lo_txt}."
    return "Mid-range on all selected features for this filter."


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "has_cdta_geojson": CDTA_GEO_JSON.is_file(),
        "has_anthropic_key": bool(os.getenv("ANTHROPIC_API_KEY")),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "supabase_configured": bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
    }


@app.get("/api/feature-ranges", response_model=FeatureRangesResponse)
def feature_ranges(vintage: Vintage = "present") -> FeatureRangesResponse:
    df = load_features(vintage)

    numeric_cols = [
        "subway_station_count",
        "avg_pedestrian",
        "storefront_density_per_km2",
        "storefront_filing_count",
        "commercial_activity_score",
        "competitive_score",
        "shooting_incident_count",
        "transit_activity_score",
        "category_entropy",
        "category_diversity",
        "peak_pedestrian",
        "subway_density_per_km2",
        "total_jobs",
    ]

    ranges: dict[str, FeatureRange] = {}
    for col in numeric_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty:
                ranges[col] = FeatureRange(min=float(vals.min()), max=float(vals.max()))

    # NFH columns are optional (depend on whether the NFH raw CSV was present at pipeline time).
    has_nfh_goal4 = (
        "nfh_goal4_fin_shocks_score" in df.columns
        and df["nfh_goal4_fin_shocks_score"].notna().any()
    )
    has_nfh_overall = (
        "nfh_overall_score" in df.columns and df["nfh_overall_score"].notna().any()
    )
    if has_nfh_goal4:
        v = pd.to_numeric(df["nfh_goal4_fin_shocks_score"], errors="coerce").dropna()
        ranges["nfh_goal4_fin_shocks_score"] = FeatureRange(min=float(v.min()), max=float(v.max()))
    if has_nfh_overall:
        v = pd.to_numeric(df["nfh_overall_score"], errors="coerce").dropna()
        ranges["nfh_overall_score"] = FeatureRange(min=float(v.min()), max=float(v.max()))

    activity_columns = sorted(
        c for c in df.columns if c.startswith("act_") and c.endswith("_storefront")
    )

    return FeatureRangesResponse(
        boroughs=sorted(df["borough"].dropna().unique().tolist()),
        ranges=ranges,
        has_nfh_goal4=bool(has_nfh_goal4),
        has_nfh_overall=bool(has_nfh_overall),
        activity_columns=activity_columns,
    )


@app.get("/api/geo/cdta")
def geo_cdta() -> dict:
    """CDTA boundary GeoJSON. Cached. Returns empty FeatureCollection if shapefile missing."""
    geojson = load_cdta_geojson()
    minx, miny, maxx, maxy = load_cdta_bounds()
    return {
        "geojson": geojson,
        "bounds": {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy},
        "center": {"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2},
    }


@app.post("/api/cluster", response_model=ClusterResponse)
def cluster(req: ClusterRequest) -> ClusterResponse:
    df = load_features(req.vintage)
    if req.boroughs:
        df = df[df["borough"].isin(req.boroughs)].copy()

    missing = [f for f in req.features if f not in df.columns]
    if missing:
        raise HTTPException(400, f"Unknown feature columns: {missing}")

    df_clean = df.dropna(subset=req.features).reset_index(drop=True)
    n = len(df_clean)
    if n < 4:
        raise HTTPException(422, f"Only {n} neighborhoods after filtering — need ≥4 to cluster.")

    X_raw = df_clean[req.features].values.astype(float)
    X, mean, std = _zscore(X_raw)

    effective_max_k = min(req.max_k, n - 1)
    k_range = list(range(2, effective_max_k + 1))

    inertias: list[float] = []
    sil_numpy: list[float] = []
    sil_sklearn: list[float] = []
    # Cache labels/centroids from the sweep so viz_k doesn't need a second run.
    _sweep_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in k_range:
        labels, centroids, _ = kmeans_plus_plus(X, k, random_state=req.random_state)
        _sweep_cache[k] = (labels, centroids)
        inertias.append(compute_inertia(X, labels, centroids))
        sil_numpy.append(silhouette_score(X, labels))
        sil_sklearn.append(float(sklearn_silhouette_score(X, labels)))

    elbow_k = _find_elbow(k_range, inertias)
    elbow_k_kneedle = _find_elbow_curvature_knee(k_range, inertias)
    best_sil_k = k_range[int(np.argmax(sil_sklearn))]

    if req.chosen_k is not None:
        viz_k = int(req.chosen_k)
        if viz_k not in k_range:
            raise HTTPException(
                422,
                f"chosen_k must be one of {k_range} after filtering (n={n}); got {viz_k}.",
            )
    else:
        viz_k = int(elbow_k)

    viz_labels, viz_centroids = _sweep_cache[viz_k]

    # Build per-row points with raw feature values + cd/borough for the map.
    points: list[ClusterPoint] = []
    for i, row in df_clean.iterrows():
        cd = row.get("cd")
        borough = row.get("borough")
        map_key = f"{cd} | {borough}" if pd.notna(cd) and pd.notna(borough) else None
        points.append(
            ClusterPoint(
                neighborhood=str(row["neighborhood"]),
                cd=str(cd) if pd.notna(cd) else None,
                borough=str(borough) if pd.notna(borough) else None,
                map_key=map_key,
                cluster=int(viz_labels[i]),
                raw={f: float(row[f]) for f in req.features},
            )
        )

    summaries: list[ClusterSummary] = []
    for c in range(viz_k):
        size = int((viz_labels == c).sum())
        summaries.append(
            ClusterSummary(
                cluster=c,
                size=size,
                description=_cluster_brief(viz_centroids[c], req.features),
                centroid_z=[float(v) for v in viz_centroids[c]],
            )
        )

    return ClusterResponse(
        k_range=k_range,
        inertias=[float(v) for v in inertias],
        silhouettes_numpy=[float(v) for v in sil_numpy],
        silhouettes_sklearn=[float(v) for v in sil_sklearn],
        elbow_k=int(elbow_k),
        elbow_k_kneedle=int(elbow_k_kneedle),
        best_silhouette_k=int(best_sil_k),
        chosen_k=int(viz_k),
        features=req.features,
        feature_means=[float(v) for v in mean],
        feature_stds=[float(v) for v in std],
        points=points,
        centroids_z=[[float(v) for v in row] for row in viz_centroids],
        cluster_summaries=summaries,
    )


def _build_sql(filters, boroughs_in_data: list[str]) -> tuple[str, list]:
    """Return (sql, parameters) using DuckDB-style ? bindings for safety."""
    where: list[str] = []
    params: list = []

    boroughs = filters.boroughs if filters.boroughs else boroughs_in_data
    placeholders = ", ".join(["?"] * len(boroughs))
    where.append(f"borough IN ({placeholders})")
    params.extend(boroughs)

    if filters.min_subway_stations is not None:
        where.append("subway_station_count >= ?")
        params.append(float(filters.min_subway_stations))
    if filters.min_avg_pedestrian is not None:
        where.append("avg_pedestrian >= ?")
        params.append(float(filters.min_avg_pedestrian))
    if filters.min_storefront_density is not None:
        where.append("storefront_density_per_km2 >= ?")
        params.append(float(filters.min_storefront_density))
    if filters.min_storefront_filings is not None:
        where.append("storefront_filing_count >= ?")
        params.append(float(filters.min_storefront_filings))
    if filters.min_commercial_activity is not None:
        where.append("commercial_activity_score >= ?")
        params.append(float(filters.min_commercial_activity))
    if filters.max_competitive_score is not None:
        where.append("competitive_score <= ?")
        params.append(float(filters.max_competitive_score))
    if filters.max_shooting_incident_count is not None:
        where.append("shooting_incident_count <= ?")
        params.append(float(filters.max_shooting_incident_count))
    if filters.min_nfh_goal4 is not None:
        where.append("nfh_goal4_fin_shocks_score >= ?")
        params.append(float(filters.min_nfh_goal4))
    if filters.min_nfh_overall is not None:
        where.append("nfh_overall_score >= ?")
        params.append(float(filters.min_nfh_overall))

    sql = (
        "SELECT * FROM nbhd WHERE "
        + " AND ".join(where)
        + " ORDER BY commercial_activity_score DESC"
    )
    return sql, params


def _supabase_client():
    """Return a Supabase client if env vars are set, else None. Lazy-imports supabase-py."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    from supabase import create_client  # imported lazily so the dep is optional

    return create_client(url, key)


def _rank_via_supabase(req: RankRequest, client) -> RankResponse:
    """Query Supabase via the match_neighborhoods RPC, then blend in Python."""
    # Per-category competitive scoring needs act_*_storefront columns the RPC
    # doesn't return — let the caller fall back to the CSV path in that case.
    if req.competitive_source and req.competitive_source != "__overall__":
        raise NotImplementedError(
            "competitive_source != '__overall__' requires the CSV path (per-category act_* data)."
        )

    query_embedding = embed_texts([req.query])[0].tolist()

    # match_count default in the SQL is 50; bump to 200 to comfortably cover all CDTAs.
    rpc_args = {
        "query_embedding": query_embedding,
        "boroughs": req.filters.boroughs or None,
        "min_subway_station_count": (
            int(req.filters.min_subway_stations) if req.filters.min_subway_stations is not None else None
        ),
        "min_avg_pedestrian": req.filters.min_avg_pedestrian,
        "min_storefront_density": req.filters.min_storefront_density,
        "min_storefront_filing_count": (
            int(req.filters.min_storefront_filings) if req.filters.min_storefront_filings is not None else None
        ),
        "min_commercial_activity": req.filters.min_commercial_activity,
        "max_competitive_score": req.filters.max_competitive_score,
        "max_shooting_incident_count": (
            int(req.filters.max_shooting_incident_count)
            if req.filters.max_shooting_incident_count is not None
            else None
        ),
        "min_nfh_goal4_score": req.filters.min_nfh_goal4,
        "min_nfh_overall_score": req.filters.min_nfh_overall,
        "match_count": 200,
    }
    resp = client.rpc("match_neighborhoods", rpc_args).execute()
    rows_raw: list[dict] = resp.data or []

    if not rows_raw:
        return RankResponse(rows=[], n_total=0, n_filtered=0, sql="-- via supabase RPC --")

    sims = np.array([float(r["similarity"]) for r in rows_raw], dtype=float)
    competitive = np.array(
        [float(r.get("competitive_score") or 0.0) for r in rows_raw],
        dtype=float,
    )

    # Higher competition penalises rank, so negate before MinMax (matches the CSV path).
    X = np.column_stack([sims, -competitive])
    if X.shape[0] == 1:
        scaled = np.ones((1, 2)) * 0.5
    else:
        scaled = MinMaxScaler().fit_transform(X)
    alpha, beta = req.alpha, 1.0 - req.alpha
    final = scaled @ np.array([alpha, beta], dtype=float)

    order = np.argsort(-final)
    cmap = req.cluster_assignments or {}
    bmap = req.cluster_briefs or {}
    out: list[RankRow] = []
    for rank_pos, idx in enumerate(order, start=1):
        r = rows_raw[idx]
        name = r["neighborhood"]
        cd = r.get("cd")
        borough = r.get("borough")
        cluster_id = cmap.get(name)
        out.append(
            RankRow(
                rank=rank_pos,
                neighborhood=name,
                cd=cd,
                borough=borough,
                map_key=f"{cd} | {borough}" if cd and borough else None,
                semantic_similarity=float(sims[idx]),
                specific_competitive_score=float(competitive[idx]),
                blended_score=float(final[idx]),
                cluster=int(cluster_id) if cluster_id is not None else None,
                cluster_description=bmap.get(str(cluster_id)) if cluster_id is not None else None,
            )
        )
    return RankResponse(rows=out, n_total=len(rows_raw), n_filtered=len(rows_raw), sql="-- via supabase RPC --")


def _clean_for_json(rows: list[dict]) -> list[dict]:
    """Replace NaN / inf with None and downcast numpy scalars so FastAPI can serialise."""
    out: list[dict] = []
    for row in rows:
        clean: dict = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                fv = float(v)
                clean[k] = fv if np.isfinite(fv) else None
            elif v is pd.NA:
                clean[k] = None
            else:
                try:
                    if pd.isna(v):
                        clean[k] = None
                        continue
                except (TypeError, ValueError):
                    pass
                clean[k] = v
        out.append(clean)
    return out


@app.post("/api/filter", response_model=FilterResponse)
def filter_endpoint(req: FilterRequest) -> FilterResponse:
    """Apply the hard filters via DuckDB and return the raw filtered rows.

    Powers the "Hard-filtered neighborhoods" preview on the Ranking page —
    independent of the semantic ranking pipeline.
    """
    df_full = load_features(req.vintage)
    boroughs_in_data = sorted(df_full["borough"].dropna().unique().tolist())

    con = duckdb.connect()
    con.register("nbhd", df_full)
    sql, params = _build_sql(req.filters, boroughs_in_data)
    df_filtered = con.execute(sql, params).fetchdf()
    con.close()

    rows = _clean_for_json(df_filtered.to_dict(orient="records"))
    return FilterResponse(rows=rows, n_total=len(df_full), n_filtered=len(df_filtered), sql=sql)


@app.post("/api/rank", response_model=RankResponse)
def rank(req: RankRequest) -> RankResponse:
    # Prefer Supabase when configured.
    client = _supabase_client()
    if client is not None:
        try:
            return _rank_via_supabase(req, client)
        except Exception as e:
            logger.warning("Supabase rank failed, falling back to CSV: %s", e)

    # Fallback: CSV + DuckDB + cached embeddings.
    df_full = load_features(req.vintage)
    boroughs_in_data = sorted(df_full["borough"].dropna().unique().tolist())

    con = duckdb.connect()
    con.register("nbhd", df_full)
    sql, params = _build_sql(req.filters, boroughs_in_data)
    df_filtered = con.execute(sql, params).fetchdf()
    con.close()

    if df_filtered.empty:
        return RankResponse(rows=[], n_total=len(df_full), n_filtered=0, sql=sql)

    try:
        all_embeddings, _all_texts = embed_neighborhood_features()
    except Exception as e:
        raise HTTPException(503, f"Embeddings unavailable: {e}. Run `python -m src.embeddings` or set OPENAI_API_KEY.")

    full_neighborhoods = df_full["neighborhood"].tolist()
    idx_map = {name: i for i, name in enumerate(full_neighborhoods)}
    keep_names = [n for n in df_filtered["neighborhood"].tolist() if n in idx_map]
    if not keep_names:
        raise HTTPException(500, "No filtered neighborhoods could be aligned with embeddings.")

    filtered_indices = [idx_map[n] for n in keep_names]
    filtered_embeddings = all_embeddings[filtered_indices]

    query_embedding = embed_texts([req.query])[0]
    sim_scores = cosine_similarity(query_embedding, filtered_embeddings)

    source = req.competitive_source or "__overall__"
    if source == "__overall__":
        if "competitive_score" not in df_filtered.columns:
            raise HTTPException(500, "competitive_score column missing from filtered data.")
        competitive = np.array(
            [
                float(df_filtered.loc[df_filtered["neighborhood"] == n, "competitive_score"].iloc[0])
                for n in keep_names
            ],
            dtype=float,
        )
    else:
        if not (source.startswith("act_") and source.endswith("_storefront")):
            raise HTTPException(
                400,
                f"Invalid competitive_source '{source}'. Use '__overall__' or an act_*_storefront column.",
            )
        if source not in df_filtered.columns:
            raise HTTPException(400, f"Column '{source}' missing from feature table.")
        counts = np.array(
            [
                float(df_filtered.loc[df_filtered["neighborhood"] == n, source].iloc[0])
                for n in keep_names
            ],
            dtype=float,
        )
        ped_scores = np.array(
            [
                float(df_filtered.loc[df_filtered["neighborhood"] == n, "avg_pedestrian"].iloc[0])
                for n in keep_names
            ],
            dtype=float,
        )
        competitive = np.log1p(np.maximum(counts / (ped_scores + 1.0), 0.0))

    # Higher competition should lower rank, so use the negated competition signal.
    X = np.column_stack([sim_scores.astype(float), -competitive])
    if X.shape[0] == 1:
        scaled = np.ones((1, 2)) * 0.5
    else:
        scaled = MinMaxScaler().fit_transform(X)

    alpha, beta = req.alpha, 1.0 - req.alpha
    final_scores = scaled @ np.array([alpha, beta], dtype=float)

    order = np.argsort(-final_scores)
    rows: list[RankRow] = []
    cmap = req.cluster_assignments or {}
    bmap = req.cluster_briefs or {}
    for rank_pos, idx in enumerate(order, start=1):
        name = keep_names[idx]
        meta = df_filtered.loc[df_filtered["neighborhood"] == name].iloc[0]
        cd = meta.get("cd")
        borough = meta.get("borough")
        cluster_id = cmap.get(name)
        rows.append(
            RankRow(
                rank=rank_pos,
                neighborhood=name,
                cd=str(cd) if pd.notna(cd) else None,
                borough=str(borough) if pd.notna(borough) else None,
                map_key=f"{cd} | {borough}" if pd.notna(cd) and pd.notna(borough) else None,
                semantic_similarity=float(sim_scores[idx]),
                specific_competitive_score=float(competitive[idx]),
                blended_score=float(final_scores[idx]),
                cluster=int(cluster_id) if cluster_id is not None else None,
                cluster_description=bmap.get(str(cluster_id)) if cluster_id is not None else None,
            )
        )

    return RankResponse(rows=rows, n_total=len(df_full), n_filtered=len(df_filtered), sql=sql)


@app.post("/api/agent")
def agent_analysis(req: RankRequest) -> dict:
    """Optional Claude analysis of filtered data. Requires ANTHROPIC_API_KEY on the server."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not set on the server.")

    df_full = load_features(req.vintage)
    boroughs_in_data = sorted(df_full["borough"].dropna().unique().tolist())
    con = duckdb.connect()
    con.register("nbhd", df_full)
    sql, params = _build_sql(req.filters, boroughs_in_data)
    df_filtered = con.execute(sql, params).fetchdf()
    con.close()

    if df_filtered.empty:
        raise HTTPException(422, "No neighborhoods passed the hard filters.")

    from src.agent import run_agent

    prompt = (
        f"The user is looking for: {req.query}\n\n"
        f"There are {len(df_filtered)} neighborhoods that passed the hard filters. "
        f"Use the run_sql tool to explore the data and recommend the top 3-5 "
        f"neighborhoods that best match the user's soft preferences. "
        f"Explain your reasoning with specific data points."
    )
    try:
        answer = run_agent(prompt, df_filtered, max_turns=20)
    except Exception as e:
        raise HTTPException(500, f"Claude agent error: {e}")
    return {"answer": answer, "n_filtered": len(df_filtered)}
