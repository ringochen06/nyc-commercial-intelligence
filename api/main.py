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

from api.cluster_helpers import (
    _cluster_rich_description,
    _find_cluster_reps,
    _find_elbow,
    _find_elbow_curvature_knee,
    _get_required_features,
    _zscore,
)
from api.loaders import (
    CDTA_GEO_JSON,
    load_cdta_bounds,
    load_cdta_geojson,
    load_features,
)
from api.rank_helpers import (
    _build_sql,
    _clean_for_json,
    _interpolate_sql,
    _rank_via_supabase,
    _supabase_client,
    _top5_markdown,
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

_EXCLUDED_ACTIVITY_COLUMNS = {
    "act_NO_BUSINESS_ACTIVITY_IDENTIFIED_storefront",
    "act_UNKNOWN_storefront",
}


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
        c
        for c in df.columns
        if c.startswith("act_")
        and c.endswith("_storefront")
        and c not in _EXCLUDED_ACTIVITY_COLUMNS
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
    df_master = load_features(req.vintage)

    # Prepend required activity density features to the request features.
    required_features = _get_required_features(df_master)
    # Remove duplicates while preserving order: required features first, then optional.
    # Ignore act_*_storefront count columns so clustering uses category densities only.
    optional_features = [
        f
        for f in req.features
        if f not in required_features
        and not (f.startswith("act_") and f.endswith("_storefront"))
    ]
    all_features = required_features + optional_features

    df = df_master
    if req.boroughs:
        df = df_master[df_master["borough"].isin(req.boroughs)].copy()

    missing = [f for f in all_features if f not in df.columns]
    if missing:
        raise HTTPException(400, f"Unknown feature columns: {missing}")

    df_clean = df.dropna(subset=all_features).reset_index(drop=True)
    n = len(df_clean)
    if n < 4:
        raise HTTPException(422, f"Only {n} neighborhoods after filtering — need ≥4 to cluster.")

    X_raw = df_clean[all_features].values.astype(float)
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
                raw={f: float(row[f]) for f in all_features},
            )
        )

    summaries: list[ClusterSummary] = []
    for c in range(viz_k):
        size = int((viz_labels == c).sum())
        member_df = df_clean[viz_labels == c]
        reps = _find_cluster_reps(df_master, df_clean, viz_labels, c)
        description = _cluster_rich_description(c, viz_centroids[c], all_features, df_master, member_df, reps)
        summaries.append(
            ClusterSummary(
                cluster=c,
                size=size,
                description=description,
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
        features=all_features,
        feature_means=[float(v) for v in mean],
        feature_stds=[float(v) for v in std],
        points=points,
        centroids_z=[[float(v) for v in row] for row in viz_centroids],
        cluster_summaries=summaries,
    )


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
    return FilterResponse(
        rows=rows,
        n_total=len(df_full),
        n_filtered=len(df_filtered),
        sql=_interpolate_sql(sql, params),
    )


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

    return RankResponse(
        rows=rows,
        n_total=len(df_full),
        n_filtered=len(df_filtered),
        sql=_interpolate_sql(sql, params),
    )


@app.post("/api/agent")
def agent_analysis(req: RankRequest) -> dict:
    """Explain the top 5 soft-ranked neighborhoods. Requires ANTHROPIC_API_KEY."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not set on the server.")

    # Compute the actual blended ranking the same way /api/rank does, so Claude
    # explains the *real* recommendations rather than re-ranking on its own.
    rank_resp = rank(req)
    if not rank_resp.rows:
        raise HTTPException(422, "No neighborhoods passed the hard filters.")

    top5 = rank_resp.rows[:5]
    top5_table = _top5_markdown(rank_resp)
    top5_names = [r.neighborhood for r in top5]

    # Hard-filtered raw rows so the run_sql tool can look up extra detail
    # (foot traffic, activity counts, density, etc.) about the fixed top 5.
    df_full = load_features(req.vintage)
    boroughs_in_data = sorted(df_full["borough"].dropna().unique().tolist())
    con = duckdb.connect()
    con.register("nbhd", df_full)
    sql_text, params = _build_sql(req.filters, boroughs_in_data)
    df_filtered = con.execute(sql_text, params).fetchdf()
    con.close()

    from src.agent import run_agent

    prompt = (
        f"The user's query is:\n\n> {req.query}\n\n"
        f"The semantic + competitive blended ranker has already produced the "
        f"final top-5 recommendations. These are FIXED and FINAL — your job is "
        f"only to explain why each one matches the user's query, **in the exact "
        f"order given**. Do not re-rank, drop, replace, or reorder them. Do not "
        f"suggest other neighborhoods.\n\n"
        f"## Top 5 (final, fixed, ordered)\n\n"
        f"{top5_table}\n\n"
        f"## Your task\n\n"
        f"Identify the **key intent words** in the user's query (e.g. 'restaurant', "
        f"'family', 'tech', 'quiet', 'walkable') and map them to the most relevant "
        f"feature columns (e.g. 'restaurant' → `act_FOOD_SERVICES_storefront`, "
        f"`food_services`, `avg_pedestrian`, `competitive_score`). For each of the "
        f"5 neighborhoods (in the same order, highest blended_score → lowest):\n"
        f"  1. Render the neighborhood name as a bold markdown header (numbered 1–5).\n"
        f"  2. Write a 2–3 sentence explanation that grounds the recommendation in "
        f"specific data points from this CDTA — foot traffic, dominant business "
        f"categories from the act_*_storefront columns, density metrics, the "
        f"competitive_score, and any other meaningful columns that connect to "
        f"the user's intent.\n"
        f"  3. Tie each explanation directly to the user's query — name the intent "
        f"words and the columns you used.\n\n"
        f"Avoid generic statements. Every claim must reference a concrete number or "
        f"category from the data.\n\n"
        f"You may call the `run_sql` tool to look up additional detail about any of "
        f"the 5 neighborhoods — but ONLY for retrieving more context, NEVER for "
        f"ranking, reordering, filtering, or replacing the fixed top 5. The "
        f"`neighborhoods` table you query is the hard-filtered set ({len(df_filtered)} rows). "
        f"When you're done, call the `done` tool with the final markdown answer "
        f"as a numbered list (1–5) with bold neighborhood-name headers.\n\n"
        f"The 5 neighborhoods you must explain, in this exact order: "
        f"{', '.join(top5_names)}."
    )
    try:
        answer = run_agent(prompt, df_filtered, max_turns=20)
    except Exception as e:
        raise HTTPException(500, f"Claude agent error: {e}")
    return {"answer": answer, "n_filtered": len(df_filtered)}
