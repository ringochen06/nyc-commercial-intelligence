"""SQL building, Supabase integration, and ranking helpers."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from api.schemas import HardFilters, RankRequest, RankResponse, RankRow
from src.embeddings import embed_texts


def _build_sql(filters: HardFilters, boroughs_in_data: list[str]) -> tuple[str, list]:
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


def _interpolate_sql(sql: str, params: list) -> str:
    """Inline ``?`` placeholders with their literal SQL values for display only.

    DuckDB execution still uses the parameterized form; this is purely so the
    user can read the actual query in the UI instead of '?' placeholders.
    """
    parts = sql.split("?")
    if len(parts) - 1 != len(params):
        return sql
    out: list[str] = [parts[0]]
    for i, value in enumerate(params):
        if value is None:
            literal = "NULL"
        elif isinstance(value, bool):
            literal = "TRUE" if value else "FALSE"
        elif isinstance(value, (int,)):
            literal = str(value)
        elif isinstance(value, float):
            literal = f"{int(value)}" if float(value).is_integer() else f"{value:g}"
        else:
            s = str(value).replace("'", "''")
            literal = f"'{s}'"
        out.append(literal)
        out.append(parts[i + 1])
    return "".join(out)


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


def _top5_markdown(rank_resp: RankResponse) -> str:
    """Render the top 5 ranked rows as a markdown table for Claude's prompt."""
    header = (
        "| # | neighborhood | semantic_similarity | specific_competitive_score | blended_score |\n"
        "|---|---|---|---|---|\n"
    )
    body = "\n".join(
        f"| {row.rank} | {row.neighborhood} | {row.semantic_similarity:.4f} | "
        f"{row.specific_competitive_score:.4f} | {row.blended_score:.4f} |"
        for row in rank_resp.rows[:5]
    )
    return header + body
