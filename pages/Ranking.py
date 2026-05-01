"""
NYC Commercial Intelligence — Streamlit dashboard.

Hard constraints  : sidebar controls → DuckDB SQL filters (deterministic)
Soft ranking      : α·semantic + β·(negative competitive penalty)
                    (MinMax-scaled to [0,1] on the filtered set; β = 1 − α from one slider)
Semantic search   : embedding cosine similarity on neighborhood profiles (OpenAI or sentence-transformers)
Optional Claude   : read-only SQL on filtered data

Opened from the sidebar (**Ranking**) when you run `streamlit run app.py` (home = K-Selection / clustering).
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# Repo root is one level above pages/
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from agent import run_agent  # noqa: E402
from embeddings import (  # noqa: E402
    cosine_similarity,
    embed_neighborhood_features,
    embed_texts,
)
from config import (  # noqa: E402
    CDTA_SHAPE_PATH,
    load_cdta_gdf_for_map,
    load_neighborhood_features,
)
from feature_engineering import is_act_storefront_column  # noqa: E402

DEFAULT_SOFT_QUERY = (
    "quiet residential area suitable for retail with good subway access and good NFH stability"
)
DEFAULT_ALPHA_SEMANTIC = 0.8

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ranking · NYC Commercial Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Ranking")
st.caption(
    "Hard filters (DuckDB SQL), then α·semantic + β·competitive score "
    "(MinMax on the filtered set). Optional Claude. "
    "**Cluster** columns appear when you have run **K-Selection Analysis** on the home page."
)

# ── Load data ───────────────────────────────────────────────────────────────

df_full = load_neighborhood_features()
SHOOTING_COUNT_COL = (
    "shooting_incident_count"
    if "shooting_incident_count" in df_full.columns
    else (
        "shooting_incident_count_2024"
        if "shooting_incident_count_2024" in df_full.columns
        else None
    )
)

# ── Sidebar: Hard Filters ──────────────────────────────────────────────────

st.sidebar.header("Hard Filters")
st.sidebar.caption(
    "Deterministic constraints applied via DuckDB SQL before the model sees the data."
)

# Borough
boroughs = sorted(df_full["borough"].unique().tolist())
selected_boroughs = st.sidebar.multiselect(
    "Borough",
    options=boroughs,
    default=boroughs,
    help="Select one or more boroughs to include.",
)

# Subway station count
subway_min, subway_max = int(df_full["subway_station_count"].min()), int(
    df_full["subway_station_count"].max()
)
subway_range = st.sidebar.slider(
    "Min subway stations",
    min_value=subway_min,
    max_value=subway_max,
    value=subway_min,
    help="Minimum number of subway stations in the neighborhood.",
)

# Average pedestrian traffic
ped_min_val = int(df_full["avg_pedestrian"].min())
ped_max_val = int(df_full["avg_pedestrian"].max())
ped_threshold = st.sidebar.slider(
    "Min avg pedestrian count",
    min_value=ped_min_val,
    max_value=ped_max_val,
    value=ped_min_val,
    help="Minimum average pedestrian foot traffic.",
)

# Storefront filing density
density_min = float(df_full["storefront_density_per_km2"].min())
density_max = float(df_full["storefront_density_per_km2"].max())
density_threshold = st.sidebar.slider(
    "Min storefront density (per km\u00b2)",
    min_value=density_min,
    max_value=density_max,
    value=density_min,
    step=0.5,
    help="Minimum non-vacant storefront filings per km\u00b2 (CDTA area).",
)

# Total storefront filings
poi_min = int(df_full["storefront_filing_count"].min())
poi_max = int(df_full["storefront_filing_count"].max())
poi_threshold = st.sidebar.slider(
    "Min storefront filing count",
    min_value=poi_min,
    max_value=poi_max,
    value=poi_min,
    help="Minimum count of storefront filings (non-vacant) in the CDTA.",
)

# Commercial activity score (log1p scale — step must be << range or Streamlit slider breaks)
comm_min = float(
    pd.to_numeric(df_full["commercial_activity_score"], errors="coerce").min()
)
comm_max = float(
    pd.to_numeric(df_full["commercial_activity_score"], errors="coerce").max()
)
if not np.isfinite(comm_min):
    comm_min = 0.0
if not np.isfinite(comm_max):
    comm_max = 1.0
if comm_max <= comm_min:
    comm_max = comm_min + 1e-3
comm_span = comm_max - comm_min
comm_step = max(0.001, min(0.5, round(comm_span / 200.0, 6)))
comm_threshold = st.sidebar.slider(
    "Min commercial activity score",
    min_value=comm_min,
    max_value=comm_max,
    value=comm_min,
    step=comm_step,
    help="Minimum commercial activity score: log1p(storefront filings × avg pedestrian).",
)

# Competitive score upper bound (higher is worse; keep below this cap)
comp_min = float(pd.to_numeric(df_full["competitive_score"], errors="coerce").min())
comp_max = float(pd.to_numeric(df_full["competitive_score"], errors="coerce").max())
if not np.isfinite(comp_min):
    comp_min = 0.0
if not np.isfinite(comp_max):
    comp_max = 1.0
if comp_max <= comp_min:
    comp_max = comp_min + 1e-3
comp_span = comp_max - comp_min
comp_step = max(0.001, min(0.5, round(comp_span / 200.0, 6)))
comp_max_threshold = st.sidebar.slider(
    "Max competitive score",
    min_value=comp_min,
    max_value=comp_max,
    value=comp_max,
    step=comp_step,
    help="Upper bound on competition pressure: log1p(storefront filings / (avg pedestrian + 1)).",
)

# Crime upper bound
crime_max_threshold = None
if SHOOTING_COUNT_COL is not None:
    crime_min = float(pd.to_numeric(df_full[SHOOTING_COUNT_COL], errors="coerce").min())
    crime_max = float(pd.to_numeric(df_full[SHOOTING_COUNT_COL], errors="coerce").max())
    crime_max_threshold = st.sidebar.slider(
        "Max shooting incident count",
        min_value=crime_min,
        max_value=crime_max,
        value=crime_max,
        step=1.0,
        help="Upper bound on total NYPD shooting incidents per neighborhood.",
    )

# NFH minimums (same style as other hard filters; only shown when columns exist)
has_nfh_shocks = "nfh_goal4_fin_shocks_score" in df_full.columns
nfh_goal4_threshold = None
if has_nfh_shocks and df_full["nfh_goal4_fin_shocks_score"].notna().any():
    nfh_goal4_vals = pd.to_numeric(
        df_full["nfh_goal4_fin_shocks_score"], errors="coerce"
    ).dropna()
    if not nfh_goal4_vals.empty:
        nfh_goal4_threshold = st.sidebar.slider(
            "Min NFH Goal 4 score (Financial Shocks)",
            min_value=float(nfh_goal4_vals.min()),
            max_value=float(nfh_goal4_vals.max()),
            value=float(nfh_goal4_vals.min()),
            step=0.1,
            help="Higher values indicate stronger financial-shock resilience in this index.",
        )

has_nfh_overall = "nfh_overall_score" in df_full.columns
nfh_overall_threshold = None
if has_nfh_overall and df_full["nfh_overall_score"].notna().any():
    nfh_overall_vals = pd.to_numeric(
        df_full["nfh_overall_score"], errors="coerce"
    ).dropna()
    if not nfh_overall_vals.empty:
        nfh_overall_threshold = st.sidebar.slider(
            "Min NFH Overall index score",
            min_value=float(nfh_overall_vals.min()),
            max_value=float(nfh_overall_vals.max()),
            value=float(nfh_overall_vals.min()),
            step=0.1,
            help="Higher values indicate stronger overall neighborhood financial health (relative index).",
        )

# ── Apply hard filters with DuckDB ─────────────────────────────────────────


def _interpolate_sql(sql: str, params: list) -> str:
    """Inline ``?`` placeholders for display only (FastAPI `/api/filter` parity)."""
    parts = sql.split("?")
    if len(parts) - 1 != len(params):
        return sql
    out: list[str] = [parts[0]]
    for i, value in enumerate(params):
        if value is None:
            literal = "NULL"
        elif isinstance(value, bool):
            literal = "TRUE" if value else "FALSE"
        elif isinstance(value, int):
            literal = str(value)
        elif isinstance(value, float):
            literal = f"{int(value)}" if float(value).is_integer() else f"{value:g}"
        else:
            s = str(value).replace("'", "''")
            literal = f"'{s}'"
        out.append(literal)
        out.append(parts[i + 1])
    return "".join(out)


def _top5_markdown_for_agent(blend_df: pd.DataFrame) -> str:
    header = (
        "| # | neighborhood | semantic_similarity | specific_competitive_score | blended_score |\n"
        "|---|---|---|---|---|\n"
    )
    body = "\n".join(
        f"| {int(row['rank'])} | {row['neighborhood']} | {float(row['semantic_similarity']):.4f} | "
        f"{float(row['specific_competitive_score']):.4f} | {float(row['blended_score']):.4f} |"
        for _, row in blend_df.head(5).iterrows()
    )
    return header + body


def _summarize_cluster_description(description: str) -> str:
    normalized = " ".join(str(description or "").split()).strip()
    if not normalized:
        return ""

    title = normalized
    if normalized.lower().startswith("cluster ") and " - " in normalized:
        _, remainder = normalized.split(" - ", 1)
        title = remainder.split(".", 1)[0].strip() or normalized

    characterized_sentence = ""
    marker = "Characterized by elevated "
    start = normalized.find(marker)
    if start != -1:
        end = normalized.find(".", start)
        characterized_sentence = (
            normalized[start : end + 1].strip() if end != -1 else normalized[start:].strip()
        )

    return f"{title}.\n{characterized_sentence}".strip()


def apply_hard_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, str, list]:
    """Parameterized DuckDB query (same binding style as `api.main._build_sql`)."""
    con = duckdb.connect()
    con.register("nbhd", df)

    where: list[str] = []
    params: list = []

    ph = ", ".join(["?"] * len(selected_boroughs))
    where.append(f"borough IN ({ph})")
    params.extend(selected_boroughs)

    where.append("subway_station_count >= ?")
    params.append(float(subway_range))
    where.append("avg_pedestrian >= ?")
    params.append(float(ped_threshold))
    where.append("storefront_density_per_km2 >= ?")
    params.append(float(density_threshold))
    where.append("storefront_filing_count >= ?")
    params.append(float(poi_threshold))
    where.append("commercial_activity_score >= ?")
    params.append(float(comm_threshold))
    where.append("competitive_score <= ?")
    params.append(float(comp_max_threshold))

    if SHOOTING_COUNT_COL is not None and crime_max_threshold is not None:
        where.append(f"{SHOOTING_COUNT_COL} <= ?")
        params.append(float(crime_max_threshold))
    if nfh_goal4_threshold is not None:
        where.append("nfh_goal4_fin_shocks_score >= ?")
        params.append(float(nfh_goal4_threshold))
    if nfh_overall_threshold is not None:
        where.append("nfh_overall_score >= ?")
        params.append(float(nfh_overall_threshold))

    sql = (
        "SELECT * FROM nbhd WHERE "
        + " AND ".join(where)
        + " ORDER BY commercial_activity_score DESC"
    )
    result = con.execute(sql, params).fetchdf()
    con.close()
    return result, sql, params


df_filtered, _hard_sql_template, _hard_sql_params = apply_hard_filters(df_full)

# ── Show hard-filter results ────────────────────────────────────────────────

st.subheader(f"Hard-filtered neighborhoods ({len(df_filtered)} of {len(df_full)})")

with st.expander("View generated SQL", expanded=False):
    st.code(
        _interpolate_sql(_hard_sql_template, _hard_sql_params),
        language="sql",
    )

if df_filtered.empty:
    st.warning(
        "No neighborhoods match the current hard filters. Loosen your constraints."
    )
    st.stop()

# Hard-filter neighborhood search
hard_filter_search = st.text_input(
    "Search neighborhoods",
    key="hard_filter_search",
    help="Filter by neighborhood name",
)

df_hard_display = df_filtered[
    df_filtered["neighborhood"].str.contains(
        hard_filter_search, case=False, na=False
    )
] if hard_filter_search else df_filtered

_act_cols = sorted(c for c in df_hard_display.columns if is_act_storefront_column(c))
display_cols = [
    "neighborhood",
    "borough",
    "storefront_filing_count",
]
for _x in ("category_diversity", "category_entropy"):
    if _x in df_hard_display.columns:
        display_cols.append(_x)
display_cols.extend(_act_cols)
display_cols.extend(
    [
        "subway_station_count",
        "avg_pedestrian",
        "storefront_density_per_km2",
        "commercial_activity_score",
        "competitive_score",
        "transit_activity_score",
    ]
)
if SHOOTING_COUNT_COL is not None:
    display_cols.insert(
        display_cols.index("commercial_activity_score"), SHOOTING_COUNT_COL
    )
for optional_col in ["nfh_overall_score", "nfh_goal4_fin_shocks_score"]:
    if optional_col in df_hard_display.columns:
        display_cols.append(optional_col)
display_cols = [c for c in display_cols if c in df_hard_display.columns]
st.caption(
    f"Showing **{len(_act_cols)}** storefront activity columns (`act_*_storefront`) from your feature table; "
    "semantic profiles include every activity with count > 0 (`src/embeddings.py`)."
)
st.dataframe(
    df_hard_display[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=360,
)

with st.expander("About zeros, nulls, and refreshing data", expanded=False):
    st.markdown(
        """
**Why you used to see nulls (`None`)**

- **NFH columns** (`nfh_*`) only load if the Neighborhood Financial Health CSV is parsed and joined on Community District. Older exports used labels like `BX Community District 8`; the pipeline now normalizes those to `BX08` so scores merge correctly.
- **Profile / NFH numeric gaps** after the CDTA join are filled in the pipeline with **borough median, then citywide median** (a proxy for CDTAs that do not match a single profile row). They are useful for dashboards, not exact tract-level truth.

**`commercial_activity_score` and `transit_activity_score`**

- **`commercial_activity_score` = `log1p`(`storefront_filing_count` × `avg_pedestrian`)** (after filling missing storefront counts with 0 and missing pedestrian averages with the citywide mean, then any remaining NaN pedestrian with 0; linear product clipped at 0 before `log1p`).
- **`transit_activity_score` = `log1p`(`subway_station_count` × `avg_pedestrian`)** with the same pedestrian handling.
- A row can still show **0** when the **linear** product is 0 (no filings, no stations, or zero pedestrian signal after imputation), since **`log1p(0) = 0`**. The log compresses heavy tails so sidebar thresholds and MinMax blending behave more evenly across CDTAs.

**If numbers look stale after changing the pipeline**

- Regenerate with `python run_pipeline.py`, then in Streamlit use **Rerun** or **Clear cache** (menu) so the cached feature table picks up the new `neighborhood_features_final.csv`.
        """
    )

# ── Sidebar: Soft Preferences ──────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.header("Soft Preferences")

if "soft_query_committed" not in st.session_state:
    st.session_state.soft_query_committed = DEFAULT_SOFT_QUERY
if "soft_query_draft" not in st.session_state:
    st.session_state.soft_query_draft = st.session_state.soft_query_committed

with st.sidebar.form("soft_preferences_form"):
    st.caption(
        "Edit your description, then click **Update soft ranking** to refresh semantic similarity "
        "(and the blended table). In a multi-line box, use the button; "
        "you can also focus the button and press Enter."
    )
    st.text_area(
        "Describe your ideal area",
        height=100,
        key="soft_query_draft",
    )
    submitted_soft = st.form_submit_button("Update soft ranking")

if submitted_soft:
    st.session_state.soft_query_committed = st.session_state.soft_query_draft

soft_query = st.session_state.soft_query_committed

st.sidebar.markdown("**Blend** (α + β = 1)")
alpha = st.sidebar.slider(
    "\u03b1 — semantic similarity",
    0.0,
    1.0,
    DEFAULT_ALPHA_SEMANTIC,
    0.05,
    help="Weight on embedding cosine similarity; β = 1 − α goes to competitive_score.",
)
beta = 1.0 - alpha
_w = np.array([alpha, beta], dtype=float)
st.sidebar.caption(
    f"\u03b2 — competitive score = 1 − \u03b1 → \u03b1={alpha:.3f}, \u03b2={beta:.3f}"
)

# ── Competitive score source selector ───────────────────────────────────────

_activity_count_cols = sorted(c for c in df_full.columns if is_act_storefront_column(c))


def _activity_label(col: str) -> str:
    return (
        col.removeprefix("act_").removesuffix("_storefront").replace("_", " ").title()
    )


_competitive_source_options: list[str] = ["__overall__"] + _activity_count_cols


def _competitive_source_label(source: str) -> str:
    if source == "__overall__":
        return "Overall (all storefront filings)"
    return _activity_label(source)


selected_competitive_source = st.selectbox(
    "Competitive score source",
    options=_competitive_source_options,
    format_func=_competitive_source_label,
    help=(
        "Choose which storefront count to use in competitive_score. "
        "Formula: log1p(count / (avg_pedestrian + 1))."
    ),
)

# ── Soft ranking: semantic + competitive penalty (aligned with `/api/rank`) ──

st.subheader("Soft ranking (α·semantic + β·competitive score)")
st.caption(
    "Semantic similarity uses the **submitted** text from the sidebar (after **Update soft ranking**). "
    "Changing α still updates the blend immediately. "
    "**Cluster** / **Cluster description** use the latest **K-Selection Analysis** on the home page "
    "(same session). Neighborhoods not in that run show empty cluster cells."
)
st.caption(
    f"Current competitive source: **{_competitive_source_label(selected_competitive_source)}** "
    "(computed as `log1p(count / (avg_pedestrian + 1))` on filtered rows)."
)


@st.cache_data(show_spinner="Loading neighborhood embeddings...")
def get_all_embeddings():
    return embed_neighborhood_features()


try:
    all_embeddings, all_texts = get_all_embeddings()

    full_neighborhoods = df_full["neighborhood"].tolist()
    filtered_neighborhoods = df_filtered["neighborhood"].tolist()
    idx_map = {name: i for i, name in enumerate(full_neighborhoods)}
    filtered_indices = [idx_map[n] for n in filtered_neighborhoods if n in idx_map]

    if filtered_indices:
        filtered_embeddings = all_embeddings[filtered_indices]
        query_embedding = embed_texts([soft_query])[0]
        sim_scores = cosine_similarity(query_embedding, filtered_embeddings)

        keep_names = [n for n in filtered_neighborhoods if n in idx_map]
        if "competitive_score" not in df_filtered.columns:
            raise KeyError("Column competitive_score missing from feature table.")

        comp_scores = np.array(
            [
                float(
                    df_filtered.loc[
                        df_filtered["neighborhood"] == n,
                        (
                            "storefront_filing_count"
                            if selected_competitive_source == "__overall__"
                            else selected_competitive_source
                        ),
                    ].iloc[0]
                )
                for n in keep_names
            ],
            dtype=float,
        )
        ped_scores = np.array(
            [
                float(
                    df_filtered.loc[
                        df_filtered["neighborhood"] == n, "avg_pedestrian"
                    ].iloc[0]
                )
                for n in keep_names
            ],
            dtype=float,
        )
        comp_scores = np.log1p(np.maximum(comp_scores / (ped_scores + 1.0), 0.0))

        # Higher competition should hurt ranking, so blend with the negated signal.
        X = np.column_stack([sim_scores.astype(float), -comp_scores])
        if X.shape[0] == 1:
            scores_scaled = np.ones((1, 2)) * 0.5
        else:
            scores_scaled = MinMaxScaler().fit_transform(X)

        final_scores = scores_scaled @ _w

        ranking_df = (
            pd.DataFrame(
                {
                    "neighborhood": keep_names,
                    "semantic_similarity": sim_scores.round(4),
                    "specific_competitive_score": comp_scores.round(3),
                    "blended_score": final_scores.round(4),
                }
            )
            .sort_values("blended_score", ascending=False)
            .reset_index(drop=True)
        )

        ranking_df.index = ranking_df.index + 1
        ranking_df.index.name = "rank"

        cmap = st.session_state.get("ks_cluster_by_neighborhood")
        bmap = st.session_state.get("ks_cluster_brief") or {}
        if cmap:
            rk = ranking_df.reset_index()
            rk["cluster"] = rk["neighborhood"].map(cmap)
            rk["cluster_description"] = rk["cluster"].apply(
                lambda x: bmap.get(int(x), "") if pd.notna(x) else ""
            )
            rk["cluster_description"] = rk["cluster_description"].apply(
                _summarize_cluster_description
            )
            rk["cluster"] = (
                rk["cluster"]
                .apply(lambda x: int(x) if pd.notna(x) else pd.NA)
                .astype("Int64")
            )
            show_cols = [
                "rank",
                "neighborhood",
                "cluster",
                "cluster_description",
                "semantic_similarity",
                "specific_competitive_score",
                "blended_score",
            ]
            ranking_df = rk[show_cols].set_index("rank")
        else:
            st.caption(
                "_No cluster labels in this session yet — on **home**, run **K-Selection Analysis** to "
                "fill **cluster** and **cluster_description**._"
            )

        _agent_cols = [
            "rank",
            "neighborhood",
            "semantic_similarity",
            "specific_competitive_score",
            "blended_score",
        ]
        _blend_snap = ranking_df.reset_index()
        st.session_state["rank_agent_blend_df"] = _blend_snap[
            [c for c in _agent_cols if c in _blend_snap.columns]
        ].copy()
        st.session_state["rank_agent_soft_query"] = soft_query

        # Ranking table neighborhood search
        ranking_search = st.text_input(
            "Search ranked neighborhoods",
            key="ranking_search",
            help="Filter by neighborhood name",
        )
        ranking_display_df = ranking_df.reset_index()
        if ranking_search:
            ranking_display_df = ranking_display_df[
                ranking_display_df["neighborhood"].str.contains(
                    ranking_search, case=False, na=False
                )
            ]
        st.dataframe(
            ranking_display_df.set_index("rank"),
            use_container_width=True,
            height=380,
        )

        # NYC map: blended score on CDTA polygons (sequential greens)
        st.markdown(
            "**NYC map** *(blended score — light green = lower, dark green = higher on the filtered set)*"
        )
        if not CDTA_SHAPE_PATH.is_file():
            st.caption(f"No shapefile at `{CDTA_SHAPE_PATH}` — map skipped.")
        else:
            geo_gdf = load_cdta_gdf_for_map(CDTA_SHAPE_PATH)
            shape_geojson = geo_gdf.__geo_interface__
            loc_keys = df_filtered[["neighborhood", "cd", "borough"]].drop_duplicates(
                subset=["neighborhood"], keep="first"
            )
            map_df = ranking_df.reset_index().merge(
                loc_keys, on="neighborhood", how="left"
            )
            map_df["map_key"] = map_df["cd"] + " | " + map_df["borough"]
            map_df["blended_score"] = pd.to_numeric(
                map_df["blended_score"], errors="coerce"
            )
            map_plot = map_df.dropna(
                subset=["map_key", "blended_score", "cd", "borough"]
            )
            n_bad = len(map_df) - len(map_plot)
            if n_bad:
                st.caption(
                    f"{n_bad} ranked row(s) missing borough/CD join for the map were dropped."
                )
            if map_plot.empty:
                st.info("No rows to plot on the map.")
            else:
                zmin = float(map_plot["blended_score"].min())
                zmax = float(map_plot["blended_score"].max())
                if zmax <= zmin:
                    zmax = zmin + 1e-9
                bounds = geo_gdf.geometry.total_bounds
                lon0 = float((bounds[0] + bounds[2]) / 2)
                lat0 = float((bounds[1] + bounds[3]) / 2)
                map_fig = go.Figure(
                    go.Choroplethmapbox(
                        geojson=shape_geojson,
                        locations=map_plot["map_key"],
                        z=map_plot["blended_score"],
                        featureidkey="properties.map_key",
                        colorscale=[
                            [0.0, "#e8f5e9"],
                            [0.35, "#a5d6a7"],
                            [0.65, "#43a047"],
                            [1.0, "#1b5e20"],
                        ],
                        zmin=zmin,
                        zmax=zmax,
                        marker_opacity=0.82,
                        marker_line_width=0.6,
                        marker_line_color="#ffffff",
                        colorbar=dict(
                            title=dict(text="Blended score", side="right"),
                            tickformat=".3f",
                        ),
                        text=map_plot["neighborhood"]
                        + "<br>rank "
                        + map_plot["rank"].astype(str)
                        + "<br>blended "
                        + map_plot["blended_score"].astype(str),
                        hovertemplate="<b>%{text}</b><extra></extra>",
                    )
                )
                map_fig.update_layout(
                    height=480,
                    margin=dict(l=0, r=0, t=8, b=0),
                    mapbox=dict(
                        style="open-street-map",
                        center=dict(lat=lat0, lon=lon0),
                        zoom=9,
                    ),
                )
                st.plotly_chart(map_fig, use_container_width=True)
    else:
        st.session_state.pop("rank_agent_blend_df", None)
        st.session_state.pop("rank_agent_soft_query", None)
        st.info("Could not map filtered neighborhoods to embeddings.")

except Exception as e:
    st.session_state.pop("rank_agent_blend_df", None)
    st.session_state.pop("rank_agent_soft_query", None)
    st.warning(
        f"Semantic search unavailable (embeddings not cached or embedding backend error): {e}\n\n"
        "Run `python -m src.embeddings` to generate embeddings. Default: OpenAI when `OPENAI_API_KEY` is set, else local sentence-transformers; "
        "to force local-only vectors, set `EMBEDDING_BACKEND=sentence_transformers` in `.env` (see `.env.example`)."
    )

# ── Claude agent analysis ──────────────────────────────────────────────────

st.subheader("AI analysis (Claude)")
st.caption(
    "Same contract as FastAPI **`/api/agent`**: Claude explains the **fixed top 5** from the soft ranker "
    "(semantic + competitive blend above). It must **not** re-rank or substitute other neighborhoods."
)

if st.button("Ask Claude to explain top 5 (fixed ranking)", type="primary"):
    with st.spinner("Claude is analyzing the top-ranked neighborhoods..."):
        blend_store = st.session_state.get("rank_agent_blend_df")
        query_for_agent = st.session_state.get("rank_agent_soft_query", soft_query)
        try:
            if blend_store is None or blend_store.empty:
                raise RuntimeError(
                    "No soft ranking snapshot — fix embeddings (see warning above) or widen filters."
                )
            rank_resp_preview = _top5_markdown_for_agent(blend_store)
            top5_names = blend_store.head(5)["neighborhood"].astype(str).tolist()
            prompt = (
                f"The user's query is:\n\n> {query_for_agent}\n\n"
                f"The semantic + competitive blended ranker has already produced the "
                f"final top-5 recommendations. These are FIXED and FINAL — your job is "
                f"only to explain why each one matches the user's query, **in the exact "
                f"order given**. Do not re-rank, drop, replace, or reorder them. Do not "
                f"suggest other neighborhoods.\n\n"
                f"## Top 5 (final, fixed, ordered)\n\n"
                f"{rank_resp_preview}\n\n"
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
            answer = run_agent(prompt, df_filtered, max_turns=20)
            st.markdown(answer)
        except Exception as e:
            st.error(
                f"Claude agent error: {e}\n\nMake sure ANTHROPIC_API_KEY is set in your .env file."
            )

# ── Feature summary ─────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Feature summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Hard filter columns** (DuckDB SQL)")
    st.markdown(
        "| Column | Type | Description |\n"
        "|--------|------|-------------|\n"
        "| `borough` | categorical | NYC borough (5 values) |\n"
        "| `subway_station_count` | int | Subway stations in neighborhood |\n"
        "| `avg_pedestrian` | float | Average pedestrian count |\n"
        "| `storefront_density_per_km2` | float | Storefront filings per km\u00b2 |\n"
        "| `storefront_filing_count` | int | Non-vacant storefront filings (CDTA) |\n"
        "| `commercial_activity_score` | float | `log1p`(storefront filings \u00d7 avg pedestrian) |\n"
        "| `competitive_score` | float | `log1p`(storefront filings / (avg pedestrian + 1)); used as a **max** hard-filter cap |\n"
        "| `shooting_incident_count` | float | Total shooting incidents; used as a **max** hard-filter cap |\n"
        "| `nfh_goal4_fin_shocks_score` | float | NFH Goal 4 (financial shocks) index (when column exists) |\n"
        "| `nfh_overall_score` | float | NFH overall index (when column exists) |\n"
        "| `act_*_storefront` | int | **Not** in the default `WHERE`; available in `SELECT *` and the table above. Add thresholds in SQL if you need category-specific hard filters. |"
    )

with col2:
    st.markdown(
        "**Soft / embedded columns** (vectors from `src/embeddings.py`: OpenAI `text-embedding-3-small` when a key is set, "
        "else local sentence-transformers; override with `EMBEDDING_BACKEND`)"
    )
    _act_ref = sorted(c for c in df_full.columns if is_act_storefront_column(c))
    _act_rows = "".join(
        f"| `{c}` | Filing count for **{c.removeprefix('act_').removesuffix('_storefront').replace('_', ' ')}** (in profile text when count is above zero) |\n"
        for c in _act_ref
    )
    st.markdown(
        "| Column | Used in text profile |\n"
        "|--------|---------------------|\n"
        "| `neighborhood` | Area name |\n"
        "| `borough` | Borough |\n"
        "| `area_km2` | CDTA footprint (km\u00b2) |\n"
        "| `storefront_filing_count` | Total non-vacant filings |\n"
        "| `storefront_density_per_km2` | Filings per km\u00b2 (plus density wording) |\n"
        "| `category_diversity` | Count of distinct Primary Business Activity buckets |\n"
        "| `category_entropy` | Mix across all `act_*_storefront` counts |\n"
        f"{_act_rows}"
        "| `avg_pedestrian` | Foot traffic level |\n"
        "| `subway_station_count` | Transit access |\n"
        "| `pop_black` | MOCEJ-style Black resident count (own sentence in profile) |\n"
        "| `pop_hispanic` | MOCEJ-style Hispanic resident count (own sentence) |\n"
        "| `pop_asian` | MOCEJ-style Asian resident count (own sentence; pairs with e.g. Asian restaurant queries) |\n"
        "| `total_population_proxy` | Sum of the three `pop_*` groups (own sentence) |\n"
        "| `nfh_median_income` | NFH median income (when column exists) |\n"
        "| `pct_bachelors_plus` | Share with bachelor's degree or higher |\n"
        "| `commute_public_transit` | Public transit commute share |\n"
        "| `competitive_score` | Competition pressure (log-scaled storefronts per pedestrian) |\n"
        "| `transit_activity_score` | Transit activity (log-scaled product) |\n"
        "| `nfh_overall_score` | NFH overall financial-health composite |\n"
        "| `nfh_goal4_fin_shocks_score` | NFH Goal 4 financial-shock resilience |\n"
        "| *Blended* | MinMax([semantic, -competitive_score]) on filtered rows, then α·semantic + (1−α)·competitive-penalty (Ranking only; not embedded) |"
    )

st.caption(
    "**Ranking-only (not embedding inputs):** `cluster` and `cluster_description` come from the latest "
    "**K-Selection Analysis** on the home page (`app.py`), joined by `neighborhood`."
)

st.markdown(
    "**Pipeline**: Hard filters \u2192 DuckDB SQL \u2192 MinMaxScaler on "
    "[cosine semantic, -competitive_score] \u2192 "
    "α·col0 + β·col1 (β = 1 − α) \u2192 optional Claude SQL analysis."
)
