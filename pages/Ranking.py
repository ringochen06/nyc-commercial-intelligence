"""
NYC Commercial Intelligence — Streamlit dashboard.

Hard constraints  : sidebar controls → DuckDB SQL filters (deterministic)
Soft ranking      : α·semantic + β·commercial_activity
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
    load_neighborhood_test_features,
)
from feature_engineering import is_act_density_column, is_act_storefront_column  # noqa: E402

DEFAULT_SOFT_QUERY = (
    "quiet residential area suitable for boutique retail with good subway access"
)
DEFAULT_ALPHA_SEMANTIC = 0.8
_VALIDATION_OUTPUT_DIRS = [
    _REPO / "tests" / "outputs" / "validation" / "rank_stability_business_queries",
    _REPO / "outputs" / "validation" / "rank_stability_business_queries",
]

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ranking · NYC Commercial Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Ranking")
st.caption(
    "Hard filters (DuckDB SQL), then α·semantic + β·commercial activity "
    "(MinMax on the filtered set). Optional Claude. "
    "**Cluster** columns appear when you have run **K-Selection Analysis** on the home page."
)

# ── Load data ───────────────────────────────────────────────────────────────

_data_choice = st.sidebar.radio(
    "Data vintage",
    options=["Present", "Past"],
    index=0,
    help=(
        "**Present** — `data/processed/neighborhood_features_final.csv` (latest pipeline run).\n\n"
        "**Past** — `tests/data/neighborhood_features_final.csv` (historical snapshot used for testing)."
    ),
)
df_full = load_neighborhood_features() if _data_choice == "Present" else load_neighborhood_test_features()

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
comm_min = float(pd.to_numeric(df_full["commercial_activity_score"], errors="coerce").min())
comm_max = float(pd.to_numeric(df_full["commercial_activity_score"], errors="coerce").max())
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


def apply_hard_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Build and execute a DuckDB SQL query from sidebar filter values."""
    con = duckdb.connect()
    con.register("nbhd", df)

    borough_list = ", ".join(f"'{b}'" for b in selected_boroughs)
    sql = f"""
        SELECT *
        FROM nbhd
        WHERE borough IN ({borough_list})
          AND subway_station_count >= {subway_range}
          AND avg_pedestrian >= {ped_threshold}
          AND storefront_density_per_km2 >= {density_threshold}
          AND storefront_filing_count >= {poi_threshold}
          AND commercial_activity_score >= {comm_threshold}
          {"AND nfh_goal4_fin_shocks_score >= " + str(float(nfh_goal4_threshold)) if nfh_goal4_threshold is not None else ""}
          {"AND nfh_overall_score >= " + str(float(nfh_overall_threshold)) if nfh_overall_threshold is not None else ""}
        ORDER BY commercial_activity_score DESC
    """

    result = con.execute(sql).fetchdf()
    con.close()
    return result


df_filtered = apply_hard_filters(df_full)

# ── Show hard-filter results ────────────────────────────────────────────────

st.subheader(f"Hard-filtered neighborhoods ({len(df_filtered)} of {len(df_full)})")

with st.expander("View generated SQL", expanded=False):
    borough_list_display = ", ".join(f"'{b}'" for b in selected_boroughs)
    st.code(
        f"SELECT * FROM neighborhoods\n"
        f"WHERE borough IN ({borough_list_display})\n"
        f"  AND subway_station_count >= {subway_range}\n"
        f"  AND avg_pedestrian >= {ped_threshold}\n"
        f"  AND storefront_density_per_km2 >= {density_threshold}\n"
        f"  AND storefront_filing_count >= {poi_threshold}\n"
        f"  AND commercial_activity_score >= {comm_threshold}\n"
        f"{'  AND nfh_goal4_fin_shocks_score >= ' + str(float(nfh_goal4_threshold)) + chr(10) if nfh_goal4_threshold is not None else ''}"
        f"{'  AND nfh_overall_score >= ' + str(float(nfh_overall_threshold)) + chr(10) if nfh_overall_threshold is not None else ''}"
        f"ORDER BY commercial_activity_score DESC;",
        language="sql",
    )

if df_filtered.empty:
    st.warning(
        "No neighborhoods match the current hard filters. Loosen your constraints."
    )
    st.stop()

_act_cols = sorted(c for c in df_filtered.columns if is_act_storefront_column(c))
display_cols = [
    "neighborhood",
    "borough",
    "storefront_filing_count",
]
for _x in ("category_diversity", "category_entropy"):
    if _x in df_filtered.columns:
        display_cols.append(_x)
display_cols.extend(_act_cols)
display_cols.extend(
    [
        "subway_station_count",
        "avg_pedestrian",
        "storefront_density_per_km2",
        "commercial_activity_score",
        "transit_activity_score",
    ]
)
for optional_col in ["nfh_overall_score", "nfh_goal4_fin_shocks_score"]:
    if optional_col in df_filtered.columns:
        display_cols.append(optional_col)
display_cols = [c for c in display_cols if c in df_filtered.columns]
st.caption(
    f"Showing **{len(_act_cols)}** storefront activity columns (`act_*_storefront`) from your feature table; "
    "semantic profiles include every activity with count > 0 (`src/embeddings.py`)."
)
st.dataframe(
    df_filtered[display_cols].reset_index(drop=True),
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
    help="Weight on embedding cosine similarity; β = 1 − α goes to commercial_activity_score.",
)
beta = 1.0 - alpha
_w = np.array([alpha, beta], dtype=float)
st.sidebar.caption(
    f"\u03b2 — commercial activity = 1 − \u03b1 → \u03b1={alpha:.3f}, \u03b2={beta:.3f}"
)

# ── Business activity density column selector ───────────────────────────────

_density_cols = sorted(c for c in df_full.columns if is_act_density_column(c))


def _density_label(col: str) -> str:
    return col.removeprefix("act_").removesuffix("_density").replace("_", " ").title()


def _sanitize_query_slug(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "query"


def _find_validation_output_dir() -> Path | None:
    for path in _VALIDATION_OUTPUT_DIRS:
        if (path / "query_rank_correlations.csv").is_file():
            return path
    return None


@st.cache_data(show_spinner=False)
def _load_validation_summary(output_dir: str | Path) -> pd.DataFrame | None:
    path = Path(output_dir) / "query_rank_correlations.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


@st.cache_data(show_spinner=False)
def _load_validation_detail(output_dir: str | Path, query_slug: str) -> pd.DataFrame | None:
    path = Path(output_dir) / f"rank_compare_{query_slug}.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _validation_metric_options(rank_df: pd.DataFrame) -> dict[str, str]:
    options = {
        "Rank 2022": "rank_2022",
        "Rank 2024": "rank_2024",
        "Rank Delta (2024 − 2022)": "rank_delta",
        "Absolute Rank Delta": "rank_delta_abs",
    }
    return {label: col for label, col in options.items() if col in rank_df.columns}


def _build_validation_map(
    rank_df: pd.DataFrame,
    *,
    metric_col: str,
    metric_label: str,
) -> go.Figure | None:
    if not CDTA_SHAPE_PATH.is_file():
        return None

    geo_gdf = load_cdta_gdf_for_map(CDTA_SHAPE_PATH)
    shape_geojson = geo_gdf.__geo_interface__

    map_df = rank_df.copy()
    map_df[metric_col] = pd.to_numeric(map_df[metric_col], errors="coerce")
    map_df["map_key"] = map_df["cd"].astype(str) + " | " + map_df["borough"].astype(str)
    map_plot = map_df.dropna(subset=["map_key", metric_col, "cd", "borough"])
    if map_plot.empty:
        return None

    zvals = map_plot[metric_col]
    zmin = float(zvals.min())
    zmax = float(zvals.max())
    if zmax <= zmin:
        zmax = zmin + 1e-9

    if metric_col == "rank_delta":
        bound = max(abs(zmin), abs(zmax))
        zmin, zmax = -bound, bound
        colorscale = [
            [0.0, "#b2182b"],
            [0.5, "#f7f7f7"],
            [1.0, "#2166ac"],
        ]
        reversescale = False
    elif metric_col in {"rank_2022", "rank_2024"}:
        colorscale = [
            [0.0, "#08306b"],
            [0.5, "#4292c6"],
            [1.0, "#deebf7"],
        ]
        reversescale = True
    else:
        colorscale = [
            [0.0, "#fff5eb"],
            [0.5, "#fdae6b"],
            [1.0, "#a63603"],
        ]
        reversescale = False

    bounds = geo_gdf.geometry.total_bounds
    lon0 = float((bounds[0] + bounds[2]) / 2)
    lat0 = float((bounds[1] + bounds[3]) / 2)

    hover_parts = [
        map_plot["neighborhood"].astype(str),
        "<br>rank 2022 " + map_plot["rank_2022"].astype(str),
        "<br>rank 2024 " + map_plot["rank_2024"].astype(str),
    ]
    if "rank_delta" in map_plot.columns:
        hover_parts.append("<br>delta " + map_plot["rank_delta"].astype(str))
    hover_text = hover_parts[0]
    for part in hover_parts[1:]:
        hover_text = hover_text + part

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=shape_geojson,
            locations=map_plot["map_key"],
            z=map_plot[metric_col],
            featureidkey="properties.map_key",
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            reversescale=reversescale,
            marker_opacity=0.82,
            marker_line_width=0.6,
            marker_line_color="#ffffff",
            colorbar=dict(
                title=dict(text=metric_label, side="right"),
                tickformat=".1f",
            ),
            text=hover_text,
            hovertemplate="<b>%{text}</b><br>selected value %{z:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=8, b=0),
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat0, lon=lon0),
            zoom=9,
        ),
    )
    return fig


selected_density_col: str | None = None
if _density_cols:
    selected_density_col = st.selectbox(
        "Business activity for density column in ranking table",
        options=_density_cols,
        format_func=_density_label,
    )

# ── Soft ranking: semantic + commercial activity ────────────────────────────

st.subheader("Soft ranking (α·semantic + β·commercial activity)")
st.caption(
    "Semantic similarity uses the **submitted** text from the sidebar (after **Update soft ranking**). "
    "Changing α still updates the blend immediately. "
    "**Cluster** / **Cluster description** use the latest **K-Selection Analysis** on the home page "
    "(same session). Neighborhoods not in that run show empty cluster cells."
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
        if "commercial_activity_score" not in df_filtered.columns:
            raise KeyError(
                "Column commercial_activity_score missing from feature table."
            )

        act_scores = np.array(
            [
                float(
                    df_filtered.loc[
                        df_filtered["neighborhood"] == n, "commercial_activity_score"
                    ].iloc[0]
                )
                for n in keep_names
            ],
            dtype=float,
        )

        X = np.column_stack([sim_scores.astype(float), act_scores])
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
                    "commercial_activity_score": act_scores.round(3),
                    "blended_score": final_scores.round(4),
                }
            )
            .sort_values("blended_score", ascending=False)
            .reset_index(drop=True)
        )

        _density_col_label: str | None = None
        if selected_density_col and selected_density_col in df_filtered.columns:
            _density_col_label = (
                f"present {_density_label(selected_density_col).lower()} density"
            )
            _density_merge = df_filtered[["neighborhood", selected_density_col]].rename(
                columns={selected_density_col: _density_col_label}
            )
            ranking_df = ranking_df.merge(_density_merge, on="neighborhood", how="left")

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
                "commercial_activity_score",
                "blended_score",
            ]
            if _density_col_label and _density_col_label in rk.columns:
                show_cols.append(_density_col_label)
            ranking_df = rk[show_cols].set_index("rank")
        else:
            st.caption(
                "_No cluster labels in this session yet — on **home**, run **K-Selection Analysis** to "
                "fill **cluster** and **cluster_description**._"
            )

        st.dataframe(ranking_df, use_container_width=True, height=380)

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
        st.info("Could not map filtered neighborhoods to embeddings.")

except Exception as e:
    st.warning(
        f"Semantic search unavailable (embeddings not cached or embedding backend error): {e}\n\n"
        "Run `python -m src.embeddings` to generate embeddings. Default: OpenAI when `OPENAI_API_KEY` is set, else local sentence-transformers; "
        "to force local-only vectors, set `EMBEDDING_BACKEND=sentence_transformers` in `.env` (see `.env.example`)."
    )

# ── Validation choropleth (saved 2022 vs 2024 outputs) ───────────────────

st.subheader("Validation Choropleth (2022 vs 2024)")
st.caption(
    "Saved rank-stability outputs for a fixed validation query. This map is independent of the live hard-filtered ranking above: "
    "use it to compare 2022 rank, 2024 rank, and rank shifts across CDTAs."
)

validation_dir = _find_validation_output_dir()
if validation_dir is None:
    st.info(
        "No validation outputs found. Run `python tests/rank_stability_validation_business_queries.py` from the repo root, "
        "or run it from `tests/` with `--output-dir ../tests/outputs/validation/rank_stability_business_queries`."
    )
else:
    validation_summary = _load_validation_summary(validation_dir)
    if validation_summary is None:
        st.info("Validation summary CSV is missing or empty.")
    else:
        query_options = validation_summary["query"].astype(str).tolist()
        selected_validation_query = st.selectbox(
            "Validation query",
            options=query_options,
            index=0,
            help="Choose which saved validation query to map.",
        )

        query_slug = _sanitize_query_slug(selected_validation_query)
        validation_detail = _load_validation_detail(validation_dir, query_slug)
        if validation_detail is None:
            st.warning(f"No rank comparison CSV found for query '{selected_validation_query}'.")
        else:
            metric_options = _validation_metric_options(validation_detail)
            selected_metric_label = st.radio(
                "Map view",
                options=list(metric_options.keys()),
                horizontal=True,
            )
            metric_col = metric_options[selected_metric_label]

            summary_row = validation_summary.loc[
                validation_summary["query"].astype(str) == selected_validation_query
            ].head(1)
            if not summary_row.empty:
                row = summary_row.iloc[0]
                st.markdown(
                    f"**Spearman** {float(row['spearman_r']):.3f}  |  "
                    f"**Kendall Tau** {float(row['kendall_tau']):.3f}  |  "
                    f"**CDTA overlap** {int(row['n_cdta_overlap'])}"
                )

            validation_map_fig = _build_validation_map(
                validation_detail,
                metric_col=metric_col,
                metric_label=selected_metric_label,
            )
            if validation_map_fig is None:
                st.warning("Validation map could not be rendered because the shapefile or joined rows are missing.")
            else:
                st.plotly_chart(validation_map_fig, use_container_width=True)

            if {"rank_2022", "rank_2024", "rank_delta", "rank_delta_abs"}.issubset(
                validation_detail.columns
            ):
                movers_df = (
                    validation_detail[
                        [
                            "neighborhood",
                            "cd",
                            "borough",
                            "rank_2022",
                            "rank_2024",
                            "rank_delta",
                            "rank_delta_abs",
                        ]
                    ]
                    .sort_values("rank_delta_abs", ascending=False)
                    .reset_index(drop=True)
                )
                st.markdown("**Largest Movers**")
                st.dataframe(movers_df.head(15), use_container_width=True, hide_index=True)

# ── Claude agent analysis ──────────────────────────────────────────────────

st.subheader("AI analysis (Claude)")

if st.button("Ask Claude to analyze filtered data", type="primary"):
    with st.spinner("Claude is analyzing the filtered neighborhoods..."):
        try:
            prompt = (
                f"The user is looking for: {soft_query}\n\n"
                f"There are {len(df_filtered)} neighborhoods that passed the hard filters. "
                f"Use the run_sql tool to explore the data and recommend the top 3-5 "
                f"neighborhoods that best match the user's soft preferences. "
                f"Explain your reasoning with specific data points."
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
        "| `commercial_activity_score` | Activity level (log-scaled product) |\n"
        "| `transit_activity_score` | Transit activity (log-scaled product) |\n"
        "| `nfh_overall_score` | NFH overall financial-health composite |\n"
        "| `nfh_goal4_fin_shocks_score` | NFH Goal 4 financial-shock resilience |\n"
        "| *Blended* | MinMax([semantic, commercial_activity]) on filtered rows, then α·semantic + (1−α)·activity (Ranking only; not embedded) |"
    )

st.caption(
    "**Ranking-only (not embedding inputs):** `cluster` and `cluster_description` come from the latest "
    "**K-Selection Analysis** on the home page (`app.py`), joined by `neighborhood`."
)

st.markdown(
    "**Pipeline**: Hard filters \u2192 DuckDB SQL \u2192 MinMaxScaler on "
    "[cosine semantic, commercial_activity_score] \u2192 "
    "α·col0 + β·col1 (β = 1 − α) \u2192 optional Claude SQL analysis."
)
