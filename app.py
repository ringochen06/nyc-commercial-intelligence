"""
NYC Commercial Intelligence — Streamlit dashboard.

Hard constraints  : sidebar controls → DuckDB SQL filters (deterministic)
Soft ranking      : α·semantic + β·commercial_activity
                    (MinMax-scaled to [0,1] on the filtered set; β = 1 − α from one slider)
Semantic search   : OpenAI embeddings cosine similarity on neighborhood profiles
Optional Claude   : read-only SQL on filtered data

Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agent import run_agent  # noqa: E402
from embeddings import (  # noqa: E402
    cosine_similarity,
    embed_neighborhood_features,
    embed_texts,
)

DEFAULT_SOFT_QUERY = (
    "quiet residential area suitable for boutique retail with good subway access"
)
DEFAULT_ALPHA_SEMANTIC = 0.8

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NYC Commercial Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("NYC Commercial Intelligence")
st.caption(
    "Rank NYC neighborhoods using hard filters (DuckDB SQL), then α·semantic + β·commercial activity "
    "(MinMax-normalized on the filtered set; one blend slider: β = 1 − α). Optional Claude analysis."
)

# ── Load data ───────────────────────────────────────────────────────────────

DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "processed"
    / "neighborhood_features_final.csv"
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


df_full = load_data()

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

# POI density
density_min = float(df_full["poi_density_per_km2"].min())
density_max = float(df_full["poi_density_per_km2"].max())
density_threshold = st.sidebar.slider(
    "Min POI density (per km\u00b2)",
    min_value=density_min,
    max_value=density_max,
    value=density_min,
    step=0.5,
    help="Minimum points-of-interest per square kilometer.",
)

# Total POI
poi_min = int(df_full["total_poi"].min())
poi_max = int(df_full["total_poi"].max())
poi_threshold = st.sidebar.slider(
    "Min total POI count",
    min_value=poi_min,
    max_value=poi_max,
    value=poi_min,
    help="Minimum total number of businesses/POIs.",
)

# Commercial activity score
comm_min = float(df_full["commercial_activity_score"].min())
comm_max = float(df_full["commercial_activity_score"].max())
comm_threshold = st.sidebar.slider(
    "Min commercial activity score",
    min_value=comm_min,
    max_value=comm_max,
    value=comm_min,
    step=1000.0,
    help="Minimum commercial activity score (POI count x pedestrian traffic).",
)

# Optional NFH filters (available only when NFH columns exist)
st.sidebar.markdown("---")
st.sidebar.markdown("**NFH (optional)**")

has_nfh_shocks = "nfh_goal4_fin_shocks_score" in df_full.columns
use_nfh_goal4_filter = False
nfh_goal4_threshold = None
if has_nfh_shocks and df_full["nfh_goal4_fin_shocks_score"].notna().any():
    nfh_goal4_vals = pd.to_numeric(
        df_full["nfh_goal4_fin_shocks_score"], errors="coerce"
    ).dropna()
    if not nfh_goal4_vals.empty:
        use_nfh_goal4_filter = st.sidebar.checkbox(
            "Enable NFH Goal 4 (Financial Shocks)",
            value=False,
            help="Apply a minimum threshold on Goal 4: stable housing & capacity to limit financial shocks.",
        )
        nfh_goal4_threshold = st.sidebar.slider(
            "Min NFH Goal 4 score (Financial Shocks)",
            min_value=float(nfh_goal4_vals.min()),
            max_value=float(nfh_goal4_vals.max()),
            value=float(nfh_goal4_vals.min()),
            step=0.1,
            help="Higher values indicate stronger financial-shock resilience in this index.",
            disabled=not use_nfh_goal4_filter,
        )

has_nfh_overall = "nfh_overall_score" in df_full.columns
use_nfh_overall_filter = False
nfh_overall_threshold = None
if has_nfh_overall and df_full["nfh_overall_score"].notna().any():
    nfh_overall_vals = pd.to_numeric(
        df_full["nfh_overall_score"], errors="coerce"
    ).dropna()
    if not nfh_overall_vals.empty:
        use_nfh_overall_filter = st.sidebar.checkbox(
            "Enable NFH Overall index",
            value=False,
            help="Apply a minimum threshold on the combined Neighborhood Financial Health index.",
        )
        nfh_overall_threshold = st.sidebar.slider(
            "Min NFH Overall index score",
            min_value=float(nfh_overall_vals.min()),
            max_value=float(nfh_overall_vals.max()),
            value=float(nfh_overall_vals.min()),
            step=0.1,
            help="Higher values indicate stronger overall neighborhood financial health (relative index).",
            disabled=not use_nfh_overall_filter,
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
          AND poi_density_per_km2 >= {density_threshold}
          AND total_poi >= {poi_threshold}
          AND commercial_activity_score >= {comm_threshold}
          {"AND nfh_goal4_fin_shocks_score >= " + str(float(nfh_goal4_threshold)) if (use_nfh_goal4_filter and nfh_goal4_threshold is not None) else ""}
          {"AND nfh_overall_score >= " + str(float(nfh_overall_threshold)) if (use_nfh_overall_filter and nfh_overall_threshold is not None) else ""}
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
        f"  AND poi_density_per_km2 >= {density_threshold}\n"
        f"  AND total_poi >= {poi_threshold}\n"
        f"  AND commercial_activity_score >= {comm_threshold}\n"
        f"{'  AND nfh_goal4_fin_shocks_score >= ' + str(float(nfh_goal4_threshold)) + chr(10) if (use_nfh_goal4_filter and nfh_goal4_threshold is not None) else ''}"
        f"{'  AND nfh_overall_score >= ' + str(float(nfh_overall_threshold)) + chr(10) if (use_nfh_overall_filter and nfh_overall_threshold is not None) else ''}"
        f"ORDER BY commercial_activity_score DESC;",
        language="sql",
    )

if df_filtered.empty:
    st.warning(
        "No neighborhoods match the current hard filters. Loosen your constraints."
    )
    st.stop()

display_cols = [
    "neighborhood",
    "borough",
    "total_poi",
    "subway_station_count",
    "avg_pedestrian",
    "poi_density_per_km2",
    "commercial_activity_score",
    "transit_activity_score",
]
for optional_col in ["nfh_overall_score", "nfh_goal4_fin_shocks_score"]:
    if optional_col in df_filtered.columns:
        display_cols.append(optional_col)
st.dataframe(
    df_filtered[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=300,
)

with st.expander("About zeros, nulls, and refreshing data", expanded=False):
    st.markdown(
        """
**Why you used to see nulls (`None`)**

- **NFH columns** (`nfh_*`) only load if the Neighborhood Financial Health CSV is parsed and joined on Community District. Older exports used labels like `BX Community District 8`; the pipeline now normalizes those to `BX08` so scores merge correctly.
- **Profile / NFH numeric gaps** after the CDTA join are filled in the pipeline with **borough median, then citywide median** (a proxy for CDTAs that do not match a single profile row). They are useful for dashboards, not exact tract-level truth.

**`commercial_activity_score` and `transit_activity_score`**

- **`commercial_activity_score` = `total_poi` × `avg_pedestrian`** (after filling missing POI counts with 0 and missing pedestrian averages with the citywide mean, then any remaining NaN pedestrian with 0).
- **`transit_activity_score` = `subway_station_count` × `avg_pedestrian`** with the same pedestrian handling.
- A row can still show **0** when **`total_poi` is 0** (no businesses in the data for that CDTA) or **`subway_station_count` is 0**, or when pedestrian signal is still 0 after imputation. That is a real signal from the inputs, not a broken join.

**If numbers look stale after changing the pipeline**

- Regenerate with `python run_pipeline.py`, then in Streamlit use **Rerun** or **Clear cache** (menu) so `load_data()` picks up the new `neighborhood_features_final.csv`.
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

# ── Soft ranking: semantic + commercial activity ────────────────────────────

st.subheader("Soft ranking (α·semantic + β·commercial activity)")
st.caption(
    "Semantic similarity uses the **submitted** text from the sidebar (after **Update soft ranking**). "
    "Changing α still updates the blend immediately."
)


@st.cache_data(show_spinner="Loading neighborhood embeddings...")
def get_all_embeddings():
    return embed_neighborhood_features()


try:
    all_embeddings, all_texts = get_all_embeddings()

    full_neighborhoods = load_data()["neighborhood"].tolist()
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
                    "commercial_activity_score": act_scores.round(0).astype(int),
                    "blended_score": final_scores.round(4),
                }
            )
            .sort_values("blended_score", ascending=False)
            .reset_index(drop=True)
        )

        ranking_df.index = ranking_df.index + 1
        ranking_df.index.name = "rank"

        st.dataframe(ranking_df, use_container_width=True, height=350)
    else:
        st.info("Could not map filtered neighborhoods to embeddings.")

except Exception as e:
    st.warning(
        f"Semantic search unavailable (embeddings not cached or OPENAI_API_KEY not set): {e}\n\n"
        "Run `python -m src.embeddings` to generate embeddings first, or set OPENAI_API_KEY in .env."
    )

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
            answer = run_agent(prompt, df_filtered)
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
        "| `poi_density_per_km2` | float | Business density per km\u00b2 |\n"
        "| `total_poi` | int | Total points of interest |\n"
        "| `commercial_activity_score` | float | POI \u00d7 pedestrian activity |\n"
        "| `nfh_goal4_fin_shocks_score` | float | Optional: NFH Goal 4 (financial shocks) index |\n"
        "| `nfh_overall_score` | float | Optional: NFH overall index |"
    )

with col2:
    st.markdown("**Soft / embedded columns** (OpenAI `text-embedding-3-small`)")
    st.markdown(
        "| Column | Used in text profile |\n"
        "|--------|---------------------|\n"
        "| `neighborhood` | Name context |\n"
        "| `borough` | Geographic context |\n"
        "| `total_poi` | Business count |\n"
        "| `category_entropy` | Industry diversity |\n"
        "| `avg_pedestrian` | Foot traffic level |\n"
        "| `subway_station_count` | Transit access |\n"
        "| `poi_density_per_km2` | Density descriptor |\n"
        "| `commercial_activity_score` | Activity level |\n"
        "| `transit_activity_score` | Transit activity |\n"
        "| *Blended* | MinMax([semantic, commercial_activity]) then α·semantic + (1−α)·activity |"
    )

st.markdown(
    "**Pipeline**: Hard filters \u2192 DuckDB SQL \u2192 MinMaxScaler on "
    "[cosine semantic, commercial_activity_score] \u2192 "
    "α·col0 + β·col1 (β = 1 − α) \u2192 optional Claude SQL analysis."
)
