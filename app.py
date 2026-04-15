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

DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "neighborhood_features_final.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


df_full = load_data()

# ── Sidebar: Hard Filters ──────────────────────────────────────────────────

st.sidebar.header("Hard Filters")
st.sidebar.caption("Deterministic constraints applied via DuckDB SQL before the model sees the data.")

# Borough
boroughs = sorted(df_full["borough"].unique().tolist())
selected_boroughs = st.sidebar.multiselect(
    "Borough",
    options=boroughs,
    default=boroughs,
    help="Select one or more boroughs to include.",
)

# Subway station count
subway_min, subway_max = int(df_full["subway_station_count"].min()), int(df_full["subway_station_count"].max())
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
        f"ORDER BY commercial_activity_score DESC;",
        language="sql",
    )

if df_filtered.empty:
    st.warning("No neighborhoods match the current hard filters. Loosen your constraints.")
    st.stop()

display_cols = [
    "neighborhood", "borough", "total_poi", "subway_station_count",
    "avg_pedestrian", "poi_density_per_km2", "commercial_activity_score",
    "transit_activity_score",
]
st.dataframe(
    df_filtered[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=300,
)

# ── Sidebar: Soft Preferences ──────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.header("Soft Preferences")
st.sidebar.caption(
    "Free-text description of what you're looking for. Claude will analyze "
    "the hard-filtered data and recommend neighborhoods."
)

soft_query = st.sidebar.text_area(
    "Describe your ideal area",
    value="quiet residential area suitable for boutique retail with good subway access",
    height=100,
)

st.sidebar.markdown("**Blend** (α + β = 1)")
alpha = st.sidebar.slider(
    "\u03b1 — semantic similarity",
    0.0,
    1.0,
    0.5,
    0.05,
    help="Weight on embedding cosine similarity; β = 1 − α goes to commercial_activity_score.",
)
beta = 1.0 - alpha
_w = np.array([alpha, beta], dtype=float)
st.sidebar.caption(f"\u03b2 — commercial activity = 1 − \u03b1 → \u03b1={alpha:.3f}, \u03b2={beta:.3f}")

# ── Soft ranking: semantic + commercial activity ────────────────────────────

st.subheader("Soft ranking (α·semantic + β·commercial activity)")

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
            raise KeyError("Column commercial_activity_score missing from feature table.")

        act_scores = np.array(
            [
                float(df_filtered.loc[df_filtered["neighborhood"] == n, "commercial_activity_score"].iloc[0])
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

        ranking_df = pd.DataFrame({
            "neighborhood": keep_names,
            "semantic_similarity": sim_scores.round(4),
            "commercial_activity_score": act_scores.round(0).astype(int),
            "blended_score": final_scores.round(4),
        }).sort_values("blended_score", ascending=False).reset_index(drop=True)

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
            st.error(f"Claude agent error: {e}\n\nMake sure ANTHROPIC_API_KEY is set in your .env file.")

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
        "| `commercial_activity_score` | float | POI \u00d7 pedestrian activity |"
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
