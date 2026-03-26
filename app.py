"""
Streamlit entry: natural-language query, rankings, cluster and persistence visuals.

Run from repo root: ``streamlit run app.py``
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="NYC Commercial Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("NYC Commercial Intelligence")
st.caption(
    "Explore NYC neighborhoods using licensing + demographics, semantic search, "
    "K-means (NumPy), and persistence prediction."
)

st.sidebar.header("Query")
query = st.sidebar.text_input(
    "Describe the kind of area you want",
    value="quiet residential area for boutique retail",
)

alpha = st.sidebar.slider("Weight: semantic similarity (α)", 0.0, 1.0, 0.5, 0.05)
beta = 1.0 - alpha
st.sidebar.caption(f"Persistence weight (β) = {beta:.2f}")

st.info(
    "Pipeline modules are under `src/`. After preprocessing and artifacts exist in "
    "`outputs/`, wire `semantic_search`, `ranking`, and plots here for the full demo."
)

st.subheader("Your query")
st.write(query)

st.subheader("Results (placeholder)")
st.dataframe(
    {
        "neighborhood_id": [],
        "similarity": [],
        "predicted_persistence": [],
        "final_score": [],
    },
    use_container_width=True,
)
