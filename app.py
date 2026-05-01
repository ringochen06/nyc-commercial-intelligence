"""
K-Selection / Clustering - default Streamlit home page (`streamlit run app.py`).

Runs K-means for k = 2 ... max_k, elbow and silhouette charts, maps, and cluster notes.
Results are stored in session state for the **Ranking** page (neighborhood -> cluster + brief).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kmeans_numpy import (
    compute_inertia,
    kmeans_plus_plus,
    kmeans_plus_plus_with_caching,
    silhouette_score,
)  # noqa: E402
from streamlit_app.cluster_helpers import (  # noqa: E402
    cluster_brief_description,
    clustering_density_feature_options,
    cluster_semantics_from_embeddings,
    find_elbow,
    find_elbow_curvature_knee,
    zscore_normalize,
)
from streamlit_app.constants import (  # noqa: E402
    BASE_CANDIDATE_FEATURES,
    BASE_DEFAULT_FEATURES,
    color_for_cluster,
    readable_feature_label,
)
from config import (  # noqa: E402
    CDTA_SHAPE_PATH,
    load_cdta_gdf_for_map,
    load_neighborhood_features,
)

# ---

st.set_page_config(
    page_title="K-Selection - Clustering",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("K-Selection / Clustering")
st.caption(
    "Compare inertia and silhouette across k = 2 ... max_k, then explore clusters. "
    "Run **K-Selection Analysis** to refresh labels used on the **Ranking** page."
)

# ---


df_full = load_neighborhood_features()

activity_density_features = clustering_density_feature_options(df_full)
CANDIDATE_FEATURES: list[str] = BASE_CANDIDATE_FEATURES + activity_density_features
DEFAULT_FEATURES: list[str] = BASE_DEFAULT_FEATURES + activity_density_features

# ---

st.sidebar.header("Filters & Settings")

boroughs = sorted(df_full["borough"].unique().tolist())
selected_boroughs = st.sidebar.multiselect(
    "Borough",
    options=boroughs,
    default=boroughs,
    help="Restrict clustering to selected boroughs.",
)

selected_features = st.sidebar.multiselect(
    "Features for clustering",
    options=CANDIDATE_FEATURES,
    default=DEFAULT_FEATURES,
    help="Numeric columns used for clustering, including optional business category density features. "
    "Features are z-score normalised before K-means runs.",
    format_func=readable_feature_label,
)

max_k = st.sidebar.slider(
    "Maximum k",
    min_value=2,
    max_value=15,
    value=8,
    help="Upper bound for the k sweep (same default range as the Next.js K-Selection page). "
    "Automatically capped at (n_neighborhoods - 1).",
)

# ---

if not selected_boroughs:
    st.warning("Select at least one borough.")
    st.stop()

df_filtered = df_full[df_full["borough"].isin(selected_boroughs)].copy()

if not selected_features:
    st.warning("Select at least one feature.")
    st.stop()

# Drop rows with NaN in any selected feature
df_clean = df_filtered.dropna(subset=selected_features).reset_index(drop=True)
n = len(df_clean)

st.markdown(
    f"**{n}** neighborhoods available after borough filter "
    f"(out of {len(df_full)} total, {len(df_filtered) - n} dropped for missing values)."
)

if n < 4:
    st.error(
        "Fewer than 4 neighborhoods match the current filters. "
        "Loosen borough selection or add more boroughs to run clustering."
    )
    st.stop()

# ---


X_raw = df_clean[selected_features].values.astype(float)
X = zscore_normalize(X_raw)

effective_max_k = min(max_k, n - 1)
k_range = list(range(2, effective_max_k + 1))

if effective_max_k < max_k:
    st.info(
        f"max_k capped at {effective_max_k} (= n - 1 = {n} - 1) "
        f"because you cannot have more clusters than neighborhoods."
    )

# ---

if st.button("Run K-Selection Analysis", type="primary"):
    inertias: list[float] = []
    sil_numpy: list[float] = []
    sil_sklearn: list[float] = []

    progress = st.progress(0, text="Running K-means...")
    total_steps = len(k_range)

    for step, k in enumerate(k_range):
        labels, centroids, _ = kmeans_plus_plus(X, k, random_state=42)
        inertias.append(compute_inertia(X, labels, centroids))
        sil_numpy.append(silhouette_score(X, labels))
        sil_sklearn.append(float(sklearn_silhouette_score(X, labels)))
        progress.progress((step + 1) / total_steps, text=f"k = {k} / {effective_max_k}")

    progress.empty()

    elbow_k = find_elbow(k_range, inertias)
    elbow_k_kneedle = find_elbow_curvature_knee(k_range, inertias)
    best_sil_k = k_range[int(np.argmax(sil_sklearn))]

    # Persist to session state so the viz section survives Streamlit reruns
    st.session_state["ks_k_range"] = k_range
    st.session_state["ks_inertias"] = inertias
    st.session_state["ks_sil_numpy"] = sil_numpy
    st.session_state["ks_sil_sklearn"] = sil_sklearn
    st.session_state["ks_elbow_k"] = elbow_k
    st.session_state["ks_elbow_k_kneedle"] = elbow_k_kneedle
    st.session_state["ks_best_sil_k"] = best_sil_k
    # Match FastAPI: default partition k = perpendicular-distance elbow when user has not chosen yet.
    st.session_state["ks_user_k"] = elbow_k
    st.session_state["ks_X"] = X
    st.session_state["ks_X_raw"] = X_raw
    st.session_state["ks_df_clean"] = df_clean
    st.session_state["ks_features"] = selected_features
    st.session_state["ks_n"] = n

# ---

if "ks_k_range" in st.session_state:
    k_range_s: list[int] = st.session_state["ks_k_range"]
    inertias_s: list[float] = st.session_state["ks_inertias"]
    sil_numpy_s: list[float] = st.session_state["ks_sil_numpy"]
    sil_sklearn_s: list[float] = st.session_state["ks_sil_sklearn"]
    elbow_k: int = st.session_state["ks_elbow_k"]
    elbow_k_kneedle: int = st.session_state["ks_elbow_k_kneedle"]
    best_sil_k: int = st.session_state["ks_best_sil_k"]
    X_s: np.ndarray = st.session_state["ks_X"]
    X_raw_s: np.ndarray = st.session_state["ks_X_raw"]
    df_s: pd.DataFrame = st.session_state["ks_df_clean"]
    features_s: list[str] = st.session_state["ks_features"]
    n_s: int = st.session_state["ks_n"]

    _default_viz_k = st.session_state.get("ks_user_k", k_range_s[0])
    if _default_viz_k not in k_range_s:
        _default_viz_k = k_range_s[0]
    viz_k = st.sidebar.select_slider(
        "Clusters (k)",
        options=k_range_s,
        value=_default_viz_k,
        help="How many clusters to draw on the charts and export to the Ranking page. "
        "Grey / green elbow markers match FastAPI `/api/cluster`; silhouette is a separate cue.",
        key="ks_viz_k_slider",
    )
    st.session_state["ks_user_k"] = viz_k

    st.success(
        f"Heuristic references - elbow (**chord distance**, grey): **k = {elbow_k}** · "
        f"alt. elbow (**Delta^2 inertia**, green when shown): **k = {elbow_k_kneedle}** · "
        f"best silhouette (sklearn): **k = {best_sil_k}**. "
        f"Visualizations use **k = {viz_k}** (sidebar)."
    )

    # ---

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=k_range_s,
            y=inertias_s,
            mode="lines+markers",
            marker=dict(size=8, color="#4A90D9"),
            line=dict(width=2, color="#4A90D9"),
            name="Inertia (WCSS)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=k_range_s,
            y=sil_numpy_s,
            mode="lines+markers",
            marker=dict(size=8, color="#E74C3C", symbol="square"),
            line=dict(width=2, color="#E74C3C", dash="dash"),
            name="Silhouette (NumPy)",
        ),
        secondary_y=True,
    )

    fig.add_vline(
        x=elbow_k,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"elbow chord k={elbow_k}",
        annotation_position="top right",
    )
    if elbow_k_kneedle != elbow_k:
        fig.add_vline(
            x=elbow_k_kneedle,
            line_dash="dot",
            line_color="darkgreen",
            annotation_text=f"Delta^2 elbow k={elbow_k_kneedle}",
            annotation_position="top left",
        )

    if viz_k != elbow_k:
        fig.add_vline(
            x=viz_k,
            line_dash="solid",
            line_color="#ea580c",
            annotation_text=f"Clusters (k)={viz_k}",
            annotation_position="bottom right",
        )

    fig.update_xaxes(title_text="k (number of clusters)", tickvals=k_range_s)
    fig.update_yaxes(title_text="Inertia (WCSS)", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
    fig.update_layout(
        title=f"Elbow Method: Inertia & Silhouette vs. k  ({n_s} neighborhoods, {len(features_s)} features)",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Grey dashed** = elbow via **max perpendicular distance** to the inertia chord (same as FastAPI `elbow_k`). "
        "**Green dotted** (when shown) = alternate elbow from **|Delta^2 inertia|** on the normalized curve (`elbow_k_kneedle`). "
        "**Orange** (when shown) = sidebar **Clusters (k)** if it differs from grey. "
        "**Yellow row** in the table matches **Clusters (k)**. "
        "**Silhouette** (red) is a separate cue."
    )

    # ---

    st.subheader("Cluster Visualization")

    st.caption(
        f"Partition with **k = {viz_k}** (sidebar). "
        f"Scatter, map, centroid bars, and notes below show all **{viz_k}** clusters."
    )

    viz_labels, viz_centroids, _ = kmeans_plus_plus_with_caching(
        features_s, X_s, k=viz_k, random_state=42
    )
    sem = cluster_semantics_from_embeddings(
        df_full,
        df_s,
        viz_labels,
        viz_k,
        centroids=viz_centroids,
        features=features_s,
    )

    # Share with Ranking page: neighborhood -> cluster id, and per-cluster profile text
    _names_v = df_s["neighborhood"].astype(str).tolist()
    st.session_state["ks_cluster_by_neighborhood"] = {
        _names_v[i]: int(viz_labels[i]) for i in range(len(_names_v))
    }
    _semantic_descriptions = (
        {int(block["cluster"]): str(block.get("description") or "") for block in sem}
        if sem is not None
        else {}
    )
    st.session_state["ks_cluster_brief"] = {
        c: _semantic_descriptions.get(
            c,
            cluster_brief_description(viz_centroids[c], features_s),
        )
        for c in range(viz_k)
    }
    st.session_state["ks_cluster_k"] = int(viz_k)

    col_left, col_right = st.columns(2)

    # ---

    with col_left:
        st.markdown("**Feature Scatter**")
        xf = st.selectbox(
            "X axis",
            options=features_s,
            index=(
                features_s.index("avg_pedestrian")
                if "avg_pedestrian" in features_s
                else 0
            ),
            key="scatter_x",
        )
        yf = st.selectbox(
            "Y axis",
            options=features_s,
            index=(
                features_s.index("storefront_filing_count")
                if "storefront_filing_count" in features_s
                else min(1, len(features_s) - 1)
            ),
            key="scatter_y",
        )

        xi = features_s.index(xf)
        yi = features_s.index(yf)

        scatter_fig = go.Figure()

        for c in range(viz_k):
            mask = viz_labels == c
            scatter_fig.add_trace(
                go.Scatter(
                    x=X_raw_s[mask, xi],
                    y=X_raw_s[mask, yi],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=color_for_cluster(c),
                        opacity=0.85,
                        line=dict(width=0.5, color="white"),
                    ),
                    name=f"Cluster {c}",
                    text=df_s["neighborhood"].values[mask],
                    hovertemplate="<b>%{text}</b><br>"
                    + xf
                    + ": %{x:.1f}<br>"
                    + yf
                    + ": %{y:.1f}<extra></extra>",
                )
            )

        # Centroid stars (in raw space: centroid in z-score -> back to raw)
        x_mean = X_raw_s.mean(axis=0)
        x_std = X_raw_s.std(axis=0) + 1e-8
        centroids_raw = viz_centroids * x_std + x_mean

        for c in range(viz_k):
            scatter_fig.add_trace(
                go.Scatter(
                    x=[centroids_raw[c, xi]],
                    y=[centroids_raw[c, yi]],
                    mode="markers",
                    marker=dict(
                        size=18,
                        symbol="star",
                        color=color_for_cluster(c),
                        line=dict(width=1.5, color="black"),
                    ),
                    name=f"Centroid {c}",
                    showlegend=False,
                    hovertemplate=f"<b>Centroid {c}</b><br>"
                    + xf
                    + ": %{x:.2f}<br>"
                    + yf
                    + ": %{y:.2f}<extra></extra>",
                )
            )

        scatter_fig.update_layout(
            xaxis_title=xf,
            yaxis_title=yf,
            height=420,
            legend=dict(orientation="v", x=1.01, y=1),
            margin=dict(r=120),
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    # ---

    with col_right:
        st.markdown("**Centroid Profiles** *(z-score space)*")

        bar_fig = go.Figure()
        for c in range(viz_k):
            bar_fig.add_trace(
                go.Bar(
                    name=f"Cluster {c}",
                    x=features_s,
                    y=viz_centroids[c].tolist(),
                    marker_color=color_for_cluster(c),
                    opacity=0.85,
                )
            )

        bar_fig.update_layout(
            barmode="group",
            xaxis_title="Feature",
            yaxis_title="Normalized value (z-score)",
            xaxis_tickangle=-35,
            height=420,
            legend=dict(orientation="v", x=1.01, y=1),
            margin=dict(r=120),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # ---

    st.markdown("**NYC map** *(CDTA polygons filled by cluster)*")

    if not CDTA_SHAPE_PATH.is_file():
        st.info(
            f"No boundary file at `{CDTA_SHAPE_PATH}` - map is skipped. "
            "The CDTA shapefile is normally under `data/raw/nyc_boundaries/`."
        )
    else:
        shape_gdf = load_cdta_gdf_for_map(CDTA_SHAPE_PATH)
        shape_df = shape_gdf[["neighborhood", "cd", "borough", "map_key", "geometry"]]
        shape_geojson = shape_df.__geo_interface__
        map_df = df_s[["neighborhood", "cd", "borough"]].copy()
        map_df["cluster"] = viz_labels.astype(int)
        map_df["map_key"] = map_df["cd"] + " | " + map_df["borough"]
        map_df = map_df.merge(
            shape_df[["map_key", "geometry"]],
            on="map_key",
            how="left",
        )
        n_missing = int(map_df["geometry"].isna().sum())
        if n_missing:
            st.warning(
                f"{n_missing} row(s) could not be matched to the shapefile on "
                "`cd` + `borough`; they are omitted from the map."
            )
        map_plot = map_df.dropna(subset=["geometry"]).copy()
        if map_plot.empty:
            st.warning("No polygons to plot on the map.")
        else:
            map_plot["cluster_label"] = map_plot["cluster"].map(
                lambda c: f"Cluster {c}"
            )
            map_fig = go.Figure()
            for c in range(viz_k):
                sub = map_plot[map_plot["cluster"] == c]
                if sub.empty:
                    continue
                map_fig.add_trace(
                    go.Choroplethmapbox(
                        geojson=shape_geojson,
                        locations=sub["map_key"],
                        z=[1] * len(sub),
                        featureidkey="properties.map_key",
                        colorscale=[
                            [0.0, color_for_cluster(c)],
                            [1.0, color_for_cluster(c)],
                        ],
                        showscale=False,
                        marker_opacity=0.65,
                        marker_line_width=1.0,
                        marker_line_color="white",
                        name=f"Cluster {c}",
                        text=sub["neighborhood"] + " (" + sub["cd"] + ")",
                        hovertemplate=(
                            "<b>%{text}</b><br>" "cluster=" + str(c) + "<extra></extra>"
                        ),
                    )
                )
            bounds = shape_df.geometry.total_bounds
            lon0 = float((bounds[0] + bounds[2]) / 2)
            lat0 = float((bounds[1] + bounds[3]) / 2)
            map_fig.update_layout(
                height=480,
                margin=dict(l=0, r=0, t=8, b=0),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=lat0, lon=lon0),
                    zoom=9,
                ),
            )
            st.plotly_chart(map_fig, use_container_width=True)

    # ---

    st.subheader("Cluster notes (cached embeddings)")
    st.caption(
        "Same vectors as the main app (`outputs/embeddings/`). This page does **not** call an embedding API. "
        "Per cluster: average embedding of members, then the neighborhoods whose vectors are "
        "closest to that mean (cosine). Text is the saved profile string from `src.embeddings`."
    )
    if sem is None:
        st.info(
            "No embedding cache found, or embedding row count does not match "
            "`neighborhood_features_final.csv`. Run `python -m src.embeddings` from the repo root "
            "(add `--force` after changing the feature CSV)."
        )
    else:
        for block in sem:
            c = int(block["cluster"])
            n_cluster = int(block["n"])
            reps = block["reps"]
            description = str(
                block.get("description")
                or cluster_brief_description(viz_centroids[c], features_s)
            )
            with st.expander(
                f"Cluster {c} - n={n_cluster} neighborhoods", expanded=(c == 0)
            ):
                st.markdown(f"**Cluster description:** {description}")
                if not reps:
                    st.write("No members matched the embedding index.")
                else:
                    for rank, rep in enumerate(reps, start=1):
                        sim = float(rep["cosine_to_mean"])
                        nm = str(rep["neighborhood"])
                        excerpt = str(rep["profile_excerpt"])
                        st.markdown(
                            f"**{rank}.** `{nm}` - cosine to cluster mean **{sim:.3f}**"
                        )
                        st.caption(excerpt)

    # ---

    st.subheader("Results table")
    st.caption(
        "The **yellow row** matches **Clusters (k)** in the sidebar. "
        "The **grey** vertical line is the perpendicular-distance elbow (aligned with the deployed API)."
    )

    results_df = pd.DataFrame(
        {
            "k": k_range_s,
            "inertia": [round(v, 2) for v in inertias_s],
            "silhouette_numpy": [round(v, 4) for v in sil_numpy_s],
            "silhouette_sklearn": [round(v, 4) for v in sil_sklearn_s],
        }
    )

    # High-contrast row for dark Streamlit themes (pale yellow + default text was hard to read).
    _elbow_row_style = (
        "background-color: #3d3520; color: #fef9e8; font-weight: 600; "
        "border-top: 2px solid #eab308; border-bottom: 2px solid #eab308"
    )

    def highlight_chosen_k(row: pd.Series) -> list[str]:
        if row["k"] == viz_k:
            return [_elbow_row_style] * len(row)
        return [""] * len(row)

    styled = results_df.style.apply(highlight_chosen_k, axis=1).format(
        {
            "inertia": "{:,.2f}",
            "silhouette_numpy": "{:.4f}",
            "silhouette_sklearn": "{:.4f}",
        }
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ---

    with st.expander("Feature details", expanded=False):
        st.markdown(
            "Features were **z-score normalised** before clustering "
            f"(mean=0, std~1 per column). {n_s} neighborhoods x {len(features_s)} features."
        )
        feat_stats = pd.DataFrame(
            {
                "feature": features_s,
                "mean (raw)": X_raw_s.mean(axis=0).round(3),
                "std (raw)": X_raw_s.std(axis=0).round(3),
            }
        )
        st.dataframe(feat_stats, use_container_width=True, hide_index=True)
