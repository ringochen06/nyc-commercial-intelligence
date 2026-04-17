"""
K-Selection Analysis — Streamlit page.

Runs K-means for k = 2 … max_k on the filtered neighborhood dataset and
plots inertia (elbow) and silhouette score on a dual-axis Plotly chart,
helping the user identify the optimal number of clusters via the elbow method.
After the sweep, offers an interactive cluster visualization: a feature scatter
(user-chosen axes) and a centroid bar chart.
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

# Ensure src/ is importable when the page is loaded directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.kmeans_numpy import compute_inertia, kmeans, silhouette_score  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "processed" / "neighborhood_features_final.csv"
)

CANDIDATE_FEATURES: list[str] = [
    "total_poi",
    "avg_pedestrian",
    "subway_station_count",
    "poi_density_per_km2",
    "commercial_activity_score",
    "transit_activity_score",
    "category_entropy",
    "unique_poi",
    "peak_pedestrian",
    "retail_density_per_km2",
    "subway_density_per_km2",
    "ratio_retail",
    "food",
]

DEFAULT_FEATURES: list[str] = [
    "total_poi",
    "avg_pedestrian",
    "subway_station_count",
    "poi_density_per_km2",
    "commercial_activity_score",
    "transit_activity_score",
    "category_entropy",
]

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="K-Selection Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("K-Selection Analysis")
st.caption(
    "Choose the optimal number of clusters (k) by comparing the inertia elbow curve "
    "and silhouette score across k = 2 … max_k on the hard-filtered neighborhood data."
)

# ── Load data ────────────────────────────────────────────────────────────────


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


df_full = load_data()

# ── Sidebar controls ─────────────────────────────────────────────────────────

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
    help="Numeric columns used to build the feature matrix. "
    "Features are z-score normalised before K-means runs.",
)

max_k = st.sidebar.slider(
    "Maximum k",
    min_value=3,
    max_value=20,
    value=15,
    help="Upper bound for the k sweep. Automatically capped at (n_neighborhoods − 1).",
)

# ── Apply borough filter ─────────────────────────────────────────────────────

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

# ── Build feature matrix ─────────────────────────────────────────────────────


def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return (arr - mean) / (std + 1e-8)


X_raw = df_clean[selected_features].values.astype(float)
X = zscore_normalize(X_raw)

effective_max_k = min(max_k, n - 1)
k_range = list(range(2, effective_max_k + 1))

if effective_max_k < max_k:
    st.info(
        f"max_k capped at {effective_max_k} (= n − 1 = {n} − 1) "
        f"because you cannot have more clusters than neighborhoods."
    )

# ── Helpers ──────────────────────────────────────────────────────────────────

CLUSTER_PALETTE: list[str] = [
    "#4A90D9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#3498DB", "#E91E63", "#00BCD4",
    "#8BC34A", "#FF5722", "#795548", "#607D8B", "#FF9800",
    "#673AB7", "#009688", "#F44336", "#CDDC39", "#03A9F4",
]


def find_elbow(k_range: list[int], inertias: list[float]) -> int:
    """Return the k at the elbow using the perpendicular-distance (kneedle) method.

    Normalises both axes to [0,1] and finds the point with the maximum
    orthogonal distance from the chord connecting the first and last points.
    """
    ks = np.array(k_range, dtype=float)
    ys = np.array(inertias, dtype=float)
    # Normalize both axes to [0, 1]
    ks_n = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
    ys_n = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    # Direction vector of the chord from first to last point
    dx = ks_n[-1] - ks_n[0]
    dy = ys_n[-1] - ys_n[0]
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
    # Perpendicular distance of each point from the chord
    distances = np.abs(dy * ks_n - dx * ys_n + ks_n[-1] * ys_n[0] - ys_n[-1] * ks_n[0]) / norm
    return k_range[int(np.argmax(distances))] 


def find_elbow_minima(k_range: list[int], inertias: list[float]) -> int:
    """Return the k at the smallest local minimum of the inertia curve.

    A local minimum at index i means inertias[i] < inertias[i-1] and
    inertias[i] < inertias[i+1].  Among all such points, returns the one
    with the smallest inertia value (the lowest dip on the curve).

    Falls back to find_elbow() when fewer than 3 points are available or
    when the curve is strictly monotone (no interior local minimum exists).
    """
    if len(k_range) <= 3:
        return k_range[np.argmin(inertias)]
        #return find_elbow(k_range, inertias)

    ys = np.array(inertias, dtype=float)
    local_minima = [
        i for i in range(1, len(ys) - 1)
        if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]
    ]

    if not local_minima:
        return find_elbow(k_range, inertias)

    # Among all local minima, pick the one with the lowest inertia value
    best_i = min(local_minima, key=lambda i: ys[i])
    return k_range[best_i]




def _color_for_cluster(c: int) -> str:
    return CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]


# ── Run analysis ─────────────────────────────────────────────────────────────

if st.button("Run K-Selection Analysis", type="primary"):
    inertias: list[float] = []
    sil_numpy: list[float] = []
    sil_sklearn: list[float] = []

    progress = st.progress(0, text="Running K-means…")
    total_steps = len(k_range)

    for step, k in enumerate(k_range):
        labels, centroids, _ = kmeans(X, k, random_state=42)
        inertias.append(compute_inertia(X, labels, centroids))
        sil_numpy.append(silhouette_score(X, labels))
        sil_sklearn.append(float(sklearn_silhouette_score(X, labels)))
        progress.progress((step + 1) / total_steps, text=f"k = {k} / {effective_max_k}")

    progress.empty()

    elbow_k = find_elbow_minima(k_range, inertias)
    elbow_k_kneedle = find_elbow(k_range, inertias)
    best_sil_k = k_range[int(np.argmax(sil_sklearn))]

    # Persist to session state so the viz section survives Streamlit reruns
    st.session_state["ks_k_range"] = k_range
    st.session_state["ks_inertias"] = inertias
    st.session_state["ks_sil_numpy"] = sil_numpy
    st.session_state["ks_sil_sklearn"] = sil_sklearn
    st.session_state["ks_elbow_k"] = elbow_k
    st.session_state["ks_elbow_k_kneedle"] = elbow_k_kneedle
    st.session_state["ks_best_sil_k"] = best_sil_k
    st.session_state["ks_X"] = X
    st.session_state["ks_X_raw"] = X_raw
    st.session_state["ks_df_clean"] = df_clean
    st.session_state["ks_features"] = selected_features
    st.session_state["ks_n"] = n

# ── Display results (persisted via session state) ────────────────────────────

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

    elbow_idx = k_range_s.index(elbow_k)
    st.success(
        f"Elbow k (second-difference): **k = {elbow_k}**  ·  "
        f"Elbow k (kneedle): **k = {elbow_k_kneedle}**  ·  "
        f"Best silhouette k (sklearn): **k = {best_sil_k}**"
    )

    # ── Dual-axis Plotly chart ────────────────────────────────────────────────

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
        annotation_text=f"2nd-diff k={elbow_k}",
        annotation_position="top right",
    )
    if elbow_k_kneedle != elbow_k:
        fig.add_vline(
            x=elbow_k_kneedle,
            line_dash="dot",
            line_color="darkgreen",
            annotation_text=f"kneedle k={elbow_k_kneedle}",
            annotation_position="top left",
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
        "**Grey dashed line** = elbow k from the second-difference method (primary).  "
        "**Green dotted line** = elbow k from the kneedle (perpendicular distance) method, shown only when it differs.  "
        "Inertia (blue) is the primary signal; silhouette (red dashed) is shown for reference."
    )

    # ── Cluster visualization ─────────────────────────────────────────────────

    st.subheader("Cluster Visualization")

    viz_k = st.select_slider(
        "k for visualization",
        options=k_range_s,
        value=elbow_k,
        help="Run K-means with this k and explore results. Defaults to elbow k.",
    )

    viz_labels, viz_centroids, _ = kmeans(X_s, viz_k, random_state=42)

    col_left, col_right = st.columns(2)

    # ── View 1: Feature scatter ───────────────────────────────────────────────

    with col_left:
        st.markdown("**Feature Scatter**")
        xf = st.selectbox("X axis", options=features_s, index=features_s.index("avg_pedestrian") if "avg_pedestrian" in features_s else 0, key="scatter_x")
        yf = st.selectbox("Y axis", options=features_s, index=features_s.index("total_poi") if "total_poi" in features_s else min(1, len(features_s) - 1), key="scatter_y")

        xi = features_s.index(xf)
        yi = features_s.index(yf)

        scatter_fig = go.Figure()

        for c in range(viz_k):
            mask = viz_labels == c
            scatter_fig.add_trace(go.Scatter(
                x=X_raw_s[mask, xi],
                y=X_raw_s[mask, yi],
                mode="markers",
                marker=dict(size=9, color=_color_for_cluster(c), opacity=0.85,
                            line=dict(width=0.5, color="white")),
                name=f"Cluster {c}",
                text=df_s["neighborhood"].values[mask],
                hovertemplate="<b>%{text}</b><br>" + xf + ": %{x:.1f}<br>" + yf + ": %{y:.1f}<extra></extra>",
            ))

        # Centroid stars (in raw space: centroid in z-score → back to raw)
        x_mean = X_raw_s.mean(axis=0)
        x_std = X_raw_s.std(axis=0) + 1e-8
        centroids_raw = viz_centroids * x_std + x_mean

        for c in range(viz_k):
            scatter_fig.add_trace(go.Scatter(
                x=[centroids_raw[c, xi]],
                y=[centroids_raw[c, yi]],
                mode="markers",
                marker=dict(size=18, symbol="star", color=_color_for_cluster(c),
                            line=dict(width=1.5, color="black")),
                name=f"Centroid {c}",
                showlegend=False,
                hovertemplate=f"<b>Centroid {c}</b><br>" + xf + ": %{x:.2f}<br>" + yf + ": %{y:.2f}<extra></extra>",
            ))

        scatter_fig.update_layout(
            xaxis_title=xf,
            yaxis_title=yf,
            height=420,
            legend=dict(orientation="v", x=1.01, y=1),
            margin=dict(r=120),
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    # ── View 2: Centroid bar chart ────────────────────────────────────────────

    with col_right:
        st.markdown("**Centroid Profiles** *(z-score space)*")

        bar_fig = go.Figure()
        for c in range(viz_k):
            bar_fig.add_trace(go.Bar(
                name=f"Cluster {c}",
                x=features_s,
                y=viz_centroids[c].tolist(),
                marker_color=_color_for_cluster(c),
                opacity=0.85,
            ))

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

    # ── Summary table ─────────────────────────────────────────────────────────

    st.subheader("Results table")

    results_df = pd.DataFrame({
        "k": k_range_s,
        "inertia": [round(v, 2) for v in inertias_s],
        "silhouette_numpy": [round(v, 4) for v in sil_numpy_s],
        "silhouette_sklearn": [round(v, 4) for v in sil_sklearn_s],
    })

    def highlight_elbow(row: pd.Series) -> list[str]:
        if row["k"] == elbow_k:
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    styled = results_df.style.apply(highlight_elbow, axis=1).format(
        {"inertia": "{:,.2f}", "silhouette_numpy": "{:.4f}", "silhouette_sklearn": "{:.4f}"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Feature details ───────────────────────────────────────────────────────

    with st.expander("Feature details", expanded=False):
        st.markdown(
            "Features were **z-score normalised** before clustering "
            f"(mean=0, std≈1 per column). {n_s} neighborhoods × {len(features_s)} features."
        )
        feat_stats = pd.DataFrame({
            "feature": features_s,
            "mean (raw)": X_raw_s.mean(axis=0).round(3),
            "std (raw)": X_raw_s.std(axis=0).round(3),
        })
        st.dataframe(feat_stats, use_container_width=True, hide_index=True)
