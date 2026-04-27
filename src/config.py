"""Repository paths, constants, and Streamlit cached loaders (``app.py`` + ``pages/Ranking.py``)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
NEIGHBORHOOD_FEATURES_CSV = DATA_PROCESSED / "neighborhood_features_final.csv"
NEIGHBORHOOD_TEST_FEATURES_CSV = Path(__file__).parent.parent / "tests" / "data" / "neighborhood_features_final.csv"
CDTA_SHAPE_PATH = REPO_ROOT / "data" / "raw" / "nyc_boundaries" / "nycdta2020.shp"

try:
    from feature_engineering import load_boundaries
except ImportError:
    from src.feature_engineering import load_boundaries


@st.cache_data
def load_neighborhood_features() -> pd.DataFrame:
    return pd.read_csv(NEIGHBORHOOD_FEATURES_CSV)
@st.cache_data
def load_neighborhood_test_features() -> pd.DataFrame:
    return pd.read_csv(NEIGHBORHOOD_TEST_FEATURES_CSV)


@st.cache_data(show_spinner=False)
def load_cdta_gdf_for_map(shape_path_str: str | Path) -> gpd.GeoDataFrame:
    """CDTA polygons in WGS84 with ``map_key`` = ``cd | borough`` for choropleth joins."""
    gdf = load_boundaries(Path(shape_path_str))
    out = gdf.copy()
    out["map_key"] = out["cd"] + " | " + out["borough"]
    return out
