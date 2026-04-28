"""Repository paths and cached loaders.

The Streamlit app uses ``@st.cache_data`` when streamlit is installed (so
re-running app.py reuses cached DataFrames). The FastAPI backend on
Railway runs without streamlit installed, in which case we fall back to a
no-op decorator and rely on the module-level callers caching themselves.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
NEIGHBORHOOD_FEATURES_CSV = DATA_PROCESSED / "neighborhood_features_final.csv"
NEIGHBORHOOD_TEST_FEATURES_CSV = Path(__file__).parent.parent / "tests" / "data" / "neighborhood_features_final.csv"
CDTA_SHAPE_PATH = REPO_ROOT / "data" / "raw" / "nyc_boundaries" / "nycdta2020.shp"


try:
    import streamlit as _st

    _cache_data = _st.cache_data
except ImportError:
    def _cache_data(*dargs, **dkwargs):
        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return dargs[0]

        def _wrap(fn):
            return fn

        return _wrap


@_cache_data
def load_neighborhood_features() -> pd.DataFrame:
    return pd.read_csv(NEIGHBORHOOD_FEATURES_CSV)


@_cache_data
def load_neighborhood_test_features() -> pd.DataFrame:
    return pd.read_csv(NEIGHBORHOOD_TEST_FEATURES_CSV)


@_cache_data(show_spinner=False)
def load_cdta_gdf_for_map(shape_path_str: str | Path):
    """CDTA polygons in WGS84 with ``map_key`` = ``cd | borough``. Streamlit-only path."""
    # geopandas / feature_engineering only needed in the Streamlit app.
    try:
        from feature_engineering import load_boundaries
    except ImportError:
        from src.feature_engineering import load_boundaries
    gdf = load_boundaries(Path(shape_path_str))
    out = gdf.copy()
    out["map_key"] = out["cd"] + " | " + out["borough"]
    return out
