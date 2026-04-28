"""Cached data loaders for the FastAPI backend.

Mirrors src/config.py but without streamlit decorators — uses functools.lru_cache
so the same DataFrame is reused across requests in a single worker.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
NEIGHBORHOOD_FEATURES_CSV = DATA_PROCESSED / "neighborhood_features_final.csv"
CDTA_SHAPE_PATH = REPO_ROOT / "data" / "raw" / "nyc_boundaries" / "nycdta2020.shp"


@lru_cache(maxsize=2)
def load_features(vintage: str = "present") -> pd.DataFrame:
    path = NEIGHBORHOOD_FEATURES_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV not found at {path}. Run the pipeline or include the file in the deployment.")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_cdta_geojson() -> dict:
    """CDTA polygons as GeoJSON with map_key = 'cd | borough' on each feature."""
    from src.feature_engineering import load_boundaries

    if not CDTA_SHAPE_PATH.is_file():
        return {"type": "FeatureCollection", "features": []}
    gdf = load_boundaries(CDTA_SHAPE_PATH)
    out: gpd.GeoDataFrame = gdf.copy()
    out["map_key"] = out["cd"] + " | " + out["borough"]
    return out[["neighborhood", "cd", "borough", "map_key", "geometry"]].__geo_interface__


@lru_cache(maxsize=1)
def load_cdta_bounds() -> tuple[float, float, float, float]:
    """Total bounds (minx, miny, maxx, maxy) — used for map centering."""
    from src.feature_engineering import load_boundaries

    if not CDTA_SHAPE_PATH.is_file():
        return (-74.26, 40.49, -73.69, 40.92)
    gdf = load_boundaries(CDTA_SHAPE_PATH)
    b = gdf.geometry.total_bounds
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
