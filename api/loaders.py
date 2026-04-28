"""Cached data loaders for the FastAPI backend.

Reads CSVs and the pre-rendered CDTA GeoJSON. No geopandas at runtime;
the shapefile is converted once to data/processed/cdta_geo.json by
scripts/build_cdta_geojson.py.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
NEIGHBORHOOD_FEATURES_CSV = DATA_PROCESSED / "neighborhood_features_final.csv"
NEIGHBORHOOD_TEST_FEATURES_CSV = REPO_ROOT / "tests" / "data" / "neighborhood_features_final.csv"
CDTA_GEO_JSON = DATA_PROCESSED / "cdta_geo.json"

# Diagnostic only (reported by /api/health). Runtime no longer reads it.
CDTA_SHAPE_PATH = REPO_ROOT / "data" / "raw" / "nyc_boundaries" / "nycdta2020.shp"


@lru_cache(maxsize=2)
def load_features(vintage: str = "present") -> pd.DataFrame:
    path = NEIGHBORHOOD_FEATURES_CSV if vintage == "present" else NEIGHBORHOOD_TEST_FEATURES_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV not found at {path}.")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_geo_payload() -> dict:
    if CDTA_GEO_JSON.is_file():
        return json.loads(CDTA_GEO_JSON.read_text())
    return {
        "geojson": {"type": "FeatureCollection", "features": []},
        "bounds": {"minx": -74.26, "miny": 40.49, "maxx": -73.69, "maxy": 40.92},
        "center": {"lat": 40.705, "lon": -73.975},
    }


def load_cdta_geojson() -> dict:
    return _load_geo_payload()["geojson"]


def load_cdta_bounds() -> tuple[float, float, float, float]:
    b = _load_geo_payload()["bounds"]
    return (b["minx"], b["miny"], b["maxx"], b["maxy"])
