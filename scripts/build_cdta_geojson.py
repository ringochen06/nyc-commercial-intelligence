"""Convert the CDTA shapefile to a static GeoJSON the API can ship without geopandas.

Run once after changing the shapefile:
    uv run python scripts/build_cdta_geojson.py

Output: data/processed/cdta_geo.json — read at runtime by api/loaders.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.feature_engineering import load_boundaries  # noqa: E402

SHAPE = REPO_ROOT / "data" / "raw" / "nyc_boundaries" / "nycdta2020.shp"
OUT = REPO_ROOT / "data" / "processed" / "cdta_geo.json"


def main() -> None:
    if not SHAPE.is_file():
        raise SystemExit(f"shapefile not found: {SHAPE}")
    gdf = load_boundaries(SHAPE)
    gdf["map_key"] = gdf["cd"] + " | " + gdf["borough"]
    keep = gdf[["neighborhood", "cd", "borough", "map_key", "geometry"]]
    geojson = keep.__geo_interface__
    b = gdf.geometry.total_bounds
    payload = {
        "geojson": geojson,
        "bounds": {
            "minx": float(b[0]),
            "miny": float(b[1]),
            "maxx": float(b[2]),
            "maxy": float(b[3]),
        },
        "center": {
            "lat": float((b[1] + b[3]) / 2),
            "lon": float((b[0] + b[2]) / 2),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(geojson['features'])} features)")


if __name__ == "__main__":
    main()
