#!/usr/bin/env python3
"""
Replay `merge_all_features` (storefront + ped + subway + profiles) and report NaNs.

Run from repo root: python scripts/audit_imputation_fraction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_processing import clean_storefront_data  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    build_pedestrian_features,
    build_storefront_features,
    build_subway_features,
    compute_area_features,
    empty_storefront_features,
    load_boundaries,
    merge_all_features,
    spatial_join_points,
)


def main() -> None:
    processed = ROOT / "data/processed"
    raw = ROOT / "data/raw"
    boundary_path = raw / "nyc_boundaries/nycdta2020.shp"

    ped = pd.read_csv(processed / "ped_clean.csv")
    subway = pd.read_csv(processed / "subway_clean.csv")
    nbhd = pd.read_csv(processed / "nbhd_clean.csv")

    boundary_gdf = load_boundaries(boundary_path)
    area_df = compute_area_features(boundary_gdf)
    ped_joined = spatial_join_points(ped, boundary_gdf)
    subway_joined = spatial_join_points(subway, boundary_gdf)

    sf_path = raw / "Storefronts_Reported_Vacant_or_Not_20260424.csv"
    if sf_path.is_file():
        sf_joined = spatial_join_points(clean_storefront_data(sf_path), boundary_gdf)
        sf_feat = build_storefront_features(sf_joined)
    else:
        sf_feat = empty_storefront_features(area_df)

    ped_feat = build_pedestrian_features(ped_joined)
    subway_feat = build_subway_features(subway_joined)

    df = merge_all_features(area_df, ped_feat, subway_feat, nbhd, sf_feat)
    na_by_col = df.isna().sum()
    na_cols = na_by_col[na_by_col > 0]

    print("=== merge_all_features replay (storefront, no license/inspection POI) ===\n")
    print(f"Shape: {df.shape}")
    print(f"Total NaN cells: {int(na_by_col.sum())}")
    if len(na_cols):
        print("\nColumns with NaN:")
        print(na_cols.to_string())
    else:
        print("No NaN values.")


if __name__ == "__main__":
    main()
