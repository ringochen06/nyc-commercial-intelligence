#!/usr/bin/env python3
"""
One-off audit: replay merge_all_features checkpoints to estimate what fraction of
cells were filled by borough/city median, 0-fill, or pedestrian mean-fill.

Run from repo root: python scripts/audit_imputation_fraction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import (  # noqa: E402
    build_pedestrian_features,
    build_poi_features,
    build_subway_features,
    compute_area_features,
    load_boundaries,
    normalize_cdta_join_key,
    safe_divide,
    spatial_join_points,
)


def main() -> None:
    processed = ROOT / "data/processed"
    raw = ROOT / "data/raw"
    boundary_path = raw / "nyc_boundaries/nycdta2020.shp"

    poi = pd.read_csv(processed / "poi_clean.csv")
    ped = pd.read_csv(processed / "ped_clean.csv")
    subway = pd.read_csv(processed / "subway_clean.csv")
    nbhd = pd.read_csv(processed / "nbhd_clean.csv")

    boundary_gdf = load_boundaries(boundary_path)
    poi_joined = spatial_join_points(poi, boundary_gdf)
    ped_joined = spatial_join_points(ped, boundary_gdf)
    subway_joined = spatial_join_points(subway, boundary_gdf)

    area_df = compute_area_features(boundary_gdf)
    poi_feat = build_poi_features(poi_joined)
    ped_feat = build_pedestrian_features(ped_joined)
    subway_feat = build_subway_features(subway_joined)

    df = area_df.copy()
    for other in [poi_feat, ped_feat, subway_feat]:
        df = df.merge(other, on=["neighborhood", "cd", "borough"], how="left")

    df["_cd_join"] = df["cd"].map(normalize_cdta_join_key)
    nb = nbhd.copy()
    nb["_cd_join"] = nb["cd"].map(normalize_cdta_join_key)
    nb = nb.dropna(subset=["_cd_join"]).drop_duplicates(subset=["_cd_join"], keep="first")
    profile_cols = [c for c in nb.columns if c not in ("neighborhood", "borough", "cd", "_cd_join")]
    if profile_cols:
        df = df.merge(nb[["_cd_join"] + profile_cols], on="_cd_join", how="left")
    df = df.drop(columns=["_cd_join"])

    n_rows = len(df)

    # --- Profile: NaN before borough / city median imputation ---
    prof_pre = df[profile_cols].apply(pd.to_numeric, errors="coerce")
    profile_nan_cells = int(prof_pre.isna().sum().sum())
    profile_cells = int(prof_pre.size)
    rows_any_profile_nan = int(prof_pre.isna().any(axis=1).sum())

    for col in profile_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df["borough"].notna().any():
            borough_med = df.groupby("borough")[col].transform("median")
            df[col] = df[col].fillna(borough_med)
        gmed = df[col].median()
        if pd.notna(gmed):
            df[col] = df[col].fillna(gmed)

    # --- Densities (NaN from divide) ---
    if "area_km2" in df.columns:
        if "total_poi" in df.columns:
            df["poi_density_per_km2"] = safe_divide(df["total_poi"], df["area_km2"])
        if "retail" in df.columns:
            df["retail_density_per_km2"] = safe_divide(df["retail"], df["area_km2"])
        if "food" in df.columns:
            df["food_density_per_km2"] = safe_divide(df["food"], df["area_km2"])
        if "subway_station_count" in df.columns:
            df["subway_density_per_km2"] = safe_divide(df["subway_station_count"], df["area_km2"])

    density_cols = [
        c
        for c in [
            "poi_density_per_km2",
            "retail_density_per_km2",
            "food_density_per_km2",
            "subway_density_per_km2",
        ]
        if c in df.columns
    ]
    dens_pre = df[density_cols].apply(pd.to_numeric, errors="coerce")
    density_nan_cells = int(dens_pre.isna().sum().sum())
    density_cells = int(dens_pre.size)

    poi_like_cols = [
        "total_poi",
        "unique_poi",
        "category_diversity",
        "category_entropy",
        "poi_density_per_km2",
        "food",
        "other",
        "retail",
    ]
    poi_like_cols = [c for c in poi_like_cols if c in df.columns]
    retail_extra = ["ratio_retail", "retail_density_per_km2", "food_density_per_km2"]
    retail_extra = [c for c in retail_extra if c in df.columns]
    subway_cols = [c for c in ["subway_station_count", "subway_density_per_km2"] if c in df.columns]

    block_poi = poi_like_cols + [c for c in retail_extra if c not in poi_like_cols]
    block_poi = list(dict.fromkeys(block_poi))
    pre_poi = df[block_poi].apply(pd.to_numeric, errors="coerce")
    poi_nan_cells = int(pre_poi.isna().sum().sum())
    poi_cells = int(pre_poi.size)

    pre_sub = df[subway_cols].apply(pd.to_numeric, errors="coerce") if subway_cols else pd.DataFrame()
    subway_nan_cells = int(pre_sub.isna().sum().sum()) if not pre_sub.empty else 0
    subway_cells = int(pre_sub.size) if not pre_sub.empty else 0

    ped_cols = [c for c in ["avg_pedestrian", "peak_pedestrian"] if c in df.columns]
    pre_ped = df[ped_cols].apply(pd.to_numeric, errors="coerce")
    ped_nan_cells = int(pre_ped.isna().sum().sum())
    ped_cells = int(pre_ped.size)
    rows_any_ped_nan = int(pre_ped.isna().any(axis=1).sum()) if ped_cols else 0

    ped_point_col = "pedestrian_count_points"
    if ped_point_col in df.columns:
        pre_pts = pd.to_numeric(df[ped_point_col], errors="coerce")
        ped_pts_nan = int(pre_pts.isna().sum())
        ped_pts_cells = len(pre_pts)
    else:
        ped_pts_nan = ped_pts_cells = 0

    # Cells that receive any of the above "pipeline fills" (non-overlapping groups)
    # Note: density NaNs overlap with poi_block's poi_density etc. — count union carefully
    # Here we report each bucket separately; "overlap" means same cell counted twice if we sum naively
    # For union of *positions* with any NaN before fills in (profile + poi block + subway + ped + ped_pts):
    mask_any = np.zeros((n_rows,), dtype=bool)
    mask_any |= prof_pre.isna().any(axis=1).values
    if block_poi:
        mask_any |= pre_poi.isna().any(axis=1).values
    if subway_cols:
        mask_any |= pre_sub.isna().any(axis=1).values
    if ped_cols:
        mask_any |= pre_ped.isna().any(axis=1).values
    if ped_point_col in df.columns:
        mask_any |= pre_pts.isna().values

    rows_touched_by_any_fill = int(mask_any.sum())

    # Approximate cell-level union: profile + (poi block excluding duplicate density cols already in poi_like)
    # Simpler headline: sum of nan cells by stage, minus double-count where same cell (rare: ratio nan & total_poi nan)
    total_nan_events = (
        profile_nan_cells
        + density_nan_cells
        + poi_nan_cells
        + subway_nan_cells
        + ped_nan_cells
        + ped_pts_nan
    )
    # Double count: poi_block includes poi_density which is same as density_nan for poi_density row
    # poi_nan_cells already counts poi_density nan; density_nan_cells also counts poi_density — fix:
    poi_only_cols = [c for c in block_poi if c not in density_cols]
    pre_poi_only = df[poi_only_cols].apply(pd.to_numeric, errors="coerce") if poi_only_cols else pd.DataFrame()
    poi_only_nan = int(pre_poi_only.isna().sum().sum()) if not pre_poi_only.empty else 0

    total_nan_unique_estimate = profile_nan_cells + density_nan_cells + poi_only_nan + subway_nan_cells + ped_nan_cells + ped_pts_nan

    final_cols = pd.read_csv(processed / "neighborhood_features_final.csv", nrows=0).columns
    numeric_final_cells = n_rows * len([c for c in final_cols if c not in ("neighborhood", "cd", "borough")])

    def pct(part: float, whole: float) -> str:
        if whole <= 0:
            return "n/a"
        return f"{100.0 * part / whole:.2f}%"

    print("=== Imputation audit (replay merge_all_features) ===\n")
    print(f"CDTA rows: {n_rows}")
    print(f"Profile columns (from nbhd_clean): {len(profile_cols)}")
    print()
    print("1) MOCEJ + NFH profile block — NaN before borough → city median fill:")
    print(f"   NaN cells: {profile_nan_cells} / {profile_cells}  ({pct(profile_nan_cells, profile_cells)})")
    print(f"   Rows with ≥1 NaN in profile: {rows_any_profile_nan} / {n_rows}  ({pct(rows_any_profile_nan, n_rows)})")
    print()
    print("2) Density columns — NaN before fillna(0) (mostly divide-by-zero / missing inputs):")
    print(f"   NaN cells: {density_nan_cells} / {density_cells}  ({pct(density_nan_cells, density_cells)})")
    print()
    print("3) POI / retail block (incl. densities listed twice above) — NaN before fillna(0):")
    print(f"   NaN cells: {poi_nan_cells} / {poi_cells}")
    poi_only_total = max(len(poi_only_cols) * n_rows, 1)
    print(f"   Same but excluding density cols (to avoid double-count with §2): {poi_only_nan} / {poi_only_total}")
    print()
    print("4) Subway — NaN before fillna(0):")
    print(f"   NaN cells: {subway_nan_cells} / {max(subway_cells, 1)}  ({pct(subway_nan_cells, subway_cells)})")
    print()
    print("5) Pedestrian — NaN before column-mean fill:")
    print(f"   NaN cells: {ped_nan_cells} / {max(ped_cells, 1)}  ({pct(ped_nan_cells, ped_cells)})")
    print(f"   Rows with ≥1 NaN: {rows_any_ped_nan} / {n_rows}  ({pct(rows_any_ped_nan, n_rows)})")
    if ped_pts_cells:
        print(f"   pedestrian_count_points NaN: {ped_pts_nan} / {ped_pts_cells}  ({pct(ped_pts_nan, ped_pts_cells)})")
    print()
    print("--- Headline (no double-count between §2 density and §3 poi_density) ---")
    print(f"   Sum of unique-ish NaN events: {total_nan_unique_estimate}")
    print(f"   Rough share of all numeric cells in final CSV (~{numeric_final_cells}): {pct(total_nan_unique_estimate, numeric_final_cells)}")
    print()
    print("Rows with any NaN in profile OR poi-only OR subway OR ped OR ped_pts (before fills):")
    print(f"   {rows_touched_by_any_fill} / {n_rows}  ({pct(rows_touched_by_any_fill, n_rows)})")


if __name__ == "__main__":
    main()
