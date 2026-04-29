from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from data_eval_processing import run_eval_processing
from src.feature_engineering import (
    build_pedestrian_features,
    build_subway_features,
    compute_area_features,
    load_boundaries,
    merge_all_features,
    spatial_join_points,
)

_TESTS_DATA = Path(__file__).parent / "data"
_PROCESSED = _ROOT / "data" / "processed"
_RAW = _ROOT / "data" / "raw"


def run_eval_pipeline(
    storefront_path: str | Path = _RAW / "Storefronts_Reported_Vacant_or_Not_20260424.csv",
    ped_raw_path: str | Path = _RAW / "Bi-Annual_Pedestrian_Counts.csv",
    nbhd_path: str | Path = _RAW / "Public - Neighborhood Profiles 2018 - All.csv",
    nfhd_raw_path: str | Path = _RAW / "Neighborhood_Financial_Health_Digital_Mapping_and_Data_Tool.xlsx",
    boundary_path: str | Path = _RAW / "nyc_boundaries" / "nycdta2020.shp",
    subway_path: str | Path = _PROCESSED / "subway_clean.csv",
    output_dir: str | Path = _TESTS_DATA,
    max_year: int = 2022,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Step 1: running eval data processing (cap year={max_year})...")
    run_eval_processing(
        storefront_path=storefront_path,
        ped_raw_path=ped_raw_path,
        nbhd_path=nbhd_path,
        nfhd_raw_path=nfhd_raw_path,
        boundary_path=boundary_path,
        output_dir=output_dir,
        max_year=max_year,
    )

    print("\nStep 2: running feature engineering...")

    boundary_gdf = load_boundaries(boundary_path)
    area_df = compute_area_features(boundary_gdf)

    ped = pd.read_csv(output_dir / "ped_clean_test.csv")
    subway = pd.read_csv(subway_path)
    nbhd = pd.read_csv(output_dir / "nbhd_clean_test.csv")
    storefront_feat = pd.read_csv(output_dir / "storefront_features_test.csv")

    ped_joined = spatial_join_points(ped, boundary_gdf)
    subway_joined = spatial_join_points(subway, boundary_gdf)

    ped_feat = build_pedestrian_features(ped_joined)
    subway_feat = build_subway_features(subway_joined)

    final_df = merge_all_features(
        area_df=area_df,
        ped_feat=ped_feat,
        subway_feat=subway_feat,
        shooting_feat=area_df.assign(shooting_incident_count=0)[
            ["neighborhood", "cd", "borough", "shooting_incident_count"]
        ],
        nbhd_clean=nbhd,
        storefront_feat=storefront_feat,
    )

    final_df.to_csv(output_dir / "neighborhood_features_final.csv", index=False)

    print("Feature engineering finished.")
    print("Final feature table shape:", final_df.shape)
    na_counts = final_df.isna().sum()
    na_cols = na_counts[na_counts > 0].sort_values(ascending=False)
    if len(na_cols):
        print("\nNaN counts (investigate before shipping):")
        print(na_cols.head(20))
    else:
        print("\nNo missing values in the final table.")

    print("\nPreview:")
    print(final_df.columns.to_list())
    print(final_df.head())


if __name__ == "__main__":
    run_eval_pipeline()
