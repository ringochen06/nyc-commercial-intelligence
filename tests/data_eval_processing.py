"""
data_eval_processing.py

Generates evaluation snapshots with default year caps at 2022 and point-in-time
pedestrian extracts for 2022 (tests) and 2024 (processed).

Outputs:
    tests/data/{max_year}/ — all max_year-filtered evaluation datasets
    tests/data/2024/ — symlinks to data/processed/ for 2024 vintage
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing import clean_neighborhood_profiles, standardize_borough
from src.feature_engineering import (
    build_storefront_features,
    compute_area_features,
    load_boundaries,
    spatial_join_points,
)

MAX_YEAR = 2022
TEST_PED_YEAR = 2022
PROCESSED_PED_YEAR = 2024


def parse_reporting_year_max(s: str) -> int | None:
    """Return the maximum 4-digit year found in s, or None if none is present.

    Handles plain integers ("2023") and multi-year strings ("2020 and 2021").
    """
    years = [int(y) for y in re.findall(r"\b(\d{4})\b", str(s))]
    return max(years) if years else None


def parse_ped_col_year(col: str) -> int | None:
    """Return the 4-digit year encoded in a pedestrian count column name.

    Pattern: <MonthYY>_<AM|PM|MD>  e.g. May13_AM -> 2013, Oct23_PM -> 2023.
    Returns None if the column does not match the expected pattern.
    """
    m = re.search(r"(\d{2})_(?:AM|PM|MD)$", col)
    return 2000 + int(m.group(1)) if m else None


def clean_pedestrian_data_eval(
    ped_raw_path: str | Path,
    *,
    year: int,
    mode: str = "exact",
) -> pd.DataFrame:
    """Clean pedestrian counts using either exact-year or capped-year columns.

    mode:
      - "exact": use only columns whose parsed year == year
      - "upto": use only columns whose parsed year <= year
    """
    df_ped = pd.read_csv(ped_raw_path)

    if "the_geom" in df_ped.columns:
        geom = df_ped["the_geom"].astype(str)
        lon = pd.to_numeric(
            geom.str.extract(r"POINT \((-?\d+\.\d+)")[0], errors="coerce"
        )
        lat = pd.to_numeric(
            geom.str.extract(r"POINT \(-?\d+\.\d+ (\d+\.\d+)\)")[0], errors="coerce"
        )
    else:
        lon = pd.to_numeric(df_ped["longitude"], errors="coerce")
        lat = pd.to_numeric(df_ped["latitude"], errors="coerce")

    all_ped_cols = [
        col for col in df_ped.columns if "_AM" in col or "_PM" in col or "_MD" in col
    ]
    if mode == "exact":
        ped_cols = [col for col in all_ped_cols if parse_ped_col_year(col) == year]
    elif mode == "upto":
        ped_cols = [
            col for col in all_ped_cols if (parse_ped_col_year(col) or 9999) <= year
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'exact' or 'upto'.")

    if ped_cols:
        avg_ped = df_ped[ped_cols].mean(axis=1)
        peak_ped = df_ped[ped_cols].max(axis=1)
    else:
        avg_ped = pd.Series(np.nan, index=df_ped.index)
        peak_ped = pd.Series(np.nan, index=df_ped.index)

    derived = pd.DataFrame(
        {
            "longitude": lon,
            "latitude": lat,
            "avg_pedestrian": avg_ped,
            "peak_pedestrian": peak_ped,
        },
        index=df_ped.index,
    )
    drop_lon_lat = [c for c in ("longitude", "latitude") if c in df_ped.columns]
    df_ped = pd.concat(
        [df_ped.drop(columns=drop_lon_lat, errors="ignore"), derived], axis=1
    )

    keep_cols = [
        c
        for c in [
            "Borough",
            "Street",
            "From",
            "To",
            "latitude",
            "longitude",
            "avg_pedestrian",
            "peak_pedestrian",
        ]
        if c in df_ped.columns
    ]
    df = df_ped[keep_cols].copy()

    rename_map = {
        "Borough": "borough",
        "Street": "street",
        "From": "from_street",
        "To": "to_street",
    }
    df = df.rename(columns=rename_map)

    df["borough"] = standardize_borough(df["borough"])
    df = df[~df["borough"].isin(["EAST RIVER BRIDGES", "HARLEM RIVER BRIDGES"])]
    df = df.dropna(subset=["borough", "latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    df = df[df["avg_pedestrian"].fillna(0) > 0]

    mode_label = f"year == {year}" if mode == "exact" else f"year <= {year}"
    print(
        f"  Pedestrian columns used ({mode_label}): {len(ped_cols)} of {len(all_ped_cols)}"
    )
    if ped_cols:
        print(f"  Earliest: {ped_cols[0]}  Latest: {ped_cols[-1]}")

    return df.reset_index(drop=True)


def clean_shooting_data_eval(
    shooting_path: str | Path,
    max_year: int = MAX_YEAR,
) -> pd.DataFrame:
    """Clean shooting incidents, filtering by incident year <= max_year.

    Groups incidents by neighborhood (CDTA) and returns aggregated counts.
    """
    df = pd.read_csv(shooting_path)

    df["OCCUR_DATE"] = pd.to_datetime(df["OCCUR_DATE"], format="%m/%d/%Y", errors="coerce")
    df["incident_year"] = df["OCCUR_DATE"].dt.year

    mask = (df["incident_year"] <= max_year) & (df["incident_year"].notna())
    n_before = len(df)
    df = df[mask].copy()
    print(
        f"  Shooting incidents: {n_before:,} -> {len(df):,} after year filter (<= {max_year})"
    )

    lat = pd.to_numeric(df["Latitude"], errors="coerce")
    lon = pd.to_numeric(df["Longitude"], errors="coerce")
    df["latitude"] = lat
    df["longitude"] = lon
    df["borough"] = standardize_borough(df["BORO"])

    df = df.dropna(subset=["borough", "latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]

    return df.reset_index(drop=True)


def clean_storefront_data_eval(
    storefront_path: str | Path,
    max_year: int = MAX_YEAR,
) -> pd.DataFrame:
    """Clean storefront filings keeping only rows where the max reporting year <= max_year.

    Non-numeric Reporting Year values like "2020 and 2021" are handled by extracting
    all 4-digit years and using the maximum. Rows with no parseable year are dropped.
    """
    df_raw = pd.read_csv(storefront_path, low_memory=False)

    rename_map = {
        "Filing Due Date": "filing_due_date",
        "Reporting Year": "reporting_year",
        "Borough Block Lot": "boro_block_lot",
        "Property Street Address or Storefront Address": "property_address",
        "Borough": "borough_alt",
        "Zip Code": "zip_code",
        "Sold Date": "sold_date",
        "Vacant on 12/31": "vacant_on_dec31_raw",
        "Construction Reported": "construction_raw",
        "Vacant 6/30 or Date Sold": "vacant_midyear_raw",
        "Primary Business Activity": "primary_business_activity",
        "Expiration date of the most recent lease": "lease_expiration",
        "Property Number": "property_number",
        "Property Street": "property_street",
        "Unit": "unit",
        "Borough1": "borough",
        "Postcode": "postcode",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Lat/Long": "lat_long_wkt",
        "Community Board": "community_board",
        "Council District": "council_district",
        "Census Tract": "census_tract",
        "BIN": "bin",
        "BBL": "bbl",
        "NTA": "nta_code",
        "NTA Neighborhood": "nta_neighborhood",
    }
    missing = [c for c in rename_map if c not in df_raw.columns]
    if missing:
        raise ValueError(
            "Storefront CSV is missing expected Open Data columns: "
            f"{missing}. Available: {df_raw.columns.tolist()}"
        )

    year_max = df_raw["Reporting Year"].astype(str).map(parse_reporting_year_max)
    mask = year_max.apply(lambda y: y is not None and y <= max_year)
    n_before = len(df_raw)
    df_raw = df_raw[mask].copy()
    print(
        f"  Storefront rows: {n_before:,} -> {len(df_raw):,} after year filter (<= {max_year})"
    )

    df = df_raw.rename(columns=rename_map).copy()

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    if "lat_long_wkt" in df.columns:
        wkt = df["lat_long_wkt"].astype(str)
        lon_w = pd.to_numeric(
            wkt.str.extract(r"POINT \((-?\d+\.?\d*)", expand=False), errors="coerce"
        )
        lat_w = pd.to_numeric(
            wkt.str.extract(r"POINT \(-?\d+\.?\d*\s+(-?\d+\.?\d*)\)", expand=False),
            errors="coerce",
        )
        lon = lon.where(lon.notna(), lon_w)
        lat = lat.where(lat.notna(), lat_w)

    df["latitude"] = lat
    df["longitude"] = lon

    df["borough"] = standardize_borough(df["borough"])
    df["reporting_year"] = pd.to_numeric(df["reporting_year"], errors="coerce")

    v = df["vacant_on_dec31_raw"].astype(str).str.strip().str.upper()
    df["vacant_on_dec31_yes"] = (v == "YES").astype(np.int8)
    c_col = df["construction_raw"].astype(str).str.strip().str.upper()
    df["construction_yes"] = (c_col == "YES").astype(np.int8)
    m = df["vacant_midyear_raw"].astype(str).str.strip().str.upper()
    df["vacant_midyear_yes"] = (m == "YES").astype(np.int8)

    df = df[(df["vacant_on_dec31_yes"] == 0) & (df["vacant_midyear_yes"] == 0)].copy()

    act = df["primary_business_activity"].fillna("").astype(str).str.strip()
    act_upper = act.str.upper()
    df["business_activity_category"] = act
    df.loc[act_upper == "OTHER", "business_activity_category"] = "other"
    df.loc[act_upper == "MISCELLANEOUS OTHER SERVICE", "business_activity_category"] = (
        "other"
    )
    df.loc[act == "", "business_activity_category"] = "UNKNOWN"

    keep = [
        "filing_due_date",
        "reporting_year",
        "boro_block_lot",
        "property_address",
        "borough",
        "zip_code",
        "postcode",
        "sold_date",
        "vacant_on_dec31_raw",
        "construction_raw",
        "vacant_midyear_raw",
        "vacant_on_dec31_yes",
        "construction_yes",
        "vacant_midyear_yes",
        "primary_business_activity",
        "business_activity_category",
        "lease_expiration",
        "property_number",
        "property_street",
        "unit",
        "latitude",
        "longitude",
        "community_board",
        "council_district",
        "census_tract",
        "bin",
        "bbl",
        "nta_code",
        "nta_neighborhood",
        "borough_alt",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.dropna(subset=["borough", "latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]

    return df.reset_index(drop=True)


def run_eval_processing(
    storefront_path: (
        str | Path
    ) = "../data/raw/Storefronts_Reported_Vacant_or_Not_20260424.csv",
    ped_raw_path: str | Path = "../data/raw/Bi-Annual_Pedestrian_Counts.csv",
    nbhd_path: str | Path = "../data/raw/Public - Neighborhood Profiles 2018 - All.csv",
    nfhd_raw_path: (
        str | Path
    ) = "../data/raw/Neighborhood_Financial_Health_Digital_Mapping_and_Data_Tool.xlsx",
    boundary_path: str | Path = "../data/raw/nyc_boundaries/nycdta2020.shp",
    shooting_path: str | Path = "../data/raw/NYC_historic_shooting_incidents.csv",
    output_dir: str | Path = "tests/data",
    max_year: int = MAX_YEAR,
    test_ped_year: int = TEST_PED_YEAR,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Year-specific subdirectory for filtered datasets
    year_dir = output_dir / str(max_year)
    year_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating eval snapshots capped at {max_year}...\n")

    boundary_gdf = load_boundaries(boundary_path)
    area_df = compute_area_features(boundary_gdf)

    print("Step 1: pedestrian counts (tests point-in-time)")
    ped_df = clean_pedestrian_data_eval(
        ped_raw_path,
        year=test_ped_year,
        mode="exact",
    )
    ped_out = year_dir / "ped_clean_test.csv"
    ped_df.to_csv(ped_out, index=False)
    print(f"  Written: {ped_out}  ({len(ped_df)} rows)\n")

    print("Step 2: storefront features")
    sf_clean = clean_storefront_data_eval(storefront_path, max_year=max_year)
    sf_joined = spatial_join_points(sf_clean, boundary_gdf)
    sf_feat = build_storefront_features(sf_joined)
    if "act_UNKNOWN_storefront" not in sf_feat.columns:
        keys = ["neighborhood", "cd", "borough"]
        unknown_counts = (
            sf_joined[sf_joined["business_activity_category"] == "UNKNOWN"]
            .groupby(keys, dropna=False)
            .size()
            .reset_index(name="act_UNKNOWN_storefront")
        )
        sf_feat = sf_feat.merge(unknown_counts, on=keys, how="left")
        sf_feat["act_UNKNOWN_storefront"] = (
            sf_feat["act_UNKNOWN_storefront"].fillna(0).astype(int)
        )
        insert_after = "act_RETAIL_storefront"
        if insert_after in sf_feat.columns:
            col = sf_feat.pop("act_UNKNOWN_storefront")
            pos = sf_feat.columns.get_loc(insert_after) + 1
            sf_feat.insert(pos, "act_UNKNOWN_storefront", col)
    sf_out = year_dir / "storefront_features_test.csv"
    sf_feat.to_csv(sf_out, index=False)
    print(f"  Written: {sf_out}  ({len(sf_feat)} rows x {sf_feat.shape[1]} cols)\n")

    print("Step 3: neighborhood profiles (MOCEJ + NFH)")
    nbhd_df = clean_neighborhood_profiles(nbhd_path, nfh_path=nfhd_raw_path)
    nbhd_out = year_dir / "nbhd_clean_test.csv"
    nbhd_df.to_csv(nbhd_out, index=False)
    print(f"  Written: {nbhd_out}  ({len(nbhd_df)} rows x {nbhd_df.shape[1]} cols)\n")

    print("Step 4: shooting incidents")
    shoot_clean = clean_shooting_data_eval(shooting_path, max_year=max_year)
    shoot_joined = spatial_join_points(shoot_clean, boundary_gdf)
    keys = ["neighborhood", "cd", "borough"]

    shooting_counts = (
        shoot_joined.groupby(keys, dropna=False)
        .size()
        .reset_index(name="shooting_incident_count")
    )

    shooting_feat = shooting_counts.merge(
        area_df[keys + ["area_km2"]], on=keys, how="left"
    )
    shooting_feat["shooting_density_per_km2"] = (
        shooting_feat["shooting_incident_count"] / shooting_feat["area_km2"]
    )

    shooting_feat = shooting_feat[keys + ["area_km2", "shooting_incident_count", "shooting_density_per_km2"]]

    shoot_out = year_dir / "shooting_features_test.csv"
    shooting_feat.to_csv(shoot_out, index=False)
    print(f"  Written: {shoot_out}  ({len(shooting_feat)} rows x {shooting_feat.shape[1]} cols)\n")

    print("Done. subway_clean.csv unchanged.")


if __name__ == "__main__":
    run_eval_processing()
