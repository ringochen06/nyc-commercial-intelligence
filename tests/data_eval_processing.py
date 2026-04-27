"""
data_eval_processing.py

Generates ped_clean_test.csv and storefront_features_test.csv containing only
data up to and including 2023. Run this script for evaluation snapshots capped at
2023 without touching the main processed files used by the app.

Outputs:
    data/processed/ped_clean_test.csv
    data/processed/storefront_features_test.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing import clean_neighborhood_profiles, standardize_borough
from src.feature_engineering import build_storefront_features, load_boundaries, spatial_join_points

MAX_YEAR = 2023


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
    max_year: int = MAX_YEAR,
) -> pd.DataFrame:
    """Clean pedestrian counts using only time-series columns from years <= max_year."""
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
    ped_cols = [
        col for col in all_ped_cols if (parse_ped_col_year(col) or 9999) <= max_year
    ]

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
        for c in ["Borough", "Street", "From", "To", "latitude", "longitude", "avg_pedestrian", "peak_pedestrian"]
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

    print(f"  Pedestrian columns used (year <= {max_year}): {len(ped_cols)} of {len(all_ped_cols)}")
    if ped_cols:
        print(f"  Earliest: {ped_cols[0]}  Latest: {ped_cols[-1]}")

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
    print(f"  Storefront rows: {n_before:,} → {len(df_raw):,} after year filter (<= {max_year})")

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
    df.loc[act_upper == "MISCELLANEOUS OTHER SERVICE", "business_activity_category"] = "other"
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


def enrich_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process the merged final features DataFrame.

    1. Adds ``total_poi`` column sourced from ``total_businesses`` (MOCEJ 2016
       neighborhood profile) so downstream consumers have a stable POI proxy.
    2. For neighborhoods where storefront data is absent (``storefront_filing_count``
       is 0 or NaN), fills ``storefront_density_per_km2`` using
       ``total_poi / area_km2`` so every row has a non-null density estimate.
    """
    df = df.copy()

    # total_poi: stable alias for the MOCEJ total-businesses POI proxy
    if "total_businesses" in df.columns:
        df["total_poi"] = pd.to_numeric(df["total_businesses"], errors="coerce").fillna(0)
    else:
        df["total_poi"] = 0.0

    # Fill storefront_density_per_km2 for neighborhoods with no storefront filings
    if "storefront_density_per_km2" in df.columns and "area_km2" in df.columns:
        missing_mask = (
            df["storefront_filing_count"].fillna(0) == 0
        ) & (df["area_km2"].fillna(0) > 0)
        fallback_density = df["total_poi"] / df["area_km2"].replace(0, float("nan"))
        fallback_filing_count = df["total_poi"]
        df.loc[missing_mask, "storefront_density_per_km2"] = fallback_density[missing_mask]
        df.loc[missing_mask, "storefront_filing_count"] = fallback_filing_count[missing_mask]
        n_filled = int(missing_mask.sum())
        if n_filled:
            print(
                f"  storefront_density_per_km2: filled {n_filled} neighborhoods "
                "using total_poi / area_km2 (no storefront filings)"
            )

    return df


def run_eval_processing(
    storefront_path: str | Path = "../data/raw/Storefronts_Reported_Vacant_or_Not_20260424.csv",
    ped_raw_path: str | Path = "../data/raw/Bi-Annual_Pedestrian_Counts.csv",
    nbhd_path: str | Path = "../data/raw/Public - Neighborhood Profiles 2018 - All.csv",
    nfhd_raw_path: str | Path = "../data/raw/Neighborhood_Financial_Health_Digital_Mapping_and_Data_Tool.xlsx",
    boundary_path: str | Path = "../data/raw/nyc_boundaries/nycdta2020.shp",
    output_dir: str | Path = "tests/data",
    max_year: int = MAX_YEAR,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating eval snapshots capped at {max_year}...\n")

    print("Step 1: pedestrian counts")
    ped_df = clean_pedestrian_data_eval(ped_raw_path, max_year=max_year)
    ped_out = output_dir / "ped_clean_test.csv"
    ped_df.to_csv(ped_out, index=False)
    print(f"  Written: {ped_out}  ({len(ped_df)} rows)\n")

    print("Step 2: storefront features")
    boundary_gdf = load_boundaries(boundary_path)
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
    sf_out = output_dir / "storefront_features_test.csv"
    sf_feat.to_csv(sf_out, index=False)
    print(f"  Written: {sf_out}  ({len(sf_feat)} rows x {sf_feat.shape[1]} cols)\n")

    print("Step 3: neighborhood profiles (MOCEJ + NFH)")
    nbhd_df = clean_neighborhood_profiles(nbhd_path, nfh_path=nfhd_raw_path)
    nbhd_out = output_dir / "nbhd_clean_test.csv"
    nbhd_df.to_csv(nbhd_out, index=False)
    print(f"  Written: {nbhd_out}  ({len(nbhd_df)} rows x {nbhd_df.shape[1]} cols)\n")

    print("Done. subway_clean.csv unchanged.")


if __name__ == "__main__":
    run_eval_processing()
