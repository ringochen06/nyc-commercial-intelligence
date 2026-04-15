from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# =========================================================
# Basic helpers
# =========================================================

BOROUGH_MAP = {
    "MN": "MANHATTAN",
    "BX": "BRONX",
    "BK": "BROOKLYN",
    "QN": "QUEENS",
    "SI": "STATEN ISLAND",
}


def standardize_borough(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()

    s = s.replace({
        "MN": "MANHATTAN",
        "M": "MANHATTAN",
        "BX": "BRONX",
        "X": "BRONX",
        "BK": "BROOKLYN",
        "B": "BROOKLYN",
        "QN": "QUEENS",
        "Q": "QUEENS",
        "SI": "STATEN ISLAND",
        "S": "STATEN ISLAND",
        "STATEN ISLA": "STATEN ISLAND",
        "MANHATTAN ": "MANHATTAN",
        "BROOKLYN ": "BROOKLYN",
        "QUEENS ": "QUEENS",
        "BRONX ": "BRONX",
    })

    return s


def extract_borough_from_cd(cd: str) -> Optional[str]:
    cd = str(cd)
    for prefix, borough in BOROUGH_MAP.items():
        if prefix in cd:
            return borough
    return None


def clean_numeric_string(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


# =========================================================
# 1. Pedestrian counts
# =========================================================

def clean_pedestrian_data(ped_path: str | Path) -> pd.DataFrame:
    df_ped = pd.read_csv(ped_path)

    if "the_geom" in df_ped.columns:
        geom = df_ped["the_geom"].astype(str)
        lon = pd.to_numeric(geom.str.extract(r"POINT \((-?\d+\.\d+)")[0], errors="coerce")
        lat = pd.to_numeric(geom.str.extract(r"POINT \(-?\d+\.\d+ (\d+\.\d+)\)")[0], errors="coerce")
    else:
        lon = pd.to_numeric(df_ped["longitude"], errors="coerce")
        lat = pd.to_numeric(df_ped["latitude"], errors="coerce")

    ped_cols = [col for col in df_ped.columns if "_AM" in col or "_PM" in col or "_MD" in col]
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
    df_ped = pd.concat([df_ped.drop(columns=drop_lon_lat, errors="ignore"), derived], axis=1)

    keep_cols = [c for c in [
        "Borough",
        "Street",
        "From",
        "To",
        "latitude",
        "longitude",
        "avg_pedestrian",
        "peak_pedestrian"
    ] if c in df_ped.columns]

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

    return df.reset_index(drop=True)


# =========================================================
# 2. Subway station data
# =========================================================

def clean_subway_data(subway_path: str | Path) -> pd.DataFrame:
    df_subway = pd.read_csv(subway_path)

    col_map = {
        "Stop Name": "station_name",
        "Borough": "borough",
        "GTFS Latitude": "latitude",
        "GTFS Longitude": "longitude",
        "Daytime Routes": "routes",
    }

    df = df_subway.rename(columns=col_map).copy()

    required = ["station_name", "borough", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Subway data is missing required columns after renaming: {missing}. "
            f"Available columns: {df_subway.columns.tolist()}"
        )

    keep_cols = ["station_name", "borough", "latitude", "longitude", "routes"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df["borough"] = standardize_borough(df["borough"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["station_name", "borough", "latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    df = df.drop_duplicates(subset=["station_name", "latitude", "longitude"])

    return df.reset_index(drop=True)
# =========================================================
# 3. Restaurant POI
# =========================================================

def clean_restaurant_data(restaurant_path: str | Path) -> pd.DataFrame:
    df_rest = pd.read_csv(restaurant_path)


    name_col = "DBA" if "DBA" in df_rest.columns else "dba"
    borough_col = "BORO" if "BORO" in df_rest.columns else "boro"
    lat_col = "Latitude" if "Latitude" in df_rest.columns else "latitude"
    lon_col = "Longitude" if "Longitude" in df_rest.columns else "longitude"
    cuisine_col = "CUISINE DESCRIPTION" if "CUISINE DESCRIPTION" in df_rest.columns else "cuisine description"

    keep_cols = [c for c in [name_col, cuisine_col, borough_col, lat_col, lon_col] if c in df_rest.columns]
    df = df_rest[keep_cols].copy()

    df = df.rename(columns={
        name_col: "business_name",
        cuisine_col: "category",
        borough_col: "borough",
        lat_col: "latitude",
        lon_col: "longitude",
    })

    df["borough"] = standardize_borough(df["borough"])
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["business_name", "borough", "latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    df = df.drop_duplicates(subset=["business_name", "latitude", "longitude"])

    df["category"] = df["category"].fillna("unknown").astype(str).str.strip().str.lower()
    df["description"] = df["category"] + " restaurant in " + df["borough"]

    return df.reset_index(drop=True)


# =========================================================
# 4. Retail POI from business licensing
# =========================================================

def clean_retail_data(license_path: str | Path) -> pd.DataFrame:
    df_license = pd.read_csv(license_path)
    df_license["Industry"] = df_license["Industry"].astype(str)

    keep_keywords = [
    "store",
    "retail",
    "dealer",
    "shop",
    "electronics",
    "tobacco",
    "secondhand",
    "market",
    "grocery",
    "pharmacy",
    "apparel",
    "clothing",
    "jewelry",
    "furniture",
    "gift",
    "beauty",
    "cosmetics",
    "hardware",
    "home"
    ]
    df = df_license[
        (df_license["License Type"] == "Business") &
        (df_license["License Status"] == "Active") &
        (df_license["Industry"].str.contains("|".join(keep_keywords), case=False, na=False))
    ].copy()

    df = df[[
        "Business Name",
        "Industry",
        "Address Borough",
        "Latitude",
        "Longitude"
    ]].copy()

    df = df.rename(columns={
        "Business Name": "business_name",
        "Industry": "category",
        "Address Borough": "borough",
        "Latitude": "latitude",
        "Longitude": "longitude"
    })

    df["borough"] = standardize_borough(df["borough"])
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["business_name", "borough", "latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    df = df[df["borough"] != "OUTSIDE NYC"]
    df = df.drop_duplicates(subset=["business_name", "latitude", "longitude"])

    df["poi_type"] = "retail"
    df["description"] = df["category"] + " store in " + df["borough"]

    return df.reset_index(drop=True)


def build_poi_table(restaurant_path: str | Path, license_path: str | Path) -> pd.DataFrame:
    df_rest = clean_restaurant_data(restaurant_path)
    df_retail = clean_retail_data(license_path)
    df_poi = pd.concat([df_rest, df_retail], ignore_index=True)
    return df_poi.reset_index(drop=True)


# =========================================================
# 5. Neighborhood profile data
# =========================================================

def clean_neighborhood_profiles(nbhd_path: str | Path) -> pd.DataFrame:
    df_nbhd = pd.read_csv(nbhd_path)

    keep_cols = [
        "Neighborhoods",
        "Community District",
        "2016 Construction",
        "2016 Manufacturing",
        "2016 Wholesale Trade",
        "2016 Black",
        "2016 Hispanic",
        "2016 Asian",
        "2016 Food Services and Drinking Places",
        "2016 Total Number of Businesses",
        "2016 Median Household Income",
        "2016 Commute via Public Transit",
        "2016 Percentage of Population with Bachelor's or Higher",
    ]

    df = df_nbhd[keep_cols].copy()

    df = df.rename(columns={
        "Neighborhoods": "neighborhood",
        "Community District": "cd",
        "2016 Construction": "construction_jobs",
        "2016 Manufacturing": "manufacturing_jobs",
        "2016 Wholesale Trade": "wholesale_jobs",
        "2016 Black": "pop_black",
        "2016 Hispanic": "pop_hispanic",
        "2016 Asian": "pop_asian",
        "2016 Food Services and Drinking Places": "food_services",
        "2016 Total Number of Businesses": "total_businesses",
        "2016 Median Household Income": "median_household_income",
        "2016 Commute via Public Transit": "commute_public_transit",
        "2016 Percentage of Population with Bachelor's or Higher": "pct_bachelors_plus",
    })

    numeric_cols = [
        "construction_jobs",
        "manufacturing_jobs",
        "wholesale_jobs",
        "pop_black",
        "pop_hispanic",
        "pop_asian",
        "food_services",
        "total_businesses",
        "median_household_income",
        "commute_public_transit",
        "pct_bachelors_plus",
    ]

    for col in numeric_cols:
        df[col] = clean_numeric_string(df[col])

    df["borough"] = df["cd"].apply(extract_borough_from_cd)
    df = df[df["borough"].notna()]
    df = df[df["cd"].astype(str).str.contains("BX|BK|MN|QN|SI", na=False)]
    df = df.drop_duplicates(subset=["cd"])

    df["total_jobs"] = (
        df["construction_jobs"].fillna(0) +
        df["manufacturing_jobs"].fillna(0) +
        df["wholesale_jobs"].fillna(0)
    )

    df["total_population_proxy"] = (
        df["pop_black"].fillna(0) +
        df["pop_hispanic"].fillna(0) +
        df["pop_asian"].fillna(0)
    )

    denom = df["total_population_proxy"].replace(0, np.nan)
    df["pct_hispanic"] = df["pop_hispanic"] / denom
    df["pct_black"] = df["pop_black"] / denom
    df["pct_asian"] = df["pop_asian"] / denom

    return df.reset_index(drop=True)


# =========================================================
# 6. Save everything
# =========================================================

def run_data_processing(
    *,
    pedestrian_path: str | Path,
    subway_path: str | Path,
    restaurant_path: str | Path,
    license_path: str | Path,
    nbhd_path: str | Path,
    output_dir: str | Path = "data/processed",
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ped = clean_pedestrian_data(pedestrian_path)
    subway = clean_subway_data(subway_path)
    poi = build_poi_table(restaurant_path, license_path)
    nbhd = clean_neighborhood_profiles(nbhd_path)

    ped.to_csv(output_dir / "ped_clean.csv", index=False)
    subway.to_csv(output_dir / "subway_clean.csv", index=False)
    poi.to_csv(output_dir / "poi_clean.csv", index=False)
    nbhd.to_csv(output_dir / "nbhd_clean.csv", index=False)

    return {
        "pedestrian": ped,
        "subway": subway,
        "poi": poi,
        "neighborhood": nbhd,
    }
