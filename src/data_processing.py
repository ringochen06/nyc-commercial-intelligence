from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import re


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

    s = s.replace(
        {
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
        }
    )

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
        errors="coerce",
    )


def normalize_cd_code(cd: str | float | None) -> Optional[str]:
    """
    Normalize CD / CDTA-style strings to DCP-style keys (e.g. MN01, BK18).

    Keep logic aligned with ``feature_engineering.normalize_cdta_join_key`` so
    NFH merges and the CDTA master merge use the same join key.
    """
    if cd is None or (isinstance(cd, float) and pd.isna(cd)):
        return None
    s = str(cd).strip().upper()
    if not s or s == "NAN":
        return None

    if re.fullmatch(r"\d{3}", s):
        boro_digit, dist_str = s[0], s[1:3]
        bmap = {"1": "MN", "2": "BX", "3": "BK", "4": "QN", "5": "SI"}
        pref = bmap.get(boro_digit)
        if pref is None:
            return None
        dist = int(dist_str)
        if dist < 1:
            return None
        return f"{pref}{dist:02d}"

    m = re.match(r"^(BX|BK|MN|QN|SI)\s*0*(\d{1,2})$", s)
    if m:
        return f"{m.group(1)}{int(m.group(2)):02d}"
    m = re.search(r"(BX|BK|MN|QN|SI)\s*0*(\d{1,2})\b", s)
    if m:
        return f"{m.group(1)}{int(m.group(2)):02d}"

    # NFH / Planning exports: "BX Community District 8", "BX Community Districts 3 & 6"
    m = re.search(
        r"(BX|BK|MN|QN|SI)\s+COMMUNITY\s+DISTRICTS?\s+(\d{1,2})(?:\s*&\s*(\d{1,2}))?",
        s,
    )
    if m:
        pref = m.group(1).upper()
        dist = int(m.group(2))
        return f"{pref}{dist:02d}"

    return None


# =========================================================
# 1. Pedestrian counts
# =========================================================


def clean_pedestrian_data(ped_path: str | Path) -> pd.DataFrame:
    df_ped = pd.read_csv(ped_path)

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

    ped_cols = [
        col for col in df_ped.columns if "_AM" in col or "_PM" in col or "_MD" in col
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
    cuisine_col = (
        "CUISINE DESCRIPTION"
        if "CUISINE DESCRIPTION" in df_rest.columns
        else "cuisine description"
    )

    keep_cols = [
        c
        for c in [name_col, cuisine_col, borough_col, lat_col, lon_col]
        if c in df_rest.columns
    ]
    df = df_rest[keep_cols].copy()

    df = df.rename(
        columns={
            name_col: "business_name",
            cuisine_col: "category",
            borough_col: "borough",
            lat_col: "latitude",
            lon_col: "longitude",
        }
    )

    df["borough"] = standardize_borough(df["borough"])
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["business_name", "borough", "latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    df = df.drop_duplicates(subset=["business_name", "latitude", "longitude"])

    df["category"] = (
        df["category"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    # DOHMH rows are food-service inspections; drives `simplify_category` default to food vs other.
    df["poi_type"] = "restaurant"
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
        "home",
    ]
    df = df_license[
        (df_license["License Type"] == "Business")
        & (df_license["License Status"] == "Active")
        & (
            df_license["Industry"].str.contains(
                "|".join(keep_keywords), case=False, na=False
            )
        )
    ].copy()

    df = df[
        ["Business Name", "Industry", "Address Borough", "Latitude", "Longitude"]
    ].copy()

    df = df.rename(
        columns={
            "Business Name": "business_name",
            "Industry": "category",
            "Address Borough": "borough",
            "Latitude": "latitude",
            "Longitude": "longitude",
        }
    )

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


def build_poi_table(
    restaurant_path: str | Path, license_path: str | Path
) -> pd.DataFrame:
    df_rest = clean_restaurant_data(restaurant_path)
    df_retail = clean_retail_data(license_path)
    df_poi = pd.concat([df_rest, df_retail], ignore_index=True)
    return df_poi.reset_index(drop=True)


# =========================================================
# 5. Neighborhood profile data
# =========================================================


def clean_nfh_profiles(nfh_path: str | Path) -> pd.DataFrame:
    """
    Build one row per CD with selected Neighborhood Financial Health features.
    """
    df_nfh = pd.read_csv(nfh_path)

    base_cols = [
        "CD",
        "Median_Income",
        "NYC_Poverty_Rate",
        "Perc_White",
        "Perc_Black",
        "Perc_Asian",
        "Perc_Hispanic",
    ]
    goal_cols = ["CD", "Goal", "IndexScore", "GoalRank"]
    if not set(base_cols).issubset(df_nfh.columns):
        return pd.DataFrame(columns=["cd"])
    if not set(goal_cols).issubset(df_nfh.columns):
        return pd.DataFrame(columns=["cd"])

    goal_map = {
        "Overall Index": "nfh_overall",
        "Financial Services": "nfh_goal1_fin_services",
        "Goods & Services": "nfh_goal2_goods_services",
        "Jobs & Income": "nfh_goal3_jobs_income",
        "Financial Shocks": "nfh_goal4_fin_shocks",
        "Build Assets": "nfh_goal5_build_assets",
    }

    base = df_nfh[base_cols].copy()
    base["cd"] = base["CD"].map(normalize_cd_code)
    base = base.dropna(subset=["cd"]).drop_duplicates(subset=["cd"], keep="first")
    base = base.rename(
        columns={
            "Median_Income": "nfh_median_income",
            "NYC_Poverty_Rate": "nfh_poverty_rate",
            "Perc_White": "nfh_pct_white",
            "Perc_Black": "nfh_pct_black",
            "Perc_Asian": "nfh_pct_asian",
            "Perc_Hispanic": "nfh_pct_hispanic",
        }
    )
    for col in [
        "nfh_median_income",
        "nfh_poverty_rate",
        "nfh_pct_white",
        "nfh_pct_black",
        "nfh_pct_asian",
        "nfh_pct_hispanic",
    ]:
        base[col] = clean_numeric_string(base[col])

    goals = df_nfh[goal_cols].copy()
    goals["cd"] = goals["CD"].map(normalize_cd_code)
    goals = goals.dropna(subset=["cd"])
    goals = goals[goals["Goal"].isin(goal_map.keys())].copy()
    goals["metric"] = goals["Goal"].map(goal_map)
    goals["IndexScore"] = clean_numeric_string(goals["IndexScore"])
    goals["GoalRank"] = clean_numeric_string(goals["GoalRank"])

    score_wide = goals.pivot_table(
        index="cd", columns="metric", values="IndexScore", aggfunc="first"
    ).reset_index()
    score_wide = score_wide.rename(
        columns={c: f"{c}_score" for c in score_wide.columns if c != "cd"}
    )

    rank_wide = goals.pivot_table(
        index="cd", columns="metric", values="GoalRank", aggfunc="first"
    ).reset_index()
    rank_wide = rank_wide.rename(
        columns={c: f"{c}_rank" for c in rank_wide.columns if c != "cd"}
    )

    out = base.merge(score_wide, on="cd", how="left").merge(
        rank_wide, on="cd", how="left"
    )
    out = out.drop(columns=["CD"], errors="ignore")
    return out.reset_index(drop=True)


def clean_neighborhood_profiles(
    nbhd_path: str | Path, nfh_path: str | Path | None = None
) -> pd.DataFrame:
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
        "2016 Employed",
        "2016 Commute via Public Transit",
        "2016 Percentage of Population with Bachelor's or Higher",
    ]

    df = df_nbhd[keep_cols].copy()

    df = df.rename(
        columns={
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
            "2016 Employed": "employed_2016",
            "2016 Commute via Public Transit": "commute_public_transit",
            "2016 Percentage of Population with Bachelor's or Higher": "pct_bachelors_plus",
        }
    )

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
        "employed_2016",
        "commute_public_transit",
        "pct_bachelors_plus",
    ]

    for col in numeric_cols:
        df[col] = clean_numeric_string(df[col])

    # MOCEJ publishes transit commuters as a count; convert to % of 2016 employed (same table).
    emp = df["employed_2016"]
    comm = df["commute_public_transit"]
    df["commute_public_transit"] = np.where(
        emp.notna() & (emp > 0) & comm.notna(),
        100.0 * comm / emp,
        np.nan,
    )
    df = df.drop(columns=["employed_2016"])

    df["borough"] = df["cd"].apply(extract_borough_from_cd)
    df = df[df["borough"].notna()]
    df = df[df["cd"].astype(str).str.contains("BX|BK|MN|QN|SI", na=False)]
    df = df.drop_duplicates(subset=["cd"])

    df["total_jobs"] = (
        df["construction_jobs"].fillna(0)
        + df["manufacturing_jobs"].fillna(0)
        + df["wholesale_jobs"].fillna(0)
    )

    df["total_population_proxy"] = (
        df["pop_black"].fillna(0)
        + df["pop_hispanic"].fillna(0)
        + df["pop_asian"].fillna(0)
    )

    denom = df["total_population_proxy"].replace(0, np.nan)
    df["pct_hispanic"] = df["pop_hispanic"] / denom
    df["pct_black"] = df["pop_black"] / denom
    df["pct_asian"] = df["pop_asian"] / denom

    if nfh_path is not None and Path(nfh_path).exists():
        nfh = clean_nfh_profiles(nfh_path)
        if not nfh.empty:
            df["_cd_norm"] = df["cd"].map(normalize_cd_code)
            df = df.merge(
                nfh.rename(columns={"cd": "_cd_norm"}),
                on="_cd_norm",
                how="left",
            )
            df = df.drop(columns=["_cd_norm"])

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
    nfh_path: str | Path | None = None,
    output_dir: str | Path = "data/processed",
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ped = clean_pedestrian_data(pedestrian_path)
    subway = clean_subway_data(subway_path)
    poi = build_poi_table(restaurant_path, license_path)
    nbhd = clean_neighborhood_profiles(nbhd_path, nfh_path=nfh_path)

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
