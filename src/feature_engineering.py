from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd


WGS84 = "EPSG:4326"
NYC_PROJECTED = "EPSG:2263"


# =========================================================
# Helpers
# =========================================================

def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def simplify_category(cat: str) -> str:
    cat = str(cat).lower()

    if any(k in cat for k in ["pizza", "restaurant", "chinese", "italian", "mexican", "coffee"]):
        return "food"
    elif any(k in cat for k in ["electronics", "tobacco", "store", "shop", "dealer", "retail"]):
        return "retail"
    else:
        return "other"


def entropy_from_counts(values: np.ndarray) -> float:
    values = values.astype(float)
    total = values.sum()
    if total <= 0:
        return 0.0
    p = values[values > 0] / total
    return float(-(p * np.log(p)).sum())


# =========================================================
# Boundary loader
# =========================================================

def load_boundaries(boundary_path: str | Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(boundary_path)

    if "geometry" not in gdf.columns:
        raise ValueError("Boundary file must contain geometry.")

    required = ["CDTAName", "CDTA2020", "BoroName", "geometry"]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(
            f"Boundary file is missing required columns: {missing}. "
            f"Available columns: {gdf.columns.tolist()}"
        )

    gdf = gdf.rename(columns={
        "CDTAName": "neighborhood",
        "CDTA2020": "cd",
        "BoroName": "borough",
    }).copy()

    gdf["neighborhood"] = gdf["neighborhood"].astype(str).str.strip()
    gdf["cd"] = gdf["cd"].astype(str).str.strip()
    gdf["borough"] = gdf["borough"].astype(str).str.strip().str.upper()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:2263")
    gdf = gdf.to_crs("EPSG:4326")

    return gdf[["neighborhood", "cd", "borough", "geometry"]].copy()

# =========================================================
# Spatial join
# =========================================================

def spatial_join_points(
    df: pd.DataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> gpd.GeoDataFrame:
    x = df.copy()
    x[lat_col] = pd.to_numeric(x[lat_col], errors="coerce")
    x[lon_col] = pd.to_numeric(x[lon_col], errors="coerce")
    x = x.dropna(subset=[lat_col, lon_col])

    gdf = gpd.GeoDataFrame(
        x,
        geometry=gpd.points_from_xy(x[lon_col], x[lat_col]),
        crs=WGS84
    )

    boundary_use = boundary_gdf[["neighborhood", "cd", "borough", "geometry"]].copy()

    # first pass: within
    joined = gpd.sjoin(
        gdf,
        boundary_use,
        how="left",
        predicate="within"
    )

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    # fix possible suffixes after sjoin
    for base_col in ["neighborhood", "cd", "borough"]:
        if base_col not in joined.columns:
            right_col = f"{base_col}_right"
            left_col = f"{base_col}_left"

            if right_col in joined.columns:
                joined[base_col] = joined[right_col]
            elif left_col in joined.columns:
                joined[base_col] = joined[left_col]

    # nearest fallback for unmatched rows
    missing = joined["neighborhood"].isna()
    if missing.any():
        unmatched = gdf.loc[missing].copy().to_crs(NYC_PROJECTED)
        boundary_proj = boundary_use.to_crs(NYC_PROJECTED)

        nearest = gpd.sjoin_nearest(
            unmatched,
            boundary_proj,
            how="left",
            distance_col="distance_to_boundary"
        )

        # fix suffixes in nearest result
        for base_col in ["neighborhood", "cd", "borough"]:
            if base_col not in nearest.columns:
                right_col = f"{base_col}_right"
                left_col = f"{base_col}_left"

                if right_col in nearest.columns:
                    nearest[base_col] = nearest[right_col]
                elif left_col in nearest.columns:
                    nearest[base_col] = nearest[left_col]

        joined.loc[missing, "neighborhood"] = nearest["neighborhood"].values
        joined.loc[missing, "cd"] = nearest["cd"].values
        joined.loc[missing, "borough"] = nearest["borough"].values

    # optional cleanup of suffixed columns
    drop_cols = [c for c in joined.columns if c.endswith("_left") or c.endswith("_right")]
    joined = joined.drop(columns=drop_cols, errors="ignore")

    return joined


# =========================================================
# Geometry features
# =========================================================

def compute_area_features(boundary_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    gdf = boundary_gdf.to_crs(NYC_PROJECTED).copy()
    gdf["area_km2"] = gdf.geometry.area / 1_000_000
    return gdf[["neighborhood", "cd", "borough", "area_km2"]].copy()


# =========================================================
# POI features
# =========================================================

def build_poi_features(poi_joined: pd.DataFrame) -> pd.DataFrame:
    df = poi_joined.copy()
    df["simple_category"] = df["category"].apply(simplify_category)

    base = (
        df.groupby(["neighborhood", "cd", "borough"], dropna=False)
        .agg(
            total_poi=("business_name", "count"),
            unique_poi=("business_name", "nunique"),
            category_diversity=("simple_category", "nunique"),
        )
        .reset_index()
    )

    poi_type = (
        df.groupby(["neighborhood", "cd", "borough", "poi_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    rename_map = {}
    for c in poi_type.columns:
        if c not in ["neighborhood", "cd", "borough"]:
            rename_map[c] = f"num_{c}"
    poi_type = poi_type.rename(columns=rename_map)

    cat_counts = (
        df.groupby(["neighborhood", "cd", "borough", "simple_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    feat = base.merge(poi_type, on=["neighborhood", "cd", "borough"], how="left")
    feat = feat.merge(cat_counts, on=["neighborhood", "cd", "borough"], how="left", suffixes=("", "_cat"))

    # ratio
    for col in feat.columns:
        if col.startswith("num_"):
            feat[col.replace("num_", "ratio_")] = safe_divide(feat[col], feat["total_poi"])

    # entropy
    category_cols = [c for c in ["food", "retail", "other"] if c in feat.columns]
    if category_cols:
        feat["category_entropy"] = feat[category_cols].apply(
            lambda row: entropy_from_counts(row.values),
            axis=1
        )
    else:
        feat["category_entropy"] = 0.0

    if "num_restaurant" in feat.columns and "num_retail" in feat.columns:
        feat["food_to_retail_ratio"] = safe_divide(feat["num_restaurant"], feat["num_retail"])

    return feat


# =========================================================
# Pedestrian features
# =========================================================

def build_pedestrian_features(ped_joined: pd.DataFrame) -> pd.DataFrame:
    feat = (
        ped_joined.groupby(["neighborhood", "cd", "borough"], dropna=False)
        .agg(
            avg_pedestrian=("avg_pedestrian", "mean"),
            peak_pedestrian=("peak_pedestrian", "max"),
            pedestrian_count_points=("street", "count") if "street" in ped_joined.columns else ("avg_pedestrian", "count"),
        )
        .reset_index()
    )
    return feat


# =========================================================
# Subway features
# =========================================================

def build_subway_features(subway_joined: pd.DataFrame) -> pd.DataFrame:
    feat = (
        subway_joined.groupby(["neighborhood", "cd", "borough"], dropna=False)
        .agg(
            subway_station_count=("station_name", "nunique"),
        )
        .reset_index()
    )
    return feat


# =========================================================
# Final merge
# =========================================================

def merge_all_features(
    area_df: pd.DataFrame,
    poi_feat: pd.DataFrame,
    ped_feat: pd.DataFrame,
    subway_feat: pd.DataFrame,
    nbhd_clean: pd.DataFrame,
) -> pd.DataFrame:
    # Use CDTA-based spatial features as the master table
    df = area_df.copy()

    for other in [poi_feat, ped_feat, subway_feat]:
        df = df.merge(other, on=["neighborhood", "cd", "borough"], how="left")

    # density features
    if "area_km2" in df.columns:
        if "total_poi" in df.columns:
            df["poi_density_per_km2"] = safe_divide(df["total_poi"], df["area_km2"])
        if "num_restaurant" in df.columns:
            df["restaurant_density_per_km2"] = safe_divide(df["num_restaurant"], df["area_km2"])
        if "num_retail" in df.columns:
            df["retail_density_per_km2"] = safe_divide(df["num_retail"], df["area_km2"])
        if "subway_station_count" in df.columns:
            df["subway_density_per_km2"] = safe_divide(df["subway_station_count"], df["area_km2"])

    # interaction features
    if {"avg_pedestrian", "total_poi"}.issubset(df.columns):
        df["commercial_activity_score"] = (
            df["avg_pedestrian"].fillna(0) * df["total_poi"].fillna(0)
        )

    if {"subway_station_count", "avg_pedestrian"}.issubset(df.columns):
        df["transit_activity_score"] = (
            df["subway_station_count"].fillna(0) * df["avg_pedestrian"].fillna(0)
        )

    # ========= POI =========
    for col in [
        "total_poi",
        "unique_poi",
        "category_diversity",
        "category_entropy",
        "poi_density_per_km2",
        "food",
        "other",
        "retail",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ========= retail =========
    for col in ["num_retail", "ratio_retail", "retail_density_per_km2"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ========= subway =========
    for col in ["subway_station_count", "subway_density_per_km2"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ========= pedestrian =========
    for col in ["avg_pedestrian", "peak_pedestrian"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    if "pedestrian_count_points" in df.columns:
        df["pedestrian_count_points"] = df["pedestrian_count_points"].fillna(0)

    return df

# =========================================================
# Full pipeline
# =========================================================

def run_feature_engineering(
    *,
    poi_path: str | Path,
    pedestrian_path: str | Path,
    subway_path: str | Path,
    nbhd_clean_path: str | Path,
    boundary_path: str | Path,
    output_dir: str | Path = "data/processed",
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    poi = pd.read_csv(poi_path)
    ped = pd.read_csv(pedestrian_path)
    subway = pd.read_csv(subway_path)
    nbhd = pd.read_csv(nbhd_clean_path)

    boundary_gdf = load_boundaries(boundary_path)

    poi_joined = spatial_join_points(poi, boundary_gdf)
    ped_joined = spatial_join_points(ped, boundary_gdf)
    subway_joined = spatial_join_points(subway, boundary_gdf)

    area_df = compute_area_features(boundary_gdf)
    poi_feat = build_poi_features(poi_joined)
    ped_feat = build_pedestrian_features(ped_joined)
    subway_feat = build_subway_features(subway_joined)

    final_df = merge_all_features(
        area_df=area_df,
        poi_feat=poi_feat,
        ped_feat=ped_feat,
        subway_feat=subway_feat,
        nbhd_clean=nbhd,
    )

    poi_joined.drop(columns="geometry", errors="ignore").to_csv(output_dir / "poi_with_neighborhood.csv", index=False)
    ped_joined.drop(columns="geometry", errors="ignore").to_csv(output_dir / "ped_with_neighborhood.csv", index=False)
    subway_joined.drop(columns="geometry", errors="ignore").to_csv(output_dir / "subway_with_neighborhood.csv", index=False)

    poi_feat.to_csv(output_dir / "poi_features.csv", index=False)
    ped_feat.to_csv(output_dir / "ped_features.csv", index=False)
    subway_feat.to_csv(output_dir / "subway_features.csv", index=False)
    final_df.to_csv(output_dir / "neighborhood_features.csv", index=False)

    return {
        "poi_joined": poi_joined,
        "ped_joined": ped_joined,
        "subway_joined": subway_joined,
        "poi_features": poi_feat,
        "ped_features": ped_feat,
        "subway_features": subway_feat,
        "neighborhood_features": final_df,
    }
