from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

try:
    from data_processing import normalize_cdta_join_key
except ImportError:
    from src.data_processing import normalize_cdta_join_key


WGS84 = "EPSG:4326"
NYC_PROJECTED = "EPSG:2263"
# EPSG:2263 (NY State Plane, ftUS): geometry.area is in square feet, not m².
_SQFT_TO_KM2 = (0.3048**2) / 1_000_000.0


# =========================================================
# Helpers
# =========================================================


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


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

    gdf = gdf.rename(
        columns={
            "CDTAName": "neighborhood",
            "CDTA2020": "cd",
            "BoroName": "borough",
        }
    ).copy()

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
        x, geometry=gpd.points_from_xy(x[lon_col], x[lat_col]), crs=WGS84
    )

    boundary_use = boundary_gdf[["neighborhood", "cd", "borough", "geometry"]].copy()

    # first pass: within
    joined = gpd.sjoin(gdf, boundary_use, how="left", predicate="within")

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
            unmatched, boundary_proj, how="left", distance_col="distance_to_boundary"
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
    drop_cols = [
        c for c in joined.columns if c.endswith("_left") or c.endswith("_right")
    ]
    joined = joined.drop(columns=drop_cols, errors="ignore")

    return joined


# =========================================================
# Geometry features
# =========================================================


def compute_area_features(boundary_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    gdf = boundary_gdf.to_crs(NYC_PROJECTED).copy()
    gdf["area_km2"] = gdf.geometry.area * _SQFT_TO_KM2
    return gdf[["neighborhood", "cd", "borough", "area_km2"]].copy()


# =========================================================
# Pedestrian features
# =========================================================


def build_pedestrian_features(ped_joined: pd.DataFrame) -> pd.DataFrame:
    feat = (
        ped_joined.groupby(["neighborhood", "cd", "borough"], dropna=False)
        .agg(
            avg_pedestrian=("avg_pedestrian", "mean"),
            peak_pedestrian=("peak_pedestrian", "max"),
            pedestrian_count_points=(
                ("street", "count")
                if "street" in ped_joined.columns
                else ("avg_pedestrian", "count")
            ),
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
# NYPD shooting incidents (points → CDTA aggregates)
# =========================================================


def build_shooting_features(shooting_joined: pd.DataFrame) -> pd.DataFrame:
    keys = ["neighborhood", "cd", "borough"]
    id_col = "incident_key" if "incident_key" in shooting_joined.columns else None
    feat = (
        shooting_joined.groupby(keys, dropna=False)
        .agg(
            shooting_incident_count=(
                (id_col, "nunique") if id_col else ("latitude", "count")
            ),
        )
        .reset_index()
    )
    return feat


def build_shooting_neighborhood_features(
    shooting_feat: pd.DataFrame, area_df: pd.DataFrame
) -> pd.DataFrame:
    out = area_df[["neighborhood", "cd", "borough", "area_km2"]].merge(
        shooting_feat, on=["neighborhood", "cd", "borough"], how="left"
    )
    out["shooting_incident_count"] = pd.to_numeric(
        out["shooting_incident_count"], errors="coerce"
    ).fillna(0)
    out["shooting_density_per_km2"] = safe_divide(
        out["shooting_incident_count"], out["area_km2"]
    ).fillna(0)
    return out


# =========================================================
# Storefront filings (points → CDTA aggregates by business activity)
# =========================================================


def is_act_storefront_column(name: str) -> bool:
    s = str(name)
    return s.startswith("act_") and s.endswith("_storefront")


def is_act_density_column(name: str) -> bool:
    s = str(name)
    return s.startswith("act_") and s.endswith("_density")


def storefront_activity_column_name(activity: str) -> str:
    """
    Stable feature column name for one Primary Business Activity category.

    Pattern ``act_<SLUG>_storefront`` (e.g. ``act_RETAIL_storefront``).
    ``other`` (mapped from MISCELLANEOUS OTHER SERVICE) → ``act_other_storefront``.
    """
    label = str(activity).strip()
    if not label:
        return "act_UNKNOWN_storefront"
    if label.lower() == "other":
        return "act_other_storefront"
    t = label.upper().replace("&", "AND").replace("/", "_")
    t = re.sub(r"[^A-Z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        return "act_UNKNOWN_storefront"
    return f"act_{t}_storefront"


def build_storefront_features(storefront_joined: pd.DataFrame) -> pd.DataFrame:
    df = storefront_joined.copy()
    keys = ["neighborhood", "cd", "borough"]
    cat_col = "business_activity_category"
    if cat_col not in df.columns:
        df[cat_col] = "UNKNOWN"

    totals = (
        df.groupby(keys, dropna=False)
        .size()
        .reset_index(name="storefront_filing_count")
    )

    counts = df.groupby(keys + [cat_col], dropna=False).size().reset_index(name="_n")
    counts["_feat_col"] = counts[cat_col].map(storefront_activity_column_name)
    wide = counts.pivot_table(
        index=keys,
        columns="_feat_col",
        values="_n",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    wide.columns.name = None

    feat = totals.merge(wide, on=keys, how="left")
    for c in feat.columns:
        if is_act_storefront_column(c):
            feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0)
    return feat


# =========================================================
# Final merge
# =========================================================


def merge_all_features(
    area_df: pd.DataFrame,
    ped_feat: pd.DataFrame,
    subway_feat: pd.DataFrame,
    shooting_feat: pd.DataFrame,
    nbhd_clean: pd.DataFrame,
    storefront_feat: pd.DataFrame,
) -> pd.DataFrame:
    # Use CDTA-based spatial features as the master table (storefront replaces license/inspection POI).
    df = area_df.copy()
    for other in (ped_feat, subway_feat, shooting_feat, storefront_feat):
        df = df.merge(other, on=["neighborhood", "cd", "borough"], how="left")

    # Neighborhood profile (MOCEJ / Planning-style CSV + optional NFH), keyed by CD ~ CDTA2020.
    # Unmatched CDTAs get NaN on merge; we impute numeric profile columns below (borough then city median).
    df["_cd_join"] = df["cd"].map(normalize_cdta_join_key)
    nb = nbhd_clean.copy()
    nb["_cd_join"] = nb["cd"].map(normalize_cdta_join_key)
    nb = nb.dropna(subset=["_cd_join"]).drop_duplicates(
        subset=["_cd_join"], keep="first"
    )
    _profile_skip = (
        "neighborhood",
        "borough",
        "cd",
        "_cd_join",
        "pct_hispanic",
        "pct_black",
        "pct_asian",
    )
    profile_cols = [c for c in nb.columns if c not in _profile_skip]
    if profile_cols:
        n_with_key = df["_cd_join"].notna().sum()
        df = df.merge(nb[["_cd_join"] + profile_cols], on="_cd_join", how="left")
        probe = profile_cols[0]
        n_matched = int(df[probe].notna().sum()) if probe in df.columns else 0
        if n_with_key > 0 and n_matched < n_with_key * 0.5:
            warnings.warn(
                "Less than half of rows with a parsed CD code matched neighborhood profile data. "
                "Check that raw `Community District` values align with CDTA2020 (see data_processing.normalize_cdta_join_key).",
                stacklevel=2,
            )
    df = df.drop(columns=["_cd_join"])

    # All merged profile columns (MOCEJ counts/income/commute/education, derived pct/totals, nfh_*):
    # borough median then citywide median for remaining NaN (proxy where join missed a CD row).
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

    # Storefront counts (CDTAs with no filings stay NaN until here)
    if "storefront_filing_count" in df.columns:
        df["storefront_filing_count"] = df["storefront_filing_count"].fillna(0)
    for col in df.columns:
        if is_act_storefront_column(col):
            df[col] = df[col].fillna(0)

    act_cols = [c for c in df.columns if is_act_storefront_column(c)]
    if act_cols:
        mat = df[act_cols].to_numpy(dtype=float)
        df["category_diversity"] = (mat > 0).sum(axis=1)
        df["category_entropy"] = [entropy_from_counts(row.astype(float)) for row in mat]
    else:
        df["category_diversity"] = 0
        df["category_entropy"] = 0.0

    # business activity density: share of total filings per category (zero-safe)
    if "storefront_filing_count" in df.columns and act_cols:
        for col in act_cols:
            density_col = col[: -len("_storefront")] + "_density"
            df[density_col] = safe_divide(
                df[col], df["storefront_filing_count"]
            ).fillna(0)

    # density features
    if "area_km2" in df.columns:
        if "subway_station_count" in df.columns:
            df["subway_density_per_km2"] = safe_divide(
                df["subway_station_count"], df["area_km2"]
            )
        if "storefront_filing_count" in df.columns:
            df["storefront_density_per_km2"] = safe_divide(
                df["storefront_filing_count"], df["area_km2"]
            )

    for col in [
        "subway_density_per_km2",
        "storefront_density_per_km2",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in ["subway_station_count"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if "shooting_incident_count" in df.columns:
        df["shooting_incident_count"] = pd.to_numeric(
            df["shooting_incident_count"], errors="coerce"
        ).fillna(0)

    # Pedestrian: avoid a single citywide mean for all missing CDTAs (that collapses values).
    for col in ["avg_pedestrian", "peak_pedestrian"]:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df["borough"].notna().any():
            borough_med = df.groupby("borough")[col].transform("median")
            df[col] = df[col].fillna(borough_med)
        gmed = df[col].median()
        if pd.notna(gmed):
            df[col] = df[col].fillna(gmed)
        df[col] = df[col].fillna(0)

    if "pedestrian_count_points" in df.columns:
        df["pedestrian_count_points"] = df["pedestrian_count_points"].fillna(0)

    if {"avg_pedestrian", "storefront_filing_count"}.issubset(df.columns):
        raw_comm = df["avg_pedestrian"] * df["storefront_filing_count"]
        df["commercial_activity_score"] = np.log1p(np.maximum(raw_comm, 0))
        print(
            f"Computed commercial_activity_score: {df['commercial_activity_score'].describe()}"
        )
        raw_competitive = df["storefront_filing_count"] / (df["avg_pedestrian"] + 1.0)
        df["competitive_score"] = np.log1p(np.maximum(raw_competitive, 0))
        print(f"Computed competitive_score: {df['competitive_score'].describe()}")

    if {"subway_station_count", "avg_pedestrian"}.issubset(df.columns):
        raw_trans = df["subway_station_count"] * df["avg_pedestrian"]
        df["transit_activity_score"] = np.log1p(np.maximum(raw_trans, 0))

    # Keep MOCEJ income as fallback when NFH income is unavailable.
    if "nfh_median_income" in df.columns and "median_household_income" in df.columns:
        df = df.drop(columns=["median_household_income"], errors="ignore")
    _nfh_ranks = [
        c for c in df.columns if str(c).startswith("nfh_") and str(c).endswith("_rank")
    ]
    if _nfh_ranks:
        df = df.drop(columns=_nfh_ranks, errors="ignore")

    # Keep pop counts + total population proxy adjacent for readability
    cols = list(df.columns)
    pop_block = [
        c
        for c in ("pop_black", "pop_hispanic", "pop_asian", "total_population_proxy")
        if c in cols
    ]
    if pop_block:
        block_set = set(pop_block)
        pos = min(cols.index(c) for c in pop_block)
        before = [c for i, c in enumerate(cols) if i < pos and c not in block_set]
        after = [c for i, c in enumerate(cols) if i >= pos and c not in block_set]
        df = df[before + pop_block + after]

    return df


# =========================================================
# Full pipeline
# =========================================================


def empty_storefront_features(area_df: pd.DataFrame) -> pd.DataFrame:
    """Per-CDTA zero row when no raw storefront file is available."""
    out = area_df[["neighborhood", "cd", "borough"]].copy()
    out["storefront_filing_count"] = 0
    return out


def run_feature_engineering(
    *,
    pedestrian_path: str | Path,
    subway_path: str | Path,
    nbhd_clean_path: str | Path,
    boundary_path: str | Path,
    storefront_raw_path: str | Path | None = None,
    shooting_raw_path: str | Path | None = None,
    output_dir: str | Path = "data/processed",
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from data_processing import clean_storefront_data
    except ImportError:
        from src.data_processing import clean_storefront_data

    ped = pd.read_csv(pedestrian_path)
    subway = pd.read_csv(subway_path)
    nbhd = pd.read_csv(nbhd_clean_path)

    boundary_gdf = load_boundaries(boundary_path)
    area_df = compute_area_features(boundary_gdf)

    ped_joined = spatial_join_points(ped, boundary_gdf)
    subway_joined = spatial_join_points(subway, boundary_gdf)
    if shooting_raw_path is not None and Path(shooting_raw_path).exists():
        shooting_raw = pd.read_csv(shooting_raw_path, low_memory=False)
        shooting_joined = spatial_join_points(shooting_raw, boundary_gdf)
        shooting_feat = build_shooting_features(shooting_joined)
    else:
        shooting_feat = area_df[["neighborhood", "cd", "borough"]].copy()
        shooting_feat["shooting_incident_count"] = 0

    if storefront_raw_path is not None and Path(storefront_raw_path).exists():
        sf_clean = clean_storefront_data(storefront_raw_path)
        sf_joined = spatial_join_points(sf_clean, boundary_gdf)
        storefront_feat = build_storefront_features(sf_joined)
    else:
        storefront_feat = empty_storefront_features(area_df)

    ped_feat = build_pedestrian_features(ped_joined)
    subway_feat = build_subway_features(subway_joined)

    final_df = merge_all_features(
        area_df=area_df,
        ped_feat=ped_feat,
        subway_feat=subway_feat,
        shooting_feat=shooting_feat,
        nbhd_clean=nbhd,
        storefront_feat=storefront_feat,
    )

    shooting_neighborhood_feat = build_shooting_neighborhood_features(
        shooting_feat, area_df
    )
    shooting_neighborhood_feat.to_csv(output_dir / "shooting_features.csv", index=False)
    storefront_feat.to_csv(output_dir / "storefront_features.csv", index=False)
    final_df.to_csv(output_dir / "neighborhood_features_final.csv", index=False)

    return {
        "ped_joined": ped_joined,
        "subway_joined": subway_joined,
        "ped_features": ped_feat,
        "subway_features": subway_feat,
        "shooting_features": shooting_neighborhood_feat,
        "storefront_features": storefront_feat,
        "neighborhood_features": final_df,
    }
