"""Pydantic request/response models for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Vintage = Literal["present"]


class ClusterRequest(BaseModel):
    features: list[str] = Field(..., min_length=1, description="Feature column names used for clustering.")
    boroughs: list[str] | None = Field(None, description="If set, restrict to these boroughs.")
    max_k: int = Field(
        8,
        ge=2,
        le=15,
        description="Upper bound for the k sweep (k runs from 2 through min(max_k, n − 1)).",
    )
    chosen_k: int | None = Field(
        None,
        description="Number of clusters for assignments / maps / summaries. Must lie in the swept k range. "
        "If omitted, falls back to the inertia elbow heuristic.",
    )
    vintage: Vintage = Field(
        "present",
        description='Feature snapshot. Only "present" is implemented (reads neighborhood_features_final.csv).',
    )
    random_state: int = 42


class ClusterPoint(BaseModel):
    neighborhood: str
    cd: str | None = None
    borough: str | None = None
    map_key: str | None = None
    cluster: int
    raw: dict[str, float]


class ClusterSummary(BaseModel):
    cluster: int
    size: int
    description: str
    centroid_z: list[float]


class ClusterResponse(BaseModel):
    k_range: list[int]
    inertias: list[float]
    silhouettes_numpy: list[float]
    silhouettes_sklearn: list[float]
    elbow_k: int = Field(..., description="k from max perpendicular distance to the inertia chord (normalized axes).")
    elbow_k_kneedle: int = Field(
        ...,
        description="Alternative elbow: k where |Δ² inertia| is largest on the normalized inertia curve.",
    )
    best_silhouette_k: int
    chosen_k: int
    features: list[str]
    feature_means: list[float]
    feature_stds: list[float]
    points: list[ClusterPoint]
    centroids_z: list[list[float]]
    cluster_summaries: list[ClusterSummary]


class HardFilters(BaseModel):
    boroughs: list[str] | None = None
    min_subway_stations: float | None = None
    min_avg_pedestrian: float | None = None
    min_storefront_density: float | None = None
    min_storefront_filings: float | None = None
    min_commercial_activity: float | None = None
    max_competitive_score: float | None = None
    max_shooting_incident_count: float | None = None
    min_nfh_goal4: float | None = None
    min_nfh_overall: float | None = None


class RankRequest(BaseModel):
    query: str = Field(
        "quiet residential area suitable for retail with good subway access and good NFH stability"
    )
    alpha: float = Field(0.8, ge=0.0, le=1.0, description="Semantic weight; competitive weight = 1 - alpha.")
    filters: HardFilters = HardFilters()
    vintage: Vintage = Field(
        "present",
        description='Feature snapshot. Only "present" is implemented (reads neighborhood_features_final.csv).',
    )
    competitive_source: str = Field(
        "__overall__",
        description=(
            "'__overall__' uses storefront_filing_count for the per-row competitive penalty; "
            "any 'act_*_storefront' name uses that single category's filings instead."
        ),
    )
    cluster_assignments: dict[str, int] | None = None
    cluster_briefs: dict[str, str] | None = None


class RankRow(BaseModel):
    rank: int
    neighborhood: str
    cd: str | None = None
    borough: str | None = None
    map_key: str | None = None
    semantic_similarity: float
    specific_competitive_score: float
    blended_score: float
    cluster: int | None = None
    cluster_description: str | None = None


class RankResponse(BaseModel):
    rows: list[RankRow]
    n_total: int
    n_filtered: int
    sql: str


class FilterRequest(BaseModel):
    filters: HardFilters = HardFilters()
    vintage: Vintage = Field(
        "present",
        description='Feature snapshot. Only "present" is implemented.',
    )


class FilterResponse(BaseModel):
    rows: list[dict[str, Any]] = Field(
        ...,
        description="Hard-filtered rows from the feature table, in CSV column order.",
    )
    n_total: int
    n_filtered: int
    sql: str


class FeatureRange(BaseModel):
    min: float
    max: float


class FeatureRangesResponse(BaseModel):
    boroughs: list[str]
    ranges: dict[str, FeatureRange]
    has_nfh_goal4: bool = False
    has_nfh_overall: bool = False
    activity_columns: list[str] = Field(
        default_factory=list,
        description="act_*_storefront columns present in the feature CSV (for the competitive-source picker).",
    )
