"""Pydantic request/response models for the FastAPI backend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Vintage = Literal["present", "past"]


class ClusterRequest(BaseModel):
    features: list[str] = Field(..., min_length=1, description="Feature column names used for clustering.")
    boroughs: list[str] | None = Field(None, description="If set, restrict to these boroughs.")
    max_k: int = Field(8, ge=2, le=15)
    vintage: Vintage = "present"
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
    elbow_k: int
    elbow_k_kneedle: int
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
    min_nfh_goal4: float | None = None
    min_nfh_overall: float | None = None


class RankRequest(BaseModel):
    query: str = Field("quiet residential area suitable for boutique retail with good subway access")
    alpha: float = Field(0.8, ge=0.0, le=1.0, description="Semantic weight; commercial weight = 1 - alpha.")
    filters: HardFilters = HardFilters()
    vintage: Vintage = "present"
    cluster_assignments: dict[str, int] | None = None
    cluster_briefs: dict[str, str] | None = None


class RankRow(BaseModel):
    rank: int
    neighborhood: str
    cd: str | None = None
    borough: str | None = None
    map_key: str | None = None
    semantic_similarity: float
    commercial_activity_score: float
    blended_score: float
    cluster: int | None = None
    cluster_description: str | None = None


class RankResponse(BaseModel):
    rows: list[RankRow]
    n_total: int
    n_filtered: int
    sql: str


class FeatureRange(BaseModel):
    min: float
    max: float


class FeatureRangesResponse(BaseModel):
    boroughs: list[str]
    ranges: dict[str, FeatureRange]
    has_nfh_goal4: bool
    has_nfh_overall: bool
    activity_columns: list[str]
    density_columns: list[str]
