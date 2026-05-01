"""Constants and display labels for the Streamlit clustering app."""

from __future__ import annotations

BASE_CANDIDATE_FEATURES: list[str] = [
    "storefront_filing_count",
    "avg_pedestrian",
    "subway_station_count",
    "storefront_density_per_km2",
    "commercial_activity_score",
    "competitive_score",
    "shooting_incident_count",
    "transit_activity_score",
    "category_entropy",
    "category_diversity",
    "peak_pedestrian",
    "subway_density_per_km2",
    "nfh_overall_score",
    "nfh_goal4_fin_shocks_score",
    "total_jobs",
]

BASE_DEFAULT_FEATURES: list[str] = [
    "storefront_filing_count",
    "avg_pedestrian",
    "subway_station_count",
    "storefront_density_per_km2",
    "commercial_activity_score",
    "competitive_score",
    "shooting_incident_count",
    "transit_activity_score",
    "category_entropy",
    "nfh_overall_score",
    "total_jobs",
]

ACTIVITY_LABEL_REPLACEMENTS: dict[str, str] = {
    "accounting services": "accounting services",
    "broadcasting telecomm": "broadcasting and telecom",
    "educational services": "education",
    "finance and insurance": "finance and insurance",
    "food services": "food service",
    "health care or social assistance": "health care and social assistance",
    "information services": "information services",
    "legal services": "legal services",
    "manufacturing": "manufacturing",
    "movies video sound": "media and entertainment",
    "no business activity identified": "no identified business activity",
    "publishing": "publishing",
    "real estate": "real estate",
    "retail": "retail",
    "unknown": "unknown activity",
    "wholesale": "wholesale",
    "other": "other services",
}

CLUSTER_PALETTE: list[str] = [
    "#4A90D9",
    "#E74C3C",
    "#2ECC71",
    "#F39C12",
    "#9B59B6",
    "#1ABC9C",
    "#E67E22",
    "#3498DB",
    "#E91E63",
    "#00BCD4",
    "#8BC34A",
    "#FF5722",
    "#795548",
    "#607D8B",
    "#FF9800",
    "#673AB7",
    "#009688",
    "#F44336",
    "#CDDC39",
    "#03A9F4",
]


def readable_feature_label(name: str) -> str:
    if name.startswith("act_") and name.endswith("_density"):
        base = (
            name.removeprefix("act_")
            .removesuffix("_density")
            .replace("_", " ")
            .lower()
        )
        return f"{ACTIVITY_LABEL_REPLACEMENTS.get(base, base)} density"
    return name.replace("_", " ").lower()


def color_for_cluster(c: int) -> str:
    return CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)]

