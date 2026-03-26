"""
Build neighborhood feature vectors, text profiles, and persistence proxy labels.

Placeholder for checkpoint — density, diversity, demographics, persistence score.
"""

from __future__ import annotations

import pandas as pd


def compute_business_density(licenses: pd.DataFrame, area_col: str) -> pd.Series:
    """Businesses per unit area or per neighborhood."""
    raise NotImplementedError


def compute_category_diversity(licenses: pd.DataFrame, category_col: str) -> pd.Series:
    """Category entropy or similar diversity measure per neighborhood."""
    raise NotImplementedError


def build_neighborhood_text_profile(row: pd.Series) -> str:
    """Single string description of a neighborhood for embedding (stub)."""
    raise NotImplementedError


def compute_persistence_labels(
    licenses: pd.DataFrame,
    *,
    active_threshold_days: int,
) -> pd.Series:
    """Proportion (or rate) of businesses active beyond threshold — target for supervised models."""
    raise NotImplementedError


def build_feature_matrix(neighborhood_table: pd.DataFrame) -> pd.DataFrame:
    """Return numeric features used for clustering and persistence prediction."""
    raise NotImplementedError
