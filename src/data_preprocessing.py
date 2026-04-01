"""
Ingest, clean, and merge NYC Open Data licensing with ACS (or other) neighborhood keys.

Placeholder for checkpoint — implement loaders, schema alignment, and merged tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_raw_data(raw_dir: Path | str) -> dict[str, Any]:
    """Load raw CSV/API extracts from ``data/raw``. Not implemented yet."""
    raise NotImplementedError("load_raw_data: wire NYC Open Data + ACS paths and parsers")


def clean_licenses(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize license records (dates, status, geography)."""
    raise NotImplementedError


def clean_acs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ACS tract/ZIP or neighborhood aggregates."""
    raise NotImplementedError

def categorize_businesses(df: pd.DataFrame, license_df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical labels to business records based on license industry."""
    raise NotImplementedError

def merge_to_neighborhood_table(
    licenses: pd.DataFrame,
    acs: pd.DataFrame,
    geo_key: str,
) -> pd.DataFrame:
    """Join cleaned tables on a shared neighborhood / tract identifier."""
    raise NotImplementedError
