"""Small column-name helpers shared between feature_engineering and the API.

Kept here (separate from feature_engineering.py) so importing them does
not pull in geopandas. The Railway backend imports these without needing
the GIS stack.
"""

from __future__ import annotations


def is_act_storefront_column(name: str) -> bool:
    s = str(name)
    return s.startswith("act_") and s.endswith("_storefront")


def is_act_density_column(name: str) -> bool:
    s = str(name)
    return s.startswith("act_") and s.endswith("_density")
