"""Display and statistical utilities shared across cluster and rank helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _pretty_feature(name: str) -> str:
    return name.replace("_", " ")


def _activity_label_from_col(col: str) -> str:
    label = (
        str(col)
        .removeprefix("act_")
        .removesuffix("_storefront")
        .removesuffix("_density")
        .replace("_", " ")
        .lower()
    )
    replacements = {
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
    return replacements.get(label, label)


def _display_borough(name: str) -> str:
    return str(name).strip().title()


def _fmt_list(items: list[str], *, limit: int = 3) -> str:
    vals = [str(x) for x in items if str(x).strip()][:limit]
    if not vals:
        return ""
    if len(vals) == 1:
        return vals[0]
    return ", ".join(vals[:-1]) + f", and {vals[-1]}"


def _percentile_rank(series: pd.Series, value: float) -> int | None:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0 or not np.isfinite(value):
        return None
    return int(round(100.0 * float(np.mean(vals <= value))))


def _level_from_percentile(pct: int | None) -> str:
    if pct is None:
        return "typical"
    if pct >= 95:
        return "extreme"
    if pct >= 80:
        return "very high"
    if pct >= 65:
        return "high"
    if pct <= 20:
        return "low"
    return "moderate"


def _series_max(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.notna().sum() == 0:
        return 0.0
    return float(vals.max())


def _series_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
