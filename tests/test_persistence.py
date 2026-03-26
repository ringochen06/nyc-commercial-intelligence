"""Tests for persistence label logic in ``src.feature_engineering``."""

import pandas as pd
import pytest

from src import feature_engineering


@pytest.mark.skip(reason="Implement compute_persistence_labels first")
def test_persistence_label_range():
    """Persistence proxy should be in a sensible range (e.g. [0, 1]) for grouped data."""
    df = pd.DataFrame()  # replace with minimal synthetic license rows
    s = feature_engineering.compute_persistence_labels(df, active_threshold_days=365)
    assert s.min() >= 0
    assert s.max() <= 1
