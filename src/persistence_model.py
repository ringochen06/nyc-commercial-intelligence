"""
Supervised models for persistence score: Ridge Regression, Random Forest (sklearn OK).

Train/eval with borough-based splits to reduce geographic leakage where applicable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.serialization import ensure_directory, load_joblib, save_joblib

MODEL_PATH = ensure_directory(Path("outputs/models/")) / "persistence_model.joblib"

def train_ridge(X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> Any:
    """Fit Ridge regression; return fitted estimator."""
    raise NotImplementedError


def train_random_forest(X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> Any:
    """Fit RandomForestRegressor; return fitted estimator."""
    raise NotImplementedError

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate model performance on test set; return metrics + graph."""
    raise NotImplementedError

def predict_persistence(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return predicted persistence scores."""
    raise NotImplementedError

def save_model(model: Any, path: Path) -> None:
    """Persist with joblib or pickle under outputs/models/."""
    save_joblib(model, path)
    raise NotImplementedError


def load_model(path: Path) -> Any:
    """Load persisted model."""
    load_joblib(path)
    raise NotImplementedError


