"""Shared metrics and cross-validation (DECISIONS.md D7, D8).

Everyone scores models through these functions, so benchmark, tuning and final
results are directly comparable. Do not re-implement metrics elsewhere.
"""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit

from src import config


def recall_at_budget(y_true, y_score, budget: float = config.FLAG_BUDGET) -> float:
    """Share of all late orders caught by flagging the top `budget` share of orders by score.

    This is the operating metric: ops can only review a fixed share of orders per day.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same length.")
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1].")
    n_late = y_true.sum()
    if n_late == 0:
        raise ValueError("No late orders in y_true; recall is undefined.")
    k = math.ceil(budget * len(y_true))
    # Stable sort so ties are broken by original (time) order, making results reproducible.
    top_k = np.argsort(-y_score, kind="stable")[:k]
    return float(y_true[top_k].sum() / n_late)


def scorers() -> dict:
    """Scoring dict for sklearn cross_validate / RandomizedSearchCV.

    Pass refit="pr_auc" when tuning: PR-AUC is the primary metric.
    """
    out = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
    for b in config.REPORT_BUDGETS:
        out[f"recall_top{int(b * 100)}"] = make_scorer(
            recall_at_budget, response_method="predict_proba", budget=b
        )
    return out


def cv_splitter() -> TimeSeriesSplit:
    """The one CV strategy for the project: expanding-window time-series folds.

    Data must be sorted by purchase time, which build_population() guarantees.
    """
    return TimeSeriesSplit(n_splits=config.N_CV_SPLITS)


def summarize_cv(cv_results: dict, model_name: str) -> pd.DataFrame:
    """One row per model: mean and std of each test score from cross_validate."""
    row = {"model": model_name}
    for key, values in cv_results.items():
        if key.startswith("test_"):
            name = key.removeprefix("test_")
            row[f"{name}_mean"] = np.mean(values)
            row[f"{name}_std"] = np.std(values)
    return pd.DataFrame([row])
