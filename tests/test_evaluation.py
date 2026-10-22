"""Checks for the shared metric and CV helpers (run: pytest -q)."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from src import config
from src.evaluation import cv_splitter, recall_at_budget, scorers, summarize_cv


def test_recall_at_budget_counts_top_share_only():
    # 10 orders, 2 late. Top 10% = 1 order, which is late -> recall 1/2.
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
    assert recall_at_budget(y, s, budget=0.10) == 0.5
    # Top 30% = 3 orders (0.9, 0.8, 0.2) -> both late caught.
    assert recall_at_budget(y, s, budget=0.30) == 1.0


def test_recall_at_budget_fails_loudly():
    with pytest.raises(ValueError):
        recall_at_budget([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        recall_at_budget([1, 0], [0.5])


def test_scorers_run_through_cross_validate():
    # Synthetic rare-class data, only to prove the scorers work end to end.
    rng = np.random.default_rng(config.RANDOM_STATE)
    X = rng.normal(size=(2000, 3))
    y = (X[:, 0] + rng.normal(scale=1.5, size=2000) > 2.2).astype(int)
    res = cross_validate(LogisticRegression(), X, y, cv=cv_splitter(), scoring=scorers())
    table = summarize_cv(res, "logit")
    assert {"pr_auc_mean", "pr_auc_std", "recall_top10_mean", "roc_auc_mean"} <= set(table.columns)
    assert 0 <= table.loc[0, "recall_top10_mean"] <= 1
    assert len(res["test_pr_auc"]) == config.N_CV_SPLITS
