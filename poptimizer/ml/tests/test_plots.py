import numpy as np
import pandas as pd
import pytest

from poptimizer import config
from poptimizer.ml import plots

ML_PARAMS = (
    (
        (True, {"days": 58}),
        (True, {"days": 195}),
        (False, {}),
        (True, {"days": 282}),
        (True, {"days": 332}),
    ),
    {
        "bagging_temperature": 0.9388504407881838,
        "depth": 5,
        "l2_leaf_reg": 3.2947929042414654,
        "learning_rate": 0.07663371920281654,
        "one_hot_max_size": 100,
        "random_strength": 0.9261064363697566,
        "ignored_features": [1],
    },
)
TICKERS = ("BANEP", "DSKY", "LKOH", "MOEX", "NKNCP")
DATE = pd.Timestamp("2019-01-03")


@pytest.fixture(autouse=True)
def patch_params_and_show(monkeypatch):
    monkeypatch.setattr(config, "ML_PARAMS", ML_PARAMS)
    monkeypatch.setattr(plots.plt, "show", lambda: None)


def test_learning_curve(monkeypatch):
    monkeypatch.setattr(plots, "FRACTIONS", [0.1, 0.5, 1.0])
    train_sizes, train_scores, test_scores = plots.learning_curve(TICKERS, DATE)
    assert np.allclose([11, 55, 110], train_sizes)
    assert np.allclose([0.71800467, 0.83144712, 0.8870297], train_scores)
    assert np.allclose([0.9919804, 0.9883989, 1.01204766], test_scores)


def test_partial_dependence_curve(monkeypatch):
    monkeypatch.setattr(plots, "QUANTILE", [0.3, 0.7])
    result = plots.partial_dependence_curve(TICKERS, DATE)
    assert len(result) == 3
    assert len(result[0]) == 2


def test_draw_cross_val_predict():
    x, y = plots.cross_val_predict_plot(TICKERS, DATE)
    assert len(x) == len(y) == 116
    assert np.allclose(x[:3].values, [-0.13702667, 0.16367253, 0.16657164])
    assert np.allclose(y[-3:].values, [0.17200295, -0.02747856, 0.02692788])
