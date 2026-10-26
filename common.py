"""Shared pieces for every walk-forward script: data loading, cleaning, folds, OLS, Newey-West, ML estimators.

Conventions (all scripts):
- daily log returns r; variances annualised (×252); vols as decimals (0.20 = 20%) unless a name says "vol_pts"
- H = 21 trading days = the 1-month horizon of every target
- folds(): expanding window, refit every Jan 1, training rows whose target window reaches the test year dropped
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
H = 21
MIN_TRAIN_YEARS = 4


def load():
    px = pd.read_csv("data/universe_prices.csv", index_col=0, parse_dates=True)
    ivx = pd.read_csv("data/vol_indices.csv", index_col=0, parse_dates=True)
    return px, ivx


def clean_iv(s):
    """Drop isolated bad prints and stale stretches (index stopped updating)."""
    s = s.dropna()
    s = s[np.abs(np.log(s / s.rolling(5, center=True, min_periods=3).median())) < 0.5]
    run = s.diff().eq(0).groupby(s.diff().ne(0).cumsum()).cumsum()
    return s[run < 5]


def ewm_mean(x, halflife):
    return x.ewm(halflife=halflife, adjust=False).mean()


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b, lambda Z: np.column_stack([np.ones(len(Z)), Z]) @ b


def nw_t(d, lag=H):
    """Mean of d and its Newey-West t-stat (overlapping 21-day targets => lag 21)."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]; n = len(d)
    e = d - d.mean()
    s = e @ e / n + 2 * sum((1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, min(lag, n - 1) + 1))
    return float(d.mean()), float(d.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def folds(index, min_years=MIN_TRAIN_YEARS):
    """Yield (year, train_mask, test_mask): expanding window, refit every Jan 1, embargo of H rows."""
    years = index.year
    for y in range(years.min() + min_years, years.max() + 1):
        if not (years == y).any():                # e.g. Cboe paused VXFXI/VXGDX/VXSLV 2022-2025
            continue
        start = np.searchsorted(index, pd.Timestamp(f"{y}-01-01"))
        train = np.zeros(len(index), bool); train[:max(start - H, 0)] = True
        yield y, train, years == y


# ---------------------------------------------------------------- machine learning
# Hyperparameters are fixed a priori (shallow trees, heavy leaf size, slow learning rate) and never tuned on test
# data. The elastic net picks its penalty by time-ordered CV inside each training window only.
def boosting():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=False, random_state=0)


def elastic_net():
    return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=TimeSeriesSplit(4),
                                                        max_iter=20000))


def permuted(X, cols, rng):
    """Copy of X with the columns in `cols` shuffled jointly across rows (group permutation importance)."""
    Xp = X.copy()
    Xp[cols] = X[cols].values[rng.permutation(len(X))]
    return Xp
