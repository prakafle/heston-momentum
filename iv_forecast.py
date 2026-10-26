"""Predict option prices: forecast each Cboe 30-day implied-vol index h = 5 and 21 trading days ahead.

A Cboe vol index is a model-free price of a strip of 30-day options, so forecasting it is forecasting option prices.
Under Heston, 30-day implied variance x = IV² is affine in the instantaneous variance v_t, so

    Heston:            E_t[x_{t+h}] - x_t = a + (b - 1) x_t                (b = e^{-κh}, a = θ(1 - b))

Each modification of the model adds one state variable to the variance drift, i.e. one regressor:

    + realised-variance feedback   recent realised variance pulls implied variance (P-measure v feeds Q prices)
    + return trend (momentum)      Guyon-Lekeufack path dependence: variance drift depends on R1 = EWMA(returns)
    + IV momentum                  implied variance itself trends (5-day change of x)
    + second factor (double Heston) a slow variance factor: the long-run level x_slow = EWMA(x, 63d) moves
    + jumps (Bates proxy)          a recent large move (in daily sigmas) changes how fast x reverts
    + term structure (VIX only)    VIX9D/VIX3M/VIX6M: the market's own forecast of where variance is going

Then machine learning (gradient boosting, elastic net) on all of the above, with and without the trend group.
Benchmarks: random walk (x_{t+h} = x_t, the hard one to beat) and plain Heston.
Walk-forward exactly as panel_backtest.py (common.folds). Errors are in implied-vol points.
"""
import json
import numpy as np
import pandas as pd

from common import H, load, clean_iv, ewm_mean, ols, nw_t, folds, boosting, elastic_net
from download_data import VOL_PAIRS

HORIZONS = [5, 21]
px, ivx = load()

GROUPS = {"base": ["x"], "realised": ["rv", "jump"], "trend": ["R1", "R1neg", "R1slow"],
          "factor2": ["x_slow", "ivmom"]}
LINEAR = {
    "Heston (mean reversion)": ["x"],
    "+ realised-variance feedback": ["x", "rv"],
    "+ return trend (momentum)": ["x", "R1", "R1neg"],
    "+ IV momentum": ["x", "ivmom"],
    "+ second factor (double Heston)": ["x", "x_slow"],
    "+ jumps (Bates proxy)": ["x", "jump"],
    "All modifications": ["x", "rv", "jump", "R1", "R1neg", "R1slow", "x_slow", "ivmom"],
    "All modifications, no trend": ["x", "rv", "jump", "x_slow", "ivmom"],
}
TERM = ["ts_short", "ts_long"]                     # VIX only
ML = {
    "Boosting, all": (boosting, ["base", "realised", "trend", "factor2"]),
    "Boosting, no trend": (boosting, ["base", "realised", "factor2"]),
    "Elastic net, all": (elastic_net, ["base", "realised", "trend", "factor2"]),
}
TESTS = [  # (model, benchmark)
    ("Heston (mean reversion)", "Random walk"),
    ("+ realised-variance feedback", "Heston (mean reversion)"),
    ("+ return trend (momentum)", "Heston (mean reversion)"),
    ("+ IV momentum", "Heston (mean reversion)"),
    ("+ second factor (double Heston)", "Heston (mean reversion)"),
    ("+ jumps (Bates proxy)", "Heston (mean reversion)"),
    ("All modifications", "Heston (mean reversion)"),
    ("All modifications", "All modifications, no trend"),
    ("Boosting, all", "All modifications"),
    ("Boosting, all", "Boosting, no trend"),
    ("Elastic net, all", "All modifications"),
    ("Boosting, all", "Random walk"),
]


def features(iv_name, h):
    under = VOL_PAIRS[iv_name]
    r = np.log(px[under].dropna()).diff().dropna()
    iv = clean_iv(ivx[iv_name]).reindex(r.index) / 100
    x = iv**2
    sq = np.sqrt(ewm_mean(r**2, 20) * 252)
    f = pd.DataFrame({
        "iv": iv, "x": x, "y": x.shift(-h) - x,                           # target: change in implied variance
        "rv": sq**2, "R1": ewm_mean(r, 10) * 252, "R1slow": ewm_mean(r, 63) * 252,
        "jump": np.abs(r).rolling(5).max() / (sq / np.sqrt(252)),
        "x_slow": ewm_mean(x.ffill(limit=5), 63), "ivmom": x - x.shift(5),
    })
    f["R1neg"] = (-f.R1).clip(lower=0)
    if iv_name == "VIX":
        ts = ivx[["VIX9D", "VIX3M", "VIX6M"]].reindex(r.index) / 100
        f["ts_short"] = ts.VIX9D**2 - x                                    # front of the variance curve
        f["ts_long"] = ts.VIX6M**2 - x                                     # back of the variance curve
    return f.iloc[100:].dropna(subset=sum(GROUPS.values(), []))


def run(iv_name, h):
    f = features(iv_name, h)
    linear = dict(LINEAR)
    if iv_name == "VIX":
        f = f[f.index >= "2011-01-04"].dropna(subset=TERM)               # VIX9D starts 2011: same sample for all
        linear["+ term structure (VIX only)"] = ["x"] + TERM
        linear["All modifications + term structure"] = LINEAR["All modifications"] + TERM
    names = list(linear) + list(ML)
    fc = {m: pd.Series(np.nan, f.index) for m in names}
    for year, tr, te in folds(f.index):
        d = f[tr].dropna(subset=["y"])
        for m, cols in linear.items():
            fc[m][te] = ols(d[cols].values, d.y.values)[1](f.loc[te, cols].values)
        for m, (make, groups) in ML.items():
            cols = sum((GROUPS[g] for g in groups), [])
            fc[m][te] = make().fit(d[cols], d.y.values).predict(f.loc[te, cols])
    ok = fc[names[0]].notna() & f.y.notna()
    x, y = f.x[ok], f.y[ok]
    actual = 100 * np.sqrt(x + y)                                          # realised future IV, vol points
    pred = {"Random walk": 100 * np.sqrt(x)}
    pred.update({m: 100 * np.sqrt(np.maximum(x + s[ok], 1e-4)) for m, s in fc.items()})
    err = pd.DataFrame({m: (actual - p) ** 2 for m, p in pred.items()})
    move = np.sign(actual - pred["Random walk"])
    out = {"pair": iv_name, "h": h, "test_start": str(x.index[0].date()), "n": int(ok.sum()),
           "rmse": {m: float(np.sqrt(e.mean())) for m, e in err.items()},
           "oos_r2_vs_rw": {m: float(100 * (1 - e.sum() / err["Random walk"].sum())) for m, e in err.items()},
           "hit_rate": {m: float(100 * (np.sign(p - pred["Random walk"]) == move)[move != 0].mean())
                        for m, p in pred.items() if m != "Random walk"}}
    return out, err


rows, pooled = [], []
for h in HORIZONS:
    res = [run(n, h) for n in VOL_PAIRS]
    rows += [r for r, _ in res]
    for m, b in TESTS:
        diff = pd.concat({r["pair"]: e[m] - e[b] for r, e in res}, axis=1)
        mean, t = nw_t(diff.mean(1).values, lag=h)
        per = [nw_t(e[m] - e[b], lag=h) for _, e in res]
        pooled.append({"h": h, "model": m, "benchmark": b, "pooled_d_mse": mean, "pooled_dm_t": t,
                       "pairs_better": sum(p[0] < 0 for p in per), "pairs_sig_better": sum(p[1] < -1.96 for p in per),
                       "pairs_sig_worse": sum(p[1] > 1.96 for p in per), "n_pairs": len(per)})

summary = []
for r in rows:
    for m in r["rmse"]:
        summary.append({"h": r["h"], "pair": r["pair"], "model": m, "rmse": r["rmse"][m],
                        "oos_r2_vs_rw": r["oos_r2_vs_rw"][m], "hit_rate": r["hit_rate"].get(m, np.nan)})
summary = pd.DataFrame(summary)
avg = summary.groupby(["h", "model"])[["rmse", "oos_r2_vs_rw", "hit_rate"]].mean().round(3)
pooled = pd.DataFrame(pooled)
vix = summary[summary.pair == "VIX"].set_index(["h", "model"])[["rmse", "oos_r2_vs_rw", "hit_rate"]].round(3)
json.dump({"per_pair": rows, "average": avg.reset_index().to_dict("records"),
           "pooled_tests": pooled.to_dict("records"), "vix": vix.reset_index().to_dict("records")},
          open("results_iv_forecast.json", "w"), indent=1, default=float)

pd.set_option("display.width", 200)
print("== average across 18 pairs (RMSE in vol pts; OOS R² vs random walk, %; hit rate on direction, %) ==")
print(avg.to_string())
print("\n== pooled Diebold-Mariano on squared IV error (negative = model better) ==")
print(pooled.round(4).to_string(index=False))
print("\n== VIX (2011+, includes term-structure models) ==")
print(vix.to_string())
