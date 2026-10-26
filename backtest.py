"""Out-of-sample tests of momentum added to a Heston-style model.

A. Momentum in the drift: does an EWMA-of-returns signal predict next-month returns?
B. Momentum in the variance: does a trend term improve 1-month realized-vol forecasts?
C. Options: trade 1-month SPX variance (VIX^2 as the strike) using those forecasts.

Everything is fit on TRAIN (<= 2014) and scored on TEST (>= 2015). No parameter is tuned on TEST.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

H = 21                      # forecast horizon, trading days
SPLIT = "2015-01-01"
COST_VOL_PTS = 0.5          # bid/ask paid on each variance-swap entry, in vol points
px = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)


def ewm_mean(x, halflife):
    return x.ewm(halflife=halflife, adjust=False).mean()


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b, lambda Z: np.column_stack([np.ones(len(Z)), Z]) @ b


# ---------------------------------------------------------------- A. drift momentum
def drift_test(ticker, halflife_days):
    r = np.log(px[ticker]).diff().dropna()
    m = ewm_mean(r, halflife_days) * 252                  # annualised trend estimate m_t
    fwd = r.rolling(H).sum().shift(-H)                    # next-21d log return
    df = pd.DataFrame({"m": m, "y": fwd}).dropna().iloc[::H]   # non-overlapping months
    tr, te = df[df.index < SPLIT], df[df.index >= SPLIT]
    b, f = ols(tr[["m"]].values, tr.y.values)
    sse_model = ((te.y - f(te[["m"]].values)) ** 2).sum()
    sse_mean = ((te.y - tr.y.mean()) ** 2).sum()
    # trading rule: daily, vol-targeted sign(m) (classic time-series momentum), 2bp per unit turnover
    vol = np.sqrt(ewm_mean(r**2, 30) * 252)
    pos = (np.sign(m) * 0.10 / vol).clip(-2, 2).shift(1)
    strat = pos * r - 0.0002 * pos.diff().abs()
    bh = (0.10 / vol).clip(0, 2).shift(1) * r            # same vol targeting, always long
    t = r.index >= SPLIT
    sharpe = lambda x: x[t].mean() / x[t].std() * np.sqrt(252)
    return {
        "ticker": ticker, "halflife_days": halflife_days,
        "beta_train": float(b[1]), "oos_r2_pct": float(100 * (1 - sse_model / sse_mean)),
        "n_test_months": int(len(te)),
        "sharpe_tsmom": float(sharpe(strat)), "sharpe_long": float(sharpe(bh)),
    }


# ---------------------------------------------------------------- B. vol momentum
r = np.log(px["^GSPC"]).diff().dropna()
vix = px["^VIX"].reindex(r.index) / 100
rv_fwd = (r**2).rolling(H).sum().shift(-H) * 252 / H     # next-21d realised variance (annualised)

R2 = ewm_mean(r**2, 20) * 252                             # recent squared returns (variance level)
R2s = ewm_mean(r**2, 5) * 252                             # short-memory squared returns
R1 = ewm_mean(r, 10) * 252                                # trend; negative after a selloff

feat = pd.DataFrame({"y": rv_fwd, "sq": np.sqrt(R2), "sqs": np.sqrt(R2s), "R1": R1,
                     "R1neg": (-R1).clip(lower=0), "vix": vix}).dropna()
tr, te = feat[feat.index < SPLIT], feat[feat.index >= SPLIT]


def fit_vol_model(cols):
    """Regress next-month realised VOL on features; return variance forecasts on TEST."""
    b, f = ols(tr[cols].values, np.sqrt(tr.y.values))
    ptr, pte = np.maximum(f(tr[cols].values), 0.02), np.maximum(f(te[cols].values), 0.02)
    smear = tr.y.mean() / (ptr**2).mean()                 # fixes E[vol]^2 < E[var] bias, fit on TRAIN
    predict = lambda Z: np.maximum(f(Z), 0.02) ** 2 * smear
    return dict(zip(["const"] + cols, b.round(4).tolist())), pd.Series(pte**2 * smear, te.index), predict


def garch_forecast():
    x = r.values - r[r.index < SPLIT].mean()
    ntr = (r.index < SPLIT).sum()

    def filt(p, x):
        w, a, bb = p
        s2 = np.empty(len(x)); s2[0] = x[:250].var()
        for i in range(1, len(x)):
            s2[i] = w + a * x[i - 1] ** 2 + bb * s2[i - 1]
        return s2

    def nll(p):
        w, a, bb = p
        if w <= 0 or a < 0 or bb < 0 or a + bb >= 0.9999:
            return 1e10
        s2 = filt(p, x[:ntr])
        return 0.5 * np.sum(np.log(s2) + x[:ntr] ** 2 / s2)

    p = minimize(nll, [1e-6, 0.08, 0.9], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-10}).x
    w, a, bb = p
    s2 = filt(p, x)                                       # s2[i] = variance for day i given info to i-1
    lr = w / (1 - a - bb)
    nxt = w + a * x**2 + bb * s2                          # variance for day i+1 given info to i
    k = np.arange(H)
    avg = lr + (nxt - lr) * ((a + bb) ** k).mean()        # mean variance over the 21 days ahead
    return pd.Series(avg * 252, r.index), dict(omega=w, alpha=a, beta=bb)


def qlike(y, f):
    return float(np.mean(y / f - np.log(y / f) - 1))


def score(f, name, params=None):
    y = te.y
    f = f.reindex(y.index)
    return {"model": name, "qlike": qlike(y.values, f.values),
            "rmse_vol_pts": float(100 * np.sqrt(np.mean((np.sqrt(y) - np.sqrt(f)) ** 2))),
            "params": params}


heston_p, heston_f, heston_tr = fit_vol_model(["sq"])                         # mean-reverting variance
pdv_p, pdv_f, pdv_tr = fit_vol_model(["sq", "sqs", "R1"])                     # + trend (Guyon-Lekeufack)
pdvn_p, pdvn_f, _ = fit_vol_model(["sq", "sqs", "R1neg"])                     # + downside trend only
noR1_p, noR1_f, _ = fit_vol_model(["sq", "sqs"])                              # ablation: no trend
g_f, g_p = garch_forecast()
vol_scores = [
    score(heston_f, "Heston-style mean reversion (√R2)", heston_p),
    score(g_f, "GARCH(1,1)", {k: float(v) for k, v in g_p.items()}),
    score(noR1_f, "Short + long squared returns, no trend", noR1_p),
    score(pdv_f, "+ trend R1 (momentum in variance)", pdv_p),
    score(pdvn_f, "+ downside trend only", pdvn_p),
    score(te.vix**2, "VIX² (option market, unadjusted)"),
]

# ---------------------------------------------------------------- C. variance-swap strategy
def varswap_backtest(forecast, rule):
    """Sell 1-month variance at strike VIX^2 when the rule says so. P&L in vol points per unit vega."""
    out = []
    for off in range(H):                                  # 21 staggered monthly calendars, averaged
        d = te.iloc[off::H]
        K = (d.vix * 100 - COST_VOL_PTS).clip(lower=1) ** 2 / 1e4
        pnl_unit = (K - d.y) / (2 * np.sqrt(K)) * 100      # short var swap, vega-notional convention
        size = rule(d, forecast.reindex(d.index))
        out.append(pd.Series(size * pnl_unit.values, d.index))
    allp = pd.concat(out).sort_index()
    by_cal = [s for s in out]
    sharpe = np.mean([s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else 0 for s in by_cal])
    cum = [s.cumsum() for s in by_cal]
    mdd = np.mean([(c.cummax() - c).max() for c in cum])
    return {"mean_pnl_per_month": float(allp.mean()), "pct_months_traded": float(100 * (allp != 0).mean()),
            "sharpe": float(sharpe), "worst_month": float(allp.min()), "avg_max_drawdown": float(mdd),
            "total_pnl": float(np.mean([s.sum() for s in by_cal]))}


always = lambda d, f: np.ones(len(d))
edge = lambda d, f: ((d.vix**2 - f) > 0).astype(float).values
strategies = {
    "Always short 1m variance": varswap_backtest(heston_f, always),
    "Short only if VIX² > Heston-style forecast": varswap_backtest(heston_f, edge),
    "Short only if VIX² > GARCH forecast": varswap_backtest(g_f, edge),
    "Short only if VIX² > momentum-vol forecast": varswap_backtest(pdv_f, edge),
    # chosen AFTER seeing the vol-forecast table (best QLIKE), so this row is not a clean out-of-sample test
    "Short only if VIX² > downside-trend forecast (post-hoc)": varswap_backtest(pdvn_f, edge),
}

# monthly P&L path of one calendar (offset 0) for charting
d0 = te.iloc[::H]
K0 = (d0.vix * 100 - COST_VOL_PTS).clip(lower=1) ** 2 / 1e4
u0 = (K0 - d0.y) / (2 * np.sqrt(K0)) * 100
paths = {
    "dates": [x.strftime("%Y-%m") for x in d0.index],
    "always": u0.cumsum().round(2).tolist(),
    "heston": (u0 * edge(d0, heston_f.reindex(d0.index))).cumsum().round(2).tolist(),
    "momentum": (u0 * edge(d0, pdv_f.reindex(d0.index))).cumsum().round(2).tolist(),
}
# forecast vs realised series (weekly) for charting
wk = te.iloc[::5]
series = {"dates": [x.strftime("%Y-%m-%d") for x in wk.index],
          "realised": (100 * np.sqrt(wk.y)).round(2).tolist(),
          "vix": (100 * wk.vix).round(2).tolist(),
          "momentum": (100 * np.sqrt(pdv_f.reindex(wk.index))).round(2).tolist(),
          "heston": (100 * np.sqrt(heston_f.reindex(wk.index))).round(2).tolist()}

drift = [drift_test(t, hl) for t in ["SPY", "QQQ", "EFA", "TLT", "GLD", "USO"] for hl in (21, 63)]
now = pd.DataFrame({"sq": np.sqrt(R2), "sqs": np.sqrt(R2s), "R1": R1}).iloc[[-1]]
res = {
    "train_end": "2014-12-31", "test_start": SPLIT, "test_end": str(te.index[-1].date()),
    "drift": drift, "vol_scores": vol_scores, "varswap": strategies,
    "pnl_paths": paths, "vol_series": series,
    "vrp_mean_vol_pts": float(100 * (np.sqrt(te.vix**2) - np.sqrt(te.y)).mean()),
    "today": {"date": str(r.index[-1].date()), "vix": float(px["^VIX"].iloc[-1]),
              "momentum_forecast_vol": float(100 * np.sqrt(pdv_tr(now[["sq", "sqs", "R1"]].values)[0])),
              "heston_forecast_vol": float(100 * np.sqrt(heston_tr(now[["sq"]].values)[0])),
              "trend_R1": float(now.R1.iloc[0])},
}
json.dump(res, open("results.json", "w"), indent=1)

pd.set_option("display.width", 160)
print(pd.DataFrame(drift).round(3).to_string(index=False))
print()
print(pd.DataFrame(vol_scores).drop(columns="params").round(4).to_string(index=False))
print()
print(pd.DataFrame(strategies).T.round(3).to_string())
print("\nmean VIX - realised vol in TEST (vol pts):", round(res["vrp_mean_vol_pts"], 2))
print("today:", res["today"])
