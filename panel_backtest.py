"""Walk-forward, multi-asset version of backtest.py (equation-based models + machine learning).

A. Drift momentum on 49 assets: does a trend signal predict next-month returns? (OOS R², Clark-West, TSMOM Sharpe)
B. Vol momentum on 18 (asset, Cboe implied-vol index) pairs: does a trend term improve 1-month realised-vol
   forecasts, with and without the option market's implied vol as a regressor? (QLIKE, Diebold-Mariano)
   Machine learning (gradient boosting, elastic net) is run on the same folds, with and without the trend
   feature group, plus group-permutation importance.
C. Short 1-month variance at strike IV² on each pair, filtered by each forecast.

Every model is refit at the start of each calendar year on all data before it (expanding window). Training rows
whose 21-day target overlaps the test year are dropped (embargo), so no test-period return enters any fit.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from common import H, MIN_TRAIN_YEARS, load, clean_iv, ewm_mean, ols, nw_t, folds, boosting, elastic_net, permuted
from download_data import VOL_PAIRS, DRIFT_UNIVERSE

COST_VOL_PTS = {"index": 0.5, "etf": 1.0, "stock": 1.5}      # bid/ask paid per variance-swap entry
KIND = {**{k: "index" for k in ["VIX", "VXN", "RVX", "VXD"]},
        **{k: "stock" for k in ["VXAPL", "VXAZN", "VXGOG", "VXIBM", "VXGS"]}}
px, ivx = load()


# ---------------------------------------------------------------- A. drift momentum
def drift_features(p):
    r = np.log(p.dropna()).diff().dropna()
    f = pd.DataFrame({
        "r": r, "y": r.rolling(H).sum().shift(-H),
        "ewma21": ewm_mean(r, 21) * 252, "ewma63": ewm_mean(r, 63) * 252,
        "mom12_1": r.rolling(231).sum().shift(H),                    # classic 12-minus-1-month momentum
        "vol": np.sqrt(ewm_mean(r**2, 30) * 252),
    })
    return f.iloc[252:]


SIGNALS = ["ewma21", "ewma63", "mom12_1"]


def drift_test(ticker):
    f = drift_features(px[ticker]).dropna(subset=SIGNALS + ["vol"])
    pred = {s: pd.Series(np.nan, f.index) for s in SIGNALS}
    bench = pd.Series(np.nan, f.index)
    for _, tr, te in folds(f.index, 5):
        d = f[tr].dropna(subset=["y"])
        bench[te] = d.y.mean()
        for s in SIGNALS:
            pred[s][te] = ols(d[[s]].values, d.y.values)[1](f.loc[te, [s]].values)
    ok = bench.notna() & f.y.notna()
    y, b = f.y[ok], bench[ok]
    out = {"ticker": ticker, "test_start": str(f.index[ok][0].date()), "test_years": round(ok.sum() / 252, 1)}
    t = bench.notna()
    long = ((0.10 / f.vol).clip(0, 2).shift(1) * f.r)[t]
    out["sharpe_long"] = float(long.mean() / long.std() * np.sqrt(252))
    for s in SIGNALS:
        m = pred[s][ok]
        out[f"oos_r2_{s}"] = float(100 * (1 - ((y - m) ** 2).sum() / ((y - b) ** 2).sum()))
        cw = (y - b) ** 2 - ((y - m) ** 2 - (b - m) ** 2)                   # Clark-West adjusted MSPE gain
        out[f"cw_t_{s}"] = nw_t(cw.values)[1]
        pos = (np.sign(f[s]) * 0.10 / f.vol).clip(-2, 2).shift(1)
        strat = (pos * f.r - 0.0002 * pos.diff().abs())[t]
        out[f"sharpe_tsmom_{s}"] = float(strat.mean() / strat.std() * np.sqrt(252))
        out.setdefault("_daily", {})[s] = strat
    out["_daily"]["long"] = long
    out["_sq"] = {s: pd.DataFrame({"e_m": (y - pred[s][ok]) ** 2, "e_b": (y - b) ** 2}) for s in SIGNALS}
    return out


# ---------------------------------------------------------------- B. vol momentum
def vol_features(p, iv):
    r = np.log(p.dropna()).diff().dropna()
    f = pd.DataFrame({
        "r": r, "y": (r**2).rolling(H).sum().shift(-H) * 252 / H,            # next-21d realised variance
        "sq": np.sqrt(ewm_mean(r**2, 20) * 252), "sqs": np.sqrt(ewm_mean(r**2, 5) * 252),
        "R1": ewm_mean(r, 10) * 252,
        "rv_d": np.sqrt(r**2 * 252), "rv_w": np.sqrt((r**2).rolling(5).mean() * 252),
        "rv_m": np.sqrt((r**2).rolling(22).mean() * 252),
        "iv": iv.reindex(r.index) / 100,
        "R1slow": ewm_mean(r, 63) * 252,                                     # quarter-scale trend
    })
    f["R1neg"] = (-f.R1).clip(lower=0)
    f["jump"] = np.abs(r).rolling(5).max() / (f.sq / np.sqrt(252))         # largest recent move, in daily sigmas
    f["vrp"] = f.iv - f.sq                                                  # implied minus recent realised vol
    return f.iloc[60:]


VOL_MODELS = {                                    # name -> regressors for next-month realised VOL
    "Heston-style (√R2)": ["sq"],
    "HAR-RV": ["rv_d", "rv_w", "rv_m"],
    "Two-scale, no trend": ["sq", "sqs"],
    "Two-scale + trend R1": ["sq", "sqs", "R1"],
    "Two-scale + downside trend": ["sq", "sqs", "R1neg"],
    "IV regression": ["iv"],
    "IV + two-scale": ["iv", "sq", "sqs"],
    "IV + two-scale + trend R1": ["iv", "sq", "sqs", "R1"],
    "IV + two-scale + downside trend": ["iv", "sq", "sqs", "R1neg"],
}
# ML models see feature GROUPS; dropping the trend group is the ML version of the "does momentum add?" test
GROUPS = {"history": ["sq", "sqs", "rv_d", "rv_w", "rv_m", "jump"], "trend": ["R1", "R1neg", "R1slow"],
          "implied": ["iv", "vrp"]}
ML_MODELS = {                                     # name -> (estimator factory, feature groups)
    "Boosting, history": (boosting, ["history", "trend"]),
    "Boosting, history, no trend": (boosting, ["history"]),
    "Boosting + IV": (boosting, ["history", "trend", "implied"]),
    "Boosting + IV, no trend": (boosting, ["history", "implied"]),
    "Elastic net + IV": (elastic_net, ["history", "trend", "implied"]),
}
IMPORTANCE_MODELS = ["Boosting, history", "Boosting + IV"]
HYBRID_BASE = "IV + two-scale + downside trend"     # equation model whose residual the hybrid's trees learn
POOLED = {"Pooled boosting + IV": ["level", "history", "trend"], "Pooled boosting + IV, no trend": ["level", "history"]}
# scale-free versions of the features so one model can be trained on all 18 pairs at once
SF_GROUPS = {"level": ["liv"], "history": ["lsq", "lsqs", "lrv_d", "lrv_w", "lrv_m", "jump"],
             "trend": ["tR1", "tR1neg", "tR1slow"]}
# (model, benchmark) pairs that isolate the momentum question
VOL_TESTS = [
    ("Two-scale + trend R1", "Two-scale, no trend"),
    ("Two-scale + downside trend", "Two-scale, no trend"),
    ("IV + two-scale + trend R1", "IV + two-scale"),
    ("IV + two-scale + downside trend", "IV + two-scale"),
    ("Two-scale, no trend", "Heston-style (√R2)"),
    ("GARCH(1,1)", "Heston-style (√R2)"),
    ("IV + two-scale", "IV regression"),
    ("IV regression", "Two-scale + downside trend"),
    ("Boosting, history", "Two-scale + downside trend"),
    ("Boosting, history", "Boosting, history, no trend"),
    ("Boosting + IV", "IV + two-scale + downside trend"),
    ("Boosting + IV", "Boosting + IV, no trend"),
    ("Elastic net + IV", "IV + two-scale + downside trend"),
    ("Boosting + IV", "IV² (raw)"),
    ("Hybrid: equation + boosted residual", HYBRID_BASE),
    ("Pooled boosting + IV", HYBRID_BASE),
    ("Pooled boosting + IV", "Pooled boosting + IV, no trend"),
    ("Pooled boosting + IV", "IV² (raw)"),
]


def scale_free(f):
    """Everything relative to implied vol: trees then see a crisis in one asset as familiar data from another."""
    lg = lambda a: np.log(np.maximum(a, 1e-3) / f.iv)
    return pd.DataFrame({"liv": np.log(f.iv), "lsq": lg(f.sq), "lsqs": lg(f.sqs), "lrv_d": lg(f.rv_d),
                         "lrv_w": lg(f.rv_w), "lrv_m": lg(f.rv_m), "jump": f.jump, "tR1": f.R1 / f.iv,
                         "tR1neg": f.R1neg / f.iv, "tR1slow": f.R1slow / f.iv,
                         "z": np.log(f.y / f.iv**2)}, index=f.index)        # target: log realised/implied variance


def pooled_forecasts():
    """One boosting model per year, trained on every pair's history before that year (35-day embargo)."""
    sf = {n: scale_free(vol_features(px[u], clean_iv(ivx[n])).dropna(subset=sum(GROUPS.values(), [])))
          for n, u in VOL_PAIRS.items()}
    out = {n: {m: pd.Series(np.nan, s.index) for m in POOLED} for n, s in sf.items()}
    years = sorted(set().union(*[set(s.index.year) for s in sf.values()]))
    for y in years[MIN_TRAIN_YEARS:]:
        cut = pd.Timestamp(f"{y}-01-01")
        train = pd.concat([s[s.index < cut - pd.Timedelta(days=35)] for s in sf.values()])
        train = train[np.isfinite(train.z)]
        for m, groups in POOLED.items():
            cols = sum((SF_GROUPS[g] for g in groups), [])
            est = boosting().fit(train[cols], train.z)
            smear = np.mean(np.exp(train.z - est.predict(train[cols])))
            for n, s in sf.items():
                te = s.index.year == y
                if te.any():
                    out[n][m][te] = s.liv[te].pipe(np.exp) ** 2 * np.exp(est.predict(s.loc[te, cols])) * smear
    return out


def garch_fit(x):
    def filt(p, x):
        w, a, b = p
        s0 = x[:250].var()
        return np.concatenate([[s0], lfilter([1], [1, -b], w + a * x[:-1] ** 2, zi=[b * s0])[0]])

    def nll(p):
        w, a, b = p
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
            return 1e10
        s2 = filt(p, x)
        return 0.5 * np.sum(np.log(s2) + x**2 / s2)

    v = x.var()
    p = minimize(nll, [v * 0.02, 0.08, 0.9], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-12}).x
    return p, filt


def garch_forecast(r, mu, p, filt):
    w, a, b = p
    x = r - mu
    s2 = filt(p, x)
    lr = w / (1 - a - b)
    nxt = w + a * x**2 + b * s2
    return (lr + (nxt - lr) * ((a + b) ** np.arange(H)).mean()) * 252


def qlike(y, f):
    return y / f - np.log(y / f) - 1


def vol_test(iv_name):
    under = VOL_PAIRS[iv_name]
    f = vol_features(px[under], clean_iv(ivx[iv_name])).dropna(subset=sum(GROUPS.values(), []))
    fc = {m: pd.Series(np.nan, f.index) for m in list(VOL_MODELS) + ["GARCH(1,1)"] + list(ML_MODELS)
          + ["Hybrid: equation + boosted residual"]}
    perm = {(m, g): pd.Series(np.nan, f.index) for m in IMPORTANCE_MODELS for g in ML_MODELS[m][1]}
    rng = np.random.default_rng(0)
    coefs = []
    for year, tr, te in folds(f.index):
        d = f[tr].dropna(subset=["y"])
        for m, cols in VOL_MODELS.items():
            b, g = ols(d[cols].values, np.sqrt(d.y.values))
            smear = d.y.mean() / (np.maximum(g(d[cols].values), 0.02) ** 2).mean()
            fc[m][te] = np.maximum(g(f.loc[te, cols].values), 0.02) ** 2 * smear
            coefs.append({"year": year, "model": m, **dict(zip(["const"] + cols, b.tolist()))})
        # GARCH uses the asset's full return history up to the fold, not only the IV-available sample
        rr = np.log(px[under].dropna()).diff().dropna()
        cut = rr.index < pd.Timestamp(f"{year}-01-01")
        p, filt = garch_fit(rr[cut].values - rr[cut].mean())
        g = pd.Series(garch_forecast(rr.values, rr[cut].mean(), p, filt), rr.index)
        fc["GARCH(1,1)"][te] = g.reindex(f.index[te]).values
        for m, (make, groups) in ML_MODELS.items():
            cols = sum((GROUPS[g] for g in groups), [])
            est = make().fit(d[cols], np.sqrt(d.y.values))
            smear = d.y.mean() / (np.maximum(est.predict(d[cols]), 0.02) ** 2).mean()
            pred = lambda X: np.maximum(est.predict(X), 0.02) ** 2 * smear
            fc[m][te] = pred(f.loc[te, cols])
            if m == "Boosting + IV":               # hybrid: same features, trees fit the equation's error
                base = VOL_MODELS[HYBRID_BASE]
                b, g = ols(d[base].values, np.sqrt(d.y.values))
                res = make().fit(d[cols], np.sqrt(d.y.values) - g(d[base].values))
                fit = lambda Z, X: np.maximum(g(Z[base].values) + res.predict(X), 0.02)
                smear_h = d.y.mean() / (fit(d, d[cols]) ** 2).mean()
                fc["Hybrid: equation + boosted residual"][te] = fit(f.loc[te], f.loc[te, cols]) ** 2 * smear_h
            if m in IMPORTANCE_MODELS:
                for g in groups:
                    perm[(m, g)][te] = pred(permuted(f.loc[te, cols], GROUPS[g], rng))
    fc.update({m: s.reindex(f.index) for m, s in POOLED_FC[iv_name].items()})
    ok = fc["Heston-style (√R2)"].notna() & f.y.notna() & (f.y > 0)
    y = f.y[ok]
    loss = pd.DataFrame({m: qlike(y, s[ok]) for m, s in fc.items()})
    loss["IV² (raw)"] = qlike(y, f.iv[ok] ** 2)
    importance = {f"{m} | {g}": float((qlike(y, s[ok]) - loss[m]).mean()) for (m, g), s in perm.items()}
    rmse = {m: float(100 * np.sqrt(np.mean((np.sqrt(y) - np.sqrt(s[ok])) ** 2))) for m, s in fc.items()}
    return {"pair": iv_name, "underlying": under, "test_start": str(y.index[0].date()),
            "test_years": round(len(y) / 252, 1), "_loss": loss, "_fc": {m: s[ok] for m, s in fc.items()},
            "_f": f[ok], "rmse": rmse, "importance": importance, "_coefs": pd.DataFrame(coefs)}


# ---------------------------------------------------------------- C. variance-swap strategy
def varswap(res):
    """Daily entries (21 overlapping monthly calendars). P&L in vol points per unit vega notional."""
    f, cost = res["_f"], COST_VOL_PTS[KIND.get(res["pair"], "etf")]
    K = (f.iv * 100 - cost).clip(lower=1) ** 2 / 1e4
    unit = (K - f.y) / (2 * np.sqrt(K)) * 100
    pnl = {"Always short": unit}
    for m in ["Heston-style (√R2)", "GARCH(1,1)", "Two-scale, no trend", "Two-scale + trend R1",
              "Two-scale + downside trend", "HAR-RV", "Boosting, history", "Boosting + IV",
              "Hybrid: equation + boosted residual", "Pooled boosting + IV"]:
        pnl[f"Short if IV² > {m}"] = unit * (f.iv**2 > res["_fc"][m])
    return pd.DataFrame(pnl)


def sharpe_monthly(x):
    """Annualised Sharpe of a daily-entry, 21-day-hold P&L series, averaged over the 21 calendars."""
    s = [x.iloc[o::H] for o in range(H)]
    return float(np.mean([c.mean() / c.std() * np.sqrt(12) if c.std() > 0 else 0 for c in s]))


# ---------------------------------------------------------------- run
drift = [drift_test(t) for t in DRIFT_UNIVERSE if t in px]
POOLED_FC = pooled_forecasts()
vol = [vol_test(n) for n in VOL_PAIRS]

# A: per-asset table + pooled
dtab = pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("_")} for d in drift])
pooled_drift = {}
for s in SIGNALS:
    sq = pd.concat([d["_sq"][s] for d in drift])
    tsm = pd.concat({d["ticker"]: d["_daily"][s] for d in drift}, axis=1).mean(1)
    pooled_drift[s] = {
        "pooled_oos_r2": float(100 * (1 - sq.e_m.sum() / sq.e_b.sum())),
        "n_assets_r2_pos": int((dtab[f"oos_r2_{s}"] > 0).sum()),
        "n_assets_cw_sig_5pct": int((dtab[f"cw_t_{s}"] > 1.645).sum()),
        "median_sharpe_tsmom": float(dtab[f"sharpe_tsmom_{s}"].median()),
        "n_assets_tsmom_beats_long": int((dtab[f"sharpe_tsmom_{s}"] > dtab.sharpe_long).sum()),
        "diversified_tsmom_sharpe": float(tsm.mean() / tsm.std() * np.sqrt(252)),
    }
lng = pd.concat({d["ticker"]: d["_daily"]["long"] for d in drift}, axis=1).mean(1)
pooled_drift["diversified_long_sharpe"] = float(lng.mean() / lng.std() * np.sqrt(252))
pooled_drift["n_assets"] = len(drift)

# B: per-pair DM tests + pooled (cross-sectional mean loss differential per date, then Newey-West)
vtab, dm_rows = [], []
for v in vol:
    row = {"pair": v["pair"], "underlying": v["underlying"], "test_start": v["test_start"], "test_years": v["test_years"]}
    row.update({f"qlike:{m}": float(v["_loss"][m].mean()) for m in v["_loss"]})
    vtab.append(row)
    for m, b in VOL_TESTS:
        mean, t = nw_t((v["_loss"][m] - v["_loss"][b]).values)
        dm_rows.append({"pair": v["pair"], "model": m, "benchmark": b, "dqlike": mean, "dm_t": t})
vtab, dm = pd.DataFrame(vtab), pd.DataFrame(dm_rows)
pooled_vol = []
for m, b in VOL_TESTS:
    diff = pd.concat({v["pair"]: v["_loss"][m] - v["_loss"][b] for v in vol}, axis=1)
    mean, t = nw_t(diff.mean(1).values)
    by_year = diff.mean(1).groupby(diff.index.year).mean()
    sub = dm[(dm.model == m) & (dm.benchmark == b)]
    pooled_vol.append({"model": m, "benchmark": b, "pooled_dqlike": mean, "pooled_dm_t": t,
                       "pairs_better": int((sub.dqlike < 0).sum()), "pairs_sig_better": int((sub.dm_t < -1.96).sum()),
                       "pairs_sig_worse": int((sub.dm_t > 1.96).sum()), "n_pairs": len(sub),
                       "years_better": int((by_year < 0).sum()), "n_years": len(by_year)})
pooled_vol = pd.DataFrame(pooled_vol)
avg_qlike = vtab.filter(like="qlike:").mean().rename(lambda c: c[6:]).sort_values()

# ML group-permutation importance: QLIKE increase when a feature group is shuffled (bigger = more relied on)
importance = pd.DataFrame([{"pair": v["pair"], **v["importance"]} for v in vol])
importance_summary = importance.drop(columns="pair").agg(["mean", "median", lambda c: int((c > 0).sum())])
importance_summary.index = ["mean", "median", "n_pairs_positive"]

# what the trend coefficient looks like with and without IV (last refit, all pairs)
trend_coef = pd.DataFrame([{
    "pair": v["pair"],
    "R1neg_no_iv": float(v["_coefs"].query("model == 'Two-scale + downside trend'").R1neg.iloc[-1]),
    "R1neg_with_iv": float(v["_coefs"].query("model == 'IV + two-scale + downside trend'").R1neg.iloc[-1]),
    "R1_no_iv": float(v["_coefs"].query("model == 'Two-scale + trend R1'").R1.iloc[-1]),
    "R1_with_iv": float(v["_coefs"].query("model == 'IV + two-scale + trend R1'").R1.iloc[-1]),
} for v in vol])

# C: variance swaps
vs_rows, vs_pool = [], {}
for v in vol:
    p = varswap(v)
    for rule in p:
        mean, t = nw_t((p[rule] - p["Always short"]).values) if rule != "Always short" else (0.0, np.nan)
        vs_rows.append({"pair": v["pair"], "rule": rule, "sharpe": sharpe_monthly(p[rule]),
                        "mean_pnl": float(p[rule].mean()), "worst": float(p[rule].min()),
                        "pct_traded": float(100 * (p[rule] != 0).mean()), "vs_always_t": t})
    vs_pool[v["pair"]] = p
vs = pd.DataFrame(vs_rows)
port = pd.concat(vs_pool, axis=1)                         # equal vega-notional across pairs alive that day
vs_portfolio = {rule: {"sharpe": sharpe_monthly(port.xs(rule, axis=1, level=1).mean(1).dropna()),
                       "mean_pnl": float(port.xs(rule, axis=1, level=1).mean(1).mean())}
                for rule in vs_pool["VIX"].columns}
vs_summary = vs.groupby("rule").agg(median_sharpe=("sharpe", "median"), mean_pnl=("mean_pnl", "mean"),
                                    pairs_sig_better=("vs_always_t", lambda t: int((t > 1.96).sum())),
                                    pairs_sig_worse=("vs_always_t", lambda t: int((t < -1.96).sum())))

out = {
    "drift_assets": dtab.to_dict("records"), "drift_pooled": pooled_drift,
    "vol_pairs": vtab.to_dict("records"), "vol_rmse": {v["pair"]: v["rmse"] for v in vol},
    "vol_dm": dm.to_dict("records"), "vol_pooled": pooled_vol.to_dict("records"),
    "vol_avg_qlike": avg_qlike.to_dict(),
    "ml_importance": importance.to_dict("records"), "trend_coef_last_fit": trend_coef.to_dict("records"),
    "varswap_pairs": vs.to_dict("records"), "varswap_summary": vs_summary.reset_index().to_dict("records"),
    "varswap_portfolio": vs_portfolio,
}
json.dump(out, open("results_panel.json", "w"), indent=1, default=float)

pd.set_option("display.width", 200, "display.max_columns", 30)
print("== A. drift momentum, per asset ==")
print(dtab.round(2).to_string(index=False))
print(json.dumps(pooled_drift, indent=1, default=float))
print("\n== B. average test QLIKE across pairs ==")
print(avg_qlike.round(4).to_string())
print("\n== B. pooled Diebold-Mariano (negative dqlike = model better) ==")
print(pooled_vol.round(4).to_string(index=False))
print("\n== B. ML permutation importance (QLIKE increase when group shuffled) ==")
print(importance_summary.T.round(4).to_string())
print("\n== B. trend coefficients at last refit ==")
print(trend_coef.round(3).to_string(index=False))
print("\n== C. variance swaps ==")
print(vs_summary.round(3).to_string())
print(pd.DataFrame(vs_portfolio).T.round(3).to_string())
