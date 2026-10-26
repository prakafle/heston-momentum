"""Evaluate the TimesFM forecasts (timesfm_forecast.py, zero-shot TFM* and fine-tuned FT*) against HAR and GARCH.

Test region and target are those of timesfm_forecast.py: next-day realised variance RV_t (Garman-Klass, rv_data.py),
every day in the second half of each index's sample. Benchmarks (all refit every Jan 1 on data before it,
common.folds, so no look-ahead):
    RW      : RV_{t-1}
    HAR     : OLS of RV_t on RV_{t-1}, 5-day and 22-day means (Corsi 2009)
    HARlog  : the same in logs, mapped back with exp() and no bias correction, like the paper's (·)^log models
    GARCH   : GARCH(1,1) on close-to-close returns (Gaussian MLE), rescaled by the training-window ratio
              mean(RV)/mean(σ²) because GK RV is open-to-close and excludes the overnight return
TimesFM forecasts, for each tag in (TFM = zero-shot, FT = fine-tuned) and context L in (64, 128, 512):
    {tag}{L}, {tag}{L}log : the model's point output = its MEDIAN (paper replication; log variants exp() back)
    {tag}{L}mean          : the mean head (QLIKE is minimised by the conditional mean, the median is biased low
                            for right-skewed RV)
    {tag}{L}logbc         : lognormal bias correction of the log model, exp(median + s²/2), s = (ln q90 - ln q10)/2.563
Post-processing of one base forecast per tag, {tag}512mean (post-hoc choice, made on theory after the median results
were seen: longest context, mean head), using only earlier test-region forecasts (refit every Jan 1 from the second
test year on, 21-day embargo; the first, partial test year is left unadjusted):
    {tag}cal   : base × mean(RV/base) over the past (the QLIKE-optimal constant rescaling)
    HAR+{tag}  : OLS of RV on the HAR regressors and the base forecast (HAR alone until 500 past rows exist)
    combo{tag} : (HAR + base) / 2, no fitting
Every forecast is floored at the smallest RV of the index's first half (a near-zero forecast makes QLIKE explode).
Losses: QLIKE (main, robust to the noisy RV proxy) and MSE. Diebold-Mariano: each model vs HAR, Newey-West lag 21
like the rest of the project (conservative for 1-day targets); pooled = loss differential averaged across indices
per date, then Newey-West. Negative t = better than HAR.
Usage: timesfm_eval.py [--quick]   (--quick reads the *_quick.csv forecast files)
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from common import H, ols, nw_t, folds

quick = "--quick" in sys.argv
sfx = "_quick" if quick else ""
rv = pd.read_csv("data/rv_daily.csv", index_col=0, parse_dates=True)
ret = pd.read_csv("data/rv_returns.csv", index_col=0, parse_dates=True)
files = [f"data/timesfm_forecasts{sfx}.csv", f"data/timesfm_ft_forecasts{sfx}.csv"]
tfm = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files if os.path.exists(f)], ignore_index=True)
TAGS = [t for t in ("TFM", "FT") if tfm.variant.str.startswith(t).any()]


def garch_filter(r, w, a, b):
    """σ²_t given returns up to t-1 (σ²_0 = sample variance)."""
    x = w + a * np.r_[r.var(), r[:-1] ** 2]
    return lfilter([1.0], [1.0, -b], x, zi=[b * r.var()])[0]


def garch_fit(r):
    def nll(p):
        w, a, b = np.exp(p[0]), *(1 / (1 + np.exp(-p[1:])))
        if a + b >= 0.9999:
            return 1e10
        s2 = garch_filter(r, w, a, b)
        return 0.5 * np.sum(np.log(s2) + r**2 / s2)
    p0 = [np.log(r.var() * 0.05), np.log(0.08 / 0.92), np.log(0.9 / 0.1)]
    p = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6}).x
    return np.exp(p[0]), *(1 / (1 + np.exp(-p[1:])))


def benchmarks(name):
    s = rv[name].dropna()
    r = ret[name].reindex(s.index).fillna(0.0)            # return of the same bar; 0 where the bar was dropped
    r = r - r.mean()
    f = pd.DataFrame({"y": s, "d": s.shift(1), "w": s.rolling(5).mean().shift(1), "m": s.rolling(22).mean().shift(1)})
    lf = np.log(f)
    out = pd.DataFrame(index=s.index, columns=["RW", "HAR", "HARlog", "GARCH"], dtype=float)
    out["RW"] = f.d
    for _, tr, te in folds(s.index):
        d, ld = f[tr].dropna(), lf[tr].dropna()
        if len(d) < 500:
            continue
        out.loc[te, "HAR"] = ols(d[["d", "w", "m"]].values, d.y.values)[1](f.loc[te, ["d", "w", "m"]].values)
        out.loc[te, "HARlog"] = np.exp(ols(ld[["d", "w", "m"]].values, ld.y.values)[1](lf.loc[te, ["d", "w", "m"]].values))
        w, a, b = garch_fit(r[tr].values)
        s2 = pd.Series(garch_filter(r.values, w, a, b), s.index)   # parameters frozen, filter runs through test year
        out.loc[te, "GARCH"] = s2[te] * s[tr].mean() / s2[tr].mean()
    return s, f[["d", "w", "m"]], out


def timesfm_models(name):
    d = tfm[tfm["index"] == name]
    piv = {c: d.pivot(index="date", columns="variant", values=c) for c in ("forecast", "mean", "q10", "q90")}
    out = piv["forecast"].copy()                           # medians: {tag}{L} and {tag}{L}log
    for v in piv["forecast"].columns:
        if v.endswith("log"):
            sd = (np.log(piv["q90"][v]) - np.log(piv["q10"][v])) / (2 * 1.2816)
            out[v + "bc"] = piv["forecast"][v] * np.exp(sd**2 / 2)
        else:
            out[v + "mean"] = piv["mean"][v]
    return out


def post_process(y, X, har, base, tag):
    """{tag}cal, HAR+{tag}, combo{tag}, each fitted on earlier test-region rows only."""
    out = pd.DataFrame({f"{tag}cal": base, f"HAR+{tag}": har, f"combo{tag}": (har + base) / 2})
    Z = X.assign(b=base, y=y)
    for _, tr, te in folds(base.index, min_years=1):
        past = Z[tr].dropna()
        if len(past) >= 250:
            out.loc[te, f"{tag}cal"] = base[te] * (past.y / past.b).mean()
        if len(past) >= 500:
            out.loc[te, f"HAR+{tag}"] = ols(past[["d", "w", "m", "b"]].values, past.y.values)[1](
                Z.loc[te, ["d", "w", "m", "b"]].values)
    return out


def qlike(y, f):
    return y / f - np.log(y / f) - 1


rows, diffs = [], {"QLIKE": {}, "MSE": {}}
for name in tfm["index"].unique():
    y, X, bench = benchmarks(name)
    fc = timesfm_models(name).join(bench, how="inner")
    for tag in TAGS:
        fc = fc.join(post_process(y.reindex(fc.index), X.reindex(fc.index), fc.HAR, fc[f"{tag}512mean"], tag))
    fc = fc.dropna().clip(lower=y.iloc[:len(y) // 2].min())
    yy = y.reindex(fc.index)
    for loss_name, L in (("QLIKE", qlike), ("MSE", lambda a, b: (a - b) ** 2)):
        loss = pd.DataFrame({m: L(yy, fc[m]) for m in fc.columns})
        for m in fc.columns:
            d = loss[m] - loss["HAR"]
            rows.append({"index": name, "loss": loss_name, "model": m, "n": len(d),
                         "ratio_vs_HAR": float(loss[m].mean() / loss["HAR"].mean()),
                         "dm_t": nw_t(d.values)[1] if m != "HAR" else np.nan})
            diffs[loss_name].setdefault(m, []).append(d.rename(name))
res = pd.DataFrame(rows)

pooled = []
for loss_name, by_model in diffs.items():
    for m, ds in by_model.items():
        if m == "HAR":
            continue
        avg = pd.concat(ds, axis=1).mean(axis=1)            # cross-index average per date, then Newey-West
        sub = res[(res.loss == loss_name) & (res.model == m)]
        pooled.append({"loss": loss_name, "model": m, "pooled_dm_t": nw_t(avg.values)[1],
                       "median_ratio": float(sub.ratio_vs_HAR.median()),
                       "better": int((sub.ratio_vs_HAR < 1).sum()), "sig_better": int((sub.dm_t < -1.96).sum()),
                       "sig_worse": int((sub.dm_t > 1.96).sum()), "n_indices": len(sub)})
pooled = pd.DataFrame(pooled).sort_values(["loss", "pooled_dm_t"])

json.dump({"quick": quick, "tags": TAGS, "per_index": res.to_dict("records"), "pooled": pooled.to_dict("records")},
          open(f"results_timesfm{sfx}.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
for loss_name in ("QLIKE", "MSE"):
    print(f"== {loss_name}: pooled DM t vs HAR (negative = better) and counts over indices, best first ==")
    print(pooled[pooled.loss == loss_name].drop(columns="loss").round(3).to_string(index=False), "\n")
for loss_name in ("QLIKE", "MSE"):
    print(f"== {loss_name}: loss ratio vs HAR (< 1 = better), per index ==")
    print(res[res.loss == loss_name].pivot(index="index", columns="model", values="ratio_vs_HAR").round(3).to_string(),
          "\n")
