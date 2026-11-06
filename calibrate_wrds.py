"""Daily calibration of Heston, Bates and double Heston to the SPX smile, from OptionMetrics (one year per run).

    .venv/bin/python calibrate_wrds.py 2024        # reads data/wrds/opprcd_2024.csv.gz, fwdprd, zerocd, secprd

Rules, all fixed BEFORE the first run (none tuned on results):
- Options: SPX (secid 108105), PM-settled (am_settlement 0), standard (ss_flag 0). Each day, the listed expiry closest
  to each of 14, 30, 60, 91, 182, 273, 365 calendar days (distinct). Out-of-the-money side only (put if K < F, call
  if K >= F), moneyness K/F in [0.80, 1.15], best bid > 0, mid >= 0.05, spread <= 0.5 × mid (as smile_models.py).
  Thinned to ~30 strikes per expiry (smile_models.thin).
- Inputs: forward F = OptionMetrics fwdprd for that (date, expiry, settlement); discount df = exp(-r T) with r the
  zerocd rate linearly interpolated at the expiry's calendar days; T = calendar days / 365. Market IV = Black IV of
  the mid on the forward (SPX options are European), so model and market IVs use identical inputs.
- Fit: least squares on vega-weighted price errors (= IV errors to first order), same objective, models and bounds
  as smile_models.py (pricing.MODELS). Day 1: smile_models' three starts. Later days: warm start from the previous
  day's parameters AND the default start; keep the lower cost.
- Error: IV RMSE in vol points over all selected options of a day.
Tests:
  1. in-sample: daily full refit, RMSE per day.
  2. next day: day-d parameters, variance state moved by the realised SPX log return (smile_models.step_state:
     leverage E[dv | r] = ρ ξ r), priced on day d+1's options for the expiries both days share. Benchmarks:
     sticky-moneyness and sticky-strike (interpolate day d's smile at the same K/F or the same K).
  3. "state fast, parameters slow": all parameters calibrated on each month's first trading day; on the other days
     only the variance state (v0; v01 and v02 for double Heston) is refit. In-sample and next-day RMSE as above.
Significance: mean daily RMSE difference with a Newey-West t (lag 5, since next-day errors do not overlap).
Writes results_calibration_wrds_<year>.json (daily RMSEs and parameter paths: derived numbers, no quotes) and
prints the summary tables (save them as logs_calibration_wrds_<year>.txt).
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from common import nw_t
from pricing import MODELS, call_prices, black, black_vega, implied

SECID = 108105
TARGET_DAYS = [14, 30, 60, 91, 182, 273, 365]
PER_EXPIRY = 30
MAX_DAYS = int(os.environ.get("CALIB_MAX_DAYS", 0))      # smoke test only (CALIB_MAX_DAYS=5); 0 = whole year


def load_day_sets(year):
    o = pd.read_csv(f"data/wrds/opprcd_{year}.csv.gz", parse_dates=["date", "exdate"],
                    usecols=["secid", "date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer",
                             "am_settlement", "ss_flag"])
    o = o[(o.secid == SECID) & (o.am_settlement == 0) & (o.ss_flag == 0)].copy()
    o["days"] = (o.exdate - o.date).dt.days
    o = o[o.days.between(7, 365)]

    f = pd.read_csv("data/wrds/fwdprd.csv.gz", parse_dates=["date", "expiration"])
    f = f[(f.secid == SECID) & (f.AMSettlement == 0)][["date", "expiration", "ForwardPrice"]]
    o = o.merge(f, left_on=["date", "exdate"], right_on=["date", "expiration"], how="inner")

    z = pd.read_csv("data/wrds/zerocd.csv.gz", parse_dates=["date"])
    z = z[z.date.isin(o.date.unique())]
    zr = {d: (g.days.values, g.rate.values / 100) for d, g in z.groupby("date")}
    o["r"] = [np.interp(n, *zr[d]) for d, n in zip(o.date, o.days)]

    s = pd.read_csv("data/wrds/secprd.csv.gz", parse_dates=["date"], usecols=["secid", "date", "close"])
    spot = s[s.secid == SECID].set_index("date").close

    o["K"] = o.strike_price / 1000
    o["F"] = o.ForwardPrice
    o["m"] = o.K / o.F
    o["mid"] = (o.best_bid + o.best_offer) / 2
    o["call"] = o.cp_flag.eq("C")
    otm = np.where(o.call, o.K >= o.F, o.K < o.F)
    o = o[otm & o.m.between(0.80, 1.15) & (o.best_bid > 0) & (o.mid >= 0.05)
          & (o.best_offer - o.best_bid <= 0.5 * o.mid)]

    days = {}
    for d, g in o.groupby("date"):
        exp = g.groupby("exdate").days.first()
        pick = sorted({exp.index[np.abs(exp.values - t).argmin()] for t in TARGET_DAYS})
        g = g[g.exdate.isin(pick)].sort_values(["exdate", "K"]).copy()
        g["T"] = g.days / 365
        g["df"] = np.exp(-g.r * g["T"])
        g = g.rename(columns={"exdate": "expiry"})
        step = g.groupby("expiry").expiry.transform(lambda x: max(1, len(x) // PER_EXPIRY))
        g = g[g.groupby("expiry").cumcount() % step == 0].copy()
        g["iv"] = [implied(p / F, 1.0, m, T, df, c) for p, F, m, T, df, c in
                   zip(g.mid, g.F, g.m, g["T"], g.df, g.call)]           # per unit forward: price / F
        g = g[np.isfinite(g.iv)]
        days[d] = g[["expiry", "T", "df", "F", "K", "m", "iv"]].reset_index(drop=True)
        if MAX_DAYS and len(days) >= MAX_DAYS:
            break
    return days, spot


def resid_fn(name, d):
    call = d.m.values >= 1
    mkt = black(1.0, d.m.values, d["T"].values, d.df.values, d.iv.values, call)
    vega = black_vega(1.0, d.m.values, d["T"].values, d.df.values, d.iv.values)
    groups = [(g.index.values, T, df, g.m.values) for (e, T, df), g in d.groupby(["expiry", "T", "df"])]

    def resid(p):
        out = np.empty(len(d))
        for idx, T, df, m in groups:
            c = call_prices(MODELS[name][0], p, 1.0, m, T, df)
            out[idx] = np.where(m >= 1, c, c - df * (1 - m))
        return (out - mkt) / vega
    return resid


def fit(name, d, starts, free=None, base=None):
    """Least squares from each start; `free` = indices refit (others fixed at `base`)."""
    _, _, _, lo, hi = MODELS[name]
    lo, hi = np.array(lo, float), np.array(hi, float)
    r = resid_fn(name, d)
    if free is None:
        free, base = np.arange(len(lo)), np.zeros(len(lo))
    def full(q):
        p = np.array(base, float)
        p[free] = q
        return p
    best = None
    for s in starts:
        s = np.clip(np.asarray(s, float)[free], lo[free], hi[free])
        f = least_squares(lambda q: r(full(q)), s, bounds=(lo[free], hi[free]), x_scale="jac", max_nfev=400)
        if best is None or f.cost < best.cost:
            best = f
    return full(best.x)


def model_iv(name, p, d):
    out = np.empty(len(d))
    for (e, T, df), g in d.groupby(["expiry", "T", "df"]):
        c = call_prices(MODELS[name][0], p, 1.0, g.m.values, T, df)
        call = g.m.values >= 1
        prices = np.where(call, c, c - df * (1 - g.m.values))
        out[d.index.get_indexer(g.index)] = [implied(pr, 1.0, k, T, df, cl) for pr, k, cl in zip(prices, g.m, call)]
    return out


def rmse(a, b):
    return float(100 * np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def step_state(name, p, ret, dt):
    """smile_models.step_state: move v by mean reversion + leverage given the day's log return."""
    p = np.array(p, float)
    facs = [0, 5] if name.startswith("Double") else [0]
    vtot = sum(p[i + 4] for i in facs)
    for i in facs:
        k, th, xi, rho, v = p[i:i + 5]
        p[i + 4] = max(v + k * (th - v) * dt + rho * xi * (v / vtot) * ret, 1e-5)
    return p


def state_idx(name):
    return np.array([4, 9]) if name.startswith("Double") else np.array([4])


def run_model(args):
    name, year = args
    days, spot = load_day_sets(year)
    dates = sorted(days)
    x0 = np.array(MODELS[name][2], float)
    lo = np.array(MODELS[name][3])
    first = [x0 * np.where(lo < 0, 1, s) for s in (1.0, 0.6, 1.5)]
    daily, slow, prev, month_p = {}, {}, None, None
    for i, d in enumerate(dates):
        g = days[d]
        prev = fit(name, g, first if prev is None else [prev, x0])
        daily[d] = prev
        if month_p is None or d.month != dates[i - 1].month:
            month_p = prev.copy()                                           # month's first day: full calibration
            slow[d] = prev.copy()
        else:
            slow[d] = fit(name, g, [month_p], free=state_idx(name), base=month_p)
        print(f"{name[:6]} {d.date()} n={len(g)} rmse={rmse(model_iv(name, prev, g), g.iv):.2f}", flush=True)

    rows = []
    for i, d in enumerate(dates):
        g = days[d]
        row = {"date": str(d.date()), "n": len(g), "in_daily": rmse(model_iv(name, daily[d], g), g.iv),
               "in_slow": rmse(model_iv(name, slow[d], g), g.iv),
               "params": dict(zip(MODELS[name][1], map(float, daily[d])))}
        if i + 1 < len(dates):
            d1 = dates[i + 1]
            b = days[d1][days[d1].expiry.isin(g.expiry.unique())].reset_index(drop=True)
            ret, dt = np.log(spot[d1] / spot[d]), (d1 - d).days / 365
            row["next_daily"] = rmse(model_iv(name, step_state(name, daily[d], ret, dt), b), b.iv)
            row["next_slow"] = rmse(model_iv(name, step_state(name, slow[d], ret, dt), b), b.iv)
        rows.append(row)
    return name, rows


def sticky(days, dates):
    out = {}
    for d0, d1 in zip(dates[:-1], dates[1:]):
        a, b = days[d0], days[d1][days[d1].expiry.isin(days[d0].expiry.unique())]
        pm, pk = [], []
        for e, g in b.groupby("expiry", sort=False):
            old = a[a.expiry == e].sort_values("m")
            pm += list(np.interp(g.m, old.m, old.iv))
            pk += list(np.interp(g.K, old.K, old.iv))
        out[str(d0.date())] = {"sticky_moneyness": rmse(pm, b.iv), "sticky_strike": rmse(pk, b.iv)}
    return out


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    days, _ = load_day_sets(year)
    dates = sorted(days)
    print(f"{year}: {len(dates)} days, {np.mean([len(g) for g in days.values()]):.0f} options/day, "
          f"{np.mean([g.expiry.nunique() for g in days.values()]):.1f} expiries/day", flush=True)
    st = sticky(days, dates)
    with Pool(len(MODELS)) as pool:
        res = dict(pool.map(run_model, [(n, year) for n in MODELS]))

    tab = pd.DataFrame({"date": list(st), **{k: [v[k] for v in st.values()] for k in ["sticky_moneyness",
                                                                                         "sticky_strike"]}})
    for n, rows in res.items():
        r = pd.DataFrame(rows).set_index("date")
        for c in ["in_daily", "in_slow", "next_daily", "next_slow"]:
            tab[f"{n}|{c}"] = r[c].reindex(tab.date).values
    summary = []
    for c in tab.columns[1:]:
        x = tab[c].values
        row = {"series": c, "mean": float(np.nanmean(x)), "median": float(np.nanmedian(x)),
               "p90": float(np.nanpercentile(x, 90))}
        if "next" in c or "sticky" in c:
            for bm in ["sticky_moneyness", "sticky_strike"]:
                if c != bm:
                    row[f"vs_{bm}_t"] = nw_t(x - tab[bm].values, lag=5)[1]
        summary.append(row)
    summary = pd.DataFrame(summary).set_index("series")

    out = {"year": year, "n_days": len(dates), "rules": __doc__.split("Rules")[1].split("Writes")[0],
           "summary": summary.reset_index().to_dict("records"), "sticky": st,
           "models": {n: rows for n, rows in res.items()}}
    json.dump(out, open(f"results_calibration_wrds_{year}.json", "w"), indent=1, default=float)

    pd.set_option("display.width", 200)
    print(f"\nSPX {year}, IV RMSE in vol points per day (t: Newey-West lag 5 of the daily RMSE difference; "
          "negative = better than the sticky rule)\n")
    print(summary.round(3).to_string())
    print("\nParameter medians (daily refit):")
    for n, rows in res.items():
        p = pd.DataFrame([r["params"] for r in rows])
        print(f"{n}: " + ", ".join(f"{k} {v:.3g} [{p[k].quantile(.1):.3g}, {p[k].quantile(.9):.3g}]"
                                   for k, v in p.median().items()))
