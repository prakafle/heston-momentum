"""Which modified pricing equation best fits and predicts the SPY option smile?

Models (pricing.py): Heston, Bates (Heston + jumps), Double Heston (fast + slow variance factor).
Tests:
  1. in-sample fit to the latest chain (IV RMSE, vol points)
  2. leave-one-expiry-out: calibrate on 7 expiries, price the held-out one (does the model get the term structure?)
  3. next day: calibrate on day d, then predict day d+1's smile from the spot move alone, against the
     practitioner rules sticky-moneyness and sticky-strike. The model's prediction moves the variance state by
     its leverage effect: E[dv | return] = ρ ξ · return (per factor for double Heston).
Every run snapshots the live chain to data/options/SPY_<date>.csv, so test 3 gains a day each time it is run.
The 2026-10-23 snapshot is rebuilt from heston_calibration.json (IVs and moneyness; F from parity is not stored, so
strikes are approximated as moneyness × S·e^{(r-q)T} with q = 1.1% - only matters for the sticky-strike rule).

Everything is priced per unit forward (F = 1, strike = moneyness), which loses nothing: prices are homogeneous.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pricing import MODELS, call_prices, black, black_vega, implied

os.makedirs("data/options", exist_ok=True)


def snapshot_live():
    """Fetch SPY OTM options like calibrate_heston.py and save them. Returns the snapshot date."""
    import yfinance as yf
    spy = yf.Ticker("SPY")
    hist = spy.history(period="5d")
    S, last = float(hist["Close"].iloc[-1]), hist.index[-1].date()
    path = f"data/options/SPY_{last}.csv"
    if os.path.exists(path):
        return last
    r = float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]) / 100
    today = pd.Timestamp(last)
    exps = [e for e in spy.options if 10 <= (pd.Timestamp(e) - today).days <= 400]
    pick = sorted(set([exps[i] for i in np.linspace(0, len(exps) - 1, 8).round().astype(int)]))
    rows = []
    for e in pick:
        T = (pd.Timestamp(e) - today).days / 365
        ch = spy.option_chain(e)
        both = ch.calls.set_index("strike").join(ch.puts.set_index("strike"), lsuffix="_c", rsuffix="_p", how="inner")
        live = (both.bid_c > 0) & (both.bid_p > 0)
        fresh = (both.lastTradeDate_c.dt.date == last) & (both.lastTradeDate_p.dt.date == last)
        both = both[live | fresh].copy(); live = live[live | fresh]
        both["mc"] = np.where(live, (both.bid_c + both.ask_c) / 2, both.lastPrice_c)
        both["mp"] = np.where(live, (both.bid_p + both.ask_p) / 2, both.lastPrice_p)
        df = np.exp(-r * T)
        near = both.iloc[(both.index - S).to_series().abs().argsort()[:6]]
        F = float(np.median(near.index + (near.mc - near.mp) / df))
        for K, row in both.iterrows():
            call = K >= F
            mid = row.mc if call else row.mp
            spread = (row.ask_c - row.bid_c) if call else (row.ask_p - row.bid_p)
            if not 0.80 <= K / F <= 1.15 or mid < 0.05 or (live[K] and spread > 0.5 * mid):
                continue
            iv = implied(mid, F, K, T, df, call)
            if np.isfinite(iv):
                rows.append(dict(date=str(last), spot=S, expiry=e, T=T, F=F, K=K, m=K / F, df=df, iv=iv))
    pd.DataFrame(rows).to_csv(path, index=False)
    return last


def seed_from_calibration():
    path = "data/options/SPY_2026-10-23.csv"
    if os.path.exists(path) or not os.path.exists("heston_calibration.json"):
        return
    c = json.load(open("heston_calibration.json"))
    rows = []
    for s in c["smiles"]:
        F = c["spot"] * np.exp((c["r"] - 0.011) * s["T"])
        for m, iv in zip(s["moneyness"], s["market_iv"]):
            rows.append(dict(date=c["date"], spot=c["spot"], expiry=s["expiry"], T=s["T"], F=F, K=m * F, m=m,
                             df=np.exp(-c["r"] * s["T"]), iv=iv / 100))
    pd.DataFrame(rows).to_csv(path, index=False)


def thin(d, per_expiry=30):
    step = d.groupby("expiry").expiry.transform(lambda g: max(1, len(g) // per_expiry))
    return d[d.groupby("expiry").cumcount() % step == 0]


def model_iv(name, p, d):
    """Model implied vols for rows of d (columns expiry, T, m, df)."""
    out = np.empty(len(d))
    for (e, T, df), g in d.groupby(["expiry", "T", "df"]):
        c = call_prices(MODELS[name][0], p, 1.0, g.m.values, T, df)
        call = g.m.values >= 1
        prices = np.where(call, c, c - df * (1 - g.m.values))
        out[d.index.get_indexer(g.index)] = [implied(pr, 1.0, k, T, df, cl) for pr, k, cl in zip(prices, g.m, call)]
    return out


def calibrate(name, d):
    d = d.reset_index(drop=True)
    call = d.m.values >= 1
    mkt = black(1.0, d.m.values, d["T"].values, d.df.values, d.iv.values, call)
    vega = black_vega(1.0, d.m.values, d["T"].values, d.df.values, d.iv.values)

    def resid(p):
        out = np.empty(len(d))
        for (e, T, df), g in d.groupby(["expiry", "T", "df"]):
            c = call_prices(MODELS[name][0], p, 1.0, g.m.values, T, df)
            out[g.index] = np.where(g.m.values >= 1, c, c - df * (1 - g.m.values))
        return (out - mkt) / vega

    _, names, x0, lo, hi = MODELS[name]
    best = None
    for scale in (1.0, 0.6, 1.5):                       # a few starts: the multi-factor fits are not convex
        start = np.clip(np.array(x0) * np.where(np.array(lo) < 0, 1, scale), lo, hi)
        fit = least_squares(resid, start, bounds=(lo, hi), x_scale="jac", max_nfev=400)
        if best is None or fit.cost < best.cost:
            best = fit
    return best.x


def rmse(a, b):
    return float(100 * np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def step_state(name, p, ret, dt):
    """Move the variance state one day given the realised log return (leverage effect), params fixed."""
    p = np.array(p, float)
    facs = [0, 5] if name.startswith("Double") else [0]
    vtot = sum(p[i + 4] for i in facs)
    for i in facs:
        k, th, xi, rho, v = p[i:i + 5]
        p[i + 4] = max(v + k * (th - v) * dt + rho * xi * (v / vtot) * ret, 1e-5)
    return p


seed_from_calibration()
try:
    snapshot_live()
except Exception as e:                                   # offline: keep going with what is on disk
    print("live snapshot failed:", e)
snaps = {os.path.basename(p)[4:14]: thin(pd.read_csv(p)) for p in sorted(glob.glob("data/options/SPY_*.csv"))}
dates = sorted(snaps)
latest = snaps[dates[-1]].reset_index(drop=True)
print("snapshots:", dates)

res = {"date": dates[-1], "models": {}}
params = {}
for name in MODELS:
    p = calibrate(name, latest)
    params[name] = p
    ins = rmse(model_iv(name, p, latest), latest.iv)
    loo = []
    for e in sorted(latest.expiry.unique()):
        tr, te = latest[latest.expiry != e].reset_index(drop=True), latest[latest.expiry == e].reset_index(drop=True)
        loo.append({"expiry": e, "rmse": rmse(model_iv(name, calibrate(name, tr), te), te.iv)})
    res["models"][name] = {"params": dict(zip(MODELS[name][1], map(float, p))), "in_sample_rmse": ins,
                           "leave_one_expiry_out_rmse": float(np.mean([x["rmse"] for x in loo])), "loo": loo}
    print(f"{name:28s} in-sample {ins:.2f}  leave-one-expiry-out {res['models'][name]['leave_one_expiry_out_rmse']:.2f}")

# next-day prediction on every consecutive pair of snapshots
nd = []
for d0, d1 in zip(dates[:-1], dates[1:]):
    a, b = snaps[d0].reset_index(drop=True), snaps[d1].reset_index(drop=True)
    b = b[b.expiry.isin(a.expiry.unique())].reset_index(drop=True)
    ret = np.log(b.spot.iloc[0] / a.spot.iloc[0])
    dt = (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365
    row = {"from": d0, "to": d1, "spot_return_pct": float(100 * ret), "n": len(b)}
    pred_m, pred_k = [], []
    for e, g in b.groupby("expiry", sort=False):
        old = a[a.expiry == e].sort_values("m")
        pred_m += list(np.interp(g.m, old.m, old.iv))                        # same moneyness, same IV
        pred_k += list(np.interp(g.K, old.K, old.iv))                        # same strike, same IV
    row["sticky_moneyness"] = rmse(pred_m, b.iv)
    row["sticky_strike"] = rmse(pred_k, b.iv)
    for name in MODELS:
        p0 = calibrate(name, a)
        row[name] = rmse(model_iv(name, step_state(name, p0, ret, dt), b), b.iv)
    nd.append(row)
    print("next day", row)
res["next_day"] = nd
json.dump(res, open("results_smile.json", "w"), indent=1, default=float)
