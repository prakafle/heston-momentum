"""Calibrate Heston to today's SPY option chain (the risk-neutral, Q-measure view).

Pricing: Heston characteristic function ("little trap" form) + Gil-Pelaez inversion, on forwards.
Forwards and discounting come from put-call parity at each expiry, so dividends need no assumption.
Fit: vega-weighted price errors (≈ implied-vol errors) over OTM options.
"""
import json
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import least_squares, brentq
from scipy.stats import norm

U, W = np.polynomial.legendre.leggauss(160)
U = 0.5 * (U + 1) * 250 + 1e-8; W = W * 0.5 * 250        # integrate u on (0, 250]


def heston_cf(u, T, F, kappa, theta, xi, rho, v0):
    """E[exp(i u ln F_T)] under Q, forward measure (driftless)."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0 + iu * np.log(F))


def heston_call(F, K, T, df, p):
    K = np.atleast_1d(K)[:, None]
    phi = heston_cf(U, T, F, *p)
    phi1 = heston_cf(U - 1j, T, F, *p) / F
    lk = np.exp(-1j * U * np.log(K))
    P1 = 0.5 + (W * (lk * phi1 / (1j * U)).real).sum(1) / np.pi
    P2 = 0.5 + (W * (lk * phi / (1j * U)).real).sum(1) / np.pi
    return df * (F * P1 - K[:, 0] * P2)


def black(F, K, T, df, vol, call):
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T)); d2 = d1 - vol * np.sqrt(T)
    c = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return np.where(call, c, c - df * (F - K))


def black_vega(F, K, T, df, vol):
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    return df * F * norm.pdf(d1) * np.sqrt(T)


def implied(price, F, K, T, df, call):
    try:
        return brentq(lambda s: black(F, K, T, df, s, call) - price, 1e-3, 3.0)
    except ValueError:
        return np.nan


# sanity check: tiny vol-of-vol with v0 = theta must reproduce Black
chk = heston_call(100.0, np.array([90.0, 100.0, 110.0]), 0.5, 0.98, (2.0, 0.04, 1e-3, 0.0, 0.04))
assert np.allclose(chk, black(100.0, np.array([90.0, 100.0, 110.0]), 0.5, 0.98, 0.2, True), atol=2e-3), chk

spy = yf.Ticker("SPY")
hist = spy.history(period="5d")
S = float(hist["Close"].iloc[-1])
last_session = hist.index[-1].date()
r = float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]) / 100
today = pd.Timestamp(last_session)
exps = [e for e in spy.options if 10 <= (pd.Timestamp(e) - today).days <= 400]
# a spread of maturities: a few short, a few long
pick = sorted(set([exps[i] for i in np.linspace(0, len(exps) - 1, 8).round().astype(int)]))

rows = []
for e in pick:
    T = (pd.Timestamp(e) - today).days / 365
    ch = spy.option_chain(e)
    c = ch.calls.set_index("strike"); p = ch.puts.set_index("strike")
    both = c.join(p, lsuffix="_c", rsuffix="_p", how="inner")
    # live mid when quotes exist; outside market hours Yahoo zeroes them, so use the last session's trades
    live = (both.bid_c > 0) & (both.bid_p > 0)
    fresh = (both.lastTradeDate_c.dt.date == last_session) & (both.lastTradeDate_p.dt.date == last_session)
    both = both[live | fresh].copy(); live = live[live | fresh]
    both["mc"] = np.where(live, (both.bid_c + both.ask_c) / 2, both.lastPrice_c)
    both["mp"] = np.where(live, (both.bid_p + both.ask_p) / 2, both.lastPrice_p)
    both["live"] = live
    df = np.exp(-r * T)
    near = both.iloc[(both.index - S).to_series().abs().argsort()[:6]]
    F = float(np.median(near.index + (near.mc - near.mp) / df))          # put-call parity forward
    for K, row in both.iterrows():
        if not 0.80 <= K / F <= 1.15:
            continue
        call = K >= F
        mid = row.mc if call else row.mp
        spread = (row.ask_c - row.bid_c) if call else (row.ask_p - row.bid_p)
        if mid < 0.05 or (row.live and spread > 0.5 * mid):
            continue
        iv = implied(mid, F, K, T, df, call)
        if np.isfinite(iv):
            rows.append(dict(expiry=e, tau=T, F=F, df=df, K=K, call=call, mid=mid, iv=iv))
opt = pd.DataFrame(rows)
assert len(opt), "no usable quotes or same-day trades"
opt = opt[opt.groupby("expiry").cumcount() % opt.groupby("expiry").expiry.transform(
    lambda g: max(1, len(g) // 30)) == 0].astype({c: float for c in ["tau", "F", "df", "K", "mid", "iv"]})  # ≤ ~30 per expiry
opt["vega"] = black_vega(opt.F, opt.K, opt.tau, opt.df, opt.iv)


def model_prices(p):
    out = np.empty(len(opt))
    for (e, T, F, df), g in opt.groupby(["expiry", "tau", "F", "df"]):
        c = heston_call(F, g.K.values, T, df, p)
        out[opt.index.get_indexer(g.index)] = np.where(g.call, c, c - df * (F - g.K.values))
    return out


opt = opt.reset_index(drop=True)
resid = lambda p: (model_prices(p) - opt.mid.values) / opt.vega.values
fit = least_squares(resid, x0=[2.0, 0.04, 0.6, -0.7, opt.iv.min() ** 2 * 1.5],
                    bounds=([0.05, 0.001, 0.05, -0.999, 0.0005], [20, 1.0, 4.0, 0.5, 1.0]))
p = fit.x
mp = model_prices(p)
opt["model_iv"] = [implied(m, F, K, T, df, c) for m, F, K, T, df, c in
                   zip(mp, opt.F, opt.K, opt.tau, opt.df, opt.call)]
rmse = float(100 * np.sqrt(np.nanmean((opt.model_iv - opt.iv) ** 2)))
names = ["kappa", "theta", "xi", "rho", "v0"]
out = {
    "date": str(today.date()), "spot": S, "r": r, "n_options": int(len(opt)),
    "params": dict(zip(names, map(float, p))), "iv_rmse_vol_pts": rmse,
    "feller": float(2 * p[0] * p[1] - p[2] ** 2),
    "smiles": [{"expiry": e, "T": float(g.tau.iloc[0]), "moneyness": (g.K / g.F).round(4).tolist(),
                "market_iv": (100 * g.iv).round(2).tolist(), "model_iv": (100 * g.model_iv).round(2).tolist()}
               for e, g in opt.groupby("expiry")],
}
json.dump(out, open("heston_calibration.json", "w"), indent=1)
print(f"SPY {S:.2f}  r={r:.4f}  options={len(opt)}  expiries={pick}")
print({k: round(v, 4) for k, v in out["params"].items()}, "IV RMSE (vol pts):", round(rmse, 2))
print("sqrt(v0) =", round(100 * np.sqrt(p[4]), 2), " sqrt(theta) =", round(100 * np.sqrt(p[1]), 2),
      " Feller 2κθ-ξ² =", round(out["feller"], 4))
print(opt.groupby("expiry").apply(lambda g: pd.Series({
    "n": len(g), "rmse": 100 * np.sqrt(np.nanmean((g.model_iv - g.iv) ** 2))})).round(2))
