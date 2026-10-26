"""Monte Carlo: would our tests find momentum if it were there, and do they read "already priced" correctly?

Simulated world (P measure, daily Euler, full truncation), Heston with a downside-trend term in the variance drift:

    r_t   = (μ - v/2) dt + √(v dt) ε1 + γ·m_{t-1} dt                     (γ: momentum in the DRIFT)
    m_t   = EWMA_10d(r) · 252,   n_t = max(-m_t, 0)                       (R1 and the downside trend R1neg)
    dv    = κ(θ + β n - v) dt + ξ √v dZ,   corr(ε1, dZ) = ρ                (β: momentum in the VARIANCE)

Two option markets are simulated on the same paths:
    priced   : IV² = E_t[average v over 21 days] including the trend term (option market knows β)
    unpriced : IV² = plain-Heston expectation, ignoring the trend (option market misses β)
Both carry a 1.15 variance risk premium and 5% lognormal quote noise, like real vol indices.

Each path then goes through the same walk-forward pipeline as panel_backtest.py part B (two-scale vs
two-scale + downside trend, with and without IV; QLIKE; Diebold-Mariano with Newey-West lag 21), and part A
(Clark-West test of the trend signal for next-month returns). Output: rejection rates (power) per β / γ.
β is also calibrated by matching the simulated regression coefficient on R1neg to the real S&P 500 / VIX one.

Usage:  monte_carlo.py                          all in one process (N_PATHS per point, reproduces the committed run)
        monte_carlo.py --task K --chunks C --paths N   one HPC array task: point K // C, chunk K % C of N paths
        monte_carlo.py --aggregate               combine mc_parts/*.csv into results_monte_carlo.json
Points are the BETAS (vol test) then the GAMMAS (drift test), so an array covers 0..(len(BETAS)+len(GAMMAS))·C-1.
Chunk c of a point uses seed (base + 10000·c), so chunk 0 is the single-process run.
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd

from common import H, ewm_mean, ols, nw_t, folds

YEARS, N_PATHS = 20, 200
KAPPA, THETA, XI, RHO, MU = 5.0, 0.03, 0.6, -0.7, 0.07
DT = 1 / 252
A10 = 1 - 2 ** (-1 / 10)                                   # EWMA weight for a 10-day half-life
LAM = np.log(2) / (10 / 252)                               # decay rate of the trend in expectation
TAU = H / 252
BETAS = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
GAMMAS = [0.0, 0.01, 0.02, 0.04]
REAL_R1NEG_COEF = 0.084                                    # VIX/SPX "Two-scale + downside trend", last refit


def simulate(beta, gamma, n_paths, years, seed):
    rng = np.random.default_rng(seed)
    T = 252 * years + 300                                  # 300-day burn-in, dropped
    v = np.full(n_paths, THETA); m = np.zeros(n_paths)
    R, V, M = (np.empty((T, n_paths)) for _ in range(3))
    for t in range(T):
        e1, e2 = rng.standard_normal((2, n_paths))
        r = (MU - v / 2) * DT + gamma * m * DT + np.sqrt(v * DT) * e1
        n = np.maximum(-m, 0)
        v = np.maximum(v + KAPPA * (THETA + beta * n - v) * DT
                       + XI * np.sqrt(v * DT) * (RHO * e1 + np.sqrt(1 - RHO**2) * e2), 1e-5)
        m = (1 - A10) * m + A10 * r * 252
        R[t], V[t], M[t] = r, v, m
    return R[300:], V[300:], M[300:]


def avg_factor(k):
    return (1 - np.exp(-k * TAU)) / (k * TAU)


def implied_var(v, m, beta, priced, rng):
    """21-day expected average variance: dE[v]/ds = κ(θ + β n e^{-λs} - E[v]), n frozen at today's value."""
    base = THETA + (v - THETA) * avg_factor(KAPPA)
    if priced:
        n = np.maximum(-m, 0)
        base = base + KAPPA * beta * n / (KAPPA - LAM) * (avg_factor(LAM) - avg_factor(KAPPA))
    return 1.15 * base * np.exp(0.05 * rng.standard_normal(v.shape) - 0.05**2 / 2)


def qlike(y, f):
    return y / f - np.log(y / f) - 1


def vol_tests(r, iv2):
    """Panel B on one path. Returns DM t-stats (negative = trend helps) and the R1neg coefficient."""
    idx = pd.bdate_range("2000-01-03", periods=len(r))
    r = pd.Series(r, idx)
    f = pd.DataFrame({"y": (r**2).rolling(H).sum().shift(-H) * 252 / H,
                      "sq": np.sqrt(ewm_mean(r**2, 20) * 252), "sqs": np.sqrt(ewm_mean(r**2, 5) * 252),
                      "R1neg": (-ewm_mean(r, 10) * 252).clip(lower=0), "iv": np.sqrt(iv2)}, index=idx).iloc[60:]
    models = {"base": ["sq", "sqs"], "trend": ["sq", "sqs", "R1neg"],
              "iv": ["iv", "sq", "sqs"], "iv_trend": ["iv", "sq", "sqs", "R1neg"]}
    fc = {m: pd.Series(np.nan, f.index) for m in models}
    coef = np.nan
    for _, tr, te in folds(f.index):
        d = f[tr].dropna(subset=["y"])
        for m, cols in models.items():
            b, g = ols(d[cols].values, np.sqrt(d.y.values))
            smear = d.y.mean() / (np.maximum(g(d[cols].values), 0.02) ** 2).mean()
            fc[m][te] = np.maximum(g(f.loc[te, cols].values), 0.02) ** 2 * smear
            if m == "trend":
                coef = b[-1]
    ok = fc["base"].notna() & f.y.notna() & (f.y > 0)
    loss = {m: qlike(f.y[ok], s[ok]).values for m, s in fc.items()}
    return {"t_no_iv": nw_t(loss["trend"] - loss["base"])[1], "t_with_iv": nw_t(loss["iv_trend"] - loss["iv"])[1],
            "r1neg_coef": coef}


def drift_test(r):
    """Panel A on one path: Clark-West t-stat of the 21-day-EWMA trend signal for next-month returns."""
    idx = pd.bdate_range("2000-01-03", periods=len(r))
    r = pd.Series(r, idx)
    f = pd.DataFrame({"y": r.rolling(H).sum().shift(-H), "s": ewm_mean(r, 21) * 252}).iloc[252:]
    pred, bench = pd.Series(np.nan, f.index), pd.Series(np.nan, f.index)
    for _, tr, te in folds(f.index, 5):
        d = f[tr].dropna()
        bench[te] = d.y.mean()
        pred[te] = ols(d[["s"]].values, d.y.values)[1](f.loc[te, ["s"]].values)
    ok = bench.notna() & f.y.notna()
    y, b, m = f.y[ok], bench[ok], pred[ok]
    return nw_t(((y - b) ** 2 - ((y - m) ** 2 - (b - m) ** 2)).values)[1]


def vol_chunk(i, c, n):
    beta = BETAS[i]
    R, V, M = simulate(beta, 0.0, n, YEARS, seed=100 + i + 10000 * c)
    rng = np.random.default_rng(7 + 10000 * c)
    rows = []
    for j in range(n):
        for priced in (True, False):
            iv2 = implied_var(V[:, j], M[:, j], beta, priced, rng)
            rows.append({"beta": beta, "world": "priced" if priced else "unpriced", **vol_tests(R[:, j], iv2)})
    return pd.DataFrame(rows)


def drift_chunk(k, c, n):
    R, _, _ = simulate(0.0, GAMMAS[k], n, 30, seed=500 + k + 10000 * c)
    return pd.DataFrame([{"gamma": GAMMAS[k], "cw_t": drift_test(R[:, j])} for j in range(n)])


ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int)
ap.add_argument("--chunks", type=int, default=1)
ap.add_argument("--paths", type=int, default=N_PATHS, help="paths per chunk")
ap.add_argument("--aggregate", action="store_true")
args = ap.parse_args()

if args.task is not None:
    point, c = divmod(args.task, args.chunks)
    os.makedirs("mc_parts", exist_ok=True)
    if point < len(BETAS):
        vol_chunk(point, c, args.paths).to_csv(f"mc_parts/vol_{point}_{c}.csv", index=False)
    else:
        drift_chunk(point - len(BETAS), c, args.paths).to_csv(f"mc_parts/drift_{point - len(BETAS)}_{c}.csv", index=False)
    raise SystemExit
if args.aggregate:
    vol = pd.concat(map(pd.read_csv, sorted(glob.glob("mc_parts/vol_*.csv"))), ignore_index=True)
    drift = pd.concat(map(pd.read_csv, sorted(glob.glob("mc_parts/drift_*.csv"))), ignore_index=True)
else:
    vol = pd.concat([vol_chunk(i, 0, N_PATHS) for i in range(len(BETAS))], ignore_index=True)
    drift = pd.concat([drift_chunk(k, 0, N_PATHS) for k in range(len(GAMMAS))], ignore_index=True)
n_vol, n_drift = vol.groupby(["beta", "world"]).size().min(), drift.groupby("gamma").size().min()
power_vol = vol.groupby(["beta", "world"]).agg(
    power_no_iv=("t_no_iv", lambda t: float((t < -1.96).mean())),
    power_with_iv=("t_with_iv", lambda t: float((t < -1.96).mean())),
    median_t_no_iv=("t_no_iv", "median"), median_t_with_iv=("t_with_iv", "median"),
    r1neg_coef=("r1neg_coef", "median"))
coef_by_beta = power_vol.xs("priced", level="world").r1neg_coef
beta_match = float(np.interp(REAL_R1NEG_COEF, coef_by_beta.values, coef_by_beta.index.values))

power_drift = drift.groupby("gamma").cw_t.agg(power=lambda t: float((t > 1.645).mean()), median_t="median")

json.dump({"settings": {"years_vol": YEARS, "years_drift": 30, "n_paths_vol": int(n_vol), "n_paths_drift": int(n_drift), "kappa": KAPPA, "theta": THETA,
                        "xi": XI, "rho": RHO, "mu": MU},
           "power_vol": power_vol.reset_index().to_dict("records"), "beta_matching_real_coef": beta_match,
           "power_drift": power_drift.reset_index().to_dict("records")},
          open("results_monte_carlo.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print("== variance momentum: rejection rate of 'trend does not help' at 5% (DM t < -1.96) ==")
print(power_vol.round(3).to_string())
print(f"\nβ whose simulated R1neg coefficient matches the real S&P 500 one ({REAL_R1NEG_COEF}): {beta_match:.4f}")
print("\n== drift momentum (30-year paths): Clark-West rejection rate at 5% one-sided ==")
print(power_drift.round(3).to_string())
