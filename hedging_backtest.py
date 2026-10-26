"""Delta-hedged option selling on 18 (asset, Cboe implied-vol index) pairs: can volatility information make it pay?

Each trading day, sell one 1-month (21 trading days) at-the-money straddle on the underlying, priced with Black at
the vol index (minus the same bid/ask cost as panel_backtest C), and delta-hedge it daily in the underlying until
expiry. For a delta-hedged option the P&L is approximately

    Σ_k ½ Γ_k S_k² (realised variance_k - hedge variance) Δt        (+ the premium edge IV - hedge vol)

so the trade is a bet on realised vs implied vol, and vol forecasts enter in two places:
  1. hedge vol: which σ goes into the delta (Ahmad & Wilmott 2005: hedging at the vol that will be realised locks
     in the IV-RV spread; hedging at implied vol gives a smoother but path-dependent P&L)
  2. trade filter: sell only when IV² exceeds the forecast of realised variance; or go long when it does not

All choices below were fixed before the first run (none were tuned on results):
- hedge vols: none (naked), entry IV (constant), current IV (refreshed daily), EWMA (√EWMA r², halflife 20, no
  fit), HAR-RV and IV + two-scale forecasts (OLS on √RV, as in panel_backtest B), and the realised vol of the
  option's life (ORACLE: uses the future, infeasible; it shows the ceiling)
- filters (hedged at entry IV): always short, short if IV² > forecast (HAR-RV; IV + two-scale), long/short by sign
- costs: option sold at IV - 0.5 / 1.0 / 1.5 vol pts (index / ETF / stock, as panel C) and bought at IV + same;
  hedge trades pay 1 / 2 / 3 bp of notional; hedge closed at expiry
- forecasts refit every Jan 1 with the H-row embargo (common.folds); entries only in test years

P&L unit: vol points per unit of entry vega (a straddle sold 1 vol pt rich and hedged perfectly earns +1).
Simplifications: zero rates and dividends (forward = spot); the hedge earns no carry. A straddle starts nearly
delta-neutral, so the ignored carry is a small effect. Cboe indices for ETFs/stocks are not directly tradeable
(see README limitations). Profit from always-short is the variance risk premium: pay for crash risk, not alpha.

BIAS, the main caveat: a Cboe index prices a strip of OTM options (variance-swap style), so with a put skew it sits
ABOVE at-the-money IV (on the 2026-10-23 SPY snapshot: VIX 19.45 vs 30-day ATM IV 15.4). Pricing an ATM straddle at
the index therefore overstates the premium. Because P&L is in vol points per vega, every vol point of index-minus-ATM
gap lowers mean P&L by about one point, so the break-even gap equals the mean P&L. A real test needs ATM IV history.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import H, load, clean_iv, ewm_mean, ols, nw_t, folds
from download_data import VOL_PAIRS
from pricing import black, black_vega

COST_VOL_PTS = {"index": 0.5, "etf": 1.0, "stock": 1.5}
HEDGE_BP = {"index": 1.0, "etf": 2.0, "stock": 3.0}
KIND = {**{k: "index" for k in ["VIX", "VXN", "RVX", "VXD"]},
        **{k: "stock" for k in ["VXAPL", "VXAZN", "VXGOG", "VXIBM", "VXGS"]}}
FORECASTS = {"HAR-RV": ["rv_d", "rv_w", "rv_m"], "IV + two-scale": ["iv", "sq", "sqs"]}
HEDGES = ["No hedge", "Entry IV", "Current IV", "EWMA", "HAR-RV", "IV + two-scale", "ORACLE realised vol"]
T = H / 252
px, ivx = load()


def features(p, iv):
    r = np.log(p).diff().dropna()
    f = pd.DataFrame({
        "y": (r**2).rolling(H).sum().shift(-H) * 252 / H,                    # realised variance over the option's life
        "sq": np.sqrt(ewm_mean(r**2, 20) * 252), "sqs": np.sqrt(ewm_mean(r**2, 5) * 252),
        "rv_d": np.sqrt(r**2 * 252), "rv_w": np.sqrt((r**2).rolling(5).mean() * 252),
        "rv_m": np.sqrt((r**2).rolling(22).mean() * 252),
        "iv": iv.reindex(r.index) / 100,
    })
    return f.iloc[60:].dropna(subset=["sq", "rv_m", "iv"])


def forecasts(f):
    """Walk-forward forecasts of the option-life realised variance (same recipe as panel_backtest B)."""
    fc = {m: pd.Series(np.nan, f.index) for m in FORECASTS}
    for _, tr, te in folds(f.index):
        d = f[tr].dropna(subset=["y"])
        for m, cols in FORECASTS.items():
            _, g = ols(d[cols].values, np.sqrt(d.y.values))
            smear = d.y.mean() / (np.maximum(g(d[cols].values), 0.02) ** 2).mean()
            fc[m][te] = np.maximum(g(f.loc[te, cols].values), 0.02) ** 2 * smear
    return fc


def straddle_delta(S, K, tau, vol):
    d1 = (np.log(S / K) + 0.5 * vol**2 * tau) / (vol * np.sqrt(tau))
    return 2 * norm.cdf(d1) - 1


def run_pair(name):
    under, kind = VOL_PAIRS[name], KIND.get(name, "etf")
    p = px[under].dropna()
    iv_all = clean_iv(ivx[name])
    f = features(p, iv_all)
    fc = forecasts(f)
    ok = fc["HAR-RV"].notna() & f.y.notna()
    f, fc = f[ok], {m: s[ok] for m, s in fc.items()}
    pos = p.index.get_indexer(f.index)
    keep = pos + H < len(p)
    f, pos, fc = f[keep], pos[keep], {m: s[keep] for m, s in fc.items()}
    S = p.values[pos[:, None] + np.arange(H + 1)]                             # (entries, 22) price paths
    K = S[:, :1]
    tau = (H - np.arange(H)) / 252                                            # time left at each hedge date
    iv0 = f.iv.values
    cost, bp = COST_VOL_PTS[kind] / 100, HEDGE_BP[kind] / 1e4
    vega = 2 * black_vega(K[:, 0], K[:, 0], T, 1.0, iv0)                     # straddle vega per 1.00 of vol
    straddle = lambda vol: 2 * black(K[:, 0], K[:, 0], T, 1.0, vol, True)    # ATM: call + put = 2 × call
    payoff = np.abs(S[:, -1] - K[:, 0])
    iv_path = iv_all.reindex(p.index).ffill().values[pos[:, None] + np.arange(H)] / 100

    hedge_vol = {"Entry IV": iv0[:, None], "Current IV": iv_path, "EWMA": f.sq.values[:, None],
                 "HAR-RV": np.sqrt(fc["HAR-RV"].values)[:, None],
                 "IV + two-scale": np.sqrt(fc["IV + two-scale"].values)[:, None],
                 "ORACLE realised vol": np.sqrt(f.y.values)[:, None]}

    def hedge_pnl(vol):
        """Gains of the long-delta hedge of a short straddle, net of trading costs."""
        if vol is None:
            return np.zeros(len(S))
        D = straddle_delta(S[:, :-1], K, tau, np.maximum(vol, 0.01))
        trades = np.abs(np.diff(D, axis=1, prepend=0)) * S[:, :-1]
        return (D * np.diff(S, axis=1)).sum(1) - bp * (trades.sum(1) + np.abs(D[:, -1]) * S[:, -1])

    hp = {h: hedge_pnl(hedge_vol.get(h)) for h in HEDGES}
    short = {h: (straddle(iv0 - cost) - payoff + hp[h]) / vega * 100 for h in HEDGES}
    long_ = (payoff - straddle(iv0 + cost) - hp["Entry IV"]) / vega * 100
    gross = (straddle(iv0) - payoff + hp["Entry IV"]) / vega * 100           # no option bid/ask (hedge cost kept)

    hedges = pd.DataFrame({h: short[h] for h in HEDGES}, f.index)
    rules = {"Always short (gross of option bid/ask)": gross, "Always short": short["Entry IV"]}
    for m in FORECASTS:
        sell = f.iv.values**2 > fc[m].values
        rules[f"Short if IV² > {m}"] = np.where(sell, short["Entry IV"], 0.0)
        rules[f"Long/short by {m}"] = np.where(sell, short["Entry IV"], long_)
    return {"pair": name, "underlying": under, "hedges": hedges, "rules": pd.DataFrame(rules, f.index)}


def sharpe_monthly(x):
    """Annualised Sharpe of a daily-entry, 21-day-hold P&L series, averaged over the 21 calendars."""
    s = [x.iloc[o::H] for o in range(H)]
    return float(np.mean([c.mean() / c.std() * np.sqrt(12) if c.std() > 0 else 0 for c in s]))


def summary(df, base):
    rows = []
    for c in df:
        x = df[c].dropna()
        mean, t = nw_t(x.values)
        tail = x[x <= x.quantile(0.05)].mean()
        dt = nw_t((df[c] - df[base]).dropna().values)[1] if c != base else np.nan
        rows.append({"strategy": c, "mean_pnl": mean, "nw_t": t, "sharpe": sharpe_monthly(x), "std": x.std(),
                     "es5": tail, "worst": x.min(), "pct_traded": 100 * (x != 0).mean(), f"vs_{base}_t": dt})
    return pd.DataFrame(rows).set_index("strategy")


res = [run_pair(n) for n in VOL_PAIRS]
per_pair = []
for r in res:
    for kind, base in [("hedges", "Entry IV"), ("rules", "Always short")]:
        s = summary(r[kind], base).reset_index()
        s.insert(0, "pair", r["pair"]); s.insert(1, "table", kind)
        per_pair.append(s)
per_pair = pd.concat(per_pair, ignore_index=True)

# portfolio: equal vega notional in every pair alive that day (averaging across pairs per date, as in panel C)
port = {kind: pd.concat({r["pair"]: r[kind] for r in res}, axis=1) for kind in ["hedges", "rules"]}
port = {kind: pd.DataFrame({c: d.xs(c, axis=1, level=1).mean(1) for c in d.columns.levels[1]})
        for kind, d in port.items()}
port_hedges = summary(port["hedges"][HEDGES], "Entry IV")
port_rules = summary(port["rules"], "Always short")
counts = per_pair.groupby(["table", "strategy"]).agg(
    pairs_profitable=("mean_pnl", lambda m: int((m > 0).sum())),
    pairs_sig_profitable=("nw_t", lambda t: int((t > 1.96).sum())),
    pairs_sig_loss=("nw_t", lambda t: int((t < -1.96).sum())), n_pairs=("pair", "size"))
std_vs_iv = per_pair[per_pair.table == "hedges"].pivot(index="pair", columns="strategy", values="std")
std_vs_iv = (std_vs_iv.div(std_vs_iv["Entry IV"], axis=0))[HEDGES]
spx = next(r for r in res if r["pair"] == "VIX")
spx_years = pd.DataFrame({"Always short": spx["rules"]["Always short"],
                          "Long/short by IV + two-scale": spx["rules"]["Long/short by IV + two-scale"],
                          "No hedge": spx["hedges"]["No hedge"]}).groupby(lambda d: d.year).sum() / H

out = {"portfolio_hedges": port_hedges.reset_index().to_dict("records"),
       "portfolio_rules": port_rules.reset_index().to_dict("records"),
       "pair_counts": counts.reset_index().to_dict("records"),
       "std_ratio_vs_entry_iv": std_vs_iv.reset_index().to_dict("records"),
       "spx_by_year_vol_pts": spx_years.reset_index().to_dict("records"),
       "per_pair": per_pair.to_dict("records")}
json.dump(out, open("results_hedging.json", "w"), indent=1, default=float)

pd.set_option("display.width", 200, "display.max_columns", 30)
print("P&L in vol points per unit vega, per 1-month straddle; portfolio = equal vega across live pairs\n")
print("== 1. Hedge vol choice: short ATM straddle sold at IV - cost, portfolio of 18 pairs ==")
print(port_hedges.round(3).to_string())
print("\n== 1. P&L std relative to hedging at entry IV, per pair (< 1 = smoother) ==")
print(std_vs_iv.round(3).to_string())
print("\n== 2. Trade filters (hedged at entry IV), portfolio of 18 pairs ==")
print(port_rules.round(3).to_string())
print("\n== Counts over pairs ==")
print(counts.to_string())
print("\n== SPX (VIX) by year: P&L of selling one straddle per month (sum of ~12 trades), vol pts ==")
print(spx_years.round(2).to_string())
