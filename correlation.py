"""Correlation structure of S&P 500 stocks: static, random-matrix, time-varying, asymmetric, and option-implied.

1. Universe: CURRENT S&P 500 constituents (Wikipedia, with GICS sector), Yahoo adjusted closes from 2000.
   Survivorship bias: firms that left the index are missing, so history is tilted to survivors.
2. Static structure on the last 10 years: Pearson correlation of daily log returns, ordered by sector and by
   hierarchical clustering (distance sqrt(2(1 - ρ))); within- vs cross-sector averages; do clusters = sectors?
3. Random matrix theory: eigenvalues vs the Marchenko-Pastur noise band; market mode; what eigenvectors 2-5 separate;
   RMT-cleaned correlation vs sample vs Ledoit-Wolf in an out-of-sample minimum-variance portfolio.
4. Time variation: rolling average pairwise correlation; does it rise with vol and after selloffs (R1neg, the
   downside trend from the main project)? Next-month average correlation regressed on today's state (Newey-West),
   plus a walk-forward test of whether R1neg adds forecasting power.
5. Asymmetry and nonlinearity: exceedance correlations (Longin-Solnik) vs a Gaussian benchmark; Pearson vs
   Spearman vs Kendall; correlation of |returns| (volatility co-movement); empirical tail dependence.
6. Options link: Cboe implied correlation (COR1M/COR3M, top-50 S&P names) vs realised correlation of the same
   names -> correlation risk premium, and does implied correlation forecast realised correlation?
"""
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau
from sklearn.covariance import LedoitWolf
from sklearn.metrics import adjusted_rand_score

from common import H, ewm_mean, ols, nw_t, folds

os.makedirs("figures", exist_ok=True)
warnings.filterwarnings("ignore")
WINDOW_YEARS = 10
UA = {"User-Agent": "Mozilla/5.0 (research; heston-momentum)"}


# ---------------------------------------------------------------- 1. data
def constituents():
    path = "data/sp500_constituents.csv"
    if not os.path.exists(path):
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0][["Symbol", "Security", "GICS Sector"]]
        t.columns = ["ticker", "name", "sector"]
        t["ticker"] = t.ticker.str.replace(".", "-", regex=False)
        t.to_csv(path, index=False)
    return pd.read_csv(path)


def prices(tickers):
    path = "data/sp500_prices.csv"
    if not os.path.exists(path):
        import yfinance as yf
        p = yf.download(list(tickers), start="2000-01-01", progress=False, auto_adjust=True, threads=True)["Close"]
        p.index = p.index.tz_localize(None)
        p.to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def market_caps(tickers):
    path = "data/sp500_mcap.csv"
    if not os.path.exists(path):
        import yfinance as yf

        def cap(t):
            try:
                return t, float(yf.Ticker(t).fast_info["marketCap"])
            except Exception:
                return t, np.nan
        with ThreadPoolExecutor(16) as ex:
            pd.Series(dict(ex.map(cap, tickers)), name="mcap").to_csv(path)
    return pd.read_csv(path, index_col=0)["mcap"]


def cboe(name):
    r = requests.get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv", timeout=30)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    return pd.Series(d["CLOSE"].values, pd.to_datetime(d["DATE"]), name=name)


def hac_ols(X, y, lag=H):
    """OLS with Newey-West standard errors: returns coefs and t-stats (constant first)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    u = X1 * (y - X1 @ b)[:, None]
    S = u.T @ u
    for k in range(1, lag + 1):
        G = u[k:].T @ u[:-k]
        S += (1 - k / (lag + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    return b, b / se


def avg_corr(R, window, w=None):
    """Rolling average pairwise correlation from the portfolio-variance identity:
    ρ̄ = (σ_p² - Σ w_i² σ_i²) / ((Σ w_i σ_i)² - Σ w_i² σ_i²); equal weights over stocks present unless w given."""
    if w is None:
        w = R.notna().div(R.notna().sum(1), axis=0)
    else:
        w = R.notna().mul(w, axis=1)
        w = w.div(w.sum(1), axis=0)
    port = (R.fillna(0) * w).sum(1)
    s = R.rolling(window, min_periods=window).std()
    ws = (w.rolling(window).mean() * s)
    num = port.rolling(window).var() - (ws**2).sum(1)
    den = ws.sum(1) ** 2 - (ws**2).sum(1)
    return (num / den).where(s.notna().sum(1) > 20)


cons = constituents()
px = prices(cons.ticker)
px = px.loc[:, px.notna().sum() > 0]
sector = cons.set_index("ticker").sector.reindex(px.columns)
R_all = np.log(px).diff().iloc[1:]
R_all = R_all.where(np.abs(R_all) < 1.0)                                   # drop absurd prints (splits gone wrong)
uni = pd.read_csv("data/universe_prices.csv", index_col=0, parse_dates=True)
ivx = pd.read_csv("data/vol_indices.csv", index_col=0, parse_dates=True)
spx = np.log(uni["^GSPC"].dropna()).diff().dropna()
out, log = {"survivorship_note": "current constituents only; delisted/removed firms are missing"}, []


def show(title, obj):
    s = f"\n== {title} ==\n{obj if isinstance(obj, str) else obj.to_string()}"
    print(s); log.append(s)


# ---------------------------------------------------------------- 2. static structure
end = R_all.index[-1]
W = R_all[R_all.index > end - pd.DateOffset(years=WINDOW_YEARS)]
W = W.loc[:, W.notna().mean() > 0.99].dropna(how="all")
W = W.fillna(0.0)
N, T = W.shape[1], W.shape[0]
C = np.corrcoef(W.values.T)
off = C[~np.eye(N, dtype=bool)]
sec = np.asarray(sector[W.columns].fillna("Unknown"), dtype=str)
same = (sec[:, None] == sec[None, :]) & ~np.eye(N, dtype=bool)
sec_tab = pd.DataFrame({s: {"n_stocks": int((sec == s).sum()),
                            "within": float(C[np.ix_(sec == s, sec == s)][~np.eye((sec == s).sum(), dtype=bool)].mean()),
                            "to_other_sectors": float(C[np.ix_(sec == s, sec != s)].mean())}
                        for s in sorted(set(sec))}).T.sort_values("within", ascending=False)
D = np.sqrt(np.clip(2 * (1 - C), 0, None)); np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), "ward")                         # D is Euclidean for z-scored returns
k = len(set(sec))
ari = adjusted_rand_score(sec, fcluster(Z, k, "maxclust"))
out["static"] = {"window": [str(W.index[0].date()), str(W.index[-1].date())], "n_stocks": N, "n_days": T,
                 "avg_pairwise_corr": float(off.mean()), "median_pairwise_corr": float(np.median(off)),
                 "avg_within_sector": float(C[same].mean()), "avg_cross_sector": float(C[~same & ~np.eye(N, dtype=bool)].mean()),
                 "clusters_vs_sectors_adjusted_rand": float(ari), "sector_table": sec_tab.round(4).to_dict("index")}
show("2. static correlation", pd.Series({k2: v for k2, v in out["static"].items() if k2 != "sector_table"}))
show("2. within- vs cross-sector average correlation", sec_tab.round(3))

order = []
for s in sec_tab.index:                                                    # sector blocks, clustered inside
    idx = np.where(sec == s)[0]
    if len(idx) > 2:
        sub = D[np.ix_(idx, idx)]
        idx = idx[leaves_list(linkage(squareform(sub, checks=False), "ward"))]
    order += list(idx)
for name, o, ttl in [("corr_heatmap_sector.png", order, "ordered by GICS sector (clustered within)"),
                     ("corr_heatmap_cluster.png", leaves_list(Z), "ordered by hierarchical clustering")]:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C[np.ix_(o, o)], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    if name.endswith("sector.png"):
        b = np.cumsum([0] + [int((sec == s).sum()) for s in sec_tab.index])
        for x in b:
            ax.axhline(x - 0.5, color="k", lw=0.4); ax.axvline(x - 0.5, color="k", lw=0.4)
        ax.set_yticks((b[:-1] + b[1:]) / 2); ax.set_yticklabels(sec_tab.index, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_xticks([]); fig.colorbar(im, shrink=0.8)
    ax.set_title(f"S&P 500 daily log-return correlation, {N} stocks, {W.index[0].year}-{W.index[-1].year}\n{ttl}")
    fig.tight_layout(); fig.savefig(f"figures/{name}", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 3. random matrix theory
ev, V = np.linalg.eigh(C)
ev, V = ev[::-1], V[:, ::-1]
q = N / T
lam_plus, lam_minus = (1 + np.sqrt(q)) ** 2, (1 - np.sqrt(q)) ** 2
s2 = 1 - ev[0] / N                                                          # Laloux: noise variance net of market mode
lam_plus_adj = s2 * (1 + np.sqrt(q)) ** 2
loadings = {}
for i in range(1, 5):
    v = pd.Series(V[:, i], W.columns).groupby(sec).mean()
    v = v * np.sign(v[v.abs().idxmax()])                                   # biggest sector loading positive
    loadings[f"eigvec_{i + 1}"] = {"lambda": float(ev[i]), "positive": v.nlargest(3).round(3).to_dict(),
                                   "negative": v.nsmallest(3).round(3).to_dict()}
out["rmt"] = {"q_N_over_T": q, "mp_lambda_max": lam_plus, "mp_lambda_min": lam_minus,
              "mp_lambda_max_market_adjusted": lam_plus_adj, "lambda_1": float(ev[0]),
              "market_mode_share_pct": float(100 * ev[0] / N),
              "n_eigs_above_mp": int((ev > lam_plus).sum()), "n_eigs_above_mp_adjusted": int((ev > lam_plus_adj).sum()),
              "share_of_variance_in_signal_eigs_pct": float(100 * ev[ev > lam_plus_adj].sum() / N),
              "top_eigvec_sector_loadings": loadings}
show("3. random matrix theory", pd.Series({k2: v for k2, v in out["rmt"].items() if k2 != "top_eigvec_sector_loadings"}))
show("3. eigenvectors 2-5: sectors with largest +/- average loading",
     pd.DataFrame({k2: {"lambda": round(v["lambda"], 2), "+": ", ".join(v["positive"]), "-": ", ".join(v["negative"])}
                   for k2, v in loadings.items()}).T)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.linspace(s2 * lam_minus, lam_plus_adj, 400)
mp = np.sqrt(np.clip((lam_plus_adj - x) * (x - s2 * lam_minus), 0, None)) / (2 * np.pi * q * s2 * x)
axs[0].hist(ev[ev < 6], bins=80, density=True, alpha=0.6, label="sample eigenvalues (< 6)")
axs[0].plot(x, mp, "r", label=f"Marchenko-Pastur (q={q:.2f}, σ²={s2:.2f})")
axs[0].axvline(lam_plus_adj, color="r", ls="--", lw=0.8); axs[0].legend(); axs[0].set_xlabel("eigenvalue")
axs[0].set_title(f"{(ev > lam_plus_adj).sum()} of {N} eigenvalues above the noise band")
axs[1].bar(range(1, 21), ev[:20]); axs[1].set_yscale("log"); axs[1].axhline(lam_plus_adj, color="r", ls="--")
axs[1].set_title(f"top 20 eigenvalues; market mode λ1 = {ev[0]:.0f} ({100 * ev[0] / N:.0f}% of variance)")
fig.tight_layout(); fig.savefig("figures/rmt_eigenvalues.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
for i, ax in enumerate(axs, start=1):
    v = pd.Series(V[:, i], W.columns).groupby(sec).mean().sort_values()
    ax.barh(v.index, v.values, color=np.where(v.values > 0, "tab:blue", "tab:red"))
    ax.set_title(f"eigenvector {i + 1} (λ={ev[i]:.1f})"); ax.tick_params(labelsize=7)
fig.suptitle("average eigenvector loading by sector"); fig.tight_layout()
fig.savefig("figures/eigvec_sector_loadings.png", dpi=130); plt.close(fig)


def clean_rmt(Cs, n_obs):
    e, U = np.linalg.eigh(Cs)
    lp = (1 - e.max() / len(e)) * (1 + np.sqrt(len(e) / n_obs)) ** 2
    noise = e < lp
    e = e.copy(); e[noise] = e[noise].mean()                                   # trace-preserving eigenvalue clipping
    Cc = U @ np.diag(e) @ U.T
    d = np.sqrt(np.diag(Cc))
    return Cc / np.outer(d, d)


half = T // 2
tr, te = W.iloc[:half].values, W.iloc[half:].values
sd = tr.std(0)
covs = {"sample": np.cov(tr.T), "RMT-cleaned": clean_rmt(np.corrcoef(tr.T), half) * np.outer(sd, sd),
        "Ledoit-Wolf": LedoitWolf().fit(tr).covariance_}
mv = {}
for name, S in covs.items():
    w = np.linalg.solve(S, np.ones(N)); w /= w.sum()
    mv[name] = {"oos_vol_pct": float(100 * (te @ w).std() * np.sqrt(252)),
                "predicted_vol_pct": float(100 * np.sqrt(w @ S @ w * 252)), "gross_leverage": float(np.abs(w).sum())}
mv["equal weight"] = {"oos_vol_pct": float(100 * te.mean(1).std() * np.sqrt(252)), "predicted_vol_pct": np.nan,
                      "gross_leverage": 1.0}
out["rmt"]["min_variance_portfolio"] = mv
show("3. minimum-variance portfolio: fit on first half, realised vol on second half", pd.DataFrame(mv).T.round(2))

# ---------------------------------------------------------------- 4. time variation
rho63 = avg_corr(R_all, 63)
rho21 = avg_corr(R_all, 21)
rv = np.sqrt(ewm_mean(spx**2, 20) * 252)
R1neg = (-ewm_mean(spx, 10) * 252).clip(lower=0)
vix = ivx["VIX"].reindex(spx.index) / 100
f = pd.DataFrame({"y": rho21.shift(-21), "rho63": rho63, "rho21": rho21, "spx_vol": rv, "R1neg": R1neg, "vix": vix}).dropna()
f = f[f.index >= "2001-01-01"]
tv = {"mean_avg_corr_63d": float(rho63.mean()), "min": float(rho63.min()), "max": float(rho63.max()),
      "date_of_max": str(rho63.idxmax().date()),
      "corr(avg_corr_63d, spx_vol)": float(f.rho63.corr(f.spx_vol)), "corr(avg_corr_63d, vix)": float(f.rho63.corr(f.vix))}
reg = {}
for name, cols in {"persistence only": ["rho63"], "+ SPX vol": ["rho63", "spx_vol"],
                   "+ SPX vol + downside trend R1neg": ["rho63", "spx_vol", "R1neg"],
                   "+ VIX + R1neg": ["rho63", "vix", "R1neg"]}.items():
    Xs = (f[cols] - f[cols].mean()) / f[cols].std()                         # standardised coefficients
    b, t = hac_ols(Xs.values, f.y.values)
    reg[name] = {c: f"{bb:+.4f} (t={tt:+.2f})" for c, bb, tt in zip(cols, b[1:], t[1:])}
# walk-forward: does R1neg add to (rho, vol) in forecasting next-month average correlation?
fc = {m: pd.Series(np.nan, f.index) for m in ["random walk", "persistence", "+ vol", "+ vol + R1neg"]}
spec = {"persistence": ["rho21", "rho63"], "+ vol": ["rho21", "rho63", "spx_vol"],
        "+ vol + R1neg": ["rho21", "rho63", "spx_vol", "R1neg"]}
for _, trm, tem in folds(f.index):
    d = f[trm]
    for m, cols in spec.items():
        fc[m][tem] = ols(d[cols].values, d.y.values)[1](f.loc[tem, cols].values)
fc["random walk"] = f.rho21.where(fc["persistence"].notna())
ok = fc["persistence"].notna()
e = {m: (f.y[ok] - s[ok]) ** 2 for m, s in fc.items()}
wf = {m: {"oos_r2_vs_random_walk_pct": float(100 * (1 - e[m].sum() / e["random walk"].sum()))} for m in e}
wf["+ vol + R1neg"]["dm_t_vs_+vol"] = nw_t((e["+ vol + R1neg"] - e["+ vol"]).values)[1]
wf["+ vol"]["dm_t_vs_persistence"] = nw_t((e["+ vol"] - e["persistence"]).values)[1]
out["time_variation"] = {**tv, "next_month_regression_standardised": reg, "walk_forward": wf,
                         "test_start": str(f.index[ok][0].date())}
show("4. time-varying average correlation", pd.Series(tv))
show("4. next-21d avg correlation on today's state (standardised coefs, Newey-West t)", pd.DataFrame(reg).T.fillna(""))
show("4. walk-forward forecast of next-21d avg correlation", pd.DataFrame(wf).T.round(3))

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(rho63.index, rho63, lw=0.8, label="avg pairwise correlation (63d)")
ax2 = ax.twinx(); ax2.plot(vix.index, 100 * vix, color="tab:red", lw=0.5, alpha=0.6, label="VIX")
ax.set_ylabel("average correlation"); ax2.set_ylabel("VIX")
ax.set_title("S&P 500 average pairwise correlation vs VIX"); fig.legend(loc="upper left")
fig.tight_layout(); fig.savefig("figures/rolling_avg_corr.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 5. asymmetry and nonlinearity
m = spx.reindex(W.index).fillna(0).values
Zs = (W.values - W.values.mean(0)) / W.values.std(0)
zm = (m - m.mean()) / m.std()
qs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def exceed(x, y, qq):
    """Longin-Solnik exceedance correlation: both below their qq-quantile (qq<=.5) or both above (qq>.5)."""
    if qq <= 0.5:
        sel = (x < np.quantile(x, qq)) & (y < np.quantile(y, qq))
    else:
        sel = (x > np.quantile(x, qq)) & (y > np.quantile(y, qq))
    return np.corrcoef(x[sel], y[sel])[0, 1] if sel.sum() > 10 else np.nan


rho_mkt = np.array([np.corrcoef(Zs[:, i], zm)[0, 1] for i in range(N)])
rng = np.random.default_rng(0)
exc = {}
for qq in qs:
    data = np.nanmean([exceed(Zs[:, i], zm, qq) for i in range(N)])
    g = []
    for r_ in rng.choice(rho_mkt, 60, replace=False):                       # Gaussian with each stock's own ρ
        a = rng.standard_normal(T); b_ = r_ * a + np.sqrt(1 - r_**2) * rng.standard_normal(T)
        g.append(exceed(b_, a, qq))
    exc[qq] = {"data": float(data), "gaussian": float(np.nanmean(g))}
exc = pd.DataFrame(exc).T
sub = rng.choice(N, 60, replace=False)
pairs = [(i, j) for a_, i in enumerate(sub) for j in sub[a_ + 1:]]
Ws = W.values[:, sub]
nl = {"pearson": float(np.corrcoef(Ws.T)[np.triu_indices(60, 1)].mean()),
      "spearman": float(pd.DataFrame(Ws).corr("spearman").values[np.triu_indices(60, 1)].mean()),
      "kendall": float(np.mean([kendalltau(W.values[:, i], W.values[:, j])[0] for i, j in pairs])),
      "gaussian_implied_kendall_2/pi*asin(pearson)": float(np.mean(2 / np.pi * np.arcsin(np.corrcoef(Ws.T)[np.triu_indices(60, 1)]))),
      "avg_corr_of_returns_all": float(off.mean()),
      "avg_corr_of_abs_returns_all": float(np.corrcoef(np.abs(W.values).T)[~np.eye(N, dtype=bool)].mean())}
u = 0.05
lo = np.mean([np.mean(W.values[:, i][m < np.quantile(m, u)] < np.quantile(W.values[:, i], u)) for i in range(N)])
hi = np.mean([np.mean(W.values[:, i][m > np.quantile(m, 1 - u)] > np.quantile(W.values[:, i], 1 - u)) for i in range(N)])
gl = []
for r_ in rng.choice(rho_mkt, 60, replace=False):
    a = rng.standard_normal(200000); b_ = r_ * a + np.sqrt(1 - r_**2) * rng.standard_normal(200000)
    gl.append(np.mean(b_[a < np.quantile(a, u)] < np.quantile(b_, u)))
tail = {"lower_tail_dep_5pct": float(lo), "upper_tail_dep_5pct": float(hi), "gaussian_benchmark": float(np.mean(gl)),
        "independence": u}
out["asymmetry"] = {"exceedance_vs_spx": exc.round(4).to_dict("index"), "rank_and_abs": nl, "tail_dependence": tail}
show("5. exceedance correlation with S&P 500 (avg over stocks) vs Gaussian with same ρ", exc.round(3))
show("5. linear vs rank vs |return| correlation", pd.Series(nl).round(4))
show("5. tail dependence vs S&P 500 (P(stock in its 5% tail | index in its 5% tail))", pd.Series(tail).round(4))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(exc.index, exc.data, "o-", label="S&P 500 stocks vs index")
ax.plot(exc.index, exc.gaussian, "s--", label="bivariate Gaussian, same ρ")
ax.axvline(0.5, color="gray", lw=0.6); ax.set_xlabel("exceedance quantile (left: both down, right: both up)")
ax.set_ylabel("conditional correlation"); ax.set_title("Tail correlations far above Gaussian; strongest in joint selloffs"); ax.legend()
fig.tight_layout(); fig.savefig("figures/exceedance_corr.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------- 6. implied correlation (options)
try:
    imp = pd.concat([cboe("COR1M"), cboe("COR3M")], axis=1) / 100
    caps = market_caps(list(W.columns))
    top = caps.dropna().nlargest(50).index
    w50 = caps[top] / caps[top].sum()
    R50 = R_all[top]
    real21 = avg_corr(R50, 21, w50)                                         # cap-weighted, current weights (approx.)
    real63 = avg_corr(R50, 63, w50)
    g = pd.DataFrame({"cor1m": imp.COR1M, "cor3m": imp.COR3M, "real_fwd21": real21.shift(-21),
                      "real_fwd63": real63.shift(-63), "real_past21": real21}).dropna(subset=["cor1m", "real_past21"])
    crp = (g.cor1m - g.real_fwd21).dropna()
    crp3 = (g.cor3m - g.real_fwd63).dropna()
    gf = g.dropna(subset=["real_fwd21"])
    fc = {m_: pd.Series(np.nan, gf.index) for m_ in ["historical (past 21d)", "implied COR1M", "implied + historical"]}
    spec = {"historical (past 21d)": ["real_past21"], "implied COR1M": ["cor1m"],
            "implied + historical": ["cor1m", "real_past21"]}
    for _, trm, tem in folds(gf.index):
        d = gf[trm]
        for m_, cols in spec.items():
            fc[m_][tem] = ols(d[cols].values, d.real_fwd21.values)[1](gf.loc[tem, cols].values)
    ok = fc["implied COR1M"].notna()
    y = gf.real_fwd21[ok]
    err = {"random walk": (y - gf.real_past21[ok]) ** 2, "raw implied (COR1M as forecast)": (y - gf.cor1m[ok]) ** 2}
    err.update({m_: (y - s[ok]) ** 2 for m_, s in fc.items()})
    ic = {m_: {"oos_r2_vs_random_walk_pct": float(100 * (1 - e_.sum() / err["random walk"].sum())),
               "dm_t_vs_random_walk": nw_t((e_ - err["random walk"]).values)[1] if m_ != "random walk" else np.nan}
          for m_, e_ in err.items()}
    out["implied_correlation"] = {
        "available": True, "start": str(g.index[0].date()), "top50_names": list(top),
        "note": "realised = cap-weighted avg correlation of today's 50 largest names with static current weights",
        "mean_implied_1m": float(g.cor1m.mean()), "mean_realised_next_1m": float(g.real_fwd21.mean()),
        "corr_risk_premium_1m_mean": float(crp.mean()), "crp_1m_pct_positive": float(100 * (crp > 0).mean()),
        "crp_1m_nw_t": nw_t(crp.values)[1], "corr_risk_premium_3m_mean": float(crp3.mean()),
        "crp_3m_pct_positive": float(100 * (crp3 > 0).mean()), "forecast": ic, "test_start": str(y.index[0].date())}
    show("6. implied vs realised correlation (top-50 names)",
         pd.Series({k2: v for k2, v in out["implied_correlation"].items() if k2 not in ("forecast", "top50_names")}))
    show("6. forecasting next-month realised correlation (walk-forward)", pd.DataFrame(ic).T.round(3))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(g.index, g.cor1m, lw=0.7, label="implied (Cboe COR1M)")
    ax.plot(g.index, g.real_fwd21, lw=0.7, label="realised over next month (top-50, cap-weighted)")
    ax.set_title(f"Implied vs realised correlation: premium {100 * crp.mean():.1f} pts, positive {100 * (crp > 0).mean():.0f}% of days")
    ax.legend(); fig.tight_layout(); fig.savefig("figures/implied_vs_realized_corr.png", dpi=130); plt.close(fig)
except Exception as ex:
    out["implied_correlation"] = {"available": False, "error": str(ex)}
    show("6. implied correlation", f"unavailable: {ex}")

json.dump(out, open("results_correlation.json", "w"), indent=1, default=float)
open("logs_correlation.txt", "w").write("\n".join(log))
