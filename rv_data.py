"""Daily realised-variance proxy for the 21 indices of Goel, Pasricha, Magris & Kanniainen (2025), arXiv:2505.11163.

The paper uses 5-minute realised variance from the Oxford-Man Realized Library, which was discontinued in 2022.
We substitute the Garman-Klass range estimator from daily OHLC (Yahoo), the most efficient OHLC estimator of
the open-to-close variance:  RV_t = 0.5 ln(H/L)² - (2 ln 2 - 1) ln(C/O)².
It is noisier than 5-minute RV, so absolute loss levels are not comparable to the paper, but model rankings are.
Yahoo has no data for KSE (Karachi), so 20 of the paper's 21 indices are used. Days with a stale or broken bar (H = L, or open = close = high = low) are dropped.
Writes data/rv_daily.csv (columns = paper symbols, values = daily variance, not annualised).
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
INDICES = {
    "AEX": "^AEX", "AORD": "^AORD", "BFX": "^BFX", "BVSP": "^BVSP", "DJI": "^DJI", "FCHI": "^FCHI",
    "FTSE": "^FTSE", "GDAXI": "^GDAXI", "HSI": "^HSI", "IBEX": "^IBEX", "IXIC": "^IXIC", "KS11": "^KS11",
    "KSE": "^KSE", "MXX": "^MXX", "N225": "^N225", "RUT": "^RUT", "SPX": "^GSPC", "SSEC": "000001.SS",
    "SSMI": "^SSMI", "STI": "^STI", "STOXX50E": "^STOXX50E",
}

if __name__ == "__main__":
    raw = yf.download(list(INDICES.values()), start="2000-01-01", progress=False, auto_adjust=False)
    rv, ret = {}, {}
    for name, tk in INDICES.items():
        o, h, l, c = (raw[k][tk] for k in ["Open", "High", "Low", "Close"])
        ok = (h > l) & (o > 0) & ~((o == h) & (c == l)) & ~((o == l) & (c == h))
        g = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
        rv[name] = g[ok & (g > 0)]
        ret[name] = np.log(c).diff()[ok]
    rv = pd.DataFrame(rv).dropna(axis=1, how="all"); rv.index = rv.index.tz_localize(None)
    ret = pd.DataFrame(ret)[rv.columns]; ret.index = ret.index.tz_localize(None)
    rv.to_csv("data/rv_daily.csv"); ret.to_csv("data/rv_returns.csv")
    s = rv.agg(["count", "mean"]).T
    s["first"] = rv.apply(lambda c: c.first_valid_index().date()); s["ann_vol_pct"] = 100 * np.sqrt(252 * s["mean"])
    print(s.to_string())
