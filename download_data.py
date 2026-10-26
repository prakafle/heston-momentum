"""Download the multi-asset universe.

data/universe_prices.csv : adjusted closes for the drift (return-momentum) tests and the vol-pair underlyings
data/vol_indices.csv     : Cboe 30-day implied-vol indices (VIX-style), straight from cdn.cboe.com
"""
import io
import warnings
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# implied-vol index -> underlying it is computed from
VOL_PAIRS = {
    "VIX": "^GSPC", "VXN": "^NDX", "RVX": "^RUT", "VXD": "^DJI",
    "VXEEM": "EEM", "VXEFA": "EFA", "VXFXI": "FXI", "VXEWZ": "EWZ",
    "GVZ": "GLD", "VXSLV": "SLV", "OVX": "USO", "VXGDX": "GDX", "VXTLT": "TLT",
    "VXAPL": "AAPL", "VXAZN": "AMZN", "VXGOG": "GOOG", "VXIBM": "IBM", "VXGS": "GS",
}
# S&P 500 implied vol at other maturities (9 days, 3 and 6 months): the option market's variance term structure
TERM_STRUCTURE = ["VIX9D", "VIX3M", "VIX6M"]
DRIFT_UNIVERSE = [
    # equity indices with long histories
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^N225", "^FTSE", "^GDAXI", "^HSI", "^GSPTSE", "^AXJO",
    # US sectors
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    # international / EM
    "EFA", "EEM", "EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ", "FXI", "EWY", "EWT",
    # rates and credit
    "TLT", "IEF", "SHY", "LQD", "HYG",
    # commodities and FX
    "GLD", "SLV", "USO", "DBC", "GDX", "FXE", "FXY", "UUP",
    # single names with vol indices
    "AAPL", "AMZN", "GOOG", "IBM", "GS",
]


def cboe(name):
    r = requests.get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv", timeout=30)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    col = "CLOSE" if "CLOSE" in d.columns else name
    return pd.Series(d[col].values, pd.to_datetime(d["DATE"]), name=name)


if __name__ == "__main__":                       # other scripts import the lists above without downloading
    tickers = sorted(set(DRIFT_UNIVERSE) | set(VOL_PAIRS.values()) | {"^NDX"})
    px = yf.download(tickers, start="1985-01-01", progress=False, auto_adjust=True)["Close"]
    px.index = px.index.tz_localize(None)
    px.to_csv("data/universe_prices.csv")
    vols = pd.concat([cboe(n) for n in VOL_PAIRS] + [cboe(n) for n in TERM_STRUCTURE], axis=1).sort_index()
    vols.to_csv("data/vol_indices.csv")
    print(px.notna().sum().sort_values().to_string())
    print(vols.notna().sum().to_string())
