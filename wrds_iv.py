"""30-day at-the-money implied vol for the 18 VOL_PAIRS underlyings, from the OptionMetrics volatility surface.

ATM IV = mean of the 50-delta call and -50-delta put implied vols on the 30-calendar-day surface (vsurfd, days = 30),
i.e. ATM-forward at the same 1-month horizon as the Cboe indices and hedging_backtest's 21-trading-day straddles.
Reads data/wrds/vsurfd.csv.gz and data/wrds/secids.csv; writes data/wrds/atm_iv.csv (vol points, one column per
vol-index name, like data/vol_indices.csv) and prints the index-minus-ATM gap per pair.
WRDS data: stays in data/wrds/ (gitignored); only aggregates are printed.
"""
import pandas as pd

from download_data import VOL_PAIRS

# Cboe vol index -> OptionMetrics ticker of the option underlying (indices: the cash index, DJX = DJIA / 100)
OM_TICKER = {"VIX": "SPX", "VXN": "NDX", "RVX": "RUT", "VXD": "DJX"}


def secid_map():
    """vol-index name -> list of (secid, start, end) spells (GOOG changes secid at the 2014 split)."""
    s = pd.read_csv("data/wrds/secids.csv", dtype={"start": str, "end": str})
    out = {}
    for name, under in VOL_PAIRS.items():
        t = OM_TICKER.get(name, under)
        out[name] = [(r.secid, r.start, r.end) for r in s[s.ticker == t].itertuples()]
        assert out[name], f"no secid for {name} ({t})"
    return out


def atm_iv():
    v = pd.read_csv("data/wrds/vsurfd.csv.gz", usecols=["secid", "date", "days", "delta", "impl_volatility"],
                    parse_dates=["date"])
    v = v[(v.days == 30) & v.delta.abs().eq(50) & (v.impl_volatility > 0)]
    atm = v.groupby(["secid", "date"]).impl_volatility.mean().unstack(0) * 100
    cols = {}
    for name, spells in secid_map().items():
        parts = []
        for secid, start, end in spells:
            x = atm[secid].dropna() if secid in atm else pd.Series(dtype=float)
            parts.append(x[(x.index >= (start if isinstance(start, str) else "1900"))
                           & (x.index <= (end if isinstance(end, str) else "2100"))])
        cols[name] = pd.concat(parts).sort_index()
    return pd.DataFrame(cols)


if __name__ == "__main__":
    a = atm_iv()
    a.to_csv("data/wrds/atm_iv.csv", float_format="%.4f")
    ivx = pd.read_csv("data/vol_indices.csv", index_col=0, parse_dates=True)
    rows = []
    for n in a:
        both = pd.concat([ivx[n], a[n]], axis=1, keys=["index", "atm"]).dropna()
        g = both["index"] - both["atm"]
        rows.append({"pair": n, "days": len(both), "first": both.index[0].date(), "last": both.index[-1].date(),
                     "index_mean": both["index"].mean(), "atm_mean": both["atm"].mean(), "gap_mean": g.mean(),
                     "gap_median": g.median(), "gap_p10": g.quantile(0.1), "gap_p90": g.quantile(0.9),
                     "corr": both["index"].corr(both["atm"])})
    pd.set_option("display.width", 200)
    print("Cboe index minus OptionMetrics 30-day ATM IV, vol points\n")
    print(pd.DataFrame(rows).set_index("pair").round(2).to_string())
