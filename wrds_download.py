"""Targeted OptionMetrics pulls from WRDS into data/wrds/ (gitignored; never commit, never redistribute).

Login: credentials are read by libpq from ~/.pgpass (chmod 600), one line:
    wrds-pgdata.wharton.upenn.edu:9737:wrds:kaflepra:<password>
Dev nodes cannot reach WRDS directly (TLS handshake stalls); gateways can. So log in to the dev node with a tunnel:
    ssh -R 9737:wrds-pgdata.wharton.upenn.edu:9737 dev-amd20      # from the gateway
    export WRDS_TUNNEL=1
Run on that dev node (not a gateway): module load Python/3.14.2-GCCcore-15.2.0; unset PYTHONPATH
    .venv/bin/python wrds_download.py secids               # ticker -> secid map, check it before the rest
    .venv/bin/python wrds_download.py tier1                # vsurfd, stdopd, secprd for the 18 pairs; zerocd, idxdvd, distrd
    .venv/bin/python wrds_download.py opprcd 2019 2020 ... # filtered SPX/SPY option prices, one file per year
Prints only row counts and file names, so the raw data never lands in a terminal transcript.
WRDS may send a Duo push on the first connection from a new machine: approve it on your phone.
"""
import os
import sys

import pandas as pd
import psycopg

OUT = "data/wrds"
FIRST_YEAR = 1996
# underlyings of download_data.VOL_PAIRS, as OptionMetrics tickers (indices: SPX, NDX, RUT, DJX)
TICKERS = ["SPX", "NDX", "RUT", "DJX", "EEM", "EFA", "FXI", "EWZ", "GLD", "SLV", "USO", "GDX", "TLT",
           "AAPL", "AMZN", "GOOG", "IBM", "GS", "SPY"]
RAW_SECIDS = {"SPX": 108105, "SPY": 109820}  # opprcd universe (huge tables: keep it to these two)


def connect():
    print("Connecting to WRDS (approve the Duo push on your phone if one arrives)...", flush=True)
    # dev-node egress stalls the WRDS TLS handshake; WRDS_TUNNEL=1 goes through an ssh -R/-L tunnel on 127.0.0.1:9737
    # (host= still names WRDS so ~/.pgpass matches)
    tunnel = {"hostaddr": "127.0.0.1"} if os.environ.get("WRDS_TUNNEL") else {}
    return psycopg.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds", **tunnel,
                           user=os.environ.get("WRDS_USER", "kaflepra"), sslmode="require", connect_timeout=120)


def query(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def save(df, name):
    path = f"{OUT}/{name}.csv.gz"
    df.to_csv(path, index=False)
    print(f"{name}: {len(df):,} rows -> {path}", flush=True)


def years(conn, stem):
    """Year tables optionm.<stem>YYYY that exist on WRDS."""
    t = query(conn, "select table_name from information_schema.tables where table_schema='optionm' "
                    "and table_name ~ %s", (f"^{stem}[0-9]{{4}}$",))
    return sorted(int(n[-4:]) for n in t.table_name if int(n[-4:]) >= FIRST_YEAR)


def secids(conn):
    m = query(conn, "select secid, ticker, issuer, effect_date from optionm.secnmd where ticker = any(%s) "
                    "order by ticker, effect_date", (TICKERS,))
    save(m, "secnmd_candidates")
    # current owner of each ticker = latest effect_date; check the printout (ticker reuse, GOOG/GOOGL, etc.)
    cur = m.sort_values("effect_date").groupby("ticker").tail(1)[["ticker", "secid", "issuer"]]
    cur.to_csv(f"{OUT}/secids.csv", index=False)
    print(cur.to_string(index=False))  # metadata only (names and ids)


def load_secids():
    return [int(s) for s in pd.read_csv(f"{OUT}/secids.csv").secid]


def tier1(conn):
    ids = load_secids()
    for stem in ["vsurfd", "stdopd", "secprd"]:
        parts = [query(conn, f"select * from optionm.{stem}{y} where secid = any(%s)", (ids,))
                 for y in years(conn, stem)]
        save(pd.concat(parts, ignore_index=True), stem)
    save(query(conn, "select * from optionm.zerocd"), "zerocd")
    save(query(conn, "select * from optionm.idxdvd where secid = any(%s)", (ids,)), "idxdvd")
    save(query(conn, "select * from optionm.distrd where secid = any(%s)", (ids,)), "distrd")


def opprcd(conn, yrs):
    avail = set(years(conn, "opprcd"))
    for y in yrs:
        if y not in avail:
            print(f"opprcd{y}: not on WRDS, skipped")
            continue
        df = query(conn, f"""
            select o.secid, o.date, o.exdate, o.cp_flag, o.strike_price / 1000.0 as strike,
                   o.best_bid, o.best_offer, o.volume, o.open_interest, o.impl_volatility,
                   o.delta, o.gamma, o.vega, o.optionid, o.am_settlement, o.ss_flag, s.close as spot
            from optionm.opprcd{y} o
            join optionm.secprd{y} s on s.secid = o.secid and s.date = o.date
            where o.secid = any(%s) and o.best_bid > 0
              and o.exdate - o.date between 7 and 365
              and o.strike_price / 1000.0 between 0.7 * s.close and 1.3 * s.close""",
                   (list(RAW_SECIDS.values()),))
        save(df, f"opprcd_{y}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    step = sys.argv[1] if len(sys.argv) > 1 else "secids"
    with connect() as conn:
        if step == "secids":
            secids(conn)
        elif step == "tier1":
            tier1(conn)
        elif step == "opprcd":
            opprcd(conn, [int(a) for a in sys.argv[2:]])
        else:
            sys.exit(__doc__)
