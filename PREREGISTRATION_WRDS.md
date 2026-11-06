# Pre-registered tests on OptionMetrics data (written 2026-11-05, before any of these tests was run)

These rules are fixed before the analyses are run. Any later change gets a dated entry in the change log at the bottom,
and results from changed rules are labelled post-hoc, as everywhere else in this repo (CLAUDE.md conventions:
no look-ahead, Newey-West t with lag = overlap, decimals in code).

**What had already been seen when this was written (disclosure):** the 2024 daily in-sample fit RMSEs of Heston,
Bates and double Heston on SPX (calibrate_wrds.py; mean 1.15 / 0.47 / 0.39 vol pts), a 4-day smoke test of the
next-day prediction (Jan 2–5 2024), the index-minus-ATM gap table (wrds_iv.py) and the hedging rerun at ATM IV.
No β regression of any kind and no 2008–2023 options have been looked at.

Common definitions (identical to panel_backtest.py / iv_forecast.py / monte_carlo.py):
- r_t = daily log return of the underlying (data/universe_prices.csv)
- R1_t = EWMA(r, halflife 10) · 252; downside trend D_t = max(-R1_t, 0)  ("R1neg")
- RV EWMA control: s_t = EWMA(r², halflife 20) · 252
- σ_τ(t) = OptionMetrics 30/91-day ATM IV = mean of the 50-delta call and −50-delta put on vsurfd (wrds_iv.py)

---

## Q1 (main): does the option market price momentum correctly? (β_Q vs β_P)

Hypothesis from the earlier work: a downtrend raises future realised variance (β_P > 0, panel B) and the option
market already prices it (adding D_t to IV did not help). Q1 measures the market's reaction directly.

**Market (Q) forecast of variance 1–3 months ahead** (forward variance between 30 and 91 calendar days):

    FV_t = (σ_91(t)² · 91 − σ_30(t)² · 30) / 61

**Realised (P) counterpart over the same window:** RV_t = (252 / 42) · Σ r² over trading days t+22 … t+63.

**Regressions** (one per underlying, daily data, all dates with both sides available):

    Q:     FV_t        = a + b·σ_30(t)² + c·s_t + β_Q · D_t + e
    P:     RV_t        = a + b·σ_30(t)² + c·s_t + β_P · D_t + e
    Diff:  FV_t − RV_t = a + b·σ_30(t)² + c·s_t + δ · D_t + e,     δ = β_Q − β_P

Newey-West t with lag 63 (targets overlap 63 days). **Primary test: δ for SPX, 1996-01 to 2025-08, two-sided 5%.**
- δ ≈ 0 (not significant): the market prices the momentum effect on variance correctly.
- δ < 0: options under-react (implied forward variance rises less than realised after a downtrend).
- δ > 0: options over-react.
Also report the same at 1 month: σ_30(t)² − RV over t+1 … t+21 (lag 21).

**Replication (pre-stated, not a second chance):** the same Diff regression on each of the other 17 underlyings
(per-pair δ and t). Pooled: one δ common to all 18 pairs, with pair-specific intercepts and control slopes
(each pair's target and D_t residualised on its own controls first, Frisch–Waugh), standard error by averaging the
score D̃_t·ẽ_t across pairs per date and applying Newey-West lag 63 to that series (the repo's "average across pairs
per date first" rule). Subperiods 1996–2009 and 2010–2025 reported for SPX.
**Robustness:** the same with Cboe variance-swap-style indices for SPX (VIX for 30d, VIX3M for 91d; 2007+).
**Model-based check (descriptive, no test):** from the daily Bates calibrations (calibrate_wrds.py), regress the
fitted long-run variance θ_t and the 91-day model forward variance on D_t with the same controls.

**If δ < 0 (only then), the trading follow-up**, fixed now: after a downtrend (D_t above its past-only 80th
percentile, expanding window), buy the 91-day ATM straddle at OptionMetrics IV + cost (panel C costs), delta-hedged
daily at entry IV for 21 trading days, vs the same trade on all other days. Mirror trade (sell) if δ > 0.

Limitations stated up front: ATM IV² is not the variance-swap rate (skew); the VIX/VIX3M robustness check addresses
it for SPX only. Power: the Monte Carlo study found ~10% power for the 21-day Cboe-index version; the 1996–2025 SPX
sample has ~10 non-overlapping 63-day windows per year, so a null result is weak evidence, and the write-up says so.

---

## Q2 (secondary): do jumps matter more in crises?

From calibrate_wrds.py runs on every downloaded year (2020–2023, 2024, 2025, plus 2008–2019 if downloaded):

    G_t = RMSE_Heston,t − RMSE_Bates,t   (vol points)  for (a) daily in-sample fit, (b) next-day prediction

    G_t = a + γ_1 · VIX_t + γ_2 · D_t + e,   Newey-West lag 5

**Hypothesis: γ_1 > 0** (Bates's advantage grows with the vol level); one-sided 5% on the next-day version (b),
which is the out-of-sample one; (a) reported alongside. Also a table of mean G by year and for the pre-stated
crisis windows: 2008-09-15 … 2009-03-31, 2020-02-20 … 2020-04-30, 2022-01-01 … 2022-10-31.
The same for double Heston vs Heston (descriptive). Rules of calibrate_wrds.py are not changed between years.

---

## Q3 (secondary): does the option-implied jump premium predict S&P 500 returns?

Predictors (daily, from the Bates calibrations and vsurfd):
- JQ_t = λ_t · (μJ_t² + δJ_t²): risk-neutral jump variance per year (Bates)
- VRP_t = σ_30(t)² − s_t: variance risk premium (benchmark predictor from the literature)

Target: SPX log return over the next 21 and 63 trading days, minus the zerocd rate at that horizon (excess).
Method exactly as panel A: expanding-window OLS refit every Jan 1 with the 21-row embargo (common.folds), first test
year = first year with 3 years of training data; OOS R² vs the historical mean; Clark-West test (NW lag = horizon).
Models: JQ alone, VRP alone, JQ + VRP. **Hypothesis: JQ predicts positive excess returns (OOS R² > 0, CW t > 1.645).**
Honest expectation: with 2008–2025 at most, this has low power; a null is reported as such.

---

## Data needed
- Q1: already downloaded (vsurfd, secprd/universe prices, vol_indices). Runs now.
- Q2, Q3: opprcd for SPX 2008–2023 (2020–2023 requested 2026-11-05; 2008–2019 to add, 3–4 years per query),
  2024 and 2025 (have / requested).

## Change log
- 2026-11-05: written.
