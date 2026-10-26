# Momentum, Heston and Machine Learning: can modified pricing equations predict options better?

Erdős Institute final project. All results below are out-of-sample: every model is refit once a year on past data
only (expanding window, 21-day embargo), and every comparison carries a Newey-West / Diebold-Mariano t-stat.

## Question
The Heston stochastic-volatility model is the standard "known equation" for option prices. We modify it, mainly with
**momentum** (a trend term in the variance dynamics), but also with jumps, a second variance factor and
realised-variance feedback, and add **machine learning** on the same inputs. Then we ask:

1. **Predicting option prices over time:** do the modified equations forecast 30-day implied vol (Cboe VIX-style
   indices = prices of option strips) 1 week and 1 month ahead better than plain Heston and a random walk?
2. **Pricing the option smile:** does a modified equation price SPY options across strikes and expiries better?
3. **Where does momentum live:** does trend predict returns (drift)? Realised volatility? And is it already in
   option prices? A key fact: momentum in the *drift* cannot affect option prices at all, because under the
   risk-neutral measure the drift is replaced by r. Only momentum in the *variance* can.

## Data
- 48 assets (indices, sectors, countries, bonds, commodities, FX, single stocks), Yahoo, back to 1985–2013.
- 18 Cboe implied-vol indices with their underlyings (VIX/SPX, VXN/NDX, RVX/RUT, VXD/DJI, VXEEM, VXEFA, VXFXI,
  VXEWZ, GVZ, VXSLV, OVX, VXGDX, VXTLT, VXAPL, VXAZN, VXGOG, VXIBM, VXGS), from cdn.cboe.com; plus VIX9D/VIX3M/VIX6M.
- SPY option chain snapshots (`data/options/`, one per run day).

## Findings

### 1. Predicting implied vol (option prices): plain Heston wins (`iv_forecast.py`)
Average over 18 markets, 21-day horizon. OOS R² is relative to the random walk "IV stays the same".

| model | RMSE (vol pts) | OOS R² vs random walk |
|---|---|---|
| Random walk | 7.40 | 0 |
| **Heston (mean reversion)** | **6.96** | **+10.6%** |
| + IV momentum | 6.95 | +10.7% |
| + second factor (double Heston) | 6.93 | +10.0% |
| + jumps (Bates proxy) | 6.99 | +9.7% |
| + return trend (momentum) | 7.00 | +9.5% |
| + realised-variance feedback | 6.99 | +8.6% |
| All modifications together | 7.27 | −0.8% |
| Elastic net (all features) | 7.06 | +7.5% |
| Gradient boosting (all features) | 7.72 | −11.1% |

- Heston's mean reversion beats the random walk on 16/18 markets (6 significantly; pooled t = −1.80).
- No single modification significantly improves on Heston at 21 days. Adding all of them overfits (pooled t = +5.8,
  worse). Regularisation (elastic net) fixes much of that (t = −4.8 vs unregularised) but still doesn't beat Heston.
- VIX only: the option term structure (VIX9D/3M/6M) lifts R² from 17.3% to 19.4%, and the double-Heston slow factor
  to 19.3%. A genuine second variance factor shows up when you can observe it directly.

### 2. Pricing the SPY smile: jumps help most (`smile_models.py`, 2026-10-23 chain, 219 options, 8 expiries)
| model | in-sample IV RMSE | leave-one-expiry-out IV RMSE |
|---|---|---|
| Heston | 0.64 | 0.63 |
| **Bates (Heston + jumps)** | **0.33** | **0.43** (−32%) |
| Double Heston | 0.38 | 0.51 |

Bates fits a rare crash (λ ≈ 0.09/yr, mean jump −41%), which makes the steep short-dated skew that Heston can only
match with an implausible vol-of-vol (ξ = 1.76, Feller badly violated). Double Heston hits parameter bounds (ξ₁ = 4,
ρ₁ = −1), so treat it as over-flexible. Caveat: one day of data. The next-day test (calibrate today, predict
tomorrow's smile from the spot move vs sticky-strike/moneyness) runs automatically once ≥ 2 snapshots exist.

### 3. Momentum: real in volatility, already priced by options, absent in returns (`panel_backtest.py`)
- **Returns (48 assets):** no predictability. Pooled OOS R² −0.7% to −1.9%. Only 2–6 of 48 assets pass a 5% test (≈ chance).
  A diversified trend-following portfolio (Sharpe 0.55–0.71) does not beat buy-and-hold (0.75).
- **Realised volatility (18 markets, QLIKE):** a downside-trend term significantly improves a history-only
  forecast (pooled t = −2.84, better in 13/18 markets and 22/33 years). Once implied vol is in the model the gain
  disappears (t = −0.57) and the trend coefficient shrinks 40–90%: **the option market already prices it.**
  Gold and silver show the opposite sign (vol rises after rallies): the "inverse leverage" effect.
- **Implied vol is the best single forecast of realised vol** (beats the best history model in 12/18 markets).
- **Machine learning:** per-market gradient boosting is significantly *worse* than linear models. Trees cannot
  extrapolate into vol spikes beyond the training range. Boosting the residual of the equation model (hybrid) is
  also worse. A **pooled, scale-free** boosting model (one model for all 18 markets, features relative to implied
  vol) is the best ML model: on par with implied vol (better in 8/18, none significant). Permutation importance and
  ablation agree with the linear result: without implied vol, trend is the most important group; with implied vol,
  removing the trend features changes nothing (t = −0.04).
- **Trading (short 1-month variance at IV²):** always-short has Sharpe 0.59 (portfolio of 18). No forecast-based
  filter is significantly better in any market. The best is GARCH (0.82, not significant).

### 4. Monte Carlo: how much can these tests see? (`monte_carlo.py`, 5000 paths per point, HPC)
Simulated Heston world (κ = 5, θ = 0.03, ξ = 0.6, ρ = −0.7) with a known downside-trend term β in the variance drift
(or γ in the return drift), pushed through the exact panel B / panel A pipelines. Rejection rate at 5%:

| β (variance momentum) | trend helps, no IV | trend helps beyond IV, options price β | beyond IV, options miss β |
|---|---|---|---|
| 0 | 81% | 1.7% | 1.7% |
| 0.02 | 88% | 2.0% | 3.2% |
| 0.08 (matches the real S&P 500 coefficient) | 96% | 2.4% | 9.6% |

- **The no-IV test is not a test of β.** It rejects 81% of the time at β = 0: with ρ = −0.7 (leverage), falling
  prices raise variance, so a downside-trend term helps forecast realised vol in plain Heston. The real result in
  section 3 (t = −2.84) is therefore consistent with leverage alone and is not, by itself, evidence of momentum.
- **The with-IV test is correctly sized** (1.7% at β = 0, below 5%): when options price the trend it finds nothing,
  as in the data. **But it has little power**: even at the realistic β it detects unpriced momentum only 9.6% of
  the time. So "already priced by options" is consistent with the data, not proven by it.
- **Drift momentum is undetectable**: Clark-West rejects 6–7% at every γ up to 0.04 with 30-year paths (≈ size).
  The "no return predictability" result in section 3 is uninformative rather than evidence of absence.

### One-paragraph conclusion
Momentum is real in volatility but not in returns, and the option market already prices it. The parts of the
"known equation" that add predictive value are its structure: Heston's mean reversion forecasts implied vol (+10.6%
R² over a random walk), and jumps price the smile far better (−32% error). Extra state variables and machine learning
mostly overfit, unless the ML is pooled across markets and made scale-free, in which case it matches the market.

## Reproduce
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download_data.py      # ~1 min
.venv/bin/python panel_backtest.py     # ~10 min, results_panel.json
.venv/bin/python iv_forecast.py        # ~10 min, results_iv_forecast.json
.venv/bin/python smile_models.py       # ~5 min, results_smile.json (+ a new option snapshot on a trading day)
```
`backtest.py` and `calibrate_heston.py` are the original SPX-only study and single-day Heston calibration.

## Limitations
- Cboe vol indices for ETFs/single stocks are not directly tradeable. Variance-swap P&L uses assumed costs
  (0.5 / 1.0 / 1.5 vol pts for index / ETF / stock).
- Smile results use one day. Historical option chains are not freely available, so the snapshot archive
  grows only as `smile_models.py` is run on trading days.
- The momentum-in-variance model is tested in reduced form (regressions). The full β_P vs β_Q test (momentum
  coefficient fitted under P from returns vs under Q from the smile, across many days) needs the snapshot archive.
