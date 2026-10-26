"""Characteristic-function option pricing for Heston and its modifications (all on forwards, so no dividend input).

    Heston         dv = κ(θ - v)dt + ξ√v dZ,  corr(dW, dZ) = ρ
    Bates          Heston + lognormal jumps in S: intensity λ, log-jump mean μJ, std δJ (compensated)
    Double Heston  two independent Heston variance factors, v = v1 + v2 (one fast, one slow mean reversion)

Prices use Gil-Pelaez inversion with 160-node Gauss-Legendre on u in (0, 250]. Same scheme as calibrate_heston.py.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

U, W = np.polynomial.legendre.leggauss(160)
U = 0.5 * (U + 1) * 250 + 1e-8; W = W * 0.5 * 250


def heston_log_cf(u, T, kappa, theta, xi, rho, v0):
    """log E[exp(i u (ln F_T - ln F_0))] under Q ("little trap" form)."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    return C + D * v0


def jump_log_cf(u, T, lam, muj, dj):
    iu = 1j * u
    return lam * T * (np.exp(iu * muj - 0.5 * dj**2 * u**2) - 1 - iu * (np.exp(muj + 0.5 * dj**2) - 1))


MODELS = {
    # name: (log-cf(u, T, params), parameter names, x0, lower bounds, upper bounds)
    "Heston": (lambda u, T, p: heston_log_cf(u, T, *p),
               ["kappa", "theta", "xi", "rho", "v0"],
               [2.0, 0.04, 0.6, -0.7, 0.03], [0.05, 0.001, 0.05, -0.999, 0.0005], [20, 1.0, 4.0, 0.5, 1.0]),
    "Bates (Heston + jumps)": (lambda u, T, p: heston_log_cf(u, T, *p[:5]) + jump_log_cf(u, T, *p[5:]),
               ["kappa", "theta", "xi", "rho", "v0", "lam", "muj", "dj"],
               [2.0, 0.04, 0.5, -0.7, 0.03, 0.3, -0.08, 0.08],
               [0.05, 0.001, 0.05, -0.999, 0.0005, 0.0, -0.5, 0.01], [20, 1.0, 4.0, 0.5, 1.0, 5.0, 0.2, 0.5]),
    "Double Heston (2 factors)": (lambda u, T, p: heston_log_cf(u, T, *p[:5]) + heston_log_cf(u, T, *p[5:]),
               ["kappa1", "theta1", "xi1", "rho1", "v01", "kappa2", "theta2", "xi2", "rho2", "v02"],
               [8.0, 0.02, 1.0, -0.7, 0.015, 0.5, 0.02, 0.3, -0.7, 0.015],
               [0.5, 0.0005, 0.05, -0.999, 0.0001, 0.01, 0.0005, 0.05, -0.999, 0.0001],
               [30, 1.0, 4.0, 0.5, 1.0, 3.0, 1.0, 2.0, 0.5, 1.0]),
}


def call_prices(logcf, p, F, K, T, df):
    """Undiscounted-forward Gil-Pelaez call prices for strikes K at one expiry."""
    K = np.atleast_1d(K)[:, None]
    phi = np.exp(logcf(U, T, p))
    phi1 = np.exp(logcf(U - 1j, T, p))                    # E[F_T/F_0 · e^{iu ln(F_T/F_0)}]
    lk = np.exp(-1j * U * np.log(K / F))
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
_chk = call_prices(MODELS["Heston"][0], (2.0, 0.04, 1e-3, 0.0, 0.04), 100.0, np.array([90.0, 100.0, 110.0]), 0.5, 0.98)
assert np.allclose(_chk, black(100.0, np.array([90.0, 100.0, 110.0]), 0.5, 0.98, 0.2, True), atol=2e-3), _chk
