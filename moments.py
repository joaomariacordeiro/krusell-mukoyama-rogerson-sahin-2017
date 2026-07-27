"""Business-cycle statistics (Section III.A) and the variance decomposition.

All model series are treated exactly as the paper treats the data: aggregated
to quarterly frequency (three-month averages for rates and flows, three-month
sums for output), logged, HP-filtered with smoothing parameter 1600, and then
summarised by the standard deviation, the contemporaneous correlation with
cyclical output, and the first-order autocorrelation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve


# --------------------------------------------------------------------------- #
# Filtering                                                                   #
# --------------------------------------------------------------------------- #
def hp_trend(y: NDArray, lam: float = 1600.0) -> NDArray:
    """Hodrick-Prescott trend: solve (I + lam * D'D) tau = y with D the
    second-difference operator"""
    n = y.size
    if n < 3:
        return y.copy()
    e = np.ones(n)
    D = sparse.spdiags(np.vstack([e, -2.0 * e, e]), [0, -1, -2], n, n - 2).tocsc()
    A = (sparse.eye(n, format="csc") + lam * (D @ D.T)).tocsc()
    return spsolve(A, y)


def cyclical(x: NDArray, lam: float = 1600.0) -> NDArray:
    """Cyclical component of log x"""
    lx = np.log(x)
    return lx - hp_trend(lx, lam)


def quarterly(x: NDArray, how: str = "mean") -> NDArray:
    """Aggregate monthly series to quarters (mean for rates, sum for Y)"""
    n = (x.size // 3) * 3
    blocks = x[:n].reshape(-1, 3)
    return blocks.mean(axis=1) if how == "mean" else blocks.sum(axis=1)


# --------------------------------------------------------------------------- #
# Cyclical moments (model columns of Tables 7, 8 panel C, 9)                  #
# --------------------------------------------------------------------------- #
def cycle_statistics(series: dict, burn_in: int) -> dict:
    """std, corr with output, and AR(1) for the stocks, flows and J2J rate"""
    cyc_y = cyclical(quarterly(series["Y"][burn_in:], "sum"))

    def stats(x_monthly: NDArray) -> tuple[float, float, float]:
        c = cyclical(quarterly(x_monthly[burn_in:], "mean"))
        return (float(np.std(c)),
                float(np.corrcoef(cyc_y, c)[0, 1]),
                float(np.corrcoef(c[:-1], c[1:])[0, 1]))

    urate, lfpr = series["urate"], series["lfpr"]
    employment = lfpr * (1.0 - urate)   # employment-population ratio
    out: dict = {"std_Y": float(np.std(cyc_y))}
    for name, x in [("urate", urate), ("lfpr", lfpr), ("E", employment),
                    ("feu", series["EU"]), ("fen", series["EN"]),
                    ("fue", series["UE"]), ("fun", series["UN"]),
                    ("fne", series["NE"]), ("fnu", series["NU"]),
                    ("fjj", series["JJ"])]:
        s, c, a = stats(x)
        out[f"std_{name}"] = s
        out[f"corr_{name}Y"] = c
        out[f"autocorr_{name}"] = a
    return out


# --------------------------------------------------------------------------- #
# Variance decomposition of unemployment changes (Section III.D, Table 11)     #
# --------------------------------------------------------------------------- #
def _flow_steady_u(feu, fen, fue, fun, fne, fnu) -> NDArray:
    """Flow-implied steady-state unemployment rate, month by month.

    For each month's six transition rates, the stationary distribution of the
    implied 3x3 chain gives (E*, U*, N*). The series returned is
    u* = U* / (E* + U*).  Solved as a 2x2 linear system in (E*, U*) using the
    balance equations with N* = 1 - E* - U*.
    """
    T = feu.size
    # Stationary condition pi = pi' M with M row-stochastic (rows = from):
    #   E*: (feu+fen+fne) E* - (fue-fne) U* = fne
    #   U*: (-feu+fnu) E* + (fue+fun+fnu) U* = fnu
    A = np.empty((T, 2, 2))
    A[:, 0, 0] = feu + fen + fne
    A[:, 0, 1] = -(fue - fne)
    A[:, 1, 0] = -(feu - fnu)
    A[:, 1, 1] = fue + fun + fnu
    b = np.stack([fne, fnu], axis=1)[..., None]     # (T, 2, 1) column vectors
    eu = np.linalg.solve(A, b)[..., 0]              # (T, 2): (E*, U*)
    return eu[:, 1] / (eu[:, 0] + eu[:, 1])


def variance_decomposition(series: dict, burn_in: int) -> dict:
    """Share of cyclical u* variance attributable to each flow pair.

    Counterfactual-covariance method: recompute the flow-implied u* letting
    only one pair of flows vary (the others held at their sample means), and
    attribute ``100 * cov(cyc u*, cyc u*_pair) / var(cyc u*)`` percent to that
    pair.  Shares sum to roughly 100.  This implements the spirit of the
    Elsby-Hobijn-Sahin (2015) decomposition used in the paper's Table 11. The
    paper's exact implementation differs in detail, so this exercise is
    qualitative (see readme and term paper for further details).
    """
    f = {k: series[k][burn_in:] for k in ("EU", "EN", "UE", "UN", "NE", "NU")}
    mean = {k: np.full_like(x, x.mean()) for k, x in f.items()}

    def u_star(varying: set[str]) -> NDArray:
        args = {k: (f[k] if k in varying else mean[k]) for k in f}
        return _flow_steady_u(args["EU"], args["EN"], args["UE"],
                              args["UN"], args["NE"], args["NU"])

    cyc_full = cyclical(quarterly(u_star(set(f)), "mean"))
    var_full = float(np.var(cyc_full))
    pairs = {"U_E": {"EU", "UE"}, "U_N": {"UN", "NU"}, "E_N": {"EN", "NE"}}
    out = {}
    for name, sel in pairs.items():
        cyc_pair = cyclical(quarterly(u_star(sel), "mean"))
        out[name] = 100.0 * float(np.cov(cyc_full, cyc_pair)[0, 1]) / var_full
    out["total"] = sum(out[p] for p in pairs)
    return out
