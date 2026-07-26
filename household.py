"""The household problem: budgets, Bellman recursion, and its solution.

The individual chooses consumption/saving and a discrete labour-market
activity.  Five value functions carry the recursion (Section I.B-I.C):

    W (a, z, q)      employed at match quality q
    UE(a, z, gamma)  not employed, searching actively, UI-eligible
    UN(a, z, gamma)  not employed, searching actively, not eligible
    OE(a, z)         not employed, passive, UI-eligible
    ON(a, z)         not employed, passive, not eligible

Passive search carries no search cost and no UI receipt (benefits require
active search), so OE/ON do not depend on gamma.  The composite objects of the
paper are recovered as maxima:

    J  = max{U, O}                   jobless value (participation choice)
    V  = max{W(q), J}                offer in hand (accept/reject)

Timing (Section I.B): all shocks realise at the start of the period; the
period budget is

    c + a' = (1 + r) a + (1 - tau) w z q  [if employed]
                       + (1 - tau) b(z)   [if active and eligible]
                       + T,

with a' >= 0 (no borrowing) and log utility net of the activity disutility
(alpha if working, gamma if actively searching).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from discretize import Grids
from parameters import Calibration, Frictions, Numerics

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


# --------------------------------------------------------------------------- #
# Prices and budgets                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Prices:
    """Aggregate prices, constant over the business cycle (Section II.B)"""

    w: float             #: wage per efficiency unit of labour
    r: float             #: monthly net interest rate
    transfer: float      #: lump-sum transfer T (balances the budget in steady state)
    avg_earnings: float  #: average z*q among the employed (sets the UI cap)

    @classmethod
    def from_kl(cls, cal: Calibration, kl: float, avg_earnings: float,
                transfer: float) -> "Prices":
        """Cobb-Douglas prices implied by capital per efficiency unit
        (Section II.A): w = (1-a) (K/L)^a,  r = a (K/L)^(a-1) - delta"""
        a = cal.capital_share
        return cls(w=(1.0 - a) * kl ** a,
                   r=a * kl ** (a - 1.0) - cal.depreciation,
                   transfer=transfer, avg_earnings=avg_earnings)

    @classmethod
    def converged(cls, cal: Calibration) -> "Prices":
        """Prices at the converged background-GE constants (Section II.A)."""
        return cls.from_kl(cal, cal.kl_star, cal.avgw_star, cal.transfer_star)


def ui_schedule(cal: Calibration, grids: Grids, prices: Prices) -> NDArray:
    """UI benefit b(z) = min(b0 * w * z, cap), cap = 0.465 * w * avg earnings.

    Benefits are indexed to current productivity rather than past earnings
    (Section I.C), which keeps the state space small.
    """
    return np.minimum(cal.ui_replacement * prices.w * grids.z,
                      cal.ui_cap_ratio * prices.w * prices.avg_earnings)


def cash_on_hand(cal: Calibration, grids: Grids, prices: Prices) -> dict:
    """Resources available before the savings choice, by labour-market state"""
    gross = (1.0 + prices.r) * grids.a + prices.transfer          # (A,)
    b = ui_schedule(cal, grids, prices)
    return {
        # employed: after-tax earnings w*z*q
        "W": gross[:, None, None] + (1.0 - cal.tau)
             * prices.w * grids.z[None, :, None] * grids.q[None, None, :],
        # active & eligible: after-tax UI benefit
        "UE": gross[:, None] + (1.0 - cal.tau) * b[None, :],
        # all remaining non-employed states: transfer and asset income only
        "UN": np.broadcast_to(gross[:, None], (grids.a.size, grids.z.size)),
    }


# --------------------------------------------------------------------------- #
# Value and policy containers                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Values:
    """The five value functions (shapes in the module docstring)"""

    W: NDArray    # (A, Z, Q)
    UE: NDArray   # (A, Z, G)
    UN: NDArray   # (A, Z, G)
    OE: NDArray   # (A, Z)
    ON: NDArray   # (A, Z)


@dataclass
class Policies:
    """Savings choices a'.

    The search cost enters the active-search problems additively, so it does
    not affect the savings choice: UE and UN policies are gamma-independent
    (and the values satisfy UE(a,z,gamma) = UE_base(a,z) - gamma).
    """

    W: NDArray    # (A, Z, Q)
    UE: NDArray   # (A, Z)
    UN: NDArray   # (A, Z)
    OE: NDArray   # (A, Z)
    ON: NDArray   # (A, Z)


def zero_values(grids: Grids) -> Values:
    A, Z, Q, G = grids.a.size, grids.z.size, grids.q.size, grids.gamma.size
    return Values(W=np.zeros((A, Z, Q)), UE=np.zeros((A, Z, G)),
                  UN=np.zeros((A, Z, G)), OE=np.zeros((A, Z)), ON=np.zeros((A, Z)))


# --------------------------------------------------------------------------- #
# The Bellman expectation (Section I.B)                                        #
# --------------------------------------------------------------------------- #
@dataclass
class Continuation:
    """Expected continuation values R(a', z), by current state, before the
    flow payoff.  The a'-axis indexes next-period assets."""

    W: NDArray    # (A, Z, Q)
    UE: NDArray   # (A, Z)
    UN: NDArray   # (A, Z)
    OE: NDArray   # (A, Z)
    ON: NDArray   # (A, Z)


def continuation_values(v: Values, fr: Frictions, grids: Grids) -> Continuation:
    """Take expectations over (z', q', gamma') of next period's optimal choice.

    Composite objects, conditional on next period's eligibility ("E"/"N"):

        NE, NN = max{U, O}                jobless value J
        FE, FN = E_q' max{W(q'), J}       value of receiving a fresh offer
        V      = max{W(q), J_N}           offer q in hand vs quitting
        L      = E_q' max{W(q), W(q'), J_N}   the on-the-job ladder: keep the
                                              better of current and outside q

    A worker who stays employed and later quits is not eligible (hence J_N in
    V and L). A separated worker IS eligible (Section I.C).  The employed
    continuation mixes the four mutually exclusive events of Section I.B:

        (1-sigma-lambda_e) V  +  lambda_e L
        + sigma [ lambda_s FE + (1-lambda_s) NE ],

    and the searchers' continuations mix offer arrival with eligibility decay
    at rate mu (eligible searchers keep eligibility w.p. 1-mu).
    """
    w_q, w_g, Pi = grids.w_q, grids.w_gamma, grids.Pi_z
    lu, le, ln, ls = fr.lambda_u, fr.lambda_e, fr.lambda_n, fr.lambda_s
    sig, mu = fr.sigma, fr.mu

    NE = np.maximum(v.UE, v.OE[:, :, None])                       # (A,Z,G)
    NN = np.maximum(v.UN, v.ON[:, :, None])
    W_vs_NE = np.maximum(v.W[:, :, :, None], NE[:, :, None, :])   # (A,Z,Q,G)
    W_vs_NN = np.maximum(v.W[:, :, :, None], NN[:, :, None, :])
    FE = np.einsum("azpg,p->azg", W_vs_NE, w_q)
    FN = np.einsum("azpg,p->azg", W_vs_NN, w_q)
    V = W_vs_NN                                                   # (A,Z,Q,G)
    ladder = np.maximum(v.W[:, :, :, None], v.W[:, :, None, :])   # (A,Z,Q,P)
    L = np.einsum("azqpg,p->azqg",
                  np.maximum(ladder[..., None], NN[:, :, None, None, :]), w_q)

    # Integrate over gamma' (i.i.d., weights w_g).
    e_W = (np.einsum("azqg,g->azq", (1.0 - sig - le) * V + le * L, w_g)
           + np.einsum("azg,g->az",
                       sig * (ls * FE + (1.0 - ls) * NE), w_g)[:, :, None])
    e_UE = np.einsum("azg,g->az",
                     (1.0 - mu) * (lu * FE + (1.0 - lu) * NE)
                     + mu * (lu * FN + (1.0 - lu) * NN), w_g)
    e_UN = np.einsum("azg,g->az", lu * FN + (1.0 - lu) * NN, w_g)
    e_OE = np.einsum("azg,g->az",
                     (1.0 - mu) * (ln * FE + (1.0 - ln) * NE)
                     + mu * (ln * FN + (1.0 - ln) * NN), w_g)
    e_ON = np.einsum("azg,g->az", ln * FN + (1.0 - ln) * NN, w_g)

    # Integrate over z' with the Tauchen chain: R(., z) = sum_z' Pi[z,z'] e(., z').
    return Continuation(
        W=np.einsum("zy,ayq->azq", Pi, e_W),
        UE=np.einsum("zy,ay->az", Pi, e_UE),
        UN=np.einsum("zy,ay->az", Pi, e_UN),
        OE=np.einsum("zy,ay->az", Pi, e_OE),
        ON=np.einsum("zy,ay->az", Pi, e_ON),
    )


# --------------------------------------------------------------------------- #
# Savings choice                                                               #
# --------------------------------------------------------------------------- #
def interp_at(grid: NDArray, table: NDArray, x: NDArray) -> NDArray:
    """Linear interpolation of ``table`` (n, *S) along axis 0 at points ``x`` (*S)"""
    n = grid.size
    xc = np.clip(x, grid[0], grid[-1])
    i = np.clip(np.searchsorted(grid, xc, side="right") - 1, 0, n - 2)
    t = (xc - grid[i]) / (grid[i + 1] - grid[i])
    lo = np.take_along_axis(table, i[None, ...], axis=0)[0]
    hi = np.take_along_axis(table, i[None, ...] + 1, axis=0)[0]
    return (1.0 - t) * lo + t * hi


def _objective(ap, coh, table, a_grid, beta, num):
    """Period value of saving ``ap``: log consumption plus discounted continuation.
    (Activity disutility is an additive constant and is applied by the caller.)"""
    c = np.maximum(coh - ap, num.tiny)
    return np.log(c) + beta * interp_at(a_grid, table, ap)


def maximize_savings(coh: NDArray, R: NDArray, a_grid: NDArray, beta: float,
                     num: Numerics) -> tuple[NDArray, NDArray]:
    """Golden-section maximisation of the savings problem at every grid node.

    The feasible set is a' in [0, min(coh, a_max)].  The objective (log
    consumption + interpolated continuation) is unimodal in a'; the interior
    golden-section optimum is compared against both endpoints to catch corner
    solutions (in particular the borrowing constraint a' = 0).
    """
    S = coh.shape
    table = np.broadcast_to(R[:, None, ...], (R.shape[0],) + S)

    lo = np.zeros(S)
    hi = np.minimum(a_grid[-1] - num.tiny, np.maximum(0.0, coh))
    lo0, hi0 = lo.copy(), hi.copy()

    x1 = hi - _GOLDEN * (hi - lo)
    x2 = lo + _GOLDEN * (hi - lo)
    f1 = _objective(x1, coh, table, a_grid, beta, num)
    f2 = _objective(x2, coh, table, a_grid, beta, num)
    for _ in range(60):
        move_up = f2 >= f1
        lo = np.where(move_up, x1, lo)
        hi = np.where(move_up, hi, x2)
        x1 = hi - _GOLDEN * (hi - lo)
        x2 = lo + _GOLDEN * (hi - lo)
        f1 = _objective(x1, coh, table, a_grid, beta, num)
        f2 = _objective(x2, coh, table, a_grid, beta, num)
        if float(np.max(hi - lo)) < num.tol_golden:
            break

    a_star = np.where(f1 >= f2, x1, x2)
    v_star = np.maximum(f1, f2)
    for corner in (lo0, hi0):
        f = _objective(corner, coh, table, a_grid, beta, num)
        better = f > v_star
        a_star = np.where(better, corner, a_star)
        v_star = np.where(better, f, v_star)
    return a_star, v_star


def evaluate_savings(coh: NDArray, R: NDArray, a_grid: NDArray, beta: float,
                     policy: NDArray, num: Numerics) -> NDArray:
    """Value of a fixed savings policy, with no optimisation.  Used by the
    Howard acceleration step in :func:`solve_household`"""
    table = np.broadcast_to(R[:, None, ...], (R.shape[0],) + coh.shape)
    return _objective(policy, coh, table, a_grid, beta, num)


# --------------------------------------------------------------------------- #
# Value-function iteration                                                     #
# --------------------------------------------------------------------------- #
def _expand_gamma(base: NDArray, gamma: NDArray) -> NDArray:
    """Attach the additive search cost: U(a,z,gamma) = U_base(a,z) - gamma."""
    return base[:, :, None] - gamma[None, None, :]


def _assemble(vW, vUE_base, vUN_base, vOE, vON, gamma) -> Values:
    return Values(W=vW, UE=_expand_gamma(vUE_base, gamma),
                  UN=_expand_gamma(vUN_base, gamma), OE=vOE, ON=vON)


def solve_household(cal: Calibration, num: Numerics, grids: Grids,
                    prices: Prices, fr: Frictions, *,
                    v_init: Values | None = None,
                    verbose: bool = False) -> tuple[Values, Policies, int]:
    """Solve the five-problem Bellman system by value-function iteration

    Each iteration first computes the expected continuation values and
    re-optimises the savings choice at every grid point.  The value is then
    updated several more times with that savings choice held fixed, with
    no further optimisation  costs (Howard's policy-improvement idea).  This
    accelerates convergence without changing the fixed point.  The loop stops
    once both the values and the savings choices have settled.

    Work disutility alpha and search cost gamma are additive constants, so
    they are applied after the savings maximisation.  This also makes the
    UE/UN savings problems independent of gamma.
    """
    coh = cash_on_hand(cal, grids, prices)
    a, beta, gam = grids.a, cal.beta, grids.gamma
    v = v_init if v_init is not None else zero_values(grids)
    pol = Policies(W=np.zeros_like(coh["W"]), UE=np.zeros_like(coh["UE"]),
                   UN=np.zeros_like(coh["UN"]), OE=np.zeros_like(coh["UN"]),
                   ON=np.zeros_like(coh["UN"]))

    it = 0
    for it in range(1, num.max_value_iter + 1):
        R = continuation_values(v, fr, grids)

        # Re-optimise the savings choice at every grid point, one problem at a
        # time (the activity disutility is an additive constant, added after).
        pW, vW = maximize_savings(coh["W"], R.W, a, beta, num)
        pUE, vUE = maximize_savings(coh["UE"], R.UE, a, beta, num)
        pUN, vUN = maximize_savings(coh["UN"], R.UN, a, beta, num)
        pOE, vOE = maximize_savings(coh["UN"], R.OE, a, beta, num)
        pON, vON = maximize_savings(coh["UN"], R.ON, a, beta, num)
        new_pol = Policies(W=pW, UE=pUE, UN=pUN, OE=pOE, ON=pON)
        new_v = _assemble(vW - cal.alpha, vUE, vUN, vOE, vON, gam)

        # Update the value  with the savings choice held fixed (Howard acceleration)
        for _ in range(num.howard_steps):
            R = continuation_values(new_v, fr, grids)
            new_v = _assemble(
                evaluate_savings(coh["W"], R.W, a, beta, pW, num) - cal.alpha,
                evaluate_savings(coh["UE"], R.UE, a, beta, pUE, num),
                evaluate_savings(coh["UN"], R.UN, a, beta, pUN, num),
                evaluate_savings(coh["UN"], R.OE, a, beta, pOE, num),
                evaluate_savings(coh["UN"], R.ON, a, beta, pON, num),
                gam,
            )

        err_v = sum(float(np.max(np.abs(getattr(new_v, s) - getattr(v, s))))
                    for s in ("W", "UE", "UN", "OE", "ON"))
        err_p = sum(float(np.max(np.abs(getattr(new_pol, s) - getattr(pol, s))))
                    for s in ("W", "UE", "UN", "OE", "ON"))
        v, pol = new_v, new_pol
        if verbose and it % 10 == 0:
            print(f"    VFI iteration {it:4d}  d_value={err_v:.2e}  d_policy={err_p:.2e}",
                  flush=True)
        if err_v < num.tol_value and err_p < num.tol_value:
            break
    return v, pol, it
