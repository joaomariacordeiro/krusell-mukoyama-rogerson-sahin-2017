"""Business cycle: shocks to frictions with constant prices (Sections II.B, III).

The only aggregate shock is a symmetric two-state Markov process (bad = 0,
good = 1) moving the friction rates: the good state has a high ``lambda_u``
(and proportionally high ``lambda_e``, ``lambda_n``) and a low separation
rate.  The wage per efficiency unit and the interest rate never move.

Households solve a value-function system with the aggregate state as an extra
state variable.  The economy is then simulated by propagating the full
cross-sectional distribution along one realised shock path -- a deterministic
simulation, not a Monte-Carlo panel.  Within a period, savings are chosen
under the current aggregate state, while the labour-market events and the
resulting choices occur under next period's state (the household observes the
new state before acting on the labour market, Section I.B timing).

Model output is Cobb-Douglas with a fixed capital-output ratio, so
``Y_t = w * L_t / (1 - capital share)`` with ``L_t`` aggregate efficiency
units of employed labour (Section III.A).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cross_section import (FineSolution, Population, build_operator, refine, step)
from discretize import Grids
from household import (Policies, Prices, Values, continuation_values,
                       cash_on_hand, evaluate_savings, maximize_savings,
                       zero_values, _assemble)
from parameters import BusinessCycle, Calibration, Frictions, Numerics


# --------------------------------------------------------------------------- #
# Aggregate states                                                             #
# --------------------------------------------------------------------------- #
def state_frictions(cal: Calibration, bc: BusinessCycle) -> list[Frictions]:
    """Frictions in the bad (0) and good (1) aggregate state (Section II.B)."""
    out = []
    for sgn in (-1.0, +1.0):
        lu = cal.lambda_u + sgn * bc.eps_lambda
        out.append(Frictions(
            lambda_u=lu,
            lambda_e=cal.lambda_e_ratio * lu,   # constant ratios to lambda_u
            lambda_n=cal.lambda_n_ratio * lu,
            lambda_s=lu,
            sigma=cal.sigma_sep - sgn * bc.eps_sigma,
            mu=cal.mu,
        ))
    return out


def transition_matrix(bc: BusinessCycle) -> NDArray:
    """Symmetric two-state chain with persistence rho on the diagonal."""
    return np.array([[bc.rho, 1.0 - bc.rho],
                     [1.0 - bc.rho, bc.rho]])


def shock_path(bc: BusinessCycle) -> NDArray:
    """One realised aggregate path of length ``n_months + 1``.

    Seeded and started in the good state so the simulation is reproducible.
    """
    Pi = transition_matrix(bc)
    cum = np.cumsum(Pi, axis=1)
    rng = np.random.default_rng(bc.seed)
    u = rng.random(bc.n_months + 1)
    path = np.empty(bc.n_months + 1, dtype=np.int64)
    path[0] = 1
    for t in range(1, bc.n_months + 1):
        path[t] = int(np.searchsorted(cum[path[t - 1]], u[t]))
    return path


# --------------------------------------------------------------------------- #
# Value functions with the aggregate state                                     #
# --------------------------------------------------------------------------- #
@dataclass
class CycleSolution:
    values: list[Values]      #: per aggregate state
    policies: list[Policies]
    fine: list[FineSolution]


def solve_cycle(cal: Calibration, num: Numerics, grids: Grids, prices: Prices,
                bc: BusinessCycle, *, v_init: Values | None = None,
                verbose: bool = False) -> CycleSolution:
    """Solve the household problem in both aggregate states jointly.

    The continuation in state ``s`` mixes next-period expectations across
    states with the aggregate chain:  R_s = sum_s' Pi[s, s'] * E[.| frictions
    and values of s'].  Prices are identical across states, so the budget
    sets are shared.  Warm-starting both states from the steady-state
    solution accelerates convergence without moving the fixed point.
    """
    frs = state_frictions(cal, bc)
    Pi = transition_matrix(bc)
    coh = cash_on_hand(cal, grids, prices)
    a, beta, gam = grids.a, cal.beta, grids.gamma
    n_states = len(frs)

    def clone(v: Values) -> Values:
        return Values(**{f: getattr(v, f).copy()
                         for f in ("W", "UE", "UN", "OE", "ON")})

    v = [clone(v_init) if v_init is not None else zero_values(grids)
         for _ in range(n_states)]
    pol = [Policies(W=np.zeros_like(coh["W"]), UE=np.zeros_like(coh["UE"]),
                    UN=np.zeros_like(coh["UN"]), OE=np.zeros_like(coh["UN"]),
                    ON=np.zeros_like(coh["UN"])) for _ in range(n_states)]

    def mixed_continuations(vs):
        inner = [continuation_values(vs[s], frs[s], grids) for s in range(n_states)]
        out = []
        for s in range(n_states):
            out.append(type(inner[0])(**{
                f: sum(Pi[s, sp] * getattr(inner[sp], f) for sp in range(n_states))
                for f in ("W", "UE", "UN", "OE", "ON")}))
        return out

    it = 0
    for it in range(1, num.max_value_iter + 1):
        Rs = mixed_continuations(v)
        new_v, new_pol = [], []
        for s in range(n_states):
            R = Rs[s]
            pW, vW = maximize_savings(coh["W"], R.W, a, beta, num)
            pUE, vUE = maximize_savings(coh["UE"], R.UE, a, beta, num)
            pUN, vUN = maximize_savings(coh["UN"], R.UN, a, beta, num)
            pOE, vOE = maximize_savings(coh["UN"], R.OE, a, beta, num)
            pON, vON = maximize_savings(coh["UN"], R.ON, a, beta, num)
            new_pol.append(Policies(W=pW, UE=pUE, UN=pUN, OE=pOE, ON=pON))
            new_v.append(_assemble(vW - cal.alpha, vUE, vUN, vOE, vON, gam))

        for _ in range(num.howard_steps):
            Rs = mixed_continuations(new_v)
            for s in range(n_states):
                R, p = Rs[s], new_pol[s]
                new_v[s] = _assemble(
                    evaluate_savings(coh["W"], R.W, a, beta, p.W, num) - cal.alpha,
                    evaluate_savings(coh["UE"], R.UE, a, beta, p.UE, num),
                    evaluate_savings(coh["UN"], R.UN, a, beta, p.UN, num),
                    evaluate_savings(coh["UN"], R.OE, a, beta, p.OE, num),
                    evaluate_savings(coh["UN"], R.ON, a, beta, p.ON, num),
                    gam,
                )

        err_v = sum(float(np.max(np.abs(getattr(new_v[s], f) - getattr(v[s], f))))
                    for s in range(n_states) for f in ("W", "UE", "UN", "OE", "ON"))
        err_p = sum(float(np.max(np.abs(getattr(new_pol[s], f) - getattr(pol[s], f))))
                    for s in range(n_states) for f in ("W", "UE", "UN", "OE", "ON"))
        v, pol = new_v, new_pol
        if verbose and it % 10 == 0:
            print(f"    cycle VFI sweep {it:4d}  d_value={err_v:.2e}  "
                  f"d_policy={err_p:.2e}", flush=True)
        if err_v < num.tol_value and err_p < num.tol_value:
            break
    if verbose:
        print(f"    cycle value functions solved in {it} sweeps", flush=True)

    fine = [refine(v[s], pol[s], grids) for s in range(n_states)]
    return CycleSolution(values=v, policies=pol, fine=fine)


# --------------------------------------------------------------------------- #
# Simulation                                                                   #
# --------------------------------------------------------------------------- #
def simulate(sol: CycleSolution, cal: Calibration, num: Numerics, grids: Grids,
             prices: Prices, bc: BusinessCycle, pop0: Population, *,
             verbose: bool = False) -> dict:
    """Propagate the distribution along the realised shock path.

    Returns monthly series: stocks, the six gross-flow rates, the job-to-job
    rate, efficiency units and output.  The operator for month ``t`` uses the
    savings policy of state ``s_t`` and the events/choices of state
    ``s_{t+1}``; flows and stocks are recorded at the start of the month.
    """
    frs = state_frictions(cal, bc)
    path = shock_path(bc)
    ops = {(s, sp): build_operator(sol.fine[s], sol.fine[sp], frs[sp], grids)
           for s in range(2) for sp in range(2)}

    T = bc.n_months
    keys = ("EE", "JJ", "EU", "EN", "UE", "UU", "UN", "NE", "NU", "NN",
            "urate", "lfpr", "E", "U", "N", "eL", "Y", "wage_gain")
    out = {k: np.empty(T) for k in keys}
    out["state"] = path[:T].astype(float)

    zq = grids.z[None, :, None] * grids.q[None, None, :]
    pop = pop0
    for t in range(T):
        nxt, fl = step(pop, ops[(int(path[t]), int(path[t + 1]))],
                       count_flows=True)
        eL = float((zq * pop.W).sum())
        for k in ("EE", "JJ", "EU", "EN", "UE", "UU", "UN", "NE", "NU", "NN",
                  "urate", "lfpr", "E", "U", "N", "wage_gain"):
            out[k][t] = fl[k]
        out["eL"][t] = eL
        out["Y"][t] = prices.w * eL / (1.0 - cal.capital_share)
        pop = nxt
        if verbose and (t + 1) % 1000 == 0:
            print(f"    simulated month {t + 1}/{T}", flush=True)
    return out
