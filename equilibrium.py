"""Steady state and the background-GE price calibration (Section II.A).

The model is partial equilibrium: households take the wage per efficiency
unit ``w``, the interest rate ``r`` and the transfer ``T`` as given.  KMRS
pin down the levels of these constants by requiring consistency with a
background Cobb-Douglas economy.  At the stationary distribution, the
capital-labour ratio implied by household wealth must equal the one behind
the prices, the average-earnings reference in the UI cap must equal average
earnings among the employed, and ``T`` must balance the government budget
(taxes minus UI outlays).  This is a one-time calibration , found as a damped 
fixed point, not by period-by-period market clearing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cross_section import (FineSolution, Operator, Population, build_operator,
                           refine, stationary, step)
from discretize import Grids, build_grids
from household import Policies, Prices, Values, solve_household, ui_schedule
from parameters import BusinessCycle, Calibration, Frictions, Numerics


@dataclass
class SteadyState:
    prices: Prices
    values: Values
    policies: Policies
    fine: FineSolution
    population: Population
    flows: dict            # average gross-flow rates and stocks (Table 5)
    aggregates: dict
    grids: Grids
    operator: Operator


def household_aggregates(pop: Population, grids: Grids, prices: Prices,
                         cal: Calibration) -> dict:
    """Aggregates implied by the stationary distribution.

    ``L`` is efficiency units supplied by the employed (sum of z*q); wealth is
    total asset holdings; the government budget nets the labour-income and
    UI-benefit tax base against UI outlays (only eligible *active* searchers
    collect benefits, Section I.C).
    """
    zq = grids.z[None, :, None] * grids.q[None, None, :]
    L = float((zq * pop.W).sum())
    wealth = float(grids.a_fine @ pop.asset_marginal())
    b = ui_schedule(cal, grids, prices)
    ui_paid = float((b[None, :] * pop.UE).sum())
    taxes = cal.tau * (prices.w * L + ui_paid)
    return {
        "L": L,
        "avg_earnings": L / pop.employed,
        "wealth": wealth,
        "KL": wealth / L,
        "ui_paid": ui_paid,
        "taxes": taxes,
        "transfer_balancing": taxes - ui_paid,
    }


def _solve_at(cal, num, grids, fr, prices, *, v_init=None, pop0=None,
              dist_tol=None, verbose=False):
    """Household problem + invariant distribution at one price vector"""
    v, pol, n_vfi = solve_household(cal, num, grids, prices, fr,
                                    v_init=v_init, verbose=verbose)
    fine = refine(v, pol, grids)
    op = build_operator(fine, fine, fr, grids)
    pop, n_dist = stationary(op, grids, num, pop0=pop0, tol=dist_tol,
                             verbose=verbose)
    if verbose:
        print(f"    household solved in {n_vfi} iterations; "
              f"distribution in {n_dist} iterations", flush=True)
    return v, pol, fine, op, pop


def solve_steady_state(cal: Calibration, num: Numerics, *,
                       prices: Prices | None = None,
                       verbose: bool = False) -> SteadyState:
    """Compute the stationary equilibrium of the model.

    With ``prices`` supplied, solves once at those constants.  Otherwise runs
    the background-GE calibration: iterate on (K/L, average earnings, T) with
    damping until the household-implied values reproduce the assumed ones.
    Iterations start from the previous solution, which does not affect the 
    fixed point.
    """
    grids = build_grids(cal, num)
    fr = Frictions.steady(cal)
    v_prev, pop_prev = None, None

    if prices is None:
        kl, avgw, T = num.kl_init, num.avgw_init, num.transfer_init
        for it in range(1, num.max_price_iter + 1):
            trial = Prices.from_kl(cal, kl, avgw, T)
            v, pol, fine, op, pop = _solve_at(
                cal, num, grids, fr, trial, v_init=v_prev, pop0=pop_prev,
                dist_tol=1e-8, verbose=False)
            v_prev, pop_prev = v, pop
            agg = household_aggregates(pop, grids, trial, cal)

            # Damped update toward the household-implied aggregates.
            kl = num.damp_kl * agg["KL"] + (1.0 - num.damp_kl) * kl
            avgw = num.damp_w * agg["avg_earnings"] + (1.0 - num.damp_w) * avgw
            T = (num.damp_transfer * agg["transfer_balancing"]
                 + (1.0 - num.damp_transfer) * T)
            err = (abs(kl - agg["KL"]), abs(avgw - agg["avg_earnings"]),
                   abs(T - agg["transfer_balancing"]))
            if verbose:
                print(f"  price iteration {it:2d}:  K/L={kl:9.4f}  "
                      f"avg earnings={avgw:.4f}  T={T:.4f}  "
                      f"errors=({err[0]:.1e}, {err[1]:.1e}, {err[2]:.1e})",
                      flush=True)
            if err[0] < num.tol_kl and err[1] < num.tol_avgw and err[2] < num.tol_transfer:
                break
        prices = Prices.from_kl(cal, kl, avgw, T)

    # Lastly, solve at the (calibrated or supplied) values
    v, pol, fine, op, pop = _solve_at(cal, num, grids, fr, prices,
                                      v_init=v_prev, pop0=pop_prev,
                                      verbose=verbose)
    _, flows = step(pop, op, count_flows=True)
    agg = household_aggregates(pop, grids, prices, cal)
    return SteadyState(prices=prices, values=v, policies=pol, fine=fine,
                       population=pop, flows=flows, aggregates=agg,
                       grids=grids, operator=op)


# --------------------------------------------------------------------------- #
# Gross flows by wealth (Table 6)                                              #
# --------------------------------------------------------------------------- #
def wealth_quintile_bounds(pop: Population, grids: Grids) -> list[int]:
    """Fine-grid indices splitting the population into wealth quintiles"""
    cum = np.cumsum(pop.asset_marginal())
    cuts = [int(np.searchsorted(cum, c)) for c in (0.2, 0.4, 0.6, 0.8)]
    return [0] + cuts + [grids.a_fine.size]


def flows_by_wealth(ss: SteadyState) -> dict[str, list[float]]:
    """Transition rates within each wealth quintile, relative to the aggregate.

    The flow accounting is re-run on each quintile's sub-population (without
    renormalising), each rate divided by the quintile's own source stock, and
    reported relative to the economy-wide rate as in the paper's Table 6.
    """
    bounds = wealth_quintile_bounds(ss.population, ss.grids)
    rel: dict[str, list[float]] = {}
    for k in range(5):
        sub = ss.population.restrict_assets(bounds[k], bounds[k + 1])
        _, fl = step(sub, ss.operator, count_flows=True, renormalize=False)
        for key in ("EU", "EN", "UE", "UN", "NE", "NU", "EE", "UU", "NN", "JJ"):
            rel.setdefault(key, []).append(fl[key] / ss.flows[key])
    return rel
