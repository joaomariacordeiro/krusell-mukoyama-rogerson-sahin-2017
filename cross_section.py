"""Cross-sectional dynamics: how the population distribution evolves over time.

The population is described by the mass of households at each combination of
assets and productivity, separately for each labour-market state, on a fine
and evenly spaced asset grid:

    W (a, z, q)   employed at match quality q
    UE(a, z)      active search, UI-eligible
    UN(a, z)      active search, not eligible
    OE(a, z)      passive, UI-eligible
    ON(a, z)      passive, not eligible

The search cost gamma is drawn every period at the moment the
labour-market choice is made, so last period's draw carries no information
about this period's.  The non-employed masses therefore need only (a, z).
The new gamma draw is averaged out inside the choice coefficients below.

Advancing the distribution by one period follows the timing of the model
(Section I.B of the paper) in three stages:

1.  Assets and productivity:  each household's mass moves to its chosen
    savings level (split between the two neighbouring grid points, so that
    average assets are preserved) and its productivity z evolves according
    to the Tauchen transition matrix.
    
2.  Events:  chance then divides each household's mass across the possible
    events of the period.  An employed household keeps its job, keeps it and
    receives an outside offer, separates and immediately draws a new offer,
    or separates with no offer, with probabilities (1-sigma-lambda_e,
    lambda_e, sigma*lambda_s, sigma*(1-lambda_s)).  A non-employed household
    receives an offer or not, at the arrival rate of its state, and UI
    eligibility expires with probability mu for the eligible.  Each possible
    outcome is a branch of the period.
    
3.  Choices:  within each branch the household picks the best of working,
    searching actively, or staying out, comparing the relevant value
    functions under a fresh gamma draw (and the drawn match quality q' where
    an offer arrived).  A separated worker chooses among UI-eligible values.
    One who quits a surviving job does not, because quitting forfeits UI
    (Section I.C).

Because the value functions stay fixed while the distribution evolves, the
stage-3 choices never change.  They are therefore computed once, averaged
over the possible gamma and q' draws, and stored as small coefficient arrays.
Advancing the distribution one period then reduces to a few array
multiplications.  Gross flows are counted in the same pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from discretize import Grids
from household import Policies, Values
from parameters import Frictions, Numerics

_FLOW_KEYS = ("EE", "JJ", "EU", "EN", "UE", "UU", "UN", "NE", "NU", "NN")


# --------------------------------------------------------------------------- #
# Population                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Population:
    """Mass by labour-market state on the fine asset grid (sums to one)"""

    W: NDArray    # (A, Z, Q)
    UE: NDArray   # (A, Z)
    UN: NDArray   # (A, Z)
    OE: NDArray   # (A, Z)
    ON: NDArray   # (A, Z)

    @property
    def employed(self) -> float:
        return float(self.W.sum())

    @property
    def unemployed(self) -> float:
        return float(self.UE.sum() + self.UN.sum())

    @property
    def out_of_lf(self) -> float:
        return float(self.OE.sum() + self.ON.sum())

    def total(self) -> float:
        return self.employed + self.unemployed + self.out_of_lf

    def asset_marginal(self) -> NDArray:
        """Total mass at each fine asset node (for the wealth quantiles)."""
        return (self.W.sum(axis=(1, 2)) + self.UE.sum(axis=1) + self.UN.sum(axis=1)
                + self.OE.sum(axis=1) + self.ON.sum(axis=1))

    def restrict_assets(self, lo: int, hi: int) -> "Population":
        """Sub-population with fine-grid asset index in [lo, hi)"""
        m = np.zeros(self.W.shape[0])
        m[lo:hi] = 1.0
        return Population(self.W * m[:, None, None], self.UE * m[:, None],
                          self.UN * m[:, None], self.OE * m[:, None],
                          self.ON * m[:, None])


def uniform_seed(grids: Grids) -> Population:
    """A full-support starting distribution.

    The invariant distribution is unique, so the seed affects only the number
    of iterations to convergence: start everyone at a mid-grid asset node,
    spread over z, 60% employed / 10% active / 30% passive.
    """
    A, Z, Q = grids.a_fine.size, grids.z.size, grids.q.size
    ia = A // 2
    wz = np.full(Z, 1.0 / Z)
    W = np.zeros((A, Z, Q)); W[ia] = 0.6 * wz[:, None] / Q
    UE = np.zeros((A, Z)); UE[ia] = 0.1 * wz
    UN = np.zeros((A, Z))
    OE = np.zeros((A, Z)); OE[ia] = 0.1 * wz
    ON = np.zeros((A, Z)); ON[ia] = 0.2 * wz
    return Population(W, UE, UN, OE, ON)


# --------------------------------------------------------------------------- #
# Solution objects on the fine grid                                            #
# --------------------------------------------------------------------------- #
def _to_fine(a_coarse: NDArray, a_fine: NDArray, arr: NDArray) -> NDArray:
    """Interpolate an array (n_coarse, *T) linearly onto the fine asset grid."""
    i = np.clip(np.searchsorted(a_coarse, a_fine, side="right") - 1,
                0, a_coarse.size - 2)
    t = (a_fine - a_coarse[i]) / (a_coarse[i + 1] - a_coarse[i])
    shape = (a_fine.size,) + (1,) * (arr.ndim - 1)
    return (1.0 - t).reshape(shape) * arr[i] + t.reshape(shape) * arr[i + 1]


@dataclass
class FineSolution:
    """Values and savings policies interpolated onto the fine asset grid."""

    W: NDArray    # (A, Z, Q)   values
    UE: NDArray   # (A, Z, G)
    UN: NDArray   # (A, Z, G)
    OE: NDArray   # (A, Z)
    ON: NDArray   # (A, Z)
    aW: NDArray   # (A, Z, Q)   savings policies
    aUE: NDArray  # (A, Z)
    aUN: NDArray  # (A, Z)
    aOE: NDArray  # (A, Z)
    aON: NDArray  # (A, Z)


def refine(v: Values, pol: Policies, grids: Grids) -> FineSolution:
    a, af = grids.a, grids.a_fine
    return FineSolution(
        W=_to_fine(a, af, v.W), UE=_to_fine(a, af, v.UE), UN=_to_fine(a, af, v.UN),
        OE=_to_fine(a, af, v.OE), ON=_to_fine(a, af, v.ON),
        aW=_to_fine(a, af, pol.W), aUE=_to_fine(a, af, pol.UE),
        aUN=_to_fine(a, af, pol.UN), aOE=_to_fine(a, af, pol.OE),
        aON=_to_fine(a, af, pol.ON),
    )


# --------------------------------------------------------------------------- #
# Stage-3 choice coefficients (averaged over the q' and gamma draws)           #
# --------------------------------------------------------------------------- #
# All comparisons are strict: taking a job requires W > max(U, O), and active
# search requires U > O.  A tie therefore resolves to the more passive state.

@dataclass
class OfferRouting:
    """Jobless-with-fresh-offer channel: work at drawn q' / search / stay out.

    Coefficients are per unit of channel mass at (a, z):
    ``to_E[a,z,p]`` mass accepting an offer of quality p, ``to_U``/``to_O``
    mass choosing active/passive search.
    """

    to_E: NDArray   # (A, Z, P)
    to_U: NDArray   # (A, Z)
    to_O: NDArray   # (A, Z)


@dataclass
class JoblessRouting:
    """No-offer channel: active vs passive search only."""

    to_U: NDArray   # (A, Z)
    to_O: NDArray   # (A, Z)


@dataclass
class KeepRouting:
    """Employed, no outside offer: keep the job at unchanged q, or quit."""

    stay: NDArray   # (A, Z, Q)
    to_U: NDArray   # (A, Z, Q)
    to_O: NDArray   # (A, Z, Q)


@dataclass
class LadderRouting:
    """Employed with an outside offer: keep max{q, q'} or quit (Section I.B).

    ``switch[a,z,q,p]`` is the mass moving to the outside job p (a job-to-job
    transition); ``gain`` pre-integrates the percentage wage gain (q_p-q_q)/q_q
    over accepted switches, the object behind the 3.3% calibration target.
    """

    switch: NDArray  # (A, Z, Q, P)
    keep: NDArray    # (A, Z, Q)
    to_U: NDArray    # (A, Z, Q)
    to_O: NDArray    # (A, Z, Q)
    gain: NDArray    # (A, Z, Q)


def _offer_routing(Wf, Uf, Of, w_q, w_g) -> OfferRouting:
    UO = np.maximum(Uf, Of[:, :, None])                          # (A,Z,G)
    take = Wf[:, :, :, None] > UO[:, :, None, :]                 # (A,Z,P,G)
    active = (~take) & (Uf[:, :, None, :] > Of[:, :, None, None])
    passive = (~take) & ~(Uf[:, :, None, :] > Of[:, :, None, None])
    return OfferRouting(
        to_E=np.einsum("azpg,g->azp", take, w_g) * w_q[None, None, :],
        to_U=np.einsum("azpg,p,g->az", active, w_q, w_g),
        to_O=np.einsum("azpg,p,g->az", passive, w_q, w_g),
    )


def _jobless_routing(Uf, Of, w_g) -> JoblessRouting:
    active = Uf > Of[:, :, None]                                 # (A,Z,G)
    to_U = np.einsum("azg,g->az", active, w_g)
    return JoblessRouting(to_U=to_U, to_O=1.0 - to_U)


def _keep_routing(Wf, Uf, Of, w_g) -> KeepRouting:
    UO = np.maximum(Uf, Of[:, :, None])                          # (A,Z,G)
    stay = Wf[:, :, :, None] > UO[:, :, None, :]                 # (A,Z,Q,G)
    active = (~stay) & (Uf[:, :, None, :] > Of[:, :, None, None])
    passive = (~stay) & ~(Uf[:, :, None, :] > Of[:, :, None, None])
    return KeepRouting(
        stay=np.einsum("azqg,g->azq", stay, w_g),
        to_U=np.einsum("azqg,g->azq", active, w_g),
        to_O=np.einsum("azqg,g->azq", passive, w_g),
    )


def _ladder_routing(Wf, Uf, Of, w_q, w_g, q_levels) -> LadderRouting:
    UO = np.maximum(Uf, Of[:, :, None])                          # (A,Z,G)
    Wq = Wf[:, :, :, None, None]                                 # current q
    Wp = Wf[:, :, None, :, None]                                 # outside q'
    UOx = UO[:, :, None, None, :]
    switch = Wp > np.maximum(Wq, UOx)                            # (A,Z,Q,P,G)
    keep = (~switch) & (Wq > UOx)
    quit_ = (~switch) & ~(Wq > UOx)
    active = quit_ & (Uf[:, :, None, None, :] > Of[:, :, None, None, None])
    passive = quit_ & ~(Uf[:, :, None, None, :] > Of[:, :, None, None, None])
    pct_gain = (q_levels[None, :] - q_levels[:, None]) / q_levels[:, None]  # (Q,P)
    return LadderRouting(
        switch=np.einsum("azqpg,g->azqp", switch, w_g) * w_q[None, None, None, :],
        keep=np.einsum("azqpg,p,g->azq", keep, w_q, w_g),
        to_U=np.einsum("azqpg,p,g->azq", active, w_q, w_g),
        to_O=np.einsum("azqpg,p,g->azq", passive, w_q, w_g),
        gain=np.einsum("azqpg,qp,p,g->azq", switch, pct_gain, w_q, w_g),
    )


# --------------------------------------------------------------------------- #
# The one-period operator                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Operator:
    """Everything needed to advance the distribution by one period.

    The savings moves come from one solution and the event/choice
    coefficients from another.  In the steady state the two coincide.  Over
    the business cycle the savings policy belongs to the current aggregate
    state, while the events and choices occur under next period's state,
    matching the timing of the model (Section II.B).
    """

    idx: dict         #: bracketing fine-grid index of each savings choice
    frac_hi: dict     #: mass share assigned to the upper bracketing node
    Pi_z: NDArray
    fr: Frictions
    offer_e: OfferRouting
    offer_n: OfferRouting
    jobless_e: JoblessRouting
    jobless_n: JoblessRouting
    keep_n: KeepRouting
    ladder_n: LadderRouting


def build_operator(save: FineSolution, choose: FineSolution, fr: Frictions,
                   grids: Grids) -> Operator:
    """Assemble the one-period operator.

    ``save`` supplies the savings policies (where mass moves), ``choose`` the
    value functions that decide the labour-market choices.  Pass the same
    solution twice for the steady state.
    """
    af, w_q, w_g = grids.a_fine, grids.w_q, grids.w_gamma
    idx, frac = {}, {}
    for name, pol in (("W", save.aW), ("UE", save.aUE), ("UN", save.aUN),
                      ("OE", save.aOE), ("ON", save.aON)):
        i = np.clip(np.searchsorted(af, pol, side="right") - 1, 0, af.size - 2)
        idx[name] = i
        frac[name] = (pol - af[i]) / (af[i + 1] - af[i])
    return Operator(
        idx=idx, frac_hi=frac, Pi_z=grids.Pi_z, fr=fr,
        # Eligible choices compare against (UE, OE); non-eligible against (UN, ON).
        offer_e=_offer_routing(choose.W, choose.UE, choose.OE, w_q, w_g),
        offer_n=_offer_routing(choose.W, choose.UN, choose.ON, w_q, w_g),
        jobless_e=_jobless_routing(choose.UE, choose.OE, w_g),
        jobless_n=_jobless_routing(choose.UN, choose.ON, w_g),
        keep_n=_keep_routing(choose.W, choose.UN, choose.ON, w_g),
        ladder_n=_ladder_routing(choose.W, choose.UN, choose.ON, w_q, w_g, grids.q),
    )


def _push(mass: NDArray, idx: NDArray, frac_hi: NDArray, Pi_z: NDArray) -> NDArray:
    """Stage 1: move each mass point to its chosen savings level, splitting it
    between the two neighbouring grid points, then let z evolve by the
    Tauchen transition matrix."""
    A = mass.shape[0]
    rest = int(np.prod(mass.shape[1:]))
    m = mass.reshape(A, rest)
    i = idx.reshape(A, rest)
    t = frac_hi.reshape(A, rest)
    cols = np.arange(rest)
    scattered = (
        np.bincount((i * rest + cols).ravel(), weights=(m * (1.0 - t)).ravel(),
                    minlength=A * rest)
        + np.bincount(((i + 1) * rest + cols).ravel(), weights=(m * t).ravel(),
                      minlength=A * rest)
    ).reshape(mass.shape)
    if mass.ndim == 2:
        return scattered @ Pi_z
    return np.einsum("azq,zy->ayq", scattered, Pi_z)


def step(pop: Population, op: Operator, *, count_flows: bool = False,
         renormalize: bool = True):
    """Advance the population one period

    Returns ``(next_population, flows)``; ``flows`` is None unless requested,
    otherwise a dict of transition *rates* (each destination mass divided by
    the source stock at the start of the period), the stocks, and the average
    job-to-job wage gain.  ``EE`` counts stayers only; ``JJ`` counts all job
    switches (ladder moves and separation-with-immediate-rehire), so the
    E-to-E entry of the paper's Table 5 is EE + JJ.
    """
    fr = op.fr
    lu, ln, le, ls, sig, mu = (fr.lambda_u, fr.lambda_n, fr.lambda_e,
                               fr.lambda_s, fr.sigma, fr.mu)

    # Stage 1: savings and productivity transition, one array per labour state.
    pW = _push(pop.W, op.idx["W"], op.frac_hi["W"], op.Pi_z)      # (A,Z,Q)
    pUE = _push(pop.UE, op.idx["UE"], op.frac_hi["UE"], op.Pi_z)  # (A,Z)
    pUN = _push(pop.UN, op.idx["UN"], op.frac_hi["UN"], op.Pi_z)
    pOE = _push(pop.OE, op.idx["OE"], op.frac_hi["OE"], op.Pi_z)
    pON = _push(pop.ON, op.idx["ON"], op.frac_hi["ON"], op.Pi_z)
    pWq = pW.sum(axis=2)

    A, Z = pUE.shape
    Q = pW.shape[2]
    nW = np.zeros((A, Z, Q))
    nUE = np.zeros((A, Z)); nUN = np.zeros((A, Z))
    nOE = np.zeros((A, Z)); nON = np.zeros((A, Z))
    fl = dict.fromkeys(_FLOW_KEYS, 0.0)
    gain_total = 0.0

    def route_offer(mass, rt: OfferRouting, dU, dO, kE, kU, kO):
        """Offer-in-hand channel: mass (A,Z) -> employment / search / out"""
        nonlocal nW
        nW += mass[:, :, None] * rt.to_E
        dU += mass * rt.to_U
        dO += mass * rt.to_O
        if count_flows:
            fl[kE] += float((mass * rt.to_E.sum(axis=2)).sum())
            fl[kU] += float((mass * rt.to_U).sum())
            fl[kO] += float((mass * rt.to_O).sum())

    def route_jobless(mass, rt: JoblessRouting, dU, dO, kU, kO):
        dU += mass * rt.to_U
        dO += mass * rt.to_O
        if count_flows:
            fl[kU] += float((mass * rt.to_U).sum())
            fl[kO] += float((mass * rt.to_O).sum())

    # --- Employed: four mutually exclusive events (Section I.B) ----------- #
    # Survivors decide under non-eligible values (quitting forfeits UI).
    # The separated decide under eligible values.
    m = (1.0 - sig - le) * pW                       # keep job, no outside offer
    kn = op.keep_n
    nW += m * kn.stay
    nUN += (m * kn.to_U).sum(axis=2)
    nON += (m * kn.to_O).sum(axis=2)
    if count_flows:
        fl["EE"] += float((m * kn.stay).sum())
        fl["EU"] += float((m * kn.to_U).sum())
        fl["EN"] += float((m * kn.to_O).sum())

    m = le * pW                                     # keep job + outside offer
    lad = op.ladder_n
    nW += np.einsum("azq,azqp->azp", m, lad.switch) + m * lad.keep
    nUN += (m * lad.to_U).sum(axis=2)
    nON += (m * lad.to_O).sum(axis=2)
    if count_flows:
        fl["JJ"] += float((m * lad.switch.sum(axis=3)).sum())
        fl["EE"] += float((m * lad.keep).sum())
        fl["EU"] += float((m * lad.to_U).sum())
        fl["EN"] += float((m * lad.to_O).sum())
        gain_total += float((m * lad.gain).sum())

    route_offer(sig * ls * pWq, op.offer_e, nUE, nOE, "JJ", "EU", "EN")
    route_jobless(sig * (1.0 - ls) * pWq, op.jobless_e, nUE, nOE, "EU", "EN")

    # --- Active searchers -------------------------------------------------- #
    # Eligible: keep eligibility w.p. 1-mu; lose it w.p. mu (Section I.C).
    route_offer((1.0 - mu) * lu * pUE, op.offer_e, nUE, nOE, "UE", "UU", "UN")
    route_jobless((1.0 - mu) * (1.0 - lu) * pUE, op.jobless_e, nUE, nOE, "UU", "UN")
    route_offer(mu * lu * pUE, op.offer_n, nUN, nON, "UE", "UU", "UN")
    route_jobless(mu * (1.0 - lu) * pUE, op.jobless_n, nUN, nON, "UU", "UN")
    route_offer(lu * pUN, op.offer_n, nUN, nON, "UE", "UU", "UN")
    route_jobless((1.0 - lu) * pUN, op.jobless_n, nUN, nON, "UU", "UN")

    # --- Passive searchers (offers arrive at the lower rate lambda_n) ------ #
    route_offer((1.0 - mu) * ln * pOE, op.offer_e, nUE, nOE, "NE", "NU", "NN")
    route_jobless((1.0 - mu) * (1.0 - ln) * pOE, op.jobless_e, nUE, nOE, "NU", "NN")
    route_offer(mu * ln * pOE, op.offer_n, nUN, nON, "NE", "NU", "NN")
    route_jobless(mu * (1.0 - ln) * pOE, op.jobless_n, nUN, nON, "NU", "NN")
    route_offer(ln * pON, op.offer_n, nUN, nON, "NE", "NU", "NN")
    route_jobless((1.0 - ln) * pON, op.jobless_n, nUN, nON, "NU", "NN")

    nxt = Population(nW, nUE, nUN, nOE, nON)
    if renormalize:
        s = nxt.total()
        nxt = Population(nW / s, nUE / s, nUN / s, nOE / s, nON / s)

    flows = None
    if count_flows:
        E, U, N = pop.employed, pop.unemployed, pop.out_of_lf
        flows = {k: fl[k] / E for k in ("EE", "JJ", "EU", "EN")}
        flows |= {k: fl[k] / U for k in ("UE", "UU", "UN")}
        flows |= {k: fl[k] / N for k in ("NE", "NU", "NN")}
        flows["wage_gain"] = gain_total / fl["JJ"] if fl["JJ"] > 0 else np.nan
        flows |= {"E": E, "U": U, "N": N,
                  "urate": U / (E + U), "lfpr": E + U}
    return nxt, flows


def stationary(op: Operator, grids: Grids, num: Numerics, *,
               pop0: Population | None = None, tol: float | None = None,
               verbose: bool = False) -> tuple[Population, int]:
    """Iterate the operator to its fixed point (the invariant distribution)"""
    tol = num.tol_dist if tol is None else tol
    pop = pop0 if pop0 is not None else uniform_seed(grids)
    it = 0
    for it in range(1, num.max_dist_iter + 1):
        nxt, _ = step(pop, op)
        err = max(float(np.max(np.abs(nxt.W - pop.W))),
                  float(np.max(np.abs(nxt.UE - pop.UE))),
                  float(np.max(np.abs(nxt.UN - pop.UN))),
                  float(np.max(np.abs(nxt.OE - pop.OE))),
                  float(np.max(np.abs(nxt.ON - pop.ON))))
        pop = nxt
        if verbose and it % 250 == 0:
            print(f"    distribution iter {it:5d}  err={err:.2e}  "
                  f"E={pop.employed:.4f} U={pop.unemployed:.4f}", flush=True)
        if err < tol:
            break
    return pop, it
