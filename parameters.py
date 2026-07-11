"""Parameters of the Krusell-Mukoyama-Rogerson-Sahin (2017) model.

All economic parameters follow Table 4 (p. 3461) and Section II of
"Gross Worker Flows over the Business Cycle", AER 107(11).  The model period
is one month.  This is a model-only replication: calibrated values and targets
are taken from the paper, none of the CPS/SIPP data work is redone.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    """Economic parameters (Table 4; Section II.A)."""

    # --- Preferences (Section I.A) --------------------------------------- #
    beta: float = 0.99465        #: discount factor; consistent with 1+r = 1.00327
    alpha: float = 0.485         #: flow disutility of work (target: lfpr = 0.66)
    # Search cost gamma is i.i.d. on a symmetric three-point grid
    # {(1-s)*gbar, gbar, (1+s)*gbar} with equal weights (Section I.A).
    gamma_bar_ratio: float = 0.0875   #: gbar/alpha = 3.5h search / 40h work (ATUS)
    gamma_spread: float = 0.72        #: s; implies eps_gamma = 0.030 in Table 4

    # --- Idiosyncratic shocks (Section I.A, II.A) ------------------------- #
    rho_z_annual: float = 0.955       #: persistence of log z at annual frequency
    sigma_z_annual: float = 0.20      #: annual innovation s.d. of log z
    sigma_q: float = 0.034            #: s.d. of log match quality (J2J gain 0.033)

    # --- Labour-market frictions (Section I.B; Table 4) ------------------- #
    # Table 4 prints lambda_u = 0.278; the official calibration underlying the
    # published model results uses 0.282, and the tied rates below follow it.
    lambda_u: float = 0.282           #: offer arrival, active search (u = 0.068)
    lambda_e_ratio: float = 0.428     #: lambda_e / lambda_u (J2J rate = 0.022)
    lambda_n_ratio: float = 0.645     #: lambda_n / lambda_u (N->E rate = 0.022)
    sigma_sep: float = 0.0178         #: separation rate (E->U rate = 0.014)
    # A separating worker draws a fresh offer with probability lambda_s = lambda_u.

    # --- Unemployment insurance (Section I.C) ----------------------------- #
    mu: float = 1.0 / 6.0             #: monthly eligibility-expiry probability
    ui_replacement: float = 0.23      #: b0: benefit = b0 * w * z, capped
    ui_cap_ratio: float = 0.465       #: cap = 0.465 * (average earnings)

    # --- Taxes and background technology (Section II.A) ------------------- #
    tau: float = 0.30                 #: proportional labour-income tax
    capital_share: float = 0.30       #: Cobb-Douglas capital share
    depreciation: float = 0.0067      #: monthly depreciation (in r = MPK - delta)

    # Converged constants of the background-GE price calibration (Section
    # II.A; official package).  Used only to (i) skip the price loop in quick
    # mode and (ii) cross-check the loop's output.
    kl_star: float = 129.513889       #: capital per efficiency unit of labour
    avgw_star: float = 2.477383       #: average z*q among the employed
    transfer_star: float = 1.356605   #: lump-sum transfer T

    @property
    def gamma_bar(self) -> float:
        """Mean search cost, gbar = 0.0875 * alpha (Table 4: 0.042)."""
        return self.gamma_bar_ratio * self.alpha

    @property
    def lambda_e(self) -> float:
        return self.lambda_e_ratio * self.lambda_u

    @property
    def lambda_n(self) -> float:
        return self.lambda_n_ratio * self.lambda_u


@dataclass(frozen=True)
class Frictions:
    """Arrival and separation rates faced by households in one aggregate state.

    Bundled because the business cycle (Section II.B) moves exactly these
    objects while everything else stays fixed.
    """

    lambda_u: float   #: offer arrival under active search
    lambda_e: float   #: outside-offer arrival while employed
    lambda_n: float   #: offer arrival under passive search
    lambda_s: float   #: arrival for a just-separated worker (= lambda_u)
    sigma: float      #: exogenous separation probability
    mu: float         #: UI-eligibility expiry probability

    @classmethod
    def steady(cls, cal: Calibration) -> "Frictions":
        return cls(cal.lambda_u, cal.lambda_e, cal.lambda_n, cal.lambda_u,
                   cal.sigma_sep, cal.mu)


@dataclass(frozen=True)
class BusinessCycle:
    """Aggregate two-state Markov shock to frictions (Section II.B).

    Good state: lambda_u* + eps_lambda, sigma* - eps_sigma (and conversely in
    the bad state); lambda_e, lambda_n keep constant ratios to lambda_u.
    Prices (w, r) are constant over the cycle.  eps_lambda matches the
    volatility of f_UE, eps_sigma that of f_EU (AZ-adjusted data).
    """

    eps_lambda: float = 0.0662   #: amplitude of the lambda_u shock
    eps_sigma: float = 0.00239   #: amplitude of the sigma shock
    rho: float = 0.983           #: diagonal of the symmetric transition matrix
    n_months: int = 5000         #: simulated months
    burn_in: int = 1000          #: months discarded before computing statistics
    seed: int = 1                #: RNG seed for the aggregate shock path


@dataclass(frozen=True)
class Numerics:
    """Grid sizes, tolerances and iteration limits."""

    n_a: int = 48            #: asset nodes for the household problem (log-spaced)
    n_a_fine: int = 1000     #: asset nodes for the distribution (evenly spaced)
    n_z: int = 20            #: productivity nodes (Tauchen)
    n_q: int = 7             #: match-quality nodes
    n_gamma: int = 3         #: search-cost nodes
    a_max: float = 1440.0    #: upper bound of the asset grid (a >= 0 throughout)
    grid_width: float = 2.0  #: z and q grids span +/- this many s.d.

    tol_value: float = 1e-4    #: VFI convergence (sup norm, values and policies)
    tol_dist: float = 1e-9     #: stationary-distribution convergence (sup norm)
    tol_golden: float = 1e-6   #: golden-section bracket width
    tiny: float = 1e-10        #: consumption floor / numerical epsilon
    max_value_iter: int = 2000
    max_dist_iter: int = 10000
    max_price_iter: int = 100
    howard_steps: int = 30     #: policy-evaluation sweeps per optimisation sweep

    # Damping and tolerances of the background-GE price calibration
    # (Section II.A): K/L updates slowly, earnings and transfers faster.
    damp_kl: float = 0.1
    damp_w: float = 0.5
    damp_transfer: float = 0.5
    tol_kl: float = 0.01
    tol_avgw: float = 0.001
    tol_transfer: float = 1e-4
    kl_init: float = 129.2
    avgw_init: float = 2.5
    transfer_init: float = 1.35
