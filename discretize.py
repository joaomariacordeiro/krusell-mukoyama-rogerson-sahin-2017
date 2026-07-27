"""Discretisation of the model's state space

Builds the asset grids and finite-state approximations to the three
idiosyncratic shocks of Section I.A:

* persistent productivity ``log z`` -- AR(1), approximated by a Tauchen (1986)
  chain on an evenly spaced log grid;
* match quality ``log q`` -- i.i.d. normal, approximated by bin probabilities
  on an evenly spaced log grid;
* search cost ``gamma`` -- i.i.d. on a symmetric three-point grid with equal
  weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr  # standard normal CDF

from parameters import Calibration, Numerics


# --------------------------------------------------------------------------- #
# Annual -> monthly conversion of the productivity process (Section II.A)      #
# --------------------------------------------------------------------------- #
def monthly_persistence(rho_annual: float) -> float:
    """Monthly AR(1) persistence consistent with the annual estimate"""
    return rho_annual ** (1.0 / 12.0)


def monthly_innovation_sd(rho_annual: float, sigma_annual: float) -> float:
    """Monthly innovation s.d. consistent with the annual estimate.

    The paper's estimates (rho = 0.955, sigma = 0.20) refer to annual log
    wages, i.e. to the sum of twelve monthly log realisations.  A monthly
    innovation arriving in month ``i`` contributes ``sum_{k=0}^{12-i} rho^k``
    to that sum, so the annual innovation variance is the monthly variance
    times ``S = sum_{i=1}^{12} (sum_{k=0}^{12-i} rho^k)^2``.  Matching the
    annual s.d. gives ``sigma_month = 12 * sigma_annual / sqrt(S)`` (= 0.096
    in Table 4).
    """
    rho = monthly_persistence(rho_annual)
    S = sum(sum(rho ** k for k in range(12 - i + 1)) ** 2 for i in range(1, 13))
    return 12.0 * sigma_annual / math.sqrt(S)


# --------------------------------------------------------------------------- #
# Finite-state approximations                                                  #
# --------------------------------------------------------------------------- #
def tauchen_matrix(grid: NDArray, rho: float, sigma: float) -> NDArray:
    """Tauchen (1986) transition matrix on an evenly spaced grid.

    Row ``i`` holds the probabilities of the conditional normal
    ``N(rho * grid[i], sigma^2)`` over midpoint bins, with open-ended bins at
    the two extremes.  Rows sum to one.
    """
    h = grid[1] - grid[0]
    cond = rho * grid[:, None]                       # conditional means (n, 1)
    upper = ndtr((grid[None, :] + 0.5 * h - cond) / sigma)
    lower = ndtr((grid[None, :] - 0.5 * h - cond) / sigma)
    P = upper - lower
    P[:, 0] = upper[:, 0]                            # open lower tail
    P[:, -1] = 1.0 - lower[:, -1]                    # open upper tail
    return P


def normal_bin_weights(grid: NDArray, sigma: float) -> NDArray:
    """Probabilities of a mean-zero normal over midpoint bins of ``grid``"""
    h = grid[1] - grid[0]
    upper = ndtr((grid + 0.5 * h) / sigma)
    lower = ndtr((grid - 0.5 * h) / sigma)
    w = upper - lower
    w[0] = upper[0]
    w[-1] = 1.0 - lower[-1]
    return w


# --------------------------------------------------------------------------- #
# Grids                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grids:
    """All discretised state objects, built once and passed around"""

    a: NDArray          #: asset grid for the household problem (log-spaced)
    a_fine: NDArray     #: asset grid for the distribution (evenly spaced)
    z: NDArray          #: productivity levels, exp(log-grid)
    q: NDArray          #: match-quality levels, exp(log-grid)
    gamma: NDArray      #: search-cost grid (3 points)
    Pi_z: NDArray       #: (n_z, n_z) Tauchen transition matrix for log z
    w_q: NDArray        #: i.i.d. weights over match quality
    w_gamma: NDArray    #: i.i.d. weights over the search cost (uniform)
    rho_z: float        #: monthly persistence of log z
    sigma_z: float      #: monthly innovation s.d. of log z


def build_grids(cal: Calibration, num: Numerics) -> Grids:
    """Construct the full grid bundle from the calibration."""
    # Assets: the household problem uses a grid evenly spaced in log(a + 2),
    # concentrating nodes near the borrowing constraint a = 0 where the value
    # function has the most curvature. The distribution lives on an evenly
    # spaced grid, so splitting a mass point between the two grid points that
    # bracket its savings choice is a constant-step calculation.
    ratio = (num.a_max + 2.0) / 2.0
    steps = np.arange(num.n_a) / (num.n_a - 1)
    a = 2.0 * ratio ** steps - 2.0
    a[0], a[-1] = 0.0, num.a_max
    a_fine = np.linspace(0.0, num.a_max, num.n_a_fine)

    # Productivity: monthly AR(1) on +/- grid_width unconditional s.d.
    rho = monthly_persistence(cal.rho_z_annual)
    sig = monthly_innovation_sd(cal.rho_z_annual, cal.sigma_z_annual)
    half_z = num.grid_width * sig / math.sqrt(1.0 - rho * rho)
    log_z = np.linspace(-half_z, half_z, num.n_z)
    Pi_z = tauchen_matrix(log_z, rho, sig)

    # Match quality: i.i.d. lognormal on +/- grid_width * sigma_q.
    log_q = np.linspace(-num.grid_width * cal.sigma_q,
                        num.grid_width * cal.sigma_q, num.n_q)
    w_q = normal_bin_weights(log_q, cal.sigma_q)

    # Search cost: symmetric three-point grid, equal weights (Section I.A).
    gbar, s = cal.gamma_bar, cal.gamma_spread
    gamma = np.array([(1.0 - s) * gbar, gbar, (1.0 + s) * gbar])[: num.n_gamma]
    w_gamma = np.full(num.n_gamma, 1.0 / num.n_gamma)

    return Grids(a=a, a_fine=a_fine, z=np.exp(log_z), q=np.exp(log_q),
                 gamma=gamma, Pi_z=Pi_z, w_q=w_q, w_gamma=w_gamma,
                 rho_z=rho, sigma_z=sig)
