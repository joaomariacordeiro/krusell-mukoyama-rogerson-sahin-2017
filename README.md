# Replicating Krusell, Mukoyama, Rogerson & Şahin (2017): Gross Worker Flows over the Business Cycle

## Overview

This repository contains an independent Python replication of the model in:

> Krusell, P., Mukoyama, T., Rogerson, R. and Şahin, A. (2017). "Gross Worker
> Flows over the Business Cycle". *American Economic Review*, 107(11),
> 3447–3476.


**Scope.** This replicates the **model only**. The empirical work of the
original paper (CPS gross-flow construction, Abowd–Zellner and deNUNification
corrections, seasonal adjustment, SIPP processing) is **not** replicated.
Calibration targets and parameter values are taken from the paper rather than
recomputed from microdata.

**Author:** João Maria Cordeiro  
**Date:** March 2026  
**Contact:** joaomariacordeiro@gmail.com

## Disclaimer

Any errors are my own.

## Data and calibration availability

| Data | Source | Availability |
|---|---|---|
| Model parameters (Table 4 of the paper) | Krusell et al. (2017) | Embedded in `parameters.py` |
| Business-cycle shock process (ε^λ, ε^σ, ρ) | Krusell et al. (2017), Section II.B | Embedded in `parameters.py` |
| Converged background-GE price constants | Authors' AER replication package | Embedded in `parameters.py` (used only by `--quick`) |
| Authors' published model output (`Logfile.txt`, `stats_M.txt`) | Authors' AER replication package | Included in `refs/` as validation fixtures |
| CPS/SIPP microdata and gross-flow construction | Krusell et al. (2017) | **Not replicated here**; data columns in the write-up are the published values |

No external data are downloaded or required to run the model.

## Computational Requirements

### Software

| Software | Tested version | Required |
|---|---|---|
| Python | 3.13.5 | ≥ 3.11 |
| numpy | 2.3.2 | yes |
| scipy | 1.17.1 | yes |
| pandas | 2.3.1 | yes |
| matplotlib | 3.10.7 | yes |

### Hardware

| | |
|---|---|
| CPU | Intel Core 7 240H |
| RAM | 31.5 GB |
| OS | Microsoft Windows 11 |

### Expected Runtime

Expect roughly 12 to 14 minutes.
`main.py` reports only total runtime, not per-phase timings.

## Setup Instructions

1. Download the repository folder.
2. Install the dependencies:

   ```
   py -m pip install numpy scipy pandas matplotlib
   ```

3. Verify the installation:

   ```
   py -c "import numpy, scipy, pandas, matplotlib; print('ok')"
   ```

4. Run the replication:

   ```
   py main.py
   ```

   Adding `--quick` skips the price-calibration loop and solves at the
   authors' converged prices; it produces the same exhibits.

Repository layout:

```
Replication/
├── main.py                          driver: run, compare, validate, save
├── parameters.py                    calibration, frictions, shocks, numerics
├── discretize.py                    grids and finite-state approximations
├── household.py                     household problem (VFI)
├── cross_section.py                 population distribution dynamics
├── equilibrium.py                   steady state and price calibration
├── cycle.py                         business-cycle solution and simulation
├── moments.py                       HP filter, cyclical statistics
├── refs/
│   ├── Logfile.txt                  authors' published steady-state output
│   └── stats_M.txt                  authors' published business-cycle output
├── outputs/                         generated: tables (CSV), monthly series
├── figures/                         generated: flows_combined.pdf
└── README.md                        this file
```

## Code Structure

| File | Description | Key functions |
|---|---|---|
| `parameters.py` | All calibrated parameters (Table 4), friction bundle, shock process, numerical settings | `Calibration`, `Frictions`, `BusinessCycle`, `Numerics` |
| `discretize.py` | Asset grids; Tauchen chain for z; bins for match quality; monthly conversion of the annual wage process | `build_grids`, `tauchen_matrix`, `monthly_innovation_sd` |
| `household.py` | The five-value-function household problem, solved by VFI with golden-section savings and Howard acceleration | `solve_household`, `continuation_values`, `maximize_savings`, `Prices` |
| `cross_section.py` | Population distribution and its one-period law of motion; gross-flow accounting | `build_operator`, `step`, `stationary`, `Population` |
| `equilibrium.py` | Steady state at given prices; damped background-GE price calibration; flows by wealth quintile | `solve_steady_state`, `household_aggregates`, `flows_by_wealth` |
| `cycle.py` | Value functions over the two aggregate states; shock path; deterministic simulation | `solve_cycle`, `simulate`, `shock_path`, `state_frictions` |
| `moments.py` | Quarterly aggregation, HP(1600) filter, cyclical moments, variance decomposition | `cycle_statistics`, `variance_decomposition`, `hp_trend` |
| `main.py` | Orchestrates the pipeline, prints comparisons with the paper, validates against `refs/`, writes outputs | `main`, `make_flow_figure`, `parse_logfile` |

Dependency graph (A ← B means B imports A):

```
parameters ← discretize ← household ← cross_section ← equilibrium ─┐
                                                     ← cycle ──────┼── main
                                          moments (standalone) ────┘
```

`parameters.py` and `moments.py` are standalone (no internal imports;
`moments.py` uses only numpy/scipy). `discretize.py` imports `parameters`;
`household.py` imports both; `cross_section.py` builds on `household`;
`equilibrium.py` and `cycle.py` build on all of the above; `main.py` imports
`equilibrium`, `cycle`, `moments`, `household` and `parameters`.

## Output

Key results, replication versus paper (replication numbers from the full run;
paper values from Krusell et al. 2017, Tables 5, 7, 8C, 9 and 11):

| Exhibit | Replication | Paper |
|---|---|---|
| Unemployment rate (steady state) | 0.0678 | 0.068 |
| Participation rate (steady state) | 0.662 | 0.66 |
| E→U flow rate | 0.0142 | 0.014 |
| U→E flow rate | 0.2187 | 0.219 |
| U→N flow rate | 0.1296 | 0.130 |
| N→E flow rate | 0.0221 | 0.022 |
| Average job-to-job wage gain | 0.0320 | 0.033 (target) |
| std(u), HP-filtered | 0.128 | 0.121 |
| corr(lfpr, Y) | 0.393 | 0.37 |
| corr(f_EU, Y) | −0.794 | −0.79 |
| corr(f_UN, Y) | 0.478 | 0.47 |
| corr(f_NU, Y) | −0.962 | −0.96 |
| Job-to-job rate: std, corr(·,Y) | 0.103, 0.554 | 0.098, 0.54 |
| Variance decomposition (U&E / U&N / E&N) | 84.9 / 13.2 / 1.6 | 74.1 / 31.1 / −3.8 |

The business-cycle standard deviations sit uniformly about 5% above the
paper's, and the variance decomposition is reproduced only qualitatively (see
Methodological Notes below).

Figures:

| File | Description | Paper reference |
|---|---|---|
| `figures/flows_combined.pdf` | Gross-flow cyclicality (top) and volatility (bottom), replication vs paper | Table 8, panel C |

## Methodological Notes

### Departures from the original

| Component | This replication | Krusell et al. (2017) |
|---|---|---|
| Language | Python (numpy/scipy) | Fortran + MATLAB statistics |
| Household problem | VFI, golden-section savings, Howard acceleration | VFI, golden-section savings |
| Search-cost state | analytic: savings policies are γ-free, U(a,z,γ)=Û(a,z)−γ | explicit γ dimension |
| Distribution | Young (2010) operator with pre-computed choice coefficients | Young (2010), explicit loops |
| Shock discretisation | Tauchen n_z=20, n_q=7, n_γ=3 | identical |
| Asset grids | 48 (log-spaced) solve / 1000 (linear) distribution | identical |
| Price calibration | damped fixed point on K/L, average earnings, T | identical procedure |
| Business-cycle simulation | deterministic distribution along one shock path | identical |
| Shock-path random numbers | numpy PCG64, seed 1 | IMSL generator, seed 1 |
| Statistics | in-house HP(1600), quarterly aggregation | `hpfilter.m`, quarterly |

The Howard acceleration and the two search-cost simplifications change
iteration counts, not the fixed point; at the authors' converged prices the
ten steady-state flows match their published output to ~1e-6 (`--quick`
mode; ~1e-5 when the replication calibrates its own prices).

### Discrepancies

1. **Business-cycle volatilities ~5% high.** The aggregate shock path is
   drawn with a different random-number generator (same seed, different
   algorithm), so the two simulations average different 5,000-month samples.
   Correlations and autocorrelations, which depend far less on the particular
   path, match to ~0.02, and the gap is uniform across series.
2. **Variance decomposition (Table 11): 84.9/13.2/1.6 vs 74.1/31.1/−3.8.**
   The exact Elsby–Hobijn–Şahin implementation used by the authors lives in
   an external spreadsheet, not in their model code; the counterfactual-
   covariance method used here reproduces the qualitative structure (flows
   involving U dominate jointly, E&N negligible) but splits U&E vs U&N
   differently.
3. **λ_u = 0.282 vs 0.278 printed in the paper's Table 4.** The authors'
   calibration (and their derived λ_e = 0.428·λ_u, λ_n = 0.645·λ_u) uses
   0.282; this replication follows the calibration, and notes the
   discrepancy in the write-up.

## References

- Elsby, M. W. L., Hobijn, B. and Şahin, A. (2015). "On the Importance of the
  Participation Margin for Labor Market Fluctuations". *Journal of Monetary
  Economics*, 72, 64–82.
- Goensch, J. (2025). "Lecture 1: Dynamic Programming: Certainty".
  Quantitative Macroeconomics and Numerical Methods, Goethe University
  Frankfurt.
- Krusell, P., Mukoyama, T., Rogerson, R. and Şahin, A. (2017). "Gross Worker
  Flows over the Business Cycle". *American Economic Review*, 107(11),
  3447–3476.
- Tauchen, G. (1986). "Finite State Markov-Chain Approximations to Univariate
  and Vector Autoregressions". *Economics Letters*, 20(2), 177–181.
- Young, E. R. (2010). "Solving the Incomplete Markets Model with Aggregate
  Uncertainty Using the Krusell–Smith Algorithm and Non-Stochastic
  Simulations". *Journal of Economic Dynamics and Control*, 34(1), 36–41.

## Licence

The code is released under the MIT licence.
