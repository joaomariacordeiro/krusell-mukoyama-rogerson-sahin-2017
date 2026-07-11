# KMRS (2017) — A Python Replication of the Model

An original Python implementation of the model in Krusell, Mukoyama, Rogerson
& Şahin (2017), *"Gross Worker Flows over the Business Cycle"*, American
Economic Review 107(11): 3447–3476 — written from the paper's equations, with
the official replication package (`../Replication_AER_2012_1662/`, read-only)
as the arbiter of implementation detail wherever the text is silent.

**Scope: model only.** The empirical work (CPS/SIPP gross-flow construction,
measurement-error corrections, seasonal adjustment) is not reproduced.
Calibrated parameter values and targets are taken from the paper (Table 4);
the data columns of its tables are the published values.

## What is reproduced

| Exhibit | Object | Status |
|---|---|---|
| Table 5 | average gross-flow matrix, steady state | matches published model output to ~3e-6 |
| Table 6 | flow rates by wealth quintile | matches to ~1e-5 |
| Table 7 | cyclical behaviour of stocks (u, lfpr, E) | signs and magnitudes; see COMPARISON.md |
| Table 8C | cyclical behaviour of the six gross flows | signs and magnitudes; see COMPARISON.md |
| Table 9 | job-to-job rate over the cycle | see COMPARISON.md |
| Table 11 | unemployment variance decomposition | qualitative only (see below) |

`refs/` contains the authors' published model output (`Logfile.txt`,
`stats_M.txt`), used as exact validation targets; `main.py` reports the
deviations at the end of a run.

## What is taken as given

- **Parameters** — Table 4 of the paper. One documented exception: Table 4
  prints λ_u = 0.278, but the official calibration behind the published model
  results uses **0.282** (the tied rates λ_e = 0.428·λ_u, λ_n = 0.645·λ_u
  follow it); this implementation uses 0.282.
- **Prices** — the model is partial equilibrium; the constants (w, r, T) are
  calibrated once to a background Cobb–Douglas general equilibrium
  (Section II.A) and never move over the cycle. `--quick` skips this
  calibration loop and uses the converged constants directly.
- **Business-cycle shocks** — the two-state friction process of Section II.B
  (ε_λ = 0.0662, ε_σ = 0.00239, ρ = 0.983), simulated for 5,000 months from
  seed 1 with a 1,000-month burn-in.

## Known limitations (documented, not hidden)

- **Business-cycle volatilities** run ~5% above the published model values.
  The published output was produced with a different random-number generator
  for the aggregate shock path (same seed, different algorithm), so the two
  simulations average over different 4,000-month samples. Correlations and
  autocorrelations — far less sensitive to the particular path — match to
  ~0.01–0.03.
- **Average J2J wage gain**: 0.0320 here versus 0.0329 in the published log.
  The original's accounting of the separation-with-immediate-rehire channel
  reads an out-of-scope array index, contaminating the shipped figure; this
  implementation counts wage gains on on-the-job ladder moves only. Both are
  near the 0.033 calibration target.
- **Table 11** is computed with a counterfactual-covariance decomposition of
  the flow-implied steady-state unemployment rate. It reproduces the paper's
  qualitative finding (flows involving U dominate; the E&N contribution is
  negligible) but not the published U&E / U&N split, which is produced with
  a different implementation of the Elsby–Hobijn–Şahin (2015) method that is
  external to the shipped model code.

## How to run

```bash
pip install -r requirements.txt   # numpy, scipy, pandas (pinned)
python main.py                    # full run: price calibration + cycle
python main.py --quick            # same results; skips the price loop
```

(Windows: `py main.py`.) The run prints every table against the paper's model
column, validates against `refs/`, writes CSVs to `outputs/`, and produces the
gross-flow comparison figure in `figures/`. The simulation seed is fixed
(seed 1, first period in the good state), so runs are reproducible. Full run
≈ 12–15 minutes on a laptop.

## Code map

| Module | Contents | Paper |
|---|---|---|
| `parameters.py` | calibration, frictions, shock process, numerics | Table 4, §II |
| `discretize.py` | Tauchen chain, i.i.d. bins, annual→monthly conversion, grids | §I.A, §II.A |
| `household.py` | budgets, UI, Bellman expectation, savings choice, VFI | §I.B–I.C |
| `cross_section.py` | population law of motion, routing tables, gross-flow accounting | §I.B, §I.D |
| `equilibrium.py` | steady state, background-GE price calibration, Table 6 | §II.A |
| `cycle.py` | two-state aggregate shock, cycle value functions, simulation | §II.B, §III |
| `moments.py` | quarterly aggregation, HP(1600), moments, variance decomposition | §III.A, §III.D |
| `main.py` | driver: run, compare, validate, save | — |


## Requirements

Python ≥ 3.11 with `numpy`, `scipy`, `pandas`, `matplotlib` (pinned versions
in `requirements.txt`; tested with Python 3.13.5). No other dependencies.

## Reference

Krusell, P., Mukoyama, T., Rogerson, R. and Şahin, A. (2017). *Gross Worker
Flows over the Business Cycle.* American Economic Review, 107(11): 3447–3476.
