"""Reproduce Krusell, Mukoyama, Rogerson and Sahin (2017).

Pipeline:
    1. steady state, with the constant prices calibrated to the background
       general equilibrium (Tables 5 and 6);
    2. business cycle: value functions over the two aggregate states and a
       5,000-month simulated path (Tables 7, 8 panel C, 9);
    3. unemployment variance decomposition (Table 11, qualitative);
    4. the gross-flow comparison figure, validation against the authors'
       published model output in ``refs/`` and the paper's printed tables;
       results written to ``outputs/`` and ``figures/``.

Run:  python main.py            full run (price calibration; ~15-30 min)
      python main.py --quick    solve at the authors' converged prices (skips the calibration loop)
                                
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from cycle import simulate, solve_cycle
from equilibrium import flows_by_wealth, solve_steady_state
from household import Prices
from moments import cycle_statistics, variance_decomposition
from parameters import BusinessCycle, Calibration, Numerics

OUT = "outputs"

# --------------------------------------------------------------------------- #
# Published model values (for the comparison printout only)                    #
# --------------------------------------------------------------------------- #
PAPER_TABLE5 = {("E", "E"): 0.972, ("E", "U"): 0.014, ("E", "N"): 0.014,
                ("U", "E"): 0.219, ("U", "U"): 0.652, ("U", "N"): 0.130,
                ("N", "E"): 0.022, ("N", "U"): 0.020, ("N", "N"): 0.958}
PAPER_TABLE7 = {"urate": (0.1207, -0.99, 0.87), "lfpr": (0.0015, 0.37, 0.71),
                "E": (0.0096, 0.995, 0.89)}
PAPER_TABLE8C = {"feu": (0.089, -0.79, 0.76), "fen": (0.057, 0.21, 0.21),
                 "fue": (0.088, 0.69, 0.70), "fun": (0.029, 0.47, 0.34),
                 "fne": (0.051, 0.57, 0.66), "fnu": (0.076, -0.96, 0.87)}
PAPER_TABLE9 = {"fjj": (0.098, 0.54, 0.72)}
PAPER_TABLE11 = (74.1, 31.1, -3.8)


# --------------------------------------------------------------------------- #
# Reference-output values (authors' files)                                   #
# --------------------------------------------------------------------------- #
def parse_logfile(path: str) -> dict:
    """Steady-state flows etc. from the published log.
       last occurrence prevails (converged)."""
    text = open(path).read()

    def last(label: str):
        # Lines are either "label = value" or "label = (target: x) value".
        pat = rf"{re.escape(label)}\s*=\s*(?:\(target:[^)]*\))?\s*(-?[\d.]+)"
        hits = re.findall(pat, text)
        return float(hits[-1]) if hits else None

    out = {k: last(f"{k} flow") for k in
           ("EE", "JS", "EU", "EN", "UE", "UU", "UN", "NE", "NU", "NN")}
    out["urate"] = last("urate")
    out["lfpr"] = last("lfpr")
    for q in range(1, 6):
        for k in ("EE", "JS", "EU", "EN", "UE", "UU", "UN", "NE", "NU", "NN"):
            out[f"{k}_Q{q}"] = last(f"{k} flow Q{q}")
    return out


def parse_stats(path: str) -> dict:
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


def banner(msg: str) -> None:
    print("\n" + "=" * 74 + f"\n{msg}\n" + "=" * 74, flush=True)


# --------------------------------------------------------------------------- #
# Gross-flow comparison figure                                                 #
# --------------------------------------------------------------------------- #
def make_flow_figure(stats: dict, path: str = "figures/flows_combined.pdf") -> None:
    """Bar-chart comparison of the six flow moments with the paper (Table 8C):
    cyclicality (top panel) and volatility (bottom panel).  The figure is saved
    to ``path``. """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "TeX Gyre Pagella", "Palatino",
                       "URW Palladio L", "P052", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 18, "axes.titlesize": 16, "axes.labelsize": 14,
        "legend.fontsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.dpi": 110, "savefig.bbox": "tight",
    })

    flows = ["feu", "fen", "fue", "fun", "fne", "fnu"]
    x = np.arange(len(flows))
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    panels = [("cyclicality", "corr_{}Y", 1, "corr(flow, output)"),
              ("volatility", "std_{}", 0, "std (HP-filtered log)")]
    for ax, (kind, key, col, ylab) in zip(axes, panels):
        model = [stats[key.format(f)] for f in flows]
        paper = [PAPER_TABLE8C[f][col] for f in flows]
        ax.bar(x - 0.2, model, 0.4, label="Replication")
        ax.bar(x + 0.2, paper, 0.4, label="Paper (Table 8C)")
        ax.set_xticks(x)
        ax.set_xticklabels([f.upper()[1:] for f in flows])
        ax.axhline(0, color="grey", lw=.5)
        ax.set_ylabel(ylab)
        ax.set_title(f"Gross-flow {kind}: replication vs paper")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path)
    print(f"  figure written to {path}", flush=True)


# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="use the converged background-GE constants directly")
    args = ap.parse_args()

    t0 = datetime.now()
    os.makedirs(OUT, exist_ok=True)
    cal, num, bc = Calibration(), Numerics(), BusinessCycle()

    # ----- 1. Steady state ------------------------------------------------ #
    banner("1/4  Steady state" + ("  (quick: converged constants)" if args.quick
                                  else "  (calibrating the constant prices)"))
    prices = Prices.converged(cal) if args.quick else None
    ss = solve_steady_state(cal, num, prices=prices, verbose=True)
    fl, ag = ss.flows, ss.aggregates
    print(f"\n  prices: w={ss.prices.w:.4f}  1+r={1 + ss.prices.r:.5f}  "
          f"T={ss.prices.transfer:.4f}")
    print(f"  implied: K/L={ag['KL']:.3f}  avg earnings={ag['avg_earnings']:.4f}  "
          f"T balance={ag['transfer_balancing']:.4f}")

    table5 = {("E", "E"): fl["EE"] + fl["JJ"], ("E", "U"): fl["EU"], ("E", "N"): fl["EN"],
              ("U", "E"): fl["UE"], ("U", "U"): fl["UU"], ("U", "N"): fl["UN"],
              ("N", "E"): fl["NE"], ("N", "U"): fl["NU"], ("N", "N"): fl["NN"]}
    print("\n  Table 5 (average gross flows):")
    print(f"  {'':>6}{'model':>10}{'paper':>10}{'diff':>10}")
    rows5 = []
    for (i, j), v in table5.items():
        p = PAPER_TABLE5[(i, j)]
        print(f"  {i}->{j:<3}{v:>10.5f}{p:>10.3f}{v - p:>+10.5f}")
        rows5.append({"from": i, "to": j, "model": v, "paper_model": p})
    print(f"  urate={fl['urate']:.5f} (paper 0.068)   lfpr={fl['lfpr']:.5f} (paper 0.66)")
    print(f"  avg J2J wage gain = {fl['wage_gain']:.4f} (target 0.033)")
    pd.DataFrame(rows5).to_csv(f"{OUT}/table5.csv", index=False)

    ref = parse_logfile("refs/Logfile.txt") if os.path.exists("refs/Logfile.txt") else None
    if ref:
        pairs = {"EE": "EE", "JJ": "JS", "EU": "EU", "EN": "EN", "UE": "UE",
                 "UU": "UU", "UN": "UN", "NE": "NE", "NU": "NU", "NN": "NN"}
        worst = max(abs(fl[k] - ref[rk]) for k, rk in pairs.items())
        print(f"  validation: max |flow - published model output| = {worst:.2e}")

    # ----- Table 6 ---------------------------------------------------------- #
    banner("2/4  Gross flows by wealth quintile (Table 6)")
    t6 = flows_by_wealth(ss)
    print(f"  {'flow':>6}" + "".join(f"{'Q' + str(i):>8}" for i in range(1, 6)))
    for k in ("EU", "EN", "UE", "UN", "NE", "NU", "EE", "UU", "NN", "JJ"):
        print(f"  {k:>6}" + "".join(f"{x:>8.2f}" for x in t6[k]))
    pd.DataFrame(t6, index=[f"Q{i}" for i in range(1, 6)]).T.to_csv(f"{OUT}/table6.csv")

    # ----- 3. Business cycle ------------------------------------------------ #
    banner(f"3/4  Business cycle ({bc.n_months} months, seed {bc.seed})")
    sol = solve_cycle(cal, num, ss.grids, ss.prices, bc,
                      v_init=ss.values, verbose=True)
    series = simulate(sol, cal, num, ss.grids, ss.prices, bc, ss.population,
                      verbose=True)
    np.savez_compressed(f"{OUT}/monthly_series.npz", **series)
    stats = cycle_statistics(series, bc.burn_in)

    def show(block: dict, title: str) -> list[dict]:
        print(f"\n  {title}:")
        print(f"  {'series':>7}{'std':>9}{'(paper)':>9}{'corrY':>9}{'(paper)':>9}"
              f"{'AR1':>8}{'(paper)':>9}")
        rows = []
        for k, (ps, pc, pa) in block.items():
            s, c, a = stats[f"std_{k}"], stats[f"corr_{k}Y"], stats[f"autocorr_{k}"]
            print(f"  {k:>7}{s:>9.4f}{ps:>9.4f}{c:>9.3f}{pc:>9.3f}{a:>8.3f}{pa:>9.3f}")
            rows.append({"series": k, "std": s, "std_paper": ps, "corrY": c,
                         "corrY_paper": pc, "ar1": a, "ar1_paper": pa})
        return rows

    rows7 = show(PAPER_TABLE7, "Table 7 (stocks)")
    rows8 = show(PAPER_TABLE8C, "Table 8 panel C (gross flows)")
    rows9 = show(PAPER_TABLE9, "Table 9 (job-to-job rate)")
    pd.DataFrame(rows7).to_csv(f"{OUT}/table7.csv", index=False)
    pd.DataFrame(rows8 + rows9).to_csv(f"{OUT}/table8_9.csv", index=False)

    refs = parse_stats("refs/stats_M.txt") if os.path.exists("refs/stats_M.txt") else None
    if refs:
        checks = {"corr_urateY": "corr_urateY", "corr_lfprY": "corr_lfprY",
                  "corr_feuY": "corr_feuY", "corr_fnuY": "corr_fnuY"}
        dev = max(abs(stats[k] - refs[rk]) for k, rk in checks.items() if rk in refs)
        print(f"\n  validation: max |corr - published model output| = {dev:.3f}")

    # ----- 4. Variance decomposition ---------------------------------------- #
    banner("4/4  Unemployment variance decomposition (Table 11)")
    vd = variance_decomposition(series, bc.burn_in)
    print(f"  U&E={vd['U_E']:.1f}   U&N={vd['U_N']:.1f}   E&N={vd['E_N']:.1f}"
          f"   (paper: {PAPER_TABLE11[0]} / {PAPER_TABLE11[1]} / {PAPER_TABLE11[2]};"
          f" qualitative -- see README)")

    make_flow_figure(stats)
    print(f"\nOutputs in {OUT}/.  Total runtime: {datetime.now() - t0}", flush=True)


if __name__ == "__main__":
    main()