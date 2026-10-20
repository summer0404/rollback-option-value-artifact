"""E2: Stackelberg audit allocation across carriers against a best-responding attacker.

E2a  minimax coverage x*(r1), attacker level u*(r1) and support on an r1 grid.
E2b  four commitments (uniform, phi-proportional, recovery-blind = minimax solved at
     r1 = 10, recovery-aware minimax at the true r1) at r1 in {0, 0.1, 1, 10}: attacker
     best response and expected damage, with Monte Carlo validation (5 seeds).
Writes results/e2_coverage.csv, e2_commitments.csv.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import CARRIERS, GameParams, minimax, best_response, utility, damage, simulate_damage, AB, a10_monotone  # noqa
from stats import boot_ci  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
G0 = GameParams()
R1_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0)
R1_TAB = (0.0, 0.1, 1.0, 10.0)
SEEDS = 5
NEP = 20000


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def e2a():
    rows = []
    for r1 in R1_GRID:
        g = replace(G0, r1=r1)
        x, u, sup = minimax(CARRIERS, g)
        br0, U0 = best_response(CARRIERS, g, np.zeros(len(CARRIERS)))
        row = dict(r1=r1, u_star=u, support="+".join(CARRIERS[i].name for i in sup),
                   br_uncovered=CARRIERS[br0].name if br0 >= 0 else "abstain",
                   a10_monotone=int(a10_monotone(CARRIERS, g)),
                   interior=int(all(1e-4 < x[i] < 1 - 1e-4 for i in sup)))
        for c, xi, ui in zip(CARRIERS, x, U0):
            A0, B0 = AB(c, g, 0.0)
            row[f"x_{c.name}"] = float(xi)
            row[f"U0_{c.name}"] = float(ui)
            row[f"sB0_{c.name}"] = float(c.s * B0)
            row[f"sA0_{c.name}"] = float(c.s * A0 - c.kappa)
        rows.append(row)
    write(os.path.join(RES, "e2_coverage.csv"), rows)
    return rows


def commitments(g):
    n = len(CARRIERS)
    phis = np.array([c.phi for c in CARRIERS])
    phi_prop = g.B * phis / phis.sum()
    blind, _, _ = minimax(CARRIERS, replace(g, r1=10.0))
    aware, _, _ = minimax(CARRIERS, g)
    return {"uniform": np.full(n, g.B / n), "phi_proportional": np.minimum(phi_prop, 1.0),
            "recovery_blind": blind, "recovery_aware": aware}


def e2b():
    rows = []
    for r1 in R1_TAB:
        g = replace(G0, r1=r1)
        for name, x in commitments(g).items():
            br, U = best_response(CARRIERS, g, x)
            c = CARRIERS[br]
            # Monte Carlo of the attacker's realised payoff s*L - kappa on the best-response carrier
            per_seed = []
            pooled = []
            for s in range(SEEDS):
                rng = np.random.default_rng([21, int(r1 * 100), s, list(commitments(g)).index(name)])
                L = simulate_damage(c, g, float(x[br]), NEP, rng)
                success = rng.random(NEP) < c.s
                pay = success * L - c.kappa
                per_seed.append(float(pay.mean()))
                pooled.append(pay)
            pooled = np.concatenate(pooled)
            lo, hi = boot_ci(pooled, seed=3)
            rows.append(dict(r1=r1, commitment=name, br=c.name, attacker_utility=float(U[br]),
                             mc_utility=float(pooled.mean()), mc_lo=lo, mc_hi=hi,
                             seed_sd=float(np.std(per_seed, ddof=1)),
                             **{f"x_{cc.name}": float(xx) for cc, xx in zip(CARRIERS, x)},
                             **{f"U_{cc.name}": float(uu) for cc, uu in zip(CARRIERS, U)}))
    write(os.path.join(RES, "e2_commitments.csv"), rows)
    return rows


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    e2a()
    e2b()
    print("E2 done")
