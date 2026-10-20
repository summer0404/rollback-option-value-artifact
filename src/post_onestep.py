#!/usr/bin/env python3
"""Full-action one-step controllers in the restricted model (E1).

The paper's myopic rule (Definition 4, model.MyopicPolicy) only chooses wait or audit and never
recovers blindly. This script adds one-step lookahead controllers that may wait, recover or audit,
and evaluates all of them exactly on the belief grid (model.GridPolicyEval, as the paper evaluates the
myopic rule), so the gap to the DP splits into its parts:

  onestep_term_wa   one-step lookahead, continuation = terminal cost h*D*b, actions W/A (recover only
                    after an audit)
  onestep_term_wra  the same with blind recovery allowed
  rollout_na_wa     one-step lookahead, continuation = exact cost-to-go of never auditing (one policy
                    improvement step over no audit), actions W/A
  rollout_na_wra    the same with blind recovery allowed
  myopic            the paper's Definition 4 rule
  myopic_dprec      Definition 4 audit timing with the DP's recovery decisions

Every controller recovers after an audit when its own one-step Q_R <= Q_W at the posterior. These
definitions are stated in the supplement (exploratory analyses).
Cells: the E1a grid (phi x r1, base instance of exp_e1.BASE) and the E1b one-at-a-time sensitivity grid.
A Monte Carlo cross-check (20,000 episodes, one seed) accompanies the E1a cells.
Output: results/onestep/onestep.csv, onestep_summary.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE
sys.path.insert(0, SRC)
from model import (GridDP, GridPolicyEval, MyopicPolicy, MyopicAuditDPRecover, FixedPolicy, DPPolicy,  # noqa: E402
                   simulate, EPS, W, R, A)
from exp_e1 import BASE, R1S, PHIS, G  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "results", "exploratory", "onestep"))


class ContinuationTable(GridDP):
    """GridDP whose V[t+1] is a fixed continuation instead of the optimal cost-to-go.

    cont='terminal': V[t] = h*D*b for every t (one-step lookahead against the terminal cost).
    cont=<GridPolicyEval>: V[t] = that policy's exact-on-grid cost-to-go (rollout).
    """

    def __init__(self, p, G, cont):
        self._cont = cont
        super().__init__(p, G)

    def solve(self):
        p = self.p
        if isinstance(self._cont, str):
            term = self.grid[None, :] * p.h * np.arange(self.nD)[:, None]
            self.V = [term] * (p.T + 1)
        else:
            assert self._cont.G == self.G and self._cont.nD == self.nD
            self.V = list(self._cont.V)


class LookaheadPolicy:
    def __init__(self, table, allow_blind, name):
        self.t, self.allow_blind, self.name = table, allow_blind, name

    def act(self, t, D, b):
        qw, qr, qa = self.t.qall(t, D, b)
        if not self.allow_blind:
            qr = np.full(np.shape(qr), np.inf)
        m = np.minimum(np.minimum(qw, qr), qa)
        return np.where(qw <= m + EPS, W, np.where(qr <= m + EPS, R, A))

    def post(self, t, D, b):
        return self.t.qR(t, D, b) <= self.t.qW(t, D, b) + EPS


def controllers(p, dp):
    na_eval = GridPolicyEval(p, FixedPolicy("no_audit"), G=G)
    term = ContinuationTable(p, G, "terminal")
    roll = ContinuationTable(p, G, na_eval)
    pols = {
        "onestep_term_wa": LookaheadPolicy(term, False, "onestep_term_wa"),
        "onestep_term_wra": LookaheadPolicy(term, True, "onestep_term_wra"),
        "rollout_na_wa": LookaheadPolicy(roll, False, "rollout_na_wa"),
        "rollout_na_wra": LookaheadPolicy(roll, True, "rollout_na_wra"),
        "myopic": MyopicPolicy(p),
        "myopic_dprec": MyopicAuditDPRecover(p, dp),
    }
    return pols, na_eval.value()


def evaluate(p, mc_seed=None):
    dp = GridDP(p, G=G)
    opt = dp.value()
    pols, v_na = controllers(p, dp)
    row = dict(dp=opt, no_audit=v_na)
    mc = {}
    for name, pol in pols.items():
        v = GridPolicyEval(p, pol, G=G).value()
        row[name] = v
        row[f"gap_{name}"] = 100 * (v - opt) / opt
        if mc_seed is not None:
            m = simulate(p, pol, 20000, np.random.default_rng(mc_seed))
            L = m["loss"]
            mc[name] = dict(mc=float(L.mean()), se=float(L.std(ddof=1) / np.sqrt(len(L))),
                            z=float((L.mean() - v) / (L.std(ddof=1) / np.sqrt(len(L)))),
                            audits_per100=float(100 * m["audits"].mean() / p.T),
                            blind_rec=float(m["rec_chosen"].mean()), irrev=float(m["irrev"].mean()))
    row["gap_no_audit"] = 100 * (v_na - opt) / opt
    return row, mc


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, mcs = [], {}
    for phi in PHIS:
        for r1 in R1S:
            p = replace(BASE, phi=phi, r1=r1)
            row, mc = evaluate(p, mc_seed=[42, int(phi * 10), int(r1 * 10)])
            rows.append(dict(grid="E1a", knob="base", value="", phi=phi, r1=r1, **row))
            mcs[f"phi={phi},r1={r1}"] = mc
            print(f"E1a phi={phi} r1={r1}: " + " ".join(f"{k[4:]}={v:.1f}%" for k, v in row.items() if k.startswith("gap_")),
                  flush=True)
    for knob, vals in (("d", (0.5, 0.7, 0.9, 1.0)), ("lam", (0.02, 0.05, 0.1, 0.2)),
                       ("s1", (0.1, 0.25, 0.4, 0.7)), ("c_a", (0.25, 0.5, 1.0, 2.0))):
        for v in vals:
            for phi in (0.0, 1.0):
                for r1 in (0.0, 1.0, 10.0):
                    p = replace(BASE, phi=phi, r1=r1, **{knob: v})
                    row, _ = evaluate(p)
                    rows.append(dict(grid="E1b", knob=knob, value=v, phi=phi, r1=r1, **row))
        print(f"E1b {knob} done", flush=True)
    with open(os.path.join(OUT, "onestep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    names = ("myopic", "myopic_dprec", "onestep_term_wa", "onestep_term_wra", "rollout_na_wa", "rollout_na_wra", "no_audit")
    summ = {}
    for gname in ("E1a", "E1b", "all"):
        rs = [r for r in rows if gname == "all" or r["grid"] == gname]
        summ[gname] = {n: dict(max_gap_pct=float(max(r[f"gap_{n}"] for r in rs)),
                               median_gap_pct=float(np.median([r[f"gap_{n}"] for r in rs])),
                               base_cell_gap_pct=float(next(r[f"gap_{n}"] for r in rows
                                                            if r["grid"] == "E1a" and r["phi"] == 0.5 and r["r1"] == 1.0)))
                       for n in names}
    max_abs_z = max(abs(v["z"]) for c in mcs.values() for v in c.values())
    json.dump(dict(summary=summ, monte_carlo_check=mcs, mc_max_abs_z=max_abs_z, cells=len(rows),
                   grid_points=G, base=BASE.asdict()), open(os.path.join(OUT, "onestep_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("MC cross-check max |z| =", round(max_abs_z, 2))


if __name__ == "__main__":
    main()
