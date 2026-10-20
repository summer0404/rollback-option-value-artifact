#!/usr/bin/env python3
"""Belief-grid convergence of ROVER's tables at the operating horizon T = 24.

The paper's grid check (theory_checks.json, key GRID) compares a 1,001-point grid with the exact
recursion for T <= 6 on the E1 instance. ROVER uses 501-point tables with T = 24 and the E3 carrier
parameters. Here each ROVER table (testbed.carrier_params exactly as testbed.planner builds it, variant
"rover") is solved on grids of G points and compared with a reference grid of G_REF points:

  value error   |V_G - V_ref| at t = 0..23, every D <= t (reachable before any recovery), 1,001 beliefs
  action share  states where the canonical action (ties W > R > A) differs from the reference's
  post-audit    states where the recover-after-audit rule differs
  solve time    seconds per table (machine dependent)

The reference is itself a grid, so this is a numerical convergence study, not an error bound.
Output: results/grid/grid_convergence.csv and .json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import post_common as C  # noqa: E402
from model import GridDP, EPS, W, R, A  # noqa: E402

TB = C.TB
GRIDS = (126, 251, 501, 1001, 2001, 4001)
G_REF = 8001
R1S = (0.0, 0.1, 1.0, 10.0)
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exploratory", "grid"))


def rover_params(i, r1):
    """The parameters testbed.planner(i, r1, 'rover') solves (default Knowledge, exposure window)."""
    return TB.carrier_params(i, TB.effective_r1(r1), None, rho_until=TB.T0_MAX, rho_late=TB.RHO / 4,
                             c_a=None, rho=TB.RHO)


def tables(dp, bs):
    """Values, canonical actions and post-audit recovery flags on (t, D <= t, b in bs)."""
    T = dp.p.T
    out_v, out_a, out_post = [], [], []
    for t in range(T):
        D = np.repeat(np.arange(t + 1), len(bs))
        b = np.tile(bs, t + 1)
        qw, qr, qa = dp.qall(t, D, b)
        m = np.minimum(np.minimum(qw, qr), qa)
        act = np.where(qw <= m + EPS, W, np.where(qr <= m + EPS, R, A))
        post = qr <= qw + EPS
        out_v.append(m)
        out_a.append(act)
        out_post.append(post)
    return np.concatenate(out_v), np.concatenate(out_a), np.concatenate(out_post)


def main():
    os.makedirs(OUT, exist_ok=True)
    bs = np.linspace(0.0, 1.0, 1001)
    rows = []
    for i, cname in enumerate(TB.CARRIER_NAMES):
        for r1 in R1S:
            p = rover_params(i, r1)
            t0 = time.perf_counter()
            ref = GridDP(p, G=G_REF)
            t_ref = time.perf_counter() - t0
            v_ref, a_ref, post_ref = tables(ref, bs)
            for G in GRIDS:
                t0 = time.perf_counter()
                dp = GridDP(p, G=G)
                ts = time.perf_counter() - t0
                v, a, post = tables(dp, bs)
                err = np.abs(v - v_ref)
                rel = err / np.maximum(np.abs(v_ref), 1e-9)
                nonwait_ref = a_ref != W
                rows.append(dict(carrier=cname, r1=r1, r1_eff=TB.effective_r1(r1), G=G, G_ref=G_REF,
                                 states=int(v.size), solve_s=round(ts, 4), solve_ref_s=round(t_ref, 4),
                                 V0_G=float(dp.value(0, 0, 0.0)), V0_ref=float(ref.value(0, 0, 0.0)),
                                 max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
                                 max_rel_err=float(rel.max()), action_disagree_share=float((a != a_ref).mean()),
                                 action_disagree_n=int((a != a_ref).sum()),
                                 nonwait_states_ref=int(nonwait_ref.sum()),
                                 disagree_among_nonwait_ref=float((a != a_ref)[nonwait_ref].mean()) if nonwait_ref.any() else 0.0,
                                 post_disagree_share=float((post != post_ref).mean())))
                print(f"{cname:7s} r1={r1:<5g} G={G:5d}: max|dV|={err.max():.2e} act-disagree={rows[-1]['action_disagree_share']:.4%} "
                      f"solve={ts:.3f}s", flush=True)
    import csv
    with open(os.path.join(OUT, "grid_convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for G in GRIDS:
        rs = [r for r in rows if r["G"] == G]
        summ[str(G)] = dict(max_abs_err=max(r["max_abs_err"] for r in rs),
                            max_rel_err=max(r["max_rel_err"] for r in rs),
                            max_action_disagree_share=max(r["action_disagree_share"] for r in rs),
                            mean_action_disagree_share=float(np.mean([r["action_disagree_share"] for r in rs])),
                            max_post_disagree_share=max(r["post_disagree_share"] for r in rs),
                            mean_solve_s=float(np.mean([r["solve_s"] for r in rs])))
    json.dump(dict(grids=GRIDS, G_ref=G_REF, beliefs=len(bs), horizon=TB.T_MS, summary=summ,
                   note="numerical convergence against a finer grid, not a certified error bound"),
              open(os.path.join(OUT, "grid_convergence.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
