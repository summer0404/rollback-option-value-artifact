"""E1: exact DP on the restricted model.

E1a  grid r1 in {0, 0.1, 1, 10} x phi in {0, 0.5, 1}: exact expected loss (grid policy
     evaluation) of the DP policy and heuristics, Monte Carlo audit and recovery rates
     (5 seeds x N episodes, bootstrap CIs), and the stop rule of the study proposal (RUNLOG.md).
E1b  sensitivity sweeps over audit power d and activation hazard lam.
E1c  model-misspecification and ablation: planners solved with a wrong lam, r1 or phi
     (irreversibility-blind), evaluated exactly under the true model.
Writes results/e1_grid.csv, e1_grid_seeds.csv, e1_sens.csv, e1_misspec.csv.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (Params, GridDP, GridPolicyEval, DPPolicy, MyopicPolicy, FixedPolicy,  # noqa: E402
                   FrozenDPPolicy, MyopicAuditDPRecover, simulate, simulate_misspec)
from stats import boot_ci  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
BASE = Params(T=20, pi0=0.2, rho=0.02, Delta=1, lam=0.1, phi=0.5, H=20.0, h=1.0, r0=3.0, r1=1.0,
              c_a=1.0, d=0.9, alpha=0.05, s0=0.1, s1=0.4, c_fr=0.0, D0=4)
R1S = (0.0, 0.1, 1.0, 10.0)
PHIS = (0.0, 0.5, 1.0)
SEEDS = 5
NEP = 10000
G = 1001


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def policies(p, dp):
    return {
        "dp": DPPolicy(dp),
        "myopic": MyopicPolicy(p),
        "myopic_dprec": MyopicAuditDPRecover(p, dp),
        "no_audit": FixedPolicy("no_audit"),
        "periodic4": FixedPolicy("periodic4", every=4),
    }


def e1a():
    rows, seed_rows = [], []
    for phi in PHIS:
        for r1 in R1S:
            p = replace(BASE, phi=phi, r1=r1)
            dp = GridDP(p, G=G)
            for name, pol in policies(p, dp).items():
                exact = dp.value() if name == "dp" else GridPolicyEval(p, pol, G=G).value()
                per_seed = []
                allm = {}
                for s in range(SEEDS):
                    rng = np.random.default_rng([5, int(phi * 10), int(r1 * 10), s])
                    m = simulate(p, pol, NEP, rng)
                    per_seed.append({k: float(v.mean()) for k, v in m.items()})
                    for k, v in m.items():
                        allm.setdefault(k, []).append(v)
                    seed_rows.append(dict(phi=phi, r1=r1, policy=name, seed=s,
                                          **{k: float(v.mean()) for k, v in m.items()}))
                allm = {k: np.concatenate(v) for k, v in allm.items()}
                aud = allm["audits"] / p.T
                lo, hi = boot_ci(aud, seed=11)
                llo, lhi = boot_ci(allm["loss"], seed=12)
                rows.append(dict(phi=phi, r1=r1, policy=name, exact_loss=exact,
                                 mc_loss=float(allm["loss"].mean()), mc_loss_lo=llo, mc_loss_hi=lhi,
                                 audit_rate=float(aud.mean()), audit_lo=lo, audit_hi=hi,
                                 blind_rec=float(allm["rec_chosen"].mean()), post_rec=float(allm["rec_post"].mean()),
                                 forced_rec=float(allm["rec_forced"].mean()), false_rec=float(allm["rec_false"].mean()),
                                 irrev=float(allm["irrev"].mean()), replay=float(allm["replay"].mean()),
                                 audit_cost=float(allm["audit_cost"].mean()), resid=float(allm["resid"].mean()),
                                 seed_sd_loss=float(np.std([x["loss"] for x in per_seed], ddof=1))))
            print(f"E1a phi={phi} r1={r1} done", flush=True)
    write(os.path.join(RES, "e1_grid.csv"), rows)
    write(os.path.join(RES, "e1_grid_seeds.csv"), seed_rows)
    return rows


def e1b():
    rows = []
    for knob, vals in (("d", (0.5, 0.7, 0.9, 1.0)), ("lam", (0.02, 0.05, 0.1, 0.2)),
                       ("s1", (0.1, 0.25, 0.4, 0.7)), ("c_a", (0.25, 0.5, 1.0, 2.0))):
        for v in vals:
            for phi in (0.0, 1.0):
                for r1 in (0.0, 1.0, 10.0):
                    p = replace(BASE, phi=phi, r1=r1, **{knob: v})
                    dp = GridDP(p, G=G)
                    my = GridPolicyEval(p, MyopicPolicy(p), G=G).value()
                    na = GridPolicyEval(p, FixedPolicy("no_audit"), G=G).value()
                    m = simulate(p, DPPolicy(dp), NEP, np.random.default_rng([7, int(v * 100), int(phi), int(r1)]))
                    rows.append(dict(knob=knob, value=v, phi=phi, r1=r1, dp=dp.value(), myopic=my, no_audit=na,
                                     gap_pct=100 * (my - dp.value()) / dp.value(),
                                     dp_audit_rate=float(m["audits"].mean() / p.T),
                                     dp_blind_rec=float(m["rec_chosen"].mean())))
        print(f"E1b {knob} done", flush=True)
    write(os.path.join(RES, "e1_sens.csv"), rows)
    return rows


def e1c():
    """Planners solved with a wrong model and run with their own belief (Knowledge misspecification).

    Value: Monte Carlo under the true model with the planner's belief update (5 seeds x NEP episodes),
    regret relative to the exact optimal value. For phi and r1 the belief update is unaffected and the
    exact grid evaluation is also reported.
    """
    rows = []
    variants = (("irrev_blind", dict(phi=0.0)), ("rollback_blind", "r1x10"), ("rollback_optimist", "r1x0.1"),
                ("lam_half", "lamx0.5"), ("lam_double", "lamx2"), ("alarm_optimist", dict(s1=0.6)),
                ("alarm_pessimist", dict(s1=0.25)), ("audit_optimist", dict(d=1.0, alpha=0.0)),
                ("audit_pessimist", dict(d=0.7, alpha=0.1)))
    for phi in (0.5, 1.0):
        for r1 in R1S:
            p = replace(BASE, phi=phi, r1=r1)
            dp = GridDP(p, G=G)
            opt = dp.value()
            for name, spec in variants:
                if isinstance(spec, dict):
                    q = replace(p, **spec)
                elif spec == "r1x10":
                    q = replace(p, r1=max(p.r1, 0.1) * 10)
                elif spec == "r1x0.1":
                    q = replace(p, r1=p.r1 * 0.1)
                elif spec == "lamx0.5":
                    q = replace(p, lam=p.lam * 0.5)
                else:
                    q = replace(p, lam=min(0.95, p.lam * 2))
                pol = FrozenDPPolicy(GridDP(q, G=G), name)
                ms = [simulate_misspec(p, q, pol, NEP, np.random.default_rng([9, int(phi * 10), int(r1 * 10), s]))
                      for s in range(SEEDS)]
                L = np.concatenate([m["loss"] for m in ms])
                lo, hi = boot_ci(L, seed=4)
                belief_free = name in ("irrev_blind", "rollback_blind", "rollback_optimist")
                exact = GridPolicyEval(p, pol, G=G).value() if belief_free else float("nan")
                v = exact if belief_free else float(L.mean())
                rows.append(dict(phi=phi, r1=r1, planner=name, true_opt=opt, value=v, mc=float(L.mean()),
                                 mc_lo=lo, mc_hi=hi, regret_pct=100 * (v - opt) / opt,
                                 audit_rate=float(np.concatenate([m["audits"] for m in ms]).mean() / p.T),
                                 irrev=float(np.concatenate([m["irrev"] for m in ms]).mean())))
        print(f"E1c phi={phi} done", flush=True)
    write(os.path.join(RES, "e1_misspec.csv"), rows)
    return rows


if __name__ == "__main__":
    e1a()
    e1b()
    e1c()
