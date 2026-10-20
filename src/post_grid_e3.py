#!/usr/bin/env python3
"""Testbed part of the grid study: does a denser belief grid change ROVER's E3 decisions and losses?

Runs ROVER on every held-out cell (30 streams x 8 seeds, label main) at r1 in {0, 0.1, 1, 10} with
planner tables of G points instead of the default 501, by replacing the GridDP name that
testbed.planner looks up (testbed.py itself is not edited). Compares with the committed G = 501 rows:
per-episode identity of decisions (audits, recoveries, loss), paired per-stream loss differences with
the paper's bootstrap, and H1 (tuned rollback only minus ROVER at r1 = 1) recomputed with the dense
ROVER. Post hoc and exploratory; the confirmatory result is unchanged.

  python3 -B post_grid_e3.py --grid 2001 --workers 3 --tmp <scratch dir>
  python3 -B post_grid_e3.py --grid 2001 --analyze-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import post_common as C  # noqa: E402
import model  # noqa: E402

TB = C.TB
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exploratory", "grid"))
R1S = (0.0, 0.1, 1.0, 10.0)
_REAL_GRIDDP = model.GridDP
_CUR = [None]


def _set_grid(G):
    if _CUR[0] != G:
        TB._DP_CACHE.clear()
        TB.GridDP = lambda p, G=501, _G=G: _REAL_GRIDDP(p, G=_G)   # testbed.planner passes G=501
        _CUR[0] = G


def run_episode_grid(job):
    _set_grid(job["grid"])
    return TB.run_episode(job)


def boot_mean(v, seed, B=2000):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    bs = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def per_stream(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[f"{r['family']}-{r['stream']}"].append(float(r[key]))
    return {s: float(np.mean(v)) for s, v in d.items()}


def paired(a_rows, b_rows, key, seed=7):
    a, b = per_stream(a_rows, key), per_stream(b_rows, key)
    st = sorted(set(a) & set(b))
    diff = np.array([a[s] - b[s] for s in st])
    lo, hi = boot_mean(diff, seed)
    from stats import wilcoxon_paired
    return dict(diff=float(diff.mean()), lo=lo, hi=hi, p_wilcoxon=wilcoxon_paired([a[s] for s in st], [b[s] for s in st]),
                n=len(st), a_lower=int((diff < 0).sum()), b_lower=int((diff > 0).sum()))


def analyze(G):
    dense = C.load_csv(os.path.join(OUT, f"rover_G{G}.csv"))
    held = [r for r in C.load_csv(os.path.join(C.RES, "e3_heldout.csv")) if r["repriced"] == "0" and r["label"] == "main"]
    out = dict(grid=G, rows=len(dense), by_r1={})
    for r1 in R1S:
        dn = [r for r in dense if float(r["r1"]) == r1]
        base = {C.row_key(r): r for r in held if r["policy"] == "rover" and float(r["r1"]) == r1}
        same_dec = sum(1 for r in dn if all(r[k] == base[C.row_key(r)][k] for k in ("audits", "recoveries", "regen", "irr_events")))
        same_loss = sum(1 for r in dn if abs(float(r["loss"]) - float(base[C.row_key(r)]["loss"])) < 1e-9)
        pr = paired(dn, list(base.values()), "loss", seed=7)
        out["by_r1"][str(r1)] = dict(episodes=len(dn), identical_decisions=same_dec, identical_loss=same_loss,
                                     mean_loss_dense=float(np.mean([float(r["loss"]) for r in dn])),
                                     mean_loss_501=float(np.mean([float(r["loss"]) for r in base.values()])),
                                     dense_minus_501=pr,
                                     mean_audits_dense=float(np.mean([float(r["audits"]) for r in dn])),
                                     mean_audits_501=float(np.mean([float(r["audits"]) for r in base.values()])))
    rb = [r for r in held if r["policy"] == "rollback_tuned" and float(r["r1"]) == 1.0]
    na = [r for r in held if r["policy"] == "rover_noaudit" and float(r["r1"]) == 1.0]
    dn1 = [r for r in dense if float(r["r1"]) == 1.0]
    out["H1_rollback_minus_rover_dense"] = paired(rb, dn1, "loss", seed=7)
    out["H2_noaudit_minus_rover_dense_irr_events"] = paired(na, dn1, "irr_events", seed=11)
    json.dump(out, open(os.path.join(OUT, f"rover_G{G}_analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=2001)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--r1", default="0,0.1,1,10")
    ap.add_argument("--tmp", default=os.environ.get("ROV_TMP"))
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.analyze_only:
        base = C.safe_tmp(a.tmp)
        tpl, eps = os.path.join(base, "templates"), os.path.join(base, "episodes")
        os.makedirs(eps, exist_ok=True)
        fams = TB.FAMILIES
        C.build_templates(tpl, {(f, s) for f in fams for s in range(10, 20)})
        best = C.thresholds()
        jobs = []
        for r1 in (float(x) for x in a.r1.split(",")):
            for f in fams:
                for s in range(10, 20):
                    for sd in range(8, 16):
                        jobs.append(dict(family=f, stream=s, seed=sd, policy="rover", r1=r1, label="main",
                                         scenario=dict(rb_threshold=C.th_for(r1, best)), rand_rate=0.0,
                                         template_root=tpl, tmp_base=eps, grid=a.grid))
        C.run_resumable(jobs, os.path.join(OUT, f"rover_G{a.grid}.csv"), a.workers, fn=run_episode_grid,
                        tag=f"rover G={a.grid}")
    analyze(a.grid)


if __name__ == "__main__":
    main()
