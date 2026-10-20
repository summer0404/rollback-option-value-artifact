#!/usr/bin/env python3
"""Quick reproducibility check of the executed testbed (a few minutes instead of hours).

Re-executes a fixed sample of committed held-out E3 episodes with real git operations and compares
each re-executed episode with its committed row in results/e3_heldout.csv: loss, loss components and
decision counts must match exactly (wall-clock timings are not compared). The sample is the main
condition at r1 = 1, every policy, and the first --cells (family, stream, seed) cells in a fixed order
that cycles over the three fixture families. Nothing in results/ is written.

  python3 -B check_reproduce.py --tmp <scratch dir outside any git repository>
  python3 -B check_reproduce.py --cells 6 --workers 4 --tmp <scratch dir>
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import post_common as C
import testbed as TB

EXACT = ("loss", "irr", "resid", "audit", "replay", "lost", "irr_events", "resid_lines", "audits", "recoveries",
         "rec_forced", "rec_false", "regen", "cherry", "conflicts", "discarded", "validations", "reviewed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=3, help="number of (family, stream, seed) cells, default 3")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--tmp", default=os.environ.get("ROV_TMP"))
    a = ap.parse_args()
    base = C.safe_tmp(a.tmp)

    rows = [r for r in C.load_csv(os.path.join(C.RES, "e3_heldout.csv"))
            if r["label"] == "main" and float(r["r1"]) == 1.0 and r["repriced"] == "0"]
    committed = {C.row_key(r): r for r in rows}
    cells = []
    for k in range(10 * 8):                       # streams 10-19, seeds 8-15, families interleaved
        for fam in TB.FAMILIES:
            cell = (fam, 10 + k % 10, 8 + k // 10)
            if cell not in cells:
                cells.append(cell)
    cells = cells[:a.cells]
    policies = sorted({r["policy"] for r in rows})

    tpl, eps = os.path.join(base, "templates"), os.path.join(base, "episodes")
    os.makedirs(eps, exist_ok=True)
    C.build_templates(tpl, {(f, s) for f, s, _ in cells})
    jobs = []
    for fam, s, sd in cells:
        for pol in policies:
            sc, rate = C.scenario("heldout", "main", pol, 1.0)
            jobs.append(dict(family=fam, stream=s, seed=sd, policy=pol, r1=1.0, label="main", scenario=sc,
                             rand_rate=rate, template_root=tpl, tmp_base=eps))
    print(f"re-executing {len(jobs)} committed episodes ({len(cells)} cells x {len(policies)} policies), "
          f"{a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        out = pool.map(TB.run_episode, jobs, chunksize=1)

    bad = 0
    for job, m in zip(jobs, out):
        ref = committed[C.job_key(job)]
        diffs = [k for k in EXACT if abs(float(m[k]) - float(ref[k])) > 1e-6]
        bad += bool(diffs)
        status = "ok" if not diffs else "DIFFERS in " + ", ".join(f"{k} ({m[k]} vs {ref[k]})" for k in diffs)
        print(f"{job['family']:>6} stream {job['stream']} seed {job['seed']}  {job['policy']:<20} "
              f"loss {float(m['loss']):7.2f}  {status}")
    print(f"{len(jobs) - bad} of {len(jobs)} episodes identical to results/e3_heldout.csv")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
