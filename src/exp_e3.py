"""E3: executed scripted git testbed.

Two modes, following protocol/protocol-E3-heldout.md:

  --mode tune     development streams 0..9 per family, seeds 0..7: rollback-only threshold sweep
                  (0.2..0.8) at r1 = 1 and r1 = 10 -> results/e3_tune.csv, results/e3_thresholds.json
  --mode heldout  fresh streams 10..19 per family, seeds 8..15: main matrix over r1 in {0, 0.1, 1, 10},
                  robustness (late attacker, sensor misspecification) and the gate study at r1 = 1
                  -> results/e3_heldout.csv
  --mode extra    exploratory additions on the same held-out cells, written after the confirmatory run and
                  reported as exploratory (RUNLOG.md): graded insertion shifts (2, 4, 6 milestones), lower
                  attack rates (0.05, 0.2) with the default and with a calibrated prior, a gate-aware ROVER,
                  and rollback only tuned for r1 <= 0.1 by re-pricing the development runs
                  -> results/e3_extra.csv, results/e3_extra_meta.json

Common random numbers: every policy sees the same draws per (stream, seed). Policies whose decisions
and executor do not depend on r1 (for r1 <= w) are executed at r1 = 1 and re-priced at r1 in {0, 0.1}.
The random-audit baseline is budget-matched to ROVER's realised audit rate (second pass).
The development files e3_episodes.csv (v2) and e3_episodes_v1_replayonly_consthazard.csv are
historical records produced by the development version of this script (see RUNLOG.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testbed as TB  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
R1_MAIN = 1.0
R1_SWEEP = (0.0, 0.1, 1.0, 10.0)
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DEV = dict(streams=range(0, 10), seeds=range(0, 8))
HELD = dict(streams=range(10, 20), seeds=range(8, 16))
BASELINES = ("no_audit", "periodic_review", "rollback_tuned", "reset_all", "immutable_validator", "myopic")
ROVERS = ("rover", "rover_irrev_blind", "rover_const", "rover_noaudit")
R1_DEPENDENT = ROVERS + ("myopic", "random_audit")
KEYS = ["label", "family", "stream", "seed", "policy", "r1", "repriced", "attacked", "carrier", "t0", "loss",
        "irr", "resid", "audit", "replay", "lost", "irr_events", "rev_events", "resid_lines", "remote_canary",
        "remote_refs", "audits", "recoveries", "rec_forced", "rec_false", "regen", "cherry", "conflicts",
        "discarded", "rec_seconds", "validations", "reviewed", "present", "n_ms", "clean_preserved",
        "detections", "gate_audits", "gate_blocks", "blocked_benign", "benign_pushes", "carrier_tainted_end",
        "audits_memory", "audits_skill", "audits_queue", "audits_branch", "git_calls", "wall"]


def jobs_for(cells, policy, r1, label, scenario, tpl, eps, rand_rate=0.0):
    return [dict(family=f, stream=s, seed=sd, policy=policy, r1=r1, label=label, scenario=dict(scenario),
                 rand_rate=rand_rate, template_root=tpl, tmp_base=eps) for (f, s, sd) in cells]


def run_jobs(pool, jobs, tag, t0):
    out = []
    for n, r in enumerate(pool.imap_unordered(TB.run_episode, jobs, chunksize=2), 1):
        out.append(r)
        if n % 250 == 0:
            print(f"{tag}: {n}/{len(jobs)} episodes, {time.time() - t0:.0f}s", flush=True)
    return out


def write(path, rows):
    rows.sort(key=lambda r: (r["label"], r["r1"], r["policy"], r["family"], r["stream"], r["seed"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r.get(k, 0):.6f}" if isinstance(r.get(k, 0), float) else r.get(k, 0)) for k in KEYS})


def tune(pool, tpl, eps, t0):
    cells = [(f, s, sd) for f in TB.FAMILIES for s in DEV["streams"] for sd in DEV["seeds"]]
    jobs = []
    for r1 in (1.0, 10.0):
        for th in THRESHOLDS:
            jobs += jobs_for(cells, "rollback_tuned", r1, f"tune{th:.1f}", dict(rb_threshold=th), tpl, eps)
    rows = run_jobs(pool, jobs, "tune", t0)
    for r in rows:
        r["repriced"] = 0
    write(os.path.join(RES, "e3_tune.csv"), rows)
    best = {}
    for r1 in (1.0, 10.0):
        means = {th: float(np.mean([r["loss"] for r in rows if r["r1"] == r1 and r["label"] == f"tune{th:.1f}"]))
                 for th in THRESHOLDS}
        best[str(r1)] = dict(threshold=min(means, key=means.get), means={f"{k:.1f}": v for k, v in means.items()})
    json.dump(best, open(os.path.join(RES, "e3_thresholds.json"), "w"), indent=1)
    print("tuned thresholds:", {k: v["threshold"] for k, v in best.items()}, flush=True)
    return best


def heldout(pool, tpl, eps, t0, best):
    cells = [(f, s, sd) for f in TB.FAMILIES for s in HELD["streams"] for sd in HELD["seeds"]]
    th = lambda r1: best["10.0" if r1 > TB.COST["w_lost"] else "1.0"]["threshold"]
    jobs = []
    for r1 in R1_SWEEP:                       # main matrix
        for pol in BASELINES + ROVERS + (("rover_replay_only",) if r1 > TB.COST["w_lost"] else ()):
            if not (r1 == R1_MAIN or r1 > TB.COST["w_lost"] or pol in R1_DEPENDENT):
                continue
            jobs += jobs_for(cells, pol, r1, "main", dict(rb_threshold=th(r1)), tpl, eps)
    rob = {"late": dict(late=True), "s1low": dict(sense_true=dict(s1=0.25)),
           "dalpha": dict(sense_true=dict(d=0.7, alpha=0.1))}
    for label, sc in rob.items():             # robustness at r1 = 1
        pols = ("rover", "rover_const", "rover_noaudit", "rollback_tuned", "no_audit", "myopic") if label == "late" \
            else ("rover", "rollback_tuned", "no_audit", "myopic")
        for pol in pols:
            jobs += jobs_for(cells, pol, R1_MAIN, label, dict(sc, rb_threshold=th(R1_MAIN)), tpl, eps)
    for nu in (1 / 24, 1 / 4):                # gate study at r1 = 1
        for gamma in (0.5, 1.0):
            label = f"gate_nu{nu:.3f}_g{gamma:.1f}"
            for pol in ("gate", "rover_gate"):
                jobs += jobs_for(cells, pol, R1_MAIN, label, dict(gate=(nu, gamma)), tpl, eps)
    rows = run_jobs(pool, jobs, "heldout pass 1", t0)
    jobs = []                                 # pass 2: budget-matched random audit
    for r1 in R1_SWEEP:
        rov = [r["audits"] for r in rows if r["policy"] == "rover" and r["r1"] == r1 and r["label"] == "main"]
        rate = float(np.mean(rov)) / (TB.T_MS * TB.NC)
        jobs += jobs_for(cells, "random_audit", r1, "main", {}, tpl, eps, rand_rate=rate)
    rows += run_jobs(pool, jobs, "heldout pass 2", t0)
    for r in rows:
        r["repriced"] = 0
    extra = []                                # re-price r1-independent policies at r1 in {0, 0.1}
    for r in rows:
        if r["label"] == "main" and r["r1"] == R1_MAIN and r["policy"] not in R1_DEPENDENT:
            for r1 in (0.0, 0.1):
                r2 = dict(r)
                r2["r1"] = r1
                r2.update(TB.loss_components(r2, r1))
                r2["repriced"] = 1
                extra.append(r2)
    rows += extra
    write(os.path.join(RES, "e3_heldout.csv"), rows)
    print(f"held-out: {len(rows)} rows ({sum(1 - r['repriced'] for r in rows)} executed)", flush=True)


SHIFTS = (2, 4, 6)
RATES = (0.05, 0.2)
SHIFT_POLICIES = ("rover", "rover_const", "rover_noaudit", "rollback_tuned", "no_audit")


def cheap_threshold():
    """Rollback-only threshold for r1 <= 0.1, chosen on the development runs re-priced from r1 = 1."""
    rows = [r for r in csv.DictReader(open(os.path.join(RES, "e3_tune.csv"))) if abs(float(r["r1"]) - 1) < 1e-9]
    out = {}
    for r1 in (0.0, 0.1):
        means = {}
        for th in THRESHOLDS:
            v = []
            for r in rows:
                if r["label"] != f"tune{th:.1f}":
                    continue
                d = {k: (float(x) if k not in ("label", "family", "policy", "carrier") else x) for k, x in r.items()}
                v.append(TB.loss_components(d, r1)["loss"])
            means[f"{th:.1f}"] = float(np.mean(v))
        out[str(r1)] = dict(threshold=float(min(means, key=means.get)), means=means)
    return out


def extra(pool, tpl, eps, t0, best):
    cells = [(f, s, sd) for f in TB.FAMILIES for s in HELD["streams"] for sd in HELD["seeds"]]
    th1 = best["1.0"]["threshold"]
    cheap = cheap_threshold()
    assert cheap["0.0"]["threshold"] == cheap["0.1"]["threshold"]
    jobs = jobs_for(cells, "rollback_cheap", R1_MAIN, "main", dict(rb_threshold=cheap["0.0"]["threshold"]), tpl, eps)
    for sh in SHIFTS:
        for pol in SHIFT_POLICIES:
            jobs += jobs_for(cells, pol, R1_MAIN, f"shift{sh}", dict(shift=sh, rb_threshold=th1), tpl, eps)
    for p in RATES:
        for pol in ("no_audit", "rollback_tuned", "rover", "rover_noaudit"):
            jobs += jobs_for(cells, pol, R1_MAIN, f"rate{p:.2f}_fix", dict(p_attack=p, rb_threshold=th1), tpl, eps)
        for pol in ("rollback_tuned", "rover", "rover_noaudit"):
            jobs += jobs_for(cells, pol, R1_MAIN, f"rate{p:.2f}_cal", dict(p_attack=p, prior_rate=p, rb_threshold=th1),
                             tpl, eps)
    for nu in (1 / 24, 1 / 4):
        for gamma in (0.5, 1.0):
            jobs += jobs_for(cells, "rover_gate_aware", R1_MAIN, f"gate_nu{nu:.3f}_g{gamma:.1f}", dict(gate=(nu, gamma)),
                             tpl, eps)
    rows = run_jobs(pool, jobs, "extra", t0)
    for r in rows:
        r["repriced"] = 0
    more = []
    for r in rows:
        if r["policy"] == "rollback_cheap":
            for r1 in (0.0, 0.1):
                r2 = dict(r)
                r2["r1"] = r1
                r2.update(TB.loss_components(r2, r1))
                r2["repriced"] = 1
                more.append(r2)
    rows += more
    write(os.path.join(RES, "e3_extra.csv"), rows)
    json.dump(dict(cheap_threshold=cheap, shifts=SHIFTS, rates=RATES, shift_policies=SHIFT_POLICIES,
                   executed=sum(1 - r["repriced"] for r in rows)), open(os.path.join(RES, "e3_extra_meta.json"), "w"),
              indent=1)
    print(f"extra: {len(rows)} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("tune", "heldout", "all", "extra"), default="all")
    ap.add_argument("--workers", type=int, default=max(1, min(14, (os.cpu_count() or 2) - 2)))
    a = ap.parse_args()
    base = tempfile.mkdtemp(prefix="rov-e3-", dir=os.environ.get("ROV_TMP") or None)
    tpl = os.path.join(base, "templates")
    eps = os.path.join(base, "episodes")
    os.makedirs(eps)
    t0 = time.time()
    try:
        for fam in TB.FAMILIES:
            for s in list(DEV["streams"]) + list(HELD["streams"]):
                dst = os.path.join(tpl, f"{fam}-{s}")
                os.makedirs(dst)
                TB.build_fixture(dst, fam, s)
        with Pool(a.workers) as pool:
            if a.mode in ("tune", "all"):
                best = tune(pool, tpl, eps, t0)
            else:
                best = json.load(open(os.path.join(RES, "e3_thresholds.json")))
            if a.mode in ("heldout", "all"):
                heldout(pool, tpl, eps, t0, best)
            if a.mode == "extra":
                extra(pool, tpl, eps, t0, best)
        print(f"E3 done in {time.time() - t0:.0f}s", flush=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
