"""Shared helpers for the exploratory E3 testbed runs (post_*.py).

Imports the committed testbed read-only (artifact/src is never written to; run Python with -B or
PYTHONDONTWRITEBYTECODE=1). Rebuilds the scenario of every committed E3 cell exactly as
artifact/src/exp_e3.py does, builds fixture templates, and runs episodes in a process pool while
appending each finished row to a CSV, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import testbed as TB  # noqa: E402

KEYS = ["label", "family", "stream", "seed", "policy", "r1", "repriced", "attacked", "carrier", "t0", "loss",
        "irr", "resid", "audit", "replay", "lost", "irr_events", "rev_events", "resid_lines", "remote_canary",
        "remote_refs", "audits", "recoveries", "rec_forced", "rec_false", "regen", "cherry", "conflicts",
        "discarded", "rec_seconds", "validations", "reviewed", "present", "n_ms", "clean_preserved",
        "detections", "gate_audits", "gate_blocks", "blocked_benign", "benign_pushes", "carrier_tainted_end",
        "audits_memory", "audits_skill", "audits_queue", "audits_branch", "git_calls", "wall"]
TIMING = ("wall", "rec_seconds")
NU = {"0.042": 1 / 24, "0.250": 1 / 4}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def thresholds():
    return json.load(open(os.path.join(RES, "e3_thresholds.json")))


def th_for(r1, best=None):
    best = best or thresholds()
    return best["10.0" if r1 > TB.COST["w_lost"] else "1.0"]["threshold"]


def cheap_threshold():
    return json.load(open(os.path.join(RES, "e3_extra_meta.json")))["cheap_threshold"]["0.0"]["threshold"]


_RATE = {}


def rand_rate(r1):
    """Budget-matched random-audit rate, computed as exp_e3.heldout pass 2 does, from the committed ROVER rows."""
    if r1 not in _RATE:
        rows = [r for r in load_csv(os.path.join(RES, "e3_heldout.csv"))
                if r["policy"] == "rover" and float(r["r1"]) == r1 and r["label"] == "main" and r["repriced"] == "0"]
        _RATE[r1] = float(np.mean([int(r["audits"]) for r in rows])) / (TB.T_MS * TB.NC)
    return _RATE[r1]


def scenario(source, label, policy, r1):
    """Scenario dict and random-audit rate of a committed cell, mirroring exp_e3.py."""
    if source == "tune":
        return dict(rb_threshold=float(label.replace("tune", ""))), 0.0
    if label.startswith("gate_nu"):
        nu = NU[label.split("_nu")[1].split("_g")[0]]
        gamma = float(label.split("_g")[-1])
        return dict(gate=(nu, gamma)), 0.0
    if source == "heldout":
        if policy == "random_audit":
            return {}, rand_rate(r1)
        th = th_for(r1)
        sc = {"main": {}, "late": dict(late=True), "s1low": dict(sense_true=dict(s1=0.25)),
              "dalpha": dict(sense_true=dict(d=0.7, alpha=0.1))}[label]
        return dict(sc, rb_threshold=th), 0.0
    if source == "extra":
        th1 = th_for(1.0)
        if label == "main" and policy == "rollback_cheap":
            return dict(rb_threshold=cheap_threshold()), 0.0
        if label.startswith("shift"):
            return dict(shift=int(label[5:]), rb_threshold=th1), 0.0
        if label.startswith("rate"):
            p = float(label[4:8])
            sc = dict(p_attack=p, rb_threshold=th1)
            if label.endswith("_cal"):
                sc["prior_rate"] = p
            return sc, 0.0
    raise ValueError((source, label, policy, r1))


def safe_tmp(path):
    """Parent directory for temporary repositories: required, and never inside a git work tree."""
    if not path:
        sys.exit("pass --tmp (or set ROV_TMP) to a directory outside any git repository")
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    probe = path
    while True:
        if os.path.exists(os.path.join(probe, ".git")):
            sys.exit(f"--tmp {path} is inside the git work tree {probe}; choose a scratch directory")
        parent = os.path.dirname(probe)
        if parent == probe:
            return path
        probe = parent


def build_templates(tpl_dir, cells):
    """Fixture repositories for the (family, stream) pairs in cells (built once, reused read-only)."""
    os.makedirs(tpl_dir, exist_ok=True)
    for fam, s in sorted(cells):
        dst = os.path.join(tpl_dir, f"{fam}-{s}")
        if not os.path.exists(dst):
            os.makedirs(dst)
            TB.build_fixture(dst, fam, s)


def job_key(j):
    return (j["label"], j["family"], int(j["stream"]), int(j["seed"]), j["policy"], float(j["r1"]))


def row_key(r):
    return (r["label"], r["family"], int(r["stream"]), int(r["seed"]), r["policy"], float(r["r1"]))


def run_resumable(jobs, out_csv, workers, fn=None, keys=KEYS, progress_every=25, tag="run"):
    """Run jobs not yet present in out_csv; append each finished row immediately."""
    fn = fn or TB.run_episode
    done = set()
    if os.path.exists(out_csv):
        done = {row_key(r) for r in load_csv(out_csv)}
    todo = [j for j in jobs if job_key(j) not in done]
    new = not os.path.exists(out_csv)
    t0 = time.time()
    print(f"{tag}: {len(jobs)} jobs, {len(done)} already done, {len(todo)} to run, {workers} workers", flush=True)
    if not todo:
        return 0
    with open(out_csv, "a", newline="") as f, Pool(workers) as pool:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if new:
            w.writeheader()
        for n, r in enumerate(pool.imap_unordered(fn, todo, chunksize=1), 1):
            r.setdefault("repriced", 0)
            w.writerow({k: (f"{r.get(k, 0):.6f}" if isinstance(r.get(k, 0), float) else r.get(k, 0)) for k in keys})
            f.flush()
            if n % progress_every == 0 or n == len(todo):
                el = time.time() - t0
                print(f"{tag}: {n}/{len(todo)} episodes, {el:.0f}s, {el / n:.2f}s per episode (wall)", flush=True)
    return len(todo)
