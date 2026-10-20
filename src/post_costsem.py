#!/usr/bin/env python3
"""How well do the planner's cost quantities predict what the testbed executes?

ROVER plans per carrier with derived work D = testbed.Episode.D(i) (commits since the carrier's
checkpoint that read it, directly or transitively), a recovery price r0 + min(r1, w) D, Bernoulli growth
with the read probability Delta_i, and a terminal residual cost h * P_INFL * D * b. The executed
testbed instead re-executes dependent commits plus independent ones whose cherry-pick conflicts, and
the loss counts canary lines that survive in the final tree. An instrumented Episode subclass logs,
without changing any decision (checked against the committed rows):

  per recovery  D the planner saw, re-executed commits (regen), conflicts, cherry-picks, discards, the
                planner's price r0 + r1e*D versus the executed price r0 + r1*regen + w*discards
  per carrier   at the end of the episode: D, the direct reads since insertion, taint, belief,
                surviving canary lines, and the planner's terminal prediction h*P_INFL*D*b
  per milestone mean D_i(t) against the planner's growth model t*Delta_i (no recovery yet)

Policies rover and rollback_tuned, r1 in {0.1, 1}, all held-out cells (960 episodes). Post hoc.
  python3 -B post_costsem.py --workers 3 --tmp <scratch dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import post_common as C  # noqa: E402

TB = C.TB
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exploratory", "cost_semantics"))
POLICIES = ("rover", "rollback_tuned")
R1S = (0.1, 1.0)
DECISION_KEYS = ("audits", "recoveries", "rec_forced", "rec_false", "regen", "cherry", "conflicts", "discarded",
                 "irr_events", "resid_lines", "present", "loss")


class EpisodeC11(TB.Episode):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.c11_rec = []
        self.c11_D = []            # (k, i, D) before each decision while no recovery of i has happened
        self.c11_recovered = [False] * TB.NC
        self.c11_insert_k = None

    def decide(self, k):
        for i in range(TB.NC):
            if not self.c11_recovered[i]:
                self.c11_D.append((k, i, self.D(i)))
        return super().decide(k)

    def attack(self, i):
        self.c11_insert_k = self.k
        return super().attack(i)

    def recover(self, i, forced=False):
        D = self.D(i)
        before = {k: self.m[k] for k in ("regen", "conflicts", "cherry", "discarded")}
        tainted = self.taint[i]
        super().recover(i, forced=forced)
        d = {k: self.m[k] - before[k] for k in before}
        r1e = TB.effective_r1(self.r1, self.policy == "rover_replay_only")
        self.c11_rec.append(dict(k=self.k, carrier=i, forced=bool(forced), tainted=bool(tainted), D=D,
                                 regen=d["regen"], conflicts=d["conflicts"], cherry=d["cherry"], discarded=d["discarded"],
                                 planned=TB.COST["r0"] + r1e * D,
                                 executed=TB.COST["r0"] + self.r1 * d["regen"] + TB.COST["w_lost"] * d["discarded"]))
        self.c11_recovered[i] = True

    def measure(self):
        m = super().measure()
        ends = []
        h_model = TB.COST["h"] * TB.P_INFL
        for i in range(TB.NC):
            D = self.D(i)
            direct_after = 0
            if self.taint[i] and self.c11_insert_k is not None:
                direct_after = sum(1 for e in self.log[self.ckpt[i]:]
                                   if e["kind"] == "ms" and e["reads"][i] and e["k"] > self.c11_insert_k)
            ends.append(dict(carrier=i, D=D, tainted=bool(self.taint[i]), belief=float(self.belief[i]),
                             direct_reads_after_insertion=direct_after,
                             predicted_resid=h_model * D * float(self.belief[i]),
                             predicted_resid_if_tainted=h_model * D))
        m["c11_json"] = json.dumps(dict(rec=self.c11_rec, end=ends, D=self.c11_D))
        return m


def run_episode_c11(job):
    root = tempfile.mkdtemp(prefix="rov-", dir=job["tmp_base"])
    try:
        shutil.copytree(os.path.join(job["template_root"], f"{job['family']}-{job['stream']}"), root, dirs_exist_ok=True)
        ep = EpisodeC11(root, job["family"], job["stream"], job["seed"], job["policy"], job["r1"],
                        job.get("rand_rate", 0.0), job.get("scenario"))
        t0 = time.perf_counter()
        m = ep.run()
        m["wall"] = time.perf_counter() - t0
        d = ep.dr
        m.update(family=job["family"], stream=job["stream"], seed=job["seed"], policy=job["policy"], r1=job["r1"],
                 label=job["label"], attacked=int(d["attacked"]),
                 carrier=TB.CARRIER_NAMES[d["carrier"]] if d["attacked"] else "none",
                 t0=d["t0"] if d["attacked"] else -1)
        m.update(TB.loss_components(m, job["r1"]))
        return m
    finally:
        shutil.rmtree(root, ignore_errors=True)


def analyze():
    rows = C.load_csv(os.path.join(OUT, "cost_semantics.csv"))
    held = {C.row_key(r): r for r in C.load_csv(os.path.join(C.RES, "e3_heldout.csv")) if r["label"] == "main"}
    mism = defaultdict(int)
    for r in rows:
        base = held[("main",) + C.row_key(r)[1:]]
        for k in DECISION_KEYS:
            if abs(float(r[k]) - float(base[k])) > 1e-9:
                mism[k] += 1
    out = dict(episodes=len(rows), decisions_match_committed=dict(ok=not mism, mismatches=dict(mism)), cells={})
    growth = defaultdict(list)
    for pol in POLICIES:
        for r1 in R1S:
            rs = [r for r in rows if r["policy"] == pol and float(r["r1"]) == r1]
            recs = [x for r in rs for x in json.loads(r["c11_json"])["rec"]]
            ends = [(r, x) for r in rs for x in json.loads(r["c11_json"])["end"]]
            for r in rs:
                for (k, i, D) in json.loads(r["c11_json"])["D"]:
                    growth[(i, k)].append(D)
            pl = np.array([x["planned"] for x in recs])
            ex = np.array([x["executed"] for x in recs])
            Ds = np.array([x["D"] for x in recs])
            rg = np.array([x["regen"] for x in recs])
            cf = np.array([x["conflicts"] for x in recs])
            tainted_end = [(r, x) for r, x in ends if x["tainted"]]
            pred = np.array([x["predicted_resid_if_tainted"] for _, x in tainted_end])
            pred_b = np.array([x["predicted_resid"] for _, x in ends])
            # canary lines are per episode; attribute them to the single tainted carrier at the end
            resid_by_ep = {C.row_key(r): float(r["resid_lines"]) for r in rs}
            act = np.array([resid_by_ep[C.row_key(r)] for r, _ in tainted_end])
            direct = np.array([TB.COST["h"] * TB.P_INFL * x["direct_reads_after_insertion"] for _, x in tainted_end])
            Dend = np.array([x["D"] for _, x in tainted_end])
            out["cells"][f"{pol}@r1={r1}"] = dict(
                episodes=len(rs), recoveries=len(recs),
                planned_cost_mean=float(pl.mean()) if len(pl) else None,
                executed_cost_mean=float(ex.mean()) if len(ex) else None,
                cost_error_mean=float((ex - pl).mean()) if len(pl) else None,
                cost_abs_error_mean=float(np.abs(ex - pl).mean()) if len(pl) else None,
                share_recoveries_cost_differs=float(np.mean(np.abs(ex - pl) > 1e-9)) if len(pl) else None,
                D_mean=float(Ds.mean()) if len(Ds) else None, regen_mean=float(rg.mean()) if len(rg) else None,
                regen_minus_D_mean=float((rg - Ds).mean()) if len(Ds) else None,
                conflicts_per_recovery=float(cf.mean()) if len(cf) else None,
                tainted_at_end=len(tainted_end),
                resid_predicted_if_tainted_mean=float(pred.mean()) if len(pred) else None,
                resid_actual_mean=float(act.mean()) if len(act) else None,
                resid_ratio_actual_over_predicted=float(act.sum() / pred.sum()) if len(pred) and pred.sum() > 0 else None,
                resid_predicted_from_direct_reads_mean=float(direct.mean()) if len(direct) else None,
                resid_ratio_actual_over_direct=float(act.sum() / direct.sum()) if len(direct) and direct.sum() > 0 else None,
                D_end_tainted_mean=float(Dend.mean()) if len(Dend) else None,
                belief_weighted_resid_predicted_per_episode=float(pred_b.sum() / max(len(rs), 1)),
                resid_actual_per_episode=float(np.mean([float(r["resid_lines"]) for r in rs])))
    out["growth_D_vs_model"] = {TB.CARRIER_NAMES[i]: {str(k): dict(mean_D=float(np.mean(growth[(i, k)])),
                                                                  model=float((k - 1) * TB.CARRIERS[i].Delta),
                                                                  n=len(growth[(i, k)]))
                                                     for k in (5, 9, 13, 17, 21, 24) if growth[(i, k)]}
                                for i in range(TB.NC)}
    json.dump(out, open(os.path.join(OUT, "cost_semantics_analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tmp", default=os.environ.get("ROV_TMP"))
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.analyze_only:
        base = C.safe_tmp(a.tmp)
        tpl, eps = os.path.join(base, "templates"), os.path.join(base, "episodes")
        os.makedirs(eps, exist_ok=True)
        C.build_templates(tpl, {(f, s) for f in TB.FAMILIES for s in range(10, 20)})
        jobs = []
        for pol in POLICIES:
            for r1 in R1S:
                for f in TB.FAMILIES:
                    for s in range(10, 20):
                        for sd in range(8, 16):
                            jobs.append(dict(family=f, stream=s, seed=sd, policy=pol, r1=r1, label="c11",
                                             scenario=dict(rb_threshold=C.th_for(r1)), rand_rate=0.0,
                                             template_root=tpl, tmp_base=eps))
        C.run_resumable(jobs, os.path.join(OUT, "cost_semantics.csv"), a.workers, fn=run_episode_c11,
                        keys=C.KEYS + ["c11_json"], tag="c11")
    analyze()


if __name__ == "__main__":
    main()
