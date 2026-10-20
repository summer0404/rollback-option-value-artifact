#!/usr/bin/env python3
"""Charge ordinary releases that publish contaminated work.

In the testbed a benign push (Episode.benign_push) publishes the current branch to the bare
remote. Episode.measure counts canary occurrences summed over remote refs (remote_canary), and
loss_components never charges them. This exploratory run adds releases to the main held-out
condition for policies without a gate, measures unique leaked content, and re-prices exactly.

Releases: scenario gate=(nu, 1.0) with nu in {1/24, 1/4}. For a policy outside testbed.GATING this only
adds benign pushes (Episode.gating is False, activations are not gated, ROVER plans with the main
tables), so every decision count must equal the committed main row at r1 = 1. The script checks this.

Measured per episode (EpisodePub.measure, added on top of Episode.measure):
  rel_lines_unique   distinct canary lines over all benign release refs (refs/releases/b*)
  rel_lines_new      of those, lines absent from every activation ref (refs/releases/r*), i.e. not
                     already published by an irreversible activation, which the loss charges at H
  rel_contaminated   benign releases that contain at least one canary line
  act_lines_unique   distinct canary lines over activation refs
Re-pricing (decisions do not depend on these prices, so it is exact):
  per line     loss + h_pub * rel_lines_new,     h_pub in {1, 2, 5, 20}
  per release  loss + H_rel * rel_contaminated,  H_rel in {1, 5, 20}
Post hoc and exploratory: not part of the confirmatory protocol.

  python3 -B post_publication.py --workers 3 --tmp <scratch dir>
  python3 -B post_publication.py --analyze-only
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
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exploratory", "publication"))
POLICIES = ("no_audit", "random_audit", "periodic_review", "rollback_tuned", "reset_all", "immutable_validator",
            "myopic", "rover_noaudit", "rover")
NUS = (("nu0.042", 1 / 24), ("nu0.250", 1 / 4))
H_PUB = (1.0, 2.0, 5.0, 20.0)
H_REL = (1.0, 5.0, 20.0)
EXTRA_KEYS = ["rel_lines_unique", "rel_lines_new", "rel_contaminated", "act_lines_unique", "n_release_refs",
              "n_activation_refs"]
DECISION_KEYS = ("audits", "recoveries", "rec_forced", "rec_false", "regen", "cherry", "conflicts", "discarded",
                 "irr_events", "rev_events", "resid_lines", "present", "detections", "validations", "reviewed",
                 "audits_memory", "audits_skill", "audits_queue", "audits_branch", "loss")


class EpisodePub(TB.Episode):
    def measure(self):
        m = super().measure()
        g = self.g
        refs = g.run("--git-dir", g.remote, "for-each-ref", "--format=%(refname)", cwd=g.root).stdout.split()
        rel, act = set(), set()
        n_rel = n_act = contaminated = 0
        for ref in refs:
            out = g.run("--git-dir", g.remote, "grep", "-h", "-F", self.tag, ref, "--", "src", "tests",
                        cwd=g.root, check=False).stdout
            lines = {ln.strip() for ln in out.splitlines() if self.tag in ln}
            if ref.startswith("refs/releases/b"):
                n_rel += 1
                rel |= lines
                contaminated += int(bool(lines))
            else:
                n_act += 1
                act |= lines
        m.update(rel_lines_unique=len(rel), rel_lines_new=len(rel - act), rel_contaminated=contaminated,
                 act_lines_unique=len(act), n_release_refs=n_rel, n_activation_refs=n_act)
        return m


def run_episode_pub(job):
    """testbed.run_episode with EpisodePub."""
    root = tempfile.mkdtemp(prefix="rov-", dir=job["tmp_base"])
    try:
        shutil.copytree(os.path.join(job["template_root"], f"{job['family']}-{job['stream']}"), root, dirs_exist_ok=True)
        ep = EpisodePub(root, job["family"], job["stream"], job["seed"], job["policy"], job["r1"],
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


def boot_mean(v, seed, B=2000):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    bs = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def analyze():
    rows = C.load_csv(os.path.join(OUT, "publication.csv"))
    held = {C.row_key(r): r for r in C.load_csv(os.path.join(C.RES, "e3_heldout.csv"))
            if r["label"] == "main" and float(r["r1"]) == 1.0 and r["repriced"] == "0"}
    mism = defaultdict(int)
    for r in rows:
        base = held[("main",) + C.row_key(r)[1:]]
        for k in DECISION_KEYS:
            if abs(float(r[k]) - float(base[k])) > 1e-9:
                mism[k] += 1
    out = dict(episodes=len(rows), decisions_match_committed_main=dict(mismatches=dict(mism), ok=not mism), by_nu={})
    from stats import wilcoxon_paired
    for lab, nu in NUS:
        rr = [r for r in rows if r["label"] == f"pub_{lab}"]
        by_pol = {p: [r for r in rr if r["policy"] == p] for p in POLICIES}
        pricings = [("base", lambda r: float(r["loss"]))]
        pricings += [(f"line_h{h:g}", (lambda h: lambda r: float(r["loss"]) + h * float(r["rel_lines_new"]))(h)) for h in H_PUB]
        pricings += [(f"release_H{h:g}", (lambda h: lambda r: float(r["loss"]) + h * float(r["rel_contaminated"]))(h)) for h in H_REL]
        res = dict(nu=nu, leak={}, pricing={})
        for p, rs in by_pol.items():
            res["leak"][p] = dict(rel_lines_new=float(np.mean([float(r["rel_lines_new"]) for r in rs])),
                                  rel_lines_unique=float(np.mean([float(r["rel_lines_unique"]) for r in rs])),
                                  rel_contaminated=float(np.mean([float(r["rel_contaminated"]) for r in rs])),
                                  share_episodes_leaking=float(np.mean([float(r["rel_contaminated"]) > 0 for r in rs])),
                                  remote_canary_occurrences=float(np.mean([float(r["remote_canary"]) for r in rs])),
                                  benign_pushes=float(np.mean([float(r["benign_pushes"]) for r in rs])))
        for name, fn in pricings:
            ps = {}
            for p, rs in by_pol.items():
                d = defaultdict(list)
                for r in rs:
                    d[f"{r['family']}-{r['stream']}"].append(fn(r))
                ps[p] = {s: float(np.mean(v)) for s, v in d.items()}
            streams = sorted(ps["rover"])
            rov = np.array([ps["rover"][s] for s in streams])
            table = {}
            for p in POLICIES:
                v = np.array([ps[p][s] for s in streams])
                lo, hi = boot_mean(v, 9)
                row = dict(mean=float(v.mean()), lo=lo, hi=hi)
                if p != "rover":
                    diff = v - rov
                    dlo, dhi = boot_mean(diff, 7)
                    row.update(d_other_minus_rover=float(diff.mean()), d_lo=dlo, d_hi=dhi,
                               p_wilcoxon_raw=wilcoxon_paired(v, rov), rover_lower=int((diff > 0).sum()),
                               other_lower=int((diff < 0).sum()))
                table[p] = row
            ranking = sorted(table, key=lambda p: table[p]["mean"])
            res["pricing"][name] = dict(table=table, ranking=ranking, rover_rank=ranking.index("rover") + 1)
        out["by_nu"][lab] = res
    json.dump(out, open(os.path.join(OUT, "publication_analysis.json"), "w"), indent=1)
    print(json.dumps(dict(episodes=out["episodes"], decisions_match=out["decisions_match_committed_main"]), indent=1))
    for lab, res in out["by_nu"].items():
        print(f"--- {lab}: leak per episode (rel_lines_new) " +
              ", ".join(f"{p}={v['rel_lines_new']:.2f}" for p, v in res["leak"].items()))
        for name, pr in res["pricing"].items():
            print(f"  {name:12s} rover rank {pr['rover_rank']}/{len(POLICIES)}: " +
                  ", ".join(f"{p}={pr['table'][p]['mean']:.2f}" for p in pr["ranking"][:4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tmp", default=os.environ.get("ROV_TMP"))
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.analyze_only:
        base = C.safe_tmp(a.tmp)
        tpl, eps = os.path.join(base, "templates"), os.path.join(base, "episodes")
        os.makedirs(eps, exist_ok=True)
        C.build_templates(tpl, {(f, s) for f in TB.FAMILIES for s in range(10, 20)})
        th1 = C.th_for(1.0)
        jobs = []
        for lab, nu in NUS:
            for pol in a.policies.split(","):
                rate = C.rand_rate(1.0) if pol == "random_audit" else 0.0
                for f in TB.FAMILIES:
                    for s in range(10, 20):
                        for sd in range(8, 16):
                            jobs.append(dict(family=f, stream=s, seed=sd, policy=pol, r1=1.0, label=f"pub_{lab}",
                                             scenario=dict(gate=(nu, 1.0), rb_threshold=th1), rand_rate=rate,
                                             template_root=tpl, tmp_base=eps))
        C.run_resumable(jobs, os.path.join(OUT, "publication.csv"), a.workers, fn=run_episode_pub,
                        keys=C.KEYS + EXTRA_KEYS, tag="publication")
    analyze()


if __name__ == "__main__":
    main()
