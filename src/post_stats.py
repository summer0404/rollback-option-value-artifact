#!/usr/bin/env python3
"""Re-analyses of the committed held-out E3 file. No episode is run.

Estimands. For every comparison in e3_tests.csv (other minus ROVER, per-stream means over the
8 seeds, 30 streams) and for the primary family (H1 loss, H2 irreversible activations):
  mean difference with the paper's cluster bootstrap (seed 7, 2,000 resamples)
  paired t-test and a sign-flip permutation test of the mean difference (200,000 random sign vectors,
    seeded): tests aligned with a claim about the mean, next to the paper's Wilcoxon signed-rank test
  paired win counts and the paired probability of superiority P(ROVER lower on the same stream) + 0.5 P(tie),
    next to the paper's A12, which compares every stream of one policy with every stream of the other
  Holm over H1 and H2 for each test

Adaptive attacker. For every policy at r1 = 1: attacked episodes and streams per carrier (the subset
sizes behind e3_adaptive.csv), and a split-sample estimate: choose the carrier with the largest payoff
s_c * damage - kappa_c on seeds 8-11 and evaluate that carrier on seeds 12-15, and the reverse. The
in-sample maximum over four noisy carrier estimates is biased upward; the split estimate is not.
Output: results/statistics/estimands.csv, primary.json, adaptive_split.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE
RES = os.path.normpath(os.path.join(SRC, "..", "results"))
sys.path.insert(0, SRC)
from stats import wilcoxon_paired, holm, vda  # noqa: E402
from game import CARRIERS  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "results", "exploratory", "statistics"))
NPERM = 200_000


def load():
    rows = [r for r in csv.DictReader(open(os.path.join(RES, "e3_heldout.csv")))]
    for r in rows:
        r["r1"] = float(r["r1"])
        r["sid"] = f"{r['family']}-{r['stream']}"
    return rows


def per_stream(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r["sid"]].append(float(r[key]))
    return {s: float(np.mean(v)) for s, v in d.items()}


def boot_mean(v, seed, B=2000):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    bs = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def signflip_p(diff, seed=2027, n=NPERM, block=20_000):
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    ge = 0
    done = 0
    while done < n:
        k = min(block, n - done)
        s = rng.choice((-1.0, 1.0), size=(k, len(diff)))
        ge += int((np.abs((s * diff).mean(1)) >= obs - 1e-12).sum())
        done += k
    return (ge + 1) / (n + 1)


def compare(a_rows, b_rows, key, seed=7):
    a, b = per_stream(a_rows, key), per_stream(b_rows, key)
    ss = sorted(set(a) & set(b))
    da, db = np.array([a[s] for s in ss]), np.array([b[s] for s in ss])
    diff = da - db
    lo, hi = boot_mean(diff, seed)
    t = st.ttest_1samp(diff, 0.0)
    return dict(n_streams=len(ss), mean_diff=float(diff.mean()), boot_lo=lo, boot_hi=hi,
                sd_diff=float(diff.std(ddof=1)), dz=float(diff.mean() / diff.std(ddof=1)) if diff.std() > 0 else float("nan"),
                p_wilcoxon=wilcoxon_paired(da, db), p_ttest=float(t.pvalue), p_signflip=signflip_p(diff),
                rover_lower=int((diff > 0).sum()), other_lower=int((diff < 0).sum()), ties=int((diff == 0).sum()),
                paired_prob_rover_lower=float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / len(diff)),
                a12_shipped_rover_lower=1 - vda(da, db))


def estimands(rows):
    sel = lambda pol, r1: [r for r in rows if r["label"] == "main" and r["r1"] == r1 and r["policy"] == pol]
    tests = list(csv.DictReader(open(os.path.join(RES, "e3_tests.csv"))))
    out = []
    for t in tests:
        r1 = float(t["r1"])
        c = compare(sel(t["other"], r1), sel("rover", r1), "loss")
        out.append(dict(r1=r1, other=t["other"], metric="loss", p_holm_shipped=float(t["p_holm"]), **c))
    fam = defaultdict(list)
    for i, o in enumerate(out):
        fam[o["r1"]].append(i)
    for r1, idx in fam.items():
        for pk in ("p_ttest", "p_signflip"):
            adj = holm([out[i][pk] for i in idx])
            for i, a in zip(idx, adj):
                out[i][pk + "_holm"] = float(a)
    h1 = compare(sel("rollback_tuned", 1.0), sel("rover", 1.0), "loss")
    h2 = compare(sel("rover_noaudit", 1.0), sel("rover", 1.0), "irr_events", seed=11)
    prim = dict(H1=h1, H2=h2)
    for pk in ("p_wilcoxon", "p_ttest", "p_signflip"):
        adj = holm([h1[pk], h2[pk]])
        prim[f"holm_{pk}"] = dict(H1=float(adj[0]), H2=float(adj[1]))
    return out, prim


def adaptive_split(rows):
    out = []
    pols = ("no_audit", "random_audit", "periodic_review", "rollback_tuned", "reset_all", "immutable_validator", "myopic",
            "rover_irrev_blind", "rover_const", "rover_noaudit", "rover")
    for pol in pols:
        pr = [r for r in rows if r["label"] == "main" and r["r1"] == 1.0 and r["policy"] == pol]
        if not pr:
            continue
        row = dict(policy=pol)
        for c in CARRIERS:
            att = [r for r in pr if r["carrier"] == c.name]
            row[f"n_{c.name}"] = len(att)
            row[f"streams_{c.name}"] = len({r["sid"] for r in att})

        def payoff(rs, c):
            att = [r for r in rs if r["carrier"] == c.name]
            if not att:
                return float("nan")
            dm = per_stream([dict(r, dmg=float(r["irr"]) + float(r["resid"]) + float(r["replay"]) + float(r["lost"]))
                             for r in att], "dmg")
            return c.s * float(np.mean(list(dm.values()))) - c.kappa
        full = {c.name: payoff(pr, c) for c in CARRIERS}
        row["insample_best"] = max(full, key=full.get)
        row["insample_payoff"] = full[row["insample_best"]]
        halves = ([r for r in pr if int(r["seed"]) <= 11], [r for r in pr if int(r["seed"]) >= 12])
        ests = []
        for sel_h, ev_h in ((0, 1), (1, 0)):
            ps = {c.name: payoff(halves[sel_h], c) for c in CARRIERS}
            pick = max(ps, key=ps.get)
            cc = next(c for c in CARRIERS if c.name == pick)
            ests.append((pick, ps[pick], payoff(halves[ev_h], cc)))
        row["split_pick_A"], row["split_select_A"], row["split_eval_A"] = ests[0]
        row["split_pick_B"], row["split_select_B"], row["split_eval_B"] = ests[1]
        row["split_eval_mean"] = float(np.mean([e[2] for e in ests]))
        row["selection_optimism"] = float(np.mean([e[1] - e[2] for e in ests]))
        out.append(row)
    return out


def write(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    est, prim = estimands(rows)
    write(os.path.join(OUT, "estimands.csv"), est)
    json.dump(prim, open(os.path.join(OUT, "primary.json"), "w"), indent=1)
    ad = adaptive_split(rows)
    write(os.path.join(OUT, "adaptive_split.csv"), ad)
    flips = [(e["r1"], e["other"]) for e in est
             if (e["p_holm_shipped"] < 0.05) != (e["p_signflip_holm"] < 0.05)]
    print(json.dumps(dict(primary=prim, tests=len(est), significance_flips_wilcoxon_vs_signflip=flips), indent=1))
    for r in ad:
        print(r["policy"], {k: r[k] for k in ("insample_best", "insample_payoff", "split_eval_mean", "selection_optimism")},
              [r[f"n_{c.name}"] for c in CARRIERS])


if __name__ == "__main__":
    main()
