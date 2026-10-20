"""Analysis of the held-out E3 run (confirmatory, protocol/protocol-E3-heldout.md).

Unit: per-stream mean over seeds (30 clusters). Paired Wilcoxon signed-rank on per-stream means,
Holm over all comparisons against ROVER at the same r1 (family stated in e3_tests.csv). Intervals:
cluster bootstrap over streams (2,000 resamples) of means and of paired differences.
Outputs: e3_main.csv, e3_components.csv, e3_sweep.csv, e3_tests.csv, e3_robust.csv, e3_gate.csv,
e3_carrier.csv, e3_adaptive.csv, e3_alloc.csv, e3_family.csv.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import wilcoxon_paired, holm, vda  # noqa: E402
from game import CARRIERS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
BASE = ("no_audit", "random_audit", "periodic_review", "rollback_tuned", "reset_all", "immutable_validator", "myopic")
ABL = ("rover_irrev_blind", "rover_const", "rover_noaudit", "rover_replay_only")
ORDER = BASE + ABL + ("rover",)
COMP = ("loss", "irr", "resid", "audit", "replay", "lost")
NUM = ("loss", "irr", "resid", "audit", "replay", "lost", "irr_events", "rev_events", "resid_lines", "remote_canary",
       "audits", "recoveries", "rec_forced", "rec_false", "regen", "cherry", "conflicts", "discarded", "rec_seconds",
       "clean_preserved", "git_calls", "attacked", "gate_audits", "gate_blocks", "blocked_benign", "benign_pushes",
       "audits_memory", "audits_skill", "audits_queue", "audits_branch")
B = 2000


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in NUM:
            r[k] = float(r.get(k) or 0)
        r["r1"] = float(r["r1"])
        r["sid"] = f"{r['family']}-{r['stream']}"
    return rows


def write(name, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def per_stream(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r["sid"]].append(r[key])
    return {s: float(np.mean(v)) for s, v in d.items()}


def boot_mean(vals, seed):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    bs = v[idx].mean(1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def paired(rows_a, rows_b, key, seed=0):
    """Per-stream mean of a minus b, with cluster-bootstrap CI and Wilcoxon p."""
    a, b = per_stream(rows_a, key), per_stream(rows_b, key)
    streams = sorted(set(a) & set(b))
    da, db = np.array([a[s] for s in streams]), np.array([b[s] for s in streams])
    diff = da - db
    lo, hi = boot_mean(diff, seed)
    return dict(diff=float(diff.mean()), lo=lo, hi=hi, p=wilcoxon_paired(da, db), a12=vda(da, db),
                a_lower=int((diff < 0).sum()), b_lower=int((diff > 0).sum()), n=len(streams))


def summary(rows, seed=0):
    out = dict(n=len(rows))
    for k in NUM:
        out[k] = float(np.mean([r[k] for r in rows]))
    lo, hi = boot_mean(list(per_stream(rows, "loss").values()), seed)
    out["loss_lo"], out["loss_hi"] = lo, hi
    rec = sum(r["recoveries"] for r in rows)
    out["ms_per_recovery"] = 1000 * sum(r["rec_seconds"] for r in rows) / rec if rec else 0.0
    out["irr_per100"] = 100 * out["irr_events"]
    return out


def main():
    rows = load(os.path.join(RES, "e3_heldout.csv"))

    def sel(label, r1, pol):
        return [r for r in rows if r["label"] == label and r["r1"] == r1 and r["policy"] == pol]

    main_rows, comp_rows, sweep_rows, test_rows = [], [], [], []
    for r1 in sorted({r["r1"] for r in rows if r["label"] == "main"}):
        pols = [p for p in ORDER if sel("main", r1, p)]
        rov = sel("main", r1, "rover")
        tests = []
        for p in pols:
            s = summary(sel("main", r1, p), seed=int(10 * r1) + len(p))
            row = dict(r1=r1, policy=p, **s)
            if p != "rover":
                pr = paired(sel("main", r1, p), rov, "loss", seed=7)
                row.update(d_loss=pr["diff"], d_lo=pr["lo"], d_hi=pr["hi"], p_raw=pr["p"],
                           a12_rover=1 - pr["a12"], rover_lower=pr["b_lower"], other_lower=pr["a_lower"])
                tests.append((p, pr["p"]))
            sweep_rows.append(row)
        adj = holm([t[1] for t in tests])
        for (p, praw), pa in zip(tests, adj):
            for row in sweep_rows:
                if row["r1"] == r1 and row["policy"] == p:
                    row["p_holm"] = float(pa)
            test_rows.append(dict(r1=r1, other=p, p_raw=praw, p_holm=float(pa),
                                  family=f"{len(tests)} comparisons vs ROVER at r1={r1:g}"))
    main_rows = [r for r in sweep_rows if r["r1"] == 1.0]
    rov = sel("main", 1.0, "rover")
    for p in ("rollback_tuned", "rover_noaudit", "rover_irrev_blind", "rover_const", "no_audit", "myopic"):
        other = sel("main", 1.0, p)
        for k in COMP + ("irr_events", "rec_false", "audits"):
            pr = paired(other, rov, k, seed=11)
            comp_rows.append(dict(other=p, component=k, diff=pr["diff"], lo=pr["lo"], hi=pr["hi"], p_raw=pr["p"],
                                  rover_lower=pr["b_lower"], other_lower=pr["a_lower"]))
    rob_rows = []
    for label in ("late", "s1low", "dalpha"):
        rr = [r for r in rows if r["label"] == label]
        rovl = [r for r in rr if r["policy"] == "rover"]
        for p in sorted({r["policy"] for r in rr}):
            pp = [r for r in rr if r["policy"] == p]
            s = summary(pp, seed=3)
            row = dict(label=label, policy=p, loss=s["loss"], loss_lo=s["loss_lo"], loss_hi=s["loss_hi"],
                       irr=s["irr"], irr_per100=s["irr_per100"], audits=s["audits"])
            if p != "rover":
                pr = paired(pp, rovl, "loss", seed=5)
                pe = paired(pp, rovl, "irr_events", seed=5)
                row.update(d_loss=pr["diff"], d_lo=pr["lo"], d_hi=pr["hi"], p_raw=pr["p"],
                           rover_lower=pr["b_lower"], other_lower=pr["a_lower"],
                           d_irr100=100 * pe["diff"], d_irr100_lo=100 * pe["lo"], d_irr100_hi=100 * pe["hi"],
                           p_irr=pe["p"])
            rob_rows.append(row)
    gate_rows = []
    for label in sorted({r["label"] for r in rows if r["label"].startswith("gate")}):
        gr = [r for r in rows if r["label"] == label]
        g_only = [r for r in gr if r["policy"] == "gate"]
        rg = [r for r in gr if r["policy"] == "rover_gate"]
        rovm = sel("main", 1.0, "rover")
        row = dict(label=label, nu=float(label.split("_nu")[1].split("_g")[0]), gamma=float(label.split("_g")[-1]))
        for name, rs in (("gate", g_only), ("rover_gate", rg), ("rover", rovm), ("no_audit", sel("main", 1.0, "no_audit"))):
            s = summary(rs, seed=9)
            row[f"{name}_loss"], row[f"{name}_lo"], row[f"{name}_hi"] = s["loss"], s["loss_lo"], s["loss_hi"]
            row[f"{name}_irr_per100"] = s["irr_per100"]
        for a, b, tag in ((rg, g_only, "rg_minus_gate"), (rg, rovm, "rg_minus_rover"), (g_only, rovm, "gate_minus_rover")):
            pr = paired(a, b, "loss", seed=13)
            row[tag], row[f"{tag}_lo"], row[f"{tag}_hi"], row[f"{tag}_p"] = pr["diff"], pr["lo"], pr["hi"], pr["p"]
        gate_rows.append(row)
    car_rows, adapt_rows = [], []
    rng = np.random.default_rng(17)
    for p in ORDER:
        pr = sel("main", 1.0, p)
        if not pr:
            continue
        dmg = {}
        for c in CARRIERS:
            att = [dict(r, dmg=r["irr"] + r["resid"] + r["replay"] + r["lost"]) for r in pr if r["carrier"] == c.name]
            vals = np.array(list(per_stream(att, "dmg").values()))
            dmg[c.name] = vals
            lo, hi = boot_mean(vals, seed=19)
            car_rows.append(dict(policy=p, carrier=c.name, n=len(att), damage=float(vals.mean()), lo=lo, hi=hi,
                                 irr=float(np.mean([r["irr"] for r in att]))))
        pay = {c.name: c.s * dmg[c.name].mean() - c.kappa for c in CARRIERS}
        best = max(pay, key=pay.get)
        bs = []
        for _ in range(B):
            bs.append(max(c.s * dmg[c.name][rng.integers(0, len(dmg[c.name]), len(dmg[c.name]))].mean() - c.kappa
                          for c in CARRIERS))
        adapt_rows.append(dict(policy=p, best_carrier=best, payoff=float(pay[best]), lo=float(np.quantile(bs, 0.025)),
                               hi=float(np.quantile(bs, 0.975)), **{f"pay_{k}": float(v) for k, v in pay.items()}))
    alloc_rows = []
    for r1 in sorted({r["r1"] for r in rows if r["label"] == "main"}):
        rr = sel("main", r1, "rover")
        if not rr:
            continue
        tot = {c.name: float(np.mean([r[f"audits_{c.name}"] for r in rr])) for c in CARRIERS}
        s = sum(tot.values()) or 1.0
        alloc_rows.append(dict(r1=r1, total=s, **{f"share_{k}": v / s for k, v in tot.items()},
                               **{f"audits_{k}": v for k, v in tot.items()}))
    fam_rows = []
    for fam in sorted({r["family"] for r in rows}):
        a = [r for r in sel("main", 1.0, "rollback_tuned") if r["family"] == fam]
        b = [r for r in sel("main", 1.0, "rover") if r["family"] == fam]
        pr = paired(a, b, "loss", seed=23)
        fam_rows.append(dict(family=fam, rollback_minus_rover=pr["diff"], lo=pr["lo"], hi=pr["hi"], p_raw=pr["p"],
                             rover_lower=pr["b_lower"], n=pr["n"]))
    for name, rr in (("e3_main.csv", main_rows), ("e3_components.csv", comp_rows), ("e3_sweep.csv", sweep_rows),
                     ("e3_tests.csv", test_rows), ("e3_robust.csv", rob_rows), ("e3_gate.csv", gate_rows),
                     ("e3_carrier.csv", car_rows), ("e3_adaptive.csv", adapt_rows), ("e3_alloc.csv", alloc_rows),
                     ("e3_family.csv", fam_rows)):
        write(name, rr)
    print("E3 analysis:", len(main_rows), "main rows,", len(test_rows), "tests,", len(rob_rows), "robustness rows,",
          len(gate_rows), "gate rows")
    if os.path.exists(os.path.join(RES, "e3_extra.csv")):
        extra_analysis(rows, load(os.path.join(RES, "e3_extra.csv")))


GATE_PRICES = (0.1, 0.2, 0.5, 1.0)


def reprice_gate(r, c_gate, c_block=2.0):
    """Re-price a logged episode with another price per gate check and per falsely blocked benign push.
    Gate decisions do not depend on these prices, so the re-pricing is exact."""
    from testbed import COST, C_BLOCK
    r2 = dict(r)
    delta = (c_gate - COST["c_a"]) * r["gate_audits"] + (c_block - C_BLOCK) * r["blocked_benign"]
    r2["loss"] = r["loss"] + delta
    r2["audit"] = r["audit"] + delta
    return r2


def extra_analysis(rows, ext):
    """Exploratory analyses added after the confirmatory run. Outputs e3_knowledge.csv,
    e3_gate_price.csv and e3_extra_summary.csv. Every comparison is other minus ROVER (positive: ROVER lower)."""
    def sel(rs, label, pol, r1=1.0):
        return [r for r in rs if r["label"] == label and r["policy"] == pol and r["r1"] == r1]

    know = []
    conds = [("shift0", rows, "main", "main"), ("shift2", ext, "shift2", "shift2"), ("shift4", ext, "shift4", "shift4"),
             ("shift6", ext, "shift6", "shift6"), ("shift8", rows, "late", "late"),
             ("rate0.20_fix", ext, "rate0.20_fix", "rate0.20_fix"), ("rate0.20_cal", ext, "rate0.20_cal", "rate0.20_fix"),
             ("rate0.05_fix", ext, "rate0.05_fix", "rate0.05_fix"), ("rate0.05_cal", ext, "rate0.05_cal", "rate0.05_fix"),
             ("s1low", rows, "s1low", "s1low"), ("dalpha", rows, "dalpha", "dalpha")]
    for name, src, lab, lab_na in conds:
        rov = sel(src, lab, "rover")
        s = summary(rov, seed=29)
        row = dict(condition=name, rover=s["loss"], rover_lo=s["loss_lo"], rover_hi=s["loss_hi"],
                   rover_irr100=s["irr_per100"], rover_audits=s["audits"], attacked=s["attacked"])
        for pol, src2, lab2 in (("no_audit", src, lab_na), ("rover_noaudit", src, lab), ("rover_const", src, lab),
                                ("rollback_tuned", src, lab), ("myopic", src, lab)):
            other = sel(src2, lab2, pol)
            if not other:
                continue
            pr = paired(other, rov, "loss", seed=31)
            pe = paired(other, rov, "irr_events", seed=31)
            row.update({f"{pol}_loss": float(np.mean([r["loss"] for r in other])), f"{pol}_d": pr["diff"],
                        f"{pol}_lo": pr["lo"], f"{pol}_hi": pr["hi"], f"{pol}_p": pr["p"],
                        f"{pol}_rover_lower": pr["b_lower"], f"{pol}_other_lower": pr["a_lower"],
                        f"{pol}_dirr100": 100 * pe["diff"], f"{pol}_dirr100_lo": 100 * pe["lo"],
                        f"{pol}_dirr100_hi": 100 * pe["hi"], f"{pol}_pirr": pe["p"]})
        know.append(row)
    write("e3_knowledge.csv", know)

    gp = []
    rovm = sel(rows, "main", "rover")
    L_rov = float(np.mean([r["loss"] for r in rovm]))
    for label in sorted({r["label"] for r in rows if r["label"].startswith("gate")}):
        nu = float(label.split("_nu")[1].split("_g")[0])
        gamma = float(label.split("_g")[-1])
        g_only = sel(rows, label, "gate")
        rg = sel(rows, label, "rover_gate")
        ra = sel(ext, label, "rover_gate_aware")
        row = dict(label=label, nu=nu, gamma=gamma, rover=L_rov,
                   checks_gate=float(np.mean([r["gate_audits"] for r in g_only])),
                   blocks_gate=float(np.mean([r["blocked_benign"] for r in g_only])),
                   canary_remote_gate=float(np.mean([r["remote_canary"] for r in g_only])),
                   canary_remote_aware=float(np.mean([r["remote_canary"] for r in ra])),
                   canary_remote_rover_gate=float(np.mean([r["remote_canary"] for r in rg])),
                   canary_remote_rover_main=float(np.mean([r["remote_canary"] for r in rovm])),
                   benign_pushes=float(np.mean([r["benign_pushes"] for r in g_only])))
        for c in GATE_PRICES:
            for name, rs in (("gate", g_only), ("rover_gate", rg), ("aware", ra)):
                rp = [reprice_gate(r, c) for r in rs]
                row[f"{name}_{c:g}"] = float(np.mean([r["loss"] for r in rp]))
                pr = paired(rp, rovm, "loss", seed=37)
                row[f"{name}_{c:g}_d"], row[f"{name}_{c:g}_lo"], row[f"{name}_{c:g}_hi"] = pr["diff"], pr["lo"], pr["hi"]
                row[f"{name}_{c:g}_p"] = pr["p"]
            pa = paired([reprice_gate(r, c) for r in ra], [reprice_gate(r, c) for r in g_only], "loss", seed=41)
            row[f"aware_minus_gate_{c:g}"], row[f"aware_minus_gate_{c:g}_lo"] = pa["diff"], pa["lo"]
            row[f"aware_minus_gate_{c:g}_hi"], row[f"aware_minus_gate_{c:g}_p"] = pa["hi"], pa["p"]
        # break-even gate price at which gate only equals ROVER without a gate (block price fixed at 2)
        base = float(np.mean([reprice_gate(r, 0.0)["loss"] for r in g_only]))
        row["breakeven_gate_price"] = (L_rov - base) / row["checks_gate"] if row["checks_gate"] > 0 else float("nan")
        gp.append(row)
    write("e3_gate_price.csv", gp)

    out = []
    # rollback only tuned for r1 <= 0.1 on the development streams (theta from e3_extra_meta.json)
    for r1 in (0.0, 0.1):
        rc = sel(ext, "main", "rollback_cheap", r1)
        rv = sel(rows, "main", "rover", r1)
        rt = sel(rows, "main", "rollback_tuned", r1)
        pr = paired(rc, rv, "loss", seed=43)
        out.append(dict(item=f"rollback_cheap_r{r1:g}", value=float(np.mean([r["loss"] for r in rc])), d=pr["diff"],
                        lo=pr["lo"], hi=pr["hi"], p=pr["p"], ref=float(np.mean([r["loss"] for r in rt]))))
    # monitoring baselines at zero monitoring price
    from testbed import COST
    for pol, key, price in (("periodic_review", "reviewed", COST["c_r"]), ("immutable_validator", "validations", COST["c_v"])):
        rr = []
        for r in sel(rows, "main", pol):
            r2 = dict(r)
            r2["loss"] = r["loss"] - price * float(r[key])
            rr.append(r2)
        pr = paired(rr, rovm, "loss", seed=47)
        out.append(dict(item=f"{pol}_free_monitoring", value=float(np.mean([r["loss"] for r in rr])), d=pr["diff"],
                        lo=pr["lo"], hi=pr["hi"], p=pr["p"], ref=L_rov))
    # primary family: Holm over H1 (loss vs tuned rollback) and H2 (irreversible activations vs no audits)
    h1 = paired(sel(rows, "main", "rollback_tuned"), rovm, "loss", seed=7)
    h2 = paired(sel(rows, "main", "rover_noaudit"), rovm, "irr_events", seed=11)
    adj = holm([h1["p"], h2["p"]])
    out.append(dict(item="primary_H1", value=h1["diff"], d=h1["diff"], lo=h1["lo"], hi=h1["hi"], p=float(adj[0]),
                    ref=h1["a12"]))
    out.append(dict(item="primary_H2", value=100 * h2["diff"], d=100 * h2["diff"], lo=100 * h2["lo"], hi=100 * h2["hi"],
                    p=float(adj[1]), ref=h2["a12"]))
    # H2 at every r1 (irreversible activations, ROVER without audits minus ROVER)
    for r1 in (0.0, 0.1, 1.0, 10.0):
        pe = paired(sel(rows, "main", "rover_noaudit", r1), sel(rows, "main", "rover", r1), "irr_events", seed=53)
        out.append(dict(item=f"H2_irr_r{r1:g}", value=100 * pe["diff"], d=100 * pe["diff"], lo=100 * pe["lo"],
                        hi=100 * pe["hi"], p=pe["p"], ref=pe["a_lower"]))
    # false recoveries and recoveries of tuned rollback only vs ROVER
    for k in ("rec_false", "recoveries", "regen"):
        pr = paired(sel(rows, "main", "rollback_tuned"), rovm, k, seed=59)
        out.append(dict(item=f"rollback_{k}", value=float(np.mean([r[k] for r in sel(rows, "main", "rollback_tuned")])),
                        d=pr["diff"], lo=pr["lo"], hi=pr["hi"], p=pr["p"], ref=float(np.mean([r[k] for r in rovm]))))
    # joint worst case over carrier and timing (insertion in the window or 8 milestones later)
    for pol in ("rover", "rover_const", "rover_noaudit", "rollback_tuned", "no_audit", "myopic"):
        best = (None, -1e9)
        for lab in ("main", "late"):
            pr_ = sel(rows, lab, pol)
            for c in CARRIERS:
                att = [r["irr"] + r["resid"] + r["replay"] + r["lost"] for r in pr_ if r["carrier"] == c.name]
                pay = c.s * float(np.mean(att)) - c.kappa
                if pay > best[1]:
                    best = (f"{c.name}/{lab}", pay)
        out.append(dict(item=f"joint_worst_{pol}", value=best[1], d=0.0, lo=0.0, hi=0.0, p=1.0, ref=best[0]))
    write("e3_extra_summary.csv", out)
    print("E3 extra analysis:", len(know), "knowledge rows,", len(gp), "gate rows,", len(out), "summary items")


if __name__ == "__main__":
    main()
