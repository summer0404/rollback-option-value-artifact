"""Numerical checks of the formal results (Theorems 1, 3, 4 and Propositions 2, 5 of the paper).

T1   random restricted instances, grid DP: recovery set is an interval (T1.ii); under
     A6 (h <= r1) the recovery set shrinks with D (T1.iii). Proposition 2 counts the
     violations of the unconditional two-threshold claim and re-checks flagged cells
     with the exact recursion. The explicit instance of Proposition 2 is confirmed by
     exhaustive enumeration of every deterministic policy over two periods.
(The keys T2 and T3 of theory_checks.json hold the checks of Theorems 3 and 4 of the paper, and
P3 those of Proposition 5. The key names are kept for continuity with RUNLOG.md.)
T2   random instances of the perfect-audit submodel (A7): closed-form corner costs
     equal the exact DP, the optimum is a corner, the myopic gap obeys the bound;
     a tightness family shows the gap growing with the horizon.
T3   closed-form damage vs Monte Carlo; best-response slope monotone in r1 for fixed
     coverage; minimax comparative statics vs finite differences under A10; the
     Proposition 3 counterexamples (best response is not argmax phi; the highest-phi
     carrier can lose coverage as rollback gets cheaper).
Writes results/theory_checks.json, prop2_instance.csv, t2_tightness.csv.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (Params, GridDP, ExactDP, W, R, A, ACTION_NAMES, corner_costs, theorem2_gain,  # noqa
                   myopic_gap_bound, myopic_defers, submodel_ok, S_T, MyopicPolicy)
import structure as ST  # noqa: E402
import game as GM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

PROP2 = Params(T=2, pi0=0.5, rho=0.0, Delta=1, grow_p=1.0, lam=0.2, phi=0.0, H=10.0, h=4.0, r0=4.0, r1=1.0,
               c_a=2.0, d=1.0, alpha=0.0, s0=0.2, s1=0.9, c_fr=0.0, D0=10)
PROP2_BELIEFS = (0.2, 0.55, 0.82, 0.95)
# Proposition 2(b): exact two-period instance under A6 (h = r1) with a recovery threshold rising in D
PROP2B = Params(T=2, pi0=0.5, rho=0.0, Delta=1, grow_p=1.0, lam=0.3, phi=1.0, H=20.0, h=1.0, r0=1.0, r1=1.0,
                c_a=2.0, d=0.9, alpha=0.05, s0=0.1, s1=0.4, c_fr=0.0, D0=0)
PROP2B_BELIEF = 0.4
# Proposition 2(c): exact two-period instance with imperfect audits and a disconnected audit set
PROP2C = Params(T=2, pi0=0.5, rho=0.0, Delta=1, grow_p=1.0, lam=0.2, phi=0.5, H=2.0, h=4.0, r0=2.0, r1=2.0,
                c_a=0.25, d=0.8, alpha=0.3, s0=0.2, s1=0.7, c_fr=0.0, D0=6)
PROP2C_BELIEFS = (0.05, 0.18, 0.33, 0.6, 0.95)


def t1_checks(n_inst=1000, seed=2027, G=1001, n_confirm=25):
    rng = np.random.default_rng(seed)
    tot = dict(instances=0, a6_instances=0, cells=0, rec_nonempty=0, t1_interval=0, t1_monoD=0,
               t1_monoD_cells=0, p2_upper=0, p2_audit_interval=0, p2_pattern=0, p2_thr_r_up=0, p2_thr_a_up=0)
    inst_with = {k: 0 for k in ("p2_upper", "p2_audit_interval", "p2_pattern", "p2_thr_r_up", "p2_thr_a_up",
                                "t1_interval", "t1_monoD")}
    flagged = {k: [] for k in inst_with}
    for i in range(n_inst):
        p = ST.random_params(rng)
        res, dp = ST.check_instance(p, G=G)
        tot["instances"] += 1
        tot["a6_instances"] += int(p.h <= p.r1)
        for k in tot:
            if k in res and k not in ("instances",):
                tot[k] += res[k]
        for k in inst_with:
            if res[k] > 0:
                inst_with[k] += 1
                flagged[k].append((p, res["examples"][k]))
    # exact re-check of flagged pattern / audit-interval cells (beliefs from the grid example)
    confirm = {}
    for k in ("p2_audit_interval", "p2_pattern"):
        ok = 0
        tried = 0
        for p, ex in flagged[k][:n_confirm]:
            t, D = ex[0], ex[1]
            dp = GridDP(p, G=G)
            c = ST.cell_sets(dp, t, D)
            lab = np.where(c["wait_strict"], 0, np.where(c["aud_strict"], 1, np.where(
                c["qr"] < np.minimum(c["qw"], c["qa"]) - ST.MARGIN * (1 + np.abs(c["m"])), 2, -1)))
            idx = np.flatnonzero(lab >= 0)
            seq = lab[idx]
            j = np.flatnonzero(np.diff(seq) < 0) if k == "p2_pattern" else None
            # pick representative beliefs: last cell before the order break and first after it
            if k == "p2_pattern":
                if j.size == 0:
                    continue
                b1, b2 = dp.grid[idx[j[0]]], dp.grid[idx[j[0] + 1]]
                acts = ST.exact_actions(p, t, D, [b1, b2])
                order = {W: 0, A: 1, R: 2}
                tried += 1
                ok += int(order[acts[1][4]] < order[acts[0][4]])
            else:
                i1, i2, i3 = ex[2], ex[3], ex[4]
                acts = ST.exact_actions(p, t, D, [dp.grid[i1], dp.grid[(i2 + i3) // 2], dp.grid[i3]])
                tried += 1
                ok += int(acts[0][4] == A and acts[2][4] == A and acts[1][4] != A)
        confirm[k] = dict(tried=tried, confirmed=ok)
    return tot, inst_with, confirm


def prop2_instance():
    """Exhaustive enumeration of every deterministic policy over the two periods of PROP2."""
    rows = []
    e = ExactDP(PROP2)
    for b in PROP2_BELIEFS:
        costs = ST.enumerate_policy_costs(PROP2, 0, PROP2.D0, b)
        n = int(sum(len(v) for v in costs.values()))
        best = {k: float(v.min()) for k, v in costs.items()}
        arg = min(best, key=best.get)
        rows.append(dict(belief=b, n_policies=n, best_W=best["W"], best_R=best["R"], best_A=best["A"],
                         optimal=arg, exact_value=e.V(0, PROP2.D0, b), match=int(abs(min(best.values()) - e.V(0, PROP2.D0, b)) < 1e-9)))
    # boundaries of the action regions on a fine belief scan (exact recursion)
    bs = np.linspace(0, 1, 401)
    acts = [e.action(0, PROP2.D0, float(b)) for b in bs]
    segs = []
    for b, a in zip(bs, acts):
        if not segs or segs[-1][0] != a:
            segs.append([a, float(b), float(b)])
        else:
            segs[-1][2] = float(b)
    from model import bayes_alarm
    post = {}
    for b in (PROP2_BELIEFS[1], PROP2_BELIEFS[2]):
        for o in (0, 1):
            post[f"{b}_{o}"] = float(bayes_alarm(PROP2, np.float64(b), o)[1])
    rows[0]["alarm_posteriors"] = json.dumps(post)
    with open(os.path.join(RES, "prop2_instance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows, [(ACTION_NAMES[a], lo, hi) for a, lo, hi in segs]


def scan_segments(p, t, D, n=401):
    e = ExactDP(p)
    bs = np.linspace(0, 1, n)
    segs = []
    for b in bs:
        a = e.action(t, D, float(b))
        if not segs or segs[-1][0] != a:
            segs.append([a, float(b), float(b)])
        else:
            segs[-1][2] = float(b)
    return [(ACTION_NAMES[a], lo, hi) for a, lo, hi in segs]


def enum_best(p, t, D, b):
    costs = ST.enumerate_policy_costs(p, t, D, b)
    best = {k: float(v.min()) for k, v in costs.items()}
    return dict(belief=b, n_policies=int(sum(len(v) for v in costs.values())), best=best,
                optimal=min(best, key=best.get), exact=ExactDP(p).V(t, D, b))


def plan_lines(p, t, D, b1=0.3, b2=0.7):
    """Every deterministic plan's cost is affine in b. Recover (intercept, slope) per plan and per
    first-action class from two interior beliefs (all branch probabilities positive at both)."""
    c1 = ST.enumerate_policy_costs(p, t, D, b1)
    c2 = ST.enumerate_policy_costs(p, t, D, b2)
    out = {}
    for k in c1:
        assert len(c1[k]) == len(c2[k])
        s = (c2[k] - c1[k]) / (b2 - b1)
        out[k] = (c1[k] - s * b1, s)
    return out


def exact_recovery_threshold(p, t, D, tol=1e-13):
    """theta_r(t, D) from the lower envelopes of all plans: smallest b with Q_R(b) <= min(Q_W, Q_A)(b)."""
    L = plan_lines(p, t, D)
    env = lambda k, b: float(np.min(L[k][0] + L[k][1] * b))
    F = lambda b: min(env("W", b), env("A", b)) - env("R", b)
    bs = np.linspace(0, 1, 20001)
    vals = np.array([F(b) for b in bs])
    idx = np.flatnonzero(vals >= 0)
    if idx.size == 0:
        return None, 0
    j = idx[0]
    if j == 0:
        return 0.0, int(sum(len(v[0]) for v in L.values()))
    lo, hi = float(bs[j - 1]), float(bs[j])
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if F(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi, int(sum(len(v[0]) for v in L.values()))


def prop2bc():
    """Exact instances for Proposition 2(b) and 2(c), each confirmed by exhaustive plan enumeration.
    All three instances of Proposition 2 use deterministic growth (xi = 1 every period)."""
    out = {}
    thr = {}
    thr_scan = {}
    nplans = {}
    for D in (2, 6):
        segs = scan_segments(PROP2B, 0, D)
        thr_scan[D] = [s for s in segs if s[0] == "R"][0][1]
        thr[D], nplans[D] = exact_recovery_threshold(PROP2B, 0, D)
    out["P2b"] = dict(params=PROP2B.asdict(), theta_r={str(k): v for k, v in thr.items()},
                      theta_r_scan401={str(k): v for k, v in thr_scan.items()},
                      theta_r_plans={str(k): v for k, v in nplans.items()},
                      enum_D2=enum_best(PROP2B, 0, 2, PROP2B_BELIEF), enum_D6=enum_best(PROP2B, 0, 6, PROP2B_BELIEF))
    out["P2c"] = dict(params=PROP2C.asdict(), segments=scan_segments(PROP2C, 0, PROP2C.D0),
                      enumeration=[enum_best(PROP2C, 0, PROP2C.D0, b) for b in PROP2C_BELIEFS])
    return out


def t1_perfect_audit(n_inst=400, seed=99, G=1001):
    """Theorem 1(v): with d = 1 and alpha = 0 the audit set is an interval (grid check, random instances)."""
    rng = np.random.default_rng(seed)
    cells = viol = 0
    for i in range(n_inst):
        p = replace(ST.random_params(rng), d=1.0, alpha=0.0)
        res, _ = ST.check_instance(p, G=G)
        cells += res["cells"]
        viol += res["p2_audit_interval"]
    return dict(instances=n_inst, cells=cells, audit_interval_violations=viol)


def default_thresholds():
    """Recovery and first-act thresholds vs D for the E1 base model (A6 holds: h = r1 = 1)."""
    from exp_e1 import BASE
    dp = GridDP(BASE, G=2001)
    out = []
    for t in (5, 10, 15):
        for D in range(0, 11, 2):
            a = dp.action(t, np.full(dp.G, D), dp.grid)
            rec = np.flatnonzero(a == R)
            act = np.flatnonzero(a != W)
            out.append(dict(t=t, D=D, theta_r=float(dp.grid[rec[0]]) if rec.size else None,
                            theta_act=float(dp.grid[act[0]]) if act.size else None))
    return out


def t2_checks(n_inst=600, seed=7):
    rng = np.random.default_rng(seed)
    max_err = 0.0
    corner_viol = 0
    bound_viol = 0
    defers = 0
    action_match = 0
    max_ratio_used = 0.0
    for i in range(n_inst):
        T = int(rng.integers(2, 13))
        r1 = float(rng.uniform(0, 3))
        D0 = int(rng.integers(1, 8))
        h = float(r1 + rng.uniform(0.2, 3))
        r0 = float(rng.uniform(0, (h - r1) * D0))
        p = Params(T=T, pi0=float(rng.uniform(0.02, 0.9)), rho=0.0, lam=float(rng.uniform(0.01, 0.6)),
                   phi=float(rng.uniform(0, 1)), H=float(rng.uniform(0, 40)), h=h, r0=r0, r1=r1,
                   c_a=float(rng.uniform(0.05, 4)), d=1.0, alpha=0.0, s0=0.3, s1=0.3, D0=D0)
        assert submodel_ok(p)
        cc = corner_costs(p)
        Jstar = min(min(cc["JA"]), cc["JB0"], cc["JN"])
        ex = ExactDP(p).V(0, p.D0, p.pi0)
        max_err = max(max_err, abs(ex - Jstar))
        # bang-bang: argmin over audit times is an endpoint
        ja = np.array(cc["JA"])
        if ja.min() < min(ja[0], ja[-1]) - 1e-9:
            corner_viol += 1
        g = theorem2_gain(p)
        if (g > 1e-12 and np.argmin(ja) != 0) or (g < -1e-12 and np.argmin(ja) != len(ja) - 1):
            corner_viol += 1
        # myopic gap
        if myopic_defers(p):
            defers += 1
            gap = cc["JN"] - Jstar
            if gap > myopic_gap_bound(p) + 1e-9:
                bound_viol += 1
            max_ratio_used = max(max_ratio_used, gap / max(myopic_gap_bound(p), 1e-12))
        # exact evaluation of the myopic policy equals the closed form
        my = MyopicPolicy(p)
        ev = ExactDP(p, policy=lambda t, D, b: int(my.act(t, D, b)), post=lambda t, D, b: bool(my.post(t, D, b)))
        vm = ev.V(0, p.D0, p.pi0)
        pred = cc["JN"] if myopic_defers(p) else cc["JA"][0]
        action_match += int(abs(vm - pred) < 1e-8)
    return dict(instances=n_inst, max_abs_err=max_err, corner_violations=corner_viol, myopic_defers=defers,
                bound_violations=bound_viol, max_gap_over_bound=max_ratio_used, myopic_closed_form_match=action_match)


def t2_tightness_proof(eps=0.05):
    """The family of the tightness proof: r1 = 0, phi = 1, pi*lam*H = (1-eps) c_a, h = r0/D0, r0 -> 0."""
    rows = []
    for T in (5, 20, 80):
        for r0 in (0.5, 0.1, 0.01):
            lam, pi, c_a, D0 = 0.01, 0.5, 1.0, 1
            H = (1 - eps) * c_a / (pi * lam)
            p = Params(T=T, pi0=pi, rho=0.0, lam=lam, phi=1.0, H=H, h=r0 / D0, r0=r0, r1=0.0, c_a=c_a,
                       d=1.0, alpha=0.0, s0=0.3, s1=0.3, D0=D0)
            assert submodel_ok(p) and myopic_defers(p)
            cc = corner_costs(p)
            Jstar = min(min(cc["JA"]), cc["JB0"], cc["JN"])
            gap = cc["JN"] - Jstar
            rows.append(dict(T=T, r0=r0, gap=gap, bound=myopic_gap_bound(p), ratio_gap_bound=gap / myopic_gap_bound(p)))
    return rows


def p5c_kink(h_eps=1e-4):
    """Proposition 5(c): one-sided derivatives of the queued call's coverage where it leaves the attack set."""
    g0 = GM.GameParams()
    iq = [c.name for c in GM.CARRIERS].index("queue")
    xq = lambda r1: float(GM.minimax(GM.CARRIERS, replace(g0, r1=r1))[0][iq])
    lo, hi = 0.0, 10.0
    if xq(hi) > 1e-6:
        return dict(found=False)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if xq(mid) > 1e-7:
            lo = mid
        else:
            hi = mid
    r_hat = 0.5 * (lo + hi)
    left = (xq(r_hat - h_eps) - xq(r_hat - 2 * h_eps)) / h_eps
    right = (xq(r_hat + 2 * h_eps) - xq(r_hat + h_eps)) / h_eps
    return dict(found=True, r_hat=r_hat, left_derivative=left, right_derivative=right)


def t2_tightness():
    """Family where the myopic rule defers although immediate audit is optimal."""
    rows = []
    for T in (5, 10, 20, 40, 80):
        lam = 0.01
        pi = 0.5
        c_a = 1.0
        phiH = 0.95 * c_a / (pi * lam)     # myopic defers: pi*lam*phi*H = 0.95 c_a
        p = Params(T=T, pi0=pi, rho=0.0, lam=lam, phi=1.0, H=phiH, h=2.0, r0=1.0, r1=0.0, c_a=c_a,
                   d=1.0, alpha=0.0, s0=0.3, s1=0.3, D0=1)
        assert submodel_ok(p) and myopic_defers(p)
        cc = corner_costs(p)
        Jstar = min(min(cc["JA"]), cc["JB0"], cc["JN"])
        gap = cc["JN"] - Jstar
        rows.append(dict(T=T, J_star=Jstar, J_myopic=cc["JN"], gap=gap, ratio=cc["JN"] / Jstar,
                         bound=myopic_gap_bound(p), S_T=S_T(lam, T)))
    with open(os.path.join(RES, "t2_tightness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def t3_checks(seed=11):
    rng = np.random.default_rng(seed)
    # (i) closed form vs Monte Carlo
    mc_out = 0
    mc_n = 0
    for k in range(40):
        cs, g = GM.random_game(rng, n=1)
        c = cs[0]
        for x in (0.0, float(rng.uniform(0, 1))):
            L = GM.simulate_damage(c, g, x, 60000, rng)
            se = L.std() / np.sqrt(len(L))
            mc_n += 1
            mc_out += int(abs(L.mean() - float(GM.damage(c, g, x))) > 3.5 * se + 1e-9)
    # (ii) best-response slope monotone in r1 for fixed coverage
    br_viol = 0
    br_n = 0
    r1s = np.linspace(0, 10, 41)
    for k in range(1500):
        cs, g = GM.random_game(rng)
        x = rng.dirichlet(np.ones(len(cs))) * g.B
        x = np.minimum(x, 1)
        slopes = []
        for r1 in r1s:
            gg = replace(g, r1=r1)
            br, U = GM.best_response(cs, gg, x)
            if br < 0:
                slopes.append(0.0)
            else:
                _, Bv = GM.AB(cs[br], gg, x[br])
                slopes.append(cs[br].s * float(Bv))
        br_n += 1
        br_viol += int(np.any(np.diff(slopes) < -1e-9))
    # (iii) comparative statics vs finite differences under A10
    d_n = d_ok = a10_fail = sign_ok = nonpos_level = 0
    max_rel = 0.0
    for k in range(400):
        cs, g = GM.random_game(rng)
        eps = 1e-3
        x0, u0, s0 = GM.minimax(cs, g)
        xm, um, sm = GM.minimax(cs, replace(g, r1=max(g.r1 - eps, 0.0)))
        xp, up, sp = GM.minimax(cs, replace(g, r1=g.r1 + eps))
        decreasing = GM.a10_monotone(cs, g)
        interior = all(1e-4 < x0[i] < 1 - 1e-4 for i in s0)
        outside_ok = all(x0[j] < 1e-9 and float(GM.utility(cs[j], g, 0.0)) < u0 - 1e-4
                         for j in range(len(cs)) if j not in s0)
        a10 = (s0 == sm == sp) and decreasing and interior and outside_ok and len(s0) >= 2 \
            and abs(sum(x0) - g.B) < 1e-6 and g.r1 > eps
        if not a10:
            a10_fail += 1
            continue
        if min(u0, um, up) <= 0:          # A10 requires a positive level (Definition 5's floor at 0)
            nonpos_level += 1
            continue
        pred, bbar = GM.predicted_derivative(cs, g, x0, s0)
        fd = (xp - xm) / (2 * eps)
        d_n += 1
        rel = np.max(np.abs(fd[s0] - pred[s0])) / max(np.max(np.abs(fd[s0])), 1e-6)
        max_rel = max(max_rel, float(rel))
        d_ok += int(rel < 0.05)
        sign_ok += int(np.all(np.sign(np.round(fd[s0], 6)) == np.sign(np.round(pred[s0], 6))))
    # Proposition 3: best response at r1 = 0 (no coverage) vs argmax phi; highest-phi carrier losing coverage
    not_argmax_phi = n_p3 = phi_loses = adm = phi_loses_adm = 0
    for k in range(1500):
        cs, g = GM.random_game(rng)
        g0 = replace(g, r1=0.0)
        br, U = GM.best_response(cs, g0, np.zeros(len(cs)))
        if br < 0:
            continue
        n_p3 += 1
        iphi = int(np.argmax([c.phi for c in cs]))
        not_argmax_phi += int(br != iphi)
        xa, ua, _ = GM.minimax(cs, replace(g, r1=0.0))
        xb, ub, _ = GM.minimax(cs, replace(g, r1=2.0))
        loses = int(xa[iphi] < xb[iphi] - 1e-4)
        phi_loses += loses
        if g.h >= 2.0 and ua > 0 and ub > 0:    # A9 at both prices and a positive level (A10's u* > 0)
            adm += 1
            phi_loses_adm += loses
    return dict(mc_cells=mc_n, mc_outside=mc_out, br_instances=br_n, br_violations=br_viol,
                deriv_instances=d_n, deriv_within5pct=d_ok, deriv_sign_agree=sign_ok, deriv_max_rel=max_rel,
                a10_fail=a10_fail, nonpos_level=nonpos_level, p3_instances=n_p3,
                p3_br_not_argmax_phi=not_argmax_phi, p3_highest_phi_loses_coverage=phi_loses,
                p5b_admissible=adm, p5b_loses_admissible=phi_loses_adm)


P3_CARRIERS = (GM.Carrier("a", 1.0, 0.02, 1.0, 0.5, 1.0), GM.Carrier("b", 0.1, 0.2, 0.0, 0.8, 0.5))
P3_GAME = GM.GameParams(T=20, H=20.0, h=2.0, r0=1.0, r1=0.0, d=0.9, B=0.5)   # h >= r1 on [0, 2] (A9)


def p3_instance():
    """Explicit two-carrier instance for Proposition 3(b): the phi = 1 carrier loses coverage
    when rollback gets cheaper, because its reversible exposure is the larger one."""
    out = dict(carriers=[c.__dict__ for c in P3_CARRIERS], game=P3_GAME.__dict__, rows=[])
    for r1 in np.round(np.linspace(0.0, 2.0, 21), 3):
        g = replace(P3_GAME, r1=float(r1))
        x, u, sup = GM.minimax(P3_CARRIERS, g)
        sB = [c.s * float(GM.AB(c, g, xi)[1]) for c, xi in zip(P3_CARRIERS, x)]
        out["rows"].append(dict(r1=float(r1), x=[float(v) for v in x], u=float(u), support=sup, sB=sB,
                                a10=int(GM.a10_monotone(P3_CARRIERS, g))))
    out["support_always_both"] = int(all(len(r["support"]) == 2 for r in out["rows"]))
    out["a10_always"] = int(all(r["a10"] for r in out["rows"]))
    return out


def grid_error(G=1001):
    """Belief-grid DP (linear interpolation) against the exact recursion on the E1 base instance with a
    short horizon, at 37 beliefs (most of them off the grid) and every D in 0..D0. The grid keeps D within
    D0 + T, which every reachable state satisfies."""
    from exp_e1 import BASE
    out = []
    for T in (3, 4, 5, 6):
        p = replace(BASE, T=T)
        gd = GridDP(p, G=G)
        ex = ExactDP(p)
        worst_abs = worst_rel = 0.0
        for D in range(0, p.D0 + 1):
            for b in np.linspace(0, 1, 37):
                v_ex = ex.V(0, D, float(b))
                v_g = float(gd.interp(0, np.array([D]), np.array([b]))[0])
                worst_abs = max(worst_abs, abs(v_g - v_ex))
                worst_rel = max(worst_rel, abs(v_g - v_ex) / max(abs(v_ex), 1e-12))
        out.append(dict(T=T, G=G, max_abs=worst_abs, max_rel=worst_rel))
    return out


def p5a_low_h():
    """Proposition 5(a) payoffs without coverage at r1 = 0 for h = 10 (E2) and h = 1 (E1, E3)."""
    res = {}
    for h in (10.0, 1.0):
        g = replace(GM.GameParams(), r1=0.0, h=h)
        U = {c.name: float(GM.utility(c, g, 0.0)) for c in GM.CARRIERS}
        res[str(h)] = dict(U=U, br=max(U, key=U.get), least=min(U, key=U.get))
    return res


def main():
    if "--only-p3" in sys.argv:
        path = os.path.join(RES, "theory_checks.json")
        d = json.load(open(path))
        d["P3_instance"] = p3_instance()
        json.dump(d, open(path, "w"), indent=1, default=float)
        print(d["P3_instance"])
        return
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    out = {}
    tot, inst_with, confirm = t1_checks()
    out["T1"] = dict(totals=tot, instances_with=inst_with, exact_confirm=confirm)
    print("T1 done", time.time() - t0, flush=True)
    rows, segs = prop2_instance()
    out["P2_instance"] = dict(params=PROP2.asdict(), enumeration=rows, segments=segs)
    out["default_thresholds"] = default_thresholds()
    print("P2 done", time.time() - t0, flush=True)
    out["T2"] = t2_checks()
    out["T2_tightness"] = t2_tightness()
    out["T2_tightness_proof"] = t2_tightness_proof()
    out["P2bc"] = prop2bc()
    out["T1_perfect_audit"] = t1_perfect_audit()
    out["P5c"] = p5c_kink()
    out["P5a_low_h"] = p5a_low_h()
    out["GRID"] = grid_error()
    print("T2 done", time.time() - t0, flush=True)
    out["T3"] = t3_checks()
    out["P3_instance"] = p3_instance()
    print("T3 done", time.time() - t0, flush=True)
    with open(os.path.join(RES, "theory_checks.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
