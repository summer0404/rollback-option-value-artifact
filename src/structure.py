"""Structural checks of the optimal policy (Theorem 1, Proposition 2 of the paper).

For a solved GridDP we extract, for every period t and reachable derived work D,
the canonical optimal action along the belief grid and test:
  T1(ii)  the recovery set is an interval                      (must hold)
  T1(iii) h <= r1  =>  recovery set at D+Delta is a subset of the set at D (must hold)
  P2(a)   recovery set contains b = 1 when nonempty            (upper interval)
  P2(b)   audit set is an interval
  P2(c)   action pattern along b is W* A* R*                    (two ordered thresholds)
  P2(d)   recovery threshold nonincreasing in D                 (part of the two-threshold conjecture, Definition 3)
  P2(e)   first non-wait threshold nonincreasing in D           (part of the two-threshold conjecture, Definition 3)
Violations are counted only when the deciding Q-value margin exceeds MARGIN, so
interpolation ties do not register. Candidates are re-checked with ExactDP.
"""
from __future__ import annotations

import numpy as np
from model import GridDP, ExactDP, Params, W, R, A

MARGIN = 1e-6


def cell_sets(dp: GridDP, t: int, D: int):
    G = dp.G
    b = dp.grid
    Dv = np.full(G, D)
    qw, qr, qa = dp.qall(t, Dv, b)
    m = np.minimum(np.minimum(qw, qr), qa)
    scale = 1.0 + np.abs(m)
    tol = MARGIN * scale
    # robust membership: R in set if within tol of min; strictly out if worse by > tol
    rec_in = qr <= m + 1e-12 * scale
    rec_out_strict = qr > m + tol
    aud_in_strict = qa < np.minimum(qw, qr) - tol
    aud_in = qa <= m + 1e-12 * scale
    wait_in_strict = qw < np.minimum(qr, qa) - tol
    act = np.where(qw <= m + 1e-12 * scale, W, np.where(rec_in, R, A))
    return dict(qw=qw, qr=qr, qa=qa, m=m, rec_in=rec_in, rec_out=rec_out_strict, aud_in=aud_in,
                aud_strict=aud_in_strict, wait_strict=wait_in_strict, act=act)


def interval_violation(member, strict_out):
    """True if there are i<j<k with member[i], member[k] and strict_out[j]."""
    idx = np.flatnonzero(member)
    if idx.size < 2:
        return None
    lo, hi = idx[0], idx[-1]
    gap = np.flatnonzero(strict_out[lo:hi + 1])
    if gap.size == 0:
        return None
    j = lo + gap[0]
    i = idx[idx < j][-1]
    k = idx[idx > j][0]
    return (int(i), int(j), int(k))


def threshold(member):
    idx = np.flatnonzero(member)
    return None if idx.size == 0 else int(idx[0])


def check_instance(p: Params, G: int = 2001):
    dp = GridDP(p, G=G)
    res = dict(cells=0, rec_nonempty=0,
               t1_interval=0, t1_monoD=0, t1_monoD_cells=0,
               p2_upper=0, p2_audit_interval=0, p2_pattern=0, p2_thr_r_up=0, p2_thr_a_up=0,
               examples={})
    prev = None
    for t in range(p.T):
        prev = None
        for D in range(0, p.D0 + p.Delta * t + 1):
            c = cell_sets(dp, t, D)
            res["cells"] += 1
            rec, rec_out = c["rec_in"], c["rec_out"]
            if rec.any():
                res["rec_nonempty"] += 1
                if not rec[-1] and np.flatnonzero(rec)[-1] < dp.G - 1 and rec_out[-1]:
                    res["p2_upper"] += 1
                    res["examples"].setdefault("p2_upper", (t, D, int(np.flatnonzero(rec)[-1])))
            v = interval_violation(rec, rec_out)
            if v is not None:
                res["t1_interval"] += 1
                res["examples"].setdefault("t1_interval", (t, D) + v)
            # audit-set interval: strict audit cells separated by strictly non-audit cells
            aud_strict = c["aud_strict"]
            non_aud = c["qa"] > np.minimum(c["qw"], c["qr"]) + MARGIN * (1 + np.abs(c["m"]))
            v = interval_violation(aud_strict, non_aud)
            if v is not None:
                res["p2_audit_interval"] += 1
                res["examples"].setdefault("p2_audit_interval", (t, D) + v)
            # pattern W* A* R*: robust order violations among strictly optimal actions
            strictW = c["wait_strict"]
            strictR = c["qr"] < np.minimum(c["qw"], c["qa"]) - MARGIN * (1 + np.abs(c["m"]))
            lab = np.full(dp.G, -1)
            lab[strictW] = 0
            lab[aud_strict] = 1
            lab[strictR] = 2
            seq = lab[lab >= 0]
            if seq.size and np.any(np.diff(seq) < 0):
                res["p2_pattern"] += 1
                k = int(np.flatnonzero(np.diff(seq) < 0)[0])
                res["examples"].setdefault("p2_pattern", (t, D, int(seq[k]), int(seq[k + 1])))
            # monotonicity in D, compared with the previous D at the same t
            if prev is not None:
                pr = prev
                # T1(iii): rec(D) subset of rec(D-1): violation if strictly in rec at D and strictly out at D-1
                strict_in_now = c["qr"] < np.minimum(c["qw"], c["qa"]) - MARGIN * (1 + np.abs(c["m"]))
                viol = strict_in_now & pr["rec_out"]
                if p.h <= p.r1 + 1e-12:
                    res["t1_monoD_cells"] += 1
                    if viol.any():
                        res["t1_monoD"] += 1
                        res["examples"].setdefault("t1_monoD", (t, D, int(np.flatnonzero(viol)[0])))
                # proposal's conjecture: theta_r nonincreasing in D -> violation if rec(D-1) strictly
                # contains beliefs that are strictly out at D (i.e., threshold moved up)
                viol_up = pr["strict_rec"] & rec_out
                if viol_up.any():
                    res["p2_thr_r_up"] += 1
                    res["examples"].setdefault("p2_thr_r_up", (t, D, int(np.flatnonzero(viol_up)[0])))
                # first non-wait threshold: violation if strictly act (non-W) at D-1 and strictly wait at D
                viol_a = pr["strict_act"] & strictW
                if viol_a.any():
                    res["p2_thr_a_up"] += 1
                    res["examples"].setdefault("p2_thr_a_up", (t, D, int(np.flatnonzero(viol_a)[0])))
            prev = dict(rec_out=rec_out,
                        strict_rec=c["qr"] < np.minimum(c["qw"], c["qa"]) - MARGIN * (1 + np.abs(c["m"])),
                        strict_act=c["qw"] > np.minimum(c["qr"], c["qa"]) + MARGIN * (1 + np.abs(c["m"])))
    return res, dp


def random_params(rng: np.random.Generator, Tmin=3, Tmax=8):
    T = int(rng.integers(Tmin, Tmax + 1))
    s0 = float(rng.uniform(0.0, 0.4))
    s1 = float(rng.uniform(s0, 0.95))
    r1 = float(rng.choice([0.0, rng.uniform(0, 1), rng.uniform(0, 6)]))
    return Params(T=T, pi0=float(rng.uniform(0.02, 0.7)),
                  rho=float(rng.choice([0.0, rng.uniform(0, 0.15)])),
                  Delta=1, grow_p=1.0, lam=float(rng.uniform(0.02, 0.5)),
                  phi=float(rng.uniform(0, 1)), H=float(rng.uniform(1, 40)),
                  h=float(rng.uniform(0, 4)), r0=float(rng.uniform(0, 6)), r1=r1,
                  c_a=float(rng.uniform(0.1, 3)), d=float(rng.uniform(0.5, 1.0)),
                  alpha=float(rng.uniform(0, 0.3)), s0=s0, s1=s1,
                  c_fr=float(rng.choice([0.0, rng.uniform(0, 2)])), D0=int(rng.integers(0, 7)))


def exact_actions(p: Params, t: int, D: int, bs):
    """Exact optimal action and Q-values at the listed beliefs (ExactDP, fresh cache)."""
    e = ExactDP(p)
    out = []
    for b in bs:
        qw, qr, qa = e.Q(t, D, float(b))
        out.append((float(b), qw, qr, qa, e.action(t, D, float(b))))
    return out


# ----------------------------------------------------------------------------- exhaustive enumeration
def enumerate_policy_costs(p: Params, t: int, D: int, b: float):
    """Expected costs of ALL deterministic policies from decision node (t, D, b).

    Returns dict action-class -> np.ndarray of policy costs (class 'A' covers the four
    post-audit sub-decision combinations). Feasible when T - t <= 2.
    """
    from model import bayes_alarm, predict, audit_split

    def node_costs(t, D, b):
        if t == p.T:
            return np.array([b * p.h * D])
        out = enumerate_policy_costs_node(t, D, b)
        return np.concatenate(list(out.values()))

    def combine(imm, children):
        """children: list of (prob, cost-array); Cartesian product of independent sub-policies."""
        acc = np.array([imm])
        for pr, arr in children:
            if pr <= 0:
                continue
            acc = (acc[:, None] + pr * arr[None, :]).ravel()
        return acc

    def after_action(t, D, b, act):
        """Children of a period in which act in {W,R} is taken at belief b (no further audit)."""
        ch = []
        imm = 0.0
        if act == R:
            imm += p.R(D) + (1 - b) * p.c_fr
            for inc, pr in p.incs():
                for o in (0, 1):
                    po = p.s0 if o == 1 else 1 - p.s0
                    ch.append((pr * po, node_costs(t + 1, inc, p.rho_at(t))))
            return imm, ch
        for inc, pr in p.incs():
            if b > 0:
                imm += pr * b * p.lam * (p.phi * p.H + p.R(D + inc))
                ch.append((pr * b * p.lam, node_costs(t + 1, 0, p.rho_at(t))))
            for o in (0, 1):
                P, post = bayes_alarm(p, np.float64(b), o)
                P, post = float(P), float(post)
                ch.append((pr * P, node_costs(t + 1, D + inc, float(predict(p, post, t)))))
        return imm, ch

    def enumerate_policy_costs_node(t, D, b):
        res = {}
        for act in (W, R):
            imm, ch = after_action(t, D, b, act)
            res[ACTION_NAMES_LOCAL[act]] = combine(imm, ch)
        Pp, bp, Pm, bm = (float(x) for x in audit_split(p, np.float64(b)))
        arrs = []
        for sp in (W, R):
            for sm in (W, R):
                imm = p.c_a
                ch = []
                for P, bb, s in ((Pp, bp, sp), (Pm, bm, sm)):
                    if P <= 0:
                        continue
                    i2, c2 = after_action(t, D, bb, s)
                    imm += P * i2
                    ch += [(P * pr, arr) for pr, arr in c2]
                arrs.append(combine(imm, ch))
        res["A"] = np.concatenate(arrs)
        return res

    return enumerate_policy_costs_node(t, D, b)


ACTION_NAMES_LOCAL = {W: "W", R: "R", A: "A"}
