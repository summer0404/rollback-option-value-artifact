"""Restricted recovery-audit POMDP (Definitions 1-5 of the paper).

One carrier with a latent taint bit z, observable derived work D since the last
verified checkpoint, and three adaptation actions per period:
    W  wait
    R  recover (rollback to the checkpoint and replay clean work)
    A  audit, then recover or wait depending on the audit outcome.

Timeline of period t (t = 0 .. T-1), matching assumptions A1-A5:
  1. the defender acts on (t, D, b) where b = P(z = 1 | history);
  2. derived work grows by an increment xi (A2);
  3. if z = 1 the carrier activates with probability lam (A4): irreversible
     harm phi*H, observed, and mandatory recovery at cost R(D + xi);
  4. otherwise the passive monitor emits o in {0,1} (A3) and b is updated;
  5. a clean carrier is (re)infected with probability rho (A1).
Terminal cost at T: h * D_T if z_T = 1 (residual reversible harm, A5).

Everything is deterministic given the parameters; Monte Carlo helpers take an
explicit numpy Generator.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from functools import lru_cache
import math
import numpy as np

W, R, A = 0, 1, 2
ACTION_NAMES = {W: "W", R: "R", A: "A"}
EPS = 1e-9


@dataclass(frozen=True)
class Params:
    T: int = 20          # horizon (periods / milestones)
    pi0: float = 0.3     # prior P(z_0 = 1)
    rho: float = 0.0     # insertion hazard per period for a clean carrier
    rho_until: int = 10 ** 9  # end of the exposure window (A1)
    rho_late: float = 0.0     # hazard after the exposure window (floor)
    Delta: int = 1       # derived-work increment (integer units)
    grow_p: float = 1.0  # increment happens with this probability (1.0 = deterministic)
    lam: float = 0.1     # activation hazard of a tainted carrier
    phi: float = 0.5     # irreversible fraction of activation harm
    H: float = 20.0      # activation harm
    h: float = 1.0       # residual reversible harm per unit of tainted derived work at T
    r0: float = 0.5      # fixed recovery cost
    r1: float = 1.0      # replay cost per unit of derived work
    c_a: float = 1.0     # audit cost
    d: float = 0.9       # audit power
    alpha: float = 0.05  # audit false-alarm rate
    s0: float = 0.1      # passive alarm rate, clean
    s1: float = 0.4      # passive alarm rate, tainted
    c_fr: float = 0.0    # extra penalty for recovering a clean carrier
    D0: int = 0          # derived work at t = 0

    def R(self, D):
        return self.r0 + self.r1 * D

    def rho_at(self, t):
        return self.rho if t < self.rho_until else self.rho_late

    def incs(self):
        """Distribution of the derived-work increment xi: list of (inc, prob)."""
        if self.grow_p >= 1.0:
            return [(self.Delta, 1.0)]
        if self.grow_p <= 0.0:
            return [(0, 1.0)]
        return [(0, 1.0 - self.grow_p), (self.Delta, self.grow_p)]

    def mean_inc(self):
        return self.Delta * min(max(self.grow_p, 0.0), 1.0)

    def asdict(self):
        return asdict(self)


def bayes_alarm(p: Params, b, o):
    """Posterior after no activation and alarm o (vectorised); returns (P(no act, o | b), post)."""
    l1 = p.s1 if o == 1 else 1.0 - p.s1
    l0 = p.s0 if o == 1 else 1.0 - p.s0
    num = b * (1.0 - p.lam) * l1
    P = num + (1.0 - b) * l0
    post = np.where(P > 0, num / np.where(P > 0, P, 1.0), 0.0)
    return P, post


def predict(p: Params, b, t=0):
    r = p.rho_at(t)
    return b + (1.0 - b) * r


def audit_split(p: Params, b):
    Pp = b * p.d + (1.0 - b) * p.alpha
    Pm = 1.0 - Pp
    bp = np.where(Pp > 0, b * p.d / np.where(Pp > 0, Pp, 1.0), 0.0)
    bm = np.where(Pm > 0, b * (1.0 - p.d) / np.where(Pm > 0, Pm, 1.0), 0.0)
    return Pp, bp, Pm, bm


# ----------------------------------------------------------------------------- grid DP
class GridDP:
    """Backward induction on a uniform belief grid with linear interpolation.

    V[t] has shape (nD, G). D ranges over 0 .. Dmax = D0 + Delta*T.
    """

    def __init__(self, p: Params, G: int = 1001):
        self.p = p
        self.G = G
        self.grid = np.linspace(0.0, 1.0, G)
        self.Dmax = p.D0 + p.Delta * p.T
        self.nD = self.Dmax + 1
        self.V = [None] * (p.T + 1)
        self.solve()

    # value interpolation at arbitrary (D, b) arrays for period t
    def interp(self, t, D, b):
        V = self.V[t]
        x = np.clip(b, 0.0, 1.0) * (self.G - 1)
        i0 = np.minimum(x.astype(np.int64), self.G - 2)
        w = x - i0
        D = np.minimum(D, self.Dmax)
        return V[D, i0] * (1.0 - w) + V[D, i0 + 1] * w

    def qW(self, t, D, b):
        p = self.p
        D = np.asarray(D)
        b = np.asarray(b, dtype=float)
        out = np.zeros(np.broadcast(D, b).shape)
        zero = np.zeros_like(D)
        rho_b = np.full(out.shape, p.rho_at(t))
        v_reset = self.interp(t + 1, np.broadcast_to(zero, out.shape), rho_b)
        for inc, pr in p.incs():
            Dn = np.minimum(D + inc, self.Dmax)
            acc = b * p.lam * (p.phi * p.H + p.R(D + inc) + v_reset)
            for o in (0, 1):
                P, post = bayes_alarm(p, b, o)
                acc = acc + P * self.interp(t + 1, np.broadcast_to(Dn, out.shape), predict(p, post, t))
            out = out + pr * acc
        return out

    def qR(self, t, D, b):
        p = self.p
        D = np.asarray(D)
        b = np.asarray(b, dtype=float)
        shape = np.broadcast(D, b).shape
        cont = 0.0
        for inc, pr in p.incs():
            cont = cont + pr * self.interp(t + 1, np.full(shape, min(inc, self.Dmax)), np.full(shape, p.rho_at(t)))
        return p.R(D) + (1.0 - b) * p.c_fr + cont

    def qA(self, t, D, b, return_post=False):
        p = self.p
        b = np.asarray(b, dtype=float)
        Pp, bp, Pm, bm = audit_split(p, b)
        wp, rp = self.qW(t, D, bp), self.qR(t, D, bp)
        wm, rm = self.qW(t, D, bm), self.qR(t, D, bm)
        q = p.c_a + Pp * np.minimum(wp, rp) + Pm * np.minimum(wm, rm)
        if return_post:
            return q, rp <= wp + EPS, rm <= wm + EPS
        return q

    def qall(self, t, D, b):
        return self.qW(t, D, b), self.qR(t, D, b), self.qA(t, D, b)

    def solve(self):
        p = self.p
        Dcol = np.arange(self.nD)[:, None]
        bgrid = self.grid[None, :]
        self.V[p.T] = bgrid * p.h * Dcol * np.ones_like(bgrid)
        for t in range(p.T - 1, -1, -1):
            Db = np.broadcast_to(Dcol, (self.nD, self.G))
            bb = np.broadcast_to(bgrid, (self.nD, self.G))
            qw, qr, qa = self.qall(t, Db, bb)
            self.V[t] = np.minimum(np.minimum(qw, qr), qa)

    def action(self, t, D, b):
        """Canonical optimal action (ties prefer W, then R, then A)."""
        qw, qr, qa = self.qall(t, D, b)
        m = np.minimum(np.minimum(qw, qr), qa)
        return np.where(qw <= m + EPS, W, np.where(qr <= m + EPS, R, A))

    def value(self, t=0, D=None, b=None):
        p = self.p
        D = p.D0 if D is None else D
        b = p.pi0 if b is None else b
        return float(self.interp(t, np.array([D]), np.array([b]))[0])


# ----------------------------------------------------------------------------- exact recursion
class ExactDP:
    """Exact backward induction over the reachable belief tree (no grid).

    Used to verify grid results, structural counterexamples and closed forms.
    A policy can be supplied as policy(t, D, b) -> action and post(t, D, b) -> bool
    (recover after the audit outcome); otherwise the optimal policy is used.
    """

    def __init__(self, p: Params, policy=None, post=None, nd=12):
        self.p = p
        self.policy = policy
        self.post = post
        self.nd = nd
        self.cache = {}

    def key(self, t, D, b):
        return (t, D, round(b, self.nd))

    def V(self, t, D, b):
        k = self.key(t, D, b)
        if k in self.cache:
            return self.cache[k]
        p = self.p
        if t == p.T:
            v = b * p.h * D
        else:
            qw, qr, qa = self.Q(t, D, b)
            if self.policy is None:
                v = min(qw, qr, qa)
            else:
                a = self.policy(t, D, b)
                v = (qw, qr, qa)[a]
        self.cache[k] = v
        return v

    def qW(self, t, D, b):
        p = self.p
        tot = 0.0
        for inc, pr in p.incs():
            acc = 0.0
            if b > 0:
                acc += b * p.lam * (p.phi * p.H + p.R(D + inc) + self.V(t + 1, 0, p.rho_at(t)))
            for o in (0, 1):
                P, post = bayes_alarm(p, np.float64(b), o)
                P, post = float(P), float(post)
                if P > 0:
                    acc += P * self.V(t + 1, D + inc, float(predict(p, post, t)))
            tot += pr * acc
        return tot

    def qR(self, t, D, b):
        p = self.p
        cont = sum(pr * self.V(t + 1, inc, p.rho_at(t)) for inc, pr in p.incs())
        return p.R(D) + (1.0 - b) * p.c_fr + cont

    def qA(self, t, D, b):
        p = self.p
        Pp, bp, Pm, bm = (float(x) for x in audit_split(p, np.float64(b)))
        tot = p.c_a
        for P, bb in ((Pp, bp), (Pm, bm)):
            if P <= 0:
                continue
            w, r = self.qW(t, D, bb), self.qR(t, D, bb)
            if self.post is None:
                tot += P * min(w, r)
            else:
                tot += P * (r if self.post(t, D, bb) else w)
        return tot

    def Q(self, t, D, b):
        return self.qW(t, D, b), self.qR(t, D, b), self.qA(t, D, b)

    def action(self, t, D, b):
        qw, qr, qa = self.Q(t, D, b)
        m = min(qw, qr, qa)
        if qw <= m + EPS:
            return W
        if qr <= m + EPS:
            return R
        return A


# ----------------------------------------------------------------------------- policies
class DPPolicy:
    name = "dp"

    def __init__(self, dp: GridDP):
        self.dp = dp

    def act(self, t, D, b):
        return self.dp.action(t, D, b)

    def post(self, t, D, b):
        qw, qr = self.dp.qW(t, D, b), self.dp.qR(t, D, b)
        return qr <= qw + EPS


class MyopicPolicy:
    """Definition 4 of the paper (myopic audit rule): the one-step audit-or-defer inequality.

    audit  iff  b d (r1*Delta + lam*phi*H) >= c_a + alpha (1-b) c_fq(D)
    after an audit, recover iff  b' (r1*Delta + lam*phi*H) >= (1-b') c_fq(D)
    with c_fq(D) = R(D) + c_fr the cost of a false rollback. Never recovers blind.
    """
    name = "myopic"

    def __init__(self, p: Params):
        self.p = p
        self.g = p.r1 * p.mean_inc() + p.lam * p.phi * p.H

    def act(self, t, D, b):
        p = self.p
        cfq = p.R(D) + p.c_fr
        return np.where(b * p.d * self.g >= p.c_a + p.alpha * (1.0 - b) * cfq, A, W)

    def post(self, t, D, b):
        p = self.p
        cfq = p.R(D) + p.c_fr
        return b * self.g >= (1.0 - b) * cfq


class FixedPolicy:
    """Simple model-level baselines: never audit, or audit every k periods."""

    def __init__(self, name, every=None):
        self.name = name
        self.every = every

    def act(self, t, D, b):
        b = np.asarray(b)
        if self.every is None:
            return np.full(b.shape, W)
        return np.full(b.shape, A if t % self.every == 0 else W)

    def post(self, t, D, b):
        return np.asarray(b) >= 0.5


# ----------------------------------------------------------------------------- Monte Carlo
def simulate(p: Params, policy, n: int, rng: np.random.Generator):
    """Vectorised Monte Carlo of a policy in the restricted model. Returns dict of arrays."""
    z = rng.random(n) < p.pi0
    D = np.full(n, p.D0, dtype=np.int64)
    b = np.full(n, p.pi0)
    m = {k: np.zeros(n) for k in ("audits", "rec_chosen", "rec_post", "rec_forced", "rec_false",
                                   "irrev", "replay", "audit_cost", "fr_cost", "resid", "activations")}
    for t in range(p.T):
        a = np.asarray(policy.act(t, D, b))
        a = np.broadcast_to(a, (n,)).copy()
        rec = a == R
        isA = a == A
        if isA.any():
            u = rng.random(n)
            pos = np.where(z, u < p.d, u < p.alpha)
            Pp, bp, Pm, bm = audit_split(p, b)
            bpost = np.where(pos, bp, bm)
            post_rec = np.asarray(policy.post(t, D, bpost))
            post_rec = np.broadcast_to(post_rec, (n,))
            m["audits"] += isA
            m["audit_cost"] += isA * p.c_a
            b = np.where(isA, bpost, b)
            prec = isA & post_rec
            m["rec_post"] += prec
            rec = rec | prec
        m["rec_chosen"] += (a == R)
        if rec.any():
            m["replay"] += rec * p.R(D)
            fr = rec & ~z
            m["rec_false"] += fr
            m["fr_cost"] += fr * p.c_fr
            z = z & ~rec
            D = np.where(rec, 0, D)
            b = np.where(rec, 0.0, b)
        # derived work
        if p.grow_p >= 1.0:
            inc = np.full(n, p.Delta)
        else:
            inc = np.where(rng.random(n) < p.grow_p, p.Delta, 0)
        # activation
        actv = z & (rng.random(n) < p.lam)
        m["activations"] += actv
        m["irrev"] += actv * p.phi * p.H
        m["replay"] += actv * p.R(D + inc)
        m["rec_forced"] += actv
        D = np.where(actv, 0, D + inc)
        z = z & ~actv
        # alarm and belief
        u = rng.random(n)
        o = np.where(z, u < p.s1, u < p.s0)
        _, post1 = bayes_alarm(p, b, 1)
        _, post0 = bayes_alarm(p, b, 0)
        b = np.where(actv, 0.0, np.where(o, post1, post0))
        # insertion
        newz = (~z) & (rng.random(n) < p.rho_at(t))
        z = z | newz
        b = predict(p, b, t)
    m["resid"] = p.h * D * z
    m["loss"] = m["irrev"] + m["replay"] + m["audit_cost"] + m["fr_cost"] + m["resid"]
    return m


# ----------------------------------------------------------------------------- Theorem 3 closed forms (A7 submodel)
def submodel_ok(p: Params, tol=1e-12):
    """Assumption A7 (perfect audit, single insertion, uninformative monitor, R(D) <= hD)."""
    return (abs(p.d - 1.0) < tol and p.alpha < tol and p.rho < tol and abs(p.s0 - p.s1) < tol
            and p.c_fr < tol and p.grow_p >= 1.0 and p.r0 <= (p.h - p.r1) * p.D0 + tol)


def corner_costs(p: Params):
    """Closed-form costs of the four corner policies of Theorem 3.

    J_A(tau): first audit at tau (if no activation before), recover on a positive audit
    J_B(0):  blind recovery at 0;  J_N: never intervene.
    """
    lam, pi, T = p.lam, p.pi0, p.T
    Ds = lambda s: p.D0 + p.Delta * s

    def q(tau):
        return pi * (1 - lam) ** tau + (1 - pi)

    def JA(tau):
        tot = 0.0
        for s in range(tau):
            tot += lam * (1 - lam) ** s * (p.phi * p.H + p.R(Ds(s + 1)))
        tot += (1 - lam) ** tau * p.R(Ds(tau))
        return pi * tot + p.c_a * q(tau)

    JB0 = p.R(Ds(0))
    tot = 0.0
    for s in range(T):
        tot += lam * (1 - lam) ** s * (p.phi * p.H + p.R(Ds(s + 1)))
    JN = pi * (tot + (1 - lam) ** T * p.h * Ds(T))
    JA_all = [JA(tau) for tau in range(T)]
    return {"JA": JA_all, "JB0": JB0, "JN": JN}


def theorem2_gain(p: Params):
    """g = r1*Delta + lam*(phi*H - c_a): sign of J_A(tau+1) - J_A(tau)."""
    return p.r1 * p.Delta + p.lam * (p.phi * p.H - p.c_a)


def S_T(lam, T):
    return (1 - (1 - lam) ** T) / lam


def myopic_gap_bound(p: Params):
    """Upper bound of Theorem 3(iv) when the myopic rule defers at t = 0."""
    DT = p.D0 + p.Delta * p.T
    return p.c_a * S_T(p.lam, p.T) + p.pi0 * (1 - p.lam) ** p.T * max(0.0, p.h * DT - p.R(DT))


def myopic_defers(p: Params):
    g0 = p.r1 * p.Delta + p.lam * p.phi * p.H
    return p.pi0 * g0 < p.c_a


class GridPolicyEval(GridDP):
    """Exact-on-grid expected cost of a fixed policy (same transitions as GridDP).

    policy.act(t, D, b) -> actions, policy.post(t, D, b) -> recover-after-audit flags.
    """

    def __init__(self, p: Params, policy, G: int = 1001):
        self.policy = policy
        super().__init__(p, G)

    def qA(self, t, D, b, return_post=False):
        p = self.p
        b = np.asarray(b, dtype=float)
        Pp, bp, Pm, bm = audit_split(p, b)
        D = np.broadcast_to(np.asarray(D), b.shape)
        rp = np.asarray(self.policy.post(t, D, bp), dtype=bool)
        rm = np.asarray(self.policy.post(t, D, bm), dtype=bool)
        vp = np.where(rp, self.qR(t, D, bp), self.qW(t, D, bp))
        vm = np.where(rm, self.qR(t, D, bm), self.qW(t, D, bm))
        return p.c_a + Pp * vp + Pm * vm

    def solve(self):
        p = self.p
        Dcol = np.arange(self.nD)[:, None]
        bgrid = self.grid[None, :]
        self.V[p.T] = bgrid * p.h * Dcol * np.ones_like(bgrid)
        for t in range(p.T - 1, -1, -1):
            Db = np.broadcast_to(Dcol, (self.nD, self.G))
            bb = np.broadcast_to(bgrid, (self.nD, self.G))
            a = np.broadcast_to(np.asarray(self.policy.act(t, Db, bb)), Db.shape)
            qw, qr, qa = self.qW(t, Db, bb), self.qR(t, Db, bb), self.qA(t, Db, bb)
            self.V[t] = np.where(a == W, qw, np.where(a == R, qr, qa))


class FrozenDPPolicy:
    """Act with a DP solved for a (possibly misspecified) planning model."""

    def __init__(self, dp: GridDP, name="dp"):
        self.dp = dp
        self.name = name

    def act(self, t, D, b):
        return self.dp.action(t, D, b)

    def post(self, t, D, b):
        return self.dp.qR(t, D, b) <= self.dp.qW(t, D, b) + EPS


class MyopicAuditDPRecover:
    """Robustness variant: the myopic audit rule with the DP's recovery decisions."""
    name = "myopic_dprec"

    def __init__(self, p: Params, dp: GridDP):
        self.m = MyopicPolicy(p)
        self.dp = dp

    def act(self, t, D, b):
        return self.m.act(t, D, b)

    def post(self, t, D, b):
        return self.dp.qR(t, D, b) <= self.dp.qW(t, D, b) + EPS


def simulate_misspec(p_true: Params, p_plan: Params, policy, n: int, rng: np.random.Generator):
    """Monte Carlo of a policy whose belief is tracked with the planner's model p_plan while the
    state evolves under p_true (model misspecification of the Knowledge base)."""
    p = p_true
    z = rng.random(n) < p.pi0
    D = np.full(n, p.D0, dtype=np.int64)
    b = np.full(n, p_plan.pi0)
    m = {k: np.zeros(n) for k in ("audits", "irrev", "replay", "audit_cost", "fr_cost", "rec_false")}
    for t in range(p.T):
        a = np.broadcast_to(np.asarray(policy.act(t, D, b)), (n,)).copy()
        rec = a == R
        isA = a == A
        if isA.any():
            u = rng.random(n)
            pos = np.where(z, u < p.d, u < p.alpha)
            _, bp, _, bm = audit_split(p_plan, b)
            bpost = np.where(pos, bp, bm)
            post_rec = np.broadcast_to(np.asarray(policy.post(t, D, bpost)), (n,))
            m["audits"] += isA
            m["audit_cost"] += isA * p.c_a
            b = np.where(isA, bpost, b)
            rec = rec | (isA & post_rec)
        if rec.any():
            m["replay"] += rec * p.R(D)
            fr = rec & ~z
            m["rec_false"] += fr
            m["fr_cost"] += fr * p.c_fr
            z = z & ~rec
            D = np.where(rec, 0, D)
            b = np.where(rec, 0.0, b)
        inc = np.full(n, p.Delta) if p.grow_p >= 1.0 else np.where(rng.random(n) < p.grow_p, p.Delta, 0)
        actv = z & (rng.random(n) < p.lam)
        m["irrev"] += actv * p.phi * p.H
        m["replay"] += actv * p.R(D + inc)
        D = np.where(actv, 0, D + inc)
        z = z & ~actv
        u = rng.random(n)
        o = np.where(z, u < p.s1, u < p.s0)
        _, post1 = bayes_alarm(p_plan, b, 1)
        _, post0 = bayes_alarm(p_plan, b, 0)
        b = np.where(actv, 0.0, np.where(o, post1, post0))
        z = z | ((~z) & (rng.random(n) < p.rho_at(t)))
        b = predict(p_plan, b, t)
    m["resid"] = p.h * D * z
    m["loss"] = m["irrev"] + m["replay"] + m["audit_cost"] + m["fr_cost"] + m["resid"]
    return m
