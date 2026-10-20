"""Multi-carrier audit game (Definition 5, Theorem 4, Proposition 5 of the paper).

Defender commits to per-period audit probabilities x_i (sum x_i <= B). The attacker
observes x and poisons one carrier i at t = 0 (success s_i, cost kappa_i) or abstains.
Per carrier (assumptions A8-A9): each period the carrier is audited with probability
x_i and a tainted carrier is then detected with probability d and recovered at cost
R(D) = r0 + r1 D; otherwise it activates with hazard lam_i (irreversible phi_i H plus
mandatory recovery at R(D + Delta_i)); otherwise derived work grows by Delta_i.
At T an undetected taint leaves residual reversible harm h * D_T. A9 assumes h >= r1, under which
leaving a detected taint in place never saves more than r0 (supplement), and D = 0 at poisoning.

L_i(x; r1) = F_i(x) + r1 E_i(x) is the expected damage of a successful poison. In the code the two
terms are called A_i and B_i for historical reasons (AB returns (F_i, E_i)).
minimax() solves min_x max_i U_i(x_i). This equals the commitment of Definition 5 (which adds the
abstention floor 0) whenever its level is positive, which A10 requires (u* > 0).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Carrier:
    name: str
    Delta: float   # derived-work growth per period
    lam: float     # activation hazard
    phi: float     # irreversible fraction
    s: float       # attack success probability
    kappa: float   # attack cost


@dataclass(frozen=True)
class GameParams:
    T: int = 20
    H: float = 20.0
    h: float = 10.0
    r0: float = 3.0
    r1: float = 1.0
    d: float = 0.9
    B: float = 0.6


CARRIERS = (
    Carrier("memory", 1.0, 0.05, 0.0, 0.9, 0.5),
    Carrier("skill", 0.6, 0.10, 0.6, 0.7, 1.0),
    Carrier("queue", 0.2, 0.04, 1.0, 0.5, 1.5),
    Carrier("branch", 0.8, 0.06, 0.4, 0.6, 1.0),
)


def AB(c: Carrier, g: GameParams, x):
    """Closed forms of F_i(x) and E_i(x) (Theorem 4(i)); x may be an array."""
    x = np.asarray(x, dtype=float)
    dx = g.d * x
    q = (1.0 - dx) * (1.0 - c.lam)
    t = np.arange(g.T)
    qt = q[..., None] ** t                       # q^t
    S = qt.sum(-1)
    A = (1.0 - dx) * c.lam * c.phi * g.H * S + g.r0 * (1.0 - q ** g.T) + g.h * c.Delta * g.T * q ** g.T
    Bv = (qt * (dx[..., None] * c.Delta * t + (1.0 - dx[..., None]) * c.lam * c.Delta * (t + 1))).sum(-1)
    return A, Bv


def damage(c: Carrier, g: GameParams, x):
    A, Bv = AB(c, g, x)
    return A + g.r1 * Bv


def utility(c: Carrier, g: GameParams, x):
    return c.s * damage(c, g, x) - c.kappa


def dU_dx(c: Carrier, g: GameParams, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    lo = np.clip(x - eps, 0, 1)
    hi = np.clip(x + eps, 0, 1)
    return (utility(c, g, hi) - utility(c, g, lo)) / (hi - lo)


def coverage_for_level(c: Carrier, g: GameParams, u, iters=80):
    """Smallest x in [0,1] with U(x) <= u (np.inf if none)."""
    if utility(c, g, 0.0) <= u:
        return 0.0
    if utility(c, g, 1.0) > u:
        # U may be non-monotone; scan for any feasible point
        xs = np.linspace(0, 1, 2001)
        ok = np.flatnonzero(utility(c, g, xs) <= u)
        return float(xs[ok[0]]) if ok.size else np.inf
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if utility(c, g, mid) <= u:
            hi = mid
        else:
            lo = mid
    return hi


def minimax(carriers, g: GameParams, iters=80):
    """Defender's optimal commitment against a best-responding attacker (water-filling)."""
    U0 = np.array([float(utility(c, g, 0.0)) for c in carriers])
    U1 = np.array([float(utility(c, g, 1.0)) for c in carriers])
    lo, hi = min(U1.min(), U0.min()) - 1.0, U0.max()
    def need(u):
        xs = [coverage_for_level(c, g, u) for c in carriers]
        return sum(xs), xs
    tot, xs = need(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        tot, _ = need(mid)
        if tot <= g.B:
            hi = mid
        else:
            lo = mid
    tot, xs = need(hi)
    x = np.array(xs)
    u = max(float(utility(c, g, xi)) for c, xi in zip(carriers, x))
    support = [i for i, c in enumerate(carriers) if utility(c, g, x[i]) >= u - 1e-6]
    return x, u, support


def best_response(carriers, g: GameParams, x):
    U = np.array([float(utility(c, g, xi)) for c, xi in zip(carriers, x)])
    i = int(np.argmax(U))
    return (i if U[i] > 0 else -1), U


def predicted_derivative(carriers, g: GameParams, x, support):
    """Theorem 4(iii): dx_i*/dr1 = w_i (s_i E_i - Ebar) with w_i = 1/|dU_i/dx| on the attack set."""
    w, sb = [], []
    for i in support:
        c = carriers[i]
        _, Bv = AB(c, g, x[i])
        w.append(1.0 / abs(float(dU_dx(c, g, x[i]))))
        sb.append(c.s * float(Bv))
    w, sb = np.array(w), np.array(sb)
    bbar = float((w * sb).sum() / w.sum())
    der = np.zeros(len(carriers))
    for k, i in enumerate(support):
        der[i] = (sb[k] - bbar) * w[k]
    return der, bbar


def simulate_damage(c: Carrier, g: GameParams, x: float, n: int, rng: np.random.Generator):
    """Monte Carlo estimate of L_i(x; r1) for a successful poison (validation of Theorem 4(i))."""
    D = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    loss = np.zeros(n)
    for t in range(g.T):
        det = alive & (rng.random(n) < x) & (rng.random(n) < g.d)
        loss += det * (g.r0 + g.r1 * D)
        alive &= ~det
        act = alive & (rng.random(n) < c.lam)
        loss += act * (c.phi * g.H + g.r0 + g.r1 * (D + c.Delta))
        alive &= ~act
        D = D + c.Delta
    loss += alive * g.h * D
    return loss


def random_game(rng: np.random.Generator, n=None):
    """Random instance satisfying A8-A9 at its own r1 (h >= r1). Checks at other r1 must re-check h >= r1."""
    n = int(rng.integers(2, 6)) if n is None else n
    cs = tuple(Carrier(f"c{i}", float(rng.uniform(0.1, 1.5)), float(rng.uniform(0.01, 0.3)),
                       float(rng.uniform(0, 1)), float(rng.uniform(0.2, 1.0)), float(rng.uniform(0, 3)))
               for i in range(n))
    h = float(rng.uniform(0.5, 10))
    g = GameParams(T=int(rng.integers(5, 30)), H=float(rng.uniform(5, 40)), h=h,
                   r0=float(rng.uniform(0, 4)), r1=float(rng.uniform(0, h)), d=float(rng.uniform(0.5, 1.0)),
                   B=float(rng.uniform(0.1, 1.5)))
    return cs, g


def a10_monotone(cs, g, n=401):
    """Strict decrease of every U_i on [0, min(1, B)] (the part of A10 a commitment can reach)."""
    xs = np.linspace(0.0, min(1.0, g.B), n)
    return all(np.all(np.diff(utility(c, g, xs)) < 0) for c in cs)
