#!/usr/bin/env python3
"""Exact sign check of dU_i/dx for the E2 game and the Proposition 5
instance, replacing the sampled check game.a10_monotone (401 points) by exact rational arithmetic.

U_i(x) = s_i (F_i(x) + r1 E_i(x)) - kappa_i with q = (1 - d x)(1 - lam_i) is a polynomial in x
(game.AB, Theorem 4(i)). The script builds F_i and E_i with sympy Rationals from the decimal parameter
values, checks them against game.AB in floating point, and counts the real roots of
dU_i/dx = s_i (F_i' + r1 E_i') on [0, min(1, B)] with sympy's exact real-root counting. No root and
a negative value at one point prove dU_i/dx < 0 on the whole closed interval. Because dU_i/dx is affine
in r1, a proof at both ends of an r1 interval proves it for every r1 in between.

This certifies only the monotonicity part of A10 (the part a10_monotone samples). The other parts of
A10 (a unique attack set, interior coverage, a positive level) concern the minimax solution and stay
numerical checks.
Output: results/a10/a10_certificate.json
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = HERE
sys.path.insert(0, SRC)
import game as GM  # noqa: E402
from exp_theory import P3_CARRIERS, P3_GAME  # noqa: E402
from exp_e2 import R1_GRID  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "results", "exploratory", "a10"))
X = sp.Symbol("x")


def Q(v):
    return sp.Rational(repr(float(v)))      # shortest decimal repr of the float, as written in the code


def FE(c, g):
    """Exact polynomials F(x), E(x) of game.AB."""
    d, lam, phi, Delta = Q(g.d), Q(c.lam), Q(c.phi), Q(c.Delta)
    H, h, r0, T = Q(g.H), Q(g.h), Q(g.r0), int(g.T)
    dx = d * X
    q = sp.expand((1 - dx) * (1 - lam))
    qt = [sp.Integer(1)]
    for _ in range(T):
        qt.append(sp.expand(qt[-1] * q))
    S = sp.Add(*qt[:T])
    F = sp.expand((1 - dx) * lam * phi * H * S + r0 * (1 - qt[T]) + h * Delta * T * qt[T])
    E = sp.expand(sp.Add(*[qt[t] * (dx * Delta * t + (1 - dx) * lam * Delta * (t + 1)) for t in range(T)]))
    return sp.Poly(F, X), sp.Poly(E, X)


def check_poly_vs_code(c, g, F, E):
    xs = np.linspace(0, 1, 11)
    Fn, En = GM.AB(c, g, xs)
    fF, fE = sp.lambdify(X, F.as_expr()), sp.lambdify(X, E.as_expr())
    return float(max(np.max(np.abs(np.array([fF(x) for x in xs]) - Fn)),
                     np.max(np.abs(np.array([fE(x) for x in xs]) - En))))


def certify(dU, lo, hi):
    """(roots in [lo, hi], value at the midpoint, strictly negative on [lo, hi])."""
    n = dU.count_roots(lo, hi)
    mid = dU.eval((lo + hi) / 2)
    return int(n), float(mid), bool(n == 0 and mid < 0)


def run_game(name, carriers, g0, r1_lo, r1_hi, r1_grid):
    xmax = min(sp.Integer(1), Q(g0.B))
    out = dict(game=name, x_interval=[0, float(xmax)], r1_interval=[r1_lo, r1_hi], carriers={})
    all_ok = True
    for c in carriers:
        F, E = FE(c, g0)
        err = check_poly_vs_code(c, g0, F, E)
        s = Q(c.s)
        dF, dE = F.diff(X), E.diff(X)
        res = dict(degree=F.degree(), poly_vs_code_max_abs_err=err)
        for tag, r1 in (("r1_lo", r1_lo), ("r1_hi", r1_hi)):
            dU = sp.Poly(s * (dF.as_expr() + Q(r1) * dE.as_expr()), X)
            res[tag] = dict(zip(("roots_in_interval", "dU_at_mid", "negative_on_interval"), certify(dU, 0, xmax)))
        res["certified_all_r1_in_interval"] = res["r1_lo"]["negative_on_interval"] and res["r1_hi"]["negative_on_interval"]
        grid_ok = []
        for r1 in r1_grid:
            dU = sp.Poly(s * (dF.as_expr() + Q(r1) * dE.as_expr()), X)
            grid_ok.append(certify(dU, 0, xmax)[2])
        res["certified_each_grid_r1"] = all(grid_ok)
        res["sampled_a10_monotone_each_grid_r1"] = all(GM.a10_monotone((c,), replace(g0, r1=float(r1))) for r1 in r1_grid)
        out["carriers"][c.name] = res
        all_ok &= res["certified_all_r1_in_interval"]
        print(f"{name} {c.name}: degree {F.degree()}, poly-vs-code err {err:.1e}, "
              f"certified on r1 in [{r1_lo}, {r1_hi}]: {res['certified_all_r1_in_interval']}", flush=True)
    out["all_carriers_certified"] = bool(all_ok)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = [run_game("E2 (CARRIERS, GameParams())", GM.CARRIERS, GM.GameParams(), 0.0, 10.0, R1_GRID),
           run_game("Proposition 5 instance (P3_CARRIERS, P3_GAME)", P3_CARRIERS, P3_GAME, 0.0, 2.0,
                    [float(v) for v in np.round(np.linspace(0.0, 2.0, 21), 3)])]
    json.dump(dict(results=res, method="sympy exact rational real-root counting (Sturm) on dU/dx; "
                                       "dU/dx is affine in r1, so the two ends certify the interval"),
              open(os.path.join(OUT, "a10_certificate.json"), "w"), indent=1)
    print(json.dumps([dict(game=r["game"], all_carriers_certified=r["all_carriers_certified"]) for r in res], indent=1))


if __name__ == "__main__":
    main()
