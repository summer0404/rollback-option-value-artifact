"""Planner overhead: offline table solve time per carrier and online decision latency.

Writes results/overhead.json. Timings are machine dependent and reported as an indication.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testbed as TB  # noqa: E402
from model import GridDP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def main():
    solve = []
    for rep in range(5):
        for i in range(TB.NC):
            t0 = time.perf_counter()
            GridDP(TB.carrier_params(i, 1.0), G=501)
            solve.append(time.perf_counter() - t0)
    dp = GridDP(TB.carrier_params(1, 1.0), G=501)
    rng = np.random.default_rng(0)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        t = int(rng.integers(0, TB.T_MS))
        D = np.array([int(rng.integers(0, 10))])
        b = np.array([float(rng.random())])
        dp.qall(t, D, b)
    decide = (time.perf_counter() - t0) / n
    out = dict(solve_seconds_mean=float(np.mean(solve)), solve_seconds_max=float(np.max(solve)),
               decision_microseconds=1e6 * decide, grid=501, horizon=TB.T_MS)
    json.dump(out, open(os.path.join(RES, "overhead.json"), "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
