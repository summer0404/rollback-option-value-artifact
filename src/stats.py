"""Statistics helpers: bootstrap CIs (plain and cluster), paired Wilcoxon with Holm."""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def boot_ci(x, B=2000, alpha=0.05, seed=0, stat=np.mean, rows=50):
    """Percentile bootstrap. Index rows are drawn in blocks of `rows` from one Generator: PCG64 keeps the
    unused half of a 64-bit draw in its state, so the result equals drawing the full (B, n) matrix at once
    while peak memory stays at rows * n instead of B * n (1.6 to 3.2 GB for E1 and E2)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for s in range(0, B, rows):
        idx = rng.integers(0, len(x), size=(min(rows, B - s), len(x)))
        bs[s:s + len(idx)] = stat(x[idx], axis=1) if stat is np.mean else [stat(x[i]) for i in idx]
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def cluster_boot_ci(values_by_cluster, B=2000, alpha=0.05, seed=0):
    """values_by_cluster: list of arrays (one per cluster). Resamples clusters."""
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(v) for v in values_by_cluster])
    sizes = np.array([len(v) for v in values_by_cluster])
    sums = means * sizes
    n = len(means)
    idx = rng.integers(0, n, size=(B, n))
    bs = sums[idx].sum(1) / sizes[idx].sum(1)
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return float(st.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def vda(a, b):
    """Vargha-Delaney A12: P(a < b) + 0.5 P(a = b) (probability that a is lower)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] < b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return float(gt + 0.5 * eq)
