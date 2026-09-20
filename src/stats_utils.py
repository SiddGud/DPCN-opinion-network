"""Shared statistics helpers: z-scores and false-discovery-rate correction."""
import numpy as np


def z_score(observed, null_values):
    # How many null standard deviations the real value sits above the null mean
    return (observed - null_values.mean()) / null_values.std()


def benjamini_hochberg(pvals):
    """False-discovery-rate adjusted p-values."""
    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 1.0
    for rank in range(n, 0, -1):
        idx = order[rank - 1]
        running = min(running, p[idx] * n / rank)
        adjusted[idx] = running
    return adjusted
