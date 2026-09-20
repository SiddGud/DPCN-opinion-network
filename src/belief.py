"""Statement-statement belief network, structural balance and theme match. Owner: Krishna."""
import itertools

import numpy as np
import networkx as nx
from scipy import stats

from stats_utils import benjamini_hochberg
from communities import partition_labels, compare_partitions


def statement_network(data, q_level):
    """Nodes = statements; signed edges = FDR-significant Spearman correlations."""
    cols = list(data.columns)
    pairs = []
    for a, b in itertools.combinations(cols, 2):
        both = data[[a, b]].dropna()
        rho, p = stats.spearmanr(both[a], both[b])
        pairs.append((a, b, rho, p))
    p_adj = benjamini_hochberg([x[3] for x in pairs])
    g = nx.Graph()
    g.add_nodes_from(cols)
    for (a, b, rho, p), pa in zip(pairs, p_adj):
        if pa < q_level and not np.isnan(rho):
            if rho > 0:
                sign = 1
            else:
                sign = -1
            g.add_edge(a, b, weight=abs(rho), rho=rho, sign=sign)
    return g


def structural_balance(g, n_perm, seed):
    """Share of balanced triangles (+++ or +--) vs the same graph with shuffled signs."""
    def balanced_share(signs):
        balanced = 0
        total = 0
        for tri in (c for c in nx.enumerate_all_cliques(g) if len(c) == 3):
            prod = signs[frozenset(tri[:2])] * signs[frozenset(tri[1:])] * signs[frozenset((tri[0], tri[2]))]
            total += 1
            if prod > 0:
                balanced += 1
        return balanced, total

    real = {}
    for u, v, d in g.edges(data=True):
        real[frozenset((u, v))] = d["sign"]
    bal, total = balanced_share(real)
    rng = np.random.default_rng(seed)
    keys = list(real.keys())
    values = np.array([real[k] for k in keys])
    null = []
    for r in range(n_perm):
        perm = rng.permutation(values)
        b, _ = balanced_share(dict(zip(keys, perm)))
        null.append(b / total)
    n_negative = int((values < 0).sum())
    return bal / total, np.array(null), total, n_negative


def theme_match(g, parts):
    """Do statement communities reproduce the four survey themes?"""
    found = partition_labels(parts)
    theme = {}
    for n in g.nodes():
        theme[n] = "TESV".index(n[0])
    ari, nmi = compare_partitions(found, theme)
    groups = []
    for t in "TESV":
        groups.append({n for n in g.nodes() if n[0] == t})
    q_theme = nx.community.modularity(g, groups, weight="weight")
    return {"ari": float(ari), "nmi": float(nmi), "q_theme_partition": float(q_theme)}
