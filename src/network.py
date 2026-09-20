"""Step 2: turn answer vectors into a respondent-respondent similarity network."""
import numpy as np
import networkx as nx


def center_questions(data):
    # Subtract each question's class mean so shared "consensus" answers stop looking like similarity
    return data - data.mean(axis=0)


def pairwise_similarity(data, min_overlap, method="cosine"):
    """Cosine similarity between people, computed only on questions both of them answered."""
    values = data.to_numpy(dtype=float)
    if method == "pearson":
        # Pearson = cosine after also removing each person's own average level
        values = values - np.nanmean(values, axis=1, keepdims=True)
    mask = (~np.isnan(values)).astype(float)
    filled = np.nan_to_num(values)
    # Dot product over the overlapping questions only (missing entries are 0)
    dot = filled @ filled.T
    # Squared norm of person i restricted to the questions j also answered
    sq = (filled ** 2) @ mask.T
    denom = np.sqrt(sq * sq.T)
    sim = np.zeros_like(dot)
    ok = denom > 0
    sim[ok] = dot[ok] / denom[ok]
    # Discard pairs with too few shared answers
    overlap = mask @ mask.T
    sim[overlap < min_overlap] = 0.0
    np.fill_diagonal(sim, 0.0)
    return sim


def knn_graph(sim, labels, k):
    """Connect each person to their k most similar people (positive similarity only)."""
    g = nx.Graph()
    g.add_nodes_from(labels)
    n = sim.shape[0]
    for i in range(n):
        order = np.argsort(-sim[i])
        added = 0
        for j in order:
            if added == k:
                break
            if j == i or sim[i, j] <= 0:
                continue
            g.add_edge(labels[i], labels[j], weight=float(sim[i, j]))
            added += 1
    return g


def threshold_graph(sim, labels, quantile):
    """Keep only pairs whose similarity is above a chosen quantile of all positive similarities."""
    upper = sim[np.triu_indices_from(sim, k=1)]
    cut = np.quantile(upper[upper > 0], quantile)
    g = nx.Graph()
    g.add_nodes_from(labels)
    n = sim.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= cut:
                g.add_edge(labels[i], labels[j], weight=float(sim[i, j]))
    return g


def disparity_backbone(sim, labels, alpha):
    """Disparity filter (Serrano et al., 2009): keep edges that are unusually strong for a node."""
    w = np.clip(sim, 0, None)
    strength = w.sum(axis=1)
    degree = (w > 0).sum(axis=1)
    g = nx.Graph()
    g.add_nodes_from(labels)
    n = w.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if w[i, j] <= 0:
                continue
            keep = False
            for a, b in ((i, j), (j, i)):
                if degree[a] > 1:
                    # p-value: chance of an edge this strong if node a's strength were split at random
                    p = (1 - w[a, b] / strength[a]) ** (degree[a] - 1)
                    if p < alpha:
                        keep = True
            if keep:
                g.add_edge(labels[i], labels[j], weight=float(w[i, j]))
    return g


def shuffle_columns(data, rng):
    """Null model: shuffle each question independently across people."""
    shuffled = data.copy()
    for c in shuffled.columns:
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
    return shuffled
