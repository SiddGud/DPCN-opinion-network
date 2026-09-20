"""Communities, null models, group profiles and the intensity (response-style) analysis. Owner: Siddhant."""
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from network import center_questions, pairwise_similarity, knn_graph, shuffle_columns
from stats_utils import benjamini_hochberg


def best_louvain(g, seed, runs=20):
    """Run Louvain several times and keep the partition with the highest modularity."""
    best = None
    best_q = -1.0
    for r in range(runs):
        parts = nx.community.louvain_communities(g, weight="weight", seed=seed + r)
        q = nx.community.modularity(g, parts, weight="weight")
        if q > best_q:
            best_q = q
            best = parts
    # Sort communities by size so labels are stable (0 = largest)
    best = sorted(best, key=len, reverse=True)
    return best, best_q


def best_louvain_unweighted(h, seed, runs):
    best = None
    best_q = -1.0
    for r in range(runs):
        parts = nx.community.louvain_communities(h, seed=seed + r)
        q = nx.community.modularity(h, parts)
        if q > best_q:
            best_q = q
            best = parts
    return best, best_q


def unweighted_modularity(g, seed, runs=20):
    # Modularity ignoring weights, so it is comparable with the rewired (unweighted) null
    h = nx.Graph()
    h.add_nodes_from(g.nodes())
    h.add_edges_from(g.edges())
    parts, q = best_louvain_unweighted(h, seed, runs)
    return q


def partition_labels(parts):
    # Convert list-of-sets into {node: community id}
    labels = {}
    for cid, members in enumerate(parts):
        for node in members:
            labels[node] = cid
    return labels


def compare_partitions(lab_a, lab_b):
    # ARI and NMI on the nodes both partitions contain
    common = sorted(set(lab_a) & set(lab_b))
    a = [lab_a[n] for n in common]
    b = [lab_b[n] for n in common]
    return adjusted_rand_score(a, b), normalized_mutual_info_score(a, b)


def build_graph(data, min_overlap, k):
    # Full pipeline in one call: center -> similarity -> kNN graph
    sim = pairwise_similarity(center_questions(data), min_overlap)
    return knn_graph(sim, list(data.index), k)


def null_model(data, min_overlap, k, n_runs, seed):
    """Modularity and clustering of graphs built from question-wise shuffled data."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_runs):
        g = build_graph(shuffle_columns(data, rng), min_overlap, k)
        parts = nx.community.louvain_communities(g, weight="weight", seed=seed + r)
        rows.append({
            "modularity": nx.community.modularity(g, parts, weight="weight"),
            "clustering": nx.average_clustering(g),
            "n_communities": len(parts),
        })
    return pd.DataFrame(rows)


def null_model_pearson(data, min_overlap, k, n_runs, seed):
    """Column-shuffle null for the person-centred (Pearson) network; returns modularities."""
    rng = np.random.default_rng(seed)
    values = []
    for r in range(n_runs):
        shuffled = shuffle_columns(data, rng)
        sim = pairwise_similarity(center_questions(shuffled), min_overlap, method="pearson")
        g = knn_graph(sim, list(data.index), k)
        parts = nx.community.louvain_communities(g, weight="weight", seed=seed + r)
        values.append(nx.community.modularity(g, parts, weight="weight"))
    return pd.Series(values)


def degree_preserving_null(g, n_runs, seed):
    """Rewire edges keeping every node's degree (double-edge swaps); recompute C, L and Q."""
    rows = []
    m = g.number_of_edges()
    for r in range(n_runs):
        h = nx.Graph(g)
        nx.double_edge_swap(h, nswap=10 * m, max_tries=100 * m, seed=seed + r)
        for u, v in h.edges():
            h[u][v]["weight"] = 1.0
        giant = h.subgraph(max(nx.connected_components(h), key=len))
        parts = nx.community.louvain_communities(h, seed=seed + r)
        rows.append({
            "clustering": nx.average_clustering(h),
            "transitivity": nx.transitivity(h),
            "path_length": nx.average_shortest_path_length(giant),
            "modularity": nx.community.modularity(h, parts),
        })
    return pd.DataFrame(rows)


def seed_stability(g, n_seeds, seed):
    """Pairwise ARI between Louvain partitions from different random seeds."""
    labels = []
    for r in range(n_seeds):
        parts = nx.community.louvain_communities(g, weight="weight", seed=seed + 1000 + r)
        labels.append(partition_labels(parts))
    scores = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            scores.append(compare_partitions(labels[i], labels[j])[0])
    return float(np.mean(scores)), float(np.min(scores))


def node_table(g, labels, flags):
    """Per-person centralities, community and quality flag."""
    # Distance = 1 - similarity, so strongly similar people are "close"
    for u, v, d in g.edges(data=True):
        d["distance"] = 1.0 - d["weight"]
    table = pd.DataFrame({
        "degree": dict(g.degree()),
        "strength": dict(g.degree(weight="weight")),
        "betweenness": nx.betweenness_centrality(g, weight="distance"),
        "eigenvector": nx.eigenvector_centrality_numpy(g, weight="weight"),
        "closeness": nx.closeness_centrality(g, distance="distance"),
        "clustering": nx.clustering(g),
    })
    table["community"] = pd.Series(labels)
    table["flagged"] = flags["flagged"].reindex(table.index)
    return table


def participation_coefficient(g, labels):
    """How evenly a node's links spread across communities (0 = all inside its own group)."""
    pc = {}
    for node in g.nodes():
        k = g.degree(node)
        if k == 0:
            pc[node] = 0.0
            continue
        counts = {}
        for nb in g.neighbors(node):
            c = labels[nb]
            counts[c] = counts.get(c, 0) + 1
        pc[node] = 1.0 - sum((x / k) ** 2 for x in counts.values())
    return pd.Series(pc)


def community_profiles(data, labels):
    """Mean answer of every community on every question + Kruskal-Wallis test per question."""
    grouped = data.copy()
    grouped["community"] = pd.Series(labels)
    means = grouped.groupby("community").mean().T
    rows = []
    for q in data.columns:
        samples = []
        for c in sorted(set(labels.values())):
            s = grouped.loc[grouped["community"] == c, q].dropna()
            if len(s) > 0:
                samples.append(s)
        h, p = stats.kruskal(*samples)
        n = sum(len(s) for s in samples)
        # Epsilon-squared effect size: share of answer variation explained by community
        eps2 = h / ((n ** 2 - 1) / (n + 1))
        rows.append({"question": q, "H": h, "p": p, "epsilon2": eps2})
    tests = pd.DataFrame(rows).set_index("question").sort_values("epsilon2", ascending=False)
    tests["p_fdr"] = benjamini_hochberg(tests["p"].to_numpy())
    return means, tests


def intensity_diagnostics(g, data):
    """Is the network organised by how strongly people agree overall?"""
    person_mean = data.mean(axis=1)
    for n in g.nodes():
        g.nodes[n]["mean_answer"] = float(person_mean[n])
    assort = nx.numeric_assortativity_coefficient(g, "mean_answer")
    nodes = list(g.nodes())
    deg = [g.degree(n) for n in nodes]
    rho, p = stats.spearmanr(deg, [person_mean[n] for n in nodes])
    # PCA on centred answers (missing -> 0 = class mean) via SVD
    z = (data - data.mean(axis=0)).fillna(0.0).to_numpy()
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    explained = s ** 2 / (s ** 2).sum()
    pc1_scores = u[:, 0] * s[0]
    r_pc1 = abs(np.corrcoef(pc1_scores, person_mean.to_numpy())[0, 1])
    same_sign = max((vt[0] > 0).mean(), (vt[0] < 0).mean())
    return {
        "assortativity_mean_answer": float(assort),
        "spearman_degree_mean": float(rho), "spearman_p": float(p),
        "pc1_explained": float(explained[0]), "pc2_explained": float(explained[1]),
        "pc1_corr_mean_answer": float(r_pc1), "pc1_same_sign_loadings": float(same_sign),
    }
