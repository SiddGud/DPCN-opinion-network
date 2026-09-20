"""Global network metrics and random-graph baselines. Owner: Krishna."""
import numpy as np
import networkx as nx


def global_metrics(g):
    """Standard whole-network descriptors."""
    giant = g.subgraph(max(nx.connected_components(g), key=len))
    degrees = [d for _, d in g.degree()]
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "density": nx.density(g),
        "mean_degree": float(np.mean(degrees)),
        "max_degree": int(np.max(degrees)),
        "components": nx.number_connected_components(g),
        "giant_share": giant.number_of_nodes() / g.number_of_nodes(),
        "avg_clustering": nx.average_clustering(g, weight=None),
        "transitivity": nx.transitivity(g),
        "avg_path_length": nx.average_shortest_path_length(giant),
        "diameter": nx.diameter(giant),
        "degree_assortativity": nx.degree_assortativity_coefficient(g),
    }


def random_baseline(g, seed):
    """Clustering and path length of an Erdos-Renyi graph with the same size, for small-world check."""
    n = g.number_of_nodes()
    m = g.number_of_edges()
    cs = []
    ls = []
    for r in range(20):
        er = nx.gnm_random_graph(n, m, seed=seed + r)
        giant = er.subgraph(max(nx.connected_components(er), key=len))
        cs.append(nx.average_clustering(er))
        ls.append(nx.average_shortest_path_length(giant))
    return float(np.mean(cs)), float(np.mean(ls))
