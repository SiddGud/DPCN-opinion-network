"""DeGroot opinion dynamics on the respondent network. Owner: Krishna."""
import numpy as np
import networkx as nx


def degroot(g, opinions, steps=50):
    """DeGroot averaging: everyone repeatedly moves to the weighted mean of self + neighbours."""
    nodes = list(g.nodes())
    w = nx.to_numpy_array(g, nodelist=nodes, weight="weight")
    # Self-weight equal to the person's strongest tie (stubbornness)
    np.fill_diagonal(w, w.max(axis=1))
    w = w / w.sum(axis=1, keepdims=True)
    x = np.array([opinions[n] for n in nodes], dtype=float)
    history = [x.copy()]
    for t in range(steps):
        x = w @ x
        history.append(x.copy())
    return np.array(history), w
