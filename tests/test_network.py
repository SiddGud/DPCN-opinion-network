"""Sanity checks for the respondent-network code. Run: python tests/test_network.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import network as N
import communities as CM


def test_centering_removes_means():
    # After centring, every statement should have mean 0
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 2.0, np.nan]})
    centred = N.center_questions(data)
    assert abs(centred["a"].mean()) < 1e-12
    assert abs(centred["b"].mean()) < 1e-12


def test_similarity_identical_and_opposite():
    # Same answers give +1, mirrored answers give -1
    data = pd.DataFrame([[1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [-1.0, 1.0, -2.0]])
    sim = N.pairwise_similarity(data, min_overlap=1)
    assert abs(sim[0, 1] - 1.0) < 1e-9
    assert abs(sim[0, 2] + 1.0) < 1e-9
    assert sim[0, 0] == 0.0


def test_similarity_respects_min_overlap():
    # Pairs sharing too few answers get similarity 0
    data = pd.DataFrame([[1.0, np.nan, np.nan], [1.0, 2.0, np.nan]])
    sim = N.pairwise_similarity(data, min_overlap=2)
    assert sim[0, 1] == 0.0


def test_knn_gives_every_node_k_links():
    # Each node should have at least k neighbours when enough positive similarities exist
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.integers(-2, 3, size=(30, 20)).astype(float))
    sim = N.pairwise_similarity(N.center_questions(data), min_overlap=5)
    g = N.knn_graph(sim, list(data.index), 4)
    for node in g.nodes():
        assert g.degree(node) >= 4


def test_compare_partitions_identical():
    # Identical partitions give ARI = NMI = 1
    labels = {0: 0, 1: 0, 2: 1, 3: 1}
    ari, nmi = CM.compare_partitions(labels, labels)
    assert abs(ari - 1.0) < 1e-9
    assert abs(nmi - 1.0) < 1e-9


if __name__ == "__main__":
    tests = [test_centering_removes_means, test_similarity_identical_and_opposite,
             test_similarity_respects_min_overlap, test_knn_gives_every_node_k_links,
             test_compare_partitions_identical]
    for t in tests:
        t()
        print("passed:", t.__name__)
    print("All tests passed.")
