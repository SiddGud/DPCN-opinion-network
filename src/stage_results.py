"""Stage 4 (Krishna): global metrics, robustness study, belief network and DeGroot dynamics."""
import numpy as np
import pandas as pd
import networkx as nx

import config as C
import network as N
import communities as CM
import metrics as M
import belief as B
import dynamics as D
import plots_results as PR


def run_metrics(ctx):
    # Whole-network metrics and small-world test against Erdos-Renyi graphs
    g = ctx["g"]
    gm = M.global_metrics(g)
    er_c, er_l = M.random_baseline(g, C.SEED)
    gm["er_clustering"] = er_c
    gm["er_path_length"] = er_l
    gm["small_world_sigma"] = (gm["avg_clustering"] / er_c) / (gm["avg_path_length"] / er_l)
    ctx["results"]["main_network"] = gm
    print("Main network:", gm)


def run_robustness(ctx):
    # Rebuild the network nine other ways and compare its communities with the main ones
    data = ctx["data"]
    flags = ctx["flags"]
    sim = ctx["sim"]
    sim_raw = ctx["sim_raw"]
    labels = ctx["labels"]
    labels_list = list(data.index)
    variants = {}
    for k in (4, 5, 8, 10):
        variants[f"kNN k={k}"] = N.knn_graph(sim, labels_list, k)
    sim_p = N.pairwise_similarity(N.center_questions(data), C.MIN_OVERLAP_FULL, method="pearson")
    variants["Pearson k=6"] = N.knn_graph(sim_p, labels_list, C.K_NEIGHBOURS)
    variants["Threshold top 10%"] = N.threshold_graph(sim, labels_list, 0.90)
    variants["Disparity backbone"] = N.disparity_backbone(sim, labels_list, C.BACKBONE_ALPHA)
    variants["Raw (no centering)"] = N.knn_graph(sim_raw, labels_list, C.K_NEIGHBOURS)
    keep = [n for n in labels_list if not flags.loc[n, "flagged"]]
    sim_nf = N.pairwise_similarity(N.center_questions(data.loc[keep]), C.MIN_OVERLAP_FULL)
    variants["Without flagged"] = N.knn_graph(sim_nf, keep, C.K_NEIGHBOURS)
    rows = {}
    for name, vg in variants.items():
        isolated = [n for n in list(vg.nodes()) if vg.degree(n) == 0]
        vg.remove_nodes_from(isolated)
        vparts, vq = CM.best_louvain(vg, C.SEED, runs=10)
        ari, nmi = CM.compare_partitions(labels, CM.partition_labels(vparts))
        rows[name] = {"edges": vg.number_of_edges(), "modularity": vq,
                      "n_comm": len(vparts), "ARI": ari, "NMI": nmi}
    rob = pd.DataFrame(rows).T
    rob.to_csv(C.TABLE_DIR / "robustness.csv")
    PR.robustness_plot(rob)
    ctx["results"]["robustness"] = rob.round(3).to_dict("index")
    print(rob.round(3))


def run_belief(ctx):
    # Statement-statement network with FDR-significant signed Spearman edges
    sg = B.statement_network(ctx["data"], C.FDR_Q)
    sparts, sq = CM.best_louvain(sg, C.SEED, runs=10)
    slabels = CM.partition_labels(sparts)
    neg = []
    for u, v, d in sg.edges(data=True):
        if d["sign"] < 0:
            neg.append((u, v, round(d["rho"], 3)))
    bal, bal_null, n_tri, n_neg = B.structural_balance(sg, 200, C.SEED)
    within = 0
    for u, v in sg.edges():
        if u[0] == v[0]:
            within += 1
    top_deg = pd.Series(dict(sg.degree())).sort_values(ascending=False).head(8)
    ctx["results"]["statement_network"] = {
        "edges": sg.number_of_edges(), "density": nx.density(sg), "modularity": sq,
        "communities": [sorted(p) for p in sparts],
        "within_theme_edge_share": within / max(sg.number_of_edges(), 1),
        "negative_edges": neg, "triangles": n_tri, "balanced_share": bal,
        "balance_null_mean": float(bal_null.mean()), "n_negative": n_neg,
        "top_degree": top_deg.to_dict(), "theme_match": B.theme_match(sg, sparts),
    }
    nx.write_gexf(sg, C.TABLE_DIR / "statement_network.gexf")
    PR.statement_plot(sg, slabels)
    print("Statement network:", ctx["results"]["statement_network"])


def run_dynamics(ctx):
    # DeGroot averaging on the most polarized statement
    data = ctx["data"]
    g = ctx["g"]
    labels = ctx["labels"]
    names = ctx["names"]
    question = ctx["qsum"]["polarization"].idxmax()
    opinions = data[question].fillna(data[question].mean()).to_dict()
    history, w = D.degroot(g, opinions)
    node_order = list(g.nodes())
    colors = []
    for n in node_order:
        colors.append(PR.PALETTE[labels[n]])
    PR.degroot_plot(history, colors, question)
    comm = {}
    for cid in range(len(ctx["parts"])):
        idx = [i for i, n in enumerate(node_order) if labels[n] == cid]
        comm[names[cid]] = [round(float(history[0, idx].mean()), 3), round(float(history[-1, idx].mean()), 3)]
    ctx["results"]["degroot"] = {
        "question": question, "initial_mean": float(history[0].mean()),
        "initial_sd": float(history[0].std()), "final_mean": float(history[-1].mean()),
        "sd_after_5": float(history[5].std()), "final_sd": float(history[-1].std()),
        "community_start_end": comm,
    }
    print("DeGroot:", ctx["results"]["degroot"])
