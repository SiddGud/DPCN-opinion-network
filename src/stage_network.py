"""Stage 2 (Siddhant): build the respondent network, find communities, test them against
null models, profile the groups and run the intensity (response-style) analysis."""
import numpy as np
import pandas as pd
import networkx as nx

import config as C
import network as N
import communities as CM
import plots_communities as PC
from stats_utils import z_score


def run_stage(ctx):
    data = ctx["data"]
    flags = ctx["flags"]
    texts = ctx["texts"]
    results = ctx["results"]
    labels_list = list(data.index)

    # Similarity before and after centring each statement
    sim_raw = N.pairwise_similarity(data, C.MIN_OVERLAP_FULL)
    sim = N.pairwise_similarity(N.center_questions(data), C.MIN_OVERLAP_FULL)
    PC.similarity_effect(sim_raw, sim)
    iu = np.triu_indices_from(sim, k=1)
    results["similarity"] = {
        "raw_mean": float(sim_raw[iu].mean()), "raw_share_above_0.8": float((sim_raw[iu] > 0.8).mean()),
        "centered_mean": float(sim[iu].mean()), "centered_share_negative": float((sim[iu] < 0).mean()),
    }
    g = N.knn_graph(sim, labels_list, C.K_NEIGHBOURS)
    nx.write_gexf(g, C.TABLE_DIR / "respondent_network.gexf")

    # Communities and per-person centralities
    parts, q = CM.best_louvain(g, C.SEED)
    labels = CM.partition_labels(parts)
    nodes = CM.node_table(g, labels, flags)
    nodes["participation"] = CM.participation_coefficient(g, labels)
    nodes.sort_values("betweenness", ascending=False).to_csv(C.TABLE_DIR / "node_metrics.csv")
    clustering = nx.average_clustering(g)
    results["communities"] = {
        "modularity": q, "n_communities": len(parts), "sizes": [len(p) for p in parts],
        "flagged_per_community": nodes.groupby("community")["flagged"].sum().astype(int).tolist(),
    }

    # Null model 1: shuffle each statement across students
    null = CM.null_model(data, C.MIN_OVERLAP_FULL, C.K_NEIGHBOURS, C.N_NULL, C.SEED)
    null.to_csv(C.TABLE_DIR / "null_model.csv", index=False)
    results["null_model"] = {
        "null_mod_mean": float(null["modularity"].mean()), "null_mod_sd": float(null["modularity"].std()),
        "mod_z": float(z_score(q, null["modularity"])), "mod_p": float((null["modularity"] >= q).mean()),
        "null_clust_mean": float(null["clustering"].mean()),
        "clust_z": float(z_score(clustering, null["clustering"])),
    }
    PC.degree_and_null(g, null, q, clustering)
    print("Null model:", results["null_model"])

    # Null model 2 (degree-preserving), seed stability and intensity diagnostics
    upg = CM.intensity_diagnostics(g, data)
    cm = CM.degree_preserving_null(g, C.N_NULL_CM, C.SEED)
    cm.to_csv(C.TABLE_DIR / "degree_preserving_null.csv", index=False)
    q_unw = CM.unweighted_modularity(g, C.SEED)
    upg["cm_clustering"] = [float(cm["clustering"].mean()), float(cm["clustering"].std())]
    upg["cm_path_length"] = [float(cm["path_length"].mean()), float(cm["path_length"].std())]
    upg["cm_modularity"] = [float(cm["modularity"].mean()), float(cm["modularity"].std())]
    upg["unweighted_modularity"] = q_unw
    upg["cm_mod_z"] = float(z_score(q_unw, cm["modularity"]))
    upg["cm_clust_z"] = float(z_score(clustering, cm["clustering"]))
    upg["seed_ari_mean"], upg["seed_ari_min"] = CM.seed_stability(g, C.N_SEEDS, C.SEED)

    # Person-centred (Pearson) network: removes each student's overall agreement level
    sim_p = N.pairwise_similarity(N.center_questions(data), C.MIN_OVERLAP_FULL, method="pearson")
    gp = N.knn_graph(sim_p, labels_list, C.K_NEIGHBOURS)
    pp, pq = CM.best_louvain(gp, C.SEED)
    plabels = CM.partition_labels(pp)
    pnull = CM.null_model_pearson(data, C.MIN_OVERLAP_FULL, C.K_NEIGHBOURS, C.N_NULL, C.SEED)
    upg["pearson"] = CM.intensity_diagnostics(gp, data)
    upg["pearson"]["modularity"] = pq
    upg["pearson"]["sizes"] = [len(x) for x in pp]
    upg["pearson"]["null_mod"] = [float(pnull.mean()), float(pnull.std())]
    upg["pearson"]["mod_z"] = float(z_score(pq, pnull))
    upg["pearson"]["ari_vs_main"] = float(CM.compare_partitions(labels, plabels)[0])
    results["intensity"] = upg
    print("Intensity:", upg)

    # Group profiles and figures
    means, tests = CM.community_profiles(data, labels)
    tests["text"] = pd.Series(texts)
    means.to_csv(C.TABLE_DIR / "community_means.csv")
    tests.to_csv(C.TABLE_DIR / "community_tests.csv")
    pmeans, ptests = CM.community_profiles(data, plabels)
    ptests["text"] = pd.Series(texts)
    pmeans.to_csv(C.TABLE_DIR / "pearson_community_means.csv")
    ptests.to_csv(C.TABLE_DIR / "pearson_community_tests.csv")
    names = {}
    for cid, members in enumerate(parts):
        if cid < len(C.COMMUNITY_NAMES):
            names[cid] = f"{C.COMMUNITY_NAMES[cid]} (n={len(members)})"
        else:
            names[cid] = f"C{cid + 1} (n={len(members)})"
    PC.community_heatmap(means, tests, names)
    PC.network_plot(g, labels, nodes, names, "fig4_respondent_network.png",
                    f"Respondent opinion network (kNN, k={C.K_NEIGHBOURS}); size = betweenness")
    PC.intensity_plot(g, gp, data, labels, plabels)
    PC.pearson_heatmap(pmeans, ptests)

    # Shared with later stages
    ctx["sim"] = sim
    ctx["sim_raw"] = sim_raw
    ctx["g"] = g
    ctx["parts"] = parts
    ctx["labels"] = labels
    ctx["names"] = names
