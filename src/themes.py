"""Consensus/polarization per statement and the four theme networks. Owner: Vedant."""
import pandas as pd


def question_summary(data):
    """Consensus vs division for each statement."""
    rows = []
    for q in data.columns:
        s = data[q].dropna()
        agree = (s > 0).mean()
        disagree = (s < 0).mean()
        rows.append({
            "question": q,
            "mean": s.mean(),
            "sd": s.std(),
            "agree_share": agree,
            "disagree_share": disagree,
            # Leik-style polarization proxy: high only when both sides are large
            "polarization": 4 * agree * disagree,
            "answered": len(s),
        })
    return pd.DataFrame(rows).set_index("question")


def run_stage(ctx):
    """Stage 3 (Vedant): one network per theme, each tested against its own null model."""
    import config as C
    import plots_data as PD
    from communities import build_graph, best_louvain, partition_labels, compare_partitions, null_model
    from stats_utils import z_score
    data = ctx["data"]
    qsum = ctx["qsum"]
    theme_graphs = {}
    theme_labels = {}
    rows = {}
    for prefix, tname in C.THEMES.items():
        cols = [c for c in data.columns if c.startswith(prefix)]
        tdata = data[cols]
        tg = build_graph(tdata, C.MIN_OVERLAP_THEME, C.K_NEIGHBOURS)
        isolated = [n for n in list(tg.nodes()) if tg.degree(n) == 0]
        tg.remove_nodes_from(isolated)
        tparts, tq = best_louvain(tg, C.SEED, runs=10)
        tnull = null_model(tdata, C.MIN_OVERLAP_THEME, C.K_NEIGHBOURS, C.N_NULL_THEME, C.SEED)
        theme_graphs[tname] = tg
        theme_labels[tname] = partition_labels(tparts)
        rows[tname] = {
            "mean_answer": float(tdata.stack().mean()),
            "mean_polarization": float(qsum.loc[cols, "polarization"].mean()),
            "modularity": tq, "null_mean": float(tnull["modularity"].mean()),
            "z": float(z_score(tq, tnull["modularity"])), "n_comm": len(tparts),
        }
    theme_df = pd.DataFrame(rows).T
    theme_df.to_csv(C.TABLE_DIR / "theme_networks.csv")
    all_parts = dict(theme_labels)
    all_parts["All 60"] = ctx["labels"]
    keys = list(all_parts.keys())
    nmi = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a in keys:
        for b in keys:
            nmi.loc[a, b] = compare_partitions(all_parts[a], all_parts[b])[1]
    nmi.to_csv(C.TABLE_DIR / "theme_nmi.csv")
    PD.theme_panel(theme_graphs, theme_labels, nmi)
    ctx["results"]["themes"] = theme_df.round(3).to_dict("index")
    ctx["results"]["theme_nmi"] = nmi.round(3).to_dict()
    print(theme_df.round(3))
