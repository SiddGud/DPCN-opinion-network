"""Figures for the respondent network, null models, communities and intensity. Owner: Siddhant."""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIG_DIR, THEMES
from plot_utils import PALETTE, LIKERT_ORDER, LIKERT_COLORS, save


def similarity_effect(sim_raw, sim_centered):
    """Why centering matters: raw similarities are all high, centered ones are informative."""
    iu = np.triu_indices_from(sim_raw, k=1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(sim_raw[iu], bins=50, alpha=0.6, label="Raw answers", color="#7f7f7f")
    ax.hist(sim_centered[iu], bins=50, alpha=0.6, label="Question-centered answers", color="#1f77b4")
    ax.set_xlabel("Cosine similarity between two respondents")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Consensus effect: centering spreads out similarities")
    ax.legend()
    save(fig, "fig3_similarity_centering.png")


def network_plot(g, labels, nodes, names, filename, title):
    """Spring layout, colour = community, size = betweenness."""
    pos = nx.spring_layout(g, weight="weight", seed=7, k=0.35)
    fig, ax = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, width=0.6)
    colors = []
    sizes = []
    edgecolors = []
    for n in g.nodes():
        colors.append(PALETTE[labels[n] % len(PALETTE)])
        sizes.append(60 + 2500 * nodes.loc[n, "betweenness"])
        if nodes.loc[n, "flagged"]:
            edgecolors.append("black")
        else:
            edgecolors.append("white")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=sizes,
                           edgecolors=edgecolors, linewidths=1.2)
    top = nodes.sort_values("betweenness", ascending=False).head(5).index
    nx.draw_networkx_labels(g, pos, labels={n: str(n) for n in top}, font_size=8, ax=ax)
    handles = []
    for cid, name in names.items():
        handles.append(plt.Line2D([], [], marker="o", ls="", color=PALETTE[cid], label=name))
    handles.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="white",
                              markeredgecolor="black", label="Flagged low-effort"))
    ax.legend(handles=handles, fontsize=8, loc="lower left")
    ax.set_title(title)
    ax.axis("off")
    save(fig, filename)


def degree_and_null(g, null_df, observed_q, observed_c):
    """Degree distribution + real modularity/clustering against the shuffled-data null model."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    degrees = [d for _, d in g.degree()]
    axes[0].hist(degrees, bins=range(min(degrees), max(degrees) + 2), color="#1f77b4", align="left")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Respondents")
    axes[0].set_title("Degree distribution")
    axes[1].hist(null_df["modularity"], bins=25, color="#bbbbbb", label="Null (shuffled)")
    axes[1].axvline(observed_q, color="red", lw=2, label="Real network")
    axes[1].set_xlabel("Modularity Q")
    axes[1].set_title("Modularity vs null model")
    axes[1].legend(fontsize=8)
    axes[2].hist(null_df["clustering"], bins=25, color="#bbbbbb", label="Null (shuffled)")
    axes[2].axvline(observed_c, color="red", lw=2, label="Real network")
    axes[2].set_xlabel("Average clustering")
    axes[2].set_title("Clustering vs null model")
    axes[2].legend(fontsize=8)
    save(fig, "fig5_degree_null.png")


def community_heatmap(means, tests, names, top_n=15):
    """Mean answer of each community on the statements that separate communities most."""
    top = tests.head(top_n).index
    table = means.loc[top].copy()
    table.columns = [names[c] for c in table.columns]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(table, cmap="RdBu", center=0, vmin=-2, vmax=2, annot=True, fmt=".1f",
                cbar_kws={"label": "Mean answer (-2 .. +2)"}, ax=ax)
    ax.set_title(f"Community opinion profiles (top {top_n} separating statements)")
    ax.set_ylabel("Statement")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    save(fig, "fig6_community_heatmap.png")


def intensity_plot(g, gp, data, labels, plabels):
    """Same kNN networks coloured by each person's overall mean answer."""
    person_mean = data.mean(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    pos = nx.spring_layout(g, weight="weight", seed=7, k=0.35)
    cols = []
    for n in g.nodes():
        cols.append(PALETTE[labels[n] % len(PALETTE)])
    nx.draw_networkx_edges(g, pos, ax=axes[0], alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(g, pos, ax=axes[0], node_color=cols, node_size=30)
    axes[0].set_title("(a) Main network: communities", fontsize=10)
    vals = []
    for n in g.nodes():
        vals.append(person_mean[n])
    nx.draw_networkx_edges(g, pos, ax=axes[1], alpha=0.2, width=0.5)
    sc = nx.draw_networkx_nodes(g, pos, ax=axes[1], node_color=vals, cmap="viridis", node_size=30)
    fig.colorbar(sc, ax=axes[1], fraction=0.04, label="Person's mean answer")
    axes[1].set_title("(b) Main network: overall agreement", fontsize=10)
    posp = nx.spring_layout(gp, weight="weight", seed=7, k=0.35)
    colp = []
    for n in gp.nodes():
        colp.append(PALETTE[plabels[n] % len(PALETTE)])
    nx.draw_networkx_edges(gp, posp, ax=axes[2], alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(gp, posp, ax=axes[2], node_color=colp, node_size=30)
    axes[2].set_title("(c) Person-centred network: communities", fontsize=10)
    for ax in axes:
        ax.axis("off")
    save(fig, "fig11_intensity.png")


def pearson_heatmap(means, tests, top_n=12):
    """Community profiles of the person-centred network, as deviation from the class mean."""
    top = tests.head(top_n).index
    table = means.loc[top].copy()
    table.columns = [f"P{c + 1}" for c in table.columns]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(table, cmap="RdBu", center=0, vmin=-2, vmax=2, annot=True, fmt=".1f",
                cbar_kws={"label": "Mean answer (-2 .. +2)"}, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(f"Person-centred network: community profiles (top {top_n} statements)")
    save(fig, "fig12_pearson_heatmap.png")
