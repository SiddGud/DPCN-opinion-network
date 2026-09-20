"""Figures for the robustness study, belief network and DeGroot dynamics. Owner: Krishna."""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIG_DIR, THEMES
from plot_utils import PALETTE, LIKERT_ORDER, LIKERT_COLORS, save


def robustness_plot(rob):
    """ARI/NMI of each alternative pipeline against the main partition."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(rob))
    ax.bar(x - 0.2, rob["ARI"], 0.4, label="ARI", color="#1f77b4")
    ax.bar(x + 0.2, rob["NMI"], 0.4, label="NMI", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(rob.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Agreement with main partition")
    ax.set_title("Robustness of communities to pipeline choices")
    ax.legend(fontsize=8)
    save(fig, "fig9_robustness.png")


def statement_plot(g, labels):
    """Signed belief network: blue = positive correlation, red = negative."""
    pos = nx.spring_layout(g, weight="weight", seed=3, k=0.5)
    fig, ax = plt.subplots(figsize=(8, 7))
    pos_edges = []
    neg_edges = []
    for u, v, d in g.edges(data=True):
        if d["sign"] > 0:
            pos_edges.append((u, v))
        else:
            neg_edges.append((u, v))
    nx.draw_networkx_edges(g, pos, edgelist=pos_edges, edge_color="#4a7ab5", alpha=0.35, ax=ax)
    nx.draw_networkx_edges(g, pos, edgelist=neg_edges, edge_color="#d62728", width=1.8, ax=ax)
    theme_colors = {"T": "#1f77b4", "E": "#ff7f0e", "S": "#2ca02c", "V": "#9467bd"}
    colors = []
    for n in g.nodes():
        colors.append(theme_colors[n[0]])
    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=260, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=6, ax=ax)
    handles = []
    for k, c in theme_colors.items():
        handles.append(plt.Line2D([], [], marker="o", ls="", color=c, label=THEMES[k]))
    handles.append(plt.Line2D([], [], color="#d62728", label="Negative correlation"))
    ax.legend(handles=handles, fontsize=8, loc="lower left")
    ax.set_title("Belief network: statements linked by significant correlation")
    ax.axis("off")
    save(fig, "fig8_statement_network.png")


def degroot_plot(history, colors, question):
    """Opinion trajectories under DeGroot averaging."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i in range(history.shape[1]):
        ax.plot(history[:, i], color=colors[i], alpha=0.5, lw=0.8)
    ax.set_xlabel("Discussion round")
    ax.set_ylabel("Opinion (-2 .. +2)")
    ax.set_title(f"DeGroot opinion dynamics on the network ({question})")
    save(fig, "fig10_degroot.png")
