"""Figures for the raw answers, missing data and theme networks. Owner: Vedant."""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIG_DIR, THEMES
from plot_utils import PALETTE, LIKERT_ORDER, LIKERT_COLORS, save


def diverging_bars(data):
    """Share of each answer per statement, disagree to the left, agree to the right."""
    rows = []
    for q in data.columns:
        s = data[q].dropna()
        shares = []
        for v in (-2, -1, 0, 1, 2):
            shares.append((s == v).mean())
        rows.append(shares)
    shares = np.array(rows)
    fig, ax = plt.subplots(figsize=(8, 11))
    y = np.arange(len(data.columns))
    # Neutral is split half left, half right of zero
    left = -(shares[:, 0] + shares[:, 1] + shares[:, 2] / 2)
    for i in range(5):
        ax.barh(y, shares[:, i], left=left, color=LIKERT_COLORS[i], label=LIKERT_ORDER[i], height=0.8)
        left = left + shares[:, i]
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data.columns, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Share of answers (disagree ← 0 → agree)")
    ax.set_title("Answer distribution for all 60 statements")
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig1_answer_distribution.png")


def missingness(raw_numeric, raw):
    """Blank / No-Comments answers per theme and answered-count per person."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    blanks = []
    nocom = []
    names = []
    for prefix, name in THEMES.items():
        cols = [c for c in raw.columns if c.startswith(prefix)]
        blanks.append(raw[cols].isna().sum().sum())
        nocom.append((raw[cols] == "No Comments").sum().sum())
        names.append(name)
    x = np.arange(4)
    axes[0].bar(x - 0.2, blanks, 0.4, label="Blank", color="#7f7f7f")
    axes[0].bar(x + 0.2, nocom, 0.4, label="No Comments", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_title("Missing answers by theme (survey order →)")
    axes[0].legend(fontsize=8)
    counts = raw_numeric.notna().sum(axis=1)
    axes[1].hist(counts, bins=range(0, 62, 3), color="#1f77b4")
    axes[1].axvline(30, color="red", ls="--", label="Keep threshold (30)")
    axes[1].set_xlabel("Opinions given (out of 60)")
    axes[1].set_ylabel("Respondents")
    axes[1].set_title("Answers given per respondent")
    axes[1].legend(fontsize=8)
    save(fig, "fig2_missingness.png")


def theme_panel(theme_graphs, theme_labels, nmi):
    """Four theme networks + NMI agreement between their partitions."""
    fig = plt.figure(figsize=(13, 7))
    for i, (name, g) in enumerate(theme_graphs.items()):
        ax = fig.add_subplot(2, 3, [1, 2, 4, 5][i])
        pos = nx.spring_layout(g, weight="weight", seed=7, k=0.35)
        colors = []
        for n in g.nodes():
            colors.append(PALETTE[theme_labels[name][n] % len(PALETTE)])
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.2, width=0.5)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=25)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    ax = fig.add_subplot(1, 3, 3)
    sns.heatmap(nmi, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax, cbar=False)
    ax.set_title("NMI between partitions")
    save(fig, "fig7_theme_networks.png")
