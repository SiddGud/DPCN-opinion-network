"""Shared plotting constants and the save helper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIG_DIR

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
LIKERT_ORDER = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
LIKERT_COLORS = ["#b2182b", "#ef8a62", "#d9d9d9", "#67a9cf", "#2166ac"]


def save(fig, name):
    # Tight layout and high resolution for the PDF report
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
