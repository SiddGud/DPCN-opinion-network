"""Central configuration: paths, answer encoding and pipeline parameters."""
from pathlib import Path

# Project root folder (one level above src/)
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "Survey_Results_UC.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

# Likert answers mapped to a symmetric numeric scale (-2 .. +2)
LIKERT_MAP = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
}
# "No Comments" is not neutral: the person gave no opinion, so it becomes missing
NO_OPINION = "No Comments"

# Theme prefix of each question id -> readable theme name
THEMES = {"T": "Technology", "E": "Education", "S": "Society & Ethics", "V": "Environment"}

# Keep a respondent only if they answered at least this share of the 60 statements
MIN_ANSWERED_SHARE = 0.5
# Two people must share at least this many answered questions for their similarity to count
MIN_OVERLAP_FULL = 20
MIN_OVERLAP_THEME = 8

# k for the k-nearest-neighbour graph (main network)
K_NEIGHBOURS = 6
# Significance level for the disparity-filter backbone
BACKBONE_ALPHA = 0.05
# Number of shuffled datasets for the null model
N_NULL = 200
N_NULL_THEME = 100
# Random seed so every run gives identical results
SEED = 42
# False discovery rate for statement-statement correlation edges
FDR_Q = 0.05

# Human-readable names given after reading the community heatmap (fig6); order = community id
COMMUNITY_NAMES = [
    "C1 Committed idealists, pro-online",
    "C2 Moderate sceptics",
    "C3 Low civic-concern",
    "C4 Committed idealists, pro-classroom",
    "C5 Individual learners",
]
# Degree-preserving rewirings and Louvain seeds for the stability check
N_NULL_CM = 100
N_SEEDS = 50
