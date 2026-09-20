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
