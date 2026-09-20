"""Step 1: load, clean and encode the survey answers."""
import numpy as np
import pandas as pd

from config import DATA_FILE, LIKERT_MAP, MIN_ANSWERED_SHARE


def load_raw():
    # Read the CSV and use the response id as the row index
    df = pd.read_csv(DATA_FILE)
    df = df.set_index(df.columns[0])
    df.index.name = "respondent"
    return df


def short_id(column_name):
    # "T01. Artificial Intelligence ..." -> "T01"
    return column_name.split(".")[0].strip()


def encode(raw):
    # Map each text answer to a number; "No Comments" and blanks stay NaN
    numeric = pd.DataFrame(index=raw.index)
    for c in raw.columns:
        numeric[short_id(c)] = raw[c].map(LIKERT_MAP).astype(float)
    return numeric


def question_text(raw):
    # Map short id -> full statement text (used in tables and plots)
    texts = {}
    for c in raw.columns:
        texts[short_id(c)] = c.split(".", 1)[1].strip()
    return texts


def prepare():
    # Full cleaning step; returns everything later stages need
    raw = load_raw()
    numeric = encode(raw)
    texts = question_text(raw)
    n_q = numeric.shape[1]
    answered = numeric.notna().sum(axis=1)
    keep = answered >= MIN_ANSWERED_SHARE * n_q
    report = {
        "n_raw": int(len(numeric)),
        "n_questions": int(n_q),
        "n_empty": int((answered == 0).sum()),
        "n_dropped": int((~keep).sum()),
        "n_kept": int(keep.sum()),
        "n_no_comments": int((raw == NO_OPINION).sum().sum()),
        "n_blank": int(raw.isna().sum().sum()),
    }
    clean = numeric[keep]
    flags = quality_flags(raw[keep], clean)
    report["n_flagged"] = int(flags["flagged"].sum())
    return raw, clean, texts, flags, report
