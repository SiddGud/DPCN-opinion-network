"""Step 1: load, clean and encode the survey; flag possible low-effort responses."""
import numpy as np
import pandas as pd

from config import DATA_FILE, LIKERT_MAP, NO_OPINION, MIN_ANSWERED_SHARE


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


def longest_run(values):
    # Length of the longest streak of identical consecutive non-missing answers
    best = 0
    current = 0
    previous = None
    for v in values:
        if pd.isna(v):
            current = 0
            previous = None
            continue
        if v == previous:
            current += 1
        else:
            current = 1
        previous = v
        best = max(best, current)
    return best


def quality_flags(raw, numeric):
    # Build several independent low-effort indicators per respondent
    rows = []
    for rid in numeric.index:
        answers = numeric.loc[rid]
        text_answers = raw.loc[rid]
        # Share of "Neutral" + "No Comments" among given answers
        middle = (text_answers == "Neutral").sum() + (text_answers == NO_OPINION).sum()
        middle_share = middle / max(text_answers.notna().sum(), 1)
        # Contradiction: agreeing that projects beat exams AND that exams measure knowledge well
        contradiction = bool(answers.get("E01", 0) >= 1 and answers.get("E02", 0) >= 1)
        rows.append({
            "respondent": rid,
            "answered": int(answers.notna().sum()),
            "longest_run": longest_run(text_answers.tolist()),
            "middle_share": round(float(middle_share), 3),
            "person_sd": round(float(answers.std()), 3),
            "contradiction_E01_E02": contradiction,
        })
    flags = pd.DataFrame(rows).set_index("respondent")
    # Each indicator adds one point; two or more points = flagged (flagged people are NOT deleted)
    score = (flags["longest_run"] >= 20).astype(int)
    score += (flags["middle_share"] > 0.4).astype(int)
    score += flags["contradiction_E01_E02"].astype(int)
    score += (flags["person_sd"] < 0.5).astype(int)
    flags["flag_score"] = score
    flags["flagged"] = score >= 2
    return flags


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


def run_stage(ctx):
    """Stage 1 (Vedant): clean the data, flag careless answers, draw Figs. 1 and 3."""
    import config as C
    import plots_data as PD
    from themes import question_summary
    raw, data, texts, flags, prep = prepare()
    flags.to_csv(C.TABLE_DIR / "quality_flags.csv")
    PD.diverging_bars(data)
    PD.missingness(encode(raw), raw)
    qsum = question_summary(data)
    qsum["text"] = pd.Series(texts)
    qsum.sort_values("polarization", ascending=False).to_csv(C.TABLE_DIR / "question_summary.csv")
    ctx["raw"] = raw
    ctx["data"] = data
    ctx["texts"] = texts
    ctx["flags"] = flags
    ctx["qsum"] = qsum
    ctx["results"]["data"] = prep
    print("Data:", prep)
