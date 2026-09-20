"""Step 1: load and encode the survey answers."""
import numpy as np
import pandas as pd

from config import DATA_FILE, LIKERT_MAP


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
