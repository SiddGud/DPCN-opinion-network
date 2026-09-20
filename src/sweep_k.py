"""Quick sensitivity check for the respondent network: modularity and its null z-score for several k.
Run: python src/sweep_k.py"""
import pandas as pd

import config as C
import data_prep
import communities as CM
from stats_utils import z_score


def main():
    raw, data, texts, flags, prep = data_prep.prepare()
    rows = []
    for k in (4, 5, 6, 8, 10):
        g = CM.build_graph(data, C.MIN_OVERLAP_FULL, k)
        parts, q = CM.best_louvain(g, C.SEED, runs=10)
        null = CM.null_model(data, C.MIN_OVERLAP_FULL, k, 50, C.SEED)
        rows.append({
            "k": k, "edges": g.number_of_edges(), "communities": len(parts),
            "Q": round(q, 3), "null_Q": round(float(null["modularity"].mean()), 3),
            "z": round(float(z_score(q, null["modularity"])), 2),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.TABLE_DIR / "k_sweep.csv", index=False)


if __name__ == "__main__":
    main()
