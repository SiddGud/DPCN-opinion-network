"""Run only the data and respondent-network stages and print a short summary.
Useful for checking the network part on its own: python src/run_network_only.py"""
import pandas as pd

import config as C
import data_prep
import stage_network


def print_summary(ctx):
    # Readable summary of the respondent-network results
    results = ctx["results"]
    comm = results["communities"]
    null = results["null_model"]
    inten = results["intensity"]
    print()
    print("Respondent network")
    print(f"  Communities: {comm['n_communities']} with sizes {comm['sizes']}")
    print(f"  Modularity Q = {comm['modularity']:.3f}")
    print(f"  Shuffle null: Q = {null['null_mod_mean']:.3f} +/- {null['null_mod_sd']:.3f}, z = {null['mod_z']:.2f}")
    print(f"  Degree-preserving null: z(Q) = {inten['cm_mod_z']:.2f}, z(clustering) = {inten['cm_clust_z']:.2f}")
    print(f"  Seed stability: mean ARI = {inten['seed_ari_mean']:.2f}, min = {inten['seed_ari_min']:.2f}")
    print()
    print("Intensity check")
    print(f"  Assortativity by mean answer: {inten['assortativity_mean_answer']:.2f}")
    print(f"  PC1 explains {100 * inten['pc1_explained']:.1f}% of variance; corr with mean answer = {inten['pc1_corr_mean_answer']:.2f}")
    pear = inten["pearson"]
    print(f"  Pearson network: Q = {pear['modularity']:.3f}, z = {pear['mod_z']:.2f}, assortativity = {pear['assortativity_mean_answer']:.2f}")
    print()
    tests = pd.read_csv(C.TABLE_DIR / "community_tests.csv", index_col=0)
    print("Statements that separate the communities most")
    for q, row in tests.head(5).iterrows():
        print(f"  {q}  eps2 = {row['epsilon2']:.2f}  {str(row['text'])[:70]}")
    print()
    nodes = pd.read_csv(C.TABLE_DIR / "node_metrics.csv", index_col=0)
    print("Top bridges (betweenness)")
    for rid, row in nodes.head(5).iterrows():
        print(f"  student {rid}: betweenness = {row['betweenness']:.3f}, community = C{int(row['community']) + 1}")


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ctx = {"results": {}}
    # Stage 1: cleaning and encoding
    data_prep.run_stage(ctx)
    # Stage 2: respondent network, communities, null models, intensity
    stage_network.run_stage(ctx)
    print_summary(ctx)


if __name__ == "__main__":
    main()
