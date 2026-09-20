# Respondent network stage (Siddhant)

This stage turns the cleaned answer matrix (87 students x 60 statements) into a similarity network
and finds opinion groups in it.

## Steps

1. **Centre each statement** (`network.center_questions`). Subtract the class mean so statements
   everyone agrees with stop making all students look alike.
2. **Similarity** (`network.pairwise_similarity`). Cosine similarity on statements both students
   answered; pairs sharing fewer than 20 answers get 0. `method="pearson"` also removes each
   student's own mean (used for the intensity check).
3. **Edges** (`network.knn_graph`). Each student is linked to their 6 most similar classmates.
4. **Communities** (`communities.best_louvain`). Louvain run 20 times; the highest-modularity split is kept.
5. **Null models.** `communities.null_model` shuffles each statement across students;
   `communities.degree_preserving_null` rewires edges keeping degrees. Both give z-scores.
6. **Stability** (`communities.seed_stability`). ARI between Louvain runs with 50 seeds.
7. **Profiles** (`communities.community_profiles`). Kruskal-Wallis per statement with FDR correction.
8. **Intensity check** (`communities.intensity_diagnostics`). Assortativity by mean answer and PCA.

## Extra tools

- `python src/sweep_k.py` prints modularity and its null z-score for k = 4, 5, 6, 8, 10.
- `python tests/test_network.py` runs sanity checks on centring, similarity, kNN and partition comparison.
