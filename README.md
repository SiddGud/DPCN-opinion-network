# Opinion Network Formation (DPCN Assignment 1, team ConsistentProcess)

Builds and analyses networks from a class survey of 60 opinion statements
(15 each on Technology, Education, Society & Ethics, Environment).

## Run

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/run_all.py
```

Takes about 2 minutes. Figures go to `outputs/figures/`, tables and `results.json` to `outputs/tables/`.
All random steps use seed 42, so every run gives the same numbers.

## Code layout and owners

| File | What it does | Owner |
|---|---|---|
| `src/config.py` | Paths, answer encoding, every parameter | Siddhant |
| `src/network.py` | Centring, cosine/Pearson similarity, kNN, threshold and backbone graphs, shuffling | Siddhant |
| `src/communities.py` | Louvain, both null models, seed stability, centralities, group profiles, intensity analysis | Siddhant |
| `src/stats_utils.py` | z-score and Benjamini-Hochberg FDR | Siddhant |
| `src/plot_utils.py`, `src/plots_communities.py` | Shared plot helpers; `fig3`, `fig4`, `fig5`, `fig6`, `fig11`, `fig12` | Siddhant |
| `src/stage_network.py` | Stage 2: respondent network end to end | Siddhant |
| `src/data_prep.py` | Stage 1: cleaning, encoding, careless-answer flags | Vedant |
| `src/themes.py` | Consensus/polarization per statement; stage 4: theme networks | Vedant |
| `src/plots_data.py` | `fig1`, `fig2`, `fig7` | Vedant |
| `src/metrics.py` | Global metrics and random-graph baselines | Krishna |
| `src/belief.py` | Belief network, structural balance, theme match | Krishna |
| `src/dynamics.py` | DeGroot model | Krishna |
| `src/plots_results.py` | `fig8`, `fig9`, `fig10` | Krishna |
| `src/stage_results.py` | Stages 3 and 5: metrics, robustness, belief network, dynamics | Krishna |
| `src/run_all.py` | Runs all stages in order | Krishna |
| `report/` | LaTeX source of the report | Krishna |

## Data

`data/Survey_Results_UC.csv` is the raw survey export (96 responses x 60 statements).
