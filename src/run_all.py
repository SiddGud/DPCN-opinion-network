"""Run the whole pipeline end to end: python src/run_all.py"""
import json

import config as C
import data_prep
import stage_network
import themes
import stage_results


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    ctx = {"results": {}}
    # Stage 1 (Vedant): data cleaning and encoding
    data_prep.run_stage(ctx)
    # Stage 2 (Siddhant): respondent network, communities, null models, intensity
    stage_network.run_stage(ctx)
    # Stage 3 (Krishna): global metrics
    stage_results.run_metrics(ctx)
    # Stage 4 (Vedant): theme networks
    themes.run_stage(ctx)
    # Stage 5 (Krishna): robustness, belief network, dynamics
    stage_results.run_robustness(ctx)
    stage_results.run_belief(ctx)
    stage_results.run_dynamics(ctx)
    with open(C.TABLE_DIR / "results.json", "w") as f:
        json.dump(ctx["results"], f, indent=2, default=str)
    print("Done. Figures in outputs/figures, tables in outputs/tables.")


if __name__ == "__main__":
    main()
