# Output guide

The seven canonical same-100 retrieval runs and their exact paths and hashes are in [the run registry](canonical_runs.json). [Experiment architecture and results](EXPERIMENT_ARCHITECTURE_AND_RESULTS.md) explains the flow, all fixed Top-k and adaptive policies, metric definitions, results, limitations, and cleanup. [Setup and run](SETUP_AND_RUN.md) gives commands.

Run `python scripts/compare_canonical_runs.py` from the repository root to validate the seven result files, the ordered sample, candidate/gold IDs, and fixed Top-k ranking prefixes. It regenerates `canonical_comparison.json` and `.csv` in `outputs/adaptive_v3_top2_unseen_compact_controller_same100/`. The V2.1 `three_way_comparison.json` and V3 `four_way_comparison.json` preserve detailed historical controller diagnostics. V2.1 had 37 controller errors among 100 attempts, so its retrieval metrics cover all attempts while controller diagnostics identify their valid-decision denominators.

Historical `scripts/run_experiment.py` artifacts use the answer/abstention `ExperimentRecord` and remain separate from these paragraph-retrieval `RetrievalStudyRecord` results. The earlier 500-question archive is `archive/historical_subset_500_seed88_rag_abstention.zip`.
