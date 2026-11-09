# Adaptive Multi-Round Retrieval for Evidence Recovery

A reproducible HotpotQA study of paragraph-level evidence recovery, unique evidence noise, and retrieval cost. On the same frozen 100 questions, it compares fixed single-shot Top-2/3/4/5 with Qwen3-4B-controlled Adaptive V1, evidence-chain V2.1, and compact-controller/unseen-retrieval V3. Historical answer/abstention code remains available; the main experiment does not generate answers.

- [Setup and run](documentation/SETUP_AND_RUN.md) — frozen pool, commands, caches, and experiment phases.
- [Architecture and metrics](documentation/SYSTEM_ARCHITECTURE.md) — data flow, records, duplicate counting, evaluation.
- [Experiment history and seven-run results](documentation/EXPERIMENT_ARCHITECTURE_AND_RESULTS.md) — canonical folders, complete comparison, limitations, and cleanup.
- [Canonical run registry](documentation/canonical_runs.json) — ordered sample IDs, source paths, and result hashes.
- [MuSiQue evidence-complexity experiment](documentation/MUSIQUE_VARIATION_EXPERIMENT.md) — frozen variable-gold sample, Top-2–8 baselines, Adaptive V3, and reproduction commands.
