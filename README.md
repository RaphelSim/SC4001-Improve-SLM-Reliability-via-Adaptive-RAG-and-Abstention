# Adaptive Multi-Round Retrieval for Evidence Recovery

A reproducible HotpotQA study of paragraph-level evidence recovery, unique evidence noise, and raw retrieval cost. It compares single-shot Top-5 retrieval with Qwen3-4B-controlled adaptive Top-2 retrieval for up to five rounds. Historical answer/abstention code remains available, but the main experiment does not generate answers.

- [Setup and run](documentation/SETUP_AND_RUN.md) — frozen pool, commands, caches, and experiment phases.
- [Architecture and metrics](documentation/SYSTEM_ARCHITECTURE.md) — data flow, records, duplicate counting, evaluation.
